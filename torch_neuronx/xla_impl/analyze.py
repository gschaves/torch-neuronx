# # Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# # ==============================================================================
import contextlib
import concurrent
from distutils import spawn
import logging
import os
import pathlib
import shlex
import shutil
import subprocess

from pathlib import Path
from types import FunctionType
from typing import List, Any, Union, Dict

import torch
import torch_neuronx
from torch_neuronx.xla_impl.hlo_conversion import xla_trace
from torch_neuronx.pyhlo import hlo_pb2

logger = logging.getLogger("Neuron")

OPS_TO_NOT_ANALYZE = set(
    [
        # aten::to changes the device of the tensor,
        # this is irrelevant for Neuron Models.
        # aten::to calls that cast the model to a different
        # data type will get removed by neuronx-cc
        "aten::to",
        "aten::copy",
    ]
)


class FunctionForward(torch.nn.Module):
    """
    Simple Module that wraps a callable in a torch module
    """

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *x):
        return self.func(*x)


class ScalarInputOperatorWrapper(torch.nn.Module):
    """
    Simple Wrapper class for wrapping scalar input operators (ex. aten::arange)
    for purposes of xla tracing (i.e lowering to HLO).

    This is done by surrounding by performing a simple addition operation
    with the output tensor of the input scalar op with an xla tensor input.
    """

    def __init__(self, func, args):
        super().__init__()
        # initialize with input scalar operator function and its input scalar args.
        self.func = func
        self.func_args = args

    def forward(self, data):
        x = self.func(*self.func_args)
        y = x + data
        return y


@contextlib.contextmanager
def nullcontext():
    """
    A context which does nothing.

    Used as a stand-in for a normal context manager, when a particular
    block of code is only sometimes used with a normal context manager.
    """
    yield


def convert_to_module(func: FunctionType) -> FunctionForward:
    """
    Converts a function to a FunctionForward object,
    which simply wraps the function in a torch module.

    Args:
    1) func: FunctionType

    Retuns:
    FunctionForward object
    """
    return FunctionForward(func)


def fixName(iname) -> str:
    """
    Append neuron to debug name of value for validity

    Args:
    1) iname: debug name of value

    Returns:
    string representing new valid debug name
    """
    if iname.isnumeric():
        iname = "neuron_" + iname
    return iname


def extract_attribute_value(
    node: torch.Node, attr: str
) -> Union[int, float, bool, str]:
    """
    Returns the atttribute value from a torch.Node

    Args:
    1) node: torch.Node
    2) attr: str name of attribute

    Returns:
    int, float, str or None of attribute value
    """

    node_type_kind = node.output().type().kind()

    if node_type_kind == "FloatType":
        return node.f(attr)
    elif node_type_kind == "BoolType":
        return node.i(attr)
    elif node_type_kind == "IntType":
        return node.i(attr)
    elif node_type_kind == "StringType":
        return node.s(attr)
    else:
        return None


def uniquified_tensor_val_str(val: torch.Value) -> str:
    """
    Returns the string representation of a TensorType torch.Value

    string format ex: torch.dtype([size1,size2];[strides1,strides2])

    Args:
    1) val: torch.Value

    Returns:
    str
    """
    if val.type().dtype() is None:
        return "TensorType"
    return f"{val.type().dtype()}({val.type().sizes()};{val.type().strides()})"


def uniquified_list_val_str(val: torch.Value) -> str:
    """
    Returns the string representation of a ListType torch.Value

    string format ex: int[1,,3] or torch.float32[]

    Args:
    1) val: torch.Value

    Returns:
    str
    """
    prim_val_types = set(["BoolType", "IntType", "FloatType"])
    list_val_str = f"{val.type().getElementType()}["
    for i, inp_val in enumerate(val.node().inputs()):
        if i > 0:
            list_val_str += ","
        if inp_val.type().kind() not in prim_val_types:
            break
        if inp_val.node().kind() == "prim::Constant":
            attr_val = extract_attribute_value(inp_val.node(), "value")
            if attr_val is not None:
                list_val_str += f"{attr_val}"
    list_val_str += "]"
    return list_val_str


def uniquified_val_str(node: torch.Node, val: torch.Value):
    """
    Returns the string representation of any torch.Value
    This could return string formats from other functions.

    string format ex: int:1

    Args:
    1) node: torch.Node containing val in input
    2) val: torch.Value to represent as str

    Returns:
    str
    """
    prim_val_types = set(["BoolType", "IntType", "FloatType"])
    if val.type().kind() == "TensorType":
        return uniquified_tensor_val_str(val)
    elif val.type().kind() == "TupleType":
        val_str = "("
        for i, inp_val in enumerate(node.inputs()):
            if i > 0:
                val_str += ", "
            val_str += uniquified_val_str(inp_val.node(), inp_val)
        val_str += ")"
        return val_str
    elif val.type().kind() == "ListType":
        return uniquified_list_val_str(val)
    elif val.type().kind() == "NoneType":
        return "NoneType"
    elif val.type().kind() in prim_val_types:
        attr = val.node().attributeNames()
        if len(attr) == 0:
            return str(val.type())

        attr = attr[0]
        attr_val = extract_attribute_value(val.node(), attr)
        return f"{val.type()}:{attr_val}"

    else:
        return str(val.type())


def uniquified_node_str(node: torch.Node) -> str:
    """
    Returns the string representation of a torch.Node

    string format ex (split into newlines for readability):
    aten::_convolution(
        torch.float32([20, 16, 50, 100];[80000, 5000, 100, 1]);
        TensorType; TensorType; int[2,1];
        int[4,2]; int[3,1];
        bool:0; int[0,0];
        int:1; bool:0; bool:0;
        bool:1; bool:1) -> torch.float32(
                                        [20, 33, 26, 100];
                                        [85800, 2600, 100, 1]
                                    )

    this is used for creating a unique string that can be hashed
    so that it can be used to check the support without running
    --query-compute-placement

    Args:
    1) node: torch.Node

    Returns:
    str
    """
    node_str = f"{node.kind()}("
    for i, inp_val in enumerate(node.inputs()):
        if i > 0:
            node_str += "; "
        node_str += uniquified_val_str(inp_val.node(), inp_val)

    if node.kind() == "prim::TupleConstruct":
        return node_str

    node_str += ") -> "
    out_str = ""
    multi_out = False
    for i, out_val in enumerate(node.outputs()):
        if i > 0:
            multi_out = True
            node_str += "; "
        node_str += uniquified_val_str(out_val.node(), out_val)

    if multi_out:
        node_str += "{" + out_str + "}"

    return node_str


def input_scalarlower(func, ex_inputs):
    wrapper = ScalarInputOperatorWrapper(func, ex_inputs)
    hlo_artifacts = xla_trace(wrapper, torch.rand(2))
    return hlo_artifacts[0]


def check_supported_unsupported(
    hlo: hlo_pb2.HloProto,
    compiler_workdir: Union[str, pathlib.Path] = "./compiler-cache",
    hlo_filename: str = "graph.hlo",
) -> int:
    """
    Checks the hlo lowered op for neuronx-cc support.
    Returns 0 if supported, otherwise it isn't

    Args:
    1. hlo: hlo lowered op
    2. compiler_workdir: path to compiler-cache where hlo will be stored
    3. hlo_filename: name of the hlo file which will be stored in compiler_workdir

    Returns:
    status_code which is an int
    """
    context = nullcontext()

    with context:
        # Create compiler directory if it does not exist
        if not os.path.exists(compiler_workdir):
            os.makedirs(compiler_workdir, exist_ok=True)

        # Write optimized HLO to disk
        hlo_filename = os.path.join(compiler_workdir, hlo_filename)
        with open(hlo_filename, "wb") as handle:
            handle.write(hlo.SerializeToString())

        # NOTE: neuronx-cc import checked in analyze()

        compiler_args = None
        if compiler_args is None:
            compiler_args = []
        elif isinstance(compiler_args, str):
            compiler_args = shlex.split(compiler_args)

        neuron_cc = spawn.find_executable("neuronx-cc")
        if neuron_cc is None:
            raise RuntimeError("neuronx-cc compiler binary does not exist")
        command = [
            neuron_cc,
            "compile",
            hlo_filename,
            "--framework",
            "XLA",
            "--target",
            "trn1",
            "--verbose",
            "error",
            "--query-compute-placement",
        ]
        command.extend(compiler_args)

        status = subprocess.run(command, cwd=compiler_workdir).returncode

        return status


def get_type_from_python_parameter(param) -> str:
    """
    Obtains the type from a python defined parameter string
    ex: "var: type" => "type"
    """
    return param.split(": ")[1]


def should_be_tensor(inp: Any, exp_type: str) -> Any:
    """
    If input type is expected to be a tensor and isn't, convert it

    Args:
    1) inp: Some data type (could be Tensor, int, float, etc.)
    2) exp_type: A string representation of the expected type. ex. "Tensor"

    Returns the appropriate type based on exp_type
    """
    input_type = type(inp)
    if (
        exp_type == "Tensor" and isinstance(input_type, torch.Tensor)
    ) or exp_type != "Tensor":
        return inp
    return torch.as_tensor(inp)


class NeuronTSNode:
    """
    Simple representation of TS::node with bidirectional edges for producers/consumers
    """

    def __init__(self, id: int, node: torch.Node):
        self.id = id
        self.node: torch.Node = node
        self.unique_node_str = uniquified_node_str(node)
        self.kind = node.kind()
        self.hlo = None
        self.analyzed = False
        self.support_report = {
            "supported": True,
            "failureAt": None,
            "call": None,
            "opGraph": None,
        }
        # self.ex_inputs = None  # example inputs to check support

        self.inputs = {}  # map input name string to TS::input obj
        self.outputs = {}  # map output name string to TS::output obj
        self.consumers = []  # nodes that consume this nodes' outputs
        self.producers = []  # nodes that produce this nodes' inputs

        self.class_type = False
        self.class_parent = None  # class type parent node

        self.in_return_node = (
            set()
        )  # flags whether the TS::output is in the return node of the model  # noqa: F401

        self.parameters_used = []  # parameters from paramNode

    def __str__(self):
        inputs = [node.id for node, _ in self.producers]
        outputs = [node.id for node, _ in self.consumers]
        node_str = (
            f"NeuronTSNode-{self.id}({self.kind}) inputs:{inputs} outputs:{outputs}"
        )

        if self.parameters_used:
            node_str += f" :: Parameters: {self.parameters_used}"

        if self.class_type:
            node_str += " :: ClassType"
            if self.class_parent:
                node_str += f" derived from {self.class_parent}"

        if not self.support_report["supported"]:
            node_str += " x"

        return node_str

    def check_val_in_consumers(self, val: str):
        for _, consumed_val in self.consumers:
            if consumed_val == val:
                return True

        return False

    def createOpGraph(self) -> Dict[str, object]:
        """
        Converts a TorchScript Node object to a single
        TorchScript callable Graph Function, i.e an opGraph.

        Args:
        1) node: TorchScript Node to be converted to opGraph

        Returns a tuple containing the callable function,
        node input names, node output names, node output types,
        scope, and the opGraph string object.
        """

        # create a graph for each node
        node = self.node
        opGraph = torch.Graph()
        opBlock = opGraph.block()
        opReturns = opBlock.returnNode()

        # the graph will only have the one node
        opNode = opGraph.create(node.kind())
        # make this node look like the original
        opNode.copyAttributes(node)
        scope = node.sourceRange()

        input_names = []

        # setup the inputs for this new node
        for i in node.inputs():
            iname = fixName(i.debugName())
            input_names.append(iname)

            # for each input on the original node, add an input to the graph/block
            block_in = opBlock.addInputToBlock()
            block_in.setType(i.type())
            # use the same name as the original
            block_in.setDebugName(iname)
            # set the input on the node to be the new block input
            opNode.addInput(block_in)

        # delete the output added by default
        opNode.eraseOutput(0)

        output_names = []
        output_types = []
        # setup the outputs for the new node
        multiOut = node.outputsSize() > 1
        tupleNode = None
        if (
            multiOut
        ):  # for operators that return multiple outputs (ex. aten::max on a dimension)
            tupleNode = opGraph.create("prim::TupleConstruct")

        for o in node.outputs():
            oname = fixName(o.debugName())
            output_names.append(oname)

            # add a new output
            nodeOutput = opNode.addOutput()
            nodeOutput.setDebugName(oname)
            nodeOutput.setType(o.type())
            output_types.append(o.type())
            if multiOut:  # add to tuple instead of return node if multiple outputs
                tupleNode.addInput(nodeOutput)
            else:  # add this output to the returns for the graph/block
                opReturns.addInput(nodeOutput)

        if multiOut:
            # make sure tuple is correctly structured to pass linter
            tupleNodeOut = tupleNode.addOutput()
            tupleNodeOut.setDebugName("neuron_tuple_out")
            tupleNodeOut.setType(torch.TupleType(output_types))
            tupleNode.eraseOutput(0)

            opReturns.addInput(tupleNodeOut)

        # add the new node to the graph
        opGraph.insertNode(opNode)
        if multiOut:
            opGraph.insertNode(tupleNode)

        # create the func
        func = torch._C._create_function_from_graph("support_check", opGraph)

        self.func_data = {
            "func": func,
            "input_names": input_names,
            "output_names": output_names,
            "output_types": output_types,
            "scope": scope,
            "op_graph_str": opGraph.__str__(),
        }

        return self.func_data

    def update_support_report(
        self, supported: bool, failureAt: str, call, opGraph: str
    ):
        self.support_report["supported"] = supported
        self.support_report["failureAt"] = failureAt
        self.support_report["call"] = call
        self.support_report["opGraph"] = opGraph

    def check_node_support(
        self, compiler_workdir="analyze-compiler-cache", additional_ignored_ops=set([])
    ) -> bool:
        if not self.analyzed:
            """
            Conditions to evaluate for support:

            1) Must be an aten:: op. prim:: ops do not have a
            direct lowering to HLO by themselves
            2) Output must be a tensor or tensor-like structure.
            XLA requires output of a graph to be a tensor.
            3) Must not be in OPS_TO_NOT_ANALYZE

            Ops that don't meet the above criteria are ignored (does not mean unsupported)
            """
            output_types = self.func_data["output_types"]
            all_tensors = True
            for output_type in output_types:
                all_tensors = all_tensors and isinstance(output_type, torch.TensorType)
            should_be_analyzed = (
                self.node.kind().startswith("aten::")
                and all_tensors
                and self.node.kind() not in OPS_TO_NOT_ANALYZE
                and self.node.kind() not in additional_ignored_ops
            )
            if should_be_analyzed:
                # check for neuronx-cc compiler support
                status = check_supported_unsupported(self.hlo, compiler_workdir)

                if status != 0:  # unsupported
                    self.update_support_report(
                        False,
                        "neuronx-cc",
                        self.func_data["scope"],
                        self.func_data["op_graph_str"],
                    )

            self.analyzed = True

        return self.support_report["supported"]

    def get_support_report_copy(self) -> Dict[str, str]:
        """
        This function returns a copy of the support report.
        """
        support_report_copy = {
            "kind": self.kind,
            "failureAt": self.support_report["failureAt"],
            "call": self.support_report["call"],
            "opGraph": self.support_report["opGraph"],
        }
        return support_report_copy


class NeuronTSGraph:
    def __init__(
        self,
        graph: torch.Graph,
        inputs: List[any],
        compiler_workdir=None,
        additional_ignored_ops=set([]),
        max_workers=4,
    ):
        """
        Initializes a NeuronTSGraph object from a torchscript graph and its inputs.

        For details on the torchscript terminology, check this link:
        https://github.com/pytorch/pytorch/blob/v1.13.0/torch/csrc/jit/OVERVIEW.md

        Args:
        1) graph: torchscript graph
        2) inputs: a list of the graph inputs. The first element must be
        the ScriptModule where the graph comes from
        3) compiler_workdir: Specify the location of compiler artifacts,
        otherwise it will be located in /tmp/neuron-analyze/
        4) additional_ignored_ops: A set specifying the aten ops to ignore.
        Default is an empty set.
        5) max_workers: Max number of worker threads when analyzing ops.
        """

        self.nodes: List[NeuronTSNode] = []

        # Data Structures to keep track of unique nodes
        self.unique_node_support_map: Dict[str, bool] = {}
        self.unique_node_map: Dict[str, List[int]] = {}

        # maps a torch.Value name to its actual value (Tensor, int, etc.)
        self.values: Dict[str, Any] = {}

        # maps a torch.Value name to the torch.Value object from the param node
        self.ts_param_node_values: Dict[str, torch.Value] = {}

        # maps a torch.Value name to the torch.Value object
        # for the operator nodes in the function body
        self.valueMap: Dict[str, torch.Value] = {}

        # maps torch.Value name to the NeuronTSNodes that use that
        # value as inputs and outputs respectively.
        self.inputMap = {}
        self.outputMap = {}

        # list of torch.Value names that are parameters to the graph,
        # and are sorted by order in the function signature
        self.sortedParams = []

        # list of torch.Value names that are return values
        # of the graph and sorted in their original return order
        self.sortedReturnVals = []

        self.fromTS(
            graph, inputs, compiler_workdir, additional_ignored_ops, max_workers
        )

    def __str__(self):
        inputs = {}
        for k in self.inputMap:
            inputs[k] = []
            for node in self.inputMap[k]:
                inputs[k].append(node.id)
        outputs = []
        for k in self.outputMap:
            for node in self.outputMap[k]:
                outputs.append(node.id)
        s = f"NeuronTSGraph - inputs:{inputs} outputs:{outputs}\n"
        for node in self.nodes:
            s += str(node) + "\n"
        return s

    def fromTS(
        self,
        graph: torch.Graph,
        inputs: List[Any],
        compiler_workdir=None,
        additional_ignored_ops=set([]),
        max_workers=4,
    ):
        """
        Takes in a torchscript graph and extracts the info
        necessary for analysis and partitioning.

        Args:
        1) graph: torchscript graph
        2) inputs: a list of the graph inputs. The first element must be
        the ScriptModule where the graph comes from
        3) compiler_workdir: Specify the location of compiler artifacts,
        otherwise it will be located in /tmp/neuron-analyze/
        4) additional_ignored_ops: A set specifying the aten ops to ignore.
        Default is an empty set.
        5) max_workers: Max number of worker threads when analyzing ops.

        Returns:
        No return value
        """

        # record parameters
        for idx, inp in enumerate(graph.block().paramNode().outputs()):
            self.sortedParams.append(fixName(inp.debugName()))
            self.inputMap[inp.debugName()] = []
            self.values[inp.debugName()] = inputs[idx]
            self.ts_param_node_values[inp.debugName()] = inp

        # record graph outputs
        for out in graph.block().returnNode().inputs():
            self.outputMap[fixName(out.debugName())] = []

        nodeOutputMap = {}  # map outputs from nodes to the producing nodes
        node_list = graph.nodes()

        # Convert Nodes to NeuronTSNode objects and add them to NeuronTSGraph.
        # This loop also tracks the inputs and outputs of each node
        for idx, node in enumerate(node_list):
            neuron_ts_node = self.addNode(
                node, additional_ignored_ops=additional_ignored_ops
            )

            for i in neuron_ts_node.inputs:
                # make connections to other nodes
                if i in self.inputMap:
                    # this node consumes graph inputs
                    self.inputMap[i].append(neuron_ts_node)
                    neuron_ts_node.parameters_used.append(i)
                elif i in nodeOutputMap:
                    # this node consumes outputs from other nodes
                    othNode = nodeOutputMap[i]
                    neuron_ts_node.producers.append((othNode, i))
                    othNode.consumers.append((neuron_ts_node, i))
                else:
                    raise Exception(
                        f'unable to find producer node for input: "{i}" in node: {node}'
                    )

            for o in neuron_ts_node.outputs:
                if o in self.outputMap:
                    self.outputMap[o].append(neuron_ts_node)
                else:
                    nodeOutputMap[o] = neuron_ts_node

        # record return TS Value names
        return_node = graph.block().returnNode()
        for val in return_node.inputs():
            ret_name = val.debugName()
            self.sortedReturnVals.append(fixName(ret_name))

        # check model support for all unique operators
        logger.info("Analyzing Operator Support...")
        self.check_model_support(compiler_workdir, max_workers)

    def addNode(self, node: torch.Node, additional_ignored_ops=set([])) -> NeuronTSNode:
        """
        Add Torchscript Node to NeuronTSGraph

        Args:
        1) node: torchscript node
        2) additional_ignored_ops: set of aten ops to ignore in analysis
        """
        neuron_ts_node = NeuronTSNode(len(self.nodes), node)

        # record node input torch.Value s
        for i in node.inputs():
            if i.debugName() not in self.valueMap:
                self.valueMap[i.debugName()] = i
            neuron_ts_node.inputs[fixName(i.debugName())] = i
            # check if a node's input is a ClassType, as it needs special treatment
            if i.type().kind() == "ClassType":
                neuron_ts_node.class_parent = i.debugName()

        # record node output torch. Value
        for o in node.outputs():
            if o.debugName() not in self.valueMap:
                self.valueMap[o.debugName()] = o
            neuron_ts_node.outputs[fixName(o.debugName())] = o
            # check if a node's output is a ClassType, as it needs special treatment
            if o.type().kind() == "ClassType":
                neuron_ts_node.class_type = True

        # convert operator node to a single operator graph (i.e and OpGraph)
        op_graph_data = neuron_ts_node.createOpGraph()
        func = op_graph_data["func"]

        ex_inputs = [self.values[iname] for iname in op_graph_data["input_names"]]

        # convert to tensor if type changed to tensor in
        # pytorch->torchscript->pytorch conversion
        SUPPORT_CHECK_SUBSTR = "def support_check("
        substr_start_idx = func.code.find(SUPPORT_CHECK_SUBSTR) + len(
            SUPPORT_CHECK_SUBSTR
        )
        substr_end_idx = func.code.find(")")
        function_param_substr = func.code[substr_start_idx:substr_end_idx]
        function_param_substr = function_param_substr.replace("\n", "")
        expected_types = []
        if function_param_substr:
            # note: params are separated by commas followed by 4 spaces.
            # This is because some param types might contain subtypes
            # delimited by commas (ex: Tuple)
            expected_types = [
                get_type_from_python_parameter(param.strip())
                for param in function_param_substr.split(",    ")
            ]
            ex_inputs = tuple(
                [
                    should_be_tensor(inp, exp_type)
                    for inp, exp_type in zip(ex_inputs, expected_types)
                ]
            )

        # test executing this graph on CPU
        ex_outputs = op_graph_data["func"](*ex_inputs)
        if (
            len(op_graph_data["output_names"]) > 1
        ):  # deconstruct tuple as if func(*ex_inputs) return multiple outputs
            iterable_ex_outputs = [*ex_outputs]
        else:
            iterable_ex_outputs = [ex_outputs]

        output_names = op_graph_data["output_names"]
        for idx, o in enumerate(iterable_ex_outputs):
            self.values[output_names[idx]] = o

        """
        Conditions to evaluate for support:

        1) Must be an aten:: op. prim:: ops do not have a direct
        lowering to HLO by themselves
        2) Output must be a tensor or tensor-like structure.
        XLA requires output of a graph to be a tensor.
        3) Must not be in OPS_TO_NOT_ANALYZE

        Ops that don't meet the above criteria are ignored.
        IMPORTANT: this does not mean unsupported
        """
        output_types = op_graph_data["output_types"]
        is_tensor_like_output = True
        for output_type in output_types:
            is_tensor_like_output = is_tensor_like_output and isinstance(
                output_type, torch.TensorType
            )

        logger.debug(f"{node.kind()}({expected_types})")
        should_be_analyzed = (
            node.kind().startswith("aten::")
            and is_tensor_like_output
            and node.kind() not in OPS_TO_NOT_ANALYZE
            and node.kind() not in additional_ignored_ops
        )
        try:
            if should_be_analyzed:
                # attempt to lower to hlo
                with torch_neuronx.contexts.mock_neuron_cores():
                    _func = func
                    _ex_inputs = ex_inputs
                    # if the operator takes in input scalars
                    # we need to wrap with the ScalarInputOperatorWrapper
                    # so that the hlo can be lowered successfully
                    if "Tensor" not in expected_types:
                        logger.debug("Scalar Input Operator Detected")
                        _func = ScalarInputOperatorWrapper(func, ex_inputs)
                        _ex_inputs = torch.rand(ex_outputs.shape)
                    logger.debug("Attempting HLO Lowering")
                    hlo_artifacts = xla_trace(_func, _ex_inputs)
                    hlo = hlo_artifacts[0]
                    neuron_ts_node.hlo = hlo
        except Exception as e:
            logger.debug(e)
            # ops that can't be lowered to hlo are considered unsupported
            neuron_ts_node.update_support_report(
                False,
                "Lowering to HLO",
                op_graph_data["scope"],
                op_graph_data["op_graph_str"],
            )
            self.unique_node_support_map[neuron_ts_node.unique_node_str] = (
                False,
                "Lowering to HLO",
            )
            neuron_ts_node.analyzed = True
        # initialize unique_node_support_map_entry
        else:
            if should_be_analyzed:
                self.unique_node_support_map[neuron_ts_node.unique_node_str] = (
                    True,
                    "",
                )
        finally:
            pass
        if should_be_analyzed:
            if neuron_ts_node.unique_node_str not in self.unique_node_map:
                self.unique_node_map[neuron_ts_node.unique_node_str] = []
            self.unique_node_map[neuron_ts_node.unique_node_str].append(
                neuron_ts_node.id
            )

        self.nodes.append(neuron_ts_node)
        return neuron_ts_node

    def check_model_support(self, compiler_workdir=None, max_workers=4):
        """
        Runs the analysis of operators through neuronx-cc, via multi threading.

        Args:
        1) compiler_workdir: specific directory containing compiler artifacts.
        Default is None, which means the artifacts are saved in /tmp/neuron-analyze/
        2) max_workers: Max number of worker threads. Default is 4.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            # specify parent compiler workdir
            parent_compiler_workdir = (
                compiler_workdir
                if compiler_workdir is not None
                else os.path.join("/tmp", "neuron-analyze")
            )
            logger.debug(
                f"Compiler Artifacts generated by analyze() located at {parent_compiler_workdir}"  # noqa: E501
            )
            # submit --query-compute-placement jobs to worker threads asynchronously
            for unique_node_str in self.unique_node_map.keys():
                if not self.unique_node_support_map[unique_node_str][0] or (
                    "prim::" in unique_node_str
                ):
                    continue
                unique_node: NeuronTSNode = self.get_node_by_id(
                    self.unique_node_map[unique_node_str][0]
                )
                analyze_workdir = os.path.join(
                    parent_compiler_workdir,
                    f"analyze-{unique_node.kind}-{hash(unique_node_str)}-compiler-cache",
                )
                futures[
                    executor.submit(unique_node.check_node_support, analyze_workdir)
                ] = unique_node

            # mark unique nodes with respective support classification
            nodes_to_mark_unsupported = []
            for future in concurrent.futures.as_completed(futures):
                unique_node = futures[future]
                return_val = future.result()
                unique_node_str = unique_node.unique_node_str
                if return_val:
                    self.unique_node_support_map[unique_node_str] = (
                        return_val,
                        "",
                        "",
                        "",
                    )
                else:
                    self.unique_node_support_map[unique_node_str] = (
                        return_val,
                        unique_node.support_report["failureAt"],
                        unique_node.support_report["call"],
                        unique_node.support_report["opGraph"],
                    )
                    nodes_to_mark_unsupported.append(unique_node_str)

        # mark all nodes part of a unique node category
        # with respective support classification
        for unique_node_str in nodes_to_mark_unsupported:
            unique_node_group = self.unique_node_map[unique_node_str]
            for node_id in unique_node_group:
                self.get_node_by_id(node_id).update_support_report(
                    *self.unique_node_support_map[unique_node_str]
                )

    def get_node_by_id(self, node_id: int) -> NeuronTSNode:
        return self.nodes[node_id]

    def get_param(self, val_name: str) -> torch.Value:
        return self.ts_param_node_values[val_name]


def create_final_report(
    neuron_ts_graph: NeuronTSGraph, neuronxcc_version: str
) -> Dict[str, Any]:
    """
    Create final report from analysis.

    Args:
    1) neuron_ts_graph: The NeuronTSGraph object containing analysis data
    2) neuronxcc_version: The neuronx-cc version used in analysis.

    Returns:
    A JSON like dictionary.
    """
    model_support = {
        "torch_neuronx_version": torch_neuronx._version.__version__,
        "neuronx_cc_version": neuronxcc_version,
        "support_percentage": "",
        "supported_operators": {},
        "unsupported_operators": [],
    }

    for neuron_ts_node in neuron_ts_graph.nodes:
        if "prim::" in neuron_ts_node.kind:
            continue
        if neuron_ts_node.support_report["supported"]:
            if neuron_ts_node.kind not in model_support["supported_operators"]:
                model_support["supported_operators"][neuron_ts_node.kind] = 0

            model_support["supported_operators"][neuron_ts_node.kind] += 1
        else:
            model_support["unsupported_operators"].append(
                neuron_ts_node.get_support_report_copy()
            )

    supported_ops_ct = 0
    unsupported_ops_ct = len(model_support["unsupported_operators"])

    for _, num_ops in model_support["supported_operators"].items():
        supported_ops_ct += num_ops

    percent_support = supported_ops_ct * 100 / (supported_ops_ct + unsupported_ops_ct)
    model_support["support_percentage"] = f"{percent_support:.2f}%"

    logger.info("The following operations are currently supported:")
    for supported_op in model_support["supported_operators"]:
        logger.info(supported_op)
    logger.info("The following operations are currently not supported:")
    for unsupported_op in model_support["unsupported_operators"]:
        logger.info(f"{unsupported_op['kind']}, {unsupported_op['call']}")
    logger.info(
        f"{model_support['support_percentage']} of arithmetic operations ({supported_ops_ct} of {supported_ops_ct+unsupported_ops_ct}) are supported"  # noqa: E501
    )

    return model_support


def non_error_remove(remove_func, path):
    try:
        remove_func(path)
    except Exception:
        logger.warning(f"path '{path}' doesn't exist")
        return


def analyze(
    model: Union[FunctionType, torch.nn.Module],
    ex_input: Any,
    compiler_workdir: Union[str, pathlib.Path] = None,
    additional_ignored_ops=set([]),
    max_workers=4,
    is_hf_transformers=False,
    cleanup=False,
) -> Dict:
    """
    Checks all of the models operators and splits them based on
    whether they are supported or unsupported by Neuron.

    Args:
    1) model: either a function or a torch.nn.Module
    representing the model to be analyzed
    2) ex_input: The inputs to be passed into the model.
    If multiple arguments are used, it should be packed in a tuple object
    3) compiler_workdir: The path for the compiler-cache.
    If set to None, a temp directory is created: /tmp/neuron-analyze/
    4) max_workers: This parameter specified the max number of
    worker threads to spawn. Default is 4
    5) is_hf_transformers: If the model is a huggingface transformers model,
    it is recommended to enable this option to prevent deadlocks
    from TOKENIZERS_PARALLELISM. The default is False.
    6) cleanup: This parameter is to specify whether to delete
    the compiler artifact directories that are generated after running analyze.
    Default is False.

    Returns:
    A JSON like dict object of this structure:
    {
        "torch_neuronx_version": "1.*",
        "neuronx_cc_version": "2.*",
        "support_percentage": "50%",
        "supported_operators": {
            "aten::linear": 1
        },
        "unsupported_operators": [
            {
                "kind": "aten::fft_fft",
                "compute_engine": "CPU",
                "failureAt": "neuronx-cc",
                "call": "path/to/python_code_containing_op:line_num",
                "opGraph": "string representation of opGraph".
            }
        ]
    }
    """
    # Ensure neuronx-cc is installed
    try:
        import neuronxcc
    except ImportError as e:
        raise RuntimeError(
            "neuronx-cc is not installed.\n"
            "neuronx-cc can be installed using:\n"
            "python3 -m pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com neuronx-cc"  # noqa: E501
        ) from e

    if not isinstance(ex_input, tuple):
        ex_input = (ex_input,)

    # if function passed in, convert to module, required for this flow
    if isinstance(model, FunctionType):
        model = convert_to_module(model)

    if is_hf_transformers:
        # this will suppress the tokenizers parallelism warning from transformers
        # env var will be unset at the end of analyze
        # read here for implications of setting this env var:
        # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/72926996#72926996  # noqa: E501
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    trace = torch.jit.trace(model, ex_input)
    torch._C._jit_pass_inline(trace.graph)

    # setup the values
    inputs = [model, *ex_input]

    neuron_ts_graph = NeuronTSGraph(
        trace.graph, inputs, compiler_workdir, additional_ignored_ops, max_workers
    )

    report = create_final_report(neuron_ts_graph, neuronxcc.__version__)

    if is_hf_transformers:
        os.unsetenv("TOKENIZERS_PARALLELISM")

    if cleanup:
        compiler_workdir = (
            compiler_workdir if compiler_workdir is not None else "/tmp/neuron-analyze/"
        )
        for p in Path(compiler_workdir).glob("analyze-aten*"):
            non_error_remove(shutil.rmtree, p)

    return report
