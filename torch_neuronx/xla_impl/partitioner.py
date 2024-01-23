# # Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# # ==============================================================================  # noqa: E501
from enum import Enum
import logging
import os
import warnings

from types import FunctionType
from typing import Union, Callable, Dict, List, Optional, Set, Tuple

import torch
from torch_neuronx.xla_impl.analyze import (
    NeuronTSGraph,
    NeuronTSNode,
    create_final_report,
    fixName,
    convert_to_module,
)
import torch_neuronx

logger = logging.getLogger("Neuron")


class PartitionerConfig:
    """
    A list of Options for the Graph Partitioner

    1) model_support_percentage_threshold: a number between 0-1 representing
    the maximum allowed percentage of operators that must be supported.
    If the max is breached, the function will throw a ValueError.
    Default is 0.5 (i.e 50% of operators must be supported)

    2) min_subgraph_size: the minimum number of operators in a subgraph.
    Can be >=1 or == -1. If -1, minimum
    subgraph size is not checked (i.e no minimum). If >= 1, each subgraph must contain
    at least that many operators. If not, the function will throw a ValueError.

    3) max_subgraph_count: the maximum number of subgraphs in the partitioned model.
    Can be >=1 or == -1. If -1, max subgraph count is not checked (i.e no maximum).
    If >= 1, the partitioned model must contain at most that many subgraphs.
    If not, the function will throw a ValueError.

    4) ops_to_partition: This is a set of strings of this structure "aten::<operator>.
    These are operators that will be partitioned to CPU regardless of support.
    The default is None (i.e no additional operators will be partitioned)

    5)  analyze_parameters: This is a dictionary of kwargs used for analyze.
    NOTE: Not all kwargs in torch_neuronx.analyze() are supported
    in partition().
    The following kwargs for analyze() are supported for use in partition().
        a) compiler_workdir
        b) additional_ignored_ops
        c) max_workers
    The default is None, corresponding to the default analyze() arguments.
    """

    def __init__(
        self,
        *args,
        trace_kwargs=None,
        model_support_percentage_threshold=0.5,
        min_subgraph_size=-1,
        max_subgraph_count=-1,
        ops_to_partition=None,
        analyze_parameters=None,
        **kwargs,
    ):
        if args:
            warnings.warn(
                "Use Key Word arguments, instead of arguments. Ignoring supplied args and using default values."
            )

        self.trace_kwargs = trace_kwargs
        self.model_support_percentage_threshold = model_support_percentage_threshold
        self.min_subgraph_size = min_subgraph_size
        self.max_subgraph_count = max_subgraph_count
        self.ops_to_partition = ops_to_partition
        self.analyze_parameters = analyze_parameters


class Device(Enum):
    NEURON = 0
    CPU = 1


class PartitionIterator:
    """
    Simple Class that implements __iter__ and __next__ methods
    for storing partitions
    """

    def __init__(self):
        self.neuron_partitions = []
        self.cpu_partitions = []
        self.partition_map = []
        self.num_partitions = 0

    def __iter__(self):
        self.idx = 0
        self.neuron_idx = 0
        self.cpu_idx = 0
        return self

    def __next__(self):
        if self.idx < len(self.partition_map):
            partition_to_grab = self.partition_map[self.idx]
            self.idx += 1
            if partition_to_grab == Device.NEURON:
                partition = self.neuron_partitions[self.neuron_idx]
                partition_type = Device.NEURON
                self.neuron_idx += 1
            else:
                partition = self.cpu_partitions[self.cpu_idx]
                partition_type = Device.CPU
                self.cpu_idx += 1

            return partition, partition_type
        else:
            raise StopIteration


def complete_device_partition(
    device: Device,
    device_partition: List[int],
    partition_iterator: PartitionIterator,
    min_subgraph_size: int,
) -> None:
    """
    This function completes a device partition for the partition iterator

    Args:
    1) device: An Enum specifying CPU or NEURON
    2) device_partition: A list of operators which are based on their topological order
    3) partition_iterator: An iterator for device partitions
    4) min_subgraph_size: Each subgraph must be a minimum subgraph size, or else an error is thrown.
       This can be ignored if the min_subgraph_size is -1.
    """

    if min_subgraph_size != -1:
        if len(device_partition) < min_subgraph_size:
            raise ValueError(
                f"Partitioner created a {device.name} subgraph with {len(device_partition)} operators, which is less than the specified min subgraph size of {min_subgraph_size}"  # noqa: E501
            )

    if device == device.NEURON:
        partition_iterator.neuron_partitions.append(device_partition)
    else:
        partition_iterator.cpu_partitions.append(device_partition)

    partition_iterator.partition_map.append(device)
    logger.debug(partition_iterator.partition_map)
    partition_iterator.num_partitions += 1
    return []


def determine_partition_strategy(
    neuron_ts_graph: NeuronTSGraph, min_subgraph_size: int
) -> PartitionIterator:
    """
    This function takes in a NeuronTSGraph and a min_subgraph_size
    to create partitions. It will fail if a partition cannot satisfy
    the min subgraph size.

    Args:
    1) neuron_ts_graph: A NeuronTSGraph object
    2) min_subgraph_size: An integer representing the minimum number of
    operators in a subgraph

    Returns:
    A PartitionIterator object which when iterated over
    gives a device,list of aten operators.
    """
    aten_node_ids = [node.id for node in neuron_ts_graph.nodes if "aten::" in node.kind]

    partition_iterator = PartitionIterator()

    neuron_partition: List[int] = []
    cpu_partition: List[int] = []

    for aten_node_id in aten_node_ids:
        aten_node = neuron_ts_graph.get_node_by_id(aten_node_id)

        if aten_node.support_report["supported"]:
            if len(cpu_partition) != 0:
                # complete cpu partition
                cpu_partition = complete_device_partition(
                    Device.CPU, cpu_partition, partition_iterator, min_subgraph_size
                )

            neuron_partition.append(aten_node_id)
        else:
            if len(neuron_partition) != 0:
                # complete neuron partition
                neuron_partition = complete_device_partition(
                    Device.NEURON,
                    neuron_partition,
                    partition_iterator,
                    min_subgraph_size,
                )

            cpu_partition.append(aten_node_id)

    if len(neuron_partition) != 0:
        # complete neuron partition
        complete_device_partition(
            Device.NEURON, neuron_partition, partition_iterator, min_subgraph_size
        )
    else:
        # complete cpu partition
        complete_device_partition(
            Device.CPU, cpu_partition, partition_iterator, min_subgraph_size
        )

    return partition_iterator


def determine_subgraph_param_var_split(
    partition: List[int],
    partitioned_tensor_nodes: Set[int],
    neuron_ts_graph: NeuronTSGraph,
) -> Tuple[Set[NeuronTSNode], List[int], List[int]]:
    """
    Function that splits the partition into
    a group of param values and variable values

    Args:
    1) partition: List of node ids
    2) partitioned_tensor_nodes: set of tensor nodes that have already been partitioned
    3) neuron_ts_graph: A NeuronTSGraph object

    Returns:
    A tuple of a set of NeuronTSNode that are original parameters in the full graph
    and a list of node ids corresponding to parameters
    and a list of node ids corresponding to the vars used
    """
    logger.debug("Determining the Subgraph parameter and variable split")
    included_orig_params = set()
    ts_params = []
    ts_vars = []

    # for each node retrieve the dependencies using bfs traversal
    # since the nodes are already topologically sorted, we reverse
    # iterate to maintain the dependency nodes in topological order
    for aten_node_id in reversed(partition):
        aten_node = neuron_ts_graph.get_node_by_id(aten_node_id)
        dep_stack = [(aten_node, "")] + aten_node.producers

        visited_node_ids = set()

        while len(dep_stack):
            dep_node_id = dep_stack.pop()[0].id

            # add to visited or skip visited
            if dep_node_id not in visited_node_ids:
                visited_node_ids.add(dep_node_id)
            else:
                continue

            # determine if it is a parameter or var
            if dep_node_id in partitioned_tensor_nodes:
                if dep_node_id not in ts_params:
                    ts_params.append(dep_node_id)
                continue
            elif dep_node_id not in ts_vars:
                ts_vars.append(dep_node_id)

            # add to bfs stack
            dep_node = neuron_ts_graph.get_node_by_id(dep_node_id)
            dep_stack += dep_node.producers
            for param in dep_node.parameters_used:
                included_orig_params.add(param)

    # sorting ids which correspond to a topological order
    ts_vars = sorted(ts_vars)

    return included_orig_params, ts_params, ts_vars


def add_params_to_subgraph(
    neuron_ts_graph: NeuronTSGraph,
    subgraph_block: torch.Block,
    included_orig_params: Set[NeuronTSNode],
    ts_params: List[int],
    inputs: Dict[str, torch.Value],
) -> Tuple[List[str], List[bool]]:
    """
    Adds parameters to the subgraph. Torchscript has a concept of a Block,
    which is a collection of nodes, and the function signature containing
    the parameters is a Block.

    Args:
    1) neuron_ts_graph: NeuronTSGraph
    2) subgraph_block: the Torchscript Block to add parameters to
    3) included_orig_params: the original parameters in the full graph
    4) ts_params: The list of node ids corresponding to parameters
    5) inputs: A dictionary that maps a torch.Value name
    to the torch.Value instance itself.
    Is used to map the parameter value added to the torchscript Block

    Returns:
    A tuple containing a list of parameter value names (maps to inputs)
    and a list of booleans whose index correspond to the parameter value name list
    and signifies whether the parameter is a class type or not
    """
    logger.debug("Adding Parameters to Subgraph")
    subgraph_input_names = []
    subgraph_input_class_flags = []

    for orig_param in neuron_ts_graph.sortedParams:
        if orig_param in included_orig_params:
            orig_param_value: torch.Value = neuron_ts_graph.get_param(orig_param)
            block_in: torch.Value = subgraph_block.addInputToBlock()
            block_in.copyMetadata(orig_param_value)

            # keep track of names and if it's a ClassType
            inputs[orig_param] = block_in
            subgraph_input_names.append(orig_param)
            subgraph_input_class_flags.append(
                orig_param_value.type().kind() == "ClassType"
            )

    # add additional tensor params after original params
    for tensor_param_id in ts_params:
        tensor_node: NeuronTSNode = neuron_ts_graph.get_node_by_id(tensor_param_id)
        for tensor_output_val in tensor_node.node.outputs():
            oname = fixName(tensor_output_val.debugName())
            if not tensor_node.check_val_in_consumers(oname):
                continue
            block_in: torch.Value = subgraph_block.addInputToBlock()
            block_in.copyMetadata(tensor_output_val)

            inputs[oname] = block_in
            subgraph_input_names.append(oname)
            subgraph_input_class_flags.append(
                tensor_output_val.type().kind() == "ClassType"
            )

    return subgraph_input_names, subgraph_input_class_flags


def add_var_nodes_to_subgraph(
    neuron_ts_graph: NeuronTSGraph,
    subgraph: torch.Graph,
    partitioned_tensor_nodes: List[int],
    ts_vars: List[int],
    inputs: Dict[str, torch.Value],
    deps: Dict[str, torch.Value],
) -> Set[torch.Node]:
    """
    Adds var nodes to the body of the torch.Graph subgraph object.

    Args:
    1) neuron_ts_graph: NeuronTSGraph object
    2) subgraph: the actual torchscript Graph object which nodes/values are added to
    3) partitioned_tensor_nodes: List of partitioned tensor nodes
    which gets updated as a side-effect
    4) ts_vars: list of Torchscript node ids to add to subgraph.
    5) inputs: Dictionary that maps the Torchscript input param value name
    to the actual Torchscript value object
    6) deps: Dictionary that maps the Torchscript body value name
    to the actual Torchscript value object.

    Returns:
    A set of Torchscript Node objects added to the subgraph.
    """
    logger.debug("Adding Variable Nodes to Subgraph")
    subgraph_node_set: Set[torch.Node] = set()

    for node_id in ts_vars:
        neuron_ts_node = neuron_ts_graph.get_node_by_id(node_id)
        ts_node = neuron_ts_node.node
        subgraph_node_set.add(ts_node)
        subgraph_node: torch.Node = subgraph.create(ts_node.kind())
        subgraph_node.copyAttributes(ts_node)

        for inp in ts_node.inputs():
            iname = fixName(inp.debugName())
            if iname in deps:
                subgraph_node.addInput(deps[iname])
            else:
                subgraph_node.addInput(inputs[iname])

        subgraph_node.eraseOutput(0)
        for out in ts_node.outputs():
            oname = fixName(out.debugName())

            subgraph_node_out = subgraph_node.addOutput()
            subgraph_node_out.copyMetadata(out)
            subgraph_node_out.setDebugName(oname)

            if out.type().kind() == "TensorType":
                partitioned_tensor_nodes.add(neuron_ts_node.id)
            deps[oname] = subgraph_node_out

        subgraph.insertNode(subgraph_node)

    return subgraph_node_set


def add_outputs_to_non_final_subgraph(
    neuron_ts_graph: NeuronTSGraph,
    subgraph: torch.Graph,
    subgraph_return: torch.Node,
    subgraph_node_set: Set[torch.Node],
    ts_vars: List[int],
    deps: Dict[str, torch.Value],
) -> List[str]:
    """
    Fills in the return node with all downstream dependencies of the subgraph.

    Args:
    1) neuron_ts_graph: NeuronTSGraph
    2) subgraph: torchscript Graph object that will get executed
    3) subgraph_return: the torchscript return Node object
    4) subgraph_node_set: A set of torchscript Node objects that
    were added to the subgraph
    5) ts_vars: List of Torchscript node ids that were added to subgraph.
    This is used to reference NeuronTSNode objects.
    6) deps: Dictionary that maps the Torchscript body value name
    to the actual Torchscript value object.

    Returns:
    A list of torchscript output value names in the subgraph
    """
    logger.debug("Adding outputs to return node in subgraph")
    subgraph_output_names: List[str] = []
    subgraph_output_types: List[Optional[torch.JitType]] = []

    subgraph_outputs = []
    for node_id in ts_vars:
        node = neuron_ts_graph.get_node_by_id(node_id)
        if len(node.consumers) == 0:
            for oname in node.outputs:
                if deps[oname].type().kind() != "TensorType":
                    continue
                subgraph_output_names.append(oname)
                subgraph_outputs.append(deps[oname])

        # add downstream dependencies to output
        for consumer_node, oname in node.consumers:
            cn_ts = consumer_node.node
            if deps[oname].type().kind() != "TensorType":
                continue
            elif cn_ts not in subgraph_node_set and oname not in subgraph_output_names:
                subgraph_output_names.append(oname)
                subgraph_outputs.append(deps[oname])

    # if multiple outputs, pack in a tuple
    if len(subgraph_outputs) > 1:
        tuple_node: torch.Node = subgraph.create("prim::TupleConstruct")
        for out in subgraph_outputs:
            tuple_node.addInput(out)
            subgraph_output_types.append(out.type())

        tuple_node.eraseOutput(0)
        tuple_node_out = tuple_node.addOutput()
        tuple_node_out.setDebugName("neuron_tuple_out")
        tuple_node_out.setType(torch.TupleType(subgraph_output_types))
        subgraph.insertNode(tuple_node)
        subgraph_return.addInput(tuple_node_out)
    else:
        out = subgraph_outputs[0]
        subgraph_return.addInput(out)

    return subgraph_output_names


def add_outputs_to_final_subgraph(
    neuron_ts_graph: NeuronTSGraph,
    subgraph: torch.Graph,
    subgraph_return: torch.Node,
    deps: [str, torch.Value],
) -> None:
    """
    Fills in the return node with the original model's return values

    Args:
    1) neuron_ts_graph: NeuronTSGraph
    2) subgraph: torchscript Graph object that will get executed
    3) subgraph_return: the torchscript return Node object
    4) deps: Dictionary that maps teh Torchscript body value name
    to the actual Torchscript value object.

    Returns:
    None
    """
    if len(neuron_ts_graph.sortedReturnVals) > 1:
        tuple_node: torch.Node = subgraph.create("prim::TupleConstruct")
        subgraph_ret_types = []
        for ret_name in neuron_ts_graph.sortedReturnVals:
            ret_val = deps[ret_name]
            tuple_node.addInput(ret_val)
            subgraph_ret_types.append(ret_val.type())

        tuple_node.eraseOutput(0)
        tuple_node_out = tuple_node.addOutput()
        tuple_node_out.setDebugName("neuron_tuple_out")
        tuple_node_out.setType(torch.TupleType(subgraph_ret_types))
        subgraph.insertNode(tuple_node)
        subgraph_return.addInput(tuple_node_out)
    else:
        ret_name = neuron_ts_graph.sortedReturnVals[0]
        ret_val = deps[ret_name]
        subgraph_return.addInput(ret_val)


def add_outputs_to_subgraph(
    neuron_ts_graph: NeuronTSGraph,
    subgraph: torch.Graph,
    subgraph_return: torch.Node,
    subgraph_node_set: Set[torch.Node],
    ts_vars: List[int],
    deps: Dict[str, torch.Value],
    is_final_subgraph=False,
) -> List[str]:
    """
    Fills in the return node with appropriate values of the subgraph.

    Args:
    1) neuron_ts_graph: NeuronTSGraph
    2) subgraph: torchscript Graph object that will get executed
    3) subgraph_return: the torchscript return Node object
    4) subgraph_node_set: A set of torchscript Node objects
    that were added to the subgraph
    5) ts_vars: List of Torchscript node ids that were added to subgraph.
    Used to reference NeuronTSNode objects.
    6) deps: Dictionary that maps the Torchscript body value name
    to the actual Torchscript value object.
    7) is_final_subgraph: boolean that flags a subgraph as the final one or not

    Returns:
    A list of torchscript output value names in the subgraph
    """
    if is_final_subgraph:
        add_outputs_to_final_subgraph(neuron_ts_graph, subgraph, subgraph_return, deps)
        return neuron_ts_graph.sortedReturnVals

    return add_outputs_to_non_final_subgraph(
        neuron_ts_graph, subgraph, subgraph_return, subgraph_node_set, ts_vars, deps
    )


def create_subgraph(
    partition: List[int],
    partitioned_tensor_nodes: Set[int],
    neuron_ts_graph: NeuronTSGraph,
    subgraph_name: str,
    is_final_subgraph=False,
) -> Callable:
    """
    Creates a callable function that is anagolous to a
    Torchscript RecursiveScriptModule, which is created
    from creating a Torchscript subgraph from
    the partitioned aten operator nodes.

    Args:
    1) partition: A list of ints used to retrieve NeuronTSNodes
    from the NeuronTSGraph to add to subgraph
    2) partitioned_tensor_nodes: A set of ints that corresponds to
    nodes that were already added to previous partitions
    3) neuron_ts_graph: NeuronTSGraph
    4) subgraph_name: a string that represents the name of the generated function.
    Will be of format f'{device}_partition_{number}'

    Returns:
    A Callable function representing the subgraph.
    """

    # determine which Torchscript nodes/values are parameters or body variables
    included_orig_params, ts_params, ts_vars = determine_subgraph_param_var_split(
        partition, partitioned_tensor_nodes, neuron_ts_graph
    )

    # initialize torchscript graph
    subgraph: torch.Graph = torch.Graph()
    subgraph_block: torch.Block = subgraph.block()
    subgraph_return: torch.Node = subgraph_block.returnNode()

    # inputs is for mapping to Torchscript param values
    # and deps is for mapping to Torchscript var values
    inputs: Dict[str, torch.Value] = {}
    deps: Dict[str, torch.Value] = {}

    # add input params to subgraph
    subgraph_input_names, subgraph_input_class_flags = add_params_to_subgraph(
        neuron_ts_graph, subgraph_block, included_orig_params, ts_params, inputs
    )

    # add body/var values to subgraph
    subgraph_node_set = add_var_nodes_to_subgraph(
        neuron_ts_graph, subgraph, partitioned_tensor_nodes, ts_vars, inputs, deps
    )

    # log included operators in subgraph/partition
    if logger.level != logging.NOTSET and logger.level <= logging.INFO:
        operator_count = {}
        for node in subgraph_node_set:
            if node.kind() not in operator_count:
                operator_count[node.kind()] = 0
            operator_count[node.kind()] += 1

        logger.info("The following operators will be included in this partition:")
        for op, count in operator_count.items():
            logger.info(f"{op}:{count}")

    # add appropriate outputs to subgraph
    subgraph_output_names = add_outputs_to_subgraph(
        neuron_ts_graph,
        subgraph,
        subgraph_return,
        subgraph_node_set,
        ts_vars,
        deps,
        is_final_subgraph,
    )

    # create Torchscript ScriptFunction from graph object
    func = torch._C._create_function_from_graph(subgraph_name, subgraph)
    return (
        func,
        subgraph_input_names,
        subgraph_input_class_flags,
        subgraph_output_names,
    )


class NeuronPartitionExecutor(torch.nn.Module):
    """
    A torch Module class to execute neuron partitions regardless of device type.
    These partition executors are added to the NeuronModel class in a ModuleList
    where they are executed.

    Args:
    1) func: A callable function (jit.traced cpu partition or a neuron partition)
    2) subgraph_input_names: list of torchscript value names
    3) subgraph_input_class_flags: list of booleans which align with subgraph_input_names.
    These flags indicate whether that torchscript value object is a ClassType
    4) values: A dictionary mapping the torch value name
    to the actual torchscript value object.
    """

    def __init__(
        self,
        func: Callable,
        subgraph_input_names: List[str],
        subgraph_input_class_flags: List[bool],
        values: Dict[str, torch.Value],
    ):
        super().__init__()
        classes = []
        for iname, is_class_type in zip(
            subgraph_input_names, subgraph_input_class_flags
        ):
            if is_class_type:
                classes.append(values[iname])

        self.registered_classes = torch.nn.ModuleList(classes)
        self.contains_class = len(self.registered_classes) > 0
        self.func = func

    def forward(self, *data):
        # pack data appropriately and execute function
        func_inputs = [model_module for model_module in self.registered_classes]
        for item in data:
            if isinstance(item, list) or isinstance(item, tuple):
                for tensor in item:
                    func_inputs.append(tensor)
            else:
                func_inputs.append(item)

        return self.func(*func_inputs)


class NeuronModel(torch.nn.Module):
    """
    A torch Module class that executes all partitions in the correct order.
    This module also makes it so a partitioned model can be jit.traced, which
    makes it serializable to .pt files.

    This class has a method add_subgraph_partition which adds partition executors.
    Once all partitions are added, call register_partitions to complete initialization.

    Args:
    None
    """

    def __init__(self):
        super().__init__()
        self.partitions = []
        self.class_names_to_ignore = []
        self.input_names_collection = []
        self.output_names_collection = []
        self.value_map = {}

    def forward(self, *data):
        inputs = [*data]

        # handle mapping of data to various partitions
        # and execute partition
        for partition, input_names, output_names in zip(
            self.registered_partitions,
            self.input_names_collection,
            self.output_names_collection,
        ):
            if inputs is None:
                inputs = [
                    self.value_map[inp_name]
                    for inp_name in input_names
                    if inp_name not in self.class_names_to_ignore
                ]
            else:
                self.value_map = {}
                i = 0
                for inp_name in input_names:
                    if inp_name in self.class_names_to_ignore:
                        continue
                    self.value_map[inp_name] = inputs[i]
                    i += 1
            x = partition(*inputs)
            out_collection = [x] if len(output_names) == 1 else x

            for out_item, out_name in zip(out_collection, output_names):
                self.value_map[out_name] = out_item
            inputs = None

        return x

    def add_subgraph_partition(
        self,
        traced_partition_executor: Callable,
        subgraph_input_names: List[str],
        subgraph_input_class_flags: List[bool],
        subgraph_output_names: List[str],
    ) -> None:
        """
        Takes in a traced NeuronPartitionExecutor to be executed

        Args:
        1) traced_partition_executor: A callable partition.
        Should be jit traced or neuron traced
        2) subgraph_input_names: List of strings corresponding to torchscript value names
        3) subgraph_input_class_flags: List of bools corresponding
        to whether a torchscript value is a ClassType
        4) subgraph_output_names: List of strings corresponding to
        torchscript output value names
        """
        self.partitions.append(traced_partition_executor)
        self.input_names_collection.append(subgraph_input_names)

        for inp_name, is_class_type in zip(
            subgraph_input_names, subgraph_input_class_flags
        ):
            if is_class_type and inp_name not in self.class_names_to_ignore:
                self.class_names_to_ignore.append(inp_name)

        self.output_names_collection.append(subgraph_output_names)

    def register_partitions(self):
        """
        Registers the list of traced partition executors by
        converting the list to a ModuleList.
        """
        self.registered_partitions = torch.nn.ModuleList(self.partitions)


def _construct_cpu_subgraph_with_original_module(
    neuron_partition_executor: NeuronPartitionExecutor, ex_inputs: List[any]
) -> Callable:
    """
    This function takes a neuron_partition_executor and converts the func attribute
    to a ScriptModule with the original weights of the operators via internal libtorch
    operations that are wrapped in a custom neuron operator
    "neuron::create_module_from_graph"

    Args:
    1) neuron_partition_executor: NeuronPartitionExecutor that is a CPU subgraph
    2) ex_inputs: the input passed into neuron_partition_executor

    Returns:
    A properly configured Torchscript ScriptModule that has
    the correct weights and operators

    Note: This function is considered private because it relies on the existance of
    a saved jit.traced module containing the original weights
    called orig_trace.pt (this is done in the partition function).
    It also relies on internal mechanisms that are not appropriate for users to use.
    """
    MODULE_NAME_WITH_NEW_GRAPH = "new_graph.pt"
    MODULE_NAME_WITH_CLASS_ATTRS = "orig_trace.pt"

    if not neuron_partition_executor.contains_class:
        return torch.jit.trace(neuron_partition_executor, ex_inputs)

    # save the ScriptFunction so it can get loaded by internal neuron op
    torch.jit.save(neuron_partition_executor.func, MODULE_NAME_WITH_NEW_GRAPH)
    final_cpu_subgraph_path = torch.ops.neuron.create_module_from_graph(
        MODULE_NAME_WITH_NEW_GRAPH, MODULE_NAME_WITH_CLASS_ATTRS
    )

    os.remove(MODULE_NAME_WITH_NEW_GRAPH)

    final_cpu_subgraph = torch.jit.load(final_cpu_subgraph_path)
    os.remove(final_cpu_subgraph_path)

    return final_cpu_subgraph


def process_subgraph(
    device: Device,
    func: Callable,
    neuron_ts_graph: NeuronTSGraph,
    input_names: List[str],
    input_class_flags: List[bool],
    output_names: List[str],
    trace_kwargs: Dict,
) -> Callable:
    """
    Converts a torchscript ScriptFunction to a NeuronPartitionExecutor
    which is serializable with weights.

    Args:
    1) device: string enum which could be CPU or NEURON
    2) func: the ScriptFunction callable object containing the subgraph
    3) input_names: names of the torchscript input value objects
    4) input_class_flags: flags indicating whether the
    respective torchscript value object is a ClassType
    5) output_names: names of the torchscript output value objects

    Returns:
    A traced NeuronPartitionExecutor object to be added to NeuronModel.
    """
    logger.debug("Processing Subgraph")
    ex_inputs = [
        neuron_ts_graph.values[iname]
        for iname, is_class_type in zip(input_names, input_class_flags)
        if not is_class_type
    ]
    # wrap function in a NeuronPartitionExecutor for tracing
    neuron_partition_executor = NeuronPartitionExecutor(
        func, input_names, input_class_flags, neuron_ts_graph.values
    )
    if device == Device.NEURON:
        if isinstance(ex_inputs, list):
            ex_inputs = tuple(ex_inputs)
        logger.debug("Performing Neuron trace")
        traced_neuron_partition_executor = torch_neuronx.trace(
            neuron_partition_executor, ex_inputs, **trace_kwargs
        )
    else:
        logger.debug(
            "Performing neuron::create_module_from_graph operation on CPU Subgraph"
        )
        traced_neuron_partition_executor = _construct_cpu_subgraph_with_original_module(
            neuron_partition_executor, ex_inputs
        )
        # IMPORTANT: Inlining TS partition graph prevents segfaults
        # when saving partitioned models
        # (especially if the first subgraph is a CPU subgraph)
        torch._C._jit_pass_inline(traced_neuron_partition_executor.graph)

    logger.debug("Getting Outputs from traced partition executor")
    ex_outputs = traced_neuron_partition_executor(*ex_inputs)

    if not isinstance(ex_outputs, tuple):
        ex_outputs = [ex_outputs]
    logger.debug(f"{ex_outputs}")
    logger.debug(f"{output_names}")
    for idx, out in enumerate(ex_outputs):
        neuron_ts_graph.values[output_names[idx]] = out

    return traced_neuron_partition_executor


def partition(
    model: Union[Callable, torch.nn.Module],
    ex_input: any,
    trace_kwargs: Optional[dict] = None,
    model_support_percentage_threshold=0.5,
    min_subgraph_size=-1,
    max_subgraph_count=-1,
    ops_to_partition=None,
    analyze_parameters=None,
) -> Callable:
    """
    This function partitions out unsupported operators to separate CPU subgraphs and
    keeps supported operators in neuron subgraphs. The support of these operators
    is determined by the same logic used in torch_neuronx.analyze().

    Args:
    1) model: a callable function (less predictable behavior) or a torch.nn.Module
    2) ex_input: the input to the model
    3) model_support_percentage_threshold: a number between 0-1 representing
    the maximum allowed percentage of operators that must be supported.
    If the max is breached, the function will throw a ValueError.
    Default is 0.5 (i.e 50% of operators must be supported)
    4) min_subgraph_size: the minimum number of operators in a subgraph.
    Can be >=1 or == -1. If -1, minimum
    subgraph size is not checked (i.e no minimum). If >= 1, each subgraph must contain
    at least that many operators. If not, the function will throw a ValueError.
    5) max_subgraph_count: the maximum number of subgraphs in the partitioned model.
    Can be >=1 or == -1. If -1, max subgraph count is not checked (i.e no maximum).
    If >= 1, the partitioned model must contain at most that many subgraphs.
    If not, the function will throw a ValueError.
    6) ops_to_partition: This is a set of strings of this structure "aten::<operator>.
    These are operators that will be partitioned to CPU regardless of support.
    The default is None (i.e no additional operators will be partitioned)
    7)  analyze_parameters: This is a dictionary of kwargs used for analyze.
    NOTE: Not all kwargs in torch_neuronx.analyze() are supported
    in partition().
    The following kwargs for analyze() are supported for use in partition().
        a) compiler_workdir
        b) additional_ignored_ops
        c) max_workers
    The default is None, corresponding to the default analyze() arguments.

    Returns:
    A traced NeuronModel, which is an instance of a Torchscript ScriptModule.
    This ScriptModule can be executed like a normal torchscript model
    and saved with torch.jit.save().
    """
    if isinstance(model, FunctionType):
        model = convert_to_module(model)

    if trace_kwargs is None:
        trace_kwargs = {}

    if ops_to_partition is None:
        ops_to_partition = set([])

    if analyze_parameters is None:
        analyze_parameters = {}

    # get full TS graph from jit trace
    model_trace = torch.jit.trace(model, ex_input)
    torch.jit.save(model_trace, "orig_trace.pt")
    torch._C._jit_pass_inline(model_trace.graph)
    ts_graph = model_trace.graph
    logger.debug(ts_graph)

    if not isinstance(ex_input, tuple):
        ex_input = (ex_input,)

    # setup NeuronTSGraph which offers more info on a torchscript graph
    inputs = [model, *ex_input]
    neuron_ts_graph = NeuronTSGraph(ts_graph, inputs, **analyze_parameters)

    logger.debug(neuron_ts_graph)

    # use final report to determine operator support percentage
    support_report = create_final_report(neuron_ts_graph, "")
    support_percentage = float(support_report["support_percentage"].replace("%", ""))
    if (
        torch.allclose(
            torch.Tensor([support_percentage]),
            torch.Tensor([100.00]),
            rtol=1e-4,
            atol=1e-4,
        )
        and len(ops_to_partition) == 0
    ):
        logger.info("Model operator support is 100%; tracing entire model.")
        return torch_neuronx.trace(model, ex_input, **trace_kwargs)
    elif support_percentage <= 2:
        raise ValueError(
            "Model support percentage is very close to 0%, aborting trace."
        )
    elif support_percentage < (model_support_percentage_threshold * 100):
        raise ValueError(
            f"Model support percentage of {support_percentage:.2f}% below threshold of {(model_support_percentage_threshold * 100):.2f}%"  # noqa: E501
        )
    else:
        logging.info(
            f"Model Support percentage of {support_percentage:.2f}% above threshold of {(model_support_percentage_threshold * 100):.2f}%"  # noqa: E501
        )

    # if there are extra ops to partition, treat those nodes as unsupported
    for ts_node in neuron_ts_graph.nodes:
        if ts_node.node.kind() in ops_to_partition:
            ts_node.support_report["supported"] = False

    # create a partition iterator with the device partitions
    # based on some partition strategy (heuristic)
    # TODO: develop more sophisticated partitioning heuristics
    partition_iterator = determine_partition_strategy(
        neuron_ts_graph, min_subgraph_size
    )

    # check max_subgraph_count
    if max_subgraph_count != -1:
        if partition_iterator.num_partitions > max_subgraph_count:
            raise ValueError(
                f"The partitioner has found {partition_iterator.num_partitions} subgraphs which exceeds the specified max subgraph count of {max_subgraph_count}."  # noqa: E501
            )

    partitioned_tensor_nodes = set()
    neuron_partitioned_model = NeuronModel()
    logger.info(f"Num Partitions: {partition_iterator.num_partitions}")
    for i, device_partition_info in enumerate(partition_iterator):
        device_partition, device = device_partition_info

        logger.info(f"Creating Partition #{i+1} for device: {device}")
        (
            func,
            subgraph_input_names,
            subgraph_input_class_flags,
            subgraph_output_names,
        ) = create_subgraph(
            device_partition,
            partitioned_tensor_nodes,
            neuron_ts_graph,
            f"{device}_subgraph_{i}",
            is_final_subgraph=(i == partition_iterator.num_partitions - 1),
        )

        # appropriately trace subgraph for intended device
        traced_partition_executor = process_subgraph(
            device,
            func,
            neuron_ts_graph,
            subgraph_input_names,
            subgraph_input_class_flags,
            subgraph_output_names,
            trace_kwargs,
        )

        neuron_partitioned_model.add_subgraph_partition(
            traced_partition_executor,
            subgraph_input_names,
            subgraph_input_class_flags,
            subgraph_output_names,
        )
        logger.info("")

    neuron_partitioned_model.register_partitions()
    logger.debug("Tracing NeuronModel")
    traced_neuron_partitioned_model = torch.jit.trace(
        neuron_partitioned_model, ex_input
    )

    logger.debug("Linting Traced Neuron Model, and running cse and dce passes")
    torch._C._jit_pass_lint(traced_neuron_partitioned_model.graph)
    torch._C._jit_pass_cse(traced_neuron_partitioned_model.graph)
    torch._C._jit_pass_dce(traced_neuron_partitioned_model.graph)

    torch._C._jit_pass_inline(traced_neuron_partitioned_model.graph)

    os.remove("orig_trace.pt")

    return traced_neuron_partitioned_model
