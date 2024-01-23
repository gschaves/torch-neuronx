# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================
import contextlib
from distutils import spawn
import json
import os
import shlex
import subprocess
import tempfile
import pathlib
from typing import List, Any, Union, Callable, Optional, Dict, Iterable, Tuple
import logging
import warnings

import torch
import numpy as np

from torch_neuronx.proto import metaneff_pb2
from torch_neuronx.pyhlo import xla_data_pb2, hlo_pb2
from torch_neuronx.xla_impl.hlo_conversion import xla_trace
from torch_neuronx.xla_impl.options import OptionsDefault, Options
from torch_neuronx.xla_impl.partitioner import PartitionerConfig
import torch_neuronx
from torch_neuronx.xla_impl import structure
from torch_neuronx.xla_impl import placement


logger = logging.getLogger("Neuron")


@contextlib.contextmanager
def nullcontext():
    """
    A context which does nothing.

    Used as a stand-in for a normal context manager, when a particular
    block of code is only sometimes used with a normal context manager.
    """
    yield


@contextlib.contextmanager
def revert_device_placement(func):
    """
    A context that reverts a function's device placement from the XLA device
    back to the function's original device.

    Args:
        func: The function to revert device placement for.
    """
    device = None

    if isinstance(func, torch.nn.Module):
        parameter = next(func.parameters(), None)
        if parameter is not None:
            device = parameter.device
    try:
        yield
    finally:
        if device is not None:
            placement.move(func, device)


class NeuronModule(torch.nn.Module):
    """
    A torch module that encapsulates a compiled Neuron graph.

    Args:
        neff: A compiled model artifact.
        metaneff: Serialized metadata regarding the NEFF interface.
        flattener: Extracts tensors from an expected hierarchical input structure.
        packer: Packs tensors into an expected hierarhcical output structure.
    """

    def __init__(
        self,
        neff: str,
        metaneff: str,
        flattener: structure.Flattener,
        packer: structure.Packer,
        initial_states: List = None,
    ) -> None:
        super().__init__()
        self.model = torch.classes.neuron.Model(neff, metaneff)
        self.flattener = flattener
        self.packer = packer
        self.states = (
            torch.nn.ParameterList(
                [
                    torch.nn.Parameter(tensor, requires_grad=False)
                    for tensor in initial_states
                ]
            )
            if initial_states
            else torch.nn.ParameterList([])
        )

    def forward(self, *tensors):
        inputs = self.flattener(tensors)
        inputs.extend(self.states)
        result = torch.ops.neuron.forward_v2(inputs, self.model)
        return self.packer(result)


def hlo_root_computation(hlo: hlo_pb2.HloProto) -> hlo_pb2.HloComputationProto:
    """
    Retrieve the main computation (external interface) from the HloProto

    Args:
        hlo: The HloProto to extract root computation from.

    Returns:
        The computation interface.
    """
    for computation in hlo.computations:
        if computation.id == hlo.entry_computation_id:
            return computation


def hlo_metaneff(
    hlo: hlo_pb2.HloProto, input_parameter_names: List[str], input_output_aliases: Dict
) -> metaneff_pb2.MetaNeff:
    """
    Build a tensor metadata message from the root computation of an HloProto.

    This is used by the runtime to validate input tensor shapes, names, and
    types. Secondly the resulting message can be augmented to contain metadata
    about the graph being executed and the pipeline configuration.

    Args:
        hlo: The HloProto to extract the interface metadata from.
        input_parameter_names: inputs that should be added to the metaneff
        input_output_aliases: alias mapping between the inputs and outputs

    Returns:
        The MetaNeff message containing interface metadata.
    """
    # TODO: See if metaneff can be eliminated in favor of
    #       `nrt_get_model_tensor_info`. Right now this does not seem possible
    #       since the tensor info API requires that a model was successfully
    #       loaded to be able to call it. The metaneff is used during tracing
    #       prior to an `nrt_load` to emit placeholder tensors.

    computations = hlo_root_computation(hlo)
    dtypes = {
        xla_data_pb2.F32: metaneff_pb2.MetaTensor.DataType.FLOAT,
        xla_data_pb2.F64: metaneff_pb2.MetaTensor.DataType.DOUBLE,
        xla_data_pb2.BF16: metaneff_pb2.MetaTensor.DataType.BFLOAT16,
        xla_data_pb2.F16: metaneff_pb2.MetaTensor.DataType.FLOAT16,
        xla_data_pb2.U8: metaneff_pb2.MetaTensor.DataType.UINT8,
        xla_data_pb2.S8: metaneff_pb2.MetaTensor.DataType.INT8,
        xla_data_pb2.U16: metaneff_pb2.MetaTensor.DataType.UINT16,
        xla_data_pb2.S16: metaneff_pb2.MetaTensor.DataType.INT16,
        xla_data_pb2.U32: metaneff_pb2.MetaTensor.DataType.INT32,
        xla_data_pb2.S32: metaneff_pb2.MetaTensor.DataType.INT32,
        xla_data_pb2.U64: metaneff_pb2.MetaTensor.DataType.INT64,
        xla_data_pb2.S64: metaneff_pb2.MetaTensor.DataType.INT64,
        xla_data_pb2.C64: None,
        xla_data_pb2.C128: None,
        xla_data_pb2.PRED: metaneff_pb2.MetaTensor.DataType.BOOL,
    }

    metaneff = metaneff_pb2.MetaNeff()
    program_shape = computations.program_shape

    # Create mapping from parameter name to parameter metadata
    parameter_name_to_metadata = {}
    for name, metadata in zip(program_shape.parameter_names, program_shape.parameters):
        parameter_name_to_metadata[name] = metadata

    # Iterate through input parameters to ensure correct input order at runtime
    for index, name in enumerate(input_parameter_names):
        metadata = parameter_name_to_metadata[name]

        shape = list(metadata.dimensions)
        dtype = metadata.element_type

        tensor = metaneff.input_tensors.add()
        tensor.name = f"input{index}".encode(
            "utf8"
        )  # Needs to be `input#` to avoid a `ddrs_create_lookup_key` error
        tensor.shape[:] = shape
        tensor.data_type = dtypes[dtype]

    for index, metadata in enumerate(program_shape.result.tuple_shapes):
        shape = list(metadata.dimensions)
        dtype = metadata.element_type

        tensor = metaneff.output_tensors.add()
        tensor.name = f"output{index}".encode("utf8")
        tensor.shape[:] = shape
        tensor.data_type = dtypes[dtype]

    for output_idx, input_idx in input_output_aliases.items():
        metaneff.output_aliases_to[output_idx] = input_idx

    return metaneff


def hlo_compile(
    filename: Union[str, pathlib.Path],
    compiler_workdir: Union[str, pathlib.Path],
    compiler_args: Optional[Union[List[str], str]] = None,
) -> str:
    """
    Compiles a serialized HloProto into a NEFF using `neuronx-cc`

    Compiler CLI Reference: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/neuronx-cc/api-reference-guide/neuron-compiler-cli-reference-guide.html  # noqa: E501

    Args:
        filename: The filename for the serialized HloProto or the path to the
            `model` folder
        compiler_workdir: The directory to save any compiler outputs to.
        compiler_args: Additional compiler arguments.

    Returns:
        The filename of the compiled NEFF.
    """
    # Ensure neuronx-cc is installed
    try:
        import neuronxcc
    except ImportError as e:
        raise RuntimeError(
            "neuronx-cc is not installed.\n"
            "neuronx-cc can be installed using:\n"
            "python3 -m pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com neuronx-cc"
        ) from e

    # NOTE: Checking the installation above isolates of all neuronx-cc usage to
    #       this function. This is convenient for mocking/patching for tests.

    neff_filename = os.path.join(compiler_workdir, "graph.neff")

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
        filename,
        "--framework",
        "XLA",
        "--target",
        "trn1",
        "--output",
        neff_filename,
    ]
    command.extend(compiler_args)

    # Write the command that produces the NEFF
    command_filename = os.path.join(compiler_workdir, "command.txt")
    with open(command_filename, "w") as f:
        stripped = [os.path.basename(str(item)) for item in command]
        f.write(" ".join(stripped))

    status = subprocess.run(command).returncode
    if status != 0:
        if status == -9:
            logger.warning(
                "The neuronx-cc (neuron compiler) process was killed (SIG_KILL).  "
                "This typically happens when there is insufficient memory to compile and the linux "
                "Out Of Memory (OOM) killer terminates the compiler. "
                "Consider trying compilation on an instance with more memory"
            )
        elif status == -6:
            logger.warning(
                "The neuronx-cc (neuron compiler) process aborted (SIG_ABORT). "
                "This is likely due to an unexpected condition internally (a bug).  "
                "Please lodge an issue at 'https://github.com/aws/aws-neuron-sdk/issues'"
            )
        elif status == -11:
            logger.warning(
                "The neuronx-cc (neuron compiler) crashed (SEGFAULT). "
                "This is likely due to a bug in the compiler.  "
                "Please lodge an issue at 'https://github.com/aws/aws-neuron-sdk/issues'"
            )

        raise RuntimeError(f"neuronx-cc failed with {status}")

    return neff_filename


def coalesce(first, second):
    if first is not None:
        return first
    return second


def trace(
    func: Union[Callable, torch.nn.Module],
    example_inputs: Any,
    *_,
    input_output_aliases: Dict = {},
    compiler_workdir: Optional[Union[str, pathlib.Path]] = None,
    compiler_args: Optional[Union[List[str], str]] = None,
    partitioner_config: Optional[PartitionerConfig] = None,
    **kwargs,
) -> torch.jit.ScriptModule:
    """
    Trace a torch module/function to produce a compiled Neuron ScriptModule.

    This uses torch-xla to extract the computation graph. The input `func` and
    `example_inputs` must be able to be moved to the XLA device.

    The resulting module can used with `torch.jit.save` and `torch.jit.load`

    NOTE: Please use keyword arguments for all arguments after `example_inputs`.
    Ex: `torch_neuronx.trace(func,example_inputs,compiler_workdir="some_dir") #valid`
        `torch_neuronx.trace(func,example_inputs,"some_dir") #invalid`

    Args:
        func: A module or function which defines a torch model or computation.
        example_inputs: An example set of inputs which will be passed to the
            `func` during tracing.
        input_output_aliases: alias mapping between the inputs and outputs
        compiler_workdir: The directory to save any compiler outputs to.
        compiler_args: Additional compiler arguments.
        partitioner_config: A PartitionerConfig object, which can be optionally
        supplied if there are unsupported ops in the model that need to be
        partitioned out to CPU.

    Returns:
        A Module where the HLO computation is a fused neuron::foward operation.
    """
    # Create a temporary directory for artifacts if none is specified
    context = nullcontext()
    if compiler_workdir is None:
        context = tempfile.TemporaryDirectory()
        compiler_workdir = context.name

    # code to handle deprecation of states and options
    # we're also transitioning towards making the user
    # not using positional args after example_inputs
    states = None
    options = None
    if _:
        num_args = 2 + len(_)
        warnings.warn(
            f"Received {num_args} positional arguments but expected 2. "
            "Use of positional arguments after `func` and `example_inputs` is "
            "deprecated. Please specify keyword arguments instead.",
            category=DeprecationWarning,
        )
        if len(_) > 5:
            raise RuntimeError(
                f"Received {num_args} positional arguments but expected 2"
            )
        empty = (None,) * (5 - len(_))
        states, aliases, workdir, args, options = _ + empty
        input_output_aliases = coalesce(aliases, input_output_aliases)
        compiler_workdir = coalesce(workdir, compiler_workdir)
        compiler_args = coalesce(args, compiler_args)

    options = kwargs.pop("options", options)
    if options is not None:
        warnings.warn("Argument `options` is deprecated", category=DeprecationWarning)

    states = kwargs.pop("states", states)
    if states is not None:
        warnings.warn("Argument `states` is deprecated", category=DeprecationWarning)

    if kwargs:
        warnings.warn(
            f"Unexpected keyword arguments: {list(kwargs.keys())}", category=UserWarning
        )

    if partitioner_config:
        if partitioner_config.trace_kwargs is None and (
            compiler_args or compiler_workdir
        ):
            partitioner_config.trace_kwargs = {
                "compiler_args": compiler_args,
                "compiler_workdir": compiler_workdir,
            }

        return torch_neuronx.partition(
            func, example_inputs, **(partitioner_config.__dict__)
        )

    with context:
        neff_filename, metaneff, flattener, packer = _trace(
            func,
            example_inputs,
            states,
            input_output_aliases,
            compiler_workdir,
            compiler_args,
            options,
        )
        return create_neuron_model(
            neff_filename,
            metaneff,
            flattener,
            packer,
            example_inputs,
            input_output_aliases,
        )


def create_neuron_model(
    neff_filename: Union[str, pathlib.Path],
    metaneff: str,
    flattener: structure.Flattener,
    packer: structure.Packer,
    example_inputs: Any,
    input_output_aliases: Dict,
):
    with open(neff_filename, "rb") as handle:
        neff = handle.read()

    initial_states = tuple(input_output_aliases.keys())
    result = NeuronModule(neff, metaneff, flattener, packer, initial_states)
    with torch_neuronx.contexts.disable_nrt_load():
        # NOTE: Turn on strict=False by default since the only reason we
        #       trace here is to ensure that data structures are correctly
        #       represented in torchscript (input/output)
        return torch.jit.trace(result, example_inputs, strict=False)


def _trace(
    func: Union[Callable, torch.nn.Module],
    example_inputs: Any,
    states=None,
    input_output_aliases: Dict = {},
    compiler_workdir: Optional[Union[str, pathlib.Path]] = None,
    compiler_args: Optional[Union[List[str], str]] = None,
    options: Union[Iterable[Options], Options] = None,
) -> Union[str, str, structure.Flattener, structure.Packer]:
    # Convert options to an iterable if it's not one
    if not isinstance(options, (list, tuple)):
        options = (options,)

    # Convert the function to a HloProto message
    with torch_neuronx.contexts.mock_neuron_cores(), revert_device_placement(func):
        (
            hlo,
            input_parameter_names,
            constant_parameter_tensors,
            flattener,
            packer,
            updated_input_output_aliases,
        ) = xla_trace(
            func,
            example_inputs,
            states,
            input_output_aliases,
        )

    # Create compiler directory if it does not exist
    if not os.path.exists(compiler_workdir):
        os.makedirs(compiler_workdir, exist_ok=True)

    # Set up compiler_workdir folder structure
    model_dir = os.path.join(compiler_workdir, "model")
    os.makedirs(model_dir, exist_ok=True)
    hlo_filename = os.path.join(model_dir, "graph.hlo")

    # Write weights to disk
    weight_paths = write_params(model_dir, constant_parameter_tensors)

    table = {
        "model_files": "graph.hlo",
        "version": "1.0",
        "weights": weight_paths,
    }
    with open(f"{model_dir}/metadata.json", "w") as f:
        json.dump(table, f, indent=4)

    # Write optimized HLO to disk
    with open(hlo_filename, "wb") as handle:
        handle.write(hlo.SerializeToString())

    # Compile HLO to NEFF
    neff_filename = hlo_compile(model_dir, compiler_workdir, compiler_args)

    metaneff = hlo_metaneff(hlo, input_parameter_names, updated_input_output_aliases)

    return neff_filename, metaneff.SerializeToString(), flattener, packer


def write_params(
    directory: Union[str, pathlib.Path], weights: Dict[Union[str, int], torch.Tensor]
) -> None:
    # Create the directory to store weights
    os.makedirs(f"{directory}/weights/", exist_ok=True)

    # Write tensor data to disk
    for name, weight in weights.items():
        # Represent bfloat16 as 2-byte void numpy type to allow serialization
        if weight.dtype == torch.bfloat16:
            weight = weight.view(torch.int16)
            weight = weight.numpy()
            weight = weight.view("|V2")
        else:
            weight = weight.numpy()
        np.save(f"{directory}/weights/{name}.npy", weight)

    # Write mapping file. Paths are relative to the directory
    weight_paths = {name: f"weights/{name}.npy" for name in weights}
    return weight_paths


def move_trace_to_device(trace, device_id):
    runtime = torch.classes.neuron.Runtime()
    runtime.initialize()
    runtime.set_default_neuron_cores(device_id, -1)
    # Currently we have to explicitly move the state params to device because
    # the map_location feature for privateuseone device is only added recently
    # in torch. TODO: replace this with map_location feature in jit.load once
    # this commit is pulled: https://github.com/pytorch/pytorch/commit/da322ea874a5abeb2f10f9e4c4bb8414892cb0d2
    # Ideally trace.to(device) would work(and it works when we trace and move) but it
    # fails when we load a trace module and then try to move it to device. This is because,
    # when we save the trace, the torch.save doesn't preserve the Param attribute and when
    # we try to move the loaded trace, since we don't have the param attribute to tensor,
    # it crashes in .to()
    for name, param in trace.states._parameters.items():
        trace.states._parameters[name] = param.to(f"privateuseone:{device_id}")
