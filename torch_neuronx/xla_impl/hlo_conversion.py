# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================
from typing import Any, Callable, Dict, Optional, List, Tuple
import warnings

import torch
from torch_xla.core import xla_model
import torch_xla

import torch_neuronx
from torch_neuronx.pyhlo import hlo_pb2, xla_data_pb2
from torch_neuronx.xla_impl import structure
from torch_neuronx.xla_impl import placement


def xla_trace(
    func: Callable,
    example_inputs: Any,
    states: Optional[List] = None,
    input_output_aliases: Optional[Dict] = None,
) -> Tuple[
    hlo_pb2.HloModuleProto,
    List[str],
    Dict[int, torch.Tensor],
    structure.Flattener,
    structure.Packer,
]:
    """
    Trace and optimize ``func`` (usually a ``torch.nn.Module``, or a training step definition)
    for execution on a lightweight runtime.

    Note to developers: Don't introduce new arguments unless absolutely necessary.

    Debug info:

    Python -> HLO lowering under environment variables ``XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1``
    would populate the ``metadata`` field of each HLO instruction.

    Here is a sample HLO instruction with debugging metadata provided as-is by Google.
    ```
    instructions {
      name: "transpose.7"
      opcode: "transpose"
      shape {
        element_type: F32
        dimensions: 16
        dimensions: 8
        layout {
          minor_to_major: 0
          minor_to_major: 1
          format: DENSE
        }
        is_dynamic_dimension: false
        is_dynamic_dimension: false
      }
      metadata {
        op_type: "aten__permute"
        op_name: "aten__permute"
        source_file: "linear@functional.py"
        source_line: 1848
      }
      dimensions: 1
      dimensions: 0
      id: 7
      operand_ids: 3
      frontend_attributes {
      }
    }
    ```
    """
    xla_device = xla_model.xla_device()
    xla_model.mark_step()

    if not isinstance(example_inputs, tuple):
        example_inputs = (example_inputs,)

    with torch_neuronx.contexts.lowering():
        # Traverse structure of inputs, and move any Tensors to XLA device.
        layout, uniques, constants = structure.extract(example_inputs)
        lookup = {**constants}
        inputs = list()
        for tensor, identifier in uniques.items():
            tensor = tensor.to(xla_device)
            lookup[identifier] = tensor
            inputs.append(tensor)
        example_inputs = structure.pack(layout, lookup)

        # Create a flattener from the input layout to later use in torchscript
        flattener = structure.Flattener(layout)

        named_params = {}
        if input_output_aliases:
            assert isinstance(
                func, torch.nn.Module
            ), "For input_output_aliasing, the input func is expected to be of type torch.nn.Module"
            # Here we get the name of the paramter that the user passed in the
            # input_output_aliases dict. We use this names to get the xla_tensor
            # when the module is moved to xla_device
            for name, parameter in func.named_parameters():
                for inp_param in input_output_aliases:
                    if (
                        torch.equal(inp_param, parameter)
                        and inp_param.data_ptr() == parameter.data_ptr()
                    ):
                        named_params[name] = inp_param

        # Move parameters and states to XLA device
        stateless = states is None
        if isinstance(func, torch.nn.Module) and stateless:
            states = [func]
        if states is not None:
            for state in states:
                placement.move(state, xla_device)

        aliased_inputs = {}
        if input_output_aliases:
            # Get the xla_tensor for the corresponding aliased input.
            for name, parameter in func.named_parameters():
                if name in named_params:
                    aliased_inputs[named_params[name]] = parameter

        # Execute the graph
        outputs = func(*example_inputs)

    # Create structural packer to later use in torchscript. This "presents"
    # the data to back to the application correctly (instead of flattened)
    layout, uniques, constants = structure.extract(outputs)
    tensors, identifiers = zip(*uniques.items())
    packer = structure.Packer(layout, identifiers, constants)

    # Lower the HLO graph
    context = torch_xla._XLAC.lowering.LoweringContext()
    context.build(tensors)

    # Determine which HloModule parameters should be inlined (ie. constants,
    # parameters, buffers). This should NOT include the example inputs.
    parameters = context.parameter_id_tensor_mapping()
    input_parameter_numbers = [context.tensor_parameter_id(tensor) for tensor in inputs]
    exclude = list()
    for index, identifier in enumerate(input_parameter_numbers):
        if identifier == -1:
            tensor = inputs[index]
            warnings.warn(
                f"Received an input tensor that was unused. Tensor will be ignored. "
                f"(index={index}, shape={tensor.shape}, dtype={tensor.dtype})"
            )
            exclude.append(index)
        parameters.pop(identifier, None)

    # Remove unused inputs to ensure linear order (See: linearize_indices)
    input_parameter_numbers = [
        identifier
        for index, identifier in enumerate(input_parameter_numbers)
        if index not in exclude
    ]

    # reordering external inputs and states
    updated_input_output_aliases = {}
    state_parameter_numbers = []
    if input_output_aliases:
        for i, (tensor, output_idx) in enumerate(input_output_aliases.items()):
            state_parameter_numbers.append(
                context.tensor_parameter_id(aliased_inputs[tensor])
            )
            updated_input_output_aliases[output_idx] = i + len(input_parameter_numbers)
    input_parameter_numbers = input_parameter_numbers + state_parameter_numbers

    # If we found any unused inputs, inform the input formatter to ignore these
    if exclude:
        flattener.exclude = exclude

    # Get constructed HloModule protobuf from LowingContext
    hlo = context.hlo()
    hlo_module = hlo_pb2.HloModuleProto()
    hlo_module.ParseFromString(hlo)

    # Optimize the HloModule
    hlo_opt = HloOptimizer(hlo_module)

    # Keep track of input parameter names
    input_parameter_names = [
        hlo_opt.hlo_module.host_program_shape.parameter_names[index]
        for index in input_parameter_numbers
    ]

    if stateless:
        constant_parameter_numbers = list(
            set(parameters.keys()) - set(input_parameter_numbers)
        )
        input_parameter_names, constant_parameter_tensors = hlo_opt.linearize_indices(
            parameters, input_parameter_numbers, constant_parameter_numbers
        )
        hlo_opt.dead_code_elimination()

    return (
        hlo_opt.hlo_module,
        input_parameter_names,
        constant_parameter_tensors,
        flattener,
        packer,
        updated_input_output_aliases,
    )


class OpCode:
    # https://github.com/tensorflow/tensorflow/blob/v2.8.0/tensorflow/compiler/xla/service/hlo_opcode.h
    kConstant = "constant"
    kParameter = "parameter"
    kCustomCall = "custom-call"


class HloOptimizer:
    def __init__(self, hlo_module):
        self.hlo_module = hlo_module
        self.id_to_computation = {cpt.id: cpt for cpt in hlo_module.computations}
        self.id_to_inst = {inst.id: inst for inst in self.entry_instructions}
        self.param_insts = self._param_insts()

    @property
    def entry_computation(self):
        return self.id_to_computation[self.hlo_module.entry_computation_id]

    @property
    def entry_instructions(self):
        return self.entry_computation.instructions

    def _param_insts(self):
        # Create a mapping from parameter number to the HLO instruction
        param_insts_mapping = dict()
        for instruction in self.entry_instructions:
            if instruction.opcode == OpCode.kParameter:
                param_insts_mapping[instruction.parameter_number] = instruction
        return param_insts_mapping

    def linearize_indices(
        self,
        parameters: Dict[int, torch.Tensor],
        input_parameter_numbers: List[int],
        constant_parameter_numbers: List[int],
    ) -> Tuple[List[str], Dict[str, torch.Tensor]]:
        """
        Linearize the instruction parameter numbers such that "real" input
        parameter indices come before constant parameter indices.

        Secondly, ensure parameter names occur in linear order [p0, p1, .. pn]
        instead of automatic naming [p212.746, p227.837, p0.1, ...]
        """

        # First loop through inputs to make sure they appear first
        order = {}
        index = 0
        input_parameter_names = []
        for parameter_number in input_parameter_numbers:
            inst = self.param_insts[parameter_number]
            order[inst.parameter_number] = index
            inst.parameter_number = index
            name = "p" + str(index)
            inst.name = name
            input_parameter_names.append(name)
            index += 1

        # Loop through constants and map the index to the tensor
        constant_parameter_tensors = dict()
        for parameter_number in constant_parameter_numbers:
            inst = self.param_insts[parameter_number]
            order[inst.parameter_number] = index
            inst.parameter_number = index
            inst.name = "p" + str(index)
            tensor = parameters[parameter_number]
            constant_parameter_tensors[index] = tensor
            index += 1

        # Create new program shape in corrected order
        original_shape = self.hlo_module.host_program_shape
        program_shape = xla_data_pb2.ProgramShapeProto()
        program_shape.CopyFrom(original_shape)
        for src, dst in order.items():
            program_shape.parameters[dst].CopyFrom(original_shape.parameters[src])
            program_shape.parameter_names[dst] = self.param_insts[src].name

        # Both top-level hlo module and root computation must have identical program shape
        module_params = len(self.hlo_module.host_program_shape.parameters)
        entry_params = len(self.entry_computation.program_shape.parameters)
        assert module_params == entry_params, (
            f"Mismatching parameter count in HloModule ({module_params}) and "
            f"entry computation ({entry_params})"
        )

        # Update both hlo module and root computation program shapes
        self.hlo_module.host_program_shape.CopyFrom(program_shape)
        self.entry_computation.program_shape.CopyFrom(program_shape)

        return input_parameter_names, constant_parameter_tensors

    def dead_code_elimination(self):
        # TODO: replace with C++ HLO DCE once we are able to call it

        def keep(inst):
            is_io = (
                inst.opcode == OpCode.kParameter
                or inst.id == self.entry_computation.root_id
            )
            return is_io or inst.custom_call_has_side_effect

        visited_ids = set()
        stack = [inst.id for inst in self.entry_instructions if keep(inst)]
        while stack:
            iid = stack.pop()
            if iid in visited_ids:
                continue
            visited_ids.add(iid)
            inst = self.id_to_inst[iid]
            stack.extend(inst.operand_ids)
        visited_instructions = [
            inst for inst in self.entry_instructions if inst.id in visited_ids
        ]
        while self.entry_instructions:
            self.entry_instructions.pop()
        self.entry_instructions.extend(visited_instructions)
