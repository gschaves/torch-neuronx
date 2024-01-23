# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================
import os
import ctypes
import functools
import math
import operator
from functools import wraps
import numpy as np
import torch
from torch.nn import functional as F
from torch_xla.core import xla_builder, xla_model
from torch_xla.core.xla_builder import Type
from torch_neuronx.xla_impl.base import (
    xla_call,
    xla_hlo_call,
    override,
    lazy_override,
    register_hook_function,
)
from torch_neuronx.xla_impl.custom_call_targets import (
    AwsNeuronCustomOp,
    AwsNeuronGelu,
    AwsNeuronGeluBackward,
    AwsNeuronNearestNeighbor2d,
    AwsNeuronNearestNeighbor2dBackward,
    AwsNeuronArgMax,
    AwsNeuronArgMin,
    AwsNeuronSoftmax,
    AwsNeuronSoftmaxBackward,
    AwsNeuronTopK,
    AwsNeuronRmsNorm,
    AwsNeuronRmsNormBackward,
    TorchNeuronUnloadPriorModels,
    AwsNeuronTritonKernel,
)
from torch_neuronx import pyhlo  # for CustomCPP to build HloShapes from a side channel
from torch_neuronx.pyhlo import xla_data_pb2

import base64
import json
import inspect
from typing import List, Tuple


def _build_maps(func, input_names):
    """
    Return a table where key is idx of the arg, value is the
    name of the arg
    """
    sig = inspect.signature(func)
    i = 0
    result = {}
    for arg in sig.parameters:
        if arg in input_names:
            result[i] = arg
        i += 1
    return result


class TritonKernel:
    """
    This class lowers a user defined compiler kernel to HLO-IR

    This is *experimental* FAL binding for the triton-like API for compiler
    to program Neuron Device directly.

    Parameters:
    func: the function of the triton kernel defition
    grid: launch grid configuration
    input_names: List[str], literal names of the inputs
    output_names: List[str], literal names of the outputs
    kernel_attrs: List[str], string attribute to control code injection point in compiler

    Example:
    @jit(kernel_attr=('tiled'), input=('a', 'b'), ouputs=('c'))
    def dummy(a, b, c):
        c.write(a + b)

    a = torch.ones((20, 30)).to(device=device)
    b = torch.rand((20, 30)).to(device=device)
    c = torch.zeros((20, 30)).to(device=device)
    d = torch.empty((20, 30)).to(device=device)

    # out is semantically c. If we do not use the return value of this call, XLA
    # thinks that this call has no effect at all.
    out = dummy(a, b, c)
    e = out + d

    Then it will generate HLO like:
    ENTRY SyncTensorsGraph.24 {
    p0.2 = f32[20,30]{1,0} parameter(0)
    p1.2 = f32[20,30]{1,0} parameter(1)
    p2.2 = f32[20,30]{1,0} parameter(2)
    p3 = f32[20,30]{1,0} parameter(3)
    custom-call.2 = f32[20,30]{1,0} custom-call(p0.2, p1.2),
        custom_call_target="AwsNeuronTritonKernel", api_version=API_VERSION_UNSPECIFIED,
        backend_config="{\"func_literal\":
        \"def dummy(a, b, c):\\n    c.write(a + b)\\n    # d.write(a - b)\\n\",
        \"kernel_attrs\": [], \"grid\": [], \"inputs\": [0, 1], \"outputs\": [2]}"
    .....

    """

    def __init__(self, func, kernel_attrs=("tiled")):
        self.func = func
        self.func_literal = None
        self.grid = ()
        self.kernel_attrs = kernel_attrs
        self.func_name = func.__name__
        self.input_names = None
        self.output_names = None
        self.input_map = None
        self.output_map = None

    def _gen_maps(self, input_names, output_names):
        self.input_names = input_names
        self.output_names = output_names
        self.input_map = _build_maps(self.func, input_names)
        self.output_map = _build_maps(self.func, output_names)

    def _extract_args(self, args):
        inputs = []
        outputs = []
        for i, arg in enumerate(args):
            if i in self.input_map:
                inputs.append(arg)
            else:
                outputs.append(arg)
            i += 1
        assert len(args) == len(self.input_map) + len(
            self.output_map
        ), "Insufficient number of argument to call kernel"
        return inputs, outputs

    def dump_config(self):
        config = {}
        config["func_literal"] = self.func_literal
        config["kernel_attrs"] = self.kernel_attrs
        config["grid"] = self.grid
        config["func_name"] = self.func_name
        return base64.b64encode(json.dumps(config).encode("utf-8")).decode("utf-8")

    def __getitem__(self, grid):
        if not isinstance(grid, (tuple, list)):
            grid = [grid]

        self.grid = grid
        return self

    @xla_hlo_call
    def call_impl(*args, input_len, config_str):
        assert (
            len(args) > input_len
        ), "Logic Fault: Triton must have at least #input_len of arguments provided"

        # steal the scribe
        scribe = args[0].scribe

        output_tys = args[input_len:][:]
        input_tys = args[:input_len][:]
        if len(output_tys) == 1:
            return output_tys[0].CustomCall(
                *input_tys,
                backend_config=str.encode(config_str),
                custom_call_target=AwsNeuronTritonKernel,
            )
        else:
            return scribe.tuple(*output_tys).CustomCall(
                *input_tys,
                backend_config=str.encode(config_str),
                custom_call_target=AwsNeuronTritonKernel,
            )

    @staticmethod
    def _translate_to_neuron_dtype(_dtype):
        """
        Translate a pytorch dtype to neuron specific dtype representation in numpy
        """
        if _dtype == torch.bfloat16:
            # This is a hard-coded assumption that bfloat16 must be represented as '|V2'
            return np.dtype("|V2")
        # For other dtype that is common with numpy, use builtin pytorch to do the translation
        return torch.empty(1, dtype=_dtype).numpy().dtype

    def __call__(self, *args):
        try:
            from neuronxcc.triton import decltensor, trace
        except ImportError as e:
            raise RuntimeError(
                "neuronx-cc is not installed.\n"
                "neuronx-cc can be installed using:\n"
                "python3 -m pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com neuronx-cc"
            ) from e
        virtual_tensors = [
            decltensor(
                shape=tuple(t.shape), dtype=self._translate_to_neuron_dtype(t.dtype)
            )
            for t in args
        ]
        traced = trace(
            func=self.func, grid=self.grid, kernel_attrs=self.kernel_attrs
        ).specialize(*virtual_tensors)
        self._gen_maps(
            [t.name for t in traced.inputs], [t.name for t in traced.outputs]
        )
        self.func_literal = traced.serialize_ir_string(f"{self.func_name}_ir")

        inputs, outputs = self._extract_args(args)
        # xla_hlo_call assumes that first few args are tensors, pass in inputs first, followed by outputs
        return self.call_impl(
            *inputs, *outputs, input_len=len(inputs), config_str=self.dump_config()
        )


def triton_jit(kernel_attrs=()):
    def captured_func(func):
        return TritonKernel(func, kernel_attrs)

    return captured_func


class CustomCppOp:
    """
    This class lowers a Neuron customOp to HLO-IR

    Parameters:
    `fn_name` : compute function name
    `shape_fcn` : python function to produce shapes/dtypes
    `compute_libs` : absolute paths to compute libraries, separated by comma
    `lib_md5` : compute library md5 hash
    `ucode_ver` : ucode version
    `isa_ver` : isa version

    Notes: output data type and output size are necessary because XLA
           deduces shape sizes/types as it builds the graph.   Because the C++
           is not executed until runtime, we do not know the tensor output
           sizes/dtype until /runtime/... while XLA needs them at compile time.
           (Note: someday we may be do a "sample run" of the C++ code and infer
           the sizes, but not today)
           To work around this, we ask the user to provide funcs that calcualte
           the output tensors datatype and shape.

          A Custom operator may only be invoked in "graph" code (b/c it
          lowers to HLO-IR).
          TODO: enforce that __call__ is invoked from a safe place (e.g. class
               that derived from Autograd.Function)
               perhaps with https://docs.python.org/3/library/inspect.html ?
    """

    def __init__(self, fn_name, shape_fcn, compute_libs, lib_md5, ucode_ver, isa_ver):
        self.low_fn = shape_fcn
        self.lib_paths = compute_libs
        self.fn_name = fn_name
        self.lib_md5 = lib_md5
        self.ucode_version = ucode_ver
        self.isa_version = isa_ver

    def conv_dtype(odtype):
        try:
            oprimtype = pyhlo.constant.serialize_torch._torch_dtype_to_primitive_type[
                odtype
            ]
            return oprimtype
        except KeyError:
            assert False, f"Unrecognized torch.dtype {odtype} for output dtype arg"

    def sanitize_inputs(self, inputs):
        fixed_inputs = []
        for i in inputs:
            if isinstance(i, list):
                raise RuntimeError(
                    "Found unsupported input arg type vector. Only fixed length types are supported, please use a tuple instead."  # noqa: E501
                )
            elif isinstance(i, tuple):
                raise RuntimeError("Tuple support not implemented yet")
            elif isinstance(i, (int, float)):
                # convert scalars to tensors
                fixed_inputs.append(torch.tensor(i, device=xla_model.xla_device()))
            elif torch.is_tensor(i):
                fixed_inputs.append(i)
            else:
                itype = type(i)
                raise RuntimeError(f"Found unsupported input type: {itype}")
        return fixed_inputs

    def process_outputs(self, outs):
        if isinstance(outs, list):
            raise RuntimeError(
                "Found unsupported vector return type. Only fixed length return types are supported, please use a tuple instead."  # noqa: E501
            )
        elif isinstance(outs, tuple):
            osizes = []
            odtypes = []
            # prepare each entry in the tuple
            for o in outs:
                if torch.is_tensor(o):
                    osizes.append(o.size())
                    odtypes.append(o.dtype)
                else:
                    # convert scalars to tensors
                    osizes.append(torch.Size([]))
                    if isinstance(o, int):
                        odtypes.append(torch.int64)
                    elif isinstance(o, float):
                        odtypes.append(torch.float64)
                    else:
                        otype = type(o)
                        raise RuntimeError(f"Found unsupported odtype: {otype}")
        elif torch.is_tensor(outs):
            osizes = outs.size()
            odtypes = outs.dtype
        else:
            otype = type(outs)
            raise RuntimeError(f"Found unsupported odtype: {otype}")
        return (osizes, odtypes)

    @xla_hlo_call
    def call_impl(
        *inputs, fn_name, lib_paths, lib_md5, ucode_ver, isa_ver, odtypes, osizes
    ):
        opaque_str = f"version=1.0;name={fn_name};path={lib_paths};hash={lib_md5};ulib_to_ucode_version={ucode_ver};ulib_to_isa_version={isa_ver};"  # noqa: E501

        # steal the scribe
        assert len(inputs) > 0, "CustomOp must have at least one input"
        scribe = inputs[0].scribe

        # handle tuples diff than scalars/tensors
        if isinstance(osizes, list):
            targs = []
            # build the set of size/dtype for each element in the tuple
            for i, s in enumerate(osizes):
                oprimtype = CustomCppOp.conv_dtype(odtypes[i])
                output_hlo_shape = pyhlo.scribe.HloShape(scribe, oprimtype)
                targs.append(output_hlo_shape.dtype[s])
            # splat the tuple arguments and make custom call
            return scribe.tuple(*targs).CustomCall(
                *inputs,
                backend_config=str.encode(opaque_str),
                custom_call_target=AwsNeuronCustomOp,
            )
        else:
            oprimtype = CustomCppOp.conv_dtype(odtypes)
            output_hlo_shape = pyhlo.scribe.HloShape(scribe, oprimtype)
            return (
                (output_hlo_shape)
                .dtype[osizes]
                .CustomCall(
                    *inputs,
                    backend_config=str.encode(opaque_str),
                    custom_call_target=AwsNeuronCustomOp,
                )
            )

    def __call__(self, *inputs):
        fixed_inputs = self.sanitize_inputs(inputs)

        # execute lowering function to produce output shapes/dtypes
        outs = self.low_fn(*inputs)

        osizes, odtypes = self.process_outputs(outs)

        return self.call_impl(
            *fixed_inputs,
            fn_name=self.fn_name,
            lib_paths=self.lib_paths,
            lib_md5=self.lib_md5,
            ucode_ver=self.ucode_version,
            isa_ver=self.isa_version,
            osizes=osizes,
            odtypes=odtypes,
        )


@override("torch._C._nn.gelu")
class Gelu(torch.autograd.Function):
    @xla_hlo_call
    def forward_impl(input):
        return input.dtype[input.sizes].CustomCall(
            input, custom_call_target=AwsNeuronGelu
        )

    @xla_hlo_call
    def backward_impl(grad, input):
        shape = input.dtype[input.sizes]
        dinput = shape.CustomCall(input, custom_call_target=AwsNeuronGeluBackward)
        return shape.Multiply(grad, dinput)

    @staticmethod
    def forward(ctx, input, approximate="none"):
        ctx.save_for_backward(input)
        return Gelu.forward_impl(input)

    @staticmethod
    def backward(ctx, grad):
        (input,) = ctx.saved_tensors
        return Gelu.backward_impl(grad, input), None


class FunctionalGelu:
    @register_hook_function
    def hook_function():
        orig_gelu = torch.nn.functional.gelu

        @wraps(orig_gelu)
        def gelu(input, approximate="none"):
            if not xla_model.is_xla_tensor(input):
                return orig_gelu(input, approximate=approximate)
            if approximate != "none":
                raise RuntimeError(
                    f"Gelu with approximate={approximate} is not" "supported"
                )
            return Gelu.apply(input, approximate)

        torch.nn.functional.gelu = gelu


@override("torch.randn")
class RandN:
    _XLA_PT_TYPE_MAP = {
        Type.F32: torch.float32,
        Type.F64: torch.float64,
        Type.BF16: torch.bfloat16,
        Type.F16: torch.float16,
        Type.U8: torch.uint8,
        Type.S8: torch.int8,
        Type.U16: torch.int16,
        Type.S16: torch.int16,
        Type.U32: torch.int32,
        Type.S32: torch.int32,
        Type.U64: torch.int64,
        Type.S64: torch.int64,
        Type.C64: torch.complex64,
        Type.C128: torch.complex128,
        Type.PRED: torch.bool,
    }
    _PT_XLA_TYPE_MAP = {value: key for key, value in _XLA_PT_TYPE_MAP.items()}

    @xla_hlo_call
    def impl(mu, sigma, size, dtype):
        dtype = getattr(mu.scribe, RandN._PT_XLA_TYPE_MAP[dtype])
        return dtype[size].Rng(mu, sigma, distribution=xla_data_pb2.RNG_NORMAL)

    @classmethod
    def apply(
        cls,
        *size,
        out=None,
        dtype=None,
        layout=torch.strided,
        device=None,
        requires_grad=False,
    ):
        if (
            out is not None
            or dtype not in cls._PT_XLA_TYPE_MAP
            or layout is not torch.strided
            or device is None
            or device.type != "xla"
            or requires_grad
        ):
            return torch.randn(
                *size,
                out=out,
                dtype=dtype,
                layout=layout,
                device=device,
                requires_grad=requires_grad,
            )
        mu = torch.tensor(0.0, dtype=dtype, device=device)
        sigma = torch.tensor(1.0, dtype=dtype, device=device)
        return cls.impl(mu, sigma, size=size, dtype=dtype)


class SimpleCrossEntropyLoss(torch.autograd.Function):
    """
    Forward reference:
        (pytorch) ATen/native/LossNLL.cpp, function "cross_entropy_loss"
    Backward reference:
        torch_xla/csrc/nll_loss.cpp, function "BuildNllLossBackward"
        torch_xla/csrc/softmax_builder.cpp, function "BuildLogSoftmaxGrad"
    """

    REDUCTION = "mean"
    SOFTMAX_DIM = 1
    # TODO: make it smaller when 2-D scatter is ready
    DENSE_SCATTER_FACTOR = int(os.environ.get("NEURON_DENSE_SCATTER_FACTOR", 999999999))

    @lazy_override("torch.nn.CrossEntropyLoss")
    def gen_override():
        cls = SimpleCrossEntropyLoss

        class Wrapper(torch.nn.CrossEntropyLoss):
            def forward(self, input, target):
                if (
                    xla_model.is_xla_tensor(input)
                    and xla_model.is_xla_tensor(target)
                    and len(input.shape) == 2
                    and len(target.shape) == 1
                    and self.weight is None
                    and self.reduction == cls.REDUCTION
                ):
                    return cls.apply(input, target, self.ignore_index)
                return super().forward(input, target)

        return Wrapper

    @xla_call
    def forward_impl(logits, target, ignore_index):
        builder = logits.builder()
        dim = SimpleCrossEntropyLoss.SOFTMAX_DIM
        dtype = logits.shape().dtype
        sizes = logits.shape().sizes
        minus_inf = xla_builder.Op.scalar(builder, value=-math.inf, dtype=dtype)
        scalar_shape = xla_builder.Shape.create(dtype, [])
        max_func = xla_builder.create_computation(
            name="SimpleCrossEntropyLossForwardMax",
            fn=xla_builder.Op.max,
            shapes=[scalar_shape, scalar_shape],
        )
        logits_max = logits.reduce(minus_inf, max_func, dimensions=[dim])
        broadcast_dims = list(range(dim))
        logits_max = logits_max.broadcast_in_dim(sizes, broadcast_dims)
        shifted_logits = logits - logits_max
        exp_shifted = shifted_logits.exp()
        zero = xla_builder.Op.zero(builder, dtype=dtype)
        add_func = xla_builder.create_computation(
            name="SimpleCrossEntropyLossForwardAdd",
            fn=xla_builder.Op.__add__,
            shapes=[scalar_shape, scalar_shape],
        )
        exp_shifted_reduce = exp_shifted.reduce(zero, add_func, dimensions=[dim])
        exp_shifted_reduce_log = exp_shifted_reduce.log()
        exp_shifted_reduce_log = exp_shifted_reduce_log.broadcast_in_dim(
            sizes, broadcast_dims
        )
        log_softmax = shifted_logits - exp_shifted_reduce_log
        target_sizes = target.shape().sizes
        ignore_index = xla_builder.Op.scalar(
            builder, ignore_index, target.shape().dtype
        )
        is_valid_index = target != ignore_index
        is_valid_index = is_valid_index.cast(dtype)
        if SimpleCrossEntropyLoss.should_use_dense_scatter(sizes, target_sizes):
            one = xla_builder.Op.one(builder, dtype=dtype)
            mask = SimpleCrossEntropyLoss.iota_scatter(log_softmax, target, one)
            gather = (log_softmax * mask).reduce(zero, add_func, [dim])
        else:
            target = target.reshape([*target_sizes, 1])
            iota = xla_builder.Op.iota(builder, target.shape(), iota_dimension=0)
            start_indices = xla_builder.Op.concat_in_dim([iota, target], 1, builder)
            operands = log_softmax.op, start_indices.op
            gather_kwargs = dict(
                offset_dims=[],
                collapsed_slice_dims=[0, 1],
                start_index_map=[0, 1],
                index_vector_dim=1,
                slice_sizes=[1, 1],
                indices_are_sorted=None,
            )
            gather = xla_builder.mkop("Gather", operands, **gather_kwargs)
        mul = gather * is_valid_index
        mul_sum = mul.reduce_all(zero, add_func)
        valid_index_count = is_valid_index.reduce_all(zero, add_func)
        neg_loss = mul_sum / valid_index_count
        loss = -neg_loss
        return xla_builder.Op.tuple([loss, log_softmax, is_valid_index], builder)

    @xla_call
    def backward_impl(grad_output, log_softmax, target, is_valid_index):
        dim = SimpleCrossEntropyLoss.SOFTMAX_DIM
        builder = log_softmax.builder()
        sizes = log_softmax.shape().sizes
        dtype = log_softmax.shape().dtype
        zero = xla_builder.Op.zero(builder, dtype=dtype)
        scalar_shape = xla_builder.Shape.create(dtype, [])
        add_func = xla_builder.create_computation(
            name="SimpleCrossEntropyLossBackwardAdd",
            fn=xla_builder.Op.__add__,
            shapes=[scalar_shape, scalar_shape],
        )
        valid_index_count = is_valid_index.reduce_all(zero, add_func)
        neg_grad_element = grad_output / valid_index_count
        grad_element = -neg_grad_element
        neg_grad_vector = is_valid_index * neg_grad_element
        broadcast_dims = list(range(dim))
        neg_grad_vector = neg_grad_vector.broadcast_in_dim(sizes, broadcast_dims)
        mul = log_softmax.exp() * neg_grad_vector
        target_sizes = target.shape().sizes
        if SimpleCrossEntropyLoss.should_use_dense_scatter(sizes, target_sizes):
            scatter_grad = SimpleCrossEntropyLoss.iota_scatter(
                log_softmax, target, grad_element
            )
            grad_input = mul + scatter_grad
        else:
            target = target.reshape([*target_sizes, 1])
            grad_element = grad_element.broadcast([*target_sizes])
            iota = xla_builder.Op.iota(builder, target.shape(), iota_dimension=0)
            scatter_indices = xla_builder.Op.concat_in_dim([iota, target], 1, builder)
            scatter_kwargs = dict(
                computation=add_func,
                update_window_dims=[],
                inserted_window_dims=[0, 1],
                scatter_dims_to_operand_dims=[0, 1],
                index_vector_dim=1,
            )
            grad_input = mul.scatter(scatter_indices, grad_element, **scatter_kwargs)
        return grad_input

    def iota_scatter(logits, labels, on_value):
        logits_builder = logits.builder()
        logits_shape = logits.shape()
        logits_dtype = logits_shape.dtype
        zero = xla_builder.Op.zero(logits_builder, dtype=logits_dtype)
        one_hot_labels = SimpleCrossEntropyLoss.labels_to_one_hot(
            logits_builder,
            depth=logits_shape.sizes[SimpleCrossEntropyLoss.SOFTMAX_DIM],
            axis=SimpleCrossEntropyLoss.SOFTMAX_DIM,
            indices=labels,
            on_value=on_value,
            off_value=zero,
        )
        return one_hot_labels

    @classmethod
    def labels_to_one_hot(cls, builder, depth, axis, indices, on_value, off_value):
        indices_shape = indices.shape()
        output_dimensions = list(indices_shape.sizes)
        output_dimensions.insert(axis, depth)
        iota = cls.one_hot_iota(
            builder, depth=depth, axis=axis, indices_shape=indices_shape
        )
        new_indices_shape = list(indices_shape.sizes)
        new_indices_shape.insert(axis, 1)
        indices = indices.reshape(new_indices_shape)
        broadcast_dims = list(range(len(output_dimensions)))
        indices = indices.broadcast_in_dim(output_dimensions, broadcast_dims)
        iota = iota.broadcast_in_dim(output_dimensions, broadcast_dims)
        one_hot_bool = indices == iota
        on_value = on_value.broadcast(output_dimensions)
        off_value = off_value.broadcast(output_dimensions)
        return one_hot_bool.select(on_value, off_value)

    @classmethod
    def one_hot_iota(cls, builder, depth, axis, indices_shape):
        indices_dims = indices_shape.rank
        linspace_dims = [1 for _ in range(indices_dims + 1)]
        linspace_dims[axis] = depth
        linspace_xla_shape = xla_builder.Shape.create(
            indices_shape.dtype, linspace_dims
        )
        iota = xla_builder.Op.iota(builder, linspace_xla_shape, axis)
        return iota

    @classmethod
    def should_use_dense_scatter(cls, input_sizes, index_sizes):
        input_elements = numel_sizes(input_sizes)
        index_elements = numel_sizes(index_sizes)
        return index_elements * cls.DENSE_SCATTER_FACTOR >= input_elements

    @staticmethod
    def forward(ctx, logits, target, ignore_index):
        loss, log_softmax, is_valid_index = SimpleCrossEntropyLoss.forward_impl(
            logits, target, ignore_index=ignore_index
        )
        ctx.save_for_backward(log_softmax, target, is_valid_index)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = SimpleCrossEntropyLoss.backward_impl(
            grad_output, *ctx.saved_tensors
        )
        return grad_input, None, None


def numel_sizes(sizes):
    return functools.reduce(operator.mul, sizes, 1)


class NoSeedTransferDropout(torch.autograd.Function):
    """
    Forward reference:
        torch_xla/csrc/xla_lower_util.cpp, functions "BuildDropout"
    Backward reference:
        (pytorch) Aten/native/cuda/Dropout.cu, functions "masked_scale_kernel"
    """

    SEED = None

    @lazy_override("torch.nn.Dropout")
    def gen_override():
        class Wrapper(torch.nn.Dropout):
            def forward(self, input):
                if (
                    xla_model.is_xla_tensor(input)
                    and self.training
                    and not self.inplace
                ):
                    return NoSeedTransferDropout.apply(input, self.p)
                return super().forward(input)

        return Wrapper

    @xla_hlo_call
    def forward_impl(input, seed, drop):
        scribe = input.scribe
        dtype = input.dtype
        sizes = input.sizes
        s64 = scribe.s64
        u64 = scribe.u64
        u32 = scribe.u32
        constant_5 = s64.Constant(constant_value=2531011)
        constant_3 = s64.Constant(constant_value=214013)
        multiply_4 = s64.Multiply(constant_3, seed)
        add_6 = s64.Add(constant_5, multiply_4)
        convert_13 = u64.Convert(add_6)
        reshape_15 = u64[1].Reshape(convert_13)
        constant_14 = u64.Constant(constant_value=0)
        reshape_16 = u64[1].Reshape(constant_14)
        concatenate_17 = u64[2].Concatenate(reshape_15, reshape_16, dimensions=[0])
        rng_bit_generator_18 = scribe.tuple(u64[2], u32[sizes]).RngBitGenerator(
            concatenate_17
        )
        get_tuple_element_19 = u32[sizes].GetTupleElement(
            rng_bit_generator_18, tuple_index=1
        )
        constant_21 = u32.Constant(constant_value=9)
        broadcast_22 = u32[sizes].Broadcast(constant_21, dimensions=[])
        shift_right_logical_23 = u32[sizes].ShiftRightLogical(
            get_tuple_element_19, broadcast_22
        )
        convert_24 = dtype[sizes].Convert(shift_right_logical_23)
        constant_25 = dtype.Constant(constant_value=1.1920929e-07)
        broadcast_26 = dtype[sizes].Broadcast(constant_25, dimensions=[])
        multiply_27 = dtype[sizes].Multiply(convert_24, broadcast_26)
        constant_12 = dtype.Constant(constant_value=1)
        constant_11 = dtype.Constant(constant_value=0)
        subtract_28 = dtype.Subtract(constant_12, constant_11)
        broadcast_29 = dtype[sizes].Broadcast(subtract_28, dimensions=[])
        multiply_30 = dtype[sizes].Multiply(multiply_27, broadcast_29)
        broadcast_31 = dtype[sizes].Broadcast(constant_11, dimensions=[])
        add_32 = dtype[sizes].Add(multiply_30, broadcast_31)
        keep = 1.0 - drop
        keep = dtype.Constant(constant_value=keep)
        reshape_7 = dtype[1, 1].Reshape(keep)
        broadcast_8 = dtype[1, 1].Broadcast(reshape_7, dimensions=[0, 1])
        reshape_9 = dtype.Reshape(broadcast_8)
        broadcast_10 = dtype[sizes].Broadcast(reshape_9, dimensions=[])
        compare_33 = scribe.pred[sizes].Compare(
            add_32, broadcast_10, comparison_direction="LT"
        )
        convert_34 = dtype[sizes].Convert(compare_33)
        broadcast_35 = dtype[sizes].Broadcast(keep, dimensions=[])
        divide_36 = dtype[sizes].Divide(convert_34, broadcast_35)
        multiply_43 = dtype[sizes].Multiply(input, divide_36)
        rng_state = u64[2].GetTupleElement(rng_bit_generator_18, tuple_index=0)
        seed = u64[1].Slice(
            rng_state, slice_dimensions=[dict(start=0, limit=1, stride=1)]
        )
        seed = u64.Reshape(seed)
        seed = s64.BitcastConvert(seed)
        return scribe.tuple(dtype[sizes], dtype[sizes], s64).Tuple(
            multiply_43, divide_36, seed
        )

    @staticmethod
    def forward(ctx, input, drop):
        cls = NoSeedTransferDropout
        if not (
            isinstance(cls.SEED, torch.Tensor) and xla_model.is_xla_tensor(cls.SEED)
        ):
            seed = ctypes.c_long(torch.initial_seed()).value
            cls.SEED = torch.tensor(seed, dtype=torch.long, device=input.device)
        output, mask, cls.SEED = cls.forward_impl(input, cls.SEED, drop=drop)
        ctx.save_for_backward(mask)
        return output

    @staticmethod
    def backward(ctx, grad):
        (mask,) = ctx.saved_tensors
        return grad * mask, None


class NoSeedTransferDropout1(torch.autograd.Function):
    """
    Forward reference:
        torch_xla/csrc/xla_lower_util.cpp, functions "BuildDropout"
    Backward reference:
        (pytorch) Aten/native/cuda/Dropout.cu, functions "masked_scale_kernel"
    """

    SEED = None

    @xla_hlo_call
    def forward_impl(input, seed, drop):
        scribe = input.scribe
        dtype = input.dtype
        sizes = input.sizes
        s64 = scribe.s64
        u64 = scribe.u64
        u32 = scribe.u32
        constant_5 = s64.Constant(constant_value=2531011)
        constant_3 = s64.Constant(constant_value=214013)
        multiply_4 = s64.Multiply(constant_3, seed)
        add_6 = s64.Add(constant_5, multiply_4)
        convert_13 = u64.Convert(add_6)
        reshape_15 = u64[1].Reshape(convert_13)
        constant_14 = u64.Constant(constant_value=0)
        reshape_16 = u64[1].Reshape(constant_14)
        concatenate_17 = u64[2].Concatenate(reshape_15, reshape_16, dimensions=[0])
        rng_bit_generator_18 = scribe.tuple(u64[2], u32[sizes]).RngBitGenerator(
            concatenate_17
        )
        get_tuple_element_19 = u32[sizes].GetTupleElement(
            rng_bit_generator_18, tuple_index=1
        )
        constant_21 = u32.Constant(constant_value=9)
        broadcast_22 = u32[sizes].Broadcast(constant_21, dimensions=[])
        shift_right_logical_23 = u32[sizes].ShiftRightLogical(
            get_tuple_element_19, broadcast_22
        )
        convert_24 = dtype[sizes].Convert(shift_right_logical_23)
        constant_25 = dtype.Constant(constant_value=1.1920929e-07)
        broadcast_26 = dtype[sizes].Broadcast(constant_25, dimensions=[])
        multiply_27 = dtype[sizes].Multiply(convert_24, broadcast_26)
        constant_12 = dtype.Constant(constant_value=1)
        constant_11 = dtype.Constant(constant_value=0)
        subtract_28 = dtype.Subtract(constant_12, constant_11)
        broadcast_29 = dtype[sizes].Broadcast(subtract_28, dimensions=[])
        multiply_30 = dtype[sizes].Multiply(multiply_27, broadcast_29)
        broadcast_31 = dtype[sizes].Broadcast(constant_11, dimensions=[])
        add_32 = dtype[sizes].Add(multiply_30, broadcast_31)
        keep = 1.0 - drop
        keep = dtype.Constant(constant_value=keep)
        reshape_7 = dtype[1, 1].Reshape(keep)
        broadcast_8 = dtype[1, 1].Broadcast(reshape_7, dimensions=[0, 1])
        reshape_9 = dtype.Reshape(broadcast_8)
        broadcast_10 = dtype[sizes].Broadcast(reshape_9, dimensions=[])
        compare_33 = scribe.pred[sizes].Compare(
            add_32, broadcast_10, comparison_direction="LT"
        )
        convert_34 = dtype[sizes].Convert(compare_33)
        broadcast_35 = dtype[sizes].Broadcast(keep, dimensions=[])
        divide_36 = dtype[sizes].Divide(convert_34, broadcast_35)
        multiply_43 = dtype[sizes].Multiply(input, divide_36)
        rng_state = u64[2].GetTupleElement(rng_bit_generator_18, tuple_index=0)
        seed = u64[1].Slice(
            rng_state, slice_dimensions=[dict(start=0, limit=1, stride=1)]
        )
        seed = u64.Reshape(seed)
        seed = s64.BitcastConvert(seed)
        return scribe.tuple(dtype[sizes], dtype[sizes], s64).Tuple(
            multiply_43, divide_36, seed
        )

    @staticmethod
    def forward(ctx, input, drop, mask):
        cls = NoSeedTransferDropout1
        if not (
            isinstance(cls.SEED, torch.Tensor) and xla_model.is_xla_tensor(cls.SEED)
        ):
            seed = ctypes.c_long(torch.initial_seed()).value
            cls.SEED = torch.tensor(seed, dtype=torch.long, device=input.device)

        if mask is not None:
            output = torch.mul(input, mask)
        else:
            output, mask, cls.SEED = cls.forward_impl(input, cls.SEED, drop=drop)

        ctx.save_for_backward(mask)
        return output, mask

    @staticmethod
    def backward(ctx, grad, mask):
        (mask,) = ctx.saved_tensors
        return grad * mask, None, None


class NoSeedTransferModuleDropout(torch.nn.Module):
    """
    Forward reference:
        torch_xla/csrc/xla_lower_util.cpp, functions "BuildDropout"
    Backward reference:
        (pytorch) Aten/native/cuda/Dropout.cu, functions "masked_scale_kernel"
    """

    @lazy_override("torch.nn.Dropout")
    def gen_override():
        if os.environ.get("NEURON_ENABLE_NOSEED_DROPOUT", None) == "1":
            return NoSeedTransferModuleDropout
        return torch.nn.Dropout

    @staticmethod
    def _clear_mask_on_backward(m, grad_input, grad_output):
        m.mask = None

    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.p = p
        self.inplace = inplace
        self.mask = None
        self.register_full_backward_hook(self._clear_mask_on_backward)

    def forward(self, input):
        if not self.training:
            return input
        if not xla_model.is_xla_tensor(input):
            return torch.nn.functional.dropout(
                input, self.p, self.training, self.inplace
            )
        assert (
            not self.inplace
        ), "Currently we dont support inplace option for Dropout"
        output, self.mask = NoSeedTransferDropout1.apply(input, self.p, self.mask)
        return output


class Int32PermissiveEmbedding(torch.autograd.Function):
    """
    Reference: torch_xla/csrc/tensor_ops.cpp, function "EmbeddingDenseBackward"
    """

    @lazy_override("torch.nn.Embedding")
    def gen_override():
        class Wrapper(torch.nn.Embedding):
            def forward(self, input):
                if (
                    xla_model.is_xla_tensor(input)
                    and self.padding_idx is None
                    and self.max_norm is None
                    and not self.scale_grad_by_freq
                    and not self.sparse
                ):
                    return Int32PermissiveEmbedding.apply(self.weight, input)
                return super().forward(input)

        return Wrapper

    @xla_call
    def grad_weight(weight, grad_output, indices):
        numel = numel_sizes(indices.shape().sizes)
        grad_weight = weight.zeros_like()
        indices_1d = indices.reshape([numel])
        indices_1d_ge0 = indices_1d >= indices_1d.zeros_like()
        indices_1d_wrapped_around = indices_1d + indices_1d.scalar_like(
            weight.shape().sizes[0]
        )
        indices_1d = indices_1d_ge0.select(indices_1d, indices_1d_wrapped_around)
        indices_2d = indices_1d.reshape([numel, 1])
        grad_output = grad_output.cast(grad_weight.shape().dtype)
        grad_source = grad_output.reshape([numel, grad_output.shape().sizes[-1]])
        scalar_shape = xla_builder.Shape.create(grad_weight.shape().dtype, [])
        computation = xla_builder.create_computation(
            name="Int32PermissiveEmbeddingScatterCombiner",
            fn=xla_builder.Op.__add__,
            shapes=[scalar_shape, scalar_shape],
        )
        scatter_kwargs = dict(
            computation=computation,
            update_window_dims=[1],
            inserted_window_dims=[0],
            scatter_dims_to_operand_dims=[0],
            index_vector_dim=1,
        )
        grad_weight = grad_weight.scatter(indices_2d, grad_source, **scatter_kwargs)
        return grad_weight

    @staticmethod
    def forward(ctx, weight, input):
        ctx.save_for_backward(weight, input)
        return torch.embedding(
            weight, input, padding_idx=-1, scale_grad_by_freq=False, sparse=False
        )

    @staticmethod
    def backward(ctx, grad_output):
        weight, indices = ctx.saved_tensors
        grad_weight = Int32PermissiveEmbedding.grad_weight(weight, grad_output, indices)
        return grad_weight, None


class SummingClipGradNorm:
    @lazy_override("torch.nn.utils.clip_grad_norm_")
    def gen_override():
        original_clip_grad_norm = torch.nn.utils.clip_grad_norm_

        def integer_norm_inner(grad, norm_type):
            if norm_type == 1:
                return grad.abs().sum()
            elif norm_type == 2:
                return (grad * grad).sum()
            else:
                return grad.pow(norm_type).sum()

        def clip_grad_norm(
            parameters, max_norm, norm_type=2.0, error_if_nonfinite=False
        ):
            norm_type = float(norm_type)
            if not norm_type.is_integer():
                return original_clip_grad_norm(
                    parameters, max_norm, norm_type, error_if_nonfinite
                )
            norm_type = int(norm_type)
            if isinstance(parameters, torch.Tensor):
                parameters = [parameters]
            parameters = list(filter(lambda p: p.grad is not None, parameters))
            max_norm = float(max_norm)
            if len(parameters) == 0:
                return torch.tensor(0.0)
            device = parameters[0].grad.device
            # norm_type != inf
            total_norm_inner = torch.zeros(
                [], dtype=parameters[0].grad.dtype, device=device
            )
            for param in reversed(parameters):
                total_norm_inner += integer_norm_inner(param.grad.detach(), norm_type)
            if norm_type == 1:
                total_norm = total_norm_inner
            elif norm_type == 2:
                total_norm = total_norm_inner.sqrt()
            else:
                total_norm = total_norm_inner.pow(1.0 / norm_type)
            if error_if_nonfinite and (total_norm.isnan() or total_norm.isinf()):
                raise RuntimeError(
                    f"The norm of order {norm_type} for a gradient from `parameters` "
                    "is non-finite, so it cannot be clipped. This error can be "
                    "disabled with `error_if_nonfinite=False`"
                )
            clip_coef = torch.tensor(max_norm, device=device) / (total_norm + 1e-6)
            clip_value = torch.where(
                clip_coef < 1, clip_coef, torch.tensor(1.0, device=device)
            )
            for param in reversed(parameters):
                param.grad.detach().mul_(clip_value)
            return total_norm

        return clip_grad_norm


class UpSampleNearestNeighbor2d(torch.autograd.Function):
    @staticmethod
    def should_be_lowered(input_size, output_size, scale_factors):
        if scale_factors is None:
            _, _, ih, iw = input_size
            oh, ow = output_size
            scale_factors = [oh / ih, ow / ow]

        return (scale_factors[0] * 1.0).is_integer() and (
            scale_factors[1] * 1.0
        ).is_integer()

    @lazy_override("torch._C._nn.upsample_nearest2d")
    def gen_override():
        original_func = torch._C._nn.upsample_nearest2d

        def upsample_nearest2d(input, out_size, scale_factors):
            if xla_model.is_xla_tensor(
                input
            ) and UpSampleNearestNeighbor2d.should_be_lowered(
                input.shape, out_size, scale_factors
            ):
                return UpSampleNearestNeighbor2d.apply(input, out_size, scale_factors)
            return original_func(input, out_size, scale_factors)

        return upsample_nearest2d

    @xla_hlo_call
    def forward_impl(input, out_size):
        # backend_config is 00 since align_corners and half_pixel_centers is false in this case.
        return input.dtype[out_size].CustomCall(
            input, custom_call_target=AwsNeuronNearestNeighbor2d, backend_config=b'"00"'
        )

    @xla_hlo_call
    def backward_impl(grad, out_size):
        # backend_config is 00 since align_corners and half_pixel_centers is false in this case.
        return grad.dtype[out_size].CustomCall(
            grad,
            custom_call_target=AwsNeuronNearestNeighbor2dBackward,
            backend_config=b'"00"',
        )

    @staticmethod
    def forward(ctx, input, output_size, scale_factors):
        if output_size is None:
            output_size = [
                int(input.shape[2] * scale_factors[0]),
                int(input.shape[3] * scale_factors[1]),
            ]
        output_size = tuple([input.shape[0], *output_size, input.shape[1]])
        input_permuted = input.permute((0, 2, 3, 1))
        ctx.save_for_backward(input_permuted)

        return UpSampleNearestNeighbor2d.forward_impl(
            input_permuted, out_size=output_size
        ).permute((0, 3, 1, 2))

    @staticmethod
    def backward(ctx, grad):
        (input,) = ctx.saved_tensors
        out_size = input.shape
        return (
            UpSampleNearestNeighbor2d.backward_impl(
                grad.permute((0, 2, 3, 1)), out_size=out_size
            ).permute((0, 3, 1, 2)),
            None,
            None,
        )


def gen_add_func(dtype):
    def add_func(scribe):
        p0 = dtype.Parameter(parameter_number=0)
        p1 = dtype.Parameter(parameter_number=1)
        return dtype.Add(p0, p1)

    return add_func


def gen_max_func(dtype):
    def max_func(scribe):
        p0 = dtype.Parameter(parameter_number=0)
        p1 = dtype.Parameter(parameter_number=1)
        return dtype.Maximum(p0, p1)

    return max_func


class Softmax(torch.autograd.Function):
    @lazy_override("torch.nn.Softmax")
    def gen_override():
        if os.environ.get("NEURON_FUSE_SOFTMAX", None) != "1":
            return torch.nn.Softmax

        class Wrapper(torch.nn.Softmax):
            def forward(self, input):
                if xla_model.is_xla_tensor(input) and self.dim is not None:
                    return Softmax.apply(input, self.dim)
                return super().forward(input)

        return Wrapper

    @xla_hlo_call
    def forward_impl(input, dim):
        """Reference implementation:
        non_reduce_dims = [idx for idx in range(len(input.sizes)) if idx != dim]
        reduce_sizes = tuple(input.sizes[idx] for idx in non_reduce_dims)
        dtype = input.dtype
        constant_7 = dtype.Constant(constant_value=float('-inf'))
        reduce_12 = dtype[reduce_sizes].Reduce(input, constant_7, dimensions=[dim],
                                               to_apply=gen_max_func(dtype))
        broadcast_13 = dtype[input.sizes].Broadcast(reduce_12, dimensions=non_reduce_dims)
        subtract_14 = dtype[input.sizes].Subtract(input, broadcast_13)
        exponential_15 = dtype[input.sizes].Exp(subtract_14)
        constant_16 = dtype.Constant(constant_value=0)
        reduce_21 = dtype[reduce_sizes].Reduce(exponential_15, constant_16, dimensions=[dim],
                                               to_apply=gen_add_func(dtype))
        broadcast_22 = dtype[input.sizes].Broadcast(reduce_21, dimensions=non_reduce_dims)
        divide_23 = dtype[input.sizes].Divide(exponential_15, broadcast_22)
        """
        if dim < 0:
            dim = len(input.sizes) + dim
        backend_config = str(dim).encode()
        return input.dtype[input.sizes].CustomCall(
            input,
            custom_call_target=AwsNeuronSoftmax,
            backend_config=backend_config,
        )

    @xla_hlo_call
    def backward_impl(grad_output, output, dim):
        """Reference implementation:
        non_reduce_dims = [idx for idx in range(len(output.sizes)) if idx != dim]
        reduce_sizes = tuple(output.sizes[idx] for idx in non_reduce_dims)
        dtype = output.dtype
        multiply_36 = dtype[output.sizes].Multiply(grad_output, output)
        constant_37 = dtype.Constant(constant_value=0)
        reduce_42 = dtype[reduce_sizes].Reduce(multiply_36, constant_37, dimensions=[dim],
                                               to_apply=gen_add_func(dtype))
        broadcast_43 = dtype[output.sizes].Broadcast(reduce_42, dimensions=non_reduce_dims)
        subtract_44 = dtype[output.sizes].Subtract(grad_output, broadcast_43)
        multiply_45 = dtype[output.sizes].Multiply(output, subtract_44)
        """
        if dim < 0:
            dim = len(output.sizes) + dim
        backend_config = str(dim).encode()
        return grad_output.dtype[grad_output.sizes].CustomCall(
            grad_output,
            output,
            custom_call_target=AwsNeuronSoftmaxBackward,
            backend_config=backend_config,
        )

    @staticmethod
    def forward(ctx, input, dim):
        ctx._dim = dim
        output = Softmax.forward_impl(input, dim=dim)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (output,) = ctx.saved_tensors
        dim = ctx._dim
        return Softmax.backward_impl(grad_output, output, dim=dim), None


class FunctionalSoftmax:
    @register_hook_function
    def hook_function():
        if os.environ.get("NEURON_FUSE_SOFTMAX", None) != "1":
            return
        orig_softmax = torch.nn.functional.softmax

        @wraps(orig_softmax)
        def softmax(input, dim=None, _stacklevel=3, dtype=None):
            if not xla_model.is_xla_tensor(input):
                return orig_softmax(
                    input, dim=dim, _stacklevel=_stacklevel, dtype=dtype
                )
            if dim is None:
                dim = -1
            if dtype is not None:
                input = input.to(dtype=dtype)
            return Softmax.apply(input, dim)

        torch.nn.functional.softmax = softmax


class RmsNorm(torch.autograd.Function):
    @xla_hlo_call
    def forward_impl(input, weight, eps, dim):
        eps = input.scribe.f32.Constant(constant_value=eps)
        dim = str(dim).encode()
        return input.dtype[input.sizes].CustomCall(
            input,
            weight,
            eps,
            custom_call_target=AwsNeuronRmsNorm,
            backend_config=dim
        )

    @xla_hlo_call
    def backward_impl(grad_output, output, weight, eps, dim):
        scribe = grad_output.scribe
        eps = scribe.f32.Constant(constant_value=eps)
        dim = str(dim).encode()
        return scribe.tuple(grad_output.dtype[grad_output.sizes],
                            weight.dtype[weight.sizes]).CustomCall(
            grad_output, output, weight, eps,
            custom_call_target=AwsNeuronRmsNormBackward,
            backend_config=dim
        )

    @staticmethod
    def forward(ctx, hidden_states, weight, eps, dim):
        output = RmsNorm.forward_impl(hidden_states, weight, eps=eps, dim=dim)
        ctx.eps = eps
        ctx.dim = dim
        ctx.save_for_backward(output, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, weight = ctx.saved_tensors
        eps = ctx.eps
        dim = ctx.dim
        grad_input, grad_weight = RmsNorm.backward_impl(grad_output, output,
                                                        weight, eps=eps, dim=dim)
        return grad_input, grad_weight, None


class Argmax(torch.autograd.Function):
    @lazy_override("torch.argmax")
    @lazy_override("torch._tensor.Tensor.argmax")
    def gen_override():
        original_func = torch.argmax

        def argmax(input, dim=None, keepdim=False):
            if xla_model.is_xla_tensor(input):
                return Argmax.apply(input, dim, keepdim)
            return original_func(input, dim, keepdim)

        return argmax

    @xla_hlo_call
    def impl(input, dim, output_shape):
        backend_config = str(dim).encode()
        scribe = input.scribe
        s64 = scribe.s64
        return s64.dtype[output_shape].CustomCall(
            input,
            custom_call_target=AwsNeuronArgMax,
            backend_config=backend_config,
        )

    @classmethod
    def apply(cls, input, dim, keepdim):
        # Handle reduction along all dimensions
        if dim is None:
            result = cls.impl(torch.reshape(input, (-1,)), dim=0, output_shape=tuple())
            if keepdim:
                result = torch.reshape(result, [1] * len(input.shape))
            return result

        if dim < 0:
            dim = len(input.shape) + dim
        output_shape = list(input.shape)
        del output_shape[dim]
        result = cls.impl(input, dim=dim, output_shape=tuple(output_shape))
        if keepdim:
            result = result.unsqueeze(dim)
        return result


def set_unload_prior_neuron_models_mode(value):
    @xla_hlo_call
    def func_set_unload_all(dummy_tensor):
        return dummy_tensor.dtype[(1)].CustomCall(
            custom_call_target=f"{TorchNeuronUnloadPriorModels}{value}"
        )

    dummy_tensor = torch.tensor(0, device=xla_model.xla_device())
    output = func_set_unload_all(dummy_tensor)
    output = output.to("cpu")


class Argmin(torch.autograd.Function):
    @lazy_override("torch.argmin")
    @lazy_override("torch._tensor.Tensor.argmin")
    def gen_override():
        original_func = torch.argmin

        def argmin(input, dim=None, keepdim=False):
            if xla_model.is_xla_tensor(input):
                return Argmin.apply(input, dim, keepdim)
            return original_func(input, dim, keepdim)

        return argmin

    @xla_hlo_call
    def impl(input, dim, output_shape):
        backend_config = str(dim).encode()
        scribe = input.scribe
        s64 = scribe.s64
        return s64.dtype[output_shape].CustomCall(
            input,
            custom_call_target=AwsNeuronArgMin,
            backend_config=backend_config,
        )

    @classmethod
    def apply(cls, input, dim, keepdim):
        # Handle reduction along all dimensions
        if dim is None:
            result = cls.impl(torch.reshape(input, (-1,)), dim=0, output_shape=tuple())
            if keepdim:
                result = torch.reshape(result, [1] * len(input.shape))
            return result

        if dim < 0:
            dim = len(input.shape) + dim
        output_shape = list(input.shape)
        del output_shape[dim]
        result = cls.impl(input, dim=dim, output_shape=tuple(output_shape))
        if keepdim:
            result = result.unsqueeze(dim)
        return result


class TopK:
    @lazy_override("torch.topk")
    def gen_override():
        original_func = torch.topk

        def topk(input, k, dim=None, largest=True, sorted=True, out=None):
            if (
                (out is not None or not xla_model.is_xla_tensor(input))
                or not largest
                or not sorted
                or (dim is not None and dim != -1 and dim != (len(input.shape) - 1))
            ):
                return original_func(
                    input,
                    k,
                    dim=dim if dim is not None else -1,
                    largest=largest,
                    sorted=sorted,
                    out=out,
                )
            return TopK.apply(input, k, dim)

        return topk

    @xla_hlo_call
    def impl(input, k, output_shape):
        k = str(k).encode()
        scribe = input.scribe
        u32 = scribe.u32
        dtype = input.dtype
        return scribe.tuple(dtype[output_shape], u32.dtype[output_shape]).CustomCall(
            input, custom_call_target=AwsNeuronTopK, backend_config=k
        )

    @classmethod
    def apply(cls, input, k, dim):
        output_shape = list(input.shape)
        output_shape[-1] = k
        return cls.impl(input, k=k, output_shape=tuple(output_shape))
