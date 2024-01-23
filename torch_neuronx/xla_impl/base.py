# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================
import re
import sys
import threading
from functools import wraps
from itertools import chain
import torch
import torch_xla
from torch_xla.core import xla_builder, xla_model, xla_op_registry
from torch_neuronx.pyhlo.scribe import HloScribe
from torch_neuronx.pyhlo.constant.serialize_torch import serialize_torch


def xla_call(opfn, name=None):
    """
    Register ``opfn`` as XLA Op with possibly auto-generated name.
    """
    if name is None:
        name = _get_op_name(opfn)
    return xla_op_registry.register(name, opfn)


def hlo_call(opfn):
    """
    Register ``opfn``, a Python function written in PyHLO syntax, as ``xla_builder.Op.call``.
    It can be called from XLA Op context.
    """
    return AwsNeuronCustomLoweringType.hlo_register(_get_op_name(opfn), opfn)


def xla_hlo_call(opfn):
    """
    Register ``opfn``, a Python function written in PyHLO syntax, as XLA Op.
    It can be used as any other XLA Op and can be called on PyTorch tensors directly.
    """
    return xla_call(hlo_call(opfn), name=_get_op_name(opfn))


def override(target):
    """
    Override the content of ``target`` with the decorated subclass of ``torch.autograd.Function``.
    For example, if ``target == 'torch._C._nn.gelu'``, and a class is defined as the following
    ```
    @override('torch._C._nn.gelu')
    class Gelu(torch.autograd.Function):
        ...
    ```
    then ``torch._C._nn.gelu`` will be replaced with ``Gelu.apply``.
    """

    def register_override(cls):
        def func():
            return target, AwsNeuronCustomLoweringType.wrap_function(eval(target), cls)

        AwsNeuronCustomLoweringType.OVERRIDINGS.append((cls.__name__, func))
        return cls

    return register_override


def lazy_override(target):
    """
    Override the content of ``target`` with the return value of the decorated function.
    The decorated function cannot accept arguments.
    """

    def register_override(method):
        def func():
            return target, method()

        AwsNeuronCustomLoweringType.OVERRIDINGS.append((_get_cls_name(method), func))
        return method

    return register_override


def register_hook_function(func):
    AwsNeuronCustomLoweringType.REGISTERED_HOOK_FUNCTIONS.append(
        (_get_cls_name(func), func)
    )


def _get_cls_name(func):
    mod_cls_name, *_ = func.__qualname__.split(f".{func.__name__}")
    *_, cls_name = mod_cls_name.split(".")
    return cls_name


def _get_op_name(func):
    cls_name = _get_cls_name(func)
    fn_name = _snake_to_camel(func.__name__)
    return f"{cls_name}{fn_name}"


class AwsNeuronCustomLoweringType(type(torch.autograd.Function)):
    OVERRIDINGS = []
    REGISTERED_HOOK_FUNCTIONS = []
    enabled = False
    DISABLED_BY_DEFAULT = [
        "NoSeedTransferDropout",
        "RandN",
    ]

    def __new__(mcs, name, bases, dct):
        xla_call_prefix = "xla_call_"
        hlo_call_prefix = "hlo_call_"
        xla_hlo_call_prefix = "xla_hlo_call_"
        for key, value in dct.items():
            if not callable(value):
                continue
            if not key.startswith(
                (xla_call_prefix, hlo_call_prefix, xla_hlo_call_prefix)
            ):
                continue
            xor_key = f"Call{name}"
            if key.startswith((hlo_call_prefix, xla_hlo_call_prefix)):
                dct[key] = mcs.hlo_register(xor_key, dct[key])
            if key.startswith(hlo_call_prefix):
                continue
            dct[key] = xla_op_registry.register(xor_key, dct[key])
        cls = super().__new__(mcs, name, bases, dct)
        if "hook_function" in dct:
            mcs.REGISTERED_HOOK_FUNCTIONS.append((name, cls.hook_function))
        if "overrides" in dct:
            mcs.OVERRIDINGS.append((name, cls.overrides))
        return cls

    @classmethod
    def enable(cls, pattern=None):
        if cls.enabled:
            return
        if pattern is None:
            pattern = r""
            if cls.DISABLED_BY_DEFAULT:
                pattern = r"^((?!({})).)*$".format("|".join(cls.DISABLED_BY_DEFAULT))
        cls.enabled = True
        cls.apply_overridings(pattern)
        for name, hook_fn in cls.REGISTERED_HOOK_FUNCTIONS:
            if not _re_match_any_case(pattern, name):
                continue
            hook_fn()

    @classmethod
    def apply_overridings(cls, pattern):
        for name, overrides in cls.OVERRIDINGS:
            if not _re_match_any_case(pattern, name):
                continue
            overridding_target, func_or_class = overrides()
            decorator_kwargs = (
                dict(updated=()) if isinstance(func_or_class, type) else {}
            )
            decorator = wraps(eval(overridding_target), **decorator_kwargs)
            func_or_class = decorator(func_or_class)
            *modules, fname = overridding_target.split(".")
            modules = ".".join(modules)
            module = eval(modules)
            setattr(module, fname, func_or_class)

    @staticmethod
    def wrap_function(func, custom_call_cls):
        def wrapper(*args, **kwargs):
            if any(
                is_xla_tensor(it) or is_xla_device(it)
                for it in chain(args, kwargs.values())
            ):
                return custom_call_cls.apply(*args, **kwargs)
            return func(*args, **kwargs)

        return wrapper

    @classmethod
    def hlo_register(cls, cpt_name, func):
        def call_func(*ops, **kwargs):
            def wrapper(scribe):
                params = []
                for idx, op in enumerate(ops):
                    dtype = getattr(scribe, op.shape().dtype)
                    sizes = tuple(op.shape().sizes)
                    params.append(dtype[sizes].Parameter(parameter_number=idx))
                return func(*params, **kwargs)

            if not ops:
                raise ValueError(f"Computation {cpt_name} must have >= 1 operand(s).")
            wrapper.__name__ = f"Hlo{cpt_name}"  # to let the HLO look better
            scribe = HloScribe(serialize_torch)
            serialized_cpt = scribe(wrapper).module_proto.SerializeToString()
            computation = xla_builder.computation_from_module_proto(
                cpt_name, serialized_cpt
            )
            return xla_builder.Op.call(computation, ops, ops[0].builder())

        return call_func


def is_xla_tensor(obj):
    return isinstance(obj, torch.Tensor) and xla_model.is_xla_tensor(obj)


def is_xla_device(obj):
    return isinstance(obj, torch.device) and obj.type == "xla"


def _snake_to_camel(snake):
    return "".join(s.title() for s in snake.split("_"))


def _re_match_any_case(pattern, name):
    return re.search(pattern.lower(), name.lower()) is not None


class AwsNeuronCustomLowering(
    torch.autograd.Function, metaclass=AwsNeuronCustomLoweringType
):
    def __new__(cls, *args, **kwargs):
        raise TypeError(
            "{} is meant as a holder class of forward/backward functions "
            "and cannot be instantiated".format(cls)
        )
