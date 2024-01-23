# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================
import os
import threading
from contextlib import contextmanager
from functools import wraps
import torch
import torch_xla
from torch_xla.core import xla_model
from torch_neuronx.xla_impl.base import xla_hlo_call, register_hook_function
from torch_neuronx.xla_impl.custom_call_targets import (
    AwsNeuronTransferWithStaticRing,
    TorchNeuronStartBackward,
)


def GetTransferWithStaticRingDefaultOps():
    DEFAULT = "Embedding,LayerNorm,Linear,Conv2d,BatchNorm2d"
    to_be_decorated = os.environ.get("NEURON_TRANSFER_WITH_STATIC_RING_OPS", DEFAULT)
    return to_be_decorated.split(",")


class TransferWithStaticRing(torch.autograd.Function):
    @register_hook_function
    def hook_function():
        """
        Decorate PyTorch modules so that they'll insert AwsNeuronTransferWithStaticRing.
        """
        if (
            os.environ.get("NEURON_TRANSFER_ALL_PARAMETERS_WITH_STATIC_RING", None)
            == "1"
            or os.environ.get(
                "NEURON_INTERNAL_TRANSFER_ALL_PARAMETERS_WITH_STATIC_RING", None
            )
            == "1"
        ):
            inserter = TransferWithStaticRingInserter()

            def wrap_getattr(getattr_fn):
                @wraps(getattr_fn)
                def wrapper(self, name: str):
                    return inserter.maybe_insert_transfer(getattr_fn(self, name))

                return wrapper

            def wrap_call(call_fn):
                @wraps(call_fn)
                def wrapper(self, *args, **kwargs):
                    with inserter.forward_scope():
                        return call_fn(self, *args, **kwargs)

                return wrapper

            torch.nn.Module.__getattr__ = wrap_getattr(torch.nn.Module.__getattr__)
            torch.nn.Module.__call__ = wrap_call(torch.nn.Module.__call__)
            return
        to_be_decorated_module_names = GetTransferWithStaticRingDefaultOps()

        for name in dir(torch.nn):
            module = getattr(torch.nn, name)
            if name in to_be_decorated_module_names and issubclass(
                module, torch.nn.Module
            ):
                module = TransferWithStaticRing.add_inserter(module)
                setattr(torch.nn, name, module)

    @xla_hlo_call
    def transfer(shape):
        out_shape = shape.dtype[shape.sizes]
        return out_shape.CustomCall(
            shape, custom_call_target=AwsNeuronTransferWithStaticRing
        )

    @staticmethod
    def forward(ctx, tensor):
        return TransferWithStaticRing.transfer(tensor)

    @staticmethod
    def backward(ctx, grad):
        return TransferWithStaticRing.transfer(grad)

    @staticmethod
    def add_inserter(module_cls):
        """
        Decorate a subclass of torch.nn.Module so that a consumption of nn.Parameter
        within its forward function inserts an AwsNeuronTransferWithStaticRing
        """
        inserter = TransferWithStaticRingInserter()

        def wrap_getattr(getattr_fn):
            @wraps(getattr_fn)
            def wrapper(self, name: str):
                return inserter.maybe_insert_transfer(getattr_fn(self, name))

            return wrapper

        module_cls.__getattr__ = wrap_getattr(module_cls.__getattr__)

        @wraps(module_cls, updated=())
        class ModuleClass(module_cls):
            @wraps(module_cls.forward)
            def forward(self, *args, **kwargs):
                with inserter.forward_scope():
                    return module_cls.forward(self, *args, **kwargs)

        return ModuleClass


class TransferWithStaticRingInserter:
    def __init__(self):
        self._rlock = threading.RLock()
        self._under_forward = False

    @contextmanager
    def forward_scope(self):
        with self._rlock:
            current_under_forward = self._under_forward
            self._under_forward = True
            try:
                yield
            finally:
                self._under_forward = current_under_forward

    def maybe_insert_transfer(self, attr):
        with self._rlock:
            if self._under_forward:
                if isinstance(attr, torch.nn.Parameter) and xla_model.is_xla_tensor(
                    attr
                ):
                    attr = TransferWithStaticRing.apply(attr)
                # Note: torch_xla < 1.12 will not have the FSDP module
                elif hasattr(torch_xla.distributed, "fsdp"):
                    if isinstance(
                        attr, torch_xla.distributed.fsdp.XlaFullyShardedDataParallel
                    ):
                        for sharded_p in attr.sharded_params:
                            sharded_p.data = TransferWithStaticRing.apply(
                                sharded_p.data
                            )
        return attr


class StartBackward(torch.autograd.Function):
    @register_hook_function
    def hook_function():
        if os.environ.get("NEURON_INTERNAL_MARK_START_BACKWARD", None) != "1":
            return

        def wrap_forward(forward_fn):
            @wraps(forward_fn)
            def wrapper(self, *args, **kwargs):
                return StartBackward.apply(forward_fn(self, *args, **kwargs))

            return wrapper

        to_be_decorated_module_names = {
            "Embedding",
            "LayerNorm",
            "Linear",
            "Conv2d",
            "BatchNorm2d",
        }
        for name in dir(torch.nn):
            module = getattr(torch.nn, name)
            if name in to_be_decorated_module_names and issubclass(
                module, torch.nn.Module
            ):
                module.forward = wrap_forward(module.forward)

    @xla_hlo_call
    def mark(shape):
        return shape.dtype[shape.sizes].CustomCall(
            shape, custom_call_target=TorchNeuronStartBackward
        )

    @staticmethod
    def forward(ctx, tensor):
        return StartBackward.mark(tensor)

    @staticmethod
    def backward(ctx, grad):
        return grad
