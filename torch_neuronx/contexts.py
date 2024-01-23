# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
"""
Contexts
========
A module which contains testing context managers
"""

import os
from contextlib import contextmanager
from typing import Dict
import tempfile
from unittest.mock import patch

import torch


@contextmanager
def environment(env: Dict[str, str]):
    """
    A context which temporarily sets environment variables.

    Arguments:
        env: A mapping from environment variable name to the value it should
             be set to.
    """
    env_backup_dict = {key: os.environ.get(key, None) for key in env}
    os.environ.update(**env)
    try:
        yield
    finally:
        for key, value in env_backup_dict.items():
            if value is None:
                os.environ.pop(key)
            else:
                os.environ.update(key=value)


@contextmanager
def mock_neuron_cores():
    """
    A context allows libtpu to be used without a real inferentia device.
    """
    env = {
        'NEURON_INTERNAL_MOCK_TPU': '1',
        'TPU_NUM_DEVICES': '1',
    }
    with environment(env):
        yield


@contextmanager
def lowering():
    """
    A context to be used during HLO lowering to configure `torch-xla`
    """
    from torch_xla import _XLAC

    _XLAC._xla_set_trace_mode(True)
    yield
    _XLAC._xla_set_trace_mode(False)


@contextmanager
def mock_neuronx_cc():
    """
    A context skip compilation calls in the torch_neuronx.trace API.

    This results in an empty NEFF file which must be correctly handled by the
    downstream code.
    """
    with tempfile.NamedTemporaryFile() as tmp:

        def hlo_compile(*args, **kwargs):
            return tmp.name

        with patch('torch_neuronx.xla_impl.trace.hlo_compile', hlo_compile):
            yield


@contextmanager
def unit():
    """
    A context for unit testing which skips neuron-cc and mocks out the device
    """
    with mock_neuronx_cc(), mock_neuron_cores():
        yield


class _StopNrtLoadContext:

    count = 0

    def __init__(self) -> None:
        self.runtime = torch.classes.neuron.Runtime()

    def __enter__(self) -> None:
        self.count += 1
        self.runtime.disable()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.count -= 1
        if self.count == 0:
            self.runtime.enable()


@contextmanager
def disable_nrt_load():
    """
    A context which stops libnrt nrt_loads calls in libtorchneuron.
    """
    with _StopNrtLoadContext():
        yield
