# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================
from typing import Any

import torch


def _move_script(module: torch.jit.ScriptModule, device: torch.device):
    """
    Moves ScriptModule parameters to the given device in-place.

    This function is useful for moving `torch.jit.trace`d modules to XLA
    devices.

    When a function has already been traced, its parameters are
    converted to pure `torch.Tensor` objects rather than `torch.nn.Parameter`
    objects. There is an assertion in `torch.nn.Module.to` which checks that
    the parameters are instances of `torch.nn.Parameter`. This means that
    `.to` cannot be used when moving to the XLA device. The other standard
    device loading mechanism is the `map_location` argument in
    `torch.jit.load`. This raises an error when passing an XLA device.

    Args:
        module: A torchscripted module.
        device: The device to move the module to.
    """
    for name, param in module.named_parameters(recurse=False):
        setattr(module, name, param.to(device))

    for name, buffer in module.named_buffers(recurse=False):
        setattr(module, name, buffer.to(device))

    for child in module.children():
        move(child, device)


def move(func: Any, device: torch.device):
    """
    Move parameter state to the specified device.

    This does not do anything if the object is not a known type that has
    parameter state.

    Args:
        module: An object which may have parameters.
        device: The device to move the object to.
    """
    if isinstance(func, torch.jit.ScriptModule):
        _move_script(func, device)
    elif isinstance(func, torch.nn.Module):
        func.to(device)
    elif isinstance(func, torch.optim.Optimizer):
        for param_group in func.param_groups:
            param_group['params'] = [par.to(device) for par in param_group['params']]
