# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================
"""
Bindings
========
Utilities to retrieve & manipulate of `torch.classes.neuron.Model` bindings.
"""
import typing

import torch


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def get_model_bindings(trace: torch.nn.Module) -> typing.List[torch.ScriptObject]:
    """
    Retrieves all Neuron subgraphs from a traced torch graph.

    Arguments:
        trace: A torch module which contains one or more Neuron subgraphs.

    Returns:
        The collection of stateful Neuron subgraph objects.
    """
    bindings = []

    for module in trace.modules():
        if not isinstance(module, torch.jit.ScriptModule):
            continue
        if not module._c._has_method("forward"):
            continue
        for node in module.graph.nodes():
            # Only look at used attributes
            if not node.kind() == "prim::GetAttr":
                continue

            name = node.s("name")
            if not hasattr(module, name):
                continue
            attribute = getattr(module, name)
            if hasattr(attribute, "set_neuron_devices"):
                bindings.append(attribute)

    return bindings


# -----------------------------------------------------------------------------
# User Interfaces
# -----------------------------------------------------------------------------

def dynamic_batch(trace: torch.nn.Module, enable_dynamic_batch: bool = True) -> torch.nn.Module:
    """
    Enables (or disables) dynamic batching on a traced Neuron model.

    Arguments:
        trace: A trace from `torch_neuronx.trace` to enable dynamic batching on.
        enable_dynamic_batch: Whether to enabled (or disabled) dynamic batching.

    Returns:
        trace: The original trace with dynamic batching applied.
    """
    for binding in get_model_bindings(trace):
        binding.set_dynamic_batching(enable_dynamic_batch)

    # NOTE: Returning the trace is required for backwards compatibility
    return trace


def async_load(trace: torch.nn.Module, enable_async_load: bool = True):
    """
    Enables (or disables) asynchronous model loading on a traced Neuron model.

    Arguments:
        trace: A trace from `torch_neuronx.trace` to enable dynamic batching on.
        enable_async_load: Whether to enabled (or disabled) async load.
    """
    for binding in get_model_bindings(trace):
        binding.set_async_load(enable_async_load)


def lazy_load(trace: torch.nn.Module, enable_lazy_load: bool = True):
    """
    Enables (or disables) lazy model loading on a traced Neuron model.

    Arguments:
        trace: A trace from `torch_neuronx.trace` to enable dynamic batching on.
        enable_lazy_load: Whether to enabled (or disabled) lazy load.
    """

    for binding in get_model_bindings(trace):
        binding.set_lazy_load(enable_lazy_load)
