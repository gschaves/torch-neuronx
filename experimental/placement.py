# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================

"""
Placement
=========
Methods which enable placement of torch Modules to specific NeuronCores.
"""
import contextlib
import typing

import torch

import torch_neuronx


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def _get_model_bindings(trace: torch.ScriptModule) -> typing.List[torch.ScriptObject]:
    return torch_neuronx.xla_impl.torchscript.get_model_bindings(trace)


def _get_neuron_visible_core_count() -> int:
    """
    Retieve the number of NeuronCores visible to this process.

    Returns:
        The number of visible neuron cores.

    Raises:
        RuntimeError: If the Neuron runtime cannot be initialized. This most
            commonly occurs when executing on an instance with no Neuron
            devices available or when no Neuron devices are visible to the
            process.
    """
    runtime = torch.classes.neuron.Runtime()
    try:
        nc_count = runtime.get_visible_nc_count()
    except RuntimeError as e:
        raise RuntimeError(
            "Neuron runtime cannot be initialized; cannot determine the number of available NeuronCores"  # noqa: E501
        ) from e
    return nc_count


def _validate_nc_count(nc_count: int):
    """
    A naive check to ensure `nc_count` is a reasonable value.

    This does not perform a model-specific check to ensure that the `nc_count`
    is valid. We do not have a guarantee that a model is loaded to a NeuronCore
    so this means that we can only check if the runtime has enough cores
    available. If a specific model requires more cores than the cores visible
    to the process, a runtime error will occur at model load time.

    Arguments:
        nc_count: The number of neuron cores.

    Raises:
        RuntimeError: If the Neuron runtime cannot be initialized.
        ValueError: If the nc_count is an invalid number of NeuronCores.
    """
    if nc_count == 0:
        raise ValueError(
            f"Invalid NeuronCore count nc_count={nc_count}. Cannot be set to 0"
        )

    if nc_count < -1:
        raise ValueError(
            f"Invalid NeuronCore count nc_count={nc_count}. Values less than -1 are invalid"  # noqa: E501
        )

    if nc_count > 0:
        visible_core_count = _get_neuron_visible_core_count()
        if nc_count > visible_core_count:
            raise ValueError(
                f"Invalid NeuronCore count nc_count={nc_count}. NeuronCores available: {visible_core_count}"  # noqa: E501
            )


# -----------------------------------------------------------------------------
# Public API Functions
# -----------------------------------------------------------------------------


def set_neuron_cores(trace: torch.ScriptModule, start_nc: int = -1, nc_count: int = -1):
    """
    Set the NeuronCore start/count for all Neuron subgraphs in a torch Module.

    This will unload the model from an existing NeuronCore if it is already
    loaded.

    Arguments:
        trace: A torch module which contains one or more Neuron subgraphs.
        start_nc: The starting NeuronCore index where the Module is placed. The
            value -1 automatically loads to the optimal NeuronCore (least
            used). Note that this index is always relative to NeuronCores
            visible to this process.
        nc_count: The number of NeuronCores to use. The value -1 will load
            a model to exactly the number of cores required by that model (1 for
            most models, >1 when using NeuronCore Pipeline). If nc_count
            is greater than the number of NeuronCores required by the
            model, the model will be replicated across multiple
            NeuronCores. (replications = floor(nc_count / cores_per_model))

    Raises:
        RuntimeError: If the Neuron runtime cannot be initialized.
        ValueError: If the nc_count is an invalid number of NeuronCores.

    Examples:
        Move a model to the first visible NeuronCore after loading:

        >>> model = torch.jit.load('example_neuron_model.pt')
        >>> set_neuron_cores(model, start_nc=0, nc_count=1)
    """

    _validate_nc_count(nc_count)
    bindings = _get_model_bindings(trace)
    for binding in bindings:
        binding.set_neuron_devices(start_nc, nc_count)


def set_multicore(trace: torch.ScriptModule):
    """
    Loads all Neuron subgraphs in a torch Module to all visible NeuronCores.

    This loads each Neuron subgraph within a Module to multiple NeuronCores
    without requiring multiple calls to `torch.jit.load`. This allows a single
    Module to use multiple NeuonCores for concurrent threadsafe inferences.
    Requests use a round-robin strategy to distribute across NeuronCores.

    This will unload the model from an existing NeuronCore if it is already
    loaded.

    Arguments:
        trace: A torch module which contains one or more Neuron subgraphs.

    Raises:
        RuntimeError: If the Neuron runtime cannot be initialized.

    Examples:
        Replicate a model across all visible NeuonCores after loading:

        >>> model = torch.jit.load('example_neuron_model.pt')
        >>> set_multicore(model)
    """
    nc_count = _get_neuron_visible_core_count()
    set_neuron_cores(trace, 0, nc_count)


@contextlib.contextmanager
def neuron_cores_context(start_nc: int = -1, nc_count: int = -1):
    """
    A context which sets the NeuronCore start/count for all Neuron subgraphs.

    Any calls to `torch.jit.load` or `torch_neuronx.trace` will cause the
    underlying Neuron subgraphs to load to the NeuronCores within this context.
    This context manager only needs to be used during the model load.
    After loading, inferences do not need to occur in this context in order
    to use the correct NeuronCores.

    Note that this context is *not* threadsafe. Using multiple core placement
    contexts from multiple threads may not correctly place models.

    Arguments:
        start_nc: The starting NeuronCore index where the Module is placed. The
            value -1 automatically loads to the optimal NeuronCore (least
            used). Note that this index is always relative to NeuronCores
            visible to this process.
        nc_count: The number of NeuronCores to use. The value -1 will load
            a model to exactly the number of cores required by that model (1 for
            most models, >1 when using NeuronCore Pipeline). If nc_count
            is greater than the number of NeuronCores required by the
            model, the model will be replicated across multiple
            NeuronCores. (replications = floor(nc_count / cores_per_model))

    Raises:
        RuntimeError: If the Neuron runtime cannot be initialized.
        ValueError: If the nc_count is an invalid number of NeuronCores.

    Examples:
        Load a model to the first visible NeuronCore:

        >>> with neuron_cores_context(start_nc=0, nc_count=1):
        >>>     model = torch.jit.load('example_neuron_model.pt')
    """
    runtime = torch.classes.neuron.Runtime()
    old_start_nc = runtime.get_default_start_nc()
    old_nc_count = runtime.get_default_nc_count()

    _validate_nc_count(nc_count)

    runtime.set_default_neuron_cores(start_nc, nc_count)
    try:
        yield
    finally:
        runtime.set_default_neuron_cores(old_start_nc, old_nc_count)


@contextlib.contextmanager
def multicore_context():
    """
    A context which loads all Neuron subgraphs to all visible NeuronCores.

    This loads each Neuron subgraph within a Module to multiple NeuronCores
    without requiring multiple calls to `torch.jit.load`. This allows a single
    Module to use multiple NeuonCores for concurrent threadsafe inferences.
    Requests use a round-robin strategy to distribute across NeuronCores.

    Any calls to `torch.jit.load` or `torch_neuronx.trace` will cause the
    underlying Neuron subgraphs to load to the NeuronCores within this context.
    This context manager only needs to be used during the model load.
    After loading, inferences do not need to occur in this context in order
    to use the correct NeuronCores.

    Note that this context is *not* threadsafe. Using multiple core placement
    contexts from multiple threads may not correctly place models.

    Raises:
        RuntimeError: If the Neuron runtime cannot be initialized.

    Examples:
        Load a model across all visible NeuonCores:

        >>> with multicore_context():
        >>>     model = torch.jit.load('example_neuron_model.pt')
    """
    nc_count = _get_neuron_visible_core_count()

    runtime = torch.classes.neuron.Runtime()
    old_start_nc = runtime.get_default_start_nc()
    old_nc_count = runtime.get_default_nc_count()

    runtime.set_default_neuron_cores(0, nc_count)
    try:
        yield
    finally:
        runtime.set_default_neuron_cores(old_start_nc, old_nc_count)
