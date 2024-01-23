# Remove since it can cause a cascading failure if torch_xla is import first
# Must import torch_neuronx.experimental.profiler directly
#
# from . import profiler

from .placement import (
    set_neuron_cores,
    set_multicore,
    neuron_cores_context,
    multicore_context,
)
