import os
import warnings

import torch


def load():
    """
    Loads the torch `libtorchneuron.so` plugin library.

    This plugin library contains the custom operations and classes which enable
    torchscript execution of Neuron operations.
    """
    try:
        torch.ops.neuron.forward_v2
        warnings.warn('A different version of "libtorchneuron.so" has already been loaded.')
        return
    except (RuntimeError, AttributeError):
        pass # This means libtorchneuron has not been loaded

    # Load the neuron operation torch library
    directory = os.path.dirname(__file__)
    filename = os.path.join(directory, 'lib', 'libtorchneuron.so')
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f'Did not find "libtorchneuron.so" in {directory}'
        )
    torch.ops.load_library(filename)
