# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
"""
Artifacts
=========
Functions to help with serialization of artifacts
"""

import os
import json
import tempfile
import pathlib
import warnings
from typing import Optional, Tuple, Union

import torch
import numpy as np


def pytest_path() -> Optional[str]:
    """
    Create a path from the current python test.

    This uses the full test directory path, followed by the test filename,
    followed by the test name. For example:

        unit/test_hlo_conversion/test_xla_trace_broadcast_add

    Returns:
        A unique path to the test.
    """

    name = os.environ.get('PYTEST_CURRENT_TEST', None)
    if name is None:
        return None

    # Strip off root test directory name
    if name.startswith('test/'):
        name = name[5:]

    # Remove pytest suffix which indicates test is being executed
    if name.endswith(' (call)'):
        name = name[:-7]
    # Remove pytest suffix which occurs when used in a fixture
    elif name.endswith(' (setup)'):
        name = name[:-8]

    # Split at the filename/test name boundary
    parts = name.split('.py::')

    # handling for model names such as 'google/flan-t5-xl' or 'microsfot/deBerta'
    if ('/') in parts[-1]:
        parts[-1] = parts[-1].replace("/", "-")

    path = os.path.join(*parts)
    return path


def directory(root: str = 'test-artifacts') -> pathlib.Path:
    """
    Create and return a temporary directory to store test artifacts into.

    Args:
        root: The root directory to store all artifacts to.

    Returns:
        A unique path to the test.
    """
    suffix = pytest_path()
    if suffix is None:
        os.makedirs(root, exist_ok=True)
        path = tempfile.mkdtemp(dir=root)
        warnings.warn(f'No pytest session found. Using directory: {path}')
    else:
        path = os.path.join(root, suffix)
    os.makedirs(path, exist_ok=True)
    return pathlib.Path(path)


def dump(path: Optional[Union[str, pathlib.Path]] = None, **metadata) -> None:
    """
    Serialize a key -> value collection to the artifacts directory.

    This produces a `metadata.json` file in the default directory.

    Kwargs:
        path: The directory path to write the file to.
        metadata: A mapping of keys and values.
    """
    if path is None:
        path = directory()
    path = os.path.join(path, 'metadata.json')

    with open(path, 'w') as f:
        json.dump(metadata, f, indent=4)


def save_tensors(
    inputs: Union[torch.Tensor, Tuple[torch.Tensor]],
    expected: Union[torch.Tensor, Tuple[torch.Tensor]],
    actual: Union[torch.Tensor, Tuple[torch.Tensor]],
    path: Optional[Union[str, pathlib.Path]] = None,
    ignore_errors: bool = True,
):
    """
    Write `inputs`, `actual`, and `expected` tensors as numpy arrays to `path`.

    Names are set according to the tuple order:
    - <path>/tensors/inputs/0.npy
    - <path>/tensors/inputs/1.npy
    - ...
    - <path>/tensors/actual/0.npy
    - <path>/tensors/actual/1.npy
    - ...
    - <path>/tensors/expected/0.npy
    - <path>/tensors/expected/1.npy
    - ...

    Args:
        inputs: The input tensors.
        expected: The expected output tensors (i.e. from CPU).
        actual: The actual output tensors (i.e. from Neuron).
        path: The directory path to write the file to.
        ignore_errors: Whether to error or warn on errors when serializing.
    """
    if path is None:
        path = directory()
    path = os.path.join(path, 'tensors')

    sets = {
        'inputs': inputs,
        'actual': actual,
        'expected': expected,
    }

    def error(message):
        if not ignore_errors:
            raise AssertionError(message)
        warnings.warn(message)

    for prefix, tensors in sets.items():

        os.makedirs(os.path.join(path, prefix), exist_ok=True)

        if isinstance(tensors, torch.Tensor):  # Wrap single tensors
            tensors = (tensors,)

        if not isinstance(tensors, tuple):
            error(
                f'Could not save "{prefix}" tensors. '
                f'Expected type {tuple} but got {type(tensors)}'
            )
            continue

        for i, tensor in enumerate(tensors):

            if not isinstance(tensor, torch.Tensor):
                error(
                    f'Could not save "{prefix}" tensor #{i}. '
                    f'Expected type {torch.Tensor} but got {type(tensor)}'
                )
                continue

            filename = os.path.join(path, prefix, f'{i}.npy')
            np.save(filename, tensor.detach().numpy())
