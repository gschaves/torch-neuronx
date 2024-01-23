# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
"""
Routines
========
High-level testing routines that encompass multiple testing functions.
"""

import pathlib
from typing import Tuple, Union, Callable, Optional, List

import torch

import torch_neuronx


def infer_dtype(
    func: Union[Callable, torch.nn.Module]
) -> Optional[Union[str, List[str]]]:
    """
    Infers a modules data type from its parameters

    Args:
        func: A module or function which defines a torch model or computation.

    Returns:
        The data type string, list of parmeter data type strings, or None
    """
    dtype = None
    if isinstance(func, torch.nn.Module):
        dtype = list(set(str(parameter.dtype) for parameter in func.parameters()))
        if len(dtype) == 1:
            dtype = next(iter(dtype))
        return dtype
    return dtype


def trace(
    func: Union[Callable, torch.nn.Module],
    example: Tuple[torch.Tensor, ...],
    *,
    compiler_workdir: Optional[Union[str, pathlib.Path]] = None,
    compiler_args: Optional[Union[List[str], str]] = None,
    **metadata
) -> torch.jit.ScriptModule:
    """
    A testing `torch_neuronx.trace` that adds validation and debug artifacts.

    This is a stand-in replacement for `torch_neuronx.trace`. However, in
    addition to a normal `torch_neuronx.trace`, this automatically saves:

    - Compiler artifacts (compiler_workdir/)
    - The torchscript model (model.pt)
    - Rudimentary benchmark statistics & metadata (metadata.json)
    - Inputs, expected, and actual tensor data (tensors/)

    Unlike `torch_neuronx.trace`, this does not currently support the `options`
    keyword argument or `Any` kind of `example`. The `example` is restricted to
    `Tuple` types.

    Artifacts are serialized to the standard testing path.

    Args:
        func: A module or function which defines a torch model or computation.
        example: An example set of inputs which will be passed to the
            `func` during tracing.
    Kwargs:
        compiler_workdir: The directory to serialized outputs to (uses
            the default test artifacts directory if unspecified).
        compiler_args: Additional compiler arguments.
        **metadata: A mapping of keys to values of extra data to serialize.

    Returns:
        A Module where the HLO computation is a fused neuron::foward operation.
    """

    # Save to standard path unless user specifies otherwise
    directory = torch_neuronx.testing.directory()
    if compiler_workdir is not None:
        directory = pathlib.Path(compiler_workdir)

    # Infer data type
    dtype = metadata.pop('data_type', infer_dtype(func))

    # Set evaluation mode prior to generating outputs
    if isinstance(func, torch.nn.Module):
        func.eval()

    # Note: Run on CPU before trace. Avoids running with XLA allocated params
    expected = func(*example)

    # Trace
    traced = torch_neuronx.trace(
        func,
        example,
        compiler_workdir=directory / 'compiler_workdir',
        compiler_args=compiler_args
    )

    # Serialize
    filename = directory / 'model.pt'
    torch.jit.save(traced, filename)

    # Run on Neuron
    actual = traced(*example)

    # Save tensors
    torch_neuronx.testing.save_tensors(
        path=directory,
        inputs=example,
        actual=actual,
        expected=expected,
    )

    # Benchmark
    metrics = torch_neuronx.testing.benchmark(filename, example)
    torch_neuronx.testing.dump(
        path=directory,
        compiler_args=compiler_args,
        data_type=dtype,
        **metadata,
        **metrics,
    )

    # Validate output. Do this last to avoid an early exit
    torch_neuronx.testing.assert_allclose(
        expected=expected,
        actual=actual,
        atol=1e-2,
        rtol=1e-2
    )

    return traced
