# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
"""
Validation
==========
Functions to help with data validation
"""
import torch

from typing import Union, Tuple, List, Optional


def assert_allclose(
    expected: Union[List, Tuple, torch.Tensor],
    actual: Union[List, Tuple, torch.Tensor],
    rtol: Optional[float] = 1e-5,
    atol: Optional[float] = 1e-5
):
    """
    Assert that an actual torch output is equal to the expected value.

    Unlike normal torch equality checking, this recursively traverses
    structured outputs to ensure that structures and their internal values are
    equal.

    Args:
        expected: The expected network output.
        actual: The actual network output.
        rtol: Relative tolerance when checking result equality.
        rtol: Absolute tolerance when checking result equality.

    Raises:
        AssertionError: Error when either the type, shape, or values differ.
    """
    assert type(expected) == type(actual), (
        f'Type Mismatch {type(expected)} != {type(actual)}'
    )

    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(expected, actual, rtol=rtol, atol=atol)

    elif isinstance(expected, (tuple, list)):
        for items in zip(expected, actual):
            assert_allclose(*items, rtol=rtol, atol=atol)

    else:
        assert expected == actual, (
            f'Equality Failure {expected} != {actual}'
        )
