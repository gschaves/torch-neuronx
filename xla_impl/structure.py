"""
Structure
=========
A module for handling data structures containing tensors.

This module is oriented around generating and using a `layout` object which
is the nested lists/tuples of torch.Tensor identifiers that would be output
from a torch graph. This can be used to retrieve data from neuron runtime
and present the data back in the original torch function output layout.
"""
import torch
import itertools
import collections
from typing import Union, Dict, List, Tuple, Any, Iterable, Optional

# Type Annotations
Layout = Union[Dict["Layout", "Layout"], Tuple["Layout", ...], List["Layout"], int]


def extract(structure: Any) -> Tuple[Layout, Dict[torch.Tensor, int], Dict[int, Any]]:
    """
    Create a structural layout representation of a nested structure of tensors.

    Example:
        def forward(x):
            y = x + 1
            return x, [y, x], 'a'

        result = forward(0)
        layout, uniques, constants = structure.extract(result)

    Result:
        | layout         | uniques          | constants   |
        |----------------|------------------|-------------|
        | (0, [1, 0], 2) | {0: 0, 1: 3}     | {2: 'a'}    |

    Args:
        structure: The output of a function.

    Returns:
        layout: The structural layout with values replaced by identifiers.
        uniques: Map from Tensor to layout identifier.
        constants: Map from layout identifier to constant (non-Tensor) value.
    """

    counter = itertools.count()
    uniques = collections.defaultdict(lambda: next(counter))
    constants = dict()

    def process(structure: Any) -> Layout:
        if isinstance(structure, torch.Tensor):
            return uniques[structure]
        elif isinstance(structure, (list, tuple)):
            return type(structure)(process(item) for item in structure)
        elif isinstance(structure, dict):
            # Note: Avoid dictionary comprehension for testing determinism.
            #       Python 3.8+ flips key/value function call evaluation order.
            items = list()
            for key, value in structure.items():
                value = process(value)
                key = process(key)
                items.append((key, value))
            return dict(items)
        else:
            identifier = next(counter)
            constants[identifier] = structure
            return identifier

    layout = process(structure)
    return layout, dict(uniques), constants


def pack(layout: Layout, values: Union[Dict[int, Any], List[Any]]) -> Any:
    """
    Pack a ``layout`` of identifiers with corresponding ``values``.

    Note that the maximum unique identifier in the ``layout`` is expected to
    equal the length of ``values``.

    Example:
        layout = (0, [1, 0], 2)
        values = ['a', 'b', 'c']
        result = structure.pack(layout, values)

    Result:
        | layout         | values     |  result        |
        |----------------|------------|----------------|
        | (0, [1, 0], 2) | [a, b, c]  | (a, [b, a], c) |

    Args:
        layout: The structural layout with values replaced by identifiers.
        values: The values to insert into the layout.

    Returns:
        The original layout packed with values
    """
    if isinstance(layout, (tuple, list)):
        return type(layout)(pack(item, values) for item in layout)
    if isinstance(layout, dict):
        return {pack(key, values): pack(value, values) for key, value in layout.items()}
    if layout >= len(values):
        raise IndexError(
            f"Found layout index={layout} but only {len(values)} values were given."
        )
    return values[layout]


def flatten(structure: Any) -> Tuple[torch.Tensor, ...]:
    """
    Flattens a data structure of tensors into a Tuple.

    Args:
        structure: The output of a function.

    Returns:
        The tensors from the original structure in a flattened tuple.
    """
    result = ()
    if isinstance(structure, torch.Tensor):
        result = (structure,)
    elif isinstance(structure, (list, tuple)):
        for item in structure:
            result += flatten(item)
    elif isinstance(structure, dict):
        for key, value in structure.items():
            result += flatten(key)
            result += flatten(value)
    return result


class Packer:
    """
    Packs a flat collection of values into a heirarchical layout.

    Example:

        packer = Packer(
            layout=[0, (1, 2), 1],
            identifiers=(0, 2),
            constants={1: 'a'}
        )
        result = packer(('b', 'c'))

        assert result == ['b', ('a', 'c'), 'a']

    Args:
        layout: The structural layout to pack.
        indentifiers: The layout indentifier to associate with each input.
        constants: Map from layout identifier to constant (non-Tensor) value.
    """

    def __init__(
        self, layout: Layout, identifiers: Tuple[int], constants: Dict[int, Any]
    ) -> None:
        self.layout = layout
        self.identifiers = identifiers
        self.constants = constants

    def __call__(self, values: Iterable[Any]):
        """
        Packs values associated with self.identifiers into self.layout.

        Args:
            values: The values to pack into the layout.

        Returns:
            The original layout packed with values.
        """
        assert isinstance(values, (tuple, list))
        assert len(values) == len(self.identifiers)
        lookup = dict(zip(self.identifiers, values))
        lookup.update(self.constants)
        return pack(self.layout, lookup)


class Flattener:
    """
    Flattens a data structure of tensors into a list of unique Tensors.

    This optionally validates that the layout of the input is identical to an
    existing ``layout``.

    Since some inputs may be unused, this class allows you to ``exclude``
    indices from the resulting list.

    Args:
        layout: Assert that the input layout is equivalent to this parameter.
        exclude: The resulting list indices to exclude.
    """

    def __init__(
        self,
        layout: Optional[Layout] = None,
        exclude: Optional[Iterable[int]] = None,
    ) -> None:
        self.layout = layout
        self.exclude = exclude

    def __call__(self, structure: Any) -> List[torch.Tensor]:
        layout, uniques, _ = extract(structure)
        if self.layout is not None:
            assert self.layout == layout
        tensors, _ = zip(*uniques.items())
        if self.exclude:
            return [
                tensor
                for index, tensor in enumerate(tensors)
                if index not in self.exclude
            ]
        return list(tensors)
