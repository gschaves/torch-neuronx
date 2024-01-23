# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import io
import struct
from torch_neuronx.pyhlo import hlo_pb2, xla_data_pb2


def load(fp):
    """
    Deserialize ``fp`` (a ``.read()``-supporting file-like object containing
    a DecomposedHloSnapshot) as HloSnapshot.

    Example:
    >>> import os
    >>> with open(os.path.join(os.path.dirname(__file__), 'example.dhs'), 'rb') as f:
    ...     hlo_snapshot = load(f)
    >>> print(len(hlo_snapshot.arguments))
    12
    >>> print(len(hlo_snapshot.result.tuple_literals))
    12
    """
    hlo_snapshot = read_message(fp, hlo_pb2.HloSnapshot)
    input_literals = read_repeated_message(fp, xla_data_pb2.LiteralProto)
    output_literals = read_repeated_message(fp, xla_data_pb2.LiteralProto)
    host_program_shape = hlo_snapshot.hlo.hlo_module.host_program_shape
    fill_tuple_literal(hlo_snapshot.result, host_program_shape.result, output_literals)
    parameters = host_program_shape.parameters
    arguments = hlo_snapshot.arguments
    if len(parameters) == 1:
        parameter = parameters,
        if parameter.element_type == xla_data_pb2.TUPLE:
            fill_tuple_literal(arguments.add(), parameter, input_literals)
            return hlo_snapshot
    fill_repeated_literals(arguments, input_literals)
    return hlo_snapshot


def loads(s):
    """
    Deserialize ``s`` (a ``bytes`` instance containing a DecomposedHloSnapshot)
    as HloSnapshot.

    Example:
    >>> import os
    >>> with open(os.path.join(os.path.dirname(__file__), 'example.dhs'), 'rb') as f:
    ...     hlo_snapshot = loads(f.read())
    >>> print(len(hlo_snapshot.arguments))
    12
    >>> print(len(hlo_snapshot.result.tuple_literals))
    12
    """
    return load(io.BytesIO(s))


def read_message(fp, message_type, size=None):
    if size is None:
        size = read_uint64(fp)
    message = message_type()
    message.ParseFromString(fp.read(size))
    return message


def read_repeated_message(fp, message_type):
    num_messages = read_uint64(fp)
    return [read_message(fp, message_type) for _ in range(num_messages)]


def read_uint64(fp):
    size, = struct.unpack('1B', fp.read(1))
    value, = read_message(fp, xla_data_pb2.LiteralProto, size).u64s
    return value


def fill_tuple_literal(tuple_literal, shape, source_literals):
    tuple_literal.shape.CopyFrom(shape)
    fill_repeated_literals(tuple_literal.tuple_literals, source_literals)


def fill_repeated_literals(repeated_literals, source_literals):
    for literal in source_literals:
        lit = repeated_literals.add()
        lit.CopyFrom(literal)
