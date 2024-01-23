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
from contextlib import contextmanager
import os
import traceback
from torch_neuronx.pyhlo import hlo_pb2, xla_data_pb2


class HloScribe:
    """
    "Scribe" a Python function that looks like HLO into HloModuleProto.

    Example:
    >>> from torch_neuronx.pyhlo.scribe import HloScribe
    >>>
    ... def func_dot_add(scribe):
    ...    f32 = scribe.f32
    ...    lhs = f32[16,6].Parameter(parameter_number=0)
    ...    rhs = f32[16,8].Parameter(parameter_number=1)
    ...    bias = f32[8].Parameter(parameter_number=2)
    ...    dot_dims = dict(lhs_contracting_dimensions=[0], rhs_contracting_dimensions=[0])
    ...    dot = f32[6,8].Dot(lhs, rhs, dot_dimension_numbers=dot_dims)
    ...    bias = f32[6,8].Broadcast(bias, dimensions=[0])
    ...    return f32[6,8].Add(dot, bias)
    ...
    >>>
    >>> # To write constant instructions, we will need `HloScribe(serialize_torch)`
    >>> # instead of `HloScribe()`, where `serialize_torch` is defined in `pyhlo.constant`,
    >>> # e. g,:
    >>> # `from torch_neuronx.pyhlo.constant.serialize_torch import serialize_torch`
    >>> # (there are `serialize_tf` and `serialize_numpy` as well)
    >>>
    >>> hlo_dot_add = HloScribe()(func_dot_add)
    >>> print(hlo_dot_add.module_proto)
    name: "FuncDotAdd.7"
    entry_computation_name: "FuncDotAdd.7"
    computations {
      name: "FuncDotAdd.7"
      instructions {
        name: "p0.1"
        opcode: "parameter"
        shape {
          element_type: F32
          dimensions: 16
          dimensions: 6
          layout {
            minor_to_major: 1
            minor_to_major: 0
          }
          is_dynamic_dimension: false
          is_dynamic_dimension: false
        }
        id: 1
      }
      instructions {
        name: "p1.2"
        opcode: "parameter"
        shape {
          element_type: F32
          dimensions: 16
          dimensions: 8
          layout {
            minor_to_major: 1
            minor_to_major: 0
          }
          is_dynamic_dimension: false
          is_dynamic_dimension: false
        }
        parameter_number: 1
        id: 2
      }
      instructions {
        name: "p2.3"
        opcode: "parameter"
        shape {
          element_type: F32
          dimensions: 8
          layout {
            minor_to_major: 0
          }
          is_dynamic_dimension: false
        }
        parameter_number: 2
        id: 3
      }
      instructions {
        name: "dot.4"
        opcode: "dot"
        shape {
          element_type: F32
          dimensions: 6
          dimensions: 8
          layout {
            minor_to_major: 1
            minor_to_major: 0
          }
          is_dynamic_dimension: false
          is_dynamic_dimension: false
        }
        dot_dimension_numbers {
          lhs_contracting_dimensions: 0
          rhs_contracting_dimensions: 0
        }
        id: 4
        operand_ids: 1
        operand_ids: 2
      }
      instructions {
        name: "broadcast.5"
        opcode: "broadcast"
        shape {
          element_type: F32
          dimensions: 6
          dimensions: 8
          layout {
            minor_to_major: 1
            minor_to_major: 0
          }
          is_dynamic_dimension: false
          is_dynamic_dimension: false
        }
        dimensions: 0
        id: 5
        operand_ids: 3
      }
      instructions {
        name: "add.6"
        opcode: "add"
        shape {
          element_type: F32
          dimensions: 6
          dimensions: 8
          layout {
            minor_to_major: 1
            minor_to_major: 0
          }
          is_dynamic_dimension: false
          is_dynamic_dimension: false
        }
        id: 6
        operand_ids: 4
        operand_ids: 5
      }
      program_shape {
        parameters {
          element_type: F32
          dimensions: 16
          dimensions: 6
          layout {
            minor_to_major: 1
            minor_to_major: 0
          }
          is_dynamic_dimension: false
          is_dynamic_dimension: false
        }
        parameters {
          element_type: F32
          dimensions: 16
          dimensions: 8
          layout {
            minor_to_major: 1
            minor_to_major: 0
          }
          is_dynamic_dimension: false
          is_dynamic_dimension: false
        }
        parameters {
          element_type: F32
          dimensions: 8
          layout {
            minor_to_major: 0
          }
          is_dynamic_dimension: false
        }
        result {
          element_type: F32
          dimensions: 6
          dimensions: 8
          layout {
            minor_to_major: 1
            minor_to_major: 0
          }
          is_dynamic_dimension: false
          is_dynamic_dimension: false
        }
        parameter_names: "p0"
        parameter_names: "p1"
        parameter_names: "p2"
      }
      id: 7
      root_id: 6
    }
    host_program_shape {
      parameters {
        element_type: F32
        dimensions: 16
        dimensions: 6
        layout {
          minor_to_major: 1
          minor_to_major: 0
        }
        is_dynamic_dimension: false
        is_dynamic_dimension: false
      }
      parameters {
        element_type: F32
        dimensions: 16
        dimensions: 8
        layout {
          minor_to_major: 1
          minor_to_major: 0
        }
        is_dynamic_dimension: false
        is_dynamic_dimension: false
      }
      parameters {
        element_type: F32
        dimensions: 8
        layout {
          minor_to_major: 0
        }
        is_dynamic_dimension: false
      }
      result {
        element_type: F32
        dimensions: 6
        dimensions: 8
        layout {
          minor_to_major: 1
          minor_to_major: 0
        }
        is_dynamic_dimension: false
        is_dynamic_dimension: false
      }
      parameter_names: "p0"
      parameter_names: "p1"
      parameter_names: "p2"
    }
    id: 7
    entry_computation_id: 7
    <BLANKLINE>
    """

    def __init__(self, constant_serializer=None, program_counter=0):
        for key, value in xla_data_pb2.PrimitiveType.items():
            setattr(self, key.lower(), HloShape(self, value))
        self.constant_serializer = constant_serializer
        self._program_counter = program_counter
        self._registered_aliases = []

    def __call__(self, func):
        scribe = HloScribe(self.constant_serializer, self._program_counter)
        scribe.scribe(func)
        return scribe

    def get_dtype(self, dtype_name):
        return getattr(self, dtype_name)

    def scribe(self, compute_def):
        module = hlo_pb2.HloModuleProto()
        self.module_proto = module
        computation = module.computations.add()
        self.entry_computation = computation
        with self.scribe_context():
            root_shape = compute_def(self)
        program_shape = computation.program_shape
        instructions = computation.instructions
        parameter_opcode, _ = HloShapeType.opCodeMap['Parameter']
        parameters = [inst for inst in instructions if inst.opcode == parameter_opcode]
        for inst in parameters:
            inst.name = f'p{inst.parameter_number}.{inst.id}'
        for idx, inst in enumerate(parameters):
            if inst.parameter_number != idx:
                raise ValueError(f'{inst.name}: parameter_number needs to be {idx}')
        for inst in parameters:
            program_shape.parameter_names.append(f'p{inst.parameter_number}')
            program_shape.parameters.add().CopyFrom(inst.shape)
        program_shape.result.CopyFrom(root_shape.shape_proto)
        computation.root_id = root_shape.instruction.id
        module.id = module.entry_computation_id = computation.id = self.get_program_counter()
        compute_def_name = compute_def.__name__
        if not is_camel(compute_def_name):
            compute_def_name = snake_to_camel(compute_def_name)
        computation.name = f'{compute_def_name}.{computation.id}'
        module.name = module.entry_computation_name = computation.name
        module.host_program_shape.CopyFrom(program_shape)

        # setup input_output_alias if necessary
        if not self._registered_aliases:
            return
        id_to_inst = {inst.id: inst for inst in instructions}
        pnum_to_inst = {inst.parameter_number: inst for inst in parameters}

        # 1-output aliasing
        if len(self._registered_aliases) == 1 and not is_tuple(root_shape.shape_proto):
            iid, parameter_number, is_must = next(iter(self._registered_aliases))
            inst = id_to_inst[iid]
            if iid != root_shape.instruction.id:
                raise ValueError(f'{inst.name} is not the root instruction and cannot alias.')
            add_alias_entry(module, parameter_number, is_must)
            return

        # tuple output aliasing
        if not is_tuple(root_shape.shape_proto):
            raise TypeError(f'root instruction {root_shape.instruction.name} must be a tuple'
                            f' for input output alias {self._registered_aliases}')
        # TODO: implement
        for inst in parameters:
            if is_tuple(inst.shape):
                raise NotImplemented(f'aliasing to nested tuple parameter {inst.name} is unimplemented')
        iid_to_output_index = {}
        for index, oid in enumerate(root_shape.instruction.operand_ids):
            inst = id_to_inst[oid]
            # TODO: implement
            if is_tuple(inst.shape):
                raise NotImplemented(f'aliasing from nested tuple output {inst.name} is unimplemented')
            iid_to_output_index[inst.id] = index
        root_name = root_shape.instruction.name
        for iid, parameter_number, is_must in self._registered_aliases:
            inst = id_to_inst[iid]
            if iid not in iid_to_output_index:
                raise ValueError(f'{inst.name} must be consumed by the root instruction {root_name}')
            output_index = iid_to_output_index[iid]
            entry = add_alias_entry(module, parameter_number, is_must)
            entry.output_shape_index.append(output_index)

    def get_program_counter(self):
        self._program_counter += 1
        return self._program_counter

    def set_program_counter(self, program_counter):
        self._program_counter = program_counter

    context = None

    @contextmanager
    def scribe_context(self):
        old_context = HloScribe.context
        HloScribe.context = self
        try:
            yield
        finally:
            HloScribe.context = old_context

    literalBytesFields = {'s8s', 'u8s', 'f16s', 'bf16s', 'u16s', 's16s'}

    def write_literal(self, inst, value):
        literal = inst.literal
        literal.shape.CopyFrom(inst.shape)
        element_type = inst.shape.element_type
        element_type_name = xla_data_pb2.PrimitiveType.Name(element_type)
        literal_attr_name = f'{element_type_name.lower()}s'
        if self.constant_serializer is None:
            raise ValueError('HloScribe requires constant_serializer for constant instructions')
        if literal_attr_name in HloScribe.literalBytesFields:
            setattr(literal, literal_attr_name, self.constant_serializer(value, element_type))
        else:
            if not is_iterable(value):
                value = [value]
            getattr(literal, literal_attr_name)[:] = value

    def register_alias(self, inst_id, parameter_number, is_must):
        self._registered_aliases.append((inst_id, parameter_number, is_must))


def is_camel(name):
    return name != name.lower() and name != name.upper() and '_' not in name


def snake_to_camel(snake):
    return ''.join(s.title() for s in snake.split('_'))


def add_alias_entry(module, parameter_number, is_must):
    entry = module.input_output_alias.entries.add()
    entry.parameter_number = parameter_number
    entry.kind = hlo_pb2.Kind.MAY_ALIAS
    if is_must:
        entry.kind = hlo_pb2.Kind.MUST_ALIAS
    return entry


class HloShapeType(type):

    kHloOpcodeIsVariadic = -1
    kConstantValue = 'constant_value'
    opCodeMap = {
        'Abs': ('abs', 1),
        'Add': ('add', 2),
        'AddDependency': ('add-dependency', 2),
        'AfterAll': ('after-all', kHloOpcodeIsVariadic),
        'AllGather': ('all-gather', kHloOpcodeIsVariadic),
        'AllReduce': ('all-reduce', kHloOpcodeIsVariadic),
        'AllReduceScatter': ('all-reduce-scatter', kHloOpcodeIsVariadic),
        'AllReduceStart': ('all-reduce-start', kHloOpcodeIsVariadic),
        'AllReduceDone': ('all-reduce-done', 1),
        'AllToAll': ('all-to-all', kHloOpcodeIsVariadic),
        'Atan2': ('atan2', 2),
        'BatchNormGrad': ('batch-norm-grad', 5),
        'BatchNormInference': ('batch-norm-inference', 5),
        'BatchNormTraining': ('batch-norm-training', 3),
        'Bitcast': ('bitcast', 1),
        'BitcastConvert': ('bitcast-convert', 1),
        'Broadcast': ('broadcast', 1),
        'Call': ('call', kHloOpcodeIsVariadic),
        'Ceil': ('ceil', 1),
        'Cholesky': ('cholesky', 1),
        'Clamp': ('clamp', 3),
        'CollectivePermute': ('collective-permute', kHloOpcodeIsVariadic),
        'CollectivePermuteStart': ('collective-permute-start', kHloOpcodeIsVariadic),
        'CollectivePermuteDone': ('collective-permute-done', 1),
        'Clz': ('count-leading-zeros', 1),
        'Compare': ('compare', 2),
        'Complex': ('complex', 2),
        'Concatenate': ('concatenate', kHloOpcodeIsVariadic),
        'Conditional': ('conditional', kHloOpcodeIsVariadic),
        'Constant': ('constant', 0),
        'Convert': ('convert', 1),
        'Convolution': ('convolution', 2),
        'Copy': ('copy', 1),
        'CopyDone': ('copy-done', 1),
        'CopyStart': ('copy-start', 1),
        'Cos': ('cosine', 1),
        'CustomCall': ('custom-call', kHloOpcodeIsVariadic),
        'Divide': ('divide', 2),
        'Domain': ('domain', 1),
        'Dot': ('dot', 2),
        'DynamicSlice': ('dynamic-slice', kHloOpcodeIsVariadic),
        'DynamicUpdateSlice': ('dynamic-update-slice', kHloOpcodeIsVariadic),
        'Exp': ('exponential', 1),
        'Expm1': ('exponential-minus-one', 1),
        'Fft': ('fft', 1),
        'Floor': ('floor', 1),
        'Fusion': ('fusion', kHloOpcodeIsVariadic),
        'Gather': ('gather', 2),
        'GetDimensionSize': ('get-dimension-size', 1),
        'SetDimensionSize': ('set-dimension-size', 2),
        'GetTupleElement': ('get-tuple-element', 1),
        'Imag': ('imag', 1),
        'Infeed': ('infeed', 1),
        'Iota': ('iota', 0),
        'IsFinite': ('is-finite', 1),
        'Log': ('log', 1),
        'Log1p': ('log-plus-one', 1),
        'Logistic': ('logistic', 1),
        'And': ('and', 2),
        'Not': ('not', 1),
        'Or': ('or', 2),
        'Xor': ('xor', 2),
        'Map': ('map', kHloOpcodeIsVariadic),
        'Maximum': ('maximum', 2),
        'Minimum': ('minimum', 2),
        'Multiply': ('multiply', 2),
        'Negate': ('negate', 1),
        'Outfeed': ('outfeed', 2),
        'Pad': ('pad', 2),
        'Parameter': ('parameter', 0),
        'PartitionId': ('partition-id', 0),
        'PopulationCount': ('popcnt', 1),
        'Power': ('power', 2),
        'Real': ('real', 1),
        'Recv': ('recv', 1),
        'RecvDone': ('recv-done', 1),
        'Reduce': ('reduce', kHloOpcodeIsVariadic),
        'ReducePrecision': ('reduce-precision', 1),
        'ReduceScatter': ('reduce-scatter', 1),
        'ReduceWindow': ('reduce-window', kHloOpcodeIsVariadic),
        'Remainder': ('remainder', 2),
        'ReplicaId': ('replica-id', 0),
        'Reshape': ('reshape', 1),
        'DynamicReshape': ('dynamic-reshape', kHloOpcodeIsVariadic),
        'Reverse': ('reverse', 1),
        'Rng': ('rng', kHloOpcodeIsVariadic),
        'RngGetAndUpdateState': ('rng-get-and-update-state', 0),
        'RngBitGenerator': ('rng-bit-generator', 1),
        'RoundNearestAfz': ('round-nearest-afz', 1),
        'RoundNearestEven': ('round-nearest-even', 1),
        'Rsqrt': ('rsqrt', 1),
        'Scatter': ('scatter', 3),
        'Select': ('select', 3),
        'SelectAndScatter': ('select-and-scatter', 3),
        'Send': ('send', 2),
        'SendDone': ('send-done', 1),
        'ShiftLeft': ('shift-left', 2),
        'ShiftRightArithmetic': ('shift-right-arithmetic', 2),
        'ShiftRightLogical': ('shift-right-logical', 2),
        'Sign': ('sign', 1),
        'Sin': ('sine', 1),
        'Slice': ('slice', 1),
        'Sort': ('sort', kHloOpcodeIsVariadic),
        'Sqrt': ('sqrt', 1),
        'Cbrt': ('cbrt', 1),
        'Subtract': ('subtract', 2),
        'Tanh': ('tanh', 1),
        'Trace': ('trace', 1),
        'Transpose': ('transpose', 1),
        'TriangularSolve': ('triangular-solve', 2),
        'Tuple': ('tuple', kHloOpcodeIsVariadic),
        'TupleSelect': ('tuple-select', 3),
        'While': ('while', 1),
    }
    opCodeWithSubComputation = {
        'all-reduce',
        'call',
        'map',
        'reduce',
        'reduce-scatter',
        'reduce-window',
        'scatter',
        'select-and-scatter',
        'sort',
    }
    opCodeWithReplicaGroups = {
        'all-gather',
        'all-reduce',
        'all-to-all',
        'reduce-scatter',
    }

    def __new__(mcs, name, bases, dct):
        for fname, opdef in mcs.opCodeMap.items():
            dct[fname] = mcs.gen_api(fname, opdef)
        return super().__new__(mcs, name, bases, dct)

    @classmethod
    def gen_api(cls, fname, opdef):
        opcode, arity = opdef

        def api(self, *operands, **kwargs):
            self = self.clone()
            if arity != cls.kHloOpcodeIsVariadic and len(operands) != arity:
                raise ValueError(f'{fname} accepts {arity} operands; received {len(operands)}')
            inst = self.scribe.entry_computation.instructions.add()
            inst.opcode = opcode
            inst.id = self.scribe.get_program_counter()
            inst.name = f'{opcode}.{inst.id}'
            inst.operand_ids[:] = [op.instruction.id for op in operands]
            inst.shape.CopyFrom(self.shape_proto)

            if os.environ.get('ENABLE_PYHLO_FILE_METADATA', None) == '1':
                stack = traceback.extract_stack(limit=2)
                inst.metadata.source_file = stack[0].filename
                inst.metadata.op_type = stack[0].name # Use the function name as the op_type
                inst.metadata.source_line = stack[0].lineno

            if opcode in cls.opCodeWithSubComputation:
                to_apply = kwargs.pop('to_apply')
                scribe = HloScribe(self.scribe.constant_serializer, self.scribe.get_program_counter())
                computation = scribe(to_apply).entry_computation
                computation.name = f'{inst.name}.{computation.name}'
                self.scribe.module_proto.computations.insert(0, computation)
                inst.called_computation_ids.append(computation.id)
                self.scribe.set_program_counter(computation.id)
            if opcode in cls.opCodeWithReplicaGroups:
                replica_groups = []
                for group in kwargs['replica_groups']:
                    if not isinstance(group, xla_data_pb2.ReplicaGroup):
                        group = xla_data_pb2.ReplicaGroup(replica_ids=group)
                    replica_groups.append(group)
                kwargs['replica_groups'] = replica_groups
            for key, value in kwargs.items():
                if key == cls.kConstantValue:
                    self.scribe.write_literal(inst, value)
                    continue
                proto_set_attr(inst, key, value)
            self.instruction = inst
            return self

        return api


def proto_set_attr(field, key, value):
    attr = field if key is None else getattr(field, key)
    if isinstance(value, dict):
        for sk, sv in value.items():
            proto_set_attr(attr, sk, sv)
        return
    if isinstance(attr, (int, float, str, bytes)):
        setattr(field, key, value)
    elif callable(getattr(attr, 'extend', None)):   # a "Repeated" field
        if callable(getattr(attr, 'add', None)):    # difference between RepeatedComposite and RepeatedScalar
            for item in value:
                proto_set_attr(attr.add(), None, item)
        else:
            attr[:] = value
    else:
        # Ultimate fall-back path if attr is something completely foreign -- it requires clients
        # to build proto attribute themselves and pass it to us directly.
        attr.CopyFrom(value)


class HloShape(metaclass=HloShapeType):

    def __init__(self, scribe, type_enum):
        shape_proto = xla_data_pb2.ShapeProto()
        shape_proto.element_type = type_enum
        pt = xla_data_pb2.PrimitiveType
        if type_enum not in {pt.TUPLE, pt.OPAQUE_TYPE, pt.TOKEN}:
            shape_proto.layout.memory_space = 0
        self.shape_proto = shape_proto

    def __getitem__(self, sizes):
        """
        Handle syntax such as f32[1,1024,1024]
        """
        shape = HloShape(self.scribe, self.shape_proto.element_type)
        shape_proto = shape.shape_proto
        if not is_iterable(sizes):
            sizes = [sizes]
        shape_proto.dimensions[:] = sizes
        shape_proto.is_dynamic_dimension[:] = [False for _ in sizes]
        shape_proto.layout.minor_to_major[:] = reversed(list(range(len(sizes))))
        return shape

    def __call__(self, *tensors):
        """
        Handle syntax such as tuple(bf16[1,1024,1024], bf16[1024])
        """
        shape = HloShape(self.scribe, self.shape_proto.element_type)
        for tensor in tensors:
            shape.shape_proto.tuple_shapes.add().CopyFrom(tensor.shape_proto)
        return shape

    def clone(self):
        shape = HloShape(self.scribe, self.shape_proto.element_type)
        shape.shape_proto.CopyFrom(self.shape_proto)
        return shape

    @property
    def scribe(self):
        return HloScribe.context

    @property
    def dtype(self):
        dtype_name = xla_data_pb2.PrimitiveType.Name(self.shape_proto.element_type).lower()
        return self.scribe.get_dtype(dtype_name)

    @property
    def sizes(self):
        return tuple(self.shape_proto.dimensions)

    def set_alias_to(self, shape, must=False):
        inst = shape.instruction
        parameter_opcode, _ = HloShapeType.opCodeMap['Parameter']
        if inst.opcode != parameter_opcode:
            raise TypeError(f'instruction {inst.name} is not a parameter')
        self.scribe.register_alias(self.instruction.id, inst.parameter_number, must)

    def get_tuple_element(self, tuple_index):
        if self.shape_proto.element_type != xla_data_pb2.PrimitiveType.TUPLE:
            raise TypeError('HloShape.get_tuple_element called on a non-tuple shape')
        element_shape_proto = self.shape_proto.tuple_shapes[tuple_index]
        shape = HloShape(self.scribe, element_shape_proto.element_type)
        shape.shape_proto.CopyFrom(element_shape_proto)
        return shape


def is_iterable(obj):
    try:
        iter(obj)
    except TypeError:
        return False
    return True


def is_tuple(shape_proto):
    return shape_proto.element_type == xla_data_pb2.PrimitiveType.TUPLE
