# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: metaneff.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='metaneff.proto',
  package='metaneff',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0emetaneff.proto\x12\x08metaneff\"\xbe\x02\n\nMetaTensor\x12\x0c\n\x04name\x18\x01 \x01(\x0c\x12\r\n\x05shape\x18\x02 \x03(\x03\x12\x30\n\tdata_type\x18\x03 \x01(\x0e\x32\x1d.metaneff.MetaTensor.DataType\x12\x0f\n\x07\x63ontent\x18\x04 \x01(\x0c\x12 \n\x18\x61llow_dynamic_batch_size\x18\x05 \x01(\x08\"\xad\x01\n\x08\x44\x61taType\x12\r\n\tUNDEFINED\x10\x00\x12\t\n\x05\x46LOAT\x10\x01\x12\t\n\x05INT32\x10\x02\x12\x08\n\x04\x42YTE\x10\x03\x12\n\n\x06STRING\x10\x04\x12\x08\n\x04\x42OOL\x10\x05\x12\t\n\x05UINT8\x10\x06\x12\x08\n\x04INT8\x10\x07\x12\n\n\x06UINT16\x10\x08\x12\t\n\x05INT16\x10\t\x12\t\n\x05INT64\x10\n\x12\x0b\n\x07\x46LOAT16\x10\x0c\x12\n\n\x06\x44OUBLE\x10\r\x12\x0c\n\x08\x42\x46LOAT16\x10\x0e\"r\n\x0bModelConfig\x12\x11\n\tnum_infer\x18\x01 \x01(\x03\x12\x0f\n\x07timeout\x18\x02 \x01(\x03\x12\x18\n\x10optimal_ncg_size\x18\x03 \x01(\x03\x12\x12\n\nasync_load\x18\x04 \x01(\x08\x12\x11\n\tlazy_load\x18\x05 \x01(\x08\"\xba\x02\n\x08MetaNeff\x12+\n\rinput_tensors\x18\x01 \x03(\x0b\x32\x14.metaneff.MetaTensor\x12,\n\x0eoutput_tensors\x18\x02 \x03(\x0b\x32\x14.metaneff.MetaTensor\x12+\n\x0cmodel_config\x18\x03 \x01(\x0b\x32\x15.metaneff.ModelConfig\x12\x1c\n\x14serialized_graph_def\x18\x04 \x01(\x0c\x12\x0c\n\x04name\x18\x05 \x01(\x0c\x12\x42\n\x11output_aliases_to\x18\x06 \x03(\x0b\x32\'.metaneff.MetaNeff.OutputAliasesToEntry\x1a\x36\n\x14OutputAliasesToEntry\x12\x0b\n\x03key\x18\x01 \x01(\x03\x12\r\n\x05value\x18\x02 \x01(\x03:\x02\x38\x01\x62\x06proto3'
)



_METATENSOR_DATATYPE = _descriptor.EnumDescriptor(
  name='DataType',
  full_name='metaneff.MetaTensor.DataType',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNDEFINED', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='FLOAT', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='INT32', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='BYTE', index=3, number=3,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='STRING', index=4, number=4,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='BOOL', index=5, number=5,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='UINT8', index=6, number=6,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='INT8', index=7, number=7,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='UINT16', index=8, number=8,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='INT16', index=9, number=9,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='INT64', index=10, number=10,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='FLOAT16', index=11, number=12,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='DOUBLE', index=12, number=13,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='BFLOAT16', index=13, number=14,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=174,
  serialized_end=347,
)
_sym_db.RegisterEnumDescriptor(_METATENSOR_DATATYPE)


_METATENSOR = _descriptor.Descriptor(
  name='MetaTensor',
  full_name='metaneff.MetaTensor',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='metaneff.MetaTensor.name', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='shape', full_name='metaneff.MetaTensor.shape', index=1,
      number=2, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='data_type', full_name='metaneff.MetaTensor.data_type', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='content', full_name='metaneff.MetaTensor.content', index=3,
      number=4, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='allow_dynamic_batch_size', full_name='metaneff.MetaTensor.allow_dynamic_batch_size', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _METATENSOR_DATATYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=29,
  serialized_end=347,
)


_MODELCONFIG = _descriptor.Descriptor(
  name='ModelConfig',
  full_name='metaneff.ModelConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_infer', full_name='metaneff.ModelConfig.num_infer', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='timeout', full_name='metaneff.ModelConfig.timeout', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='optimal_ncg_size', full_name='metaneff.ModelConfig.optimal_ncg_size', index=2,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='async_load', full_name='metaneff.ModelConfig.async_load', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='lazy_load', full_name='metaneff.ModelConfig.lazy_load', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=349,
  serialized_end=463,
)


_METANEFF_OUTPUTALIASESTOENTRY = _descriptor.Descriptor(
  name='OutputAliasesToEntry',
  full_name='metaneff.MetaNeff.OutputAliasesToEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='metaneff.MetaNeff.OutputAliasesToEntry.key', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='value', full_name='metaneff.MetaNeff.OutputAliasesToEntry.value', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=b'8\001',
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=726,
  serialized_end=780,
)

_METANEFF = _descriptor.Descriptor(
  name='MetaNeff',
  full_name='metaneff.MetaNeff',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='input_tensors', full_name='metaneff.MetaNeff.input_tensors', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='output_tensors', full_name='metaneff.MetaNeff.output_tensors', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='model_config', full_name='metaneff.MetaNeff.model_config', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='serialized_graph_def', full_name='metaneff.MetaNeff.serialized_graph_def', index=3,
      number=4, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='name', full_name='metaneff.MetaNeff.name', index=4,
      number=5, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='output_aliases_to', full_name='metaneff.MetaNeff.output_aliases_to', index=5,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[_METANEFF_OUTPUTALIASESTOENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=466,
  serialized_end=780,
)

_METATENSOR.fields_by_name['data_type'].enum_type = _METATENSOR_DATATYPE
_METATENSOR_DATATYPE.containing_type = _METATENSOR
_METANEFF_OUTPUTALIASESTOENTRY.containing_type = _METANEFF
_METANEFF.fields_by_name['input_tensors'].message_type = _METATENSOR
_METANEFF.fields_by_name['output_tensors'].message_type = _METATENSOR
_METANEFF.fields_by_name['model_config'].message_type = _MODELCONFIG
_METANEFF.fields_by_name['output_aliases_to'].message_type = _METANEFF_OUTPUTALIASESTOENTRY
DESCRIPTOR.message_types_by_name['MetaTensor'] = _METATENSOR
DESCRIPTOR.message_types_by_name['ModelConfig'] = _MODELCONFIG
DESCRIPTOR.message_types_by_name['MetaNeff'] = _METANEFF
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

MetaTensor = _reflection.GeneratedProtocolMessageType('MetaTensor', (_message.Message,), {
  'DESCRIPTOR' : _METATENSOR,
  '__module__' : 'metaneff_pb2'
  # @@protoc_insertion_point(class_scope:metaneff.MetaTensor)
  })
_sym_db.RegisterMessage(MetaTensor)

ModelConfig = _reflection.GeneratedProtocolMessageType('ModelConfig', (_message.Message,), {
  'DESCRIPTOR' : _MODELCONFIG,
  '__module__' : 'metaneff_pb2'
  # @@protoc_insertion_point(class_scope:metaneff.ModelConfig)
  })
_sym_db.RegisterMessage(ModelConfig)

MetaNeff = _reflection.GeneratedProtocolMessageType('MetaNeff', (_message.Message,), {

  'OutputAliasesToEntry' : _reflection.GeneratedProtocolMessageType('OutputAliasesToEntry', (_message.Message,), {
    'DESCRIPTOR' : _METANEFF_OUTPUTALIASESTOENTRY,
    '__module__' : 'metaneff_pb2'
    # @@protoc_insertion_point(class_scope:metaneff.MetaNeff.OutputAliasesToEntry)
    })
  ,
  'DESCRIPTOR' : _METANEFF,
  '__module__' : 'metaneff_pb2'
  # @@protoc_insertion_point(class_scope:metaneff.MetaNeff)
  })
_sym_db.RegisterMessage(MetaNeff)
_sym_db.RegisterMessage(MetaNeff.OutputAliasesToEntry)


_METANEFF_OUTPUTALIASESTOENTRY._options = None
# @@protoc_insertion_point(module_scope)
