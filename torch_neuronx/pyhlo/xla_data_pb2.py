# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: torch_neuronx/pyhlo/xla_data.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\"torch_neuronx/pyhlo/xla_data.proto\x12\x13torch_neuronx.pyhlo\"\xc7\x01\n\rPaddingConfig\x12M\n\ndimensions\x18\x01 \x03(\x0b\x32\x39.torch_neuronx.pyhlo.PaddingConfig.PaddingConfigDimension\x1ag\n\x16PaddingConfigDimension\x12\x18\n\x10\x65\x64ge_padding_low\x18\x01 \x01(\x03\x12\x19\n\x11\x65\x64ge_padding_high\x18\x02 \x01(\x03\x12\x18\n\x10interior_padding\x18\x03 \x01(\x03\"\x1f\n\tTileProto\x12\x12\n\ndimensions\x18\x01 \x03(\x03\"\xae\x04\n\x0bLayoutProto\x12:\n\x0f\x64im_level_types\x18\t \x03(\x0e\x32!.torch_neuronx.pyhlo.DimLevelType\x12\x12\n\ndim_unique\x18\r \x03(\x08\x12\x13\n\x0b\x64im_ordered\x18\x0e \x03(\x08\x12\x16\n\x0eminor_to_major\x18\x01 \x03(\x03\x12-\n\x05tiles\x18\x06 \x03(\x0b\x32\x1e.torch_neuronx.pyhlo.TileProto\x12\x14\n\x0cmemory_space\x18\x08 \x01(\x03\x12@\n\x14index_primitive_type\x18\x0b \x01(\x0e\x32\".torch_neuronx.pyhlo.PrimitiveType\x12\x42\n\x16pointer_primitive_type\x18\x0c \x01(\x0e\x32\".torch_neuronx.pyhlo.PrimitiveType\x12\x37\n\x0ephysical_shape\x18\n \x01(\x0b\x32\x1f.torch_neuronx.pyhlo.ShapeProto\x12+\n#dynamic_shape_metadata_prefix_bytes\x18\x0f \x01(\x03J\x04\x08\x02\x10\x03J\x04\x08\x03\x10\x04J\x04\x08\x04\x10\x05J\x04\x08\x05\x10\x06J\x04\x08\x07\x10\x08R\x11padded_dimensionsR\rpadding_valueR\x06\x66ormatR\x13max_sparse_elementsR\x14\x65lement_size_in_bits\"\xed\x01\n\nShapeProto\x12\x38\n\x0c\x65lement_type\x18\x02 \x01(\x0e\x32\".torch_neuronx.pyhlo.PrimitiveType\x12\x12\n\ndimensions\x18\x03 \x03(\x03\x12\x35\n\x0ctuple_shapes\x18\x04 \x03(\x0b\x32\x1f.torch_neuronx.pyhlo.ShapeProto\x12\x30\n\x06layout\x18\x05 \x01(\x0b\x32 .torch_neuronx.pyhlo.LayoutProto\x12\x1c\n\x14is_dynamic_dimension\x18\x06 \x03(\x08J\x04\x08\x01\x10\x02R\x04rank\"\x92\x01\n\x11ProgramShapeProto\x12\x33\n\nparameters\x18\x01 \x03(\x0b\x32\x1f.torch_neuronx.pyhlo.ShapeProto\x12/\n\x06result\x18\x02 \x01(\x0b\x32\x1f.torch_neuronx.pyhlo.ShapeProto\x12\x17\n\x0fparameter_names\x18\x03 \x03(\t\"D\n\x10\x43omputationStats\x12\x12\n\nflop_count\x18\x01 \x01(\x01\x12\x1c\n\x14transcendental_count\x18\x02 \x01(\x01\"\xea\x04\n\nOpMetadata\x12\x0f\n\x07op_type\x18\x01 \x01(\t\x12\x0f\n\x07op_name\x18\x02 \x01(\t\x12\x13\n\x0bsource_file\x18\x03 \x01(\t\x12\x13\n\x0bsource_line\x18\x04 \x01(\x05\x12:\n\x0cprofile_type\x18\x05 \x03(\x0e\x32 .torch_neuronx.pyhlo.ProfileTypeB\x02\x18\x01\x12\x18\n\x10\x63reation_pass_id\x18\x06 \x01(\x03\x12 \n\x18logical_creation_pass_id\x18\x07 \x01(\x03\x12\'\n\x1fsize_of_generated_code_in_bytes\x18\x08 \x01(\x03\x12+\n#size_of_memory_working_set_in_bytes\x18\t \x01(\x03\x12\x41\n\x0cprofile_info\x18\n \x01(\x0b\x32+.torch_neuronx.pyhlo.OpMetadata.ProfileInfo\x12\x19\n\x11\x64\x65\x64uplicated_name\x18\x0c \x01(\t\x1a\xdd\x01\n\x0bProfileInfo\x12\x36\n\x0cprofile_type\x18\x01 \x03(\x0e\x32 .torch_neuronx.pyhlo.ProfileType\x12\x18\n\x10relative_speedup\x18\x02 \x01(\x01\x12:\n\x0eprofile_source\x18\x03 \x01(\x0e\x32\".torch_neuronx.pyhlo.ProfileSource\x12@\n\x11\x63ompilation_event\x18\x04 \x01(\x0e\x32%.torch_neuronx.pyhlo.CompilationEventJ\x04\x08\x0b\x10\x0c\"\xe3\x01\n\x10\x45xecutionProfile\x12\x1d\n\x15\x63ompilation_cache_hit\x18\x01 \x01(\x08\x12\x17\n\x0f\x63ompile_time_ms\x18\x02 \x01(\x03\x12\x1b\n\x13\x63ompute_cycle_count\x18\x03 \x01(\x03\x12\x17\n\x0f\x63ompute_time_ns\x18\x04 \x01(\x03\x12$\n\x1c\x63ompute_and_transfer_time_ns\x18\x05 \x01(\x03\x12 \n\x18\x65xecutable_size_in_bytes\x18\x06 \x01(\x03\x12\x19\n\x11profile_cache_hit\x18\x07 \x01(\x08\"!\n\x0f\x45xecutionHandle\x12\x0e\n\x06handle\x18\x01 \x01(\x03\"\"\n\x10GlobalDataHandle\x12\x0e\n\x06handle\x18\x01 \x01(\x03\"4\n\x0c\x44\x65viceHandle\x12\x0e\n\x06handle\x18\x01 \x01(\x03\x12\x14\n\x0c\x64\x65vice_count\x18\x02 \x01(\x03\"\xc4\x01\n\rChannelHandle\x12\x0e\n\x06handle\x18\x01 \x01(\x03\x12<\n\x04type\x18\x02 \x01(\x0e\x32..torch_neuronx.pyhlo.ChannelHandle.ChannelType\"e\n\x0b\x43hannelType\x12\x18\n\x14\x43HANNEL_TYPE_INVALID\x10\x00\x12\x14\n\x10\x44\x45VICE_TO_DEVICE\x10\x01\x12\x12\n\x0e\x44\x45VICE_TO_HOST\x10\x02\x12\x12\n\x0eHOST_TO_DEVICE\x10\x03\"\xd5\x01\n\x15\x44\x65viceAssignmentProto\x12\x15\n\rreplica_count\x18\x01 \x01(\x05\x12\x19\n\x11\x63omputation_count\x18\x02 \x01(\x05\x12Y\n\x13\x63omputation_devices\x18\x03 \x03(\x0b\x32<.torch_neuronx.pyhlo.DeviceAssignmentProto.ComputationDevice\x1a/\n\x11\x43omputationDevice\x12\x1a\n\x12replica_device_ids\x18\x01 \x03(\x05\"\xa2\x03\n\x0cLiteralProto\x12.\n\x05shape\x18\x01 \x01(\x0b\x32\x1f.torch_neuronx.pyhlo.ShapeProto\x12\r\n\x05preds\x18\x02 \x03(\x08\x12\x0b\n\x03s4s\x18\x15 \x01(\x0c\x12\x0b\n\x03u4s\x18\x16 \x01(\x0c\x12\x0b\n\x03s8s\x18\x0f \x01(\x0c\x12\x0b\n\x03u8s\x18\x03 \x01(\x0c\x12\x0c\n\x04s32s\x18\x04 \x03(\x05\x12\x0c\n\x04s64s\x18\x05 \x03(\x03\x12\x0c\n\x04u32s\x18\x06 \x03(\r\x12\x0c\n\x04u64s\x18\x07 \x03(\x04\x12\x0c\n\x04\x66\x33\x32s\x18\x08 \x03(\x02\x12\x0c\n\x04\x66\x36\x34s\x18\t \x03(\x01\x12\x0c\n\x04\x63\x36\x34s\x18\x0c \x03(\x02\x12\r\n\x05\x63\x31\x32\x38s\x18\x12 \x03(\x01\x12\x39\n\x0etuple_literals\x18\n \x03(\x0b\x32!.torch_neuronx.pyhlo.LiteralProto\x12\x0c\n\x04\x66\x31\x36s\x18\x0b \x01(\x0c\x12\r\n\x05\x62\x66\x31\x36s\x18\r \x01(\x0c\x12\x0c\n\x04u16s\x18\x10 \x01(\x0c\x12\x0c\n\x04s16s\x18\x11 \x01(\x0c\x12\x0f\n\x07\x66\x38\x65\x35m2s\x18\x13 \x01(\x0c\x12\x11\n\tf8e4m3fns\x18\x14 \x01(\x0c\x12\x16\n\x0esparse_indices\x18\x0e \x03(\x03\"\xa3\x01\n\x0fWindowDimension\x12\x0c\n\x04size\x18\x01 \x01(\x03\x12\x0e\n\x06stride\x18\x02 \x01(\x03\x12\x13\n\x0bpadding_low\x18\x03 \x01(\x03\x12\x14\n\x0cpadding_high\x18\x04 \x01(\x03\x12\x17\n\x0fwindow_dilation\x18\x05 \x01(\x03\x12\x15\n\rbase_dilation\x18\x06 \x01(\x03\x12\x17\n\x0fwindow_reversal\x18\x07 \x01(\x08\"B\n\x06Window\x12\x38\n\ndimensions\x18\x01 \x03(\x0b\x32$.torch_neuronx.pyhlo.WindowDimension\"~\n\x16GatherDimensionNumbers\x12\x13\n\x0boffset_dims\x18\x01 \x03(\x03\x12\x1c\n\x14\x63ollapsed_slice_dims\x18\x02 \x03(\x03\x12\x17\n\x0fstart_index_map\x18\x03 \x03(\x03\x12\x18\n\x10index_vector_dim\x18\x04 \x01(\x03\"\x93\x01\n\x17ScatterDimensionNumbers\x12\x1a\n\x12update_window_dims\x18\x01 \x03(\x03\x12\x1c\n\x14inserted_window_dims\x18\x02 \x03(\x03\x12$\n\x1cscatter_dims_to_operand_dims\x18\x03 \x03(\x03\x12\x18\n\x10index_vector_dim\x18\x04 \x01(\x03\"\xd8\x02\n\x1b\x43onvolutionDimensionNumbers\x12\x1d\n\x15input_batch_dimension\x18\x07 \x01(\x03\x12\x1f\n\x17input_feature_dimension\x18\x08 \x01(\x03\x12 \n\x18input_spatial_dimensions\x18\x0b \x03(\x03\x12&\n\x1ekernel_input_feature_dimension\x18\x03 \x01(\x03\x12\'\n\x1fkernel_output_feature_dimension\x18\x04 \x01(\x03\x12!\n\x19kernel_spatial_dimensions\x18\x06 \x03(\x03\x12\x1e\n\x16output_batch_dimension\x18\t \x01(\x03\x12 \n\x18output_feature_dimension\x18\n \x01(\x03\x12!\n\x19output_spatial_dimensions\x18\x0c \x03(\x03\"\x99\x01\n\x13\x44otDimensionNumbers\x12\"\n\x1alhs_contracting_dimensions\x18\x01 \x03(\x03\x12\"\n\x1arhs_contracting_dimensions\x18\x02 \x03(\x03\x12\x1c\n\x14lhs_batch_dimensions\x18\x03 \x03(\x03\x12\x1c\n\x14rhs_batch_dimensions\x18\x04 \x03(\x03\"\xef\x01\n\x16TriangularSolveOptions\x12\x11\n\tleft_side\x18\x01 \x01(\x08\x12\r\n\x05lower\x18\x02 \x01(\x08\x12\x15\n\runit_diagonal\x18\x03 \x01(\x08\x12J\n\x0btranspose_a\x18\x04 \x01(\x0e\x32\x35.torch_neuronx.pyhlo.TriangularSolveOptions.Transpose\"P\n\tTranspose\x12\x15\n\x11TRANSPOSE_INVALID\x10\x00\x12\x10\n\x0cNO_TRANSPOSE\x10\x01\x12\r\n\tTRANSPOSE\x10\x02\x12\x0b\n\x07\x41\x44JOINT\x10\x03\" \n\x0f\x43holeskyOptions\x12\r\n\x05lower\x18\x01 \x01(\x08\"\x7f\n\x12\x46rontendAttributes\x12=\n\x03map\x18\x01 \x03(\x0b\x32\x30.torch_neuronx.pyhlo.FrontendAttributes.MapEntry\x1a*\n\x08MapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\xd0\x03\n\nOpSharding\x12\x32\n\x04type\x18\x01 \x01(\x0e\x32$.torch_neuronx.pyhlo.OpSharding.Type\x12\x33\n\ntile_shape\x18\x02 \x01(\x0b\x32\x1f.torch_neuronx.pyhlo.ShapeProto\x12\"\n\x1atile_assignment_dimensions\x18\x03 \x03(\x03\x12\x1f\n\x17tile_assignment_devices\x18\x04 \x03(\x03\x12\x38\n\x0ftuple_shardings\x18\x05 \x03(\x0b\x32\x1f.torch_neuronx.pyhlo.OpSharding\x12\"\n\x1areplicate_on_last_tile_dim\x18\x06 \x01(\x08\x12\x31\n\x08metadata\x18\x07 \x03(\x0b\x32\x1f.torch_neuronx.pyhlo.OpMetadata\x12<\n\x0elast_tile_dims\x18\x08 \x03(\x0e\x32$.torch_neuronx.pyhlo.OpSharding.Type\"E\n\x04Type\x12\x0e\n\nREPLICATED\x10\x00\x12\x0b\n\x07MAXIMAL\x10\x01\x12\t\n\x05TUPLE\x10\x02\x12\t\n\x05OTHER\x10\x03\x12\n\n\x06MANUAL\x10\x04\"#\n\x0cReplicaGroup\x12\x13\n\x0breplica_ids\x18\x01 \x03(\x03\".\n\x0cSourceTarget\x12\x0e\n\x06source\x18\x01 \x01(\x03\x12\x0e\n\x06target\x18\x02 \x01(\x03\"\xa0\x01\n\x0fPrecisionConfig\x12I\n\x11operand_precision\x18\x01 \x03(\x0e\x32..torch_neuronx.pyhlo.PrecisionConfig.Precision\"B\n\tPrecision\x12\x0b\n\x07\x44\x45\x46\x41ULT\x10\x00\x12\x08\n\x04HIGH\x10\x01\x12\x0b\n\x07HIGHEST\x10\x02\x12\x11\n\rPACKED_NIBBLE\x10\x03\":\n\x14ParameterReplication\x12\"\n\x1areplicated_at_leaf_buffers\x18\x01 \x03(\x08\"\x8b\x01\n\x16WhileLoopBackendConfig\x12T\n\x10known_trip_count\x18\x01 \x01(\x0b\x32:.torch_neuronx.pyhlo.WhileLoopBackendConfig.KnownTripCount\x1a\x1b\n\x0eKnownTripCount\x12\t\n\x01n\x18\x01 \x01(\x03\"g\n\x15OutputOperandAliasing\x12\x1a\n\x12output_shape_index\x18\x01 \x03(\x03\x12\x15\n\roperand_index\x18\x02 \x01(\x03\x12\x1b\n\x13operand_shape_index\x18\x03 \x03(\x03*\x84\x02\n\rPrimitiveType\x12\x1a\n\x16PRIMITIVE_TYPE_INVALID\x10\x00\x12\x08\n\x04PRED\x10\x01\x12\x06\n\x02S4\x10\x15\x12\x06\n\x02S8\x10\x02\x12\x07\n\x03S16\x10\x03\x12\x07\n\x03S32\x10\x04\x12\x07\n\x03S64\x10\x05\x12\x06\n\x02U4\x10\x16\x12\x06\n\x02U8\x10\x06\x12\x07\n\x03U16\x10\x07\x12\x07\n\x03U32\x10\x08\x12\x07\n\x03U64\x10\t\x12\x07\n\x03\x46\x31\x36\x10\n\x12\x07\n\x03\x46\x33\x32\x10\x0b\x12\x08\n\x04\x42\x46\x31\x36\x10\x10\x12\x07\n\x03\x46\x36\x34\x10\x0c\x12\n\n\x06\x46\x38\x45\x35M2\x10\x13\x12\x0c\n\x08\x46\x38\x45\x34M3FN\x10\x14\x12\x07\n\x03\x43\x36\x34\x10\x0f\x12\x08\n\x04\x43\x31\x32\x38\x10\x12\x12\t\n\x05TUPLE\x10\r\x12\x0f\n\x0bOPAQUE_TYPE\x10\x0e\x12\t\n\x05TOKEN\x10\x11*D\n\x0c\x44imLevelType\x12\r\n\tDIM_DENSE\x10\x00\x12\x12\n\x0e\x44IM_COMPRESSED\x10\x01\x12\x11\n\rDIM_SINGLETON\x10\x02*=\n\x0bProfileType\x12\x0b\n\x07INVALID\x10\x00\x12\n\n\x06WINDOW\x10\x01\x12\x08\n\x04\x46LAG\x10\x02\x12\x0b\n\x07INTEGER\x10\x03*j\n\rProfileSource\x12!\n\x1dPROFILE_SOURCE_UNKNOWN_SOURCE\x10\x00\x12\x1b\n\x17PROFILE_SOURCE_EMBEDDED\x10\x01\x12\x19\n\x15PROFILE_SOURCE_REMOTE\x10\x02*\x85\x01\n\x10\x43ompilationEvent\x12#\n\x1f\x43OMPILATION_EVENT_UNKNOWN_EVENT\x10\x00\x12\'\n#COMPILATION_EVENT_FIRST_COMPILATION\x10\x01\x12#\n\x1f\x43OMPILATION_EVENT_RECOMPILATION\x10\x02*G\n\x0bPaddingType\x12\x13\n\x0fPADDING_INVALID\x10\x00\x12\x11\n\rPADDING_VALID\x10\x01\x12\x10\n\x0cPADDING_SAME\x10\x02*1\n\x07\x46\x66tType\x12\x07\n\x03\x46\x46T\x10\x00\x12\x08\n\x04IFFT\x10\x01\x12\x08\n\x04RFFT\x10\x02\x12\t\n\x05IRFFT\x10\x03*F\n\x12RandomDistribution\x12\x0f\n\x0bRNG_INVALID\x10\x00\x12\x0f\n\x0bRNG_UNIFORM\x10\x01\x12\x0e\n\nRNG_NORMAL\x10\x02*E\n\x0fRandomAlgorithm\x12\x0f\n\x0bRNG_DEFAULT\x10\x00\x12\x11\n\rRNG_THREE_FRY\x10\x01\x12\x0e\n\nRNG_PHILOX\x10\x02\x42\x03\xf8\x01\x01\x62\x06proto3')

_PRIMITIVETYPE = DESCRIPTOR.enum_types_by_name['PrimitiveType']
PrimitiveType = enum_type_wrapper.EnumTypeWrapper(_PRIMITIVETYPE)
_DIMLEVELTYPE = DESCRIPTOR.enum_types_by_name['DimLevelType']
DimLevelType = enum_type_wrapper.EnumTypeWrapper(_DIMLEVELTYPE)
_PROFILETYPE = DESCRIPTOR.enum_types_by_name['ProfileType']
ProfileType = enum_type_wrapper.EnumTypeWrapper(_PROFILETYPE)
_PROFILESOURCE = DESCRIPTOR.enum_types_by_name['ProfileSource']
ProfileSource = enum_type_wrapper.EnumTypeWrapper(_PROFILESOURCE)
_COMPILATIONEVENT = DESCRIPTOR.enum_types_by_name['CompilationEvent']
CompilationEvent = enum_type_wrapper.EnumTypeWrapper(_COMPILATIONEVENT)
_PADDINGTYPE = DESCRIPTOR.enum_types_by_name['PaddingType']
PaddingType = enum_type_wrapper.EnumTypeWrapper(_PADDINGTYPE)
_FFTTYPE = DESCRIPTOR.enum_types_by_name['FftType']
FftType = enum_type_wrapper.EnumTypeWrapper(_FFTTYPE)
_RANDOMDISTRIBUTION = DESCRIPTOR.enum_types_by_name['RandomDistribution']
RandomDistribution = enum_type_wrapper.EnumTypeWrapper(_RANDOMDISTRIBUTION)
_RANDOMALGORITHM = DESCRIPTOR.enum_types_by_name['RandomAlgorithm']
RandomAlgorithm = enum_type_wrapper.EnumTypeWrapper(_RANDOMALGORITHM)
PRIMITIVE_TYPE_INVALID = 0
PRED = 1
S4 = 21
S8 = 2
S16 = 3
S32 = 4
S64 = 5
U4 = 22
U8 = 6
U16 = 7
U32 = 8
U64 = 9
F16 = 10
F32 = 11
BF16 = 16
F64 = 12
F8E5M2 = 19
F8E4M3FN = 20
C64 = 15
C128 = 18
TUPLE = 13
OPAQUE_TYPE = 14
TOKEN = 17
DIM_DENSE = 0
DIM_COMPRESSED = 1
DIM_SINGLETON = 2
INVALID = 0
WINDOW = 1
FLAG = 2
INTEGER = 3
PROFILE_SOURCE_UNKNOWN_SOURCE = 0
PROFILE_SOURCE_EMBEDDED = 1
PROFILE_SOURCE_REMOTE = 2
COMPILATION_EVENT_UNKNOWN_EVENT = 0
COMPILATION_EVENT_FIRST_COMPILATION = 1
COMPILATION_EVENT_RECOMPILATION = 2
PADDING_INVALID = 0
PADDING_VALID = 1
PADDING_SAME = 2
FFT = 0
IFFT = 1
RFFT = 2
IRFFT = 3
RNG_INVALID = 0
RNG_UNIFORM = 1
RNG_NORMAL = 2
RNG_DEFAULT = 0
RNG_THREE_FRY = 1
RNG_PHILOX = 2


_PADDINGCONFIG = DESCRIPTOR.message_types_by_name['PaddingConfig']
_PADDINGCONFIG_PADDINGCONFIGDIMENSION = _PADDINGCONFIG.nested_types_by_name['PaddingConfigDimension']
_TILEPROTO = DESCRIPTOR.message_types_by_name['TileProto']
_LAYOUTPROTO = DESCRIPTOR.message_types_by_name['LayoutProto']
_SHAPEPROTO = DESCRIPTOR.message_types_by_name['ShapeProto']
_PROGRAMSHAPEPROTO = DESCRIPTOR.message_types_by_name['ProgramShapeProto']
_COMPUTATIONSTATS = DESCRIPTOR.message_types_by_name['ComputationStats']
_OPMETADATA = DESCRIPTOR.message_types_by_name['OpMetadata']
_OPMETADATA_PROFILEINFO = _OPMETADATA.nested_types_by_name['ProfileInfo']
_EXECUTIONPROFILE = DESCRIPTOR.message_types_by_name['ExecutionProfile']
_EXECUTIONHANDLE = DESCRIPTOR.message_types_by_name['ExecutionHandle']
_GLOBALDATAHANDLE = DESCRIPTOR.message_types_by_name['GlobalDataHandle']
_DEVICEHANDLE = DESCRIPTOR.message_types_by_name['DeviceHandle']
_CHANNELHANDLE = DESCRIPTOR.message_types_by_name['ChannelHandle']
_DEVICEASSIGNMENTPROTO = DESCRIPTOR.message_types_by_name['DeviceAssignmentProto']
_DEVICEASSIGNMENTPROTO_COMPUTATIONDEVICE = _DEVICEASSIGNMENTPROTO.nested_types_by_name['ComputationDevice']
_LITERALPROTO = DESCRIPTOR.message_types_by_name['LiteralProto']
_WINDOWDIMENSION = DESCRIPTOR.message_types_by_name['WindowDimension']
_WINDOW = DESCRIPTOR.message_types_by_name['Window']
_GATHERDIMENSIONNUMBERS = DESCRIPTOR.message_types_by_name['GatherDimensionNumbers']
_SCATTERDIMENSIONNUMBERS = DESCRIPTOR.message_types_by_name['ScatterDimensionNumbers']
_CONVOLUTIONDIMENSIONNUMBERS = DESCRIPTOR.message_types_by_name['ConvolutionDimensionNumbers']
_DOTDIMENSIONNUMBERS = DESCRIPTOR.message_types_by_name['DotDimensionNumbers']
_TRIANGULARSOLVEOPTIONS = DESCRIPTOR.message_types_by_name['TriangularSolveOptions']
_CHOLESKYOPTIONS = DESCRIPTOR.message_types_by_name['CholeskyOptions']
_FRONTENDATTRIBUTES = DESCRIPTOR.message_types_by_name['FrontendAttributes']
_FRONTENDATTRIBUTES_MAPENTRY = _FRONTENDATTRIBUTES.nested_types_by_name['MapEntry']
_OPSHARDING = DESCRIPTOR.message_types_by_name['OpSharding']
_REPLICAGROUP = DESCRIPTOR.message_types_by_name['ReplicaGroup']
_SOURCETARGET = DESCRIPTOR.message_types_by_name['SourceTarget']
_PRECISIONCONFIG = DESCRIPTOR.message_types_by_name['PrecisionConfig']
_PARAMETERREPLICATION = DESCRIPTOR.message_types_by_name['ParameterReplication']
_WHILELOOPBACKENDCONFIG = DESCRIPTOR.message_types_by_name['WhileLoopBackendConfig']
_WHILELOOPBACKENDCONFIG_KNOWNTRIPCOUNT = _WHILELOOPBACKENDCONFIG.nested_types_by_name['KnownTripCount']
_OUTPUTOPERANDALIASING = DESCRIPTOR.message_types_by_name['OutputOperandAliasing']
_CHANNELHANDLE_CHANNELTYPE = _CHANNELHANDLE.enum_types_by_name['ChannelType']
_TRIANGULARSOLVEOPTIONS_TRANSPOSE = _TRIANGULARSOLVEOPTIONS.enum_types_by_name['Transpose']
_OPSHARDING_TYPE = _OPSHARDING.enum_types_by_name['Type']
_PRECISIONCONFIG_PRECISION = _PRECISIONCONFIG.enum_types_by_name['Precision']
PaddingConfig = _reflection.GeneratedProtocolMessageType('PaddingConfig', (_message.Message,), {

  'PaddingConfigDimension' : _reflection.GeneratedProtocolMessageType('PaddingConfigDimension', (_message.Message,), {
    'DESCRIPTOR' : _PADDINGCONFIG_PADDINGCONFIGDIMENSION,
    '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
    # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.PaddingConfig.PaddingConfigDimension)
    })
  ,
  'DESCRIPTOR' : _PADDINGCONFIG,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.PaddingConfig)
  })
_sym_db.RegisterMessage(PaddingConfig)
_sym_db.RegisterMessage(PaddingConfig.PaddingConfigDimension)

TileProto = _reflection.GeneratedProtocolMessageType('TileProto', (_message.Message,), {
  'DESCRIPTOR' : _TILEPROTO,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.TileProto)
  })
_sym_db.RegisterMessage(TileProto)

LayoutProto = _reflection.GeneratedProtocolMessageType('LayoutProto', (_message.Message,), {
  'DESCRIPTOR' : _LAYOUTPROTO,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.LayoutProto)
  })
_sym_db.RegisterMessage(LayoutProto)

ShapeProto = _reflection.GeneratedProtocolMessageType('ShapeProto', (_message.Message,), {
  'DESCRIPTOR' : _SHAPEPROTO,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.ShapeProto)
  })
_sym_db.RegisterMessage(ShapeProto)

ProgramShapeProto = _reflection.GeneratedProtocolMessageType('ProgramShapeProto', (_message.Message,), {
  'DESCRIPTOR' : _PROGRAMSHAPEPROTO,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.ProgramShapeProto)
  })
_sym_db.RegisterMessage(ProgramShapeProto)

ComputationStats = _reflection.GeneratedProtocolMessageType('ComputationStats', (_message.Message,), {
  'DESCRIPTOR' : _COMPUTATIONSTATS,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.ComputationStats)
  })
_sym_db.RegisterMessage(ComputationStats)

OpMetadata = _reflection.GeneratedProtocolMessageType('OpMetadata', (_message.Message,), {

  'ProfileInfo' : _reflection.GeneratedProtocolMessageType('ProfileInfo', (_message.Message,), {
    'DESCRIPTOR' : _OPMETADATA_PROFILEINFO,
    '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
    # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.OpMetadata.ProfileInfo)
    })
  ,
  'DESCRIPTOR' : _OPMETADATA,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.OpMetadata)
  })
_sym_db.RegisterMessage(OpMetadata)
_sym_db.RegisterMessage(OpMetadata.ProfileInfo)

ExecutionProfile = _reflection.GeneratedProtocolMessageType('ExecutionProfile', (_message.Message,), {
  'DESCRIPTOR' : _EXECUTIONPROFILE,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.ExecutionProfile)
  })
_sym_db.RegisterMessage(ExecutionProfile)

ExecutionHandle = _reflection.GeneratedProtocolMessageType('ExecutionHandle', (_message.Message,), {
  'DESCRIPTOR' : _EXECUTIONHANDLE,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.ExecutionHandle)
  })
_sym_db.RegisterMessage(ExecutionHandle)

GlobalDataHandle = _reflection.GeneratedProtocolMessageType('GlobalDataHandle', (_message.Message,), {
  'DESCRIPTOR' : _GLOBALDATAHANDLE,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.GlobalDataHandle)
  })
_sym_db.RegisterMessage(GlobalDataHandle)

DeviceHandle = _reflection.GeneratedProtocolMessageType('DeviceHandle', (_message.Message,), {
  'DESCRIPTOR' : _DEVICEHANDLE,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.DeviceHandle)
  })
_sym_db.RegisterMessage(DeviceHandle)

ChannelHandle = _reflection.GeneratedProtocolMessageType('ChannelHandle', (_message.Message,), {
  'DESCRIPTOR' : _CHANNELHANDLE,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.ChannelHandle)
  })
_sym_db.RegisterMessage(ChannelHandle)

DeviceAssignmentProto = _reflection.GeneratedProtocolMessageType('DeviceAssignmentProto', (_message.Message,), {

  'ComputationDevice' : _reflection.GeneratedProtocolMessageType('ComputationDevice', (_message.Message,), {
    'DESCRIPTOR' : _DEVICEASSIGNMENTPROTO_COMPUTATIONDEVICE,
    '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
    # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.DeviceAssignmentProto.ComputationDevice)
    })
  ,
  'DESCRIPTOR' : _DEVICEASSIGNMENTPROTO,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.DeviceAssignmentProto)
  })
_sym_db.RegisterMessage(DeviceAssignmentProto)
_sym_db.RegisterMessage(DeviceAssignmentProto.ComputationDevice)

LiteralProto = _reflection.GeneratedProtocolMessageType('LiteralProto', (_message.Message,), {
  'DESCRIPTOR' : _LITERALPROTO,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.LiteralProto)
  })
_sym_db.RegisterMessage(LiteralProto)

WindowDimension = _reflection.GeneratedProtocolMessageType('WindowDimension', (_message.Message,), {
  'DESCRIPTOR' : _WINDOWDIMENSION,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.WindowDimension)
  })
_sym_db.RegisterMessage(WindowDimension)

Window = _reflection.GeneratedProtocolMessageType('Window', (_message.Message,), {
  'DESCRIPTOR' : _WINDOW,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.Window)
  })
_sym_db.RegisterMessage(Window)

GatherDimensionNumbers = _reflection.GeneratedProtocolMessageType('GatherDimensionNumbers', (_message.Message,), {
  'DESCRIPTOR' : _GATHERDIMENSIONNUMBERS,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.GatherDimensionNumbers)
  })
_sym_db.RegisterMessage(GatherDimensionNumbers)

ScatterDimensionNumbers = _reflection.GeneratedProtocolMessageType('ScatterDimensionNumbers', (_message.Message,), {
  'DESCRIPTOR' : _SCATTERDIMENSIONNUMBERS,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.ScatterDimensionNumbers)
  })
_sym_db.RegisterMessage(ScatterDimensionNumbers)

ConvolutionDimensionNumbers = _reflection.GeneratedProtocolMessageType('ConvolutionDimensionNumbers', (_message.Message,), {
  'DESCRIPTOR' : _CONVOLUTIONDIMENSIONNUMBERS,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.ConvolutionDimensionNumbers)
  })
_sym_db.RegisterMessage(ConvolutionDimensionNumbers)

DotDimensionNumbers = _reflection.GeneratedProtocolMessageType('DotDimensionNumbers', (_message.Message,), {
  'DESCRIPTOR' : _DOTDIMENSIONNUMBERS,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.DotDimensionNumbers)
  })
_sym_db.RegisterMessage(DotDimensionNumbers)

TriangularSolveOptions = _reflection.GeneratedProtocolMessageType('TriangularSolveOptions', (_message.Message,), {
  'DESCRIPTOR' : _TRIANGULARSOLVEOPTIONS,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.TriangularSolveOptions)
  })
_sym_db.RegisterMessage(TriangularSolveOptions)

CholeskyOptions = _reflection.GeneratedProtocolMessageType('CholeskyOptions', (_message.Message,), {
  'DESCRIPTOR' : _CHOLESKYOPTIONS,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.CholeskyOptions)
  })
_sym_db.RegisterMessage(CholeskyOptions)

FrontendAttributes = _reflection.GeneratedProtocolMessageType('FrontendAttributes', (_message.Message,), {

  'MapEntry' : _reflection.GeneratedProtocolMessageType('MapEntry', (_message.Message,), {
    'DESCRIPTOR' : _FRONTENDATTRIBUTES_MAPENTRY,
    '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
    # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.FrontendAttributes.MapEntry)
    })
  ,
  'DESCRIPTOR' : _FRONTENDATTRIBUTES,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.FrontendAttributes)
  })
_sym_db.RegisterMessage(FrontendAttributes)
_sym_db.RegisterMessage(FrontendAttributes.MapEntry)

OpSharding = _reflection.GeneratedProtocolMessageType('OpSharding', (_message.Message,), {
  'DESCRIPTOR' : _OPSHARDING,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.OpSharding)
  })
_sym_db.RegisterMessage(OpSharding)

ReplicaGroup = _reflection.GeneratedProtocolMessageType('ReplicaGroup', (_message.Message,), {
  'DESCRIPTOR' : _REPLICAGROUP,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.ReplicaGroup)
  })
_sym_db.RegisterMessage(ReplicaGroup)

SourceTarget = _reflection.GeneratedProtocolMessageType('SourceTarget', (_message.Message,), {
  'DESCRIPTOR' : _SOURCETARGET,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.SourceTarget)
  })
_sym_db.RegisterMessage(SourceTarget)

PrecisionConfig = _reflection.GeneratedProtocolMessageType('PrecisionConfig', (_message.Message,), {
  'DESCRIPTOR' : _PRECISIONCONFIG,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.PrecisionConfig)
  })
_sym_db.RegisterMessage(PrecisionConfig)

ParameterReplication = _reflection.GeneratedProtocolMessageType('ParameterReplication', (_message.Message,), {
  'DESCRIPTOR' : _PARAMETERREPLICATION,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.ParameterReplication)
  })
_sym_db.RegisterMessage(ParameterReplication)

WhileLoopBackendConfig = _reflection.GeneratedProtocolMessageType('WhileLoopBackendConfig', (_message.Message,), {

  'KnownTripCount' : _reflection.GeneratedProtocolMessageType('KnownTripCount', (_message.Message,), {
    'DESCRIPTOR' : _WHILELOOPBACKENDCONFIG_KNOWNTRIPCOUNT,
    '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
    # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.WhileLoopBackendConfig.KnownTripCount)
    })
  ,
  'DESCRIPTOR' : _WHILELOOPBACKENDCONFIG,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.WhileLoopBackendConfig)
  })
_sym_db.RegisterMessage(WhileLoopBackendConfig)
_sym_db.RegisterMessage(WhileLoopBackendConfig.KnownTripCount)

OutputOperandAliasing = _reflection.GeneratedProtocolMessageType('OutputOperandAliasing', (_message.Message,), {
  'DESCRIPTOR' : _OUTPUTOPERANDALIASING,
  '__module__' : 'torch_neuronx.pyhlo.xla_data_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.OutputOperandAliasing)
  })
_sym_db.RegisterMessage(OutputOperandAliasing)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\370\001\001'
  _OPMETADATA.fields_by_name['profile_type']._options = None
  _OPMETADATA.fields_by_name['profile_type']._serialized_options = b'\030\001'
  _FRONTENDATTRIBUTES_MAPENTRY._options = None
  _FRONTENDATTRIBUTES_MAPENTRY._serialized_options = b'8\001'
  _PRIMITIVETYPE._serialized_start=5569
  _PRIMITIVETYPE._serialized_end=5829
  _DIMLEVELTYPE._serialized_start=5831
  _DIMLEVELTYPE._serialized_end=5899
  _PROFILETYPE._serialized_start=5901
  _PROFILETYPE._serialized_end=5962
  _PROFILESOURCE._serialized_start=5964
  _PROFILESOURCE._serialized_end=6070
  _COMPILATIONEVENT._serialized_start=6073
  _COMPILATIONEVENT._serialized_end=6206
  _PADDINGTYPE._serialized_start=6208
  _PADDINGTYPE._serialized_end=6279
  _FFTTYPE._serialized_start=6281
  _FFTTYPE._serialized_end=6330
  _RANDOMDISTRIBUTION._serialized_start=6332
  _RANDOMDISTRIBUTION._serialized_end=6402
  _RANDOMALGORITHM._serialized_start=6404
  _RANDOMALGORITHM._serialized_end=6473
  _PADDINGCONFIG._serialized_start=60
  _PADDINGCONFIG._serialized_end=259
  _PADDINGCONFIG_PADDINGCONFIGDIMENSION._serialized_start=156
  _PADDINGCONFIG_PADDINGCONFIGDIMENSION._serialized_end=259
  _TILEPROTO._serialized_start=261
  _TILEPROTO._serialized_end=292
  _LAYOUTPROTO._serialized_start=295
  _LAYOUTPROTO._serialized_end=853
  _SHAPEPROTO._serialized_start=856
  _SHAPEPROTO._serialized_end=1093
  _PROGRAMSHAPEPROTO._serialized_start=1096
  _PROGRAMSHAPEPROTO._serialized_end=1242
  _COMPUTATIONSTATS._serialized_start=1244
  _COMPUTATIONSTATS._serialized_end=1312
  _OPMETADATA._serialized_start=1315
  _OPMETADATA._serialized_end=1933
  _OPMETADATA_PROFILEINFO._serialized_start=1706
  _OPMETADATA_PROFILEINFO._serialized_end=1927
  _EXECUTIONPROFILE._serialized_start=1936
  _EXECUTIONPROFILE._serialized_end=2163
  _EXECUTIONHANDLE._serialized_start=2165
  _EXECUTIONHANDLE._serialized_end=2198
  _GLOBALDATAHANDLE._serialized_start=2200
  _GLOBALDATAHANDLE._serialized_end=2234
  _DEVICEHANDLE._serialized_start=2236
  _DEVICEHANDLE._serialized_end=2288
  _CHANNELHANDLE._serialized_start=2291
  _CHANNELHANDLE._serialized_end=2487
  _CHANNELHANDLE_CHANNELTYPE._serialized_start=2386
  _CHANNELHANDLE_CHANNELTYPE._serialized_end=2487
  _DEVICEASSIGNMENTPROTO._serialized_start=2490
  _DEVICEASSIGNMENTPROTO._serialized_end=2703
  _DEVICEASSIGNMENTPROTO_COMPUTATIONDEVICE._serialized_start=2656
  _DEVICEASSIGNMENTPROTO_COMPUTATIONDEVICE._serialized_end=2703
  _LITERALPROTO._serialized_start=2706
  _LITERALPROTO._serialized_end=3124
  _WINDOWDIMENSION._serialized_start=3127
  _WINDOWDIMENSION._serialized_end=3290
  _WINDOW._serialized_start=3292
  _WINDOW._serialized_end=3358
  _GATHERDIMENSIONNUMBERS._serialized_start=3360
  _GATHERDIMENSIONNUMBERS._serialized_end=3486
  _SCATTERDIMENSIONNUMBERS._serialized_start=3489
  _SCATTERDIMENSIONNUMBERS._serialized_end=3636
  _CONVOLUTIONDIMENSIONNUMBERS._serialized_start=3639
  _CONVOLUTIONDIMENSIONNUMBERS._serialized_end=3983
  _DOTDIMENSIONNUMBERS._serialized_start=3986
  _DOTDIMENSIONNUMBERS._serialized_end=4139
  _TRIANGULARSOLVEOPTIONS._serialized_start=4142
  _TRIANGULARSOLVEOPTIONS._serialized_end=4381
  _TRIANGULARSOLVEOPTIONS_TRANSPOSE._serialized_start=4301
  _TRIANGULARSOLVEOPTIONS_TRANSPOSE._serialized_end=4381
  _CHOLESKYOPTIONS._serialized_start=4383
  _CHOLESKYOPTIONS._serialized_end=4415
  _FRONTENDATTRIBUTES._serialized_start=4417
  _FRONTENDATTRIBUTES._serialized_end=4544
  _FRONTENDATTRIBUTES_MAPENTRY._serialized_start=4502
  _FRONTENDATTRIBUTES_MAPENTRY._serialized_end=4544
  _OPSHARDING._serialized_start=4547
  _OPSHARDING._serialized_end=5011
  _OPSHARDING_TYPE._serialized_start=4942
  _OPSHARDING_TYPE._serialized_end=5011
  _REPLICAGROUP._serialized_start=5013
  _REPLICAGROUP._serialized_end=5048
  _SOURCETARGET._serialized_start=5050
  _SOURCETARGET._serialized_end=5096
  _PRECISIONCONFIG._serialized_start=5099
  _PRECISIONCONFIG._serialized_end=5259
  _PRECISIONCONFIG_PRECISION._serialized_start=5193
  _PRECISIONCONFIG_PRECISION._serialized_end=5259
  _PARAMETERREPLICATION._serialized_start=5261
  _PARAMETERREPLICATION._serialized_end=5319
  _WHILELOOPBACKENDCONFIG._serialized_start=5322
  _WHILELOOPBACKENDCONFIG._serialized_end=5461
  _WHILELOOPBACKENDCONFIG_KNOWNTRIPCOUNT._serialized_start=5434
  _WHILELOOPBACKENDCONFIG_KNOWNTRIPCOUNT._serialized_end=5461
  _OUTPUTOPERANDALIASING._serialized_start=5463
  _OUTPUTOPERANDALIASING._serialized_end=5566
# @@protoc_insertion_point(module_scope)
