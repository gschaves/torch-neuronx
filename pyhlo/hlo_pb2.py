# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: torch_neuronx/pyhlo/hlo.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from torch_neuronx.pyhlo import xla_data_pb2 as torch__neuronx_dot_pyhlo_dot_xla__data__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1dtorch_neuronx/pyhlo/hlo.proto\x12\x13torch_neuronx.pyhlo\x1a\"torch_neuronx/pyhlo/xla_data.proto\"\xa4\x19\n\x13HloInstructionProto\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0e\n\x06opcode\x18\x02 \x01(\t\x12.\n\x05shape\x18\x03 \x01(\x0b\x32\x1f.torch_neuronx.pyhlo.ShapeProto\x12\x31\n\x08metadata\x18\x07 \x01(\x0b\x32\x1f.torch_neuronx.pyhlo.OpMetadata\x12\x32\n\x07literal\x18\x08 \x01(\x0b\x32!.torch_neuronx.pyhlo.LiteralProto\x12\x18\n\x10parameter_number\x18\t \x01(\x03\x12\x13\n\x0b\x66usion_kind\x18\x0b \x01(\t\x12\x13\n\x0btuple_index\x18\r \x01(\x03\x12\x12\n\ndimensions\x18\x0e \x03(\x03\x12+\n\x06window\x18\x0f \x01(\x0b\x32\x1b.torch_neuronx.pyhlo.Window\x12W\n\x1d\x63onvolution_dimension_numbers\x18\x10 \x01(\x0b\x32\x30.torch_neuronx.pyhlo.ConvolutionDimensionNumbers\x12\x1b\n\x13\x66\x65\x61ture_group_count\x18\x32 \x01(\x03\x12\x19\n\x11\x62\x61tch_group_count\x18: \x01(\x03\x12R\n\x10slice_dimensions\x18\x11 \x03(\x0b\x32\x38.torch_neuronx.pyhlo.HloInstructionProto.SliceDimensions\x12\x15\n\rexponent_bits\x18\x12 \x01(\x05\x12\x15\n\rmantissa_bits\x18\x13 \x01(\x05\x12\x1b\n\x13\x64ynamic_slice_sizes\x18\x14 \x03(\x03\x12:\n\x0epadding_config\x18\x15 \x01(\x0b\x32\".torch_neuronx.pyhlo.PaddingConfig\x12\x16\n\x0eoutfeed_config\x18\x16 \x01(\x0c\x12=\n\x0c\x64istribution\x18\x17 \x01(\x0e\x32\'.torch_neuronx.pyhlo.RandomDistribution\x12\x0f\n\x07\x65psilon\x18\x18 \x01(\x02\x12\x15\n\rfeature_index\x18\x19 \x01(\x03\x12\x12\n\nchannel_id\x18\x1a \x01(\x03\x12\x15\n\rinfeed_config\x18\x1b \x01(\x0c\x12\x1a\n\x12\x63ustom_call_target\x18\x1c \x01(\t\x12\x36\n\routfeed_shape\x18\x1d \x01(\x0b\x32\x1f.torch_neuronx.pyhlo.ShapeProto\x12G\n\x15\x64ot_dimension_numbers\x18\x1e \x01(\x0b\x32(.torch_neuronx.pyhlo.DotDimensionNumbers\x12.\n\x08\x66\x66t_type\x18\x1f \x01(\x0e\x32\x1c.torch_neuronx.pyhlo.FftType\x12\x12\n\nfft_length\x18  \x03(\x03\x12\x1c\n\x14\x63omparison_direction\x18? \x01(\t\x12M\n\x18gather_dimension_numbers\x18! \x01(\x0b\x32+.torch_neuronx.pyhlo.GatherDimensionNumbers\x12\x1a\n\x12gather_slice_sizes\x18\" \x03(\x03\x12\n\n\x02id\x18# \x01(\x03\x12\x13\n\x0boperand_ids\x18$ \x03(\x03\x12\x1f\n\x17\x63ontrol_predecessor_ids\x18% \x03(\x03\x12\x1e\n\x16\x63\x61lled_computation_ids\x18& \x03(\x03\x12\x31\n\x08sharding\x18( \x01(\x0b\x32\x1f.torch_neuronx.pyhlo.OpSharding\x12\x16\n\x0e\x62\x61\x63kend_config\x18+ \x01(\x0c\x12\x39\n\x0ereplica_groups\x18\x31 \x03(\x0b\x32!.torch_neuronx.pyhlo.ReplicaGroup\x12\x19\n\rall_reduce_id\x18- \x01(\x03\x42\x02\x18\x01\x12\x1d\n\x15use_global_device_ids\x18G \x01(\x08\x12\x18\n\x10is_host_transfer\x18/ \x01(\x08\x12\x11\n\tis_stable\x18< \x01(\x08\x12O\n\x19scatter_dimension_numbers\x18\x30 \x01(\x0b\x32,.torch_neuronx.pyhlo.ScatterDimensionNumbers\x12>\n\x10precision_config\x18\x33 \x01(\x0b\x32$.torch_neuronx.pyhlo.PrecisionConfig\x12>\n\x13source_target_pairs\x18\x34 \x03(\x0b\x32!.torch_neuronx.pyhlo.SourceTarget\x12>\n\x15\x64omain_entry_sharding\x18\x36 \x01(\x0b\x32\x1f.torch_neuronx.pyhlo.OpSharding\x12=\n\x14\x64omain_exit_sharding\x18\x37 \x01(\x0b\x32\x1f.torch_neuronx.pyhlo.OpSharding\x12\x18\n\x10\x63onstrain_layout\x18\x38 \x01(\x08\x12\x43\n\x1aoperand_shapes_with_layout\x18\x39 \x03(\x0b\x32\x1f.torch_neuronx.pyhlo.ShapeProto\x12M\n\x18triangular_solve_options\x18; \x01(\x0b\x32+.torch_neuronx.pyhlo.TriangularSolveOptions\x12>\n\x10\x63holesky_options\x18> \x01(\x0b\x32$.torch_neuronx.pyhlo.CholeskyOptions\x12H\n\x15parameter_replication\x18= \x01(\x0b\x32).torch_neuronx.pyhlo.ParameterReplication\x12#\n\x1b\x63ustom_call_has_side_effect\x18\x41 \x01(\x08\x12K\n\x17output_operand_aliasing\x18J \x03(\x0b\x32*.torch_neuronx.pyhlo.OutputOperandAliasing\x12\x45\n\x14\x63ustom_call_schedule\x18L \x01(\x0e\x32\'.torch_neuronx.pyhlo.CustomCallSchedule\x12\r\n\x05\x64\x65lta\x18\x42 \x01(\x03\x12\x1a\n\x12indices_are_sorted\x18\x43 \x01(\x08\x12\x44\n\x13\x66rontend_attributes\x18\x44 \x01(\x0b\x32\'.torch_neuronx.pyhlo.FrontendAttributes\x12\x16\n\x0eunique_indices\x18\x45 \x01(\x08\x12;\n\rrng_algorithm\x18\x46 \x01(\x0e\x32$.torch_neuronx.pyhlo.RandomAlgorithm\x12\x17\n\x0f\x63omparison_type\x18H \x01(\t\x12%\n\x19is_cross_program_prefetch\x18I \x01(\x08\x42\x02\x18\x01\x12&\n\x1c\x63ross_program_prefetch_index\x18P \x01(\x05H\x00\x12\x36\n\x0cpadding_type\x18K \x01(\x0e\x32 .torch_neuronx.pyhlo.PaddingType\x12J\n\x17\x63ustom_call_api_version\x18M \x01(\x0e\x32).torch_neuronx.pyhlo.CustomCallApiVersion\x12\x16\n\x0e\x61sync_group_id\x18N \x01(\x03\x12\x1e\n\x16\x61sync_execution_thread\x18O \x01(\t\x1a?\n\x0fSliceDimensions\x12\r\n\x05start\x18\x01 \x01(\x03\x12\r\n\x05limit\x18\x02 \x01(\x03\x12\x0e\n\x06stride\x18\x03 \x01(\x03\x42\'\n%optional_cross_program_prefetch_indexJ\x04\x08\n\x10\x0bJ\x04\x08\x0c\x10\rJ\x04\x08\x04\x10\x05J\x04\x08\x05\x10\x06J\x04\x08\x06\x10\x07J\x04\x08,\x10-J\x04\x08\x35\x10\x36J\x04\x08.\x10/J\x04\x08)\x10*J\x04\x08*\x10+J\x04\x08@\x10\x41R\x0eparameter_nameR\x1e\x66used_instructions_computationR\roperand_namesR\x19\x63ontrol_predecessor_namesR\x18\x63\x61lled_computation_namesR\x11replica_group_idsR\x12\x63ustom_call_opaqueR\x12\x61ll_reduce_barrier\"\x89\x02\n\x13HloComputationProto\x12\x0c\n\x04name\x18\x01 \x01(\t\x12>\n\x0cinstructions\x18\x02 \x03(\x0b\x32(.torch_neuronx.pyhlo.HloInstructionProto\x12=\n\rprogram_shape\x18\x04 \x01(\x0b\x32&.torch_neuronx.pyhlo.ProgramShapeProto\x12\n\n\x02id\x18\x05 \x01(\x03\x12\x0f\n\x07root_id\x18\x06 \x01(\x03\x12\x1d\n\x15is_fusion_computation\x18\x07 \x01(\x08\x12\x18\n\x10\x65xecution_thread\x18\x08 \x01(\tJ\x04\x08\x03\x10\x04R\troot_name\"\xf8\x01\n\x10HloScheduleProto\x12G\n\tsequences\x18\x01 \x03(\x0b\x32\x34.torch_neuronx.pyhlo.HloScheduleProto.SequencesEntry\x1a.\n\x13InstructionSequence\x12\x17\n\x0finstruction_ids\x18\x01 \x03(\x03\x1ak\n\x0eSequencesEntry\x12\x0b\n\x03key\x18\x01 \x01(\x03\x12H\n\x05value\x18\x02 \x01(\x0b\x32\x39.torch_neuronx.pyhlo.HloScheduleProto.InstructionSequence:\x02\x38\x01\"\xfc\x01\n\x18HloInputOutputAliasProto\x12N\n\x07\x65ntries\x18\x01 \x03(\x0b\x32=.torch_neuronx.pyhlo.HloInputOutputAliasProto.AliasEntryProto\x1a\x8f\x01\n\x0f\x41liasEntryProto\x12\x1a\n\x12output_shape_index\x18\x01 \x03(\x03\x12\x18\n\x10parameter_number\x18\x02 \x01(\x03\x12\x1d\n\x15parameter_shape_index\x18\x03 \x03(\x03\x12\'\n\x04kind\x18\x04 \x01(\x0e\x32\x19.torch_neuronx.pyhlo.Kind\"\x82\x02\n\x1c\x44ynamicParameterBindingProto\x12J\n\x07\x65ntries\x18\x01 \x03(\x0b\x32\x39.torch_neuronx.pyhlo.DynamicParameterBindingProto.Binding\x1a\x95\x01\n\x07\x42inding\x12\x19\n\x11\x64ynamic_param_num\x18\x01 \x01(\x03\x12\x1b\n\x13\x64ynamic_param_index\x18\x02 \x03(\x03\x12\x18\n\x10target_param_num\x18\x03 \x01(\x03\x12\x1a\n\x12target_param_index\x18\x04 \x03(\x03\x12\x1c\n\x14target_param_dim_num\x18\x05 \x01(\x03\"H\n\x14\x43rossProgramPrefetch\x12\x11\n\tparameter\x18\x01 \x01(\x03\x12\r\n\x05index\x18\x02 \x03(\x03\x12\x0e\n\x06offset\x18\x03 \x01(\x03\"\x92\t\n\x0eHloModuleProto\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x1e\n\x16\x65ntry_computation_name\x18\x02 \x01(\t\x12\x1c\n\x14\x65ntry_computation_id\x18\x06 \x01(\x03\x12>\n\x0c\x63omputations\x18\x03 \x03(\x0b\x32(.torch_neuronx.pyhlo.HloComputationProto\x12\x42\n\x12host_program_shape\x18\x04 \x01(\x0b\x32&.torch_neuronx.pyhlo.ProgramShapeProto\x12\n\n\x02id\x18\x05 \x01(\x03\x12\x37\n\x08schedule\x18\x07 \x01(\x0b\x32%.torch_neuronx.pyhlo.HloScheduleProto\x12I\n\x12input_output_alias\x18\x08 \x01(\x0b\x32-.torch_neuronx.pyhlo.HloInputOutputAliasProto\x12T\n\x19\x64ynamic_parameter_binding\x18\t \x01(\x0b\x32\x31.torch_neuronx.pyhlo.DynamicParameterBindingProto\x12K\n\x18\x63ross_program_prefetches\x18\n \x03(\x0b\x32).torch_neuronx.pyhlo.CrossProgramPrefetch\x12\x12\n\nis_dynamic\x18\x0b \x01(\x08\x12=\n\x14spmd_output_sharding\x18\x0c \x01(\x0b\x32\x1f.torch_neuronx.pyhlo.OpSharding\x12\x42\n\x19spmd_parameters_shardings\x18\x0e \x03(\x0b\x32\x1f.torch_neuronx.pyhlo.OpSharding\x12\"\n\x1ause_auto_spmd_partitioning\x18\x10 \x01(\x08\x12\x45\n\x0cprofile_info\x18\r \x03(\x0b\x32/.torch_neuronx.pyhlo.HloModuleProto.ProfileInfo\x12\x45\n\x11\x64\x65vice_assignment\x18\x0f \x01(\x0b\x32*.torch_neuronx.pyhlo.DeviceAssignmentProto\x1a\xec\x01\n\x0bProfileInfo\x12\x45\n\x0cprofile_type\x18\x01 \x01(\x0e\x32/.torch_neuronx.pyhlo.HloModuleProto.ProfileType\x12\x18\n\x10relative_speedup\x18\x02 \x01(\x01\x12:\n\x0eprofile_source\x18\x03 \x01(\x0e\x32\".torch_neuronx.pyhlo.ProfileSource\x12@\n\x11\x63ompilation_event\x18\x04 \x01(\x0e\x32%.torch_neuronx.pyhlo.CompilationEvent\"E\n\x0bProfileType\x12\x0b\n\x07INVALID\x10\x00\x12\x08\n\x04\x46LAG\x10\x01\x12\n\n\x06\x46USION\x10\x02\x12\n\n\x06LAYOUT\x10\x03\x12\x07\n\x03\x44OT\x10\x04\"\xe0\x01\n\x12LogicalBufferProto\x12\n\n\x02id\x18\x01 \x01(\x03\x12\x0c\n\x04size\x18\x02 \x01(\x03\x12\x44\n\ndefined_at\x18\x03 \x01(\x0b\x32\x30.torch_neuronx.pyhlo.LogicalBufferProto.Location\x12\r\n\x05\x63olor\x18\x04 \x01(\x03\x1a[\n\x08Location\x12\x1c\n\x10instruction_name\x18\x02 \x01(\tB\x02\x18\x01\x12\x16\n\x0einstruction_id\x18\x04 \x01(\x03\x12\x13\n\x0bshape_index\x18\x03 \x03(\x03J\x04\x08\x01\x10\x02\"\x88\x03\n\x15\x42ufferAllocationProto\x12\r\n\x05index\x18\x01 \x01(\x03\x12\x0c\n\x04size\x18\x02 \x01(\x03\x12\x17\n\x0fis_thread_local\x18\x03 \x01(\x08\x12\x10\n\x08is_tuple\x18\x0b \x01(\x08\x12&\n\x1eis_entry_computation_parameter\x18\x05 \x01(\x08\x12\x13\n\x0bis_constant\x18\x0c \x01(\x08\x12\x18\n\x10parameter_number\x18\x06 \x01(\x03\x12\x1d\n\x15parameter_shape_index\x18\n \x03(\x03\x12\x16\n\x0emaybe_live_out\x18\x07 \x01(\x08\x12\r\n\x05\x63olor\x18\x08 \x01(\x03\x12\x45\n\x08\x61ssigned\x18\t \x03(\x0b\x32\x33.torch_neuronx.pyhlo.BufferAllocationProto.Assigned\x1a\x43\n\x08\x41ssigned\x12\x19\n\x11logical_buffer_id\x18\x01 \x01(\x03\x12\x0e\n\x06offset\x18\x02 \x01(\x03\x12\x0c\n\x04size\x18\x03 \x01(\x03\"\xf6\x02\n\x12HeapSimulatorTrace\x12=\n\x06\x65vents\x18\x01 \x03(\x0b\x32-.torch_neuronx.pyhlo.HeapSimulatorTrace.Event\x12\x1f\n\x17whole_module_simulation\x18\x02 \x01(\x08\x12\x1f\n\x17\x62uffer_allocation_index\x18\x03 \x01(\x03\x1a\xde\x01\n\x05\x45vent\x12@\n\x04kind\x18\x01 \x01(\x0e\x32\x32.torch_neuronx.pyhlo.HeapSimulatorTrace.Event.Kind\x12\x11\n\tbuffer_id\x18\x02 \x01(\x03\x12\x18\n\x10\x63omputation_name\x18\x03 \x01(\t\x12\x18\n\x10instruction_name\x18\x04 \x01(\t\x12\x1f\n\x17share_with_canonical_id\x18\x05 \x01(\x03\"+\n\x04Kind\x12\t\n\x05\x41LLOC\x10\x00\x12\x08\n\x04\x46REE\x10\x01\x12\x0e\n\nSHARE_WITH\x10\x02\"]\n\x13HloModuleGroupProto\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x38\n\x0bhlo_modules\x18\x02 \x03(\x0b\x32#.torch_neuronx.pyhlo.HloModuleProto\"\xa6\x03\n\x15\x42ufferAssignmentProto\x12@\n\x0flogical_buffers\x18\x01 \x03(\x0b\x32\'.torch_neuronx.pyhlo.LogicalBufferProto\x12N\n\x0e\x62uffer_aliases\x18\x02 \x03(\x0b\x32\x36.torch_neuronx.pyhlo.BufferAssignmentProto.BufferAlias\x12\x46\n\x12\x62uffer_allocations\x18\x03 \x03(\x0b\x32*.torch_neuronx.pyhlo.BufferAllocationProto\x12\x46\n\x15heap_simulator_traces\x18\x04 \x03(\x0b\x32\'.torch_neuronx.pyhlo.HeapSimulatorTrace\x1ak\n\x0b\x42ufferAlias\x12\x18\n\x10source_buffer_id\x18\x01 \x01(\x03\x12\x42\n\x08location\x18\x02 \x01(\x0b\x32\x30.torch_neuronx.pyhlo.LogicalBufferProto.Location\"\x9e\x01\n\x08HloProto\x12\x37\n\nhlo_module\x18\x01 \x01(\x0b\x32#.torch_neuronx.pyhlo.HloModuleProto\x12\x45\n\x11\x62uffer_assignment\x18\x03 \x01(\x0b\x32*.torch_neuronx.pyhlo.BufferAssignmentProtoJ\x04\x08\x02\x10\x03R\x0chlo_ordering\"\xbe\x01\n\x0bHloSnapshot\x12*\n\x03hlo\x18\x01 \x01(\x0b\x32\x1d.torch_neuronx.pyhlo.HloProto\x12\x34\n\targuments\x18\x02 \x03(\x0b\x32!.torch_neuronx.pyhlo.LiteralProto\x12\x31\n\x06result\x18\x03 \x01(\x0b\x32!.torch_neuronx.pyhlo.LiteralProto\x12\x1a\n\x12\x65xecution_platform\x18\x04 \x01(\t\"\xc9\x01\n\x16HloModuleMetadataProto\x12\x1b\n\x13\x63\x61nonical_module_id\x18\x01 \x01(\x03\x12\x19\n\x11module_group_name\x18\x02 \x01(\t\x12\x1a\n\x12original_module_id\x18\x03 \x01(\x03\x12\x1e\n\x16partitioned_module_ids\x18\x04 \x03(\x03\x12;\n\rpass_metadata\x18\x05 \x03(\x0b\x32$.torch_neuronx.pyhlo.HloPassMetadata\"\xea\x01\n\x0fHloPassMetadata\x12\x0f\n\x07pass_id\x18\x01 \x01(\x03\x12\x11\n\tpass_name\x18\x02 \x01(\t\x12\x15\n\rpipeline_name\x18\x03 \x01(\t\x12\x16\n\x0e\x64ump_filenames\x18\x04 \x03(\t\x12\x16\n\x0emodule_changed\x18\x05 \x01(\x08\x12\x11\n\tmodule_id\x18\x06 \x01(\x03\x12\x1f\n\x17module_group_module_ids\x18\x07 \x03(\x03\x12\x1c\n\x14start_timestamp_usec\x18\x08 \x01(\x03\x12\x1a\n\x12\x65nd_timestamp_usec\x18\t \x01(\x03\"\xe3\x03\n\x17\x45ntryFunctionAttributes\x12W\n\x07\x62uffers\x18\x01 \x03(\x0b\x32\x46.torch_neuronx.pyhlo.EntryFunctionAttributes.BufferParameterAttributes\x12\x18\n\x10result_xla_shape\x18\x02 \x01(\t\x1a\x1d\n\nShapeIndex\x12\x0f\n\x07indices\x18\x01 \x03(\x03\x1a\xb5\x02\n\x19\x42ufferParameterAttributes\x12\x14\n\x0clmhlo_params\x18\x01 \x01(\x03\x12\x1c\n\x14lmhlo_params_present\x18\x06 \x01(\x08\x12X\n\x17lmhlo_param_shape_index\x18\x02 \x01(\x0b\x32\x37.torch_neuronx.pyhlo.EntryFunctionAttributes.ShapeIndex\x12\x1b\n\x13lmhlo_constant_name\x18\x03 \x01(\t\x12\x18\n\x10lmhlo_must_alias\x18\x04 \x01(\x08\x12S\n\x12lmhlo_output_index\x18\x05 \x01(\x0b\x32\x37.torch_neuronx.pyhlo.EntryFunctionAttributes.ShapeIndex\"\x81\x01\n\x19XlaRuntimeExecutableProto\x12=\n\x10hlo_module_proto\x18\x01 \x01(\x0b\x32#.torch_neuronx.pyhlo.HloModuleProto\x12\x10\n\x08obj_file\x18\x03 \x01(\x0c\x12\x13\n\x0bmlir_module\x18\x04 \x01(\t*S\n\x12\x43ustomCallSchedule\x12\x11\n\rSCHEDULE_NONE\x10\x00\x12\x13\n\x0fSCHEDULE_LATEST\x10\x01\x12\x15\n\x11SCHEDULE_EARLIEST\x10\x02*\xb4\x01\n\x14\x43ustomCallApiVersion\x12\x1b\n\x17\x41PI_VERSION_UNSPECIFIED\x10\x00\x12\x18\n\x14\x41PI_VERSION_ORIGINAL\x10\x01\x12 \n\x1c\x41PI_VERSION_STATUS_RETURNING\x10\x02\x12(\n$API_VERSION_STATUS_RETURNING_UNIFIED\x10\x03\x12\x19\n\x15\x41PI_VERSION_TYPED_FFI\x10\x04*:\n\x04Kind\x12\x13\n\x0fUNDEFINED_ALIAS\x10\x00\x12\r\n\tMAY_ALIAS\x10\x01\x12\x0e\n\nMUST_ALIAS\x10\x02\x42\x03\xf8\x01\x01\x62\x06proto3')

_CUSTOMCALLSCHEDULE = DESCRIPTOR.enum_types_by_name['CustomCallSchedule']
CustomCallSchedule = enum_type_wrapper.EnumTypeWrapper(_CUSTOMCALLSCHEDULE)
_CUSTOMCALLAPIVERSION = DESCRIPTOR.enum_types_by_name['CustomCallApiVersion']
CustomCallApiVersion = enum_type_wrapper.EnumTypeWrapper(_CUSTOMCALLAPIVERSION)
_KIND = DESCRIPTOR.enum_types_by_name['Kind']
Kind = enum_type_wrapper.EnumTypeWrapper(_KIND)
SCHEDULE_NONE = 0
SCHEDULE_LATEST = 1
SCHEDULE_EARLIEST = 2
API_VERSION_UNSPECIFIED = 0
API_VERSION_ORIGINAL = 1
API_VERSION_STATUS_RETURNING = 2
API_VERSION_STATUS_RETURNING_UNIFIED = 3
API_VERSION_TYPED_FFI = 4
UNDEFINED_ALIAS = 0
MAY_ALIAS = 1
MUST_ALIAS = 2


_HLOINSTRUCTIONPROTO = DESCRIPTOR.message_types_by_name['HloInstructionProto']
_HLOINSTRUCTIONPROTO_SLICEDIMENSIONS = _HLOINSTRUCTIONPROTO.nested_types_by_name['SliceDimensions']
_HLOCOMPUTATIONPROTO = DESCRIPTOR.message_types_by_name['HloComputationProto']
_HLOSCHEDULEPROTO = DESCRIPTOR.message_types_by_name['HloScheduleProto']
_HLOSCHEDULEPROTO_INSTRUCTIONSEQUENCE = _HLOSCHEDULEPROTO.nested_types_by_name['InstructionSequence']
_HLOSCHEDULEPROTO_SEQUENCESENTRY = _HLOSCHEDULEPROTO.nested_types_by_name['SequencesEntry']
_HLOINPUTOUTPUTALIASPROTO = DESCRIPTOR.message_types_by_name['HloInputOutputAliasProto']
_HLOINPUTOUTPUTALIASPROTO_ALIASENTRYPROTO = _HLOINPUTOUTPUTALIASPROTO.nested_types_by_name['AliasEntryProto']
_DYNAMICPARAMETERBINDINGPROTO = DESCRIPTOR.message_types_by_name['DynamicParameterBindingProto']
_DYNAMICPARAMETERBINDINGPROTO_BINDING = _DYNAMICPARAMETERBINDINGPROTO.nested_types_by_name['Binding']
_CROSSPROGRAMPREFETCH = DESCRIPTOR.message_types_by_name['CrossProgramPrefetch']
_HLOMODULEPROTO = DESCRIPTOR.message_types_by_name['HloModuleProto']
_HLOMODULEPROTO_PROFILEINFO = _HLOMODULEPROTO.nested_types_by_name['ProfileInfo']
_LOGICALBUFFERPROTO = DESCRIPTOR.message_types_by_name['LogicalBufferProto']
_LOGICALBUFFERPROTO_LOCATION = _LOGICALBUFFERPROTO.nested_types_by_name['Location']
_BUFFERALLOCATIONPROTO = DESCRIPTOR.message_types_by_name['BufferAllocationProto']
_BUFFERALLOCATIONPROTO_ASSIGNED = _BUFFERALLOCATIONPROTO.nested_types_by_name['Assigned']
_HEAPSIMULATORTRACE = DESCRIPTOR.message_types_by_name['HeapSimulatorTrace']
_HEAPSIMULATORTRACE_EVENT = _HEAPSIMULATORTRACE.nested_types_by_name['Event']
_HLOMODULEGROUPPROTO = DESCRIPTOR.message_types_by_name['HloModuleGroupProto']
_BUFFERASSIGNMENTPROTO = DESCRIPTOR.message_types_by_name['BufferAssignmentProto']
_BUFFERASSIGNMENTPROTO_BUFFERALIAS = _BUFFERASSIGNMENTPROTO.nested_types_by_name['BufferAlias']
_HLOPROTO = DESCRIPTOR.message_types_by_name['HloProto']
_HLOSNAPSHOT = DESCRIPTOR.message_types_by_name['HloSnapshot']
_HLOMODULEMETADATAPROTO = DESCRIPTOR.message_types_by_name['HloModuleMetadataProto']
_HLOPASSMETADATA = DESCRIPTOR.message_types_by_name['HloPassMetadata']
_ENTRYFUNCTIONATTRIBUTES = DESCRIPTOR.message_types_by_name['EntryFunctionAttributes']
_ENTRYFUNCTIONATTRIBUTES_SHAPEINDEX = _ENTRYFUNCTIONATTRIBUTES.nested_types_by_name['ShapeIndex']
_ENTRYFUNCTIONATTRIBUTES_BUFFERPARAMETERATTRIBUTES = _ENTRYFUNCTIONATTRIBUTES.nested_types_by_name['BufferParameterAttributes']
_XLARUNTIMEEXECUTABLEPROTO = DESCRIPTOR.message_types_by_name['XlaRuntimeExecutableProto']
_HLOMODULEPROTO_PROFILETYPE = _HLOMODULEPROTO.enum_types_by_name['ProfileType']
_HEAPSIMULATORTRACE_EVENT_KIND = _HEAPSIMULATORTRACE_EVENT.enum_types_by_name['Kind']
HloInstructionProto = _reflection.GeneratedProtocolMessageType('HloInstructionProto', (_message.Message,), {

  'SliceDimensions' : _reflection.GeneratedProtocolMessageType('SliceDimensions', (_message.Message,), {
    'DESCRIPTOR' : _HLOINSTRUCTIONPROTO_SLICEDIMENSIONS,
    '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
    # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.HloInstructionProto.SliceDimensions)
    })
  ,
  'DESCRIPTOR' : _HLOINSTRUCTIONPROTO,
  '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.HloInstructionProto)
  })
_sym_db.RegisterMessage(HloInstructionProto)
_sym_db.RegisterMessage(HloInstructionProto.SliceDimensions)

HloComputationProto = _reflection.GeneratedProtocolMessageType('HloComputationProto', (_message.Message,), {
  'DESCRIPTOR' : _HLOCOMPUTATIONPROTO,
  '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.HloComputationProto)
  })
_sym_db.RegisterMessage(HloComputationProto)

HloScheduleProto = _reflection.GeneratedProtocolMessageType('HloScheduleProto', (_message.Message,), {

  'InstructionSequence' : _reflection.GeneratedProtocolMessageType('InstructionSequence', (_message.Message,), {
    'DESCRIPTOR' : _HLOSCHEDULEPROTO_INSTRUCTIONSEQUENCE,
    '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
    # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.HloScheduleProto.InstructionSequence)
    })
  ,

  'SequencesEntry' : _reflection.GeneratedProtocolMessageType('SequencesEntry', (_message.Message,), {
    'DESCRIPTOR' : _HLOSCHEDULEPROTO_SEQUENCESENTRY,
    '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
    # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.HloScheduleProto.SequencesEntry)
    })
  ,
  'DESCRIPTOR' : _HLOSCHEDULEPROTO,
  '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.HloScheduleProto)
  })
_sym_db.RegisterMessage(HloScheduleProto)
_sym_db.RegisterMessage(HloScheduleProto.InstructionSequence)
_sym_db.RegisterMessage(HloScheduleProto.SequencesEntry)

HloInputOutputAliasProto = _reflection.GeneratedProtocolMessageType('HloInputOutputAliasProto', (_message.Message,), {

  'AliasEntryProto' : _reflection.GeneratedProtocolMessageType('AliasEntryProto', (_message.Message,), {
    'DESCRIPTOR' : _HLOINPUTOUTPUTALIASPROTO_ALIASENTRYPROTO,
    '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
    # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.HloInputOutputAliasProto.AliasEntryProto)
    })
  ,
  'DESCRIPTOR' : _HLOINPUTOUTPUTALIASPROTO,
  '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.HloInputOutputAliasProto)
  })
_sym_db.RegisterMessage(HloInputOutputAliasProto)
_sym_db.RegisterMessage(HloInputOutputAliasProto.AliasEntryProto)

DynamicParameterBindingProto = _reflection.GeneratedProtocolMessageType('DynamicParameterBindingProto', (_message.Message,), {

  'Binding' : _reflection.GeneratedProtocolMessageType('Binding', (_message.Message,), {
    'DESCRIPTOR' : _DYNAMICPARAMETERBINDINGPROTO_BINDING,
    '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
    # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.DynamicParameterBindingProto.Binding)
    })
  ,
  'DESCRIPTOR' : _DYNAMICPARAMETERBINDINGPROTO,
  '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.DynamicParameterBindingProto)
  })
_sym_db.RegisterMessage(DynamicParameterBindingProto)
_sym_db.RegisterMessage(DynamicParameterBindingProto.Binding)

CrossProgramPrefetch = _reflection.GeneratedProtocolMessageType('CrossProgramPrefetch', (_message.Message,), {
  'DESCRIPTOR' : _CROSSPROGRAMPREFETCH,
  '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.CrossProgramPrefetch)
  })
_sym_db.RegisterMessage(CrossProgramPrefetch)

HloModuleProto = _reflection.GeneratedProtocolMessageType('HloModuleProto', (_message.Message,), {

  'ProfileInfo' : _reflection.GeneratedProtocolMessageType('ProfileInfo', (_message.Message,), {
    'DESCRIPTOR' : _HLOMODULEPROTO_PROFILEINFO,
    '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
    # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.HloModuleProto.ProfileInfo)
    })
  ,
  'DESCRIPTOR' : _HLOMODULEPROTO,
  '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.HloModuleProto)
  })
_sym_db.RegisterMessage(HloModuleProto)
_sym_db.RegisterMessage(HloModuleProto.ProfileInfo)

LogicalBufferProto = _reflection.GeneratedProtocolMessageType('LogicalBufferProto', (_message.Message,), {

  'Location' : _reflection.GeneratedProtocolMessageType('Location', (_message.Message,), {
    'DESCRIPTOR' : _LOGICALBUFFERPROTO_LOCATION,
    '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
    # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.LogicalBufferProto.Location)
    })
  ,
  'DESCRIPTOR' : _LOGICALBUFFERPROTO,
  '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.LogicalBufferProto)
  })
_sym_db.RegisterMessage(LogicalBufferProto)
_sym_db.RegisterMessage(LogicalBufferProto.Location)

BufferAllocationProto = _reflection.GeneratedProtocolMessageType('BufferAllocationProto', (_message.Message,), {

  'Assigned' : _reflection.GeneratedProtocolMessageType('Assigned', (_message.Message,), {
    'DESCRIPTOR' : _BUFFERALLOCATIONPROTO_ASSIGNED,
    '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
    # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.BufferAllocationProto.Assigned)
    })
  ,
  'DESCRIPTOR' : _BUFFERALLOCATIONPROTO,
  '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.BufferAllocationProto)
  })
_sym_db.RegisterMessage(BufferAllocationProto)
_sym_db.RegisterMessage(BufferAllocationProto.Assigned)

HeapSimulatorTrace = _reflection.GeneratedProtocolMessageType('HeapSimulatorTrace', (_message.Message,), {

  'Event' : _reflection.GeneratedProtocolMessageType('Event', (_message.Message,), {
    'DESCRIPTOR' : _HEAPSIMULATORTRACE_EVENT,
    '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
    # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.HeapSimulatorTrace.Event)
    })
  ,
  'DESCRIPTOR' : _HEAPSIMULATORTRACE,
  '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.HeapSimulatorTrace)
  })
_sym_db.RegisterMessage(HeapSimulatorTrace)
_sym_db.RegisterMessage(HeapSimulatorTrace.Event)

HloModuleGroupProto = _reflection.GeneratedProtocolMessageType('HloModuleGroupProto', (_message.Message,), {
  'DESCRIPTOR' : _HLOMODULEGROUPPROTO,
  '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.HloModuleGroupProto)
  })
_sym_db.RegisterMessage(HloModuleGroupProto)

BufferAssignmentProto = _reflection.GeneratedProtocolMessageType('BufferAssignmentProto', (_message.Message,), {

  'BufferAlias' : _reflection.GeneratedProtocolMessageType('BufferAlias', (_message.Message,), {
    'DESCRIPTOR' : _BUFFERASSIGNMENTPROTO_BUFFERALIAS,
    '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
    # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.BufferAssignmentProto.BufferAlias)
    })
  ,
  'DESCRIPTOR' : _BUFFERASSIGNMENTPROTO,
  '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.BufferAssignmentProto)
  })
_sym_db.RegisterMessage(BufferAssignmentProto)
_sym_db.RegisterMessage(BufferAssignmentProto.BufferAlias)

HloProto = _reflection.GeneratedProtocolMessageType('HloProto', (_message.Message,), {
  'DESCRIPTOR' : _HLOPROTO,
  '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.HloProto)
  })
_sym_db.RegisterMessage(HloProto)

HloSnapshot = _reflection.GeneratedProtocolMessageType('HloSnapshot', (_message.Message,), {
  'DESCRIPTOR' : _HLOSNAPSHOT,
  '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.HloSnapshot)
  })
_sym_db.RegisterMessage(HloSnapshot)

HloModuleMetadataProto = _reflection.GeneratedProtocolMessageType('HloModuleMetadataProto', (_message.Message,), {
  'DESCRIPTOR' : _HLOMODULEMETADATAPROTO,
  '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.HloModuleMetadataProto)
  })
_sym_db.RegisterMessage(HloModuleMetadataProto)

HloPassMetadata = _reflection.GeneratedProtocolMessageType('HloPassMetadata', (_message.Message,), {
  'DESCRIPTOR' : _HLOPASSMETADATA,
  '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.HloPassMetadata)
  })
_sym_db.RegisterMessage(HloPassMetadata)

EntryFunctionAttributes = _reflection.GeneratedProtocolMessageType('EntryFunctionAttributes', (_message.Message,), {

  'ShapeIndex' : _reflection.GeneratedProtocolMessageType('ShapeIndex', (_message.Message,), {
    'DESCRIPTOR' : _ENTRYFUNCTIONATTRIBUTES_SHAPEINDEX,
    '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
    # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.EntryFunctionAttributes.ShapeIndex)
    })
  ,

  'BufferParameterAttributes' : _reflection.GeneratedProtocolMessageType('BufferParameterAttributes', (_message.Message,), {
    'DESCRIPTOR' : _ENTRYFUNCTIONATTRIBUTES_BUFFERPARAMETERATTRIBUTES,
    '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
    # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.EntryFunctionAttributes.BufferParameterAttributes)
    })
  ,
  'DESCRIPTOR' : _ENTRYFUNCTIONATTRIBUTES,
  '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.EntryFunctionAttributes)
  })
_sym_db.RegisterMessage(EntryFunctionAttributes)
_sym_db.RegisterMessage(EntryFunctionAttributes.ShapeIndex)
_sym_db.RegisterMessage(EntryFunctionAttributes.BufferParameterAttributes)

XlaRuntimeExecutableProto = _reflection.GeneratedProtocolMessageType('XlaRuntimeExecutableProto', (_message.Message,), {
  'DESCRIPTOR' : _XLARUNTIMEEXECUTABLEPROTO,
  '__module__' : 'torch_neuronx.pyhlo.hlo_pb2'
  # @@protoc_insertion_point(class_scope:torch_neuronx.pyhlo.XlaRuntimeExecutableProto)
  })
_sym_db.RegisterMessage(XlaRuntimeExecutableProto)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\370\001\001'
  _HLOINSTRUCTIONPROTO.fields_by_name['all_reduce_id']._options = None
  _HLOINSTRUCTIONPROTO.fields_by_name['all_reduce_id']._serialized_options = b'\030\001'
  _HLOINSTRUCTIONPROTO.fields_by_name['is_cross_program_prefetch']._options = None
  _HLOINSTRUCTIONPROTO.fields_by_name['is_cross_program_prefetch']._serialized_options = b'\030\001'
  _HLOSCHEDULEPROTO_SEQUENCESENTRY._options = None
  _HLOSCHEDULEPROTO_SEQUENCESENTRY._serialized_options = b'8\001'
  _LOGICALBUFFERPROTO_LOCATION.fields_by_name['instruction_name']._options = None
  _LOGICALBUFFERPROTO_LOCATION.fields_by_name['instruction_name']._serialized_options = b'\030\001'
  _CUSTOMCALLSCHEDULE._serialized_start=8543
  _CUSTOMCALLSCHEDULE._serialized_end=8626
  _CUSTOMCALLAPIVERSION._serialized_start=8629
  _CUSTOMCALLAPIVERSION._serialized_end=8809
  _KIND._serialized_start=8811
  _KIND._serialized_end=8869
  _HLOINSTRUCTIONPROTO._serialized_start=91
  _HLOINSTRUCTIONPROTO._serialized_end=3327
  _HLOINSTRUCTIONPROTO_SLICEDIMENSIONS._serialized_start=2982
  _HLOINSTRUCTIONPROTO_SLICEDIMENSIONS._serialized_end=3045
  _HLOCOMPUTATIONPROTO._serialized_start=3330
  _HLOCOMPUTATIONPROTO._serialized_end=3595
  _HLOSCHEDULEPROTO._serialized_start=3598
  _HLOSCHEDULEPROTO._serialized_end=3846
  _HLOSCHEDULEPROTO_INSTRUCTIONSEQUENCE._serialized_start=3691
  _HLOSCHEDULEPROTO_INSTRUCTIONSEQUENCE._serialized_end=3737
  _HLOSCHEDULEPROTO_SEQUENCESENTRY._serialized_start=3739
  _HLOSCHEDULEPROTO_SEQUENCESENTRY._serialized_end=3846
  _HLOINPUTOUTPUTALIASPROTO._serialized_start=3849
  _HLOINPUTOUTPUTALIASPROTO._serialized_end=4101
  _HLOINPUTOUTPUTALIASPROTO_ALIASENTRYPROTO._serialized_start=3958
  _HLOINPUTOUTPUTALIASPROTO_ALIASENTRYPROTO._serialized_end=4101
  _DYNAMICPARAMETERBINDINGPROTO._serialized_start=4104
  _DYNAMICPARAMETERBINDINGPROTO._serialized_end=4362
  _DYNAMICPARAMETERBINDINGPROTO_BINDING._serialized_start=4213
  _DYNAMICPARAMETERBINDINGPROTO_BINDING._serialized_end=4362
  _CROSSPROGRAMPREFETCH._serialized_start=4364
  _CROSSPROGRAMPREFETCH._serialized_end=4436
  _HLOMODULEPROTO._serialized_start=4439
  _HLOMODULEPROTO._serialized_end=5609
  _HLOMODULEPROTO_PROFILEINFO._serialized_start=5302
  _HLOMODULEPROTO_PROFILEINFO._serialized_end=5538
  _HLOMODULEPROTO_PROFILETYPE._serialized_start=5540
  _HLOMODULEPROTO_PROFILETYPE._serialized_end=5609
  _LOGICALBUFFERPROTO._serialized_start=5612
  _LOGICALBUFFERPROTO._serialized_end=5836
  _LOGICALBUFFERPROTO_LOCATION._serialized_start=5745
  _LOGICALBUFFERPROTO_LOCATION._serialized_end=5836
  _BUFFERALLOCATIONPROTO._serialized_start=5839
  _BUFFERALLOCATIONPROTO._serialized_end=6231
  _BUFFERALLOCATIONPROTO_ASSIGNED._serialized_start=6164
  _BUFFERALLOCATIONPROTO_ASSIGNED._serialized_end=6231
  _HEAPSIMULATORTRACE._serialized_start=6234
  _HEAPSIMULATORTRACE._serialized_end=6608
  _HEAPSIMULATORTRACE_EVENT._serialized_start=6386
  _HEAPSIMULATORTRACE_EVENT._serialized_end=6608
  _HEAPSIMULATORTRACE_EVENT_KIND._serialized_start=6565
  _HEAPSIMULATORTRACE_EVENT_KIND._serialized_end=6608
  _HLOMODULEGROUPPROTO._serialized_start=6610
  _HLOMODULEGROUPPROTO._serialized_end=6703
  _BUFFERASSIGNMENTPROTO._serialized_start=6706
  _BUFFERASSIGNMENTPROTO._serialized_end=7128
  _BUFFERASSIGNMENTPROTO_BUFFERALIAS._serialized_start=7021
  _BUFFERASSIGNMENTPROTO_BUFFERALIAS._serialized_end=7128
  _HLOPROTO._serialized_start=7131
  _HLOPROTO._serialized_end=7289
  _HLOSNAPSHOT._serialized_start=7292
  _HLOSNAPSHOT._serialized_end=7482
  _HLOMODULEMETADATAPROTO._serialized_start=7485
  _HLOMODULEMETADATAPROTO._serialized_end=7686
  _HLOPASSMETADATA._serialized_start=7689
  _HLOPASSMETADATA._serialized_end=7923
  _ENTRYFUNCTIONATTRIBUTES._serialized_start=7926
  _ENTRYFUNCTIONATTRIBUTES._serialized_end=8409
  _ENTRYFUNCTIONATTRIBUTES_SHAPEINDEX._serialized_start=8068
  _ENTRYFUNCTIONATTRIBUTES_SHAPEINDEX._serialized_end=8097
  _ENTRYFUNCTIONATTRIBUTES_BUFFERPARAMETERATTRIBUTES._serialized_start=8100
  _ENTRYFUNCTIONATTRIBUTES_BUFFERPARAMETERATTRIBUTES._serialized_end=8409
  _XLARUNTIMEEXECUTABLEPROTO._serialized_start=8412
  _XLARUNTIMEEXECUTABLEPROTO._serialized_end=8541
# @@protoc_insertion_point(module_scope)
