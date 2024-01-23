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
from tensorflow.python.framework import dtypes
from torch_neuronx.pyhlo.xla_data_pb2 import PrimitiveType


def serialize_tf(value, element_type):
    return _primitive_type_to_dtype[element_type].as_numpy_dtype(value).tobytes()


_primitive_type_to_dtype = {
    PrimitiveType.PRED: dtypes.bool,
    PrimitiveType.S8: dtypes.int8,
    PrimitiveType.S16: dtypes.int16,
    PrimitiveType.S32: dtypes.int32,
    PrimitiveType.S64: dtypes.int64,
    PrimitiveType.U8: dtypes.uint8,
    PrimitiveType.U16: dtypes.uint16,
    PrimitiveType.U32: dtypes.uint32,
    PrimitiveType.U64: dtypes.uint64,
    PrimitiveType.F16: dtypes.float16,
    PrimitiveType.F32: dtypes.float32,
    PrimitiveType.BF16: dtypes.bfloat16,
    PrimitiveType.F64: dtypes.float64,
    PrimitiveType.C64: dtypes.complex64,
    PrimitiveType.C128: dtypes.complex128,
}
