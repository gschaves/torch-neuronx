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
import numpy
from torch_neuronx.pyhlo.xla_data_pb2 import PrimitiveType


def serialize_numpy(value, element_type):
    return _primitive_type_to_dtype[element_type](value).tobytes()


_primitive_type_to_dtype = {
    PrimitiveType.PRED: numpy.bool,
    PrimitiveType.S8: numpy.int8,
    PrimitiveType.S16: numpy.int16,
    PrimitiveType.S32: numpy.int32,
    PrimitiveType.S64: numpy.int64,
    PrimitiveType.U8: numpy.uint8,
    PrimitiveType.U16: numpy.uint16,
    PrimitiveType.U32: numpy.uint32,
    PrimitiveType.U64: numpy.uint64,
    PrimitiveType.F16: numpy.float16,
    PrimitiveType.F32: numpy.float32,
    PrimitiveType.BF16: None,
    PrimitiveType.F64: numpy.float64,
    PrimitiveType.C64: numpy.complex64,
    PrimitiveType.C128: numpy.complex128,
}
