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
import torch
from torch_neuronx.pyhlo.xla_data_pb2 import PrimitiveType


def serialize_torch(value, element_type):
    tensor = torch.tensor(value, dtype=_primitive_type_to_dtype[element_type])
    if tensor.dtype == torch.bfloat16:  # "Got unsupported ScalarType BFloat16" workaround
        tensor = tensor.view(torch.int16)
    return tensor.numpy().tobytes()

_torch_dtype_to_primitive_type = {
    torch.bool: PrimitiveType.PRED,
    torch.int8: PrimitiveType.S8,
    torch.int16: PrimitiveType.S16,
    torch.int32: PrimitiveType.S32,
    torch.int64: PrimitiveType.S64,
    torch.uint8: PrimitiveType.U8,
    torch.float16: PrimitiveType.F16,
    torch.float32: PrimitiveType.F32,
    torch.float64: PrimitiveType.F64,
    torch.bfloat16: PrimitiveType.BF16,
    torch.complex64: PrimitiveType.C64,
    torch.complex128: PrimitiveType.C128,
}

_primitive_type_to_dtype = {
    PrimitiveType.PRED: torch.bool,
    PrimitiveType.S8: torch.int8,
    PrimitiveType.S16: torch.int16,
    PrimitiveType.S32: torch.int32,
    PrimitiveType.S64: torch.int64,
    PrimitiveType.U8: torch.uint8,
    PrimitiveType.U16: torch.int16,  # integer signedness does not matter for byte serialization
    PrimitiveType.U32: torch.int32,
    PrimitiveType.U64: torch.int64,
    PrimitiveType.F16: torch.float16,
    PrimitiveType.F32: torch.float32,
    PrimitiveType.BF16: torch.bfloat16,
    PrimitiveType.F64: torch.float64,
    PrimitiveType.C64: torch.complex64,
    PrimitiveType.C128: torch.complex128,
}
