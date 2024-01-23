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
import bfloat16
from torch_neuronx.pyhlo.xla_data_pb2 import PrimitiveType
from torch_neuronx.pyhlo.constant.serialize_numpy import serialize_numpy


def serialize_numpy_bfloat16(value, element_type):
    if element_type == PrimitiveType.BF16:
        return bfloat16.bfloat16(value).tobytes()
    return serializer_numpy(value, element_type)
