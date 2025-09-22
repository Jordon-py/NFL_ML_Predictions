# Copyright 2019 The KerasTuner Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""KerasTuner protos."""

from contextlib import contextmanager

# v3 is generated with protobuf==3.20.3 and grpcio-tools==1.48.0
# v4 is generated with protobuf==4.22.1 and grpcio-tools==1.53.0
try:
    from keras_tuner.src.protos import v4 as protos
except ImportError:
    from keras_tuner.src.protos import v3 as protos


def get_proto():
    return protos.keras_tuner_pb2


def get_service():
    return protos.service_pb2


def get_service_grpc():
    return protos.service_pb2_grpc

