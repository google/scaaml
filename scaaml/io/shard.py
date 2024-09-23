# Copyright 2021-2024 Google LLC
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
"""Dataset shard manipulation."""

import math
from typing import Any, Dict, List, Literal, Optional, Union

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from scaaml.io.tfdata import bytes_feature, int64_feature, float_feature

CompressionT = Literal["", "GZIP", "ZLIB"]


@dataclass
class ShardStats:
    # Number of examples in the shard
    examples: int
    min_values: Dict[str, float]
    max_values: Dict[str, float]


class Shard:
    """A shard contains N measurement pertaining to the same key"""

    def __init__(self,
                 path: str,
                 attack_points_info: Dict[str, Any],
                 measurements_info: Dict[str, Any],
                 *,
                 measurement_dtype: tf.DType,
                 compression: CompressionT = "GZIP") -> None:
        self.path = path
        self.attack_points_info = attack_points_info
        self.measurements_info = measurements_info
        self.measurement_dtype = measurement_dtype
        self.compression = compression

        # Writer if needed
        self.writer: Optional[tf.io.TFRecordWriter] = None

        # counters
        self.examples = 0
        self.min_values: Dict[str, float] = {}
        self.max_values: Dict[str, float] = {}
        for k in measurements_info.keys():
            # By convention minimum of an empty set is infinity.
            self.min_values[k] = math.inf
            self.max_values[k] = -math.inf

        # build and cache tffeature format
        self.features = self._build_tffeature()

    def write(self, attack_points: Dict[str, bytearray],
              measurements: Dict[str, List[float]]) -> None:
        """Write example on disk as TFRecord

        Args:
            attack_points: Attack points values.
            measurements: Measurements values.
        """
        with tf.device("/cpu:0"):
            # open writer if needed
            # !do not put in the init to avoid erasing on read
            if self.writer is None:
                self.writer = tf.io.TFRecordWriter(self.path, self.compression)
            example = self._to_tfrecord(attack_points, measurements)
            self.writer.write(example)
            self.examples += 1

    def read(
        self,
        num: int = 10
    ) -> tf.data.Dataset[Dict[str, Union[tf.Tensor, tf.SparseTensor]]]:
        """Open and read N examples from the shard"""
        shard = tf.data.TFRecordDataset(self.path,
                                        compression_type=self.compression)
        data = shard.map(self._from_tfrecord)
        return data.take(num)

    def close(self) -> ShardStats:
        "close shard and return statistics"
        if not self.writer:
            raise ValueError("Trying to close a shard that was not open")

        self.writer.close()
        return ShardStats(examples=self.examples,
                          min_values=self.min_values,
                          max_values=self.max_values)

    def _to_tfrecord(self, attack_points: Dict[str, Any],
                     measurements: Dict[str, Any]) -> bytes:
        """Convert example data into a tfrecord example

        Args:
            attack_points: attack points data
            measurements: measurements data

        Returns:
            TF.train.Example
        """

        # check there are no unexpected values
        for k in attack_points:
            if k not in self.attack_points_info:
                raise ValueError("Attack point", k, "not specified")

        for k in measurements:
            if k not in self.measurements_info:
                raise ValueError("Measurement", k, "not specified")

        feature = {}
        # attack points as integers
        for ap_name, info in self.attack_points_info.items():
            expected_len = info["len"]
            ap_value = attack_points[ap_name]

            # check that we get the len specified in the info
            if len(ap_value) != expected_len:
                raise ValueError(ap_name, len(ap_value),
                                 "don't have the right len", expected_len)

            # convert
            feature[ap_name] = int64_feature(ap_value)

        # measurements as float
        for mname, info in self.measurements_info.items():
            expected_len = info["len"]
            measurement = measurements[mname]

            # check that the measurement len match what is specified in info
            if len(measurement) != expected_len:
                raise ValueError(f"{mname} has wrong length, expected "
                                 f"{expected_len}, got {len(measurement)}.")

            # min and max
            self.min_values[mname] = min(self.min_values[mname],
                                         float(tf.reduce_min(measurement)))
            self.max_values[mname] = max(self.max_values[mname],
                                         float(tf.reduce_max(measurement)))

            # convert
            if self.measurement_dtype == tf.float32:
                feature[mname] = float_feature(measurement)
            elif self.measurement_dtype == tf.float16:
                measurement = measurement.astype(dtype=np.float16)
                feature[mname] = bytes_feature(
                    [tf.io.serialize_tensor(measurement).numpy()])
            else:
                raise ValueError(
                    f"Wrong measurement_dtype: {self.measurement_dtype}")

        tf_features = tf.train.Features(feature=feature)
        record = tf.train.Example(features=tf_features)
        return bytes(record.SerializeToString())

    def _from_tfrecord(
            self,
            tfrecord: str) -> Dict[str, Union[tf.Tensor, tf.SparseTensor]]:
        """Convert tf_record to dictionary

        Args:
            tf_record: tf_record to parse
        Returns:
            reloaded example as dictionary
        """
        rec: Dict[str, Union[tf.Tensor, tf.SparseTensor]]
        rec = tf.io.parse_single_example(tfrecord, self._build_tffeature())
        if self.measurement_dtype == tf.float16:
            for name, ipt in self.measurements_info.items():
                rec[name] = tf.io.parse_tensor(rec[name], tf.float16)
                rec[name] = tf.ensure_shape(rec[name], shape=(ipt["len"],))
        return rec

    def _build_tffeature(self) -> Dict[str, tf.io.FixedLenFeature]:
        "build tf feature dictionary based of meta data"
        features = {}

        # attack points
        for k, info in self.attack_points_info.items():
            feature_length = info["len"]
            features[k] = tf.io.FixedLenFeature([feature_length], tf.int64)

        # measurements
        if self.measurement_dtype == tf.float16:
            for k in self.measurements_info:
                features[k] = tf.io.FixedLenFeature((), tf.string)
        elif self.measurement_dtype == tf.float32:
            for k, info in self.measurements_info.items():
                feature_length = info["len"]
                features[k] = tf.io.FixedLenFeature([feature_length],
                                                    tf.float32)
        else:
            raise ValueError(
                f"Wrong measurement_dtype: {self.measurement_dtype}")

        return features
