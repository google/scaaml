# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model training."""

import argparse
import json
import sys
from termcolor import cprint

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow.keras.backend as K

from scaaml.intro.generator import create_dataset
from scaaml.intro.model import get_model
from scaaml.utils import get_model_stub
from scaaml.utils import get_num_gpu
from scaaml.utils import tf_cap_memory


def train_model(config):
    tf_cap_memory()
    algorithm = config["algorithm"]
    train_glob = f"datasets/{algorithm}/train/*"
    test_glob = f"datasets/{algorithm}/test/*"
    test_shards = 256
    num_traces_per_test_shards = 16
    batch_size = config["batch_size"] * get_num_gpu()

    for attack_byte in config["attack_bytes"]:
        for attack_point in config["attack_points"]:

            x_train, y_train = create_dataset(
                train_glob,
                batch_size=batch_size,
                attack_point=attack_point,
                attack_byte=attack_byte,
                num_shards=config["num_shards"],
                num_traces_per_shard=config["num_traces_per_shard"],
                max_trace_length=config["max_trace_len"],
                is_training=True)

            x_test, y_test = create_dataset(
                test_glob,
                batch_size=batch_size,
                attack_point=attack_point,
                attack_byte=attack_byte,
                num_shards=test_shards,
                num_traces_per_shard=num_traces_per_test_shards,
                max_trace_length=config["max_trace_len"],
                is_training=False)

            # infers shape
            input_shape = x_train.shape[1:]

            # reset graph and load a new model
            K.clear_session()

            # display config
            cprint(f"[{algorithm}]", "magenta")
            cprint(">Attack params", "green")
            cprint(f"|-attack_point:{attack_point}", "cyan")
            cprint(f"|-attack_byte:{attack_byte}", "yellow")
            cprint(f"|-input_shape:{str(input_shape)}", "cyan")

            # multi gpu
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                model = get_model(input_shape, attack_point, config)

                # model recording setup
                stub = get_model_stub(attack_point, attack_byte, config)
                cb = [
                    ModelCheckpoint(monitor="val_loss",
                                    filepath=f"models/{stub}",
                                    save_best_only=True),
                    TensorBoard(log_dir="logs/" + stub, update_freq="batch")
                ]

                model.fit(x_train,
                          y_train,
                          validation_data=(x_test, y_test),
                          verbose=1,
                          epochs=config["epochs"],
                          callbacks=cb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models")
    parser.add_argument("--config", "-c", help="Train config")
    args = parser.parse_args()
    if not args.config:
        parser.print_help()
        sys.exit()
    with open(args.config, encoding="utf-8") as config_file:
        train_model(json.loads(config_file.read()))
