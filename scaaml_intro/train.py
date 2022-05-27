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

import argparse
import json
from termcolor import cprint
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from scaaml.utils import tf_cap_memory
import tensorflow.keras.backend as K
from scaaml.utils import get_model_stub
from scaaml.intro.generator import create_dataset
from scaaml.intro.model import get_model
from scaaml.utils import get_num_gpu


def train_model(config):
    tf_cap_memory()
    TRAIN_GLOB = "datasets/%s/train/*" % config['algorithm']
    TEST_GLOB = "datasets/%s/test/*" % config['algorithm']
    TEST_SHARDS = 256
    NUM_TRACES_PER_TEST_SHARDS = 16
    BATCH_SIZE = config['batch_size'] * get_num_gpu()

    for attack_byte in config['attack_bytes']:
        for attack_point in config['attack_points']:

            x_train, y_train = create_dataset(
                TRAIN_GLOB,
                batch_size=BATCH_SIZE,
                attack_point=attack_point,
                attack_byte=attack_byte,
                num_shards=config['num_shards'],
                num_traces_per_shard=config["num_traces_per_shard"],
                max_trace_length=config['max_trace_len'],
                is_training=True)

            x_test, y_test = create_dataset(
                TEST_GLOB,
                batch_size=BATCH_SIZE,
                attack_point=attack_point,
                attack_byte=attack_byte,
                num_shards=TEST_SHARDS,
                num_traces_per_shard=NUM_TRACES_PER_TEST_SHARDS,
                max_trace_length=config['max_trace_len'],
                is_training=False)

            # infers shape
            input_shape = x_train.shape[1:]

            # reset graph and load a new model
            K.clear_session()

            # display config
            cprint('[%s]' % config['algorithm'], 'magenta')
            cprint(">Attack params", 'green')
            cprint("|-attack_point:%s" % attack_point, 'cyan')
            cprint("|-attack_byte:%s" % attack_byte, 'yellow')
            cprint("|-input_shape:%s" % str(input_shape), 'cyan')

            # multi gpu
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                model = get_model(input_shape, attack_point, config)

                # model recording setup
                stub = get_model_stub(attack_point, attack_byte, config)
                cb = [
                    ModelCheckpoint(monitor='val_loss',
                                    filepath='models/%s' % stub,
                                    save_best_only=True),
                    TensorBoard(log_dir='logs/' + stub, update_freq='batch')
                ]

                model.fit(x_train,
                          y_train,
                          validation_data=(x_test, y_test),
                          verbose=1,
                          epochs=config['epochs'],
                          callbacks=cb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train models')
    parser.add_argument('--config', '-c', help='Train config')
    args = parser.parse_args()
    if not args.config:
        parser.print_help()
        quit()
    config = json.loads(open(args.config).read())
    train_model(config)
