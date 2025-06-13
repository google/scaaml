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


#for testing
import json
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
from tensorflow.keras import metrics
from tabulate import tabulate
import matplotlib.pyplot as plt
from scaaml.aes import ap_preds_to_key_preds
from scaaml.plot import plot_trace, plot_confusion_matrix
from scaaml.utils import tf_cap_memory, from_categorical
from scaaml.model import get_models_by_attack_point, get_models_list, load_model_from_disk
from scaaml.intro.generator import list_shards, load_attack_shard
from scaaml.utils import hex_display, bytelist_to_hex

def train_model(config):
    tf_cap_memory()
    algorithm = config["algorithm"]
    train_glob = f"/kaggle/working/scaaml/scaaml_intro/datasets/{algorithm}/train/*"
    test_glob = f"/kaggle/working/scaaml/scaaml_intro/datasets/{algorithm}/test/*"
    test_shards = 256
    num_traces_per_test_shards = 16
    batch_size = config["batch_size"] * get_num_gpu()

    if( True):
    #for attack_byte in config["attack_bytes"]:   #go through all attackbyte and all attack point and take data approprioately
        # for attack_point in config["attack_points"]:
        attack_point="sub_bytes_out"
        attack_byte=2
#the shards each has 48 dataparts each has 256 plaintexts ka traces
        x_train, y_train = create_dataset(
            train_glob,
            batch_size=batch_size,
            attack_point=attack_point,
            attack_byte=attack_byte,
            num_shards=config["num_shards"],
            num_traces_per_shard=config["num_traces_per_shard"],
            max_trace_length=config["max_trace_len"],
            is_training=True)

        x_test, y_test = create_dataset( # 256 keys covered and 16 traces per key
            test_glob,
            batch_size=batch_size,
            attack_point=attack_point,
            attack_byte=attack_byte,
            num_shards=test_shards,
            num_traces_per_shard=num_traces_per_test_shards,
            max_trace_length=config["max_trace_len"],
            is_training=False)
        #print(x_train[])
        # infers shape
        input_shape = x_train.shape[1:]
        #print("hey",x_train.shape)
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
                                filepath=f"models/{stub}.keras",
                                save_best_only=True),
                TensorBoard(log_dir="logs/" + stub, update_freq="batch")
            ]
            
            
            model.fit(x_train,
                        y_train,
                        validation_data=(x_test, y_test),
                        verbose=1,
                        epochs=config["epochs"],
                        callbacks=cb)
            # model.save(f"models/{stub}.keras")
                # model.fit(x_train,
                #           y_train,
                #           validation_data=(x_test, y_test),
                #           verbose=1,
                #           epochs=30,
                #           )

                # ATTACK_POINT = attack_point
                # target = 'stm32f415_tinyaes'
                # tf_cap_memory()
                # target_config = json.loads(open("/home/sanju/scaaml/scaaml_intro/config/" + target + '.json').read())
                # BATCH_SIZE = target_config['batch_size']
                # TRACE_LEN = target_config['max_trace_len']
                # DATASET_GLOB = "/home/sanju/scaaml/scaaml_intro/datasets/%s/test/*" % target_config['algorithm']
                # shard_paths  = list_shards(DATASET_GLOB, 256)
                # NUM_TRACES = 10  # maximum number of traces to use to recover a given key byte. 10 is already overkill
                # correct_prediction_rank = defaultdict(list)
                # y_pred = []
                # y_true = []
                # model_metrics = {"acc": metrics.Accuracy()}
                # for shard in tqdm(shard_paths, desc='Recovering bytes', unit='shards'):
                #     keys, pts, x, y = load_attack_shard(shard, attack_byte, ATTACK_POINT, TRACE_LEN, num_traces=NUM_TRACES)

                #     # prediction
                #     predictions = model.predict(x)
                    
                #     # computing byte prediction from intermediate predictions
                #     key_preds = ap_preds_to_key_preds(predictions, pts, ATTACK_POINT)
                    
                #     c_preds = from_categorical(predictions)
                #     c_y = from_categorical(y)
                #     # metric tracking
                #     for metric in model_metrics.values():
                #         metric.update_state(c_y, c_preds)
                #     # for the confusion matrix
                #     y_pred.extend(c_preds)
                #     y_true.extend(c_y)
                    
                #     # accumulating probabilities and checking correct guess position.
                #     # if all goes well it will be at position 0 (highest probability)
                #     # see below on how to use for the real attack
                    
                #     key = keys[0] # all the same in the same shard - not used in real attack
                #     print("key pred: ",key_preds)
                #     vals = np.zeros((256))
                #     for trace_count, kp in enumerate(key_preds):
                #         vals = vals  + np.log10(kp + 1e-22) 
                #         guess_ranks = (np.argsort(vals, )[-256:][::-1])
                #         byte_rank = list(guess_ranks).index(key)
                       
                #         correct_prediction_rank[trace_count].append(byte_rank)
               
                # print("Accuracy: %.2f" % model_metrics['acc'].result())
                # plot_confusion_matrix(y_true, y_pred, normalize=True, title="%s byte %s prediction confusion matrix" % (ATTACK_POINT, 0))
                # NUM_TRACES_TO_PLOT = 10
                # avg_preds = np.array([correct_prediction_rank[i].count(0) for i in range(NUM_TRACES_TO_PLOT)])
                # y = avg_preds / len(correct_prediction_rank[0]) * 100 
                # x = [i + 1 for i in range(NUM_TRACES_TO_PLOT)]
                # plt.plot(x, y)
                # plt.xlabel("Num traces")
                # plt.ylabel("Recovery success rate in %")
                # plt.title("%s ap:%s byte:%s recovery performance" % (target_config['algorithm'], ATTACK_POINT, 0))
                # plt.show()
                # min_traces = 0
                # max_traces = 0
                # cumulative_aa = 0
                # for idx, val in enumerate(y):
                #     cumulative_aa += val
                #     if not min_traces and val > 0:
                #         min_traces = idx + 1
                #     if not max_traces and val == 100.0:
                #         max_traces = idx + 1
                #         break 

                # cumulative_aa = round( cumulative_aa / (idx + 1), 2) # divide by the number of steps

                # rows = [
                #     ["min traces", min_traces, round(y[min_traces -1 ], 1)],
                #     ["max traces", max_traces, round(y[max_traces - 1], 1)],
                #     ["cumulative score", cumulative_aa, '-'] 
                # ]
                # print(tabulate(rows, headers=['metric', 'num traces', '% of keys']))
                
                # break
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models")
    parser.add_argument("--config", "-c", help="Train config")
    args = parser.parse_args()
    if not args.config:
        parser.print_help()
        sys.exit()
    with open(args.config, encoding="utf-8") as config_file:
        train_model(json.loads(config_file.read()))
