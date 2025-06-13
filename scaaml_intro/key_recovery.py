import json
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf  # Add this line
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


target = 'stm32f415_tinyaes'
tf_cap_memory()
target_config = json.loads(open("/kaggle/working/scaaml/scaaml_intro/config/" + target + '.json').read())
BATCH_SIZE = target_config['batch_size']
TRACE_LEN = target_config['max_trace_len']



available_models = get_models_by_attack_point(target_config)


DATASET_GLOB = "/kaggle/working/scaaml/scaaml_intro/datasets/%s/test/*" % target_config['algorithm']
shard_paths  = list_shards(DATASET_GLOB, 256)

# let's select an attack point that have all the needed models -- Key is not a good target: it doesn't work
ATTACK_POINT = 'sub_bytes_out'

# let's also pick the key byte we want to use SCAAML to recover and load the related model
ATTACK_BYTE = 7

# load model
#model = load_model_from_disk(available_models[ATTACK_POINT][ATTACK_BYTE])
#layer=tf.keras.layers.TFSMLayer("/kaggle/working/scaaml/scaaml_intro/models/stm32f415-tinyaes-cnn-v10-ap_sub_bytes_out-byte_7-len_20000",call_endpoint='serving_default')

#layer = tf.keras.layers.TFSMLayer(
#    "/path/to/model",
 #   call_endpoint='serving_default'
#)

# Build a new model around it
# inputs = tf.keras.Input(shape=(256,20000,1))  # Match your input shape
# outputs = layer(inputs)
# model = tf.keras.Model(inputs, outputs)

# # Now use .predict()
# predictions = model.predict(x_test)




layer = tf.keras.layers.TFSMLayer("/kaggle/working/scaaml/scaaml_intro/models/stm32f415-tinyaes-cnn-v10-ap_sub_bytes_out-byte_7-len_20000",call_endpoint='serving_default')
input_layer = tf.keras.Input(shape=(20000,1))

outputs = layer(input_layer)
model = tf.keras.Model(input_layer, outputs)

model.summary()


 #tf.keras.models.load_model
NUM_TRACES = 10  # maximum number of traces to use to recover a given key byte. 10 is already overkill
correct_prediction_rank = defaultdict(list)
y_pred = []
y_true = []
model_metrics = {"acc": metrics.Accuracy()}
for shard in tqdm(shard_paths, desc='Recovering bytes', unit='shards'):
    keys, pts, x, y = load_attack_shard(shard, ATTACK_BYTE, ATTACK_POINT, TRACE_LEN, num_traces=NUM_TRACES)

    # prediction
    predictions = model.predict(x)['dense_1']
    
    # computing byte prediction from intermediate predictions
    key_preds = ap_preds_to_key_preds(predictions, pts, ATTACK_POINT)
    
    c_preds = from_categorical(predictions)
    c_y = from_categorical(y)
    # metric tracking
    for metric in model_metrics.values():
        metric.update_state(c_y, c_preds)
    # for the confusion matrix
    y_pred.extend(c_preds)
    y_true.extend(c_y)

    # accumulating probabilities and checking correct guess position.
    # if all goes well it will be at position 0 (highest probability)
    # see below on how to use for the real attack
    
    
    key = keys[0] # all the same in the same shard - not used in real attack
    vals = np.zeros((256))
    for trace_count, kp in enumerate(key_preds):
        vals = vals  + np.log10(kp + 1e-22) 
        guess_ranks = (np.argsort(vals, )[-256:][::-1])
        byte_rank = list(guess_ranks).index(key)
        correct_prediction_rank[trace_count].append(byte_rank)
for i in range(10):
    print("FINLLAY",correct_prediction_rank[i])

print("Accuracy: %.2f" % model_metrics['acc'].result())
plot_confusion_matrix(y_true, y_pred, normalize=True, title="%s byte %s prediction confusion matrix" % (ATTACK_POINT, ATTACK_BYTE))

NUM_TRACES_TO_PLOT = 10
avg_preds = np.array([correct_prediction_rank[i].count(0) for i in range(NUM_TRACES_TO_PLOT)])
y = avg_preds / len(correct_prediction_rank[0]) * 100 
x = [i + 1 for i in range(NUM_TRACES_TO_PLOT)]
plt.plot(x, y)
plt.xlabel("Num traces")
plt.ylabel("Recovery success rate in %")
plt.title("%s ap:%s byte:%s recovery performance" % (target_config['algorithm'], ATTACK_POINT, ATTACK_BYTE))
plt.show()
min_traces = 0
max_traces = 0
cumulative_aa = 0
for idx, val in enumerate(y):
    cumulative_aa += val
    if not min_traces and val > 0:
        min_traces = idx + 1
    if not max_traces and val == 100.0:
        max_traces = idx + 1
        break 

cumulative_aa = round( cumulative_aa / (idx + 1), 2) # divide by the number of steps

rows = [
    ["min traces", min_traces, round(y[min_traces -1 ], 1)],
    ["max traces", max_traces, round(y[max_traces - 1], 1)],
    ["cumulative score", cumulative_aa, '-'] 
]
print(tabulate(rows, headers=['metric', 'num traces', '% of keys']))



# ATTACK_POINT = 'sub_bytes_out' # let's pick an attack point- Key is not a good target: it doesn't work for TinyAEs
# TARGET_SHARD = 42 # a shard == a different key. Pick the one you would like
# NUM_TRACES = 5  # how many traces to use - as seen in single byte, 5 traces is enough
# # perfoming 16x the byte recovery algorithm showecased above - one for each key byte
# real_key = [] # what we are supposed to find
# recovered_key = [] # what we predicted
# pb = tqdm(total=16, desc="guessing key", unit='guesses')
# for ATTACK_BYTE in range(16):
#     # data
#     keys, pts, x, y = load_attack_shard(shard_paths[TARGET_SHARD], ATTACK_BYTE, ATTACK_POINT, TRACE_LEN, num_traces=NUM_TRACES, full_key=True)
#     real_key.append(keys[0])
    
#     # load model
#     layer = tf.keras.layers.TFSMLayer("/kaggle/working/scaaml/scaaml_intro/models/stm32f415-tinyaes-cnn-v10-ap_sub_bytes_out-byte_"+str(ATTACK_BYTE)+"-len_20000",call_endpoint='serving_default')
#     input_layer = tf.keras.Input(shape=(20000,1))

#     outputs = layer(input_layer)
#     model = tf.keras.Model(input_layer, outputs)

#     #model = load_model_from_disk(available_models[ATTACK_POINT][ATTACK_BYTE])
    
#     # prediction
#     predictions = model.predict(x)['dense_1']
    
#     # computing byte prediction from intermediate predictions
#     key_preds = ap_preds_to_key_preds(predictions, pts, ATTACK_POINT)
    
#     # accumulating probabity
#     vals = np.zeros((256))
#     for trace_count, kp in enumerate(key_preds):
#         vals = vals  + np.log10(kp + 1e-22)
    
#     # order predictions by probability
#     guess_ranks = (np.argsort(vals, )[-256:][::-1])
    
#     # take strongest guess as our key guess
#     recovered_key.append(guess_ranks[0])
    
#     # update display
#     pb.set_postfix({'Recovered key': bytelist_to_hex(recovered_key), "Real key": bytelist_to_hex(real_key)})
#     pb.update()
    
    
# pb.close()
# # check that everything worked out: the recovered key match the real keys
# hex_display(real_key, 'real key')
# hex_display(recovered_key, 'recovered key')
