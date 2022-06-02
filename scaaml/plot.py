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
"""Plotting functions."""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def plot_heatmap(batch, cmap="Reds", title=None):
    plt.figure(figsize=(15, 5))

    # its a batch
    if len(batch.shape) == 3:
        data = np.squeeze(batch)
        data = data[:16]
        data = np.tile(data, (data.shape[0] * 2, 1))

    # its a single trace
    elif len(batch.shape) == 1:
        data = np.tile(batch, (batch.shape[0] // 4, 1))
    else:
        data = batch

    if title:
        plt.title(title)

    plt.imshow(data, cmap=cmap)

    plt.yticks([])
    plt.show()


def plot_trace(trace,
               title=None,
               plot_avg=False,
               plot_std=False,
               x_labels=None):
    plt.figure(figsize=(15, 5))
    plt.plot(trace)

    if plot_avg:
        plt.plot(np.repeat(np.average(trace), len(trace)))

    if plot_std:
        plt.plot(np.repeat(np.average(trace) + np.std(trace), len(trace)))

    if title:
        plt.title(title)

    if x_labels:
        plt.xticks(list(range(len(trace))), x_labels)

    plt.show()


def plot_comparaison(traces, labels, title):
    "Color coded comparaison"
    plt.figure(figsize=(15, 5))
    for idx, trace in enumerate(traces):
        label = labels[idx]
        label_lower: str = label.lower()
        if label == "SNR":
            color = "#8E24AA"
        elif "activation" in label_lower:
            color = "#64DD17"
        elif "grad" in label_lower:
            color = "#26A69A"
        elif "scald" in label_lower:
            color = "#03A9F4"
        plt.plot(trace, label=label, color=color)

    plt.title(title)
    plt.legend()
    plt.show()


def plot_traces(traces, labels=None, title=None, xlabel=None, ylabel=None):
    plt.figure(figsize=(15, 5))
    for idx, trace in enumerate(traces):
        if labels:
            label = labels[idx]
            plt.plot(trace, label=label)
        else:
            plt.plot(trace)
    if title:
        plt.title(title)
    if labels:
        plt.legend()
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.show()


def plot_target_distribution(class_ids, title="Y distributions"):
    plt.title(f"{title} {len(class_ids)} examples")
    plt.hist(class_ids, bins=256)
    plt.xlabel("target value")
    plt.ylabel("example counts")
    plt.show()


def plot_confusion_matrix(class_ids,
                          predicted_class_ids,
                          title="Confusion matrix",
                          cmap=None,
                          normalize=True):
    """ Compute and plot the confusion matrix

    Args:
        class_ids (list(int)): Expected values
        predicted_class_ids (list(int)): predicted values
        cmap ([type], optional): Color map. Defaults to None.
        normalize (bool, optional): Normalize value between 0 and 1.
        Defaults to True.
    """

    cm = np.array(tf.math.confusion_matrix(class_ids, predicted_class_ids))
    # accuracy = np.trace(cm) / np.sum(cm).astype("float")

    if cmap is None:
        cmap = plt.get_cmap("Blues")

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.tight_layout()
    plt.ylabel("True intermediate values")
    plt.xlabel("Predicted intermediat values")
    plt.show()
