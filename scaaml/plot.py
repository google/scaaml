# Copyright 2020-2024 Google LLC
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

from typing import Optional, Sequence, Union, cast

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import tensorflow as tf


def plot_heatmap(batch: npt.NDArray[np.generic],
                 cmap: Optional[Union[str, Colormap]] = "Reds",
                 title: Optional[str] = None) -> None:
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


def plot_trace(trace: npt.NDArray[np.float64],
               title: Optional[str] = None,
               plot_avg: bool = False,
               plot_std: bool = False,
               x_labels: Optional[Sequence[str]] = None) -> None:
    plt.figure(figsize=(15, 5))
    plt.plot(trace)

    if plot_avg:
        plt.plot(np.repeat(np.average(trace), len(trace)))

    if plot_std:
        plt.plot(
            np.repeat(
                cast(np.float64, np.average(trace)) +
                cast(npt.NDArray[np.float64], np.std(trace)), len(trace)))

    if title:
        plt.title(title)

    if x_labels:
        plt.xticks(list(range(len(trace))), x_labels)

    plt.show()


def plot_comparison(traces: Sequence[npt.ArrayLike], labels: Sequence[str],
                    title: str) -> None:
    "Color coded comparison"
    plt.figure(figsize=(15, 5))
    for idx, trace in enumerate(traces):
        label = labels[idx]
        label_lower: str = label.lower()
        color: Optional[str] = None
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


def plot_trace_and_trigger(trace: npt.NDArray[np.generic],
                           trigger: npt.NDArray[np.generic],
                           fig_filename: str = "capture.png") -> None:
    """Plot trace and trigger."""
    plt.clf()
    plt.plot(trace, color="blue")
    plt.plot(trigger, color="red")
    plt.savefig(fig_filename)


def plot_traces(traces: Sequence[npt.NDArray[np.generic]],
                labels: Optional[Sequence[str]] = None,
                title: Optional[str] = None,
                xlabel: Optional[str] = None,
                ylabel: Optional[str] = None) -> None:
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


def plot_target_distribution(class_ids: npt.NDArray[np.generic],
                             title: str = "Y distributions") -> None:
    plt.title(f"{title} {len(class_ids)} examples")
    plt.hist(class_ids, bins=256)
    plt.xlabel("target value")
    plt.ylabel("example counts")
    plt.show()


def plot_confusion_matrix(class_ids: Sequence[int],
                          predicted_class_ids: Sequence[int],
                          title: str = "Confusion matrix",
                          cmap: Optional[Union[str, Colormap]] = None,
                          normalize: bool = True) -> None:
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
    plt.xlabel("Predicted intermediate values")
    plt.show()
