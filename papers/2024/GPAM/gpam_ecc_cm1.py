# Copyright 2022-2024 Google LLC
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
"""GPAM model, see https://github.com/google/scaaml/tree/main/papers/2024/GPAM

@article{bursztein2023generic,
  title={Generalized Power Attacks against Crypto Hardware using Long-Range Deep Learning},
  author={Bursztein, Elie and Invernizzi, Luca and Kr{\'a}l, Karel and Moghimi, Daniel and Picod, Jean-Michel and Zhang, Marina},
  journal={arXiv preprint arXiv:2306.07249},
  year={2023}
}

Hyperparameters are identified by a comment # hyperparameter

We found that CosineDecayWithWarmupSchedule is not necessary and one can use
Adafactor instead to get the same results with much fewer parameters to set.
"""

import argparse
from collections import defaultdict
import math
from typing import Any, Dict, List, Optional, Tuple, Union

# Doing topological sort on the relational outputs. One can do it by hand if
# they don't wish to install this package.
import networkx as nx
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow import Tensor

from scaaml.io import Dataset
from scaaml.metrics.custom import MeanRank


def clone_initializer(initializer: tf.keras.initializers.Initializer):
    """Clone an initializer (if an initializer is reused the generated
    weights are the same).
    """
    if isinstance(initializer, tf.keras.initializers.Initializer):
        return initializer.__class__.from_config(initializer.get_config())
    return initializer


def rope(x: Tensor, axis: Union[List[int], int]) -> Tensor:
    """RoPE positional encoding.

      Implementation of the Rotary Position Embedding proposed in
      https://arxiv.org/abs/2104.09864.

      Args:
          x: input tensor.
          axis: axis to add the positional encodings.

      Returns:
          The input tensor with RoPE encodings.
    """
    shape = x.shape.as_list()

    if isinstance(axis, int):
        axis = [axis]

    if isinstance(shape, (list, tuple)):
        spatial_shape = [shape[i] for i in axis]
        total_len = 1
        for i in spatial_shape:
            total_len *= i
        position = tf.reshape(
            tf.cast(tf.range(total_len, delta=1.0), tf.float32), spatial_shape)
    else:
        raise ValueError(f'Unsupported shape: {shape}')

    # we assume that the axis can not be negative (e.g., -1)
    if any(dim < 0 for dim in axis):
        raise ValueError(f'Unsupported axis: {axis}')
    for i in range(axis[-1] + 1, len(shape) - 1, 1):
        position = tf.expand_dims(position, axis=-1)

    half_size = shape[-1] // 2
    freq_seq = tf.cast(tf.range(half_size), tf.float32) / float(half_size)
    inv_freq = 10000**-freq_seq
    sinusoid = tf.einsum('...,d->...d', position, inv_freq)
    sin = tf.cast(tf.sin(sinusoid), dtype=x.dtype)
    cos = tf.cast(tf.cos(sinusoid), dtype=x.dtype)
    x1, x2 = tf.split(x, 2, axis=-1)
    return tf.concat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def toeplitz_matrix_rope(n: int, a: Tensor, b: Tensor) -> Tensor:
    """Obtain Toeplitz matrix using rope."""
    a = rope(tf.tile(a[None, :], [n, 1]), axis=0)
    b = rope(tf.tile(b[None, :], [n, 1]), axis=0)
    return tf.einsum("mk,nk->mn", a, b)


class GAU(layers.Layer):
    """Gated Attention Unit layer introduced in Transformer
    Quality in Linear Time.

    Paper reference: https://arxiv.org/abs/2202.10447
    """

    def __init__(self,
                 dim: int,
                 max_len: int = 128,
                 shared_dim: int = 128,
                 expansion_factor: int = 2,
                 activation: str = 'swish',
                 attention_activation: str = 'sqrrelu',
                 dropout_rate: float = 0.0,
                 attention_dropout_rate: float = 0.0,
                 spatial_dropout_rate: float = 0.0,
                 **kwargs) -> None:
        """
        Initialize a GAU layer.

        Args:
            dim: Dimension of GAU block.

            max_len: Maximum seq len of input.

            shared_dim: Size of shared dim. Defaults to 128.

            expansion_factor: Hidden dim expansion factor. Defaults to 2.

            activation: Activation to use in projection layers. Defaults
                to 'swish'.

            attention_activation: Activation to use on attention scores.
                Defaults to 'sqrrelu'.

            dropout_rate: Feature dropout rate. Defaults to 0.0.

            attention_dropout_rate: Feature dropout rate after attention.
                Defaults to 0.0

            spatial_dropout_rate: Spatial dropout rate. Defaults to 0.0.
        """
        super().__init__(**kwargs)

        self.dim = dim
        self.max_len = max_len
        self.shared_dim = shared_dim
        self.expansion_factor = expansion_factor
        self.activation = activation
        self.attention_activation: str = attention_activation
        self.dropout_rate = dropout_rate
        self.spatial_dropout_rate = spatial_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        # compute projection dimension
        self.expand_dim = self.dim * self.expansion_factor
        self.proj_dim = 2 * self.expand_dim + self.shared_dim

        # define layers
        self.norm = layers.LayerNormalization()
        self.proj1 = layers.Dense(self.proj_dim,
                                  use_bias=True,
                                  activation=self.activation)
        self.proj2 = layers.Dense(self.dim, use_bias=True)

        # dropout layers
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)

        if self.attention_dropout_rate:
            self.attention_dropout = layers.Dropout(self.attention_dropout_rate)

        if self.spatial_dropout_rate:
            self.spatial_dropout = layers.SpatialDropout1D(
                self.spatial_dropout_rate)

        # attention activation function
        self.attention_activation_layer = tf.keras.layers.Activation(
            self.attention_activation)

        # setting up position encoding
        self.a = tf.Variable(lambda: self.WEIGHT_INITIALIZER(
            shape=[self.max_len], dtype=tf.float32))
        self.b = tf.Variable(lambda: self.WEIGHT_INITIALIZER(
            shape=[self.max_len], dtype=tf.float32))

        # offset scaling values
        self.gamma = tf.Variable(lambda: self.WEIGHT_INITIALIZER(
            shape=[2, self.shared_dim], dtype=tf.float32))

        self.beta = tf.Variable(lambda: self.ZEROS_INITIALIZER(
            shape=[2, self.shared_dim], dtype=tf.float32))

    def call(self, x, training=False):

        shortcut = x
        x = self.norm(x)

        # input dropout
        if self.spatial_dropout_rate:
            x = self.spatial_dropout(x, training=training)

        x = self.dropout1(x, training=training)

        # initial projection to generate uv
        uv = self.proj1(x)
        uv = self.dropout2(uv, training=training)

        u, v, base = tf.split(
            uv, [self.expand_dim, self.expand_dim, self.shared_dim], axis=-1)

        # generate q, k by scaled offset
        base = tf.einsum('bnr,hr->bnhr', base, self.gamma) + self.beta
        q, k = tf.unstack(base, axis=-2)

        # compute key-query scores
        qk = tf.einsum('bnd,bmd->bnm', q, k)
        qk = qk / self.max_len

        # add relative position bias for attention
        qk += toeplitz_matrix_rope(self.max_len, self.a, self.b)

        # apply attention activation
        kernel = self.attention_activation_layer(qk)

        if self.attention_dropout_rate:
            kernel = self.attention_dropout(kernel)

        # apply values and project
        x = u * tf.einsum('bnm,bme->bne', kernel, v)

        x = self.proj2(x)
        return x + shortcut

    def get_config(self):
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'max_len': self.max_len,
            'shared_dim': self.shared_dim,
            'expansion_factor': self.expansion_factor,
            'activation': self.activation,
            'attention_activation': self.attention_activation,
            'dropout_rate': self.dropout_rate,
            'spatial_dropout_rate': self.spatial_dropout_rate
        })
        return config

    @property
    def WEIGHT_INITIALIZER(self):
        return clone_initializer(tf.random_normal_initializer(stddev=0.02))

    @property
    def ZEROS_INITIALIZER(self):
        return clone_initializer(tf.initializers.zeros())


class StopGradient(keras.layers.Layer):

    def __init__(self, stop_gradient: bool = False, **kwargs):
        """Stop gradient, or not, depending on the configuration.

        Args:

            stop_gradient (bool): If `True` then this layer stops gradient,
            otherwise it is a no-op. Defaults to `False`.

           **kwargs: Additional arguments for keras.layers.Layer.__init__.
        """
        super().__init__(**kwargs)
        self._stop_gradient = stop_gradient

    def call(self, inputs):
        if self._stop_gradient:
            # Stopping gradient.
            return keras.ops.stop_gradient(inputs)

        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "stop_gradient": self._stop_gradient,
        })
        return config


def _make_head(x, heads, name, relations, dim):
    """Make a single head."""
    activation: str = "swish"
    dense_dropout: float = 0.05

    head = x

    # Construction relations layers if needed
    if relations:
        related_outputs = []
        for rname in relations:
            related_outputs.append(
                StopGradient(stop_gradient=True)(heads[rname]))
        related_outputs.append(x)
        head = layers.Concatenate(name=f"{name}_relations")(related_outputs)
        for _ in range(3):
            head = layers.Dense(256)(head)
            head = layers.Activation(activation)(head)

    head = layers.Dropout(dense_dropout, name=f"{name}_dropout")(head)

    # Dense block
    block_name = f"{name}_dense_1"
    head = layers.Dense(dim)(head)
    head = layers.Dropout(dense_dropout)(head)
    head = layers.Activation(activation)(head)

    # Prediction
    return layers.Dense(dim, activation='softmax', name=name)(head)


def get_dag(outputs: Dict[str, Dict],
            output_relations: List[Tuple[str, str]]) -> nx.DiGraph:
    """Return graph of output relation dependencies.

    Both outputs and output_relations are needed to have even the outputs which
    are not a part of any relation.

    Args:
      outputs (Dict[str, Dict]): Description of outputs as returned by
        scaaml.io.Dataset.as_tfdataset.
      output_relations (List[Tuple[str, str]]): List of arcs (oriented edges)
        attack point name (full -- with the index) which is required for the
        second one. When (ap_1, ap_2) is present the interpretation is that
        ap_2 depends on the value of ap_1.

    Returns: A networkx.DiGraph representation of relations.
    """
    # Create graph of relations that will be topologically sorted and contains
    # all head names.
    relation_graph = nx.DiGraph()
    # Add all output names into the relation_graph (even if they appear in no
    # relations).
    for name in outputs:
        relation_graph.add_node(name)
    # Add all relation edges.
    for ap_1, ap_2 in output_relations:
        # When ap_2 depends on ap_1 then ap_1 must be created before ap_2.
        relation_graph.add_edge(ap_1, ap_2)

    return relation_graph


def get_topological_order(outputs: Dict[str, Dict],
                          output_relations: List[Tuple[str, str]]):
    """Return iterator of vertices in topological order (if attack point ap_2
    depends on ap_1 then ap_1 appears before ap_2).

    Both outputs and output_relations are needed to have even the outputs which
    are not a part of any relation.

    Args:
      outputs (Dict[str, Dict]): Description of outputs as returned by
        scaaml.io.Dataset.as_tfdataset.
      output_relations (List[Tuple[str, str]]): List of arcs (oriented edges)
        attack point name (full -- with the index) which is required for the
        second one. When (ap_1, ap_2) is present the interpretation is that
        ap_2 depends on the value of ap_1.
    """
    return nx.topological_sort(
        get_dag(outputs=outputs, output_relations=output_relations))


def create_heads_outputs(x: Tensor, outputs: Dict[str, Dict],
                         output_relations: List[Tuple[str, str]]) -> List:
    """Make a list of all heads.

    Args:
      x (FloatTensor): The trunk.
      outputs (Dict[str, Dict]): Description of outputs as returned by
        scaaml.io.Dataset.as_tfdataset.
      output_relations (List[Tuple[str, str]]): List of arcs (oriented edges)
        attack point name (full -- with the index) which is required for the
        second one. When (ap_1, ap_2) is present the interpretation is that
        ap_2 depends on the value of ap_1.

    Returns: A list of all head outputs.
    """
    # Create relations represented by lists of ingoing edges (attack points:
    # list of all attack points it depends on).
    ingoing_relations = defaultdict(list)
    for ap_1, ap_2 in output_relations:
        ingoing_relations[ap_2].append(ap_1)
    # Freeze the dict
    ingoing_relations = dict(ingoing_relations)

    # Dictionary containing the actual network heads
    heads = {}

    # Get iterator of outputs that are in topological order (if ap_2 depends on
    # ap_1 then ap_1 appears before ap_2).
    topological_order = get_topological_order(outputs=outputs,
                                              output_relations=output_relations)

    # Create heads.
    for name in topological_order:
        # Get relations (possibly an empty list).
        relations = ingoing_relations.get(name, [])

        # Get parameters for head creation.
        dim = outputs[name]['max_val'] if outputs[name]['max_val'] > 2 else 1
        head = _make_head(x, heads, name, relations, dim)
        heads[name] = head

    # Return all head outputs in a list.
    heads_outputs = [heads[name] for name in outputs.keys()]
    return heads_outputs


def get_model(inputs, outputs, output_relations, trace_len: int,
              merge_filter_1: int, merge_filter_2: int, patch_size: int,
              target_lr: float):
    # Constants:
    steps: int = trace_len // patch_size
    combine_kernel_size: int = 3
    activation: str = "swish"
    combine_strides: int = 1
    filters: int = 192

    # Input
    input = layers.Input(shape=(trace_len,), name='trace1')
    x = input

    # Reshape the trace.
    x = layers.Reshape((steps, patch_size))(x)
    # Rescale to the interval [-1, 1].
    x = 2 * ((x - inputs["trace1"]["min"]) / inputs["trace1"]["delta"]) - 1

    # Single dense after preprocess.
    x = layers.Dense(filters)(x)

    # Dropout
    x = layers.SpatialDropout1D(0.1)(x)

    # Transformer layers (with intermediate results).
    s = x
    gau_results = []  # Intermediate results (after 1st, 2nd...).
    for _ in range(3):
        s = GAU(
            dim=filters,
            max_len=steps,
            expansion_factor=2,
            attention_activation="softsign",
        )(s)
        gau_results.append(s)
    x = layers.Concatenate()(gau_results)

    # Norm after concatenate
    x = layers.BatchNormalization()(x)

    # merge blocks
    if merge_filter_1:
        x = layers.Conv1D(merge_filter_1,
                          combine_kernel_size,
                          activation=activation,
                          strides=combine_strides)(x)
        # MaxPool1D if applicable
        x = layers.MaxPool1D(pool_size=2)(x)
        # Second merge block
        if merge_filter_2:
            x = layers.BatchNormalization()(x)
            x = layers.Conv1D(merge_filter_2,
                              combine_kernel_size,
                              activation=activation,
                              strides=combine_strides)(x)

    # post merge dropouts
    x = layers.Dropout(0.1)(x)

    # flattening
    x = layers.GlobalAveragePooling1D(data_format='channels_first')(x)

    # Normalizing
    x = layers.BatchNormalization()(x)

    # Make head outputs
    heads_outputs = create_heads_outputs(
        x=x,
        outputs=outputs,
        output_relations=output_relations,
    )

    model = Model(input, heads_outputs)

    # Compile model
    optimizer = keras.optimizers.Adafactor(target_lr)
    model.compile(
        optimizer,
        loss=["categorical_crossentropy" for _ in range(len(outputs))],
        metrics={name: ["acc", MeanRank()] for name in outputs},
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train GPAM model for ECC CM1")
    parser.add_argument(
        "--dataset_path",
        "-d",
        help=
        "Dataset path, download info: https://github.com/google/scaaml/tree/main/papers/2024/GPAM",
        required=True)
    args = parser.parse_args()

    # Block of hyperparameters.
    # Length of the trace. For technical reasons patch_size should divide trace_len.
    batch_size: int = 64  # hyperparameter
    steps_per_epoch: int = 200  # hyperparameter
    epochs: int = 500  # hyperparameter
    target_lr: float = 0.006  # hyperparameter
    merge_filter_1: int = 16  # hyperparameter
    merge_filter_2: int = 8  # hyperparameter
    trace_len: int = 4_194_304  # hyperparameter
    patch_size: int = 2_048  # hyperparameter

    # Definition of outputs.
    attack_points = [
        {
            "name": "k",
            "index": 0,
            "type": "byte"
        },
        {
            "name": "km",
            "index": 0,
            "type": "byte"
        },
        {
            "name": "r",
            "index": 0,
            "type": "byte"
        },
    ]
    # Configuration driven definition of relational outputs, as described in
    # Section 5.2.3 point 2.
    output_relations = [
        ["km_0", "k_0"],
        ["r_0", "k_0"],
    ]

    traces = ["trace1"]
    trace_start: int = 0
    shuffle_size: int = 512
    val_steps: int = 16

    # loading dataset
    train_ds, inputs, outputs = Dataset.as_tfdataset(
        dataset_path=args.dataset_path,
        split="train",
        attack_points=attack_points,
        traces=traces,
        trace_start=trace_start,
        trace_len=trace_len,
        batch_size=batch_size,
        shuffle=shuffle_size,
    )
    test_ds, _, _ = Dataset.as_tfdataset(
        dataset_path=args.dataset_path,
        split="test",
        attack_points=attack_points,
        traces=traces,
        trace_start=trace_start,
        trace_len=trace_len,
        batch_size=batch_size,
        shuffle=0,
    )

    model = get_model(
        inputs=inputs,
        outputs=outputs,
        output_relations=output_relations,
        trace_len=trace_len,
        merge_filter_1=merge_filter_1,
        merge_filter_2=merge_filter_2,
        patch_size=patch_size,
        target_lr=target_lr,
    )

    model.summary()

    # Train the model.
    history = model.fit(train_ds,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        validation_data=test_ds,
                        validation_steps=val_steps)


if __name__ == "__main__":
    main()
