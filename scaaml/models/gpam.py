# Copyright 2025 Google LLC
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
"""This is the GPAM model version which can be imported. For the archived
version see /papers/2024/GPAM/gpam_ecc_cm1.py.

GPAM model, see https://github.com/google/scaaml/tree/main/papers/2024/GPAM

@article{bursztein2023generic,
  title={Generalized Power Attacks against Crypto Hardware using Long-Range
  Deep Learning},
  author={Bursztein, Elie and Invernizzi, Luca and Kr{\'a}l, Karel and Moghimi,
  Daniel and Picod, Jean-Michel and Zhang, Marina},
  journal={arXiv preprint arXiv:2306.07249},
  year={2023}
}
"""

from collections import defaultdict
from typing import Any, Union

# NetworkX is an optional dependency.
try:
    import networkx as nx
except ImportError:
    nx = None  # type: ignore[assignment]
import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow import Tensor


@keras.saving.register_keras_serializable()
class Rescale(layers.Layer):  # type: ignore[type-arg]
    """Rescale input to the interval [-1, 1].
    """

    def __init__(self, trace_min: float, trace_delta: float,
                 **kwargs: Any) -> None:
        """Information for trace rescaling.

        Args:

          trace_min (float): Minimum over all traces.

          trace_delta (float): Maximum over all traces minus `trace_min`.
        """
        super().__init__(**kwargs)
        self.trace_min: float = trace_min
        self.trace_delta: float = trace_delta

    def call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        """Rescale to the interval [-1, 1]."""
        del kwargs  # unused
        x = inputs
        x = 2 * ((x - self.trace_min) / self.trace_delta) - 1
        return x

    def get_config(self) -> dict[str, Any]:
        """Return the config to allow saving and loading of the model.
        """
        config = super().get_config()
        config.update({
            "trace_min": self.trace_min,
            "trace_delta": self.trace_delta,
        })
        return config


@keras.saving.register_keras_serializable()
class ScaledNorm(layers.Layer):  # type: ignore[type-arg]
    """ScaledNorm layer.

    Transformers without Tears: Improving the Normalization of Self-Attention
    Toan Q. Nguyen, Julian Salazar
    https://arxiv.org/abs/1910.05895
    """

    def __init__(self,
                 begin_axis: int = -1,
                 epsilon: float = 1e-5,
                 **kwargs: Any) -> None:
        """Initialize a ScaledNorm Layer.

        Args:

            begin_axis (int): Axis along which to apply norm. Defaults to -1.

            epsilon (float): Norm epsilon value. Defaults to 1e-5.
        """
        super().__init__(**kwargs)
        self._begin_axis = begin_axis
        self._epsilon = epsilon
        self._scale = self.add_weight(
            name="norm_scale",
            shape=(),
            initializer=tf.constant_initializer(value=1.0),
            trainable=True,
        )

    def call(self, inputs: Tensor) -> Tensor:
        """Return the output of this layer.
        """
        x = inputs
        axes = list(range(len(x.shape)))[self._begin_axis:]
        mean_square = tf.reduce_mean(tf.math.square(x), axes, keepdims=True)
        x = x * tf.math.rsqrt(mean_square + self._epsilon)
        return x * self._scale

    def get_config(self) -> dict[str, Any]:
        """Return the config to allow saving and loading of the model.
        """
        config = super().get_config()
        config.update({
            "begin_axis": self._begin_axis,
            "epsilon": self._epsilon
        })
        return config


def clone_initializer(initializer: tf.keras.initializers.Initializer) -> Any:
    """Clone an initializer (if an initializer is reused the generated
    weights are the same).
    """
    if isinstance(initializer, tf.keras.initializers.Initializer):
        return initializer.__class__.from_config(initializer.get_config())
    return initializer  # type: ignore[unreachable]


def rope(
    x: Tensor,
    axis: Union[list[int], int],
) -> Tensor:
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
            total_len *= i  # type: ignore[operator]
        position = tf.reshape(
            tf.cast(tf.range(total_len, delta=1.0), tf.float32), spatial_shape)
    else:
        raise ValueError(f"Unsupported shape: {shape}")

    # we assume that the axis can not be negative (e.g., -1)
    if any(dim < 0 for dim in axis):
        raise ValueError(f"Unsupported axis: {axis}")
    for i in range(axis[-1] + 1, len(shape) - 1, 1):
        position = tf.expand_dims(position, axis=-1)

    half_size = shape[-1] // 2  # type: ignore[operator]
    freq_seq = tf.cast(tf.range(half_size), tf.float32) / float(half_size)
    inv_freq = 10000**-freq_seq
    sinusoid = tf.einsum("...,d->...d", position, inv_freq)
    sin = tf.cast(tf.sin(sinusoid), dtype=x.dtype)
    cos = tf.cast(tf.cos(sinusoid), dtype=x.dtype)
    x1, x2 = tf.split(x, 2, axis=-1)
    return tf.concat(  # type: ignore[no-any-return]
        [x1 * cos - x2 * sin, x2 * cos + x1 * sin],
        axis=-1,
    )


def toeplitz_matrix_rope(
    n: int,
    a: Tensor,
    b: Tensor,
) -> Tensor:
    """Obtain Toeplitz matrix using rope."""
    a = rope(tf.tile(a[None, :], [n, 1]), axis=[0])
    b = rope(tf.tile(b[None, :], [n, 1]), axis=[0])
    return tf.einsum("mk,nk->mn", a, b)  # type: ignore[no-any-return]


@keras.saving.register_keras_serializable()
class GAU(layers.Layer):  # type: ignore[type-arg]
    """Gated Attention Unit layer introduced in Transformer
    Quality in Linear Time.

    Paper reference: https://arxiv.org/abs/2202.10447
    """

    def __init__(
            self,
            *,  # key-word only arguments
            dim: int,
            max_len: int = 128,
            shared_dim: int = 128,
            expansion_factor: int = 2,
            activation: str = "swish",
            attention_activation: str = "sqrrelu",
            dropout_rate: float = 0.0,
            attention_dropout_rate: float = 0.0,
            spatial_dropout_rate: float = 0.0,
            **kwargs: Any) -> None:
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
        self.proj1 = layers.Dense(
            self.proj_dim,
            use_bias=True,
            activation=self.activation,
        )
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

    def build(self, input_shape: tuple[int, ...]) -> None:
        del input_shape  # unused

        # setting up position encoding
        self.a = self.add_weight(
            name="a",
            shape=(self.max_len,),
            initializer=lambda *args, **kwargs: self.weight_initializer(
                shape=[self.max_len]),
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(self.max_len,),
            initializer=lambda *args, **kwargs: self.weight_initializer(
                shape=[self.max_len]),
            trainable=True,
        )

        # offset scaling values
        self.gamma = self.add_weight(
            name="gamma",
            shape=(2, self.shared_dim),
            initializer=lambda *args, **kwargs: self.weight_initializer(
                shape=[2, self.shared_dim]),
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(2, self.shared_dim),
            initializer=lambda *args, **kwargs: self.zeros_initializer(
                shape=[2, self.shared_dim]),
            trainable=True,
        )

    def call(self, x: Any, training: bool = False) -> Any:

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
        base = tf.einsum("bnr,hr->bnhr", base, self.gamma) + self.beta
        q, k = tf.unstack(base, axis=-2)

        # compute key-query scores
        qk = tf.einsum("bnd,bmd->bnm", q, k)
        qk = qk / self.max_len

        # add relative position bias for attention
        qk += toeplitz_matrix_rope(self.max_len, self.a, self.b)

        # apply attention activation
        kernel = self.attention_activation_layer(qk)

        if self.attention_dropout_rate:
            kernel = self.attention_dropout(kernel)

        # apply values and project
        x = u * tf.einsum("bnm,bme->bne", kernel, v)

        x = self.proj2(x)
        return x + shortcut

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "max_len": self.max_len,
            "shared_dim": self.shared_dim,
            "expansion_factor": self.expansion_factor,
            "activation": self.activation,
            "attention_activation": self.attention_activation,
            "dropout_rate": self.dropout_rate,
            "spatial_dropout_rate": self.spatial_dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
        })
        return config

    @property
    def weight_initializer(self) -> Any:
        return clone_initializer(tf.random_normal_initializer(stddev=0.02))

    @property
    def zeros_initializer(self) -> Any:
        return clone_initializer(tf.initializers.zeros())


@keras.saving.register_keras_serializable()
class StopGradient(
        keras.layers.Layer,  # type: ignore[misc,no-any-unimported]
):
    """Stop gradient as a Keras layer.
    """

    def __init__(
        self,
        stop_gradient: bool = False,
        **kwargs: Any,
    ) -> None:
        """Stop gradient, or not, depending on the configuration.

        Args:

            stop_gradient (bool): If `True` then this layer stops gradient,
            otherwise it is a no-op. Defaults to `False`.

           **kwargs: Additional arguments for keras.layers.Layer.__init__.
        """
        super().__init__(**kwargs)
        self._stop_gradient = stop_gradient

    def call(self, inputs):  # type: ignore[no-untyped-def]
        if self._stop_gradient:
            # Stopping gradient.
            return keras.ops.stop_gradient(inputs)

        return inputs

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({
            "stop_gradient": self._stop_gradient,
        })
        return config  # type: ignore[no-any-return]


def _make_head(  # type: ignore[no-any-unimported]
    x: keras.layers.Layer,
    heads: dict[str, keras.layers.Layer],
    name: str,
    relations: list[str],
    dim: int,
) -> keras.layers.Layer:
    """Make a single head.

    Args:

      x (Tensor): Stem of the neural network.

      heads (dict[str, keras.layers.Layer]): A dictionary of previous heads
      (those that are sooner in the topologically sorted outputs).

      name (str): Name of this output.

      relations (list[str]): Which outputs should be routed to this one. All of
      these must be already constructed and present in `heads`.

      dim (int): Number of classes of this output.
    """
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
    head = layers.Dense(dim, activation=activation)(head)
    head = layers.Dense(dim, activation=activation)(head)
    head = layers.Dense(dim, activation=activation)(head)
    head = layers.Dropout(dense_dropout)(head)
    head = layers.Dense(dim, activation=activation)(head)

    # Prediction
    return layers.Dense(dim, activation="softmax", name=name)(head)


def get_dag(
    outputs: dict[str, dict[str, int]],
    output_relations: list[tuple[str, str]],
) -> Any:
    """Return graph of output relation dependencies.

    Both `outputs` and `output_relations` are needed to have even the outputs
    which are not a part of any relation.

    Args:
      outputs (dict[str, dict]): Description of outputs as returned by
        scaaml.io.Dataset.as_tfdataset.
      output_relations (list[tuple[str, str]]): List of arcs (oriented edges)
        attack point name (full -- with the index) which is required for the
        second one. When `(ap_1, ap_2)` is present the interpretation is that
        `ap_2` depends on the value of `ap_1`.

    Returns: A networkx.DiGraph representation of relations.
    """
    # Create graph of relations that will be topologically sorted and contains
    # all head names.
    relation_graph: nx.DiGraph[str]  # pylint: disable=unsubscriptable-object
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


def get_topological_order(
    outputs: dict[str, dict[str, int]],
    output_relations: list[tuple[str, str]],
) -> list[str]:
    """Return iterator of vertices in topological order (if attack point ap_2
    depends on ap_1 then ap_1 appears before ap_2).

    Both outputs and output_relations are needed to have even the outputs which
    are not a part of any relation.

    Args:

      outputs (dict[str, dict[str, int]]): Description of outputs as returned
      by scaaml.io.Dataset.as_tfdataset.

      output_relations (list[tuple[str, str]]): List of arcs (oriented edges)
      attack point name (full -- with the index) which is required for the
      second one. When (ap_1, ap_2) is present the interpretation is that ap_2
      depends on the value of ap_1.

    """
    if output_relations:
        if nx is None:
            raise ImportError("To use the relational heads please install "
                              "networkx[default]")

        # We need to create the heads in a topological order.
        return nx.topological_sort(  # type: ignore[return-value]
            get_dag(outputs=outputs, output_relations=output_relations))
    else:
        return list(outputs)


def create_heads_outputs(  # type: ignore[no-any-unimported]
    x: Tensor,
    outputs: dict[str, dict[str, int]],
    output_relations: list[tuple[str, str]],
) -> dict[str, keras.layers.Layer]:
    """Make a mapping of all heads (name to Layer).

    Args:

      x (FloatTensor): The trunk.

      outputs (dict[str, dict[str, int]]): Description of outputs as returned
      by scaaml.io.Dataset.as_tfdataset.

      output_relations (list[tuple[str, str]]): List of arcs (oriented edges)
      attack point name (full -- with the index) which is required for the
      second one. When (ap_1, ap_2) is present the interpretation is that ap_2
      depends on the value of ap_1.

    Returns: A mapping of all head outputs (name to Layer).
    """
    # Create relations represented by lists of ingoing edges (attack points:
    # list of all attack points it depends on).
    ingoing_relations: dict[str, list[str]] = defaultdict(list)
    for ap_1, ap_2 in output_relations:
        ingoing_relations[ap_2].append(ap_1)
    # Freeze the dict
    ingoing_relations = dict(ingoing_relations)

    # Dictionary containing the actual network heads
    heads: dict[str, keras.layers.Layer] = {}  # type: ignore[no-any-unimported]

    # Get iterator of outputs that are in topological order (if ap_2 depends on
    # ap_1 then ap_1 appears before ap_2).
    topological_order = get_topological_order(
        outputs=outputs,
        output_relations=output_relations,
    )

    # Create heads.
    for name in topological_order:
        # Get relations (possibly an empty list).
        relations = ingoing_relations.get(name, [])

        # Get parameters for head creation.
        dim = outputs[name]["max_val"] if outputs[name]["max_val"] > 2 else 1
        head = _make_head(x, heads, name, relations, dim)
        heads[name] = head

    # Return all head outputs in a dict.
    heads_outputs = {name: heads[name] for name in outputs.keys()}
    return heads_outputs


def get_gpam_model(  # type: ignore[no-any-unimported]
    *,  # key-word only arguments
    inputs: dict[str, dict[str, float]],
    outputs: dict[str, dict[str, int]],
    output_relations: list[tuple[str, str]],
    trace_len: int,
    merge_filter_1: int,
    merge_filter_2: int,
    patch_size: int,
) -> keras.models.Model:
    """Get a GPAM model instance.

    Args:

      inputs (dict[str, dict[str, float]]): The following dictionary:
      {"trace1": {"min": MIN, "delta": MAX}} where `MIN` is the minimum value
      across all traces and time and `MAX` is the maximum value.

      outputs (dict[str, dict[str, int]]): A dictionary with output name and
      "max_val" being the number of possible classes. Example:
      `outputs={"sub_bytes_in_0": {"max_val": 256}}`.

      output_relations (list[tuple[str, str]]): A list of related inputs. Each
      relation is a list where the output of the first is fed to the second.
      Must form a directed acyclic graph.

      trace_len (int): The trace is assumed to be one-dimensional of length
      `trace_len`. Must be divisible by `patch_size`.

      merge_filter_1 (int): The number of filters in the first layer of
      convolutions.

      merge_filter_2 (int): The number of filters in the second layer of
      convolutions.

      patch_size (int): Cut the trace into patches of this length. Must divide
      `trace_len`.

    ```
    @article{bursztein2023generic,
      title={Generalized Power Attacks against Crypto Hardware using Long-Range
      Deep Learning},
      author={Bursztein, Elie and Invernizzi, Luca and Kr{\'a}l, Karel and
      Moghimi, Daniel and Picod, Jean-Michel and Zhang, Marina},
      journal={arXiv preprint arXiv:2306.07249},
      year={2023}
    }
    ```
    """
    # Constants:
    if trace_len % patch_size:
        raise ValueError(f"{trace_len = } is not divisible by {patch_size = }")
    steps: int = trace_len // patch_size
    combine_kernel_size: int = 3
    activation: str = "swish"
    combine_strides: int = 1
    filters: int = 192

    # Input
    model_input = layers.Input(shape=(trace_len,), name="trace1")
    x = model_input

    # Reshape the trace.
    x = layers.Reshape((steps, patch_size))(x)
    x = Rescale(  # to the interval [-1, 1].
        trace_min=inputs["trace1"]["min"],
        trace_delta=inputs["trace1"]["delta"],
    )(x)

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
            x = ScaledNorm()(x)
            x = layers.Conv1D(merge_filter_2,
                              combine_kernel_size,
                              activation=activation,
                              strides=combine_strides)(x)

    # post merge dropouts
    x = layers.Dropout(0.1)(x)

    # flattening
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)

    # Normalizing
    x = layers.BatchNormalization()(x)

    # Make head outputs
    heads_outputs = create_heads_outputs(
        x=x,
        outputs=outputs,
        output_relations=output_relations,
    )

    model = keras.models.Model(model_input, heads_outputs)
    return model
