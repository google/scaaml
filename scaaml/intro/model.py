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
"""Intro model."""

from typing import Any, Dict, Tuple

from tensorflow import Tensor
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

from scaaml.utils import display_config
from scaaml.utils import get_num_gpu


# pylint: disable=too-many-positional-arguments
def block(x: Tensor,
          filters: int,
          kernel_size: int = 3,
          strides: int = 1,
          conv_shortcut: bool = False,
          activation: str = "relu") -> Tensor:
    """Residual block with pre-activation
    From: https://arxiv.org/pdf/1603.05027.pdf

    Args:
        x: input tensor.
        filters (int): filters of the bottleneck layer.

        kernel_size(int, optional): kernel size of the bottleneck layer.
        defaults to 3.

        strides (int, optional): stride of the first layer.
        defaults to 1.

        conv_shortcut (bool, optional): Use convolution shortcut if True,
        otherwise identity shortcut. Defaults to False.

        use_batchnorm (bool, optional): Use batchnormalization if True.
        Defaults to True.

        activation (str, optional): activation function. Defaults to "relu".

    Returns:
        Output tensor for the residual block.
    """

    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    if conv_shortcut:
        shortcut = layers.Conv1D(4 * filters, 1, strides=strides)(x)
    else:
        if strides > 1:
            shortcut = layers.MaxPooling1D(1, strides=strides)(x)
        else:
            shortcut = x

    x = layers.Conv1D(filters, 1, use_bias=False, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    x = layers.Conv1D(filters,
                      kernel_size,
                      strides=strides,
                      use_bias=False,
                      padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    x = layers.Conv1D(4 * filters, 1)(x)
    x = layers.Add()([shortcut, x])
    return x


# pylint: disable=too-many-positional-arguments
def stack(x: Tensor,
          filters: int,
          blocks: int,
          kernel_size: int = 3,
          strides: int = 2,
          activation: str = "relu") -> Tensor:
    """A set of stacked residual blocks.
    Args:
        filters (int): filters of the bottleneck layer.

        blocks (int): number of conv blocks to stack.

        kernel_size(int, optional): kernel size of the bottleneck layer.
        defaults to 3.

        strides (int, optional): stride used in the last block.
        defaults to 2.

        conv_shortcut (bool, optional): Use convolution shortcut if True,
        otherwise identity shortcut. Defaults to False.

        activation (str, optional): activation function. Defaults to "relu".

    Returns:
        tensor:Output tensor for the stacked blocks.
  """
    x = block(x,
              filters,
              kernel_size=kernel_size,
              activation=activation,
              conv_shortcut=True)
    for _ in range(2, blocks):
        x = block(x, filters, kernel_size=kernel_size, activation=activation)
    x = block(x, filters, strides=strides, activation=activation)
    return x


# pylint: disable=C0103
def Resnet1D(input_shape: Tuple[int, ...], attack_point: str,
             mdl_cfg: Dict[str, Any], optim_cfg: Dict[str,
                                                      Any]) -> Model[Any, Any]:
    del attack_point  # unused

    pool_size = mdl_cfg["initial_pool_size"]
    filters = mdl_cfg["initial_filters"]
    block_kernel_size = mdl_cfg["block_kernel_size"]
    activation = mdl_cfg["activation"]
    dense_dropout = mdl_cfg["dense_dropout"]
    num_blocks = [
        mdl_cfg["blocks_stack1"], mdl_cfg["blocks_stack2"],
        mdl_cfg["blocks_stack3"], mdl_cfg["blocks_stack4"]
    ]

    inputs = layers.Input(shape=input_shape)
    x = inputs

    # stem
    x = layers.MaxPool1D(pool_size=pool_size)(x)

    # trunk: stack of residual block
    for block_idx in range(4):
        filters *= 2
        x = stack(x,
                  filters,
                  num_blocks[block_idx],
                  kernel_size=block_kernel_size,
                  activation=activation)

    # head model: dense
    x = layers.GlobalAveragePooling1D()(x)
    for _ in range(1):
        x = layers.Dropout(dense_dropout)(x)
        x = layers.Dense(256)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)

    outputs = layers.Dense(256, activation="softmax")(x)

    model: Model[Any, Any] = Model(inputs=inputs, outputs=outputs)
    model.summary()

    if get_num_gpu() > 1:
        lr = optim_cfg["multi_gpu_lr"]
    else:
        lr = optim_cfg["lr"]

    model.compile(loss=["categorical_crossentropy"],
                  metrics=["acc"],
                  optimizer=Adam(lr))
    return model


def get_model(input_shape: Tuple[int, ...], attack_point: str,
              config: Dict[str, Any]) -> Model[Any, Any]:
    """Return an instantiated model based of the config provided.

    Args:
        config (dict): scald config.
    """

    mdl_cfg = config["model_parameters"]
    optim_cfg = config["optimizer_parameters"]

    display_config("model", mdl_cfg)
    display_config("optimizer", optim_cfg)
    return Resnet1D(input_shape, attack_point, mdl_cfg, optim_cfg)
