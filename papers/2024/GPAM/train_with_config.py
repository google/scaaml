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
"""Deterministic training for main GPAM results.

Generalized Power Attacks against Crypto Hardware using Long-Range Deep Learning

Example use:
```bash
python train_with_config.py \
	--dataset_path K82F_ECC_CM2_ECC-FR256_CW308/ \
	--config configurations/ECC_CM2_blackbox.json \
	--result_file configurations/ECC_CM2_blackbox_result.json
```
"""
import argparse
import json
from pathlib import Path
from typing import Any
import random
import time

import keras
import numpy as np
import tensorflow as tf

import scaaml
from scaaml.io import Dataset
from scaaml.metrics.custom import MeanRank
from scaaml.models import get_gpam_model
from sedpack.io.types import SplitT


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic training of the GPAM model")
    parser.add_argument(
        "--dataset_path",
        "-d",
        help="Where to save the dataset",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--config",
        "-c",
        help="The config file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--result_file",
        "-r",
        help="Where to write the result (standad output if not picked)",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--seed",
        "-s",
        help="Random seed (defaults to 42, negative uses random)",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    # https://keras.io/examples/keras_recipes/reproducibility_recipes/
    seed: int = args.seed
    if seed < 0:
        seed = random.randint(0, 1_000_000)
    # Set the seed using keras.utils.set_random_seed. This will set:
    # 1) `numpy` seed
    # 2) backend random seed
    # 3) `python` random seed
    keras.utils.set_random_seed(seed)
    # If using TensorFlow, this will make GPU ops as deterministic as possible,
    # but it will affect the overall performance, so be mindful of that.
    tf.config.experimental.enable_op_determinism()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Loading the dataset
    train_ds, inputs, outputs = Dataset.as_tfdataset(
        dataset_path=args.dataset_path,
        split="train",
        attack_points=config["attack_points"],
        traces=["trace1"],
        trace_start=config["trace_start"],
        trace_len=config["trace_len"],
        batch_size=config["batch_size"],
        shuffle=config["shuffle_size"],
    )
    test_ds, _, _ = Dataset.as_tfdataset(
        dataset_path=args.dataset_path,
        split="test",
        attack_points=config["attack_points"],
        traces=["trace1"],
        trace_start=config["trace_start"],
        trace_len=config["trace_len"],
        batch_size=config["batch_size"],
        shuffle=config["shuffle_size"],
    )

    model = get_gpam_model(
        inputs=inputs,
        outputs=outputs,
        output_relations=config["output_relations"],
        trace_len=config["trace_len"],
        merge_filter_1=config["merge_filter_1"],
        merge_filter_2=config["merge_filter_2"],
        patch_size=config["patch_size"],
    )
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adafactor(config["target_lr"]),
        loss=["categorical_crossentropy" for _ in range(len(outputs))],
        metrics={name: ["acc", MeanRank()] for name in outputs},
    )
    model.summary()

    train_start = time.time()

    # Train the model.
    history = model.fit(
        train_ds,
        steps_per_epoch=config["steps_per_epoch"],
        epochs=config["epochs"],
        validation_data=test_ds,
        validation_steps=config["val_steps"],
    )

    train_duration = time.time() - train_start

    # Write the result.
    result = {
        "random_seed": seed,
        "keras_version": keras.__version__,
        "tensorflow_version": tf.__version__,
        "scaaml_version": scaaml.__version__,
        "history": history.history,
        "train_duration [s]": train_duration,
    }
    if args.result_file:
        with open(args.result_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
