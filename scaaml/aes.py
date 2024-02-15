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
"""Backward AES."""

import numpy as np
import numpy.typing as npt

# used for sub_bytes_out0
SBREV = np.array([
    82, 9, 106, 213, 48, 54, 165, 56, 191, 64, 163, 158, 129, 243, 215, 251,
    124, 227, 57, 130, 155, 47, 255, 135, 52, 142, 67, 68, 196, 222, 233, 203,
    84, 123, 148, 50, 166, 194, 35, 61, 238, 76, 149, 11, 66, 250, 195, 78, 8,
    46, 161, 102, 40, 217, 36, 178, 118, 91, 162, 73, 109, 139, 209, 37, 114,
    248, 246, 100, 134, 104, 152, 22, 212, 164, 92, 204, 93, 101, 182, 146, 108,
    112, 72, 80, 253, 237, 185, 218, 94, 21, 70, 87, 167, 141, 157, 132, 144,
    216, 171, 0, 140, 188, 211, 10, 247, 228, 88, 5, 184, 179, 69, 6, 208, 44,
    30, 143, 202, 63, 15, 2, 193, 175, 189, 3, 1, 19, 138, 107, 58, 145, 17, 65,
    79, 103, 220, 234, 151, 242, 207, 206, 240, 180, 230, 115, 150, 172, 116,
    34, 231, 173, 53, 133, 226, 249, 55, 232, 28, 117, 223, 110, 71, 241, 26,
    113, 29, 41, 197, 137, 111, 183, 98, 14, 170, 24, 190, 27, 252, 86, 62, 75,
    198, 210, 121, 32, 154, 219, 192, 254, 120, 205, 90, 244, 31, 221, 168, 51,
    136, 7, 199, 49, 177, 18, 16, 89, 39, 128, 236, 95, 96, 81, 127, 169, 25,
    181, 74, 13, 45, 229, 122, 159, 147, 201, 156, 239, 160, 224, 59, 77, 174,
    42, 245, 176, 200, 235, 187, 60, 131, 83, 153, 97, 23, 43, 4, 126, 186, 119,
    214, 38, 225, 105, 20, 99, 85, 33, 12, 125
])


def ap_preds_to_key_preds(preds: npt.NDArray[np.generic],
                          plaintexts: npt.NDArray[np.generic],
                          attack_point: str) -> npt.NDArray[np.generic]:
    "Convert attack points predictions to key byte prediction"

    preds = np.array(preds)

    if attack_point == "key":
        return preds

    elif attack_point == "sub_bytes_in":
        key_probas = []
        class_ids = np.arange(256)
        for idx, pred in enumerate(preds):
            kv = class_ids ^ plaintexts[idx]
            key_probas.append(np.take(pred, kv))
        return np.array(key_probas)

    elif attack_point == "sub_bytes_out":
        key_probas = []
        for idx, pred in enumerate(preds):
            # ! danger zone. change at your own peril, Here be dragons.
            kidxs = SBREV ^ plaintexts[idx]
            probas = np.zeros(256)
            for pidx, kidx in enumerate(kidxs):
                probas[kidx] = pred[pidx]
            key_probas.append(probas)

        return np.array(key_probas)
    else:
        raise ValueError("Invalid attack point")
