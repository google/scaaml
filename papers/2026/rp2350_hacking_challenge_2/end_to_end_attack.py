#!/usr/bin/env python3
# Copyright 2026 Google LLC
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

"""Standalone End-to-End RP2350 AES-256-CTR Key Recovery Pipeline.

Self-contained implementation of:
1. Blind `trace1 -> trace2` POI Reconstruction & `CT_BPERM` Un-Permutation:
   - Reads ONLY `trace1` (`Attributes(0)`), `plaintext` (`Attributes(2)`), and `ciphertext`
     (`Attributes(3)`) from the `holdout` split (`trace1` for key recovery; `plaintext` and
     `ciphertext` strictly for `POI-6a/6b` Welch `|t|` diagnostics).
   - Never accesses `trace2` (`Attributes(1)`), `key_a..d` (`Attributes(7..10)`), `round_keys`
     (`Attributes(11)`), `intermediate_shares` (`Attributes(12)`), or `tracewhisperer`
     (`Attributes(13)`) on `holdout`.
   - Ground-truth `key` (`Attributes(6)`) is read strictly at the end in a separate evaluation
     step (`_read_shard_ground_truth_key_only`) to compute recovery metrics.
   - Reconstructs canonical `trace2` POI windows (`remap_hdr_a/b`, `remap_a/b`, `pre_sbox`,
     `sbox_win`, plus `poi6a`, `poi6b`, and the 2-block `CT_BPERM` swap flag) from `trace1`.
2. Coupled 4-Mode Walsh + S-Box Tree Belief Propagation (BP) & Key-Schedule Bijection:
   - Extracts Pass-1 and Pass-2 Gram-projected 4-mode Walsh parity statistics across the 13
     AES-256 key-schedule S-box words (`W_{7+4m}`, `m = 0..12`).
   - Runs Global Self-Calibrated Coupled 4-Mode Walsh + S-Box Tree BP along the 5-step
     AES-256 key-schedule spine and maps posteriors via the exact algebraic bijection
     (`W_7..W_35 -> W_0..W_7`) to recover the full 256-bit master key.
3. Evaluation & Visualization:
   - Reports trace-budget curves, per-key predictions, Top-N full-key candidates, and `POI-6a/6b`
     Welch `|t|` diagnostics, saving JSON results and the success-rate curve plot.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import heapq
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import time
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sedpack.io.compress import CompressedFile
import sedpack.io.flatbuffer.shardfile.Shard as fbapi_Shard

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. Constants & Cryptographic / Walsh Lookup Tables
# ---------------------------------------------------------------------------

REF_TRACE2_STD = 188.74
_MIN_CANONICAL_T2_REF_LEN = 39821
_REQUIRED_TEMPLATE_KEYS: frozenset[str] = frozenset(
    {
        "whiten_models",
        "d0_ops_full",
        "w_blend_all",
        "s_j_vec",
        "mu_f_full",
        "w1_raw",
        "w2_raw",
        "w3_raw",
        "slopes_init",
        "sigma2_init",
        "r_oof_bits",
        "w_ca_models",
        "slopes_p1",
        "sigma2_modes",
        "w_b2_models",
        "slopes_p2",
        "sigma2_p2",
    }
)

# Precomputed Murmur3-32 swap_bit table for nb_blocks == 2 when k_m == 1 (pair_val == 3, r1 = 31 - m)
_CT_BPERM_NB2_SWAP_ON_K1: tuple[int, ...] = (
    0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1,
    0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0,
)

AES_SBOX = np.array(
    [
        0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
        0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
        0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
        0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
        0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
        0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
        0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
        0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
        0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
        0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
        0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
        0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
        0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
        0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
        0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
        0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
    ],
    dtype=np.uint8,
)
INV_SBOX = np.argsort(AES_SBOX).astype(np.uint8)
RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40]
RCON_BY_EVEN_M = {0: 0x01, 2: 0x02, 4: 0x04, 6: 0x08, 8: 0x10, 10: 0x20, 12: 0x40}

HW8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.float64)
HW8_INT = HW8.astype(np.int32)
BYTE_BITS_PM1 = np.array(
    [[(2 * ((x >> j) & 1) - 1) for j in range(8)] for x in range(256)], dtype=np.float64
)
STATES_16 = np.array(
    [[2 * ((s >> q) & 1) - 1 for q in range(4)] for s in range(16)], dtype=np.float64
)
WALSH_H4 = 0.5 * np.array(
    [
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, 1],
    ],
    dtype=np.float64,
)
D_STATES_16 = STATES_16 @ WALSH_H4.T
XOR_TABLE_16 = np.arange(16)[:, None] ^ np.arange(16)[None, :]
MASK_BITS_16 = np.array([[(c >> q) & 1 for q in range(4)] for c in range(16)], dtype=np.int32)

TREE_WORDS = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12]
FREE_WORDS = [5, 6]
M_TREE_STRONG = [0, 1, 2, 3, 7, 8, 9, 10, 11]
SBOX_QUAD_PERM_U = np.array(list(range(10, 27)) + [35, 36, 46, 47, 48, 49, 50, 51, 52])

_CACHED_CANONICAL_T2_REF: np.ndarray | None = None
_WORKER_T2_REF: np.ndarray | None = None


# ---------------------------------------------------------------------------
# 2. Dataset Discovery & Shard Readers
# ---------------------------------------------------------------------------


def _collect_shard_files_from_list(in_dir: Path, rel_sl_path: str) -> list[str]:
    """Recursively collect all relative `.fb` shard file paths referenced by `rel_sl_path`."""
    sl_file = in_dir / rel_sl_path
    if not sl_file.is_file():
        return []
    try:
        sl_data = json.loads(sl_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to parse shard list file {sl_file}: {exc}") from exc
    out_paths: list[str] = []
    for child in sl_data.get("children_shard_lists", ()):
        child_rel = child["shard_list_info_file"]["file_path"]
        out_paths.extend(_collect_shard_files_from_list(in_dir, child_rel))
    for sf in sl_data.get("shard_files", ()):
        for fi in sf.get("file_infos", ()):
            fp = fi["file_path"]
            if (in_dir / fp).is_file():
                out_paths.append(fp)
    return out_paths


def discover_split_shard_files(dataset_dir: Path, split: str) -> list[str]:
    """Discover all `.fb` shard files for `split` in `dataset_dir`."""
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}. "
            "Provide a valid sedpack dataset directory positionally or via '--dataset-path <dir>'."
        )
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_dir}")

    rel_paths = _collect_shard_files_from_list(dataset_dir, f"{split}/shards_list.json")
    full_paths = [str(dataset_dir / p) for p in rel_paths if (dataset_dir / p).is_file()]
    if not full_paths:
        split_dir = dataset_dir / split
        if split_dir.is_dir():
            full_paths = [str(p) for p in sorted(split_dir.rglob("*.fb"))]
    if not full_paths:
        raise FileNotFoundError(
            f"No `.fb` shard files found for split {split!r} in dataset directory: {dataset_dir}"
        )
    return full_paths


def load_holdout_split_metadata(
    dataset_dir: Path,
    n_total_traces: int | None = None,
) -> tuple[int, int]:
    """Read `(num_keys, traces_per_key)` for the `holdout` split from `dataset_info.json`."""
    info_path = dataset_dir / "dataset_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing dataset metadata file: {info_path}")

    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to parse dataset metadata file {info_path}: {exc}") from exc

    custom_meta = info.get("metadata", {}).get("custom_metadata", info.get("custom_metadata", {}))
    holdout_info = custom_meta.get("splits_info", {}).get("holdout", {})

    traces_per_key = int(
        holdout_info.get(
            "traces_per_key",
            custom_meta.get("poi_analysis_budgets", {}).get("recommended_attack_traces_per_key", 0),
        )
    )
    if traces_per_key <= 0:
        raise ValueError(f"Could not determine `traces_per_key` for `holdout` split in {info_path}")

    num_keys = int(holdout_info.get("num_keys", 0))
    if n_total_traces is not None:
        if n_total_traces < traces_per_key:
            traces_per_key = n_total_traces
            num_keys = 1
        else:
            num_keys = n_total_traces // traces_per_key
    elif num_keys <= 0 and "num_traces" in holdout_info:
        num_keys = int(holdout_info["num_traces"]) // traces_per_key

    if num_keys <= 0:
        raise ValueError(f"Could not determine `num_keys` for `holdout` split in {info_path}")
    if traces_per_key < 64:
        raise ValueError(
            f"Holdout `traces_per_key` ({traces_per_key}) in {info_path} must be at least 64."
        )
    return num_keys, traces_per_key


def load_canonical_t2_ref(ref_path: str | Path | None = None) -> np.ndarray:
    """Load and validate the fixed, non-secret canonical `trace2` reference profile (`canonical_t2_ref.npy`)."""
    global _CACHED_CANONICAL_T2_REF
    if _CACHED_CANONICAL_T2_REF is not None and ref_path is None:
        return _CACHED_CANONICAL_T2_REF

    base_dir = Path(__file__).resolve().parent
    default_ref_file = base_dir / "canonical_t2_ref.npy"
    target_ref_file = (
        Path(ref_path).expanduser().resolve() if ref_path is not None else default_ref_file
    )
    if not target_ref_file.is_file():
        raise FileNotFoundError(
            f"Canonical trace2 reference profile not found: {target_ref_file}. "
            f"Ensure 'canonical_t2_ref.npy' is placed alongside {Path(__file__).name}."
        )

    try:
        raw_arr = np.load(target_ref_file, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Failed to load canonical trace2 reference profile from {target_ref_file}: {exc}"
        ) from exc

    if raw_arr.ndim != 1 or raw_arr.size < _MIN_CANONICAL_T2_REF_LEN:
        raise ValueError(
            f"Invalid canonical trace2 reference profile in {target_ref_file}: "
            f"expected 1D waveform of length >= {_MIN_CANONICAL_T2_REF_LEN}, got shape {raw_arr.shape}."
        )
    arr = raw_arr.astype(np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"Canonical trace2 reference profile in {target_ref_file} contains NaN or infinite values."
        )

    if ref_path is None:
        _CACHED_CANONICAL_T2_REF = arr
    return arr


def load_walsh_tree_templates(templates_path: Path) -> dict[str, Any]:
    """Load and validate pre-profiled Coupled Walsh + S-Box Tree BP templates."""
    if not templates_path.exists():
        raise FileNotFoundError(
            f"Pre-profiled Walsh + S-Box Tree BP template file not found: {templates_path}. "
            "Provide a valid path via '--templates-path <file>'."
        )
    if not templates_path.is_file():
        raise IsADirectoryError(f"Template path is not a regular file: {templates_path}")

    try:
        with open(templates_path, "rb") as f_in:
            mdl = pickle.load(f_in)
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ValueError) as exc:
        raise ValueError(
            f"Failed to unpickle Walsh + S-Box Tree BP templates from {templates_path}: {exc}"
        ) from exc

    if not isinstance(mdl, dict):
        raise ValueError(
            f"Invalid template format in {templates_path}: expected a dict, got {type(mdl).__name__}."
        )
    missing_keys = sorted(_REQUIRED_TEMPLATE_KEYS - mdl.keys())
    if missing_keys:
        raise ValueError(
            f"Template file {templates_path} is missing required keys: {', '.join(missing_keys)}"
        )
    return mdl


def _init_holdout_trace1_worker(t2_ref: np.ndarray) -> None:
    """Initialize worker process with shared canonical `t2_ref` waveform."""
    global _WORKER_T2_REF
    _WORKER_T2_REF = t2_ref


def _read_shard_holdout_trace1_strict(sh_path: str) -> tuple[np.ndarray, ...]:
    """Strictly read ONLY `trace1` (Attr 0) on holdout, and reconstruct the canonical `trace2`
    POI windows from `trace1` alone.
    """
    comp = CompressedFile("LZ4")
    buf = comp.decompress(Path(sh_path).read_bytes())
    sh = fbapi_Shard.Shard.GetRootAs(buf, 0)
    n = sh.ExamplesLength()
    remap_hdr_a = np.empty((n, 18), dtype=np.float32)
    remap_hdr_b = np.empty((n, 18), dtype=np.float32)
    remap_a = np.empty((n, 64, 30), dtype=np.float32)
    remap_b = np.empty((n, 64, 30), dtype=np.float32)
    pre_sbox = np.empty((n, 13, 56), dtype=np.float32)
    sbox_win = np.empty((n, 13, 240), dtype=np.float32)

    t2_ref = _WORKER_T2_REF
    for j in range(n):
        ex = sh.Examples(j)
        t1 = np.frombuffer(ex.Attributes(0).AttributeBytesAsNumpy(), dtype=np.int16)
        pois = extract_reconstructed_pois_from_trace1(t1, t2_ref, include_poi6=False)
        remap_hdr_a[j] = pois["remap_hdr_a"]
        remap_hdr_b[j] = pois["remap_hdr_b"]
        remap_a[j] = pois["remap_a"]
        remap_b[j] = pois["remap_b"]
        pre_sbox[j] = pois["pre_sbox"]
        sbox_win[j] = pois["sbox_win"]
    return remap_hdr_a, remap_hdr_b, remap_a, remap_b, pre_sbox, sbox_win


def _read_shard_holdout_poi6_probe_strict(
    sh_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read ONLY `trace1` (Attr 0), `plaintext` (Attr 2), and `ciphertext` (Attr 3) on a probe shard
    to reconstruct `POI-6a`, `POI-6b`, and `bperm` from `trace1` alone.
    """
    comp = CompressedFile("LZ4")
    buf = comp.decompress(Path(sh_path).read_bytes())
    sh = fbapi_Shard.Shard.GetRootAs(buf, 0)
    n = sh.ExamplesLength()
    poi6a = np.empty((n, 57), dtype=np.float32)
    poi6b = np.empty((n, 57), dtype=np.float32)
    bperm = np.empty(n, dtype=np.int32)
    ks = np.empty((n, 32), dtype=np.uint8)

    t2_ref = _WORKER_T2_REF
    for j in range(n):
        ex = sh.Examples(j)
        t1 = np.frombuffer(ex.Attributes(0).AttributeBytesAsNumpy(), dtype=np.int16)
        pt = np.frombuffer(ex.Attributes(2).AttributeBytesAsNumpy(), dtype=np.uint8)[:32]
        ct = np.frombuffer(ex.Attributes(3).AttributeBytesAsNumpy(), dtype=np.uint8)[:32]
        pois = extract_reconstructed_pois_from_trace1(t1, t2_ref, include_poi6=True)
        poi6a[j] = pois["poi6a"]
        poi6b[j] = pois["poi6b"]
        bperm[j] = int(pois["bperm"])
        ks[j] = pt ^ ct
    return poi6a, poi6b, bperm, ks


def _read_shard_ground_truth_key_only(sh_path: str) -> np.ndarray:
    """Read ground-truth `key` (`Attributes(6)`) strictly at evaluation time."""
    comp = CompressedFile("LZ4")
    buf = comp.decompress(Path(sh_path).read_bytes())
    sh = fbapi_Shard.Shard.GetRootAs(buf, 0)
    return np.frombuffer(sh.Examples(0).Attributes(6).AttributeBytesAsNumpy(), dtype=np.uint8).copy()


# ---------------------------------------------------------------------------
# 3. Blind `trace1 -> trace2` POI Reconstruction & `CT_BPERM` Un-Permutation
# ---------------------------------------------------------------------------


def _excise_intervals(
    arr: np.ndarray, start: int, end: int, intervals: list[tuple[int, int]]
) -> np.ndarray:
    """Slice `arr[start:end]` while excising sorted half-open intervals `(s, e)`."""
    pieces: list[np.ndarray] = []
    cur = start
    for s, e in intervals:
        if s > cur:
            pieces.append(arr[cur:s])
        cur = max(cur, e)
    if end > cur:
        pieces.append(arr[cur:end])
    if not pieces:
        return arr[0:0]
    return np.concatenate(pieces) if len(pieces) > 1 else pieces[0]


def _remove_4phase_carrier(arr1: np.ndarray, adc_mul: int = 1) -> np.ndarray:
    """Remove the 4-phase clock-interleaving carrier offset from `trace1`."""
    period = 4 * adc_mul
    if arr1.shape[0] < period:
        return arr1.copy()
    arr1_f = arr1.astype(np.float64)
    overall_mean = float(np.mean(arr1_f))
    for r_phase in range(period):
        arr1_f[r_phase::period] -= float(np.mean(arr1_f[r_phase::period])) - overall_mean
    return np.clip(np.round(arr1_f), -32768, 32767).astype(np.int16)


def _extract_self_templates(arr1_4p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract SHA-256 and `jstate` refill self-templates from deterministic `trace1` positions."""
    np_segs = []
    call_segs = []
    for m in range(8):
        st = 12708 + 412 * m
        np_segs.append(arr1_4p[st : st + 188].astype(np.float64))
        call_segs.append(arr1_4p[st + 4 : st + 192].astype(np.float64))
    for st in (16004, 20477):
        np_segs.append(arr1_4p[st : st + 188].astype(np.float64))
        call_segs.append(arr1_4p[st + 4 : st + 192].astype(np.float64))
    tpl_sha_np = np.mean(np_segs, axis=0)
    tpl_sha_call = np.mean(call_segs, axis=0)
    tpl_j_17 = arr1_4p[20186:20203].astype(np.float64)
    return tpl_sha_np, tpl_sha_call, tpl_j_17


def _align_initial_trigger_offset(
    arr1: np.ndarray, t2_ref_orig: np.ndarray
) -> tuple[np.ndarray, int]:
    """Detect and align any initial ADC trigger offset `d_trig in {-1, 0, 1, 2}`."""
    if arr1.shape[0] < 4200 or t2_ref_orig.shape[0] < 4200:
        return arr1, 0
    ref_seg = t2_ref_orig[100:4100] - float(np.mean(t2_ref_orig[100:4100]))
    best_d = 0
    best_dot = -1e30
    for d in (-1, 0, 1, 2):
        seg = arr1[100 + d : 4100 + d].astype(np.float64)
        seg -= float(np.mean(seg))
        dot = float(np.dot(seg, ref_seg))
        if dot > best_dot:
            best_dot = dot
            best_d = d
    if best_d > 0:
        return np.concatenate([arr1[best_d:], np.repeat(arr1[-1:], best_d)]), best_d
    if best_d < 0:
        pad = -best_d
        return np.concatenate([np.repeat(arr1[:1], pad), arr1[:-pad]]), best_d
    return arr1, 0


def _scale_t2_ref_to_trace(
    arr1_f: np.ndarray, t2_ref_orig: np.ndarray
) -> tuple[np.ndarray, float]:
    """Affine-normalize `t2_ref_orig` to `arr1_f`'s deterministic `init_rstate [0:12634]` mean and std."""
    m_ref = float(np.mean(t2_ref_orig[:12634]))
    s_ref = float(np.std(t2_ref_orig[:12634])) + 1e-8
    m1 = float(np.mean(arr1_f[:12634]))
    s1 = float(np.std(arr1_f[:12634])) + 1e-8
    scale = s1 / s_ref
    t2_ref = (t2_ref_orig - m_ref) * scale + m1
    return t2_ref, scale


def _match_landmark(
    arr1_f: np.ndarray,
    t2_ref: np.ndarray,
    can_start: int,
    can_len: int,
    act_lo: int,
    act_hi: int,
    prior_center: float | None = None,
    prior_weight: float = 0.0,
) -> int:
    """Locate a canonical landmark `t2_ref[can_start : can_start + can_len]` in `arr1_f`."""
    tpl = t2_ref[can_start : can_start + can_len]
    tpl_c = tpl - np.mean(tpl)
    act_lo = max(0, int(act_lo))
    act_hi = min(int(arr1_f.shape[0] - can_len), int(act_hi))
    if act_hi <= act_lo:
        return act_lo
    win = sliding_window_view(arr1_f[act_lo : act_hi + can_len], can_len)
    win_c = win - np.mean(win, axis=1, keepdims=True)
    errs = np.mean(np.abs(win_c - tpl_c[None, :]), axis=1)
    if prior_center is not None and prior_weight > 0.0:
        cands = np.arange(act_lo, act_hi + 1, dtype=np.float64)
        errs = errs + prior_weight * np.abs(cands - prior_center)
    return int(act_lo + int(np.argmin(errs)))


def _resample_1d(seg: np.ndarray, target_len: int) -> np.ndarray:
    if seg.shape[0] == target_len:
        return seg.copy()
    if seg.shape[0] == 0:
        return np.zeros(target_len, dtype=np.int16)
    idx = np.linspace(0, seg.shape[0] - 1, target_len)
    return np.round(np.interp(idx, np.arange(seg.shape[0]), seg)).astype(np.int16)


def _infer_ct_bperm_swap_from_trace1(arr1: np.ndarray, c_bperm0: int, c_bperm1: int) -> int:
    """Infer the 2-block `CT_BPERM` swap flag (`0` unswapped, `1` swapped) from `trace1`
    using a 2-state Viterbi decoder across the 32 Murmur3-32 iterations.
    """
    n_samples = int(arr1.shape[0])
    loop_len = 32 * 42
    s0 = max(0, min(n_samples, int(c_bperm0) + 15))
    s1 = max(0, min(n_samples, int(c_bperm1) + 15))
    if s0 + loop_len > n_samples or s1 + loop_len > n_samples:
        return 0

    d = arr1[s1 : s1 + loop_len].astype(np.float64) - arr1[s0 : s0 + loop_len].astype(np.float64)
    for r in range(4):
        d[r::4] -= float(np.mean(d[r::4]))
    w = d.reshape(32, 42)

    s_k0 = -np.sum(w[:, 3:8], axis=1)
    s_k1_a = np.sum(w[:, 9:13], axis=1)
    s_k1_b = -np.sum(w[:, 36:40], axis=1)

    cost0 = 0.0
    cost1 = 1e18
    for m in range(32):
        sk0 = float(s_k0[m])
        sk1a = float(s_k1_a[m])
        sk1b = float(s_k1_b[m])
        e_i0_k0 = (sk0 - 410.0) ** 2 + sk1a ** 2 + sk1b ** 2
        e_i0_k1 = sk0 ** 2 + (sk1a - 650.0) ** 2 + (sk1b - 350.0) ** 2
        e_i1_k0 = (sk0 + 410.0) ** 2 + sk1a ** 2 + sk1b ** 2
        e_i1_k1 = sk0 ** 2 + (sk1a + 650.0) ** 2 + (sk1b + 350.0) ** 2

        if _CT_BPERM_NB2_SWAP_ON_K1[m] == 0:
            cost0 += min(e_i0_k0, e_i0_k1)
            cost1 += min(e_i1_k0, e_i1_k1)
        else:
            new_cost0 = min(cost0 + e_i0_k0, cost1 + e_i1_k1)
            new_cost1 = min(cost1 + e_i1_k0, cost0 + e_i0_k1)
            cost0, cost1 = new_cost0, new_cost1

    return 0 if cost0 <= cost1 else 1


def _score_pre810_ultra(
    arr1_f: np.ndarray,
    tpl_810: np.ndarray,
    tpl_sha_np: np.ndarray,
    tpl_sha_call: np.ndarray,
    tpl_j_17: np.ndarray,
    act_m_start: int,
    a_312: int,
    sb_shas: list[bool],
    rc_shas: list[bool],
    c_shas: list[tuple[bool, bool, bool]],
    rem_u_810: int,
) -> tuple[np.ndarray | None, float]:
    """Fast decoupled scorer for the 810-cycle `pre_810` key-expansion header `[20833+1440*m : 21643+1440*m]`."""
    ex_sb = 188 * sum(sb_shas)
    j0_fl, j0_s, c0_s = c_shas[0]
    ex_c0_no_u = (17 if j0_fl else 0) + 188 * (int(j0_s) + int(c0_s))

    p_sb = []
    cur_sb = act_m_start
    cum0 = 0
    pen_sb = 0.0
    for d_sb_i, is_s in enumerate(sb_shas):
        if is_s:
            s_p = act_m_start + 22 * d_sb_i + cum0
            if s_p > cur_sb:
                p_sb.append(arr1_f[cur_sb:s_p])
            cur_sb = s_p + 188
            pen_sb += 0.3 * float(np.mean(np.abs(arr1_f[s_p:cur_sb] - tpl_sha_np)))
            cum0 += 188
    c0_ent = act_m_start + 95 + cum0
    if c0_ent + 17 > cur_sb:
        p_sb.append(arr1_f[cur_sb : c0_ent + 17])
    prefix_112 = np.concatenate(p_sb) if len(p_sb) > 1 else p_sb[0]
    err_112 = float(np.sum(np.abs(prefix_112 - tpl_810[:112])))

    u0_cands: dict[int, tuple[float, np.ndarray, np.ndarray]] = {}
    for u0 in range(min(4, rem_u_810 + 1)):
        if rem_u_810 - u0 > 9:
            continue
        cand_rcpre = act_m_start + 143 + ex_sb + ex_c0_no_u + u0
        pen0 = pen_sb
        cur = c0_ent + 17 + u0
        if not j0_fl and not c0_s:
            tail_152 = arr1_f[cur : cand_rcpre + 121]
        else:
            p0 = [arr1_f[cur : cur + 3]]
            cur += 3
            if j0_fl:
                pen0 += 0.25 * float(np.mean(np.abs(arr1_f[cur : cur + 17] - tpl_j_17)))
                if j0_s:
                    pen0 += 0.3 * float(np.mean(np.abs(arr1_f[cur + 17 : cur + 205] - tpl_sha_call)))
                cur += 205 if j0_s else 17
            if c0_s:
                c_end = cand_rcpre - 8
                p0.append(arr1_f[cur : c_end - 188])
                pen0 += 0.3 * float(np.mean(np.abs(arr1_f[c_end - 188 : c_end] - tpl_sha_call)))
                cur = c_end
            p0.append(arr1_f[cur : cand_rcpre + 121])
            tail_152 = np.concatenate(p0)
        if tail_152.shape[0] != 152:
            continue

        rcpre_end_act = cand_rcpre + 121
        p_rc = []
        cur_r = rcpre_end_act
        cum_rc = 0
        pen_rc = 0.0
        for j_rc_i, is_s in enumerate(rc_shas):
            if is_s:
                s_p = rcpre_end_act + 29 * j_rc_i + cum_rc
                if s_p > cur_r:
                    p_rc.append(arr1_f[cur_r:s_p])
                cur_r = s_p + 188
                pen_rc += 0.3 * float(np.mean(np.abs(arr1_f[s_p:cur_r] - tpl_sha_np)))
                cum_rc += 188
        rc_end_act_cand = rcpre_end_act + 346 + cum_rc
        if rc_end_act_cand > cur_r:
            p_rc.append(arr1_f[cur_r:rc_end_act_cand])
        seg_rc346_cand = np.concatenate(p_rc) if len(p_rc) > 1 else p_rc[0]
        if seg_rc346_cand.shape[0] != 346:
            continue

        sc0 = (
            err_112
            + float(np.sum(np.abs(tail_152 - tpl_810[112:264])))
            + float(np.sum(np.abs(seg_rc346_cand - tpl_810[264:610])))
        ) + (264.0 * pen0 + 346.0 * pen_rc)
        u0_cands[u0] = (sc0, tail_152, seg_rc346_cand)

    if not u0_cands:
        return None, 1e18

    rc_end_base = act_m_start + 610 + ex_sb + ex_c0_no_u + 188 * sum(rc_shas)
    j1_fl, j1_s, c1_s = c_shas[1]
    j2_fl, j2_s, c2_s = c_shas[2]
    j3_fl, j3_s, c3_s = c_shas[3]
    ex_c1_no_u = (17 if j1_fl else 0) + 188 * (int(j1_s) + int(c1_s))
    ex_c2_no_u = (17 if j2_fl else 0) + 188 * (int(j2_s) + int(c2_s))
    ex_c3_no_u = (17 if j3_fl else 0) + 188 * (int(j3_s) + int(c3_s))

    part1_tail_by_u01: dict[int, tuple[float, np.ndarray]] = {}
    for u01 in range(max(0, rem_u_810 - 6), min(6, rem_u_810) + 1):
        cur = rc_end_base + 17 + u01
        pen1 = 0.0
        c1_ret = rc_end_base + 48 + ex_c1_no_u + u01
        c2_udiv = c1_ret + 55 + 17
        if not j1_fl and not c1_s:
            seg103 = arr1_f[cur:c2_udiv]
        else:
            p1 = [arr1_f[cur : cur + 3]]
            cur += 3
            if j1_fl:
                pen1 += 0.25 * float(np.mean(np.abs(arr1_f[cur : cur + 17] - tpl_j_17)))
                if j1_s:
                    pen1 += 0.3 * float(np.mean(np.abs(arr1_f[cur + 17 : cur + 205] - tpl_sha_call)))
                cur += 205 if j1_s else 17
            if c1_s:
                s_c = c1_ret - 196
                p1.append(arr1_f[cur:s_c])
                pen1 += 0.3 * float(np.mean(np.abs(arr1_f[s_c : s_c + 188] - tpl_sha_call)))
                cur = c1_ret - 8
            p1.append(arr1_f[cur:c2_udiv])
            seg103 = np.concatenate(p1)
        if seg103.shape[0] == 103:
            e_103 = (
                float(np.sum(np.abs(seg103[:31] - tpl_810[627:658])))
                + 1.5 * float(np.sum(np.abs(seg103[31:] - tpl_810[658:730])))
                + 200.0 * pen1
            )
            part1_tail_by_u01[u01] = (e_103, seg103)

    part2_by_u012: dict[int, tuple[float, np.ndarray]] = {}
    for u012 in range(max(0, rem_u_810 - 3), min(9, rem_u_810) + 1):
        cur = rc_end_base + 103 + ex_c1_no_u + 17 + u012
        c2_ret = rc_end_base + 151 + ex_c1_no_u + ex_c2_no_u + u012
        c3_udiv = c2_ret + 1 + 17
        pen2 = 0.0
        if not j2_fl and not c2_s:
            seg49 = arr1_f[cur:c3_udiv]
        else:
            p2 = [arr1_f[cur : cur + 3]]
            cur += 3
            if j2_fl:
                pen2 += 0.25 * float(np.mean(np.abs(arr1_f[cur : cur + 17] - tpl_j_17)))
                if j2_s:
                    pen2 += 0.3 * float(np.mean(np.abs(arr1_f[cur + 17 : cur + 205] - tpl_sha_call)))
                cur += 205 if j2_s else 17
            if c2_s:
                s_c = c2_ret - 196
                p2.append(arr1_f[cur:s_c])
                pen2 += 0.3 * float(np.mean(np.abs(arr1_f[s_c : s_c + 188] - tpl_sha_call)))
                cur = c2_ret - 8
            p2.append(arr1_f[cur:c3_udiv])
            seg49 = np.concatenate(p2)
        if seg49.shape[0] == 49:
            part2_by_u012[u012] = (
                float(np.sum(np.abs(seg49 - tpl_810[730:779]))) + 200.0 * pen2,
                seg49,
            )

    cur3 = a_312 - 31 - ex_c3_no_u
    pen3 = 0.0
    if not j3_fl and not c3_s:
        seg31 = arr1_f[cur3:a_312]
    else:
        p3 = [arr1_f[cur3 : cur3 + 3]]
        cur3 += 3
        if j3_fl:
            pen3 += 0.25 * float(np.mean(np.abs(arr1_f[cur3 : cur3 + 17] - tpl_j_17)))
            if j3_s:
                pen3 += 0.3 * float(np.mean(np.abs(arr1_f[cur3 + 17 : cur3 + 205] - tpl_sha_call)))
            cur3 += 205 if j3_s else 17
        if c3_s:
            s_c = a_312 - 196
            p3.append(arr1_f[cur3:s_c])
            pen3 += 0.3 * float(np.mean(np.abs(arr1_f[s_c : s_c + 188] - tpl_sha_call)))
            cur3 = a_312 - 8
        p3.append(arr1_f[cur3:a_312])
        seg31 = np.concatenate(p3)
    if seg31.shape[0] != 31:
        return None, 1e18
    err3 = float(np.sum(np.abs(seg31 - tpl_810[779:810]))) + 200.0 * pen3

    best_tot_sc = 1e18
    best_u0_sel = -1
    best_u01_sel = -1
    best_u012_sel = -1
    best_seg17_sel = None
    for u0, (sc0_sum, _, _) in u0_cands.items():
        rc_end_act = rc_end_base + u0
        seg17 = arr1_f[rc_end_act : rc_end_act + 17]
        e_17 = float(np.sum(np.abs(seg17 - tpl_810[610:627])))
        for u1 in range(4):
            u01 = u0 + u1
            if u01 not in part1_tail_by_u01:
                continue
            e_103, _ = part1_tail_by_u01[u01]
            rem_23 = rem_u_810 - u01
            if rem_23 < 0 or rem_23 > 6:
                continue
            for u2 in range(max(0, rem_23 - 3), min(3, rem_23) + 1):
                u012 = u01 + u2
                if u012 not in part2_by_u012:
                    continue
                e2, _ = part2_by_u012[u012]
                sc_all = (sc0_sum + e_17 + e_103 + e2 + err3) / 846.0
                if sc_all < best_tot_sc:
                    best_tot_sc = sc_all
                    best_u0_sel = u0
                    best_u01_sel = u01
                    best_u012_sel = u012
                    best_seg17_sel = seg17

    if best_u0_sel < 0 or best_seg17_sel is None:
        return None, 1e18
    _, best_tail_152, seg_rc346_f = u0_cands[best_u0_sel]
    seg810 = np.concatenate(
        [
            prefix_112,
            best_tail_152,
            seg_rc346_f,
            best_seg17_sel,
            part1_tail_by_u01[best_u01_sel][1],
            part2_by_u012[best_u012_sel][1],
            seg31,
        ]
    ).astype(np.int16)
    return seg810, best_tot_sc


def _decode_bridge_with_last_j(
    arr1_4p: np.ndarray,
    arr1_f: np.ndarray,
    t2_ref: np.ndarray,
    tpl_sha_np: np.ndarray,
    tpl_sha_call: np.ndarray,
    tpl_j_17: np.ndarray,
    m_ik: int,
    a_312_cur: int,
    a_312_nxt: int,
    d_start: int,
    last_j: int,
    top_k_usum: int = 1,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Decode `srk1 + srk2` of `m_ik` and `pre_810` of `m_ik + 1` between `a_312_cur + 312` and `a_312_nxt`."""
    act_start = a_312_cur + 312
    tot_ex_1128 = (a_312_nxt - act_start) - 1128
    base_cur = 1440 * m_ik
    base_nxt = 1440 * (m_ik + 1)
    k_base = 33 + 6 * m_ik
    len_srk1 = 107 - (3 if m_ik % 2 == 1 else 0)
    len_srk2 = 115 + (3 if m_ik % 2 == 1 else 0)
    can_srk1_ent = 21955 + base_cur
    can_srk2_ent = 22003 + base_cur + len_srk1
    tpl_srk1 = t2_ref[can_srk1_ent : can_srk1_ent + 48 + len_srk1]
    tpl_srk2 = t2_ref[can_srk2_ent : can_srk2_ent + 48 + len_srk2]
    tpl_810 = t2_ref[20833 + base_nxt : 21643 + base_nxt]

    best_total_sc = 1e18
    best_srk_seg = None
    best_pre810_seg = None
    best_d_end = (d_start + 6) & 7
    best_last_j = last_j

    for j_idx in (None, 0, 1, 2, 3, 4, 5):
        gap_pen = 0.0
        if j_idx is not None:
            if last_j <= 28:
                if (k_base + j_idx) < 29 or (k_base + j_idx) > 44:
                    continue
            else:
                gap = (k_base + j_idx) - last_j
                if gap < 1 or gap > 16:
                    continue
                if gap < 10:
                    gap_pen = 6.0 * (10 - gap)
        else:
            if last_j > 28 and (k_base + 5) - last_j >= 16 and (tot_ex_1128 % 188) >= 17:
                continue
            if last_j <= 28 and (k_base + 5) >= 44 and (tot_ex_1128 % 188) >= 17:
                continue

        d_cur = d_start
        j_srk1 = j_idx == 0
        js_srk1 = (d_cur == 0) if j_srk1 else False
        if j_srk1:
            d_cur = (d_cur + 1) & 7
        cs_srk1 = d_cur == 0
        d_cur = (d_cur + 1) & 7

        j_srk2 = j_idx == 1
        js_srk2 = (d_cur == 0) if j_srk2 else False
        if j_srk2:
            d_cur = (d_cur + 1) & 7
        cs_srk2 = d_cur == 0
        d_cur = (d_cur + 1) & 7

        sb_shas = [(d_cur + i) & 7 == 0 for i in range(4)]
        d_cur = (d_cur + 4) & 7

        c_shas = []
        rc_shas: list[bool] = []
        for c_i in range(4):
            j_fl = j_idx == c_i + 2
            j_s = (d_cur == 0) if j_fl else False
            if j_fl:
                d_cur = (d_cur + 1) & 7
            c_s = d_cur == 0
            d_cur = (d_cur + 1) & 7
            c_shas.append((j_fl, j_s, c_s))
            if c_i == 0:
                rc_shas = [(d_cur + i) & 7 == 0 for i in range(12)]
                d_cur = (d_cur + 12) & 7

        tot_sha = (
            int(js_srk1)
            + int(cs_srk1)
            + int(js_srk2)
            + int(cs_srk2)
            + sum(sb_shas)
            + sum(rc_shas)
            + sum(int(js) + int(cs) for _, js, cs in c_shas)
        )
        j_tot_17 = 17 if (j_idx is not None) else 0
        rem_u_all = tot_ex_1128 - 188 * tot_sha - j_tot_17
        if rem_u_all < 0 or rem_u_all > 18:
            continue

        ex_srk1_no_u = (17 if j_srk1 else 0) + 188 * (int(js_srk1) + int(cs_srk1))
        ex_srk2_no_u = (17 if j_srk2 else 0) + 188 * (int(js_srk2) + int(cs_srk2))

        best_by_usum: dict[int, tuple[float, np.ndarray]] = {}
        for u_srk1 in range(4):
            ret_srk1 = act_start + 48 + ex_srk1_no_u + u_srk1
            ent_srk2 = ret_srk1 + len_srk1
            exc1: list[tuple[int, int]] = []
            pen1 = 0.0
            if u_srk1 > 0:
                exc1.append((act_start + 17, act_start + 17 + u_srk1))
            if j_srk1:
                j_st = act_start + 20 + u_srk1
                exc1.append((j_st, j_st + 17 + (188 if js_srk1 else 0)))
                pen1 += 0.25 * float(np.mean(np.abs(arr1_f[j_st : j_st + 17] - tpl_j_17)))
                if js_srk1:
                    pen1 += 0.3 * float(np.mean(np.abs(arr1_f[j_st + 17 : j_st + 205] - tpl_sha_call)))
            if cs_srk1:
                c_end = ret_srk1 - 8
                exc1.append((c_end - 188, c_end))
                pen1 += 0.3 * float(np.mean(np.abs(arr1_f[c_end - 188 : c_end] - tpl_sha_call)))
            seg_srk1_f = _excise_intervals(arr1_f, act_start, ent_srk2, exc1)
            if seg_srk1_f.shape[0] != 48 + len_srk1:
                continue
            sc_srk1 = (
                0.5 * float(np.mean(np.abs(seg_srk1_f[:48] - tpl_srk1[:48])))
                + 0.5 * float(np.mean(np.abs(seg_srk1_f[48:] - tpl_srk1[48:])))
                + pen1
            )

            for u_srk2 in range(4):
                u_sum = u_srk1 + u_srk2
                rem_u_810 = rem_u_all - u_sum
                if rem_u_810 < 0 or rem_u_810 > 12:
                    continue
                ret_srk2 = ent_srk2 + 48 + ex_srk2_no_u + u_srk2
                act_m_nxt = ret_srk2 + len_srk2
                exc2: list[tuple[int, int]] = []
                pen2 = 0.0
                if u_srk2 > 0:
                    exc2.append((ent_srk2 + 17, ent_srk2 + 17 + u_srk2))
                if j_srk2:
                    j_st = ent_srk2 + 20 + u_srk2
                    exc2.append((j_st, j_st + 17 + (188 if js_srk2 else 0)))
                    pen2 += 0.25 * float(np.mean(np.abs(arr1_f[j_st : j_st + 17] - tpl_j_17)))
                    if js_srk2:
                        pen2 += 0.3 * float(np.mean(np.abs(arr1_f[j_st + 17 : j_st + 205] - tpl_sha_call)))
                if cs_srk2:
                    c_end = ret_srk2 - 8
                    exc2.append((c_end - 188, c_end))
                    pen2 += 0.3 * float(np.mean(np.abs(arr1_f[c_end - 188 : c_end] - tpl_sha_call)))
                seg_srk2_f = _excise_intervals(arr1_f, ent_srk2, act_m_nxt, exc2)
                if seg_srk2_f.shape[0] != 48 + len_srk2:
                    continue
                sc_srk2 = (
                    0.5 * float(np.mean(np.abs(seg_srk2_f[:48] - tpl_srk2[:48])))
                    + 0.5 * float(np.mean(np.abs(seg_srk2_f[48:] - tpl_srk2[48:])))
                    + pen2
                )
                sc_12 = (155.0 * sc_srk1 + 163.0 * sc_srk2) / 318.0
                if u_sum not in best_by_usum or sc_12 < best_by_usum[u_sum][0]:
                    best_by_usum[u_sum] = (
                        sc_12,
                        np.concatenate([seg_srk1_f, seg_srk2_f]).astype(np.int16),
                    )

        top_usums = sorted(best_by_usum.items(), key=lambda kv: kv[1][0])[:top_k_usum]
        for u_sum, (sc_12, seg_srk_pair) in top_usums:
            act_m_nxt = act_start + 318 + ex_srk1_no_u + ex_srk2_no_u + u_sum
            rem_u_810 = rem_u_all - u_sum
            seg810, sc810 = _score_pre810_ultra(
                arr1_f,
                tpl_810,
                tpl_sha_np,
                tpl_sha_call,
                tpl_j_17,
                act_m_nxt,
                a_312_nxt,
                sb_shas,
                rc_shas,
                c_shas,
                rem_u_810,
            )
            if seg810 is not None:
                tot_sc = (318.0 * sc_12 + 810.0 * sc810) / 1128.0 + gap_pen
                if tot_sc < best_total_sc:
                    best_total_sc = tot_sc
                    best_srk_seg = seg_srk_pair
                    best_pre810_seg = seg810
                    best_d_end = d_cur
                    best_last_j = (k_base + j_idx) if j_idx is not None else last_j

    if best_srk_seg is None or best_pre810_seg is None:
        return (
            _resample_1d(arr1_4p[act_start : act_start + 318], 318),
            _resample_1d(arr1_4p[act_start + 318 : a_312_nxt], 810),
            best_d_end,
            best_last_j,
        )
    return best_srk_seg, best_pre810_seg, best_d_end, best_last_j


def extract_reconstructed_pois_from_trace1(
    trace1: np.ndarray,
    t2_ref: np.ndarray | None = None,
    include_poi6: bool = False,
) -> dict[str, np.ndarray | int | float]:
    """Reconstruct canonical `trace2` POI windows (`remap_hdr_a/b`, `remap_a/b`, `pre_sbox`,
    `sbox_win`, and optionally `poi6a`, `poi6b`, `bperm`) from `trace1` alone.
    """
    if t2_ref is None:
        t2_ref = load_canonical_t2_ref()

    arr1 = np.asarray(trace1, dtype=np.int16).reshape(-1)
    arr1, d_trig = _align_initial_trigger_offset(arr1, t2_ref)
    arr1_4p = _remove_4phase_carrier(arr1, adc_mul=1)
    arr1_f = arr1_4p.astype(np.float64)
    t2_ref_s, scale = _scale_t2_ref_to_trace(arr1_f, t2_ref)
    s1 = float(np.std(arr1_f[:12634])) + 1e-8
    norm_gain = (
        np.float32(300.0 / s1)
        if abs(s1 - REF_TRACE2_STD) > 6.5
        else np.float32(1.0)
    )
    tpl_sha_np, tpl_sha_call, tpl_j_17 = _extract_self_templates(arr1_4p)

    # 1. Deterministic prefix [0:18989]
    det_excise = [(12708 + 412 * m, 12896 + 412 * m) for m in range(8)] + [
        (16004, 16192),
        (20186, 20203),
        (20477, 20665),
    ]
    seg_0_18989 = _excise_intervals(arr1_4p, 0, 20886, det_excise)

    # 2. Match `act_312[0..12]` and `act_end28`
    act_312_0 = _match_landmark(
        arr1_f,
        t2_ref_s,
        21643,
        312,
        21643 + 3100,
        21643 + 3500,
        prior_center=21643 + 3205.0,
        prior_weight=0.02 * scale,
    )
    act_end28 = _match_landmark(
        arr1_f,
        t2_ref_s,
        20714,
        119,
        act_312_0 - 929 - 700,
        act_312_0 - 929 - 450,
        prior_center=act_312_0 - 929 - 495.0,
        prior_weight=0.08 * scale,
    )
    act_312 = [act_312_0]
    for m_ik in range(1, 13):
        base = 1440 * m_ik
        can_312 = 21643 + base
        prev_312 = act_312[m_ik - 1]
        act_312.append(
            _match_landmark(
                arr1_f,
                t2_ref_s,
                can_312,
                312,
                prev_312 + 1440 + 370,
                prev_312 + 1440 + 605,
                prior_center=prev_312 + 1440 + 610.0,
                prior_weight=0.02 * scale,
            )
        )

    dc_m = [
        float(
            np.mean(arr1_f[act_312[m] : act_312[m] + 312])
            - np.mean(t2_ref_s[21643 + 1440 * m : 21955 + 1440 * m])
        )
        for m in range(13)
    ]

    # 3. Connected `m_ik = 0..12` bridge decoder (`pre_sbox` & `sbox_win`)
    act_m0_start = act_end28 + 119
    a_312_0 = act_312[0]
    tot_ex_810 = (a_312_0 - act_m0_start) - 810
    rem_188 = tot_ex_810 % 188
    has_j_810 = rem_188 >= 17
    rem_u_810 = rem_188 - (17 if has_j_810 else 0)
    j_tuples_810 = ((0,), (1,), (2,), (3,)) if has_j_810 else ((None,),)
    tpl_810_0 = t2_ref_s[20833:21643] + dc_m[0]
    best_pre0 = None
    best_sc0 = 1e18
    best_d_after_m0 = 7
    best_last_j = 25

    for d_init in (6, 7):
        for (j_idx,) in j_tuples_810:
            d_cur = d_init
            sb_shas = [(d_cur + i) & 7 == 0 for i in range(4)]
            d_cur = (d_cur + 4) & 7
            c_shas = []
            rc_shas: list[bool] = []
            for c_i in range(4):
                j_fl = j_idx == c_i
                j_s = (d_cur == 0) if j_fl else False
                if j_fl:
                    d_cur = (d_cur + 1) & 7
                c_s = d_cur == 0
                d_cur = (d_cur + 1) & 7
                c_shas.append((j_fl, j_s, c_s))
                if c_i == 0:
                    rc_shas = [(d_cur + i) & 7 == 0 for i in range(12)]
                    d_cur = (d_cur + 12) & 7
            tot_sha = sum(sb_shas) + sum(rc_shas) + sum(int(js) + int(cs) for _, js, cs in c_shas)
            if 188 * tot_sha + (17 if has_j_810 else 0) + rem_u_810 != tot_ex_810:
                continue
            seg810, sc810 = _score_pre810_ultra(
                arr1_f,
                tpl_810_0,
                tpl_sha_np,
                tpl_sha_call,
                tpl_j_17,
                act_m0_start,
                a_312_0,
                sb_shas,
                rc_shas,
                c_shas,
                rem_u_810,
            )
            if seg810 is not None and sc810 < best_sc0:
                best_sc0 = sc810
                best_pre0 = seg810
                best_d_after_m0 = d_cur
                best_last_j = (29 + j_idx) if j_idx is not None else 24

    seg_pre0 = (
        best_pre0 if best_pre0 is not None else _resample_1d(arr1_4p[act_m0_start:a_312_0], 810)
    )
    d_cur = best_d_after_m0
    last_j = best_last_j

    pre_segs: list[np.ndarray] = [seg_pre0]
    for m_ik in range(12):
        eff_last_j = last_j
        for m_fut in range(m_ik + 1, 12):
            tot_ex_fut = (act_312[m_fut + 1] - (act_312[m_fut] + 312)) - 1128
            if (tot_ex_fut % 188) >= 17:
                min_prev_j = (33 + 6 * m_fut) - 16
                if ((act_312[m_ik + 1] - (act_312[m_ik] + 312) - 1128) % 188) >= 17:
                    eff_last_j = max(eff_last_j, min_prev_j - 10)
                break
        t2_ref_local = t2_ref_s + dc_m[m_ik + 1]
        _, seg_pre_nxt, d_cur, last_j = _decode_bridge_with_last_j(
            arr1_4p,
            arr1_f,
            t2_ref_local,
            tpl_sha_np,
            tpl_sha_call,
            tpl_j_17,
            m_ik,
            act_312[m_ik],
            act_312[m_ik + 1],
            d_cur,
            eff_last_j,
            top_k_usum=3,
        )
        pre_segs.append(seg_pre_nxt)

    act_81253 = _match_landmark(arr1_f, t2_ref_s, 81253, 500, 94500, 95800)
    n_pad = act_81253 - 81253
    sha_mean = 0.5 * (float(np.mean(tpl_sha_np)) + float(np.mean(tpl_sha_call)))
    t2_mean = np.float32(
        (float(np.sum(arr1_f)) - n_pad * sha_mean + n_pad * float(arr1_f[-1])) / arr1_f.shape[0]
    )
    remap_hdr_a = (seg_0_18989[14500:14518].astype(np.float32) - t2_mean) * norm_gain
    remap_hdr_b = (seg_0_18989[16461:16479].astype(np.float32) - t2_mean) * norm_gain
    remap_a = (
        (seg_0_18989[14518 : 14518 + 64 * 30].astype(np.float32) - t2_mean) * norm_gain
    ).reshape(64, 30)
    remap_b = (
        (seg_0_18989[16479 : 16479 + 64 * 30].astype(np.float32) - t2_mean) * norm_gain
    ).reshape(64, 30)

    pre_sbox = np.empty((13, 56), dtype=np.float32)
    sbox_win = np.empty((13, 240), dtype=np.float32)
    for m_ik in range(13):
        pre_sbox[m_ik] = (
            pre_segs[m_ik][21491 - 20833 : 21547 - 20833].astype(np.float32) - t2_mean
        ) * norm_gain
        sbox_win[m_ik] = (
            arr1_4p[act_312[m_ik] : act_312[m_ik] + 240].astype(np.float32) - t2_mean
        ) * norm_gain

    out: dict[str, np.ndarray | int | float] = {
        "remap_hdr_a": remap_hdr_a,
        "remap_hdr_b": remap_hdr_b,
        "remap_a": remap_a,
        "remap_b": remap_b,
        "pre_sbox": pre_sbox,
        "sbox_win": sbox_win,
        "t2_mean": float(t2_mean),
    }
    if not include_poi6:
        return out

    # 4. Locate `best_act_ctr`, `CT_BPERM`, and `POI-6a` / `POI-6b`
    act_39235 = act_312[12] + 312
    base_12 = 1440 * 12
    tpl_srk1_12 = t2_ref_s[21955 + base_12 : 21955 + base_12 + 155]
    tpl_srk2_12 = t2_ref_s[22110 + base_12 : 22110 + base_12 + 151]
    best_srk12_sc = 1e18
    best_act_ctr = act_39235 + 306
    for j_idx in (None, 0, 1):
        d_c = d_cur
        j_s1 = j_idx == 0
        js_s1 = (d_c == 0) if j_s1 else False
        if j_s1:
            d_c = (d_c + 1) & 7
        cs_s1 = d_c == 0
        d_c = (d_c + 1) & 7
        j_s2 = j_idx == 1
        js_s2 = (d_c == 0) if j_s2 else False
        if j_s2:
            d_c = (d_c + 1) & 7
        cs_s2 = d_c == 0
        ex1_no_u = (17 if j_s1 else 0) + 188 * (int(js_s1) + int(cs_s1))
        ex2_no_u = (17 if j_s2 else 0) + 188 * (int(js_s2) + int(cs_s2))
        for u1 in range(4):
            ret1 = act_39235 + 48 + ex1_no_u + u1
            ent2 = ret1 + 107
            exc1: list[tuple[int, int]] = []
            if u1 > 0:
                exc1.append((act_39235 + 17, act_39235 + 17 + u1))
            if j_s1:
                exc1.append((act_39235 + 20 + u1, act_39235 + 37 + u1 + (188 if js_s1 else 0)))
            if cs_s1:
                exc1.append((ret1 - 196, ret1 - 8))
            s1_f = _excise_intervals(arr1_f, act_39235, ent2, exc1)
            if s1_f.shape[0] != 155:
                continue
            sc1 = float(np.mean(np.abs(s1_f - tpl_srk1_12)))
            for u2 in range(4):
                ret2 = ent2 + 48 + ex2_no_u + u2
                act_end = ret2 + 103
                exc2: list[tuple[int, int]] = []
                if u2 > 0:
                    exc2.append((ent2 + 17, ent2 + 17 + u2))
                if j_s2:
                    exc2.append((ent2 + 20 + u2, ent2 + 37 + u2 + (188 if js_s2 else 0)))
                if cs_s2:
                    exc2.append((ret2 - 196, ret2 - 8))
                s2_f = _excise_intervals(arr1_f, ent2, act_end, exc2)
                if s2_f.shape[0] != 151:
                    continue
                sc2 = float(np.mean(np.abs(s2_f - tpl_srk2_12)))
                if sc1 + sc2 < best_srk12_sc:
                    best_srk12_sc = sc1 + sc2
                    best_act_ctr = act_end

    c46 = _match_landmark(
        arr1_f,
        t2_ref_s,
        46000,
        800,
        best_act_ctr + (46000 - 39541) + 750,
        best_act_ctr + (46000 - 39541) + 1700,
    ) - (46000 - 44442)
    act_44442 = _match_landmark(arr1_f, t2_ref_s, 44442, 4753, c46 - 4, c46 + 4)
    act_b0_rk_shares = act_44442 + (46410 - 44442)
    act_ct_bperm_0 = act_b0_rk_shares + 1336 + 1449

    c62_b0 = _match_landmark(
        arr1_f,
        t2_ref_s,
        62620,
        667,
        act_b0_rk_shares + (62620 - 46410) + 1500,
        act_b0_rk_shares + (62620 - 46410) + 2900,
        prior_center=act_b0_rk_shares + (62620 - 46410) + 2180.0,
        prior_weight=0.02 * scale,
    )
    act_62620_b0 = _match_landmark(arr1_f, t2_ref_s, 62620, 667, c62_b0 - 4, c62_b0 + 4)
    act_b0_end = act_62620_b0 + 667
    phys_r13_14_0 = arr1_4p[act_62620_b0 : act_62620_b0 + 667].copy()

    act_b1_start = act_b0_end
    c63 = _match_landmark(
        arr1_f, t2_ref_s, 63383, 800, act_b1_start + 96, act_b1_start + 96 + 420
    )
    act_63383 = _match_landmark(arr1_f, t2_ref_s, 63383, 3197, c63 - 4, c63 + 4)
    act_b1_rk_shares = act_63383 + 412
    act_ct_bperm_1 = act_b1_rk_shares + 1336 + 1449
    phys_dec_tail_0 = arr1_4p[act_b1_start : act_b1_start + 13].copy()

    c80_b1 = _match_landmark(
        arr1_f,
        t2_ref_s,
        80005,
        693,
        act_b1_rk_shares + (80005 - 63795) + 1500,
        act_b1_rk_shares + (80005 - 63795) + 2900,
        prior_center=act_b1_rk_shares + (80005 - 63795) + 2180.0,
        prior_weight=0.02 * scale,
    )
    act_80005_b1 = _match_landmark(arr1_f, t2_ref_s, 80005, 693, c80_b1 - 4, c80_b1 + 4)
    phys_r13_14_1 = arr1_4p[act_80005_b1 : act_80005_b1 + 667].copy()
    phys_dec_tail_1 = arr1_4p[act_80005_b1 + 667 : act_80005_b1 + 680].copy()

    bperm_flag = _infer_ct_bperm_swap_from_trace1(arr1, act_ct_bperm_0, act_ct_bperm_1)
    full_phys_0 = np.concatenate([phys_r13_14_0, phys_dec_tail_0])
    full_phys_1 = np.concatenate([phys_r13_14_1, phys_dec_tail_1])
    if bperm_flag == 0:
        r13_14_b0, r13_14_b1 = full_phys_0, full_phys_1
    else:
        r13_14_b0, r13_14_b1 = full_phys_1, full_phys_0

    out.update(
        {
            "poi6a": r13_14_b0[623:680].copy(),
            "poi6b": r13_14_b1[623:680].copy(),
            "bperm": bperm_flag,
        }
    )
    return out


def _compute_poi6_welch_t(poi6: np.ndarray, ks_block: np.ndarray) -> float:
    """Compute max Welch `|t|` across the 4 keystream words (`HW <= 15` vs `HW >= 17`) on POI-6."""
    hw_words = HW8_INT[ks_block].reshape(-1, 4, 4).sum(axis=2)
    max_t = 0.0
    for w in range(4):
        lo = hw_words[:, w] <= 15
        hi = hw_words[:, w] >= 17
        if np.sum(lo) < 5 or np.sum(hi) < 5:
            continue
        x_lo = poi6[lo].astype(np.float64)
        x_hi = poi6[hi].astype(np.float64)
        m_diff = np.abs(np.mean(x_hi, axis=0) - np.mean(x_lo, axis=0))
        v_sum = np.var(x_hi, axis=0, ddof=1) / len(x_hi) + np.var(x_lo, axis=0, ddof=1) / len(x_lo)
        t_stat = float(np.max(m_diff / np.sqrt(np.maximum(v_sum, 1e-12))))
        if t_stat > max_t:
            max_t = t_stat
    return max_t


# ---------------------------------------------------------------------------
# 4. Algebraic AES-256 Key-Schedule Bijection & Coupled Walsh + Tree BP
# ---------------------------------------------------------------------------


def expand_aes256_rk_words(key_bytes: np.ndarray) -> np.ndarray:
    """Expand a 32-byte AES-256 key into 60 uint32 round-key words `W_0..W_59`."""
    W = np.zeros((60, 4), dtype=np.uint8)
    for i in range(8):
        W[i] = key_bytes[4 * i : 4 * i + 4]
    for i in range(8, 60):
        temp = W[i - 1].copy()
        if i % 8 == 0:
            temp = np.roll(temp, -1)
            temp = AES_SBOX[temp]
            temp[0] ^= RCON[(i // 8) - 1]
        elif i % 8 == 4:
            temp = AES_SBOX[temp]
        W[i] = W[i - 8] ^ temp
    return np.frombuffer(W.tobytes(), dtype=np.uint32).copy()


def build_key_labels(keys_bytes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute master key bytes `K_true (N, 8, 4)` and S-box words `W_true (N, 13, 4)`."""
    n_keys = len(keys_bytes)
    rks_all = np.zeros((n_keys, 60), dtype=np.uint32)
    for k in range(n_keys):
        rks_all[k] = expand_aes256_rk_words(keys_bytes[k])

    K_true = np.zeros((n_keys, 8, 4), dtype=np.uint8)
    for w_idx in range(8):
        w_val = rks_all[:, w_idx]
        for q in range(4):
            K_true[:, w_idx, q] = ((w_val >> (8 * q)) & 0xFF).astype(np.uint8)

    W_true = np.zeros((n_keys, 13, 4), dtype=np.uint8)
    for m in range(13):
        w_orig = rks_all[:, 7 + 4 * m]
        for q in range(4):
            W_true[:, m, q] = ((w_orig >> (8 * q)) & 0xFF).astype(np.uint8)

    return K_true, W_true


def fwht_256(a: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard Transform along the last axis of size 256."""
    orig_shape = a.shape
    x = a.reshape(-1, 256).copy()
    h = 1
    while h < 256:
        x = x.reshape(-1, 2, h)
        u = x[:, 0, :].copy()
        v = x[:, 1, :].copy()
        x[:, 0, :] = u + v
        x[:, 1, :] = u - v
        h *= 2
    return x.reshape(orig_shape)


def log_normalize_256(ll: np.ndarray) -> np.ndarray:
    """Normalize log-likelihoods over 256 byte values along the last axis."""
    m = np.max(ll, axis=-1, keepdims=True)
    p = np.exp(np.clip(ll - m, -60.0, 0.0))
    s = np.sum(p, axis=-1, keepdims=True)
    return np.log(np.maximum(p / s, 1e-25))


def xor_convolve_2(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    """Exact XOR convolution of two 256-ary byte probability distributions via FWHT."""
    fa = fwht_256(p_a)
    fb = fwht_256(p_b)
    out = fwht_256(fa * fb) / 256.0
    out = np.maximum(out, 1e-25)
    out /= np.sum(out, axis=-1, keepdims=True)
    return out


def xor_words_list(p_list: list[np.ndarray]) -> np.ndarray:
    """Exact multi-term XOR convolution across a list of `(4, 256)` word distributions."""
    f_prod = fwht_256(p_list[0])
    for p_next in p_list[1:]:
        f_prod *= fwht_256(p_next)
    out = fwht_256(f_prod) / 256.0
    out = np.maximum(out, 1e-25)
    out /= np.sum(out, axis=-1, keepdims=True)
    return out


def sbox_word_dist(p_word_m: np.ndarray, m: int) -> np.ndarray:
    """Push a `(4, 256)` byte distribution through AES-256 SubWord (+ RotWord + Rcon for even `m`)."""
    out = np.zeros((4, 256), dtype=np.float64)
    for q in range(4):
        q_src = (q + 1) % 4 if (m % 2 == 0) else q
        p_sb = p_word_m[q_src, INV_SBOX]
        if (m % 2 == 0) and (q == 0):
            rc = RCON_BY_EVEN_M[m]
            p_sb = p_sb[np.arange(256) ^ rc]
        out[q] = p_sb
    return out


def map_sbox_words_to_key_words(
    W_post_ll: np.ndarray,
    tree_temp: float = 2.0,
    free_temp: float = 1.0,
) -> np.ndarray:
    """Map log-posteriors of S-box words `W_{7+4m}` (`m=0..12`) to master key words `W_0..W_7`."""
    scaled_ll = np.empty_like(W_post_ll, dtype=np.float64)
    for m in range(13):
        temp = tree_temp if m in TREE_WORDS else free_temp
        scaled_ll[m] = log_normalize_256(W_post_ll[m] * temp)
    P_W = np.exp(scaled_ll)
    S = [sbox_word_dist(P_W[m], m) for m in range(7)]

    P_key = np.zeros((8, 4, 256), dtype=np.float64)
    P_key[7] = P_W[0]
    P_key[3] = xor_words_list([P_W[7], S[6]])
    P_key[5] = xor_words_list([P_W[0], P_W[4], S[3]])
    P_key[6] = xor_words_list([P_W[0], P_W[6], S[5]])
    P_key[1] = xor_words_list([P_W[3], P_W[7], S[2], S[6]])
    P_key[2] = xor_words_list([P_W[5], P_W[7], S[4], S[6]])
    P_key[4] = xor_words_list([P_W[0], P_W[2], P_W[4], P_W[6], S[1], S[3], S[5]])
    P_key[0] = xor_words_list([P_W[1], P_W[3], P_W[5], P_W[7], S[0], S[2], S[4], S[6]])
    return np.log(np.maximum(P_key, 1e-25))


def _sbox_word_deterministic(w_bytes: np.ndarray, m: int) -> np.ndarray:
    """Apply AES-256 key-schedule SubWord (+ RotWord + Rcon for even `m`) to a 4-byte word."""
    out = np.empty(4, dtype=np.uint8)
    for q in range(4):
        q_src = (q + 1) % 4 if (m % 2 == 0) else q
        b = int(AES_SBOX[int(w_bytes[q_src])])
        if (m % 2 == 0) and (q == 0):
            b ^= RCON_BY_EVEN_M[m]
        out[q] = b
    return out


def _bijection_w0_to_7_to_master_key(w0_to_7: np.ndarray) -> np.ndarray:
    """Deterministically map `W_{7+4m}` (`m=0..7`, shape `(8, 4)` uint8) to `W_0..W_7`."""
    s_words = [_sbox_word_deterministic(w0_to_7[m], m) for m in range(7)]
    k_words = np.empty((8, 4), dtype=np.uint8)
    k_words[7] = w0_to_7[0]
    k_words[3] = w0_to_7[7] ^ s_words[6]
    k_words[5] = w0_to_7[0] ^ w0_to_7[4] ^ s_words[3]
    k_words[6] = w0_to_7[0] ^ w0_to_7[6] ^ s_words[5]
    k_words[1] = w0_to_7[3] ^ w0_to_7[7] ^ s_words[2] ^ s_words[6]
    k_words[2] = w0_to_7[5] ^ w0_to_7[7] ^ s_words[4] ^ s_words[6]
    k_words[4] = (
        w0_to_7[0] ^ w0_to_7[2] ^ w0_to_7[4] ^ w0_to_7[6] ^ s_words[1] ^ s_words[3] ^ s_words[5]
    )
    k_words[0] = (
        w0_to_7[1]
        ^ w0_to_7[3]
        ^ w0_to_7[5]
        ^ w0_to_7[7]
        ^ s_words[0]
        ^ s_words[2]
        ^ s_words[4]
        ^ s_words[6]
    )
    return k_words


def compute_16state_log_psi(D_obs_m: np.ndarray, sigma2_modes: np.ndarray) -> np.ndarray:
    """Compute `(16, 8)` Gaussian log-potentials over the 16 4-byte bit-tuple states."""
    diff = D_obs_m[None, :, :] - D_STATES_16[:, :, None]
    return -np.sum((diff**2) / (2.0 * sigma2_modes[None, :, :]), axis=1)


def compute_16state_posterior_D(D_obs: np.ndarray, sigma2_modes: np.ndarray) -> np.ndarray:
    """Compute expected 4-mode Walsh bit parities under the 16-state intra-word factor."""
    diff = D_obs[:, :, None, :, :] - D_STATES_16[None, None, :, :, None]
    log_p = -np.sum((diff**2) / (2.0 * sigma2_modes[None, None, None, :, :]), axis=3)
    log_p -= np.max(log_p, axis=2, keepdims=True)
    p = np.exp(np.clip(log_p, -60.0, 0.0))
    p /= np.sum(p, axis=2, keepdims=True)
    return np.einsum("kmsj,sa->kmaj", p, D_STATES_16).astype(np.float32)


def compute_word_walsh_messages(
    D_obs_m: np.ndarray,
    sigma2_modes: np.ndarray,
    ext_byte_ll_m: np.ndarray | None = None,
) -> np.ndarray:
    """Marginalize the 16-state intra-word Walsh factor into `(4, 256)` byte log-messages."""
    log_psi_16 = compute_16state_log_psi(D_obs_m, sigma2_modes)

    if ext_byte_ll_m is not None:
        p_ext = np.exp(log_normalize_256(ext_byte_ll_m))
        p_pos = np.einsum("qx,xj->qj", p_ext, (BYTE_BITS_PM1 > 0).astype(np.float64))
        p_neg = 1.0 - p_pos
        ext_bit_llr = np.log(np.maximum(p_pos, 1e-15) / np.maximum(p_neg, 1e-15))
    else:
        ext_bit_llr = np.zeros((4, 8), dtype=np.float64)

    out_bit_llr = np.zeros((4, 8), dtype=np.float64)
    for q in range(4):
        ext_other = np.zeros((16, 8), dtype=np.float64)
        for qp in range(4):
            if qp != q:
                ext_other += 0.5 * STATES_16[:, qp : qp + 1] * ext_bit_llr[qp : qp + 1, :]
        total_16 = log_psi_16 + ext_other
        mask_pos = STATES_16[:, q] > 0
        mask_neg = ~mask_pos
        m_pos = np.max(total_16[mask_pos], axis=0)
        ll_pos = m_pos + np.log(np.sum(np.exp(total_16[mask_pos] - m_pos[None, :]), axis=0))
        m_neg = np.max(total_16[mask_neg], axis=0)
        ll_neg = m_neg + np.log(np.sum(np.exp(total_16[mask_neg] - m_neg[None, :]), axis=0))
        out_bit_llr[q] = ll_pos - ll_neg

    X_msg_ll = np.einsum("qj,xj->qx", 0.5 * out_bit_llr, BYTE_BITS_PM1)
    return log_normalize_256(X_msg_ll)


def run_coupled_walsh_tree_bp(
    D_obs_k: np.ndarray,
    sigma2_modes: np.ndarray,
    n_outer: int = 4,
) -> np.ndarray:
    """Alternate intra-word 16-state Walsh marginalization with exact 3-term AES-256
    S-box tree belief propagation along the 5-step key-schedule spine.
    """
    W_word_ll = np.zeros((13, 4, 256), dtype=np.float64)
    for m in range(13):
        X_ll_m = compute_word_walsh_messages(D_obs_k[m], sigma2_modes, ext_byte_ll_m=None)
        if m % 2 == 0:
            for q in range(4):
                W_word_ll[m, q] = X_ll_m[(q - 1) % 4]
        else:
            W_word_ll[m] = X_ll_m

    rcon_table = {
        0: (0, 0, 0, 0),
        1: (0x10, 0, 0, 0),
        2: (0, 0, 0, 0),
        3: (0x20, 0, 0, 0),
        4: (0, 0, 0, 0),
    }

    tree_ext_ll = np.zeros((13, 4, 256), dtype=np.float64)
    for outer in range(n_outer):
        P_prior = np.exp(log_normalize_256(W_word_ll))
        fwd_ll = np.zeros((6, 4, 256), dtype=np.float64)

        for s in range(5):
            for q in range(4):
                q_sb = q if (s % 2 == 0) else ((q + 1) % 4)
                rc = rcon_table[s][q]
                ll_in = W_word_ll[s + 7, q_sb] + fwd_ll[s, q_sb]
                p_in = np.exp(log_normalize_256(ll_in))
                p_sb = p_in[INV_SBOX]
                p_leaf = P_prior[s, q]
                if rc != 0:
                    p_leaf = p_leaf[np.arange(256) ^ rc]
                msg_to_next = xor_convolve_2(p_leaf, p_sb)
                fwd_ll[s + 1, q] = np.log(msg_to_next)

        bwd_ll = np.zeros((6, 4, 256), dtype=np.float64)
        tree_ext_ll = np.zeros((13, 4, 256), dtype=np.float64)

        for s in range(4, -1, -1):
            for q in range(4):
                q_sb = q if (s % 2 == 0) else ((q + 1) % 4)
                rc = rcon_table[s][q]
                ll_out = W_word_ll[s + 8, q] + bwd_ll[s + 1, q]
                p_out = np.exp(log_normalize_256(ll_out))
                p_out_rc = p_out[np.arange(256) ^ rc] if rc != 0 else p_out
                p_leaf = P_prior[s, q]
                msg_to_sb = xor_convolve_2(p_out_rc, p_leaf)
                msg_to_in = msg_to_sb[AES_SBOX]
                bwd_ll[s, q_sb] = np.log(np.maximum(msg_to_in, 1e-25))

                ll_in = W_word_ll[s + 7, q_sb] + fwd_ll[s, q_sb]
                p_in = np.exp(log_normalize_256(ll_in))
                p_sb = p_in[INV_SBOX]
                msg_to_leaf = xor_convolve_2(p_out_rc, p_sb)
                tree_ext_ll[s, q] = np.log(msg_to_leaf)

        for s in range(6):
            tree_ext_ll[7 + s] = fwd_ll[s] + bwd_ll[s]

        if outer < n_outer - 1:
            for m in range(13):
                if m in (5, 6):
                    continue
                if m % 2 == 0:
                    ext_X_m = np.array([tree_ext_ll[m, (q + 1) % 4] for q in range(4)])
                else:
                    ext_X_m = tree_ext_ll[m].copy()
                X_ll_new = compute_word_walsh_messages(
                    D_obs_k[m], sigma2_modes, ext_byte_ll_m=ext_X_m
                )
                if m % 2 == 0:
                    for q in range(4):
                        W_word_ll[m, q] = 0.3 * W_word_ll[m, q] + 0.7 * X_ll_new[(q - 1) % 4]
                else:
                    W_word_ll[m] = 0.3 * W_word_ll[m] + 0.7 * X_ll_new

    return log_normalize_256(W_word_ll + tree_ext_ll)


def _compute_leaf_4_12_joints(
    D_obs_k: np.ndarray,
    sigma2_modes: np.ndarray,
    W_post_ll: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute joint 16-state log-potentials for terminal leaf pair `(m=4, m=12)` given `W_51`."""
    log_psi_4 = compute_16state_log_psi(D_obs_k[4], sigma2_modes)
    log_psi_12 = compute_16state_log_psi(D_obs_k[12], sigma2_modes)

    p_Y4_X = np.zeros((4, 256), dtype=np.float64)
    for q in range(4):
        q_w = (q + 1) % 4
        p_w51 = np.exp(log_normalize_256(W_post_ll[11, q_w]))
        p_Y4_X[q] = p_w51[INV_SBOX]

    p1 = np.einsum("qx,xj->qj", p_Y4_X, (BYTE_BITS_PM1 > 0).astype(np.float64))
    p0 = 1.0 - p1
    P_mask = np.ones((16, 8), dtype=np.float64)
    for q in range(4):
        bit_q = MASK_BITS_16[:, q : q + 1]
        P_mask *= np.where(bit_q == 1, p1[None, q, :], p0[None, q, :])

    m12 = np.max(log_psi_12, axis=0, keepdims=True)
    psi12 = np.exp(np.clip(log_psi_12 - m12, -60.0, 0.0))
    conv_12_to_4 = np.einsum("cj,ucj->uj", P_mask, psi12[XOR_TABLE_16])
    joint_4 = log_psi_4 + m12 + np.log(np.maximum(conv_12_to_4, 1e-25))

    m4 = np.max(log_psi_4, axis=0, keepdims=True)
    psi4 = np.exp(np.clip(log_psi_4 - m4, -60.0, 0.0))
    conv_4_to_12 = np.einsum("cj,vcj->vj", P_mask, psi4[XOR_TABLE_16])
    joint_12 = log_psi_12 + m4 + np.log(np.maximum(conv_4_to_12, 1e-25))
    return joint_4, joint_12


def marg_16_sharpened(total_16: np.ndarray, beta: float = 4.0, even_m: bool = True) -> np.ndarray:
    """Sharpened 16-state MAP-consistent marginalization across the 4 bytes of a word."""
    scaled = total_16 if beta == 1.0 else (total_16 * beta)
    out_llr = np.zeros((4, 8), dtype=np.float64)
    for q in range(4):
        mask_pos = STATES_16[:, q] > 0
        mask_neg = ~mask_pos
        mp = np.max(scaled[mask_pos], axis=0)
        ll_p = mp + np.log(np.sum(np.exp(scaled[mask_pos] - mp[None, :]), axis=0))
        mn = np.max(scaled[mask_neg], axis=0)
        ll_n = mn + np.log(np.sum(np.exp(scaled[mask_neg] - mn[None, :]), axis=0))
        out_llr[q] = (ll_p - ll_n) if beta == 1.0 else ((ll_p - ll_n) / max(beta, 1e-6))
    X_ll = log_normalize_256(np.einsum("qj,xj->qx", 0.5 * out_llr, BYTE_BITS_PM1))
    if not even_m:
        return X_ll
    W_ll = np.zeros((4, 256), dtype=np.float64)
    for q in range(4):
        W_ll[q] = X_ll[(q - 1) % 4]
    return W_ll


def update_leaf_16state_posteriors(
    D_obs_k: np.ndarray,
    sigma2_modes: np.ndarray,
    W_post_ll: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Joint 16-state Walsh update for terminal leaf pair `(m=4, m=12)` given `W_51`."""
    joint_4, joint_12 = _compute_leaf_4_12_joints(D_obs_k, sigma2_modes, W_post_ll)
    return (
        marg_16_sharpened(joint_4, beta=1.0, even_m=True),
        marg_16_sharpened(joint_12, beta=1.0, even_m=True),
    )


def _post_ll_to_walsh_expected(post_ll: np.ndarray) -> np.ndarray:
    """Convert `(13, 4, 256)` byte log-posteriors into expected 4-mode Walsh bit parities."""
    p_w = np.exp(log_normalize_256(post_ll))
    B_exp = np.zeros((13, 4, 8), dtype=np.float32)
    for m in range(13):
        for q in range(4):
            q_w = (q + 1) % 4 if (m % 2 == 0) else q
            B_exp[m, q] = np.dot(p_w[m, q_w], BYTE_BITS_PM1)
    out = np.zeros((13, 4, 8), dtype=np.float32)
    out[:, 0, :] = 0.5 * (B_exp[:, 0, :] + B_exp[:, 1, :] + B_exp[:, 2, :] + B_exp[:, 3, :])
    out[:, 1, :] = 0.5 * (B_exp[:, 0, :] - B_exp[:, 1, :] + B_exp[:, 2, :] - B_exp[:, 3, :])
    out[:, 2, :] = 0.5 * (B_exp[:, 0, :] + B_exp[:, 1, :] - B_exp[:, 2, :] - B_exp[:, 3, :])
    out[:, 3, :] = 0.5 * (B_exp[:, 0, :] - B_exp[:, 1, :] - B_exp[:, 2, :] + B_exp[:, 3, :])
    return out


# ---------------------------------------------------------------------------
# 5. ZCA Whitening, Streaming Feature Extraction & Self-Calibrated BP
# ---------------------------------------------------------------------------


def fit_whiten(arr: np.ndarray, max_keys: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Fit unsupervised ZCA whitening parameters `(mu, W)` on POI windows (zero labels used)."""
    L = arr.shape[-1]
    sub = arr[:max_keys] if arr.shape[0] > max_keys else arr
    ac = sub - sub.mean(axis=1, keepdims=True)
    ac /= np.std(ac, axis=tuple(range(1, ac.ndim)), keepdims=True) + 1e-6
    X = ac.reshape(-1, L).astype(np.float64)
    cov = (X.T @ X) / len(X)
    reg = 1e-2 * np.trace(cov) / L
    ev, evec = np.linalg.eigh(cov + reg * np.eye(L))
    W = (evec @ np.diag(1.0 / np.sqrt(ev)) @ evec.T).astype(np.float32)
    return np.zeros((1, L), dtype=np.float32), W


def apply_whiten(arr: np.ndarray, params: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Apply ZCA whitening parameters `(mu, W)` to POI windows."""
    mu, W = params
    if not np.any(mu):
        ac = arr - arr.mean(axis=1, keepdims=True)
        ac /= np.std(ac, axis=tuple(range(1, ac.ndim)), keepdims=True) + 1e-6
        Z = ac @ W
        Z /= np.std(Z, axis=tuple(range(1, Z.ndim)), keepdims=True) + 1e-6
        return Z
    return (arr - mu) @ W


def _fix_bimodal_trace_gain_in_place(
    hdr_a: np.ndarray,
    hdr_b: np.ndarray,
    remap_a: np.ndarray,
    remap_b: np.ndarray,
    pre_sbox: np.ndarray,
    traces: np.ndarray,
) -> None:
    """Repair any per-trace bimodal gain jump across all 6 POI windows of a key in-place."""
    N_K, M_c = traces.shape[:2]
    if M_c < 8:
        return
    for k in range(N_K):
        vec = np.concatenate(
            [
                remap_a[k].reshape(M_c, -1),
                remap_b[k].reshape(M_c, -1),
                traces[k].reshape(M_c, -1),
            ],
            axis=1,
        ).astype(np.float64)
        vec_c = vec - vec.mean(axis=1, keepdims=True)
        t_std = np.std(vec_c, axis=1)
        med_std = float(np.median(t_std)) + 1e-8
        jump_mask = np.abs(t_std / med_std - 1.0) > 0.12
        if np.any(jump_mask) and np.any(~jump_mask):
            ref_wf = vec_c[~jump_mask].mean(axis=0)
            ref_norm2 = float(np.sum(ref_wf ** 2)) + 1e-8
            for t in np.where(jump_mask)[0]:
                scale_t = float(np.dot(vec_c[t], ref_wf) / ref_norm2)
                if 0.4 < scale_t < 2.5:
                    inv_s = np.float32(1.0 / scale_t)
                    hdr_a[k, t] *= inv_s
                    hdr_b[k, t] *= inv_s
                    remap_a[k, t] *= inv_s
                    remap_b[k, t] *= inv_s
                    pre_sbox[k, t] *= inv_s
                    traces[k, t] *= inv_s


def _whiten_and_build_base_tensors(
    hdr_a: np.ndarray,
    hdr_b: np.ndarray,
    remap_a: np.ndarray,
    remap_b: np.ndarray,
    pre_sbox: np.ndarray,
    traces: np.ndarray,
    whiten_models: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Whiten all six POI windows and assemble shared remap/S-box base tensors."""
    _fix_bimodal_trace_gain_in_place(hdr_a, hdr_b, remap_a, remap_b, pre_sbox, traces)
    N_K, M_c = traces.shape[:2]
    Za = apply_whiten(remap_a, whiten_models["ra"])
    Zb = apply_whiten(remap_b, whiten_models["rb"])
    Z_ha = apply_whiten(hdr_a, whiten_models["ha"])
    Z_hb = apply_whiten(hdr_b, whiten_models["hb"])
    Z_pre = apply_whiten(pre_sbox, whiten_models["pre"])
    Z = apply_whiten(traces, whiten_models["tr"])

    Za_ext = np.zeros((N_K, M_c, 64, 45), dtype=np.float32)
    Zb_ext = np.zeros((N_K, M_c, 64, 45), dtype=np.float32)
    Za_ext[:, :, :, :30] = Za
    Zb_ext[:, :, :, :30] = Zb
    Za_ext[:, :, :63, 30:] = Za[:, :, 1:, :15]
    Zb_ext[:, :, :63, 30:] = Zb[:, :, 1:, :15]

    Z_iter68 = np.stack(
        [Z[:, :, :, 22 + 50 * it : 22 + 50 * it + 64] for it in range(4)], axis=3
    )
    S_m68 = Z_iter68.sum(axis=3)
    Z_tilde68 = Z_iter68 - Z_iter68.mean(axis=3, keepdims=True)

    Z_pro = Z[:, :, :, :22].mean(axis=2)
    Z_r10_loop = S_m68[:, :, :, 35:48].mean(axis=2)
    Za_mean = Za.mean(axis=2)
    Zb_mean = Zb.mean(axis=2)
    Fa = np.concatenate([Z_ha, Za_mean[:, :, 5:22]], axis=2)
    Fb = np.concatenate([Z_hb, Zb_mean[:, :, 5:22]], axis=2)
    iu_f = np.triu_indices(Fa.shape[2])
    Qa = Fa[:, :, iu_f[0]] * Fa[:, :, iu_f[1]]
    Qb = Fb[:, :, iu_f[0]] * Fb[:, :, iu_f[1]]
    M_base = np.concatenate([Z_pro, Z_r10_loop, Fa, Fb, Qa, Qb], axis=2)
    M_base -= M_base.mean(axis=1, keepdims=True)

    return {
        "Z_pre": Z_pre,
        "Z_iter68": Z_iter68,
        "S_m68": S_m68,
        "Z_tilde68": Z_tilde68,
        "Za_ext": Za_ext,
        "Zb_ext": Zb_ext,
        "Z_pro": Z_pro,
        "Z_r10_loop": Z_r10_loop,
        "M_base": M_base,
    }


def prepare_operator_matrices(
    mdl: dict[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]]:
    """Precompile symmetric quadratic operator matrices (`W_c`, `W_i`, `W_q`) for fast streaming."""
    ops_fast: list[dict[str, Any]] = []
    for j in range(8):
        op = mdl["d0_ops_full"][j]
        n_act = op["n_act"]
        iu = op["iu"]
        w_c_raw = (op["w_c"] / op["sd_c"][0]).astype(np.float32)
        bias_c = float(op["y_c_mean"] - np.dot(op["mu_c"][0], w_c_raw))
        W_c = np.zeros((n_act, n_act), dtype=np.float32)
        W_c[iu[0], iu[1]] = 0.5 * w_c_raw
        W_c[iu[1], iu[0]] = 0.5 * w_c_raw
        W_c[np.diag_indices(n_act)] = w_c_raw[iu[0] == iu[1]]

        if j < 4:
            w_i_raw = (op["w_i"] / op["sd_i"][0]).astype(np.float32)
            bias_i = float(op["y_i_mean"] - np.dot(op["mu_i"][0], w_i_raw))
            W_i = np.zeros((n_act, n_act), dtype=np.float32)
            W_i[iu[0], iu[1]] = 0.5 * w_i_raw
            W_i[iu[1], iu[0]] = 0.5 * w_i_raw
            W_i[np.diag_indices(n_act)] = w_i_raw[iu[0] == iu[1]]
        else:
            W_i = None
            bias_i = 0.0

        ops_fast.append(
            {
                "j": j,
                "n_act": n_act,
                "act_u": op["act_u"],
                "mu_h": op["mu_h"],
                "sd_h": op["sd_h"],
                "u_h": op["u_h"],
                "s_h": op["s_h"],
                "vt_h": op["vt_h"],
                "u1": op["u1"],
                "W_c": W_c,
                "bias_c": bias_c,
                "W_i": W_i,
                "bias_i": bias_i,
            }
        )

    perm_u = SBOX_QUAD_PERM_U.copy()
    iu_p = np.triu_indices(len(perm_u))
    psi_mats: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    for w_raw in [mdl["w1_raw"], mdl["w2_raw"], mdl["w3_raw"]]:
        w_iter = w_raw[:64].astype(np.float32)
        w_pre = w_raw[64 : 64 + 4 * 56].reshape(4, 56).astype(np.float32)
        w_quad = w_raw[64 + 4 * 56 :].astype(np.float32)
        W_q = np.zeros((len(perm_u), len(perm_u)), dtype=np.float32)
        W_q[iu_p[0], iu_p[1]] = 0.5 * w_quad
        W_q[iu_p[1], iu_p[0]] = 0.5 * w_quad
        W_q[np.diag_indices(len(perm_u))] = w_quad[iu_p[0] == iu_p[1]]
        bias_a = float((mdl["mu_f_full"] @ w_raw)[0])
        psi_mats.append((w_iter, w_pre, W_q, bias_a))

    return ops_fast, perm_u, psi_mats


def extract_chunk_features(
    hdr_a: np.ndarray,
    hdr_b: np.ndarray,
    remap_a: np.ndarray,
    remap_b: np.ndarray,
    pre_sbox: np.ndarray,
    traces: np.ndarray,
    whiten_models: dict[str, tuple[np.ndarray, np.ndarray]],
    ops_fast: list[dict[str, Any]],
    perm_u: np.ndarray,
    psi_mats: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
) -> dict[str, np.ndarray]:
    """Extract compact per-shard and per-trace sufficient statistics for a chunk of `M_c` traces/key."""
    N_K, M_c = traces.shape[:2]
    n_sh = M_c // 64
    base = _whiten_and_build_base_tensors(
        hdr_a, hdr_b, remap_a, remap_b, pre_sbox, traces, whiten_models
    )
    Z_pre = base["Z_pre"]
    Z_iter68 = base["Z_iter68"]
    S_m68 = base["S_m68"]
    Z_tilde68 = base["Z_tilde68"]
    Za_ext = base["Za_ext"]
    Zb_ext = base["Zb_ext"]
    Z_pro = base["Z_pro"]
    Z_r10_loop = base["Z_r10_loop"]
    M_base = base["M_base"]

    d0_dir_sh = np.zeros((N_K, n_sh, 13, 8), dtype=np.float32)
    G_sh = np.zeros((N_K, n_sh, 13, 13, 8), dtype=np.float32)
    vals_i_sh = np.zeros((N_K, n_sh, 13, 4), dtype=np.float32)

    v_rem_all = np.zeros((N_K, M_c, 13, 4, 8), dtype=np.float32)
    s_r0_all = np.zeros((N_K, M_c, 13, 8), dtype=np.float32)
    s_r1_all = np.zeros((N_K, M_c, 13, 8), dtype=np.float32)
    z_r0_all = np.zeros((N_K, M_c, 13, 4, 8), dtype=np.float32)
    z_r1_all = np.zeros((N_K, M_c, 13, 4, 8), dtype=np.float32)

    c_q = 63 - np.arange(64)
    for j in range(8):
        op = ops_fast[j]
        act_u = op["act_u"]
        S_act = S_m68[:, :, :, act_u]
        Z_iter_act = Z_iter68[:, :, :, :, act_u]
        Z_tilde_act = Z_tilde68[:, :, :, :, act_u]

        if j >= 2:
            walsh_sign = (1.0 - 2.0 * ((c_q >> (j - 2)) & 1)).astype(np.float32)
            Ha = np.einsum("q,ktqu->ktu", walsh_sign, Za_ext) / 64.0
            Hb = np.einsum("q,ktqu->ktu", walsh_sign, Zb_ext) / 64.0
            H_ab = np.concatenate([Ha, Hb, Z_pro, Z_r10_loop], axis=2)
            H_ab -= H_ab.mean(axis=1, keepdims=True)
        else:
            H_ab = M_base

        H_all_n = ((H_ab.reshape(-1, H_ab.shape[2]) - op["mu_h"]) / op["sd_h"]).reshape(
            N_K, M_c, -1
        )

        v_rem = np.zeros((N_K, M_c, 13, 4), dtype=np.float32)
        d0_tr = np.zeros((N_K, M_c, 13), dtype=np.float32)
        for r_idx in range(2):
            h_r = H_all_n @ op["u_h"][:, r_idx]
            s_r = S_act @ op["vt_h"][r_idx]
            z_r = Z_tilde_act @ op["vt_h"][r_idx]
            d0_tr += op["s_h"][r_idx] * (h_r[:, :, None] * s_r)
            v_rem += op["s_h"][r_idx] * h_r[:, :, None, None] * z_r
            if j > 0:
                if r_idx == 0:
                    s_r0_all[:, :, :, j] = s_r
                    z_r0_all[:, :, :, :, j] = z_r
                else:
                    s_r1_all[:, :, :, j] = s_r
                    z_r1_all[:, :, :, :, j] = z_r
        v_rem_all[:, :, :, :, j] = v_rem

        if j == 0:
            s_r0_all[:, :, :, 0] = S_act @ op["u1"]
            z_r0_all[:, :, :, :, 0] = Z_tilde_act @ op["u1"]

        d0_dir_sh[:, :, :, j] = d0_tr.reshape(N_K, n_sh, 64, 13).mean(axis=2)

        S_Wc = S_act @ op["W_c"]
        S_act_sh = S_act.reshape(N_K, n_sh, 64, 13, -1)
        S_Wc_sh = S_Wc.reshape(N_K, n_sh, 64, 13, -1)
        G_sh[:, :, :, :, j] = (
            np.einsum("kbtmu,kbtnu->kbmn", S_act_sh, S_Wc_sh) / 64.0 + op["bias_c"]
        )

        if j < 4:
            quad_s = np.sum(S_act * (S_act @ op["W_i"]), axis=-1)
            quad_z = np.sum(Z_iter_act * (Z_iter_act @ op["W_i"]), axis=-1).sum(axis=3)
            vals_i_tr = quad_s - 0.5 * quad_z + op["bias_i"]
            vals_i_sh[:, :, :, j] = vals_i_tr.reshape(N_K, n_sh, 64, 13).mean(axis=2)

    Z_p_sub = Z_tilde68[:, :, :, :, perm_u]
    psi_raw_list = []
    for w_iter, w_pre, W_q, _ in psi_mats:
        q_raw = np.sum(Z_p_sub * (Z_p_sub @ W_q), axis=-1)
        q_raw -= q_raw.mean(axis=3, keepdims=True)
        p_raw = (Z_tilde68 @ w_iter) + np.einsum("ktmp,ip->ktmi", Z_pre, w_pre) + q_raw
        psi_raw_list.append(p_raw.astype(np.float32))

    return {
        "d0_dir_sh": d0_dir_sh,
        "G_sh": G_sh,
        "vals_i_sh": vals_i_sh,
        "v_rem": v_rem_all,
        "s_r0": s_r0_all,
        "s_r1": s_r1_all,
        "z_r0": z_r0_all,
        "z_r1": z_r1_all,
        "psi1_raw": psi_raw_list[0],
        "psi2_raw": psi_raw_list[1],
        "psi3_raw": psi_raw_list[2],
    }


W_LOO_BY_BIT: tuple[float, ...] = (0.50, 0.35, 0.30, 0.30, 0.15, 0.15, 0.05, 0.05)


def _compute_v_iter_slice(
    stream_data: dict[str, np.ndarray],
    t0: int,
    t1: int,
    D0_for_v: np.ndarray,
    ops_fast: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-iteration unmasked carrier projections `v_iter` on trace slice `[t0:t1]`."""
    v_rem = stream_data["v_rem"][:, t0:t1].astype(np.float32)
    v_rem = v_rem / (np.std(v_rem, axis=(1, 3), keepdims=True) + 1e-6)
    s_r0 = stream_data["s_r0"][:, t0:t1].astype(np.float32)
    s_r1 = stream_data["s_r1"][:, t0:t1].astype(np.float32)
    z_r0 = stream_data["z_r0"][:, t0:t1].astype(np.float32)
    z_r1 = stream_data["z_r1"][:, t0:t1].astype(np.float32)

    v_iter = np.zeros_like(v_rem)
    for j in range(8):
        D0_j = D0_for_v[:, :, j]
        w_loo = W_LOO_BY_BIT[j]
        w_rem = 1.0 - w_loo
        if j == 0:
            tot_m1 = np.einsum("km,ktm->kt", D0_j, s_r0[:, :, :, 0])[:, :, None]
            tot_m1 /= np.std(tot_m1, axis=1, keepdims=True) + 1e-6
            v_loo_u1 = tot_m1[:, :, :, None] * z_r0[:, :, :, :, 0]
            v_loo_u1 /= np.std(v_loo_u1, axis=(1, 3), keepdims=True) + 1e-6
            v_iter[:, :, :, :, 0] = w_rem * v_rem[:, :, :, :, 0] + w_loo * v_loo_u1
        else:
            s_h = ops_fast[j]["s_h"]
            tot_0 = np.einsum("km,ktm->kt", D0_j, s_r0[:, :, :, j])[:, :, None]
            tot_0 /= np.std(tot_0, axis=1, keepdims=True) + 1e-6
            tot_1 = np.einsum("km,ktm->kt", D0_j, s_r1[:, :, :, j])[:, :, None]
            tot_1 /= np.std(tot_1, axis=1, keepdims=True) + 1e-6
            v_loo = (
                s_h[0] * tot_0[:, :, :, None] * z_r0[:, :, :, :, j]
                + s_h[1] * tot_1[:, :, :, None] * z_r1[:, :, :, :, j]
            )
            v_loo /= np.std(v_loo, axis=(1, 3), keepdims=True) + 1e-6
            v_iter[:, :, :, :, j] = w_rem * v_rem[:, :, :, :, j] + w_loo * v_loo
    return v_rem, s_r0, s_r1, z_r0, z_r1, v_iter


def _calibrate_tensor_against_spine(
    X: np.ndarray,
    Y_spine: np.ndarray,
    prior_s2: np.ndarray,
    conv_keys: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate `(N_K, 13, 4, 8)` tensor `X` against spine posteriors `Y_spine` `(N_K, 9, 4, 8)`
    by removing per-`(k, a, j)` spine-consistent intercept `b_{k,a,j}` and per-`(k, a)` gain `g_{k,a}`.
    """
    N_K = X.shape[0]
    sp = M_TREE_STRONG
    if conv_keys is None or len(conv_keys) == 0:
        conv_keys = np.arange(N_K)
    X_cal = np.zeros_like(X, dtype=np.float64)
    s2_emp = np.zeros((4, 8), dtype=np.float64)

    g_ka = np.ones((N_K, 4), dtype=np.float64)
    for k in conv_keys:
        for a in range(4):
            x_sp = X[k, sp, a, :].astype(np.float64)
            y_sp = Y_spine[k, :, a, :].astype(np.float64)
            x_c = x_sp - x_sp.mean(axis=0, keepdims=True)
            y_c = y_sp - y_sp.mean(axis=0, keepdims=True)
            w_j = 1.0 / np.maximum(prior_s2[a], 0.05)
            num = float(np.sum(w_j[None, :] * x_c * y_c))
            den = float(np.sum(w_j[None, :] * (y_c ** 2))) + 1e-6
            g_ka[k, a] = max(0.12, num / den)

    med_g = np.median(g_ka[conv_keys], axis=0) if len(conv_keys) > 0 else np.ones(4)
    conv_set = set(conv_keys.tolist())
    for k in range(N_K):
        if k not in conv_set:
            g_ka[k] = med_g

    b_all = np.zeros((N_K, 4, 8), dtype=np.float64)
    for k in range(N_K):
        for a in range(4):
            x_sc = X[k, :, a, :].astype(np.float64) / g_ka[k, a]
            if k in conv_set:
                b_all[k, a, :] = np.mean(x_sc[sp, :] - Y_spine[k, :, a, :], axis=0)
            else:
                b_all[k, a, :] = np.mean(x_sc, axis=0)

    b_batch = np.mean(b_all[conv_keys], axis=0) if len(conv_keys) > 0 else np.zeros((4, 8))
    w_batch = float(len(conv_keys) - 1) / float(len(conv_keys) + 1) if len(conv_keys) > 1 else 0.0

    for a in range(4):
        for j in range(8):
            res_list = []
            for k in range(N_K):
                x_scaled = X[k, :, a, j].astype(np.float64) / g_ka[k, a]
                b_kj = (
                    (1.0 - w_batch) * b_all[k, a, j] + w_batch * b_batch[a, j]
                    if k in conv_set
                    else b_batch[a, j]
                )
                X_cal[k, :, a, j] = x_scaled - b_kj
                if k in conv_set:
                    res_list.append(X_cal[k, sp, a, j] - Y_spine[k, :, a, j])
            if res_list:
                all_res = np.concatenate(res_list)
                s2_emp[a, j] = float(np.mean(all_res ** 2)) * (9.0 / 8.0)
            else:
                s2_emp[a, j] = prior_s2[a, j]

    n_obs = float(9 * max(1, len(conv_keys)))
    n_prior = 2.5
    s2_cal = np.zeros((4, 8), dtype=np.float64)
    for a in range(4):
        alpha_a = float(np.median(s2_emp[a] / np.maximum(prior_s2[a], 1e-4)))
        prior_a = alpha_a * prior_s2[a]
        s2_cal[a] = np.maximum(1e-4, (n_obs * s2_emp[a] + n_prior * prior_a) / (n_obs + n_prior))
    return X_cal, s2_cal, g_ka


def _calibrate_1d_feature_against_spine(
    F: np.ndarray,
    Y_spine: np.ndarray,
    g_ref: np.ndarray,
    conv_keys: np.ndarray,
    prior_slope: float = 0.0,
    n_reg: float = 18.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Debias and normalize a `(N_K, 13, 8)` feature tensor `F` against `Y_spine` `(N_K, 9, 8)`
    using reference per-key scale `g_ref` `(N_K,)` and optional empirical-Bayes slope shrinkage.
    """
    N_K = F.shape[0]
    sp = M_TREE_STRONG
    F_sc = F.astype(np.float64) / g_ref[:, None, None]
    F_deb = np.zeros_like(F_sc)
    slope_j = np.zeros(8, dtype=np.float64)
    mse_j = np.ones(8, dtype=np.float64)
    conv_set = set(conv_keys.tolist())
    n_obs = float(9 * max(1, len(conv_keys)))

    for j in range(8):
        f_sp = F_sc[conv_keys][:, sp, j]
        y_sp = Y_spine[conv_keys, :, j]
        f_c = f_sp - f_sp.mean(axis=1, keepdims=True)
        y_c = y_sp - y_sp.mean(axis=1, keepdims=True)
        num = float(np.sum(f_c * y_c))
        den = float(np.sum(y_c ** 2)) + 1e-6
        sl_ols = num / den
        if prior_slope > 0.0:
            sl = (n_obs * sl_ols + n_reg * prior_slope) / (n_obs + n_reg)
            sl = float(np.clip(sl, 0.55, 1.30))
        else:
            sl = sl_ols
        slope_j[j] = sl

        b_k = np.zeros(N_K, dtype=np.float64)
        for k in range(N_K):
            if k in conv_set:
                b_k[k] = float(np.mean(F_sc[k, sp, j] - sl * Y_spine[k, :, j]))
            else:
                b_k[k] = float(np.mean(F_sc[k, :, j]))
        b_batch = float(np.mean(b_k[conv_keys])) if len(conv_keys) > 0 else 0.0
        w_b = float(len(conv_keys) - 1) / float(len(conv_keys) + 1) if len(conv_keys) > 1 else 0.0
        for k in range(N_K):
            bk = (1.0 - w_b) * b_k[k] + w_b * b_batch if k in conv_set else b_batch
            F_deb[k, :, j] = F_sc[k, :, j] - bk

        if sl > 0.15:
            res = (F_deb[conv_keys][:, sp, j] / sl) - y_sp
            mse_j[j] = max(1e-3, float(np.mean(res ** 2)) * (9.0 / 7.5))
        else:
            mse_j[j] = 100.0
    return F_deb, slope_j, mse_j


def extract_batch_p2_features_at_m(
    stream_data: dict[str, np.ndarray],
    M_eval: int,
    mdl: dict[str, Any],
    ops_fast: list[dict[str, Any]],
    psi_mats: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
) -> dict[str, np.ndarray]:
    """Run Pass-1 S-Box Tree BP and extract compact Pass-2 sufficient statistics at budget `M_eval`."""
    N_K = stream_data["d0_dir_sh"].shape[0]
    n_sh = M_eval // 64
    sp = M_TREE_STRONG

    D_pred = np.zeros((N_K, 13, 4, 8), dtype=np.float32)
    G_all_j = np.zeros((N_K, 13, 13, 8), dtype=np.float64)

    d0_dir_all = stream_data["d0_dir_sh"][:, :n_sh].mean(axis=1)
    G_mean_all = stream_data["G_sh"][:, :n_sh].mean(axis=1).astype(np.float64)
    vals_i_all = np.maximum(0.0, stream_data["vals_i_sh"][:, :n_sh].mean(axis=1))

    for j in range(8):
        d0_dir = d0_dir_all[:, :, j]
        G_all = G_mean_all[:, :, :, j].copy()
        for m in range(13):
            G_all[:, m, m] = 0.0
        wb = mdl["w_blend_all"][j]
        if j < 4 and wb[1] > 0:
            for k in range(N_K):
                G_k = G_all[k].copy()
                np.fill_diagonal(G_k, vals_i_all[k, :, j])
                ev, evec = np.linalg.eigh(G_k)
                v_eig = evec[:, -1] * np.sqrt(max(0.0, ev[-1]))
                if np.dot(v_eig, d0_dir[k]) < 0:
                    v_eig = -v_eig
                vc = v_eig - np.mean(v_eig)
                dc = d0_dir[k] - np.mean(d0_dir[k])
                cos_sim = float(np.dot(vc, dc) / (np.linalg.norm(vc) * np.linalg.norm(dc) + 1e-8))
                coh_w = max(0.0, min(1.0, (cos_sim - 0.65) / 0.25))
                D_pred[k, :, 0, j] = (
                    (wb[0] + wb[1] * (1.0 - coh_w)) * d0_dir[k] + (wb[1] * coh_w) * v_eig
                )
        else:
            D_pred[:, :, 0, j] = wb[0] * d0_dir
        G_all_j[:, :, :, j] = G_all

    def norm_p(p: np.ndarray) -> np.ndarray:
        p = p - p.mean(axis=3, keepdims=True)
        p /= np.std(p, axis=(1, 3), keepdims=True) + 1e-6
        return p

    def run_d123_components(
        D0_for_v: np.ndarray,
        D0_unb_for_16: np.ndarray | None,
        var_scale_16: float,
        D_bp_for_h: np.ndarray | None = None,
    ) -> tuple[Any, ...]:
        slice_len = 1024
        D_stage_sum = np.zeros((N_K, 13, 3, 8), dtype=np.float64)
        D_k4_sum = np.zeros((N_K, 13, 3, 8), dtype=np.float64)
        G_sr0_sum = np.zeros((N_K, 13, 13, 8), dtype=np.float64) if D_bp_for_h is not None else None
        d0_h0_sum = np.zeros((N_K, 13, 8), dtype=np.float64) if D_bp_for_h is not None else None
        d0_h1_sum = np.zeros((N_K, 13, 8), dtype=np.float64) if D_bp_for_h is not None else None
        D_sp = D_bp_for_h[:, sp] if D_bp_for_h is not None else None

        for t0 in range(0, M_eval, slice_len):
            t1 = min(M_eval, t0 + slice_len)
            L_s = t1 - t0
            v_rem, s_r0, s_r1, z_r0, z_r1, v_iter = _compute_v_iter_slice(
                stream_data, t0, t1, D0_for_v, ops_fast
            )

            p1_raw = stream_data["psi1_raw"][:, t0:t1].astype(np.float32)
            p2_raw = stream_data["psi2_raw"][:, t0:t1].astype(np.float32)
            p3_raw = stream_data["psi3_raw"][:, t0:t1].astype(np.float32)

            for a_idx, p_raw in enumerate([p1_raw, p2_raw, p3_raw]):
                D_stage_sum[:, :, a_idx, :] += np.einsum("ktmi,ktmij->kmj", p_raw, v_iter)

            p1_ptr, p2_ptr, p3_ptr = norm_p(p1_raw), norm_p(p2_raw), norm_p(p3_raw)
            pk1 = p2_ptr * p3_ptr
            pk1 -= pk1.mean(axis=3, keepdims=True)
            pk2 = p1_ptr * p3_ptr
            pk2 -= pk2.mean(axis=3, keepdims=True)
            pk3 = p1_ptr * p2_ptr
            pk3 -= pk3.mean(axis=3, keepdims=True)
            for idx, pk in enumerate([pk1, pk2, pk3]):
                D_k4_sum[:, :, idx, :] += np.einsum("ktmi,ktmij->kmj", pk, v_iter)

            if (
                D_sp is not None
                and G_sr0_sum is not None
                and d0_h0_sum is not None
                and d0_h1_sum is not None
            ):
                v_rem_sp = v_rem[:, :, sp]
                p1_sp = p1_ptr[:, :, sp]
                p2_sp = p2_ptr[:, :, sp]
                p3_sp = p3_ptr[:, :, sp]
                psi1_v = norm_p(
                    np.einsum("l,kml,ktmil->ktmi", mdl["r_oof_bits"], D_sp[:, :, 1, :], v_rem_sp)
                )
                psi2_v = norm_p(
                    np.einsum("l,kml,ktmil->ktmi", mdl["r_oof_bits"], D_sp[:, :, 2, :], v_rem_sp)
                )
                psi3_v = norm_p(
                    np.einsum("l,kml,ktmil->ktmi", mdl["r_oof_bits"], D_sp[:, :, 3, :], v_rem_sp)
                )
                psi1_comb = norm_p(p1_sp + 0.65 * psi1_v)
                psi2_comb = norm_p(p2_sp + 0.65 * psi2_v)
                psi3_comb = norm_p(p3_sp + 0.65 * psi3_v)

                z_r0_sp = z_r0[:, :, sp]
                z_r1_sp = z_r1[:, :, sp]
                s_r0_c = s_r0 - s_r0.mean(axis=1, keepdims=True)
                s_r1_c = s_r1 - s_r1.mean(axis=1, keepdims=True)
                for j in range(8):
                    G_sr0_sum[:, :, :, j] += np.einsum(
                        "ktm,ktn->kmn", s_r0_c[:, :, :, j], s_r0_c[:, :, :, j]
                    )
                    B_tilde_j = (
                        psi1_comb * D_sp[:, None, :, None, 1, j]
                        + psi2_comb * D_sp[:, None, :, None, 2, j]
                        + psi3_comb * D_sp[:, None, :, None, 3, j]
                    )
                    h_sp0 = np.einsum("ktmi,ktmi->ktm", B_tilde_j, z_r0_sp[:, :, :, :, j])
                    h_sp0 -= h_sp0.mean(axis=1, keepdims=True)
                    h_tot0 = h_sp0.sum(axis=2, keepdims=True)
                    h_sp0_13 = np.zeros((N_K, L_s, 13), dtype=np.float32)
                    h_sp0_13[:, :, sp] = h_sp0
                    hm0_all = h_tot0 - h_sp0_13
                    hm0_all /= np.std(hm0_all, axis=1, keepdims=True) + 1e-6
                    d0_h0_sum[:, :, j] += np.einsum("ktm,ktm->km", hm0_all, s_r0_c[:, :, :, j])
                    if j > 0:
                        h_sp1 = np.einsum("ktmi,ktmi->ktm", B_tilde_j, z_r1_sp[:, :, :, :, j])
                        h_sp1 -= h_sp1.mean(axis=1, keepdims=True)
                        h_tot1 = h_sp1.sum(axis=2, keepdims=True)
                        h_sp1_13 = np.zeros((N_K, L_s, 13), dtype=np.float32)
                        h_sp1_13[:, :, sp] = h_sp1
                        hm1_all = h_tot1 - h_sp1_13
                        hm1_all /= np.std(hm1_all, axis=1, keepdims=True) + 1e-6
                        d0_h1_sum[:, :, j] += np.einsum("ktm,ktm->km", hm1_all, s_r1_c[:, :, :, j])

        D_raw_stage = np.zeros((N_K, 13, 4, 8), dtype=np.float32)
        D_raw_stage[:, :, 0, :] = D0_for_v
        for a_idx in range(3):
            bias_a = psi_mats[a_idx][3]
            D_raw_stage[:, :, a_idx + 1, :] = (
                D_stage_sum[:, :, a_idx, :] / (float(M_eval) * mdl["s_j_vec"][None, None, :])
            ) - bias_a
        D_k4 = (D_k4_sum / (float(M_eval) * mdl["s_j_vec"][None, None, :])).astype(np.float32)

        D_unb_init = D_raw_stage / mdl["slopes_init"][None, None, :, :]
        if D0_unb_for_16 is not None:
            D_unb_init[:, :, 0, :] = D0_unb_for_16

        if D_sp is not None:
            D_unb_init_cal, s2_16_cal, _ = _calibrate_tensor_against_spine(
                D_unb_init, D_sp, mdl["sigma2_init"] * var_scale_16
            )
            D_post_16 = compute_16state_posterior_D(D_unb_init_cal, s2_16_cal)
            D_post_16[:, sp, :, :] = 0.75 * D_sp + 0.25 * D_post_16[:, sp, :, :]
        else:
            D_unb_init_c = D_unb_init.copy()
            for j in (0, 7):
                D_unb_init_c[:, :, :, j] -= D_unb_init_c[:, :, :, j].mean(axis=1, keepdims=True)
            s2_init_sc = mdl["sigma2_init"] * var_scale_16
            for k in range(N_K):
                for a in range(4):
                    rms_ka = float(np.sqrt(np.mean(D_unb_init_c[k, :, a, :] ** 2)))
                    exp_rms = float(np.sqrt(1.0 + np.mean(s2_init_sc[a])))
                    g_init = max(0.25, rms_ka / exp_rms)
                    D_unb_init_c[k, :, a, :] /= g_init
            D_post_16 = compute_16state_posterior_D(D_unb_init_c, s2_init_sc)

        feat_bits_sum = np.zeros((N_K, 13, 3, 8), dtype=np.float64)
        feat_bc_sum = np.zeros((N_K, 13, 3, 8), dtype=np.float64)

        for t0 in range(0, M_eval, slice_len):
            t1 = min(M_eval, t0 + slice_len)
            _, _, _, _, _, v_iter = _compute_v_iter_slice(
                stream_data, t0, t1, D0_for_v, ops_fast
            )
            v_norm = v_iter / (np.std(v_iter, axis=(1, 3), keepdims=True) + 1e-6)
            psi_tot = {
                a: np.einsum("l,kml,ktmil->ktmi", mdl["r_oof_bits"], D_post_16[:, :, a, :], v_norm)
                for a in (1, 2, 3)
            }
            for a_idx, a in enumerate((1, 2, 3)):
                mb, mc = [m_a for m_a in (1, 2, 3) if m_a != a]
                for j in range(8):
                    psi_a_bits = (
                        psi_tot[a]
                        - mdl["r_oof_bits"][j]
                        * D_post_16[:, :, a, j][:, None, :, None]
                        * v_norm[:, :, :, :, j]
                    )
                    psi_b_bits = (
                        psi_tot[mb]
                        - mdl["r_oof_bits"][j]
                        * D_post_16[:, :, mb, j][:, None, :, None]
                        * v_norm[:, :, :, :, j]
                    )
                    psi_c_bits = (
                        psi_tot[mc]
                        - mdl["r_oof_bits"][j]
                        * D_post_16[:, :, mc, j][:, None, :, None]
                        * v_norm[:, :, :, :, j]
                    )
                    psi_bc_k4 = psi_b_bits * psi_c_bits
                    psi_bc_k4 -= psi_bc_k4.mean(axis=3, keepdims=True)
                    feat_bits_sum[:, :, a_idx, j] += np.einsum(
                        "ktmi,ktmi->km", psi_a_bits, v_norm[:, :, :, :, j]
                    )
                    feat_bc_sum[:, :, a_idx, j] += np.einsum(
                        "ktmi,ktmi->km", psi_bc_k4, v_norm[:, :, :, :, j]
                    )

        feat_bits = (feat_bits_sum / (float(M_eval) * 4.0)).astype(np.float32)
        feat_bc_k4 = (feat_bc_sum / (float(M_eval) * 4.0)).astype(np.float32)
        D_ca = D_raw_stage.copy()
        for a_idx, a in enumerate((1, 2, 3)):
            for j in range(8):
                A_ho = np.stack(
                    [
                        D_raw_stage[:, :, a, j].reshape(-1),
                        feat_bits[:, :, a_idx, j].reshape(-1),
                        D_k4[:, :, a_idx, j].reshape(-1),
                        feat_bc_k4[:, :, a_idx, j].reshape(-1),
                    ],
                    axis=1,
                )
                D_ca[:, :, a, j] = (A_ho @ mdl["w_ca_models"][(a, j)]).reshape(N_K, 13)
        D_unb_out = D_ca / mdl["slopes_p1"][None, None, :, :]
        if D_bp_for_h is None or G_sr0_sum is None or d0_h0_sum is None or d0_h1_sum is None:
            return D_raw_stage, feat_bits, D_k4, feat_bc_k4, D_unb_out
        return (
            D_raw_stage,
            feat_bits,
            D_k4,
            feat_bc_k4,
            D_unb_out,
            G_sr0_sum / float(M_eval),
            (d0_h0_sum / float(M_eval)).astype(np.float32),
            (d0_h1_sum / float(M_eval)).astype(np.float32),
        )

    var_scale = float(np.sqrt(256.0 / float(M_eval)))
    D0_for_v_p1 = D_pred[:, :, 0, :].copy()
    for j in (0, 7):
        D0_for_v_p1[:, :, j] -= D0_for_v_p1[:, :, j].mean(axis=1, keepdims=True)

    _, _, _, _, D_unb_p1 = run_d123_components(D0_for_v_p1, None, var_scale, None)

    D_unb_p1_init = D_unb_p1.copy()
    for j in (0, 7):
        D_unb_p1_init[:, :, :, j] -= D_unb_p1_init[:, :, :, j].mean(axis=1, keepdims=True)
    s2_p1_base = mdl["sigma2_modes"] * var_scale
    D_bp_post_1a = np.zeros((N_K, 13, 4, 8), dtype=np.float32)
    for k in range(N_K):
        D_k = D_unb_p1_init[k].copy()
        for a in range(4):
            rms_ka = float(np.sqrt(np.mean(D_k[:, a, :] ** 2)))
            exp_rms = float(np.sqrt(1.0 + np.mean(s2_p1_base[a])))
            D_k[:, a, :] /= max(0.25, rms_ka / exp_rms)
        post_ll_1a = run_coupled_walsh_tree_bp(D_k, s2_p1_base, n_outer=4)
        D_bp_post_1a[k] = _post_ll_to_walsh_expected(post_ll_1a)

    D_unb_p1_cal, s2_p1_cal, _ = _calibrate_tensor_against_spine(
        D_unb_p1, D_bp_post_1a[:, sp], s2_p1_base
    )
    D_bp_post = np.zeros((N_K, 13, 4, 8), dtype=np.float32)
    for k in range(N_K):
        post_ll = run_coupled_walsh_tree_bp(D_unb_p1_cal[k], s2_p1_cal, n_outer=4)
        D_bp_post[k] = _post_ll_to_walsh_expected(post_ll)

    d0_g_proj_all = np.zeros((N_K, 13, 8), dtype=np.float32)
    D0_p2_raw = np.zeros((N_K, 13, 8), dtype=np.float32)
    D0_p2_unb = np.zeros((N_K, 13, 8), dtype=np.float64)
    for j in range(8):
        G_j = G_all_j[:, :, :, j]
        d0_g_proj = np.zeros((N_K, 13), dtype=np.float32)
        for m in range(13):
            ref_m = [mp for mp in sp if mp != m]
            num = np.einsum("kn,kn->k", G_j[:, m, ref_m], D_bp_post[:, ref_m, 0, j])
            den = np.sum(D_bp_post[:, ref_m, 0, j] ** 2, axis=1) + 1e-4
            d0_g_proj[:, m] = num / den
        d0_g_proj_all[:, :, j] = d0_g_proj
        A_ho = np.stack([D_pred[:, :, 0, j].reshape(-1), d0_g_proj.reshape(-1)], axis=1)
        d0_p2 = (A_ho @ mdl["w_b2_models"][j]).reshape(N_K, 13)
        D0_p2_raw[:, :, j] = d0_p2
        D0_p2_unb[:, :, j] = d0_p2 / mdl["slopes_p2"][j]

    g0_k = np.ones(N_K, dtype=np.float64)
    for k in range(N_K):
        x_sp = D_pred[k, sp, 0, :].astype(np.float64)
        y_sp = D_bp_post[k, sp, 0, :].astype(np.float64)
        x_c = x_sp - x_sp.mean(axis=0, keepdims=True)
        y_c = y_sp - y_sp.mean(axis=0, keepdims=True)
        g0_k[k] = max(0.12, float(np.sum(x_c * y_c) / (np.sum(y_c ** 2) + 1e-6)))
    Dp_deb, sl_dp, mse_dp = _calibrate_1d_feature_against_spine(
        D_pred[:, :, 0, :], D_bp_post[:, sp, 0, :], g0_k, np.arange(N_K)
    )
    dg_deb, sl_dg, mse_dg = _calibrate_1d_feature_against_spine(
        d0_g_proj_all, D_bp_post[:, sp, 0, :], g0_k ** 2, np.arange(N_K)
    )
    D0_clean = np.zeros((N_K, 13, 8), dtype=np.float64)
    for j in range(8):
        w_dp = 1.0 / mse_dp[j]
        w_dg = (1.0 / mse_dg[j]) if sl_dg[j] > 0.20 else 0.0
        D0_clean[:, :, j] = (
            w_dp * (Dp_deb[:, :, j] / max(sl_dp[j], 0.25))
            + w_dg * (dg_deb[:, :, j] / max(sl_dg[j], 0.25))
        ) / (w_dp + w_dg)

    D0_for_v_p2 = (D0_clean * mdl["slopes_p2"][None, None, :]).astype(np.float32)
    for m in sp:
        D0_for_v_p2[:, m, :] = D_bp_post[:, m, 0, :] * mdl["slopes_p2"][None, :]

    (
        D_raw_p2,
        feat_bits_p2,
        D_k4_p2,
        feat_bc_p2,
        D_unb_p2,
        G_sr0_all,
        d0_h0_all,
        d0_h1_all,
    ) = run_d123_components(D0_for_v_p2, D0_clean, var_scale, D_bp_post)
    D_unb_p2[:, :, 0, :] = D0_p2_unb

    d0_sr0_all = np.zeros((N_K, 13, 8), dtype=np.float32)
    for j in range(8):
        for m in range(13):
            ref_m = [mp for mp in sp if mp != m]
            den = np.sum(D_bp_post[:, ref_m, 0, j] ** 2, axis=1) + 1e-4
            d0_sr0_all[:, m, j] = (
                np.einsum("kn,kn->k", G_sr0_all[:, m, ref_m, j], D_bp_post[:, ref_m, 0, j]) / den
            )

    return {
        "D_pred": D_pred,
        "d0_g_proj_all": d0_g_proj_all,
        "d0_sr0_all": d0_sr0_all,
        "d0_h0_all": d0_h0_all,
        "d0_h1_all": d0_h1_all,
        "D_bp_post": D_bp_post,
        "D_raw_p2": D_raw_p2,
        "feat_bits_p2": feat_bits_p2,
        "D_k4_p2": D_k4_p2,
        "feat_bc_p2": feat_bc_p2,
        "D_unb_p2": D_unb_p2,
    }


def _compute_unsupervised_spine_metrics(
    p2_feats: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute unsupervised spine coherence `r_spine` and posterior saturation `c_sat` per key."""
    Dp = p2_feats["D_pred"]
    Du = p2_feats["D_unb_p2"]
    D_bp_post = p2_feats["D_bp_post"]
    sp = M_TREE_STRONG
    n_keys = Dp.shape[0]
    r_spine = np.zeros(n_keys, dtype=np.float64)
    for k in range(n_keys):
        dp_c = Dp[k, sp, 0, :] - Dp[k, sp, 0, :].mean(axis=0, keepdims=True)
        du_c = Du[k, sp, 1:, :] - Du[k, sp, 1:, :].mean(axis=0, keepdims=True)
        bp0_c = D_bp_post[k, sp, 0, :] - D_bp_post[k, sp, 0, :].mean(axis=0, keepdims=True)
        bp1_c = D_bp_post[k, sp, 1:, :] - D_bp_post[k, sp, 1:, :].mean(axis=0, keepdims=True)
        v_obs = np.concatenate([dp_c.reshape(-1), du_c.reshape(-1)])
        v_post = np.concatenate([bp0_c.reshape(-1), bp1_c.reshape(-1)])
        r_spine[k] = float(
            np.dot(v_obs, v_post) / (np.linalg.norm(v_obs) * np.linalg.norm(v_post) + 1e-8)
        )
    c_sat = np.mean(D_bp_post[:, sp, :, :] ** 2, axis=(1, 2, 3)) / 0.25
    return r_spine, c_sat


def solve_global_self_calibrated_bp_at_m(
    p2_feats: dict[str, np.ndarray],
    M_eval: int,
    mdl: dict[str, Any],
    domain_matched: bool = False,
    c0_scale: float = 0.70,
    beta_16: float = 4.0,
) -> np.ndarray:
    """Run global spine-consistent self-calibration and Coupled Walsh + S-Box Tree BP across all keys."""
    Dp = p2_feats["D_pred"].copy().astype(np.float64)
    dg = p2_feats["d0_g_proj_all"].copy().astype(np.float64)
    dsr0 = p2_feats["d0_sr0_all"].copy().astype(np.float64)
    dh0 = p2_feats["d0_h0_all"].copy().astype(np.float64)
    dh1 = p2_feats["d0_h1_all"].copy().astype(np.float64)
    D_bp_post = p2_feats["D_bp_post"].copy().astype(np.float64)
    Dr = p2_feats["D_raw_p2"].copy().astype(np.float64)
    fb = p2_feats["feat_bits_p2"].copy().astype(np.float64)
    Dk = p2_feats["D_k4_p2"].copy().astype(np.float64)
    fbc = p2_feats["feat_bc_p2"].copy().astype(np.float64)
    Du = p2_feats["D_unb_p2"].copy().astype(np.float64)

    N_K = Dp.shape[0]
    sp = M_TREE_STRONG
    vs = (256.0 / float(M_eval)) if M_eval <= 1024 else float(np.sqrt(256.0 / float(M_eval)))

    r_spine, c_sat = _compute_unsupervised_spine_metrics(p2_feats)
    good_mask = (r_spine >= 0.38) & (c_sat >= 0.80)
    conv_keys = np.where(good_mask)[0]
    n_conv = len(conv_keys)

    min_m_cal = 256 if domain_matched else 384
    use_self_cal = (n_conv >= max(1, int(0.25 * N_K))) and (M_eval >= min_m_cal)

    if use_self_cal:
        g0_arr = np.ones(N_K, dtype=np.float64)
        gd_arr = np.ones(N_K, dtype=np.float64)
        gh0_arr = np.ones(N_K, dtype=np.float64)
        g123_arr = np.ones(N_K, dtype=np.float64)
        for k in conv_keys:
            y0_c = D_bp_post[k, sp, 0, :] - D_bp_post[k, sp, 0, :].mean(axis=0, keepdims=True)
            den0 = float(np.sum(y0_c ** 2)) + 1e-6
            dp_c = Dp[k, sp, 0, :] - Dp[k, sp, 0, :].mean(axis=0, keepdims=True)
            dg_c = dg[k, sp, :] - dg[k, sp, :].mean(axis=0, keepdims=True)
            dh0_c = dh0[k, sp, :] - dh0[k, sp, :].mean(axis=0, keepdims=True)
            g0_arr[k] = max(0.12, float(np.sum(dp_c * y0_c) / den0))
            gd_arr[k] = max(0.08, float(np.sum(dg_c * y0_c) / den0))
            gh0_arr[k] = max(0.08, float(np.sum(dh0_c * y0_c) / den0))

            y123_c = D_bp_post[k, sp, 1:, :] - D_bp_post[k, sp, 1:, :].mean(axis=0, keepdims=True)
            du123_c = Du[k, sp, 1:, :] - Du[k, sp, 1:, :].mean(axis=0, keepdims=True)
            den123 = float(np.sum(y123_c ** 2)) + 1e-6
            g123_arr[k] = max(0.12, float(np.sum(du123_c * y123_c) / den123))

        med_g0 = float(np.median(g0_arr[conv_keys]))
        med_gd = float(np.median(gd_arr[conv_keys]))
        med_gh0 = float(np.median(gh0_arr[conv_keys]))
        med_g123 = float(np.median(g123_arr[conv_keys]))
        for k in range(N_K):
            if not good_mask[k]:
                g0_arr[k], gd_arr[k], gh0_arr[k], g123_arr[k] = med_g0, med_gd, med_gh0, med_g123

        D_cal = np.zeros((N_K, 13, 4, 8), dtype=np.float64)
        sig2_emp = np.zeros((4, 8), dtype=np.float64)

        if n_conv >= 12:
            for k in range(N_K):
                Dp[k] /= g0_arr[k]
                dg[k] /= gd_arr[k]
                dh0[k] /= gh0_arr[k]
                dh1[k] /= gh0_arr[k]
                dsr0[k] /= gh0_arr[k] ** 2
                Dr[k] /= g123_arr[k]
                fb[k] /= g123_arr[k]
                Dk[k] /= g123_arr[k]
                fbc[k] /= g123_arr[k]
                Du[k] /= g123_arr[k]

            use_cross_bits = n_conv >= 40
            if use_cross_bits:
                scale_reg = float(N_K) / float(n_conv)
                r0 = 0.01 * scale_reg
                ra = 0.05 * scale_reg
            else:
                scale_reg = max(1.0, 100.0 / float(n_conv))
                r0 = 0.02 * scale_reg
                ra = 0.08 * scale_reg

            for j in range(8):
                if use_cross_bits:
                    F_0 = np.concatenate(
                        [
                            Dp[:, :, 0, :],
                            dg[:, :, j : j + 1],
                            dsr0[:, :, j : j + 1],
                            dh0[:, :, :],
                            dh1[:, :, j : j + 1] if j > 0 else np.zeros((N_K, 13, 0)),
                            np.ones((N_K, 13, 1)),
                        ],
                        axis=2,
                    )
                    w0_prior = np.zeros(F_0.shape[2])
                else:
                    F_0 = np.concatenate(
                        [
                            Dp[:, :, 0, j : j + 1],
                            dg[:, :, j : j + 1],
                            dsr0[:, :, j : j + 1],
                            dh0[:, :, j : j + 1],
                            dh1[:, :, j : j + 1] if j > 0 else np.zeros((N_K, 13, 0)),
                            np.ones((N_K, 13, 1)),
                        ],
                        axis=2,
                    )
                    w0_prior = np.zeros(F_0.shape[2])
                    w0_prior[0] = 1.0

                X_tr = F_0[conv_keys][:, sp, :].reshape(-1, F_0.shape[2])
                y_tr = D_bp_post[conv_keys][:, sp, 0, j].reshape(-1)
                G0 = X_tr.T @ X_tr
                mult0 = np.ones(F_0.shape[2], dtype=np.float64)
                if not use_cross_bits:
                    mult0[1:-1] = 4.0
                P0 = r0 * np.diag((np.diag(G0) + 1e-4) * mult0)
                w0 = np.linalg.solve(G0 + P0, X_tr.T @ y_tr + P0 @ w0_prior)
                D_cal[:, :, 0, j] = (F_0.reshape(-1, F_0.shape[2]) @ w0).reshape(N_K, 13)
                dof0 = max(1.0, len(y_tr) - F_0.shape[2])
                sig2_emp[0, j] = float(np.sum((X_tr @ w0 - y_tr) ** 2) / dof0) * c0_scale

                for a_idx, a in enumerate((1, 2, 3)):
                    if use_cross_bits:
                        F_a = np.concatenate(
                            [
                                Du[:, :, a, :],
                                Dr[:, :, a, j : j + 1],
                                fb[:, :, a_idx, j : j + 1],
                                Dk[:, :, a_idx, j : j + 1],
                                fbc[:, :, a_idx, j : j + 1],
                                np.ones((N_K, 13, 1)),
                            ],
                            axis=2,
                        )
                        w_prior = np.zeros(F_a.shape[2])
                        w_prior[j] = 1.0
                    else:
                        F_a = np.concatenate(
                            [
                                Du[:, :, a, j : j + 1],
                                Dr[:, :, a, j : j + 1],
                                fb[:, :, a_idx, j : j + 1],
                                Dk[:, :, a_idx, j : j + 1],
                                fbc[:, :, a_idx, j : j + 1],
                                np.ones((N_K, 13, 1)),
                            ],
                            axis=2,
                        )
                        w_prior = np.zeros(F_a.shape[2])
                        w_prior[0] = 1.0

                    X_tr = F_a[conv_keys][:, sp, :].reshape(-1, F_a.shape[2])
                    y_tr = D_bp_post[conv_keys][:, sp, a, j].reshape(-1)
                    Ga = X_tr.T @ X_tr
                    Pa = ra * np.diag(np.diag(Ga) + 1e-4)
                    wa = np.linalg.solve(Ga + Pa, X_tr.T @ y_tr + Pa @ w_prior)
                    D_cal[:, :, a, j] = (F_a.reshape(-1, F_a.shape[2]) @ wa).reshape(N_K, 13)
                    dofa = max(1.0, len(y_tr) - F_a.shape[2])
                    sig2_emp[a, j] = float(np.sum((X_tr @ wa - y_tr) ** 2) / dofa)
                if not use_cross_bits:
                    sig2_emp[1:4, j] = 0.25 * sig2_emp[1:4, j] + 0.75 * float(np.mean(sig2_emp[1:4, j]))
            sig2_cal = sig2_emp
        else:
            Dp_d, sl_Dp, mse_Dp = _calibrate_1d_feature_against_spine(
                Dp[:, :, 0, :], D_bp_post[:, sp, 0, :], g0_arr, conv_keys, prior_slope=1.0, n_reg=6.0
            )
            dg_d, sl_dg, mse_dg = _calibrate_1d_feature_against_spine(
                dg, D_bp_post[:, sp, 0, :], gd_arr, conv_keys, prior_slope=0.0
            )
            dsr0_d, sl_dsr0, mse_dsr0 = _calibrate_1d_feature_against_spine(
                dsr0, D_bp_post[:, sp, 0, :], gh0_arr ** 2, conv_keys, prior_slope=0.0
            )
            dh0_d, sl_dh0, mse_dh0 = _calibrate_1d_feature_against_spine(
                dh0, D_bp_post[:, sp, 0, :], gh0_arr, conv_keys, prior_slope=0.0
            )
            dh1_d, sl_dh1, mse_dh1 = _calibrate_1d_feature_against_spine(
                dh1, D_bp_post[:, sp, 0, :], gh0_arr, conv_keys, prior_slope=0.0
            )

            for j in range(8):
                comps = [Dp_d[:, :, j] / max(sl_Dp[j], 0.25)]
                weights = [1.0 / mse_Dp[j]]
                for feat_d, sl_f, mse_f, min_sl in [
                    (dg_d[:, :, j], sl_dg[j], mse_dg[j], 0.25),
                    (dh0_d[:, :, j], sl_dh0[j], mse_dh0[j], 0.20),
                    (dh1_d[:, :, j], sl_dh1[j], mse_dh1[j], 0.20)
                    if j > 0
                    else (None, 0.0, 100.0, 1.0),
                    (dsr0_d[:, :, j], sl_dsr0[j], mse_dsr0[j], 0.20),
                ]:
                    if feat_d is not None and sl_f > min_sl and mse_f < 2.5 * mse_Dp[j]:
                        comps.append(feat_d / sl_f)
                        weights.append(0.35 / mse_f)
                w_arr = np.array(weights, dtype=np.float64)
                w_arr /= np.sum(w_arr)
                D_cal[:, :, 0, j] = sum(w * c for w, c in zip(w_arr, comps))
                res0 = D_cal[conv_keys][:, sp, 0, j] - D_bp_post[conv_keys][:, sp, 0, j]
                sig2_emp[0, j] = float(np.mean(res0 ** 2)) * (9.0 / 7.5) * c0_scale

            Dr_unb = Dr / mdl["slopes_init"][None, None, :, :]
            for a_idx, a in enumerate((1, 2, 3)):
                Du_d, sl_Du, mse_Du = _calibrate_1d_feature_against_spine(
                    Du[:, :, a, :],
                    D_bp_post[:, sp, a, :],
                    g123_arr,
                    conv_keys,
                    prior_slope=1.0,
                    n_reg=18.0,
                )
                Dr_d, sl_Dr, mse_Dr = _calibrate_1d_feature_against_spine(
                    Dr_unb[:, :, a, :],
                    D_bp_post[:, sp, a, :],
                    g123_arr,
                    conv_keys,
                    prior_slope=1.0,
                    n_reg=18.0,
                )
                fb_d, sl_fb, mse_fb = _calibrate_1d_feature_against_spine(
                    fb[:, :, a_idx, :],
                    D_bp_post[:, sp, a, :],
                    g123_arr,
                    conv_keys,
                    prior_slope=0.0,
                )
                Dk_d, sl_Dk, mse_Dk = _calibrate_1d_feature_against_spine(
                    Dk[:, :, a_idx, :],
                    D_bp_post[:, sp, a, :],
                    g123_arr,
                    conv_keys,
                    prior_slope=0.0,
                )
                fbc_d, sl_fbc, mse_fbc = _calibrate_1d_feature_against_spine(
                    fbc[:, :, a_idx, :],
                    D_bp_post[:, sp, a, :],
                    g123_arr,
                    conv_keys,
                    prior_slope=0.0,
                )
                for j in range(8):
                    comps_a = [Du_d[:, :, j] / sl_Du[j], Dr_d[:, :, j] / sl_Dr[j]]
                    weights_a = [1.0 / mse_Du[j], 0.50 / mse_Dr[j]]
                    for feat_d, sl_f, mse_f in [
                        (fb_d[:, :, j], sl_fb[j], mse_fb[j]),
                        (Dk_d[:, :, j], sl_Dk[j], mse_Dk[j]),
                        (fbc_d[:, :, j], sl_fbc[j], mse_fbc[j]),
                    ]:
                        if j < 6 and sl_f > 0.20 and mse_f < 2.0 * mse_Du[j]:
                            comps_a.append(feat_d / sl_f)
                            weights_a.append(0.25 / mse_f)
                    w_arr = np.array(weights_a, dtype=np.float64)
                    w_arr /= np.sum(w_arr)
                    D_cal[:, :, a, j] = sum(w * c for w, c in zip(w_arr, comps_a))
                    resa = D_cal[conv_keys][:, sp, a, j] - D_bp_post[conv_keys][:, sp, a, j]
                    sig2_emp[a, j] = float(np.mean(resa ** 2)) * (9.0 / 7.5)

            prior_s2 = mdl["sigma2_p2"] * vs
            n_obs = float(9 * n_conv)
            n_prior = 6.0
            sig2_cal = np.zeros((4, 8), dtype=np.float64)
            for a in range(4):
                alpha_a = float(np.median(sig2_emp[a] / np.maximum(prior_s2[a], 1e-4)))
                prior_a = alpha_a * prior_s2[a]
                sig2_cal[a] = np.maximum(
                    1e-4, (n_obs * sig2_emp[a] + n_prior * prior_a) / (n_obs + n_prior)
                )
            for j in range(8):
                sig2_cal[1:4, j] = 0.35 * sig2_cal[1:4, j] + 0.65 * float(np.mean(sig2_cal[1:4, j]))
                if j >= 6:
                    sig2_cal[1:4, j] = np.maximum(sig2_cal[1:4, j], 1.25 * prior_s2[1:4, j])
        conv_set = set(range(N_K))
    else:
        D_cal = Du.copy()
        sig2_cal = mdl["sigma2_p2"] * vs
        conv_set = set()

    post_ll_batch = np.zeros((N_K, 13, 4, 256), dtype=np.float64)
    for k in range(N_K):
        if k in conv_set:
            s2_tree = sig2_cal.copy()
            s2_free = sig2_cal.copy()
            if n_conv < 12:
                s2_free[0] *= 0.55
                s2_free[1:4, 6:] = np.maximum(s2_free[1:4, 6:] * 1.50, 0.55)
            else:
                s2_free[0] /= c0_scale
            post_ll_raw = run_coupled_walsh_tree_bp(D_cal[k], s2_tree, n_outer=4)
            post_ll = post_ll_raw.copy()
            for m_free in (5, 6):
                log_psi_free = compute_16state_log_psi(D_cal[k, m_free], s2_free)
                post_ll[m_free] = marg_16_sharpened(
                    log_psi_free, beta=beta_16, even_m=(m_free % 2 == 0)
                )
            joint_4, joint_12 = _compute_leaf_4_12_joints(D_cal[k], s2_tree, post_ll_raw)
            post_ll[4] = marg_16_sharpened(joint_4, beta=beta_16, even_m=True)
            post_ll[12] = marg_16_sharpened(joint_12, beta=beta_16, even_m=True)
            post_ll_batch[k] = post_ll
        else:
            sig2_k = mdl["sigma2_p2"] * vs
            Du_k = Du[k].copy()
            for j in (0, 7):
                Du_k[:, :, j] -= np.mean(Du_k[:, :, j], axis=0, keepdims=True)
            for a in range(4):
                rms_ka = float(np.sqrt(np.mean(Du_k[:, a, :] ** 2)))
                exp_rms = float(np.sqrt(1.0 + np.mean(sig2_k[a])))
                Du_k[:, a, :] /= max(0.25, rms_ka / exp_rms)
            post_ll_raw = run_coupled_walsh_tree_bp(Du_k, sig2_k, n_outer=4)
            W_4_j16, W_12_j16 = update_leaf_16state_posteriors(Du_k, sig2_k, post_ll_raw)
            post_ll = post_ll_raw.copy()
            for m_free in (5, 6):
                log_psi_free = compute_16state_log_psi(Du_k[m_free], sig2_k)
                post_ll[m_free] = marg_16_sharpened(
                    log_psi_free, beta=beta_16, even_m=(m_free % 2 == 0)
                )
            post_ll[4] = log_normalize_256(0.5 * post_ll_raw[4] + 0.5 * W_4_j16)
            post_ll[12] = log_normalize_256(0.5 * post_ll_raw[12] + 0.5 * W_12_j16)
            post_ll_batch[k] = post_ll
    return post_ll_batch


# ---------------------------------------------------------------------------
# 6. Candidate Enumeration, Evaluation Metrics, Plotting & CLI Workflow
# ---------------------------------------------------------------------------


def _enumerate_top_n_full_keys(
    post_ll_k: np.ndarray,
    top1_key_hex: str,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Enumerate the `top_n` most probable 256-bit master keys for key `k` via 16-state
    Mode-0-swap-aware factorized Dijkstra search over `W_{7+4m}` (`m=0..7`).
    """
    top_n = max(1, int(top_n))
    ll_w = log_normalize_256(post_ll_k[:8])
    top1_w = np.argmax(ll_w, axis=-1).astype(np.uint8)
    base_ll = float(np.sum(np.max(ll_w, axis=-1)))

    factors: list[list[tuple[float, tuple[tuple[int, int, int], ...]]]] = []
    for m in (5, 6):
        m_alts: list[tuple[float, tuple[tuple[int, int, int], ...]]] = []
        for j in range(8):
            bit_mask = 1 << j
            costs = np.array(
                [
                    float(ll_w[m, q, top1_w[m, q]] - ll_w[m, q, int(top1_w[m, q]) ^ bit_mask])
                    for q in range(4)
                ]
            )
            signs = np.array([1 if ((int(top1_w[m, q]) >> j) & 1) else -1 for q in range(4)])
            order_q = np.argsort(costs)
            q0, q1, q2, q3 = int(order_q[0]), int(order_q[1]), int(order_q[2]), int(order_q[3])

            swap_pairs: list[tuple[float, int, int]] = []
            for qa, qb in ((q0, q1), (q0, q2), (q1, q2), (q0, q3), (q1, q3), (q2, q3)):
                if signs[qa] != signs[qb]:
                    raw_sc = (
                        0.50 * float(costs[qa] + costs[qb])
                        if abs(costs[qb] - costs[qa]) < 0.25
                        else float(max(costs[qa], costs[qb]))
                    )
                    swap_pairs.append((raw_sc, int(qa), int(qb)))
            swap_pairs.sort(key=lambda x: x[0])

            is_pure_swap = (signs[q0] != signs[q1]) and (abs(costs[q1] - costs[q0]) < 0.20)
            max_raw = 4.2 if j == 7 else (0.65 if j == 6 else 0.50)
            sw_scale = 0.35 if j == 7 else (0.55 if j == 6 else 0.85)
            if is_pure_swap:
                if swap_pairs[0][0] < max_raw:
                    sc0 = sw_scale * swap_pairs[0][0]
                    for idx_s, (raw_sc, qa, qb) in enumerate(swap_pairs[:2]):
                        eff_sc = sc0 + 0.18 * (raw_sc - swap_pairs[0][0]) if idx_s > 0 else sc0
                        m_alts.append((eff_sc, ((m, qa, bit_mask), (m, qb, bit_mask))))
            elif len(set(signs.tolist())) == 1:
                for q_cand in (q0, q1, q2):
                    if costs[q_cand] < max_raw:
                        eff_c = float(costs[q0] + 0.20 * (costs[q_cand] - costs[q0]))
                        m_alts.append((eff_c, ((m, int(q_cand), bit_mask),)))
            else:
                if costs[q0] < (1.8 if j == 7 else (0.65 if j == 6 else 0.50)):
                    m_alts.append((float(costs[q0]), ((m, q0, bit_mask),)))
                if costs[q1] < (1.4 if j == 7 else 0.55) and signs[q1] == signs[q0]:
                    eff_c1 = float(costs[q0] + 0.25 * (costs[q1] - costs[q0]))
                    m_alts.append((eff_c1, ((m, q1, bit_mask),)))
                if swap_pairs and swap_pairs[0][0] < max_raw:
                    sc0 = sw_scale * swap_pairs[0][0]
                    for idx_s, (raw_sc, qa, qb) in enumerate(swap_pairs[:2]):
                        if raw_sc < (7.5 if j == 7 else 1.2):
                            eff_sc = sc0 + 0.18 * (raw_sc - swap_pairs[0][0])
                            m_alts.append((eff_sc, ((m, qa, bit_mask), (m, qb, bit_mask))))

        if m_alts:
            m_alts.sort(key=lambda x: x[0])
            factors.append(m_alts)

    for m in (0, 1, 2, 3, 4, 7):
        m_alts_tree: list[tuple[float, tuple[tuple[int, int, int], ...]]] = []
        for q in range(4):
            order_b = np.argsort(-ll_w[m, q])
            b0, b1 = int(order_b[0]), int(order_b[1])
            cost_b = float(ll_w[m, q, b0] - ll_w[m, q, b1])
            if cost_b < (2.5 if m == 4 else 1.0):
                eff_cb = (0.35 * cost_b) if m == 4 else cost_b
                m_alts_tree.append((eff_cb, ((m, q, b0 ^ b1),)))
        if m_alts_tree:
            m_alts_tree.sort(key=lambda x: x[0])
            factors.append(m_alts_tree)

    factors.sort(key=lambda alts: alts[0][0])

    pq_f: list[tuple[float, int, tuple[tuple[int, int], ...]]] = [(0.0, 0, ())]
    seen_hex: set[str] = {top1_key_hex}
    candidates: list[dict[str, Any]] = [
        {
            "rank": 1,
            "key_hex": top1_key_hex,
            "log_prob": round(base_ll, 4),
            "relative_prob": 1.0,
        }
    ]

    while pq_f and len(candidates) < top_n:
        pen, f_idx, chosen = heapq.heappop(pq_f)
        if chosen:
            w_cand = top1_w.copy()
            for fi, ai in chosen:
                for m, q, xor_mask in factors[fi][ai][1]:
                    w_cand[m, q] ^= xor_mask
            k_hex = _bijection_w0_to_7_to_master_key(w_cand).reshape(32).tobytes().hex()
            if k_hex not in seen_hex:
                seen_hex.add(k_hex)
                candidates.append(
                    {
                        "rank": len(candidates) + 1,
                        "key_hex": k_hex,
                        "log_prob": round(base_ll - pen, 4),
                        "relative_prob": round(float(np.exp(-min(pen, 60.0))), 6),
                    }
                )
        step_pen = 0.15 * len(chosen)
        for nxt_f in range(f_idx, len(factors)):
            for ai, (alt_cost, _) in enumerate(factors[nxt_f]):
                heapq.heappush(
                    pq_f,
                    (pen + alt_cost + step_pen, nxt_f + 1, chosen + ((nxt_f, ai),)),
                )

    if len(candidates) < top_n:
        ll_flat = ll_w.reshape(32, 256)
        order = np.argsort(-ll_flat, axis=-1)
        sorted_ll = np.take_along_axis(ll_flat, order, axis=-1)
        init_ranks = (0,) * 32
        pq: list[tuple[float, tuple[int, ...]]] = [(0.0, init_ranks)]
        visited: set[tuple[int, ...]] = {init_ranks}
        while pq and len(candidates) < top_n:
            penalty, ranks = heapq.heappop(pq)
            w_bytes = np.array([order[i, ranks[i]] for i in range(32)], dtype=np.uint8).reshape(
                8, 4
            )
            k_hex = _bijection_w0_to_7_to_master_key(w_bytes).reshape(32).tobytes().hex()
            if k_hex not in seen_hex:
                seen_hex.add(k_hex)
                candidates.append(
                    {
                        "rank": len(candidates) + 1,
                        "key_hex": k_hex,
                        "log_prob": round(base_ll - penalty, 4),
                        "relative_prob": round(float(np.exp(-min(penalty, 60.0))), 6),
                    }
                )
            for i in range(32):
                r_cur = ranks[i]
                if r_cur + 1 < 8:
                    nxt = list(ranks)
                    nxt[i] = r_cur + 1
                    nxt_t = tuple(nxt)
                    if nxt_t not in visited:
                        visited.add(nxt_t)
                        delta = float(sorted_ll[i, 0] - sorted_ll[i, r_cur + 1])
                        heapq.heappush(
                            pq,
                            (penalty - float(sorted_ll[i, 0] - sorted_ll[i, r_cur]) + delta, nxt_t),
                        )
    return candidates


def decode_predicted_keys_from_posteriors(
    post_ll_all: np.ndarray,
    r_spine: np.ndarray,
    c_sat: np.ndarray,
    top_k: int = 5,
    top_n_keys: int = 5,
) -> list[dict[str, Any]]:
    """Map S-box word log-posteriors `(N_K, 13, 4, 256)` to 32-byte master key predictions,
    confidence scores, top-K byte candidates, and Top-N full-key candidates.
    """
    n_keys = post_ll_all.shape[0]
    top_k = max(1, min(16, int(top_k)))
    top_n_keys = max(1, min(50, int(top_n_keys)))
    predictions: list[dict[str, Any]] = []

    for k in range(n_keys):
        post_ll_k = post_ll_all[k]
        key_ll_k = map_sbox_words_to_key_words(post_ll_k)

        p_key_k = np.exp(log_normalize_256(key_ll_k))
        p_w_k = np.exp(log_normalize_256(post_ll_k))

        pred_key_words = np.argmax(p_key_k, axis=-1).astype(np.uint8)
        pred_key_bytes = pred_key_words.reshape(32)
        pred_key_hex = pred_key_bytes.tobytes().hex()

        pred_w_words = np.argmax(p_w_k, axis=-1).astype(np.uint8)
        pred_tree44_bytes = pred_w_words[TREE_WORDS].reshape(44)
        pred_free8_bytes = pred_w_words[FREE_WORDS].reshape(8)

        key32_max_probs = np.max(p_key_k, axis=-1).reshape(32)
        tree44_max_probs = np.max(p_w_k[TREE_WORDS], axis=-1).reshape(44)
        free8_max_probs = np.max(p_w_k[FREE_WORDS], axis=-1).reshape(8)

        p_flat = p_key_k.reshape(32, 256)
        top_indices = np.argsort(-p_flat, axis=-1)[:, :top_k]
        byte_top_k = [
            [
                {
                    "byte_hex": f"{int(b_val):02x}",
                    "byte_int": int(b_val),
                    "prob": round(float(p_flat[b_idx, b_val]), 6),
                }
                for b_val in top_indices[b_idx]
            ]
            for b_idx in range(32)
        ]

        p_free_flat = p_w_k[FREE_WORDS].reshape(8, 256)
        free_top_indices = np.argsort(-p_free_flat, axis=-1)[:, :top_k]
        free8_top_k = [
            [
                {
                    "byte_hex": f"{int(b_val):02x}",
                    "byte_int": int(b_val),
                    "prob": round(float(p_free_flat[b_idx, b_val]), 6),
                }
                for b_val in free_top_indices[b_idx]
            ]
            for b_idx in range(8)
        ]

        top_n_candidates = _enumerate_top_n_full_keys(
            post_ll_k, top1_key_hex=pred_key_hex, top_n=top_n_keys
        )

        predictions.append(
            {
                "key_index": int(k),
                "predicted_key_hex": pred_key_hex,
                "predicted_key_words_hex": [pred_key_words[w].tobytes().hex() for w in range(8)],
                "predicted_tree44_hex": pred_tree44_bytes.tobytes().hex(),
                "predicted_free8_hex": pred_free8_bytes.tobytes().hex(),
                "confidence": {
                    "key32_mean_byte_prob": round(float(np.mean(key32_max_probs)), 6),
                    "key32_min_byte_prob": round(float(np.min(key32_max_probs)), 6),
                    "tree44_mean_byte_prob": round(float(np.mean(tree44_max_probs)), 6),
                    "tree44_min_byte_prob": round(float(np.min(tree44_max_probs)), 6),
                    "free8_mean_byte_prob": round(float(np.mean(free8_max_probs)), 6),
                    "spine_coherence_r": round(float(r_spine[k]), 4),
                    "posterior_saturation": round(float(c_sat[k]), 4),
                },
                "top_n_key_candidates": top_n_candidates,
                "byte_top_k_candidates": byte_top_k,
                "free8_top_k_candidates": free8_top_k,
            }
        )

    return predictions


def build_trace_budget_schedule(traces_per_key: int) -> list[int]:
    """Generate a trace budget schedule of multiples of 64 up to `traces_per_key`."""
    candidates = [
        64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16000
    ]
    budgets = [m for m in candidates if m <= traces_per_key]
    if not budgets or budgets[-1] != traces_per_key:
        if traces_per_key >= 64:
            budgets.append((traces_per_key // 64) * 64)
    return sorted(set(budgets))


def compute_metrics_at_budget(
    post_ll_all: np.ndarray,
    W_true: np.ndarray,
    K_true: np.ndarray,
    M_eval: int,
) -> dict[str, Any]:
    """Compute recovery metrics for the 32-byte master key (`W_0..W_7`),
    the 44-byte S-box tree (`m in TREE_WORDS`), and the 8 free S-box bytes (`m=5, 6`).
    """
    n_keys = len(post_ll_all)

    def eval_group(
        ll_arr: np.ndarray, true_arr: np.ndarray
    ) -> tuple[float, float, float, float, int]:
        ranks = []
        bit_ok = 0
        bit_tot = 0
        full_group_ok = 0
        for k in range(n_keys):
            pred_k = np.argmax(ll_arr[k], axis=-1)
            if np.all(pred_k == true_arr[k]):
                full_group_ok += 1
            for m_i in range(ll_arr.shape[1]):
                for q in range(4):
                    tb = int(true_arr[k, m_i, q])
                    ll_v = ll_arr[k, m_i, q]
                    rk = int(np.sum(ll_v > ll_v[tb]))
                    ranks.append(rk)
                    p = np.exp(log_normalize_256(ll_v))
                    for j in range(8):
                        b_exp = float(np.dot(p, BYTE_BITS_PM1[:, j]))
                        b_true = 1 if ((tb >> j) & 1) else -1
                        if (b_exp >= 0 and b_true == 1) or (b_exp < 0 and b_true == -1):
                            bit_ok += 1
                        bit_tot += 1
        r_arr = np.array(ranks)
        return (
            float(bit_ok / bit_tot),
            float(np.mean(r_arr == 0)),
            float(np.mean(r_arr < 5)),
            float(np.mean(r_arr)),
            int(full_group_ok),
        )

    tree_bit, tree_t1, tree_t5, tree_mr, tree_full_ok = eval_group(
        post_ll_all[:, TREE_WORDS], W_true[:, TREE_WORDS]
    )
    free_bit, free_t1, free_t5, free_mr, free_full_ok = eval_group(
        post_ll_all[:, FREE_WORDS], W_true[:, FREE_WORDS]
    )

    key_ll_all = np.zeros((n_keys, 8, 4, 256), dtype=np.float64)
    for k in range(n_keys):
        key_ll_all[k] = map_sbox_words_to_key_words(post_ll_all[k])
    key_bit, key_t1, key_t5, key_mr, full_keys_ok = eval_group(key_ll_all, K_true)

    exp_word_hw = np.zeros((n_keys, 8), dtype=np.float64)
    true_word_hw = np.zeros((n_keys, 8), dtype=np.float64)
    for k in range(n_keys):
        p_key_k = np.exp(log_normalize_256(key_ll_all[k]))
        exp_byte_hw = p_key_k @ HW8
        exp_word_hw[k] = exp_byte_hw.sum(axis=1)
        true_word_hw[k] = HW8[K_true[k]].sum(axis=1)

    exp_flat = exp_word_hw.ravel()
    true_flat = true_word_hw.ravel()
    if np.std(exp_flat) > 1e-12 and np.std(true_flat) > 1e-12:
        r_words = float(np.corrcoef(exp_flat, true_flat)[0, 1])
    else:
        r_words = 0.0
    z_words = float(r_words * math.sqrt(n_keys * 8))
    mask_neq16 = true_flat != 16
    word_hw_sr = (
        float(
            np.mean(
                (exp_flat[mask_neq16] > 16.0)
                == (true_flat[mask_neq16] > 16.0)
            )
        )
        if np.any(mask_neq16)
        else 1.0
    )

    return {
        "M": int(M_eval),
        "full_keys_recovered": int(full_keys_ok),
        "total_keys": int(n_keys),
        "full_key_success_rate": float(full_keys_ok / n_keys),
        "byte_top1_success_rate": float(key_t1),
        "byte_top5_success_rate": float(key_t5),
        "mean_byte_rank": float(key_mr),
        "guessing_entropy_bits": float(math.log2(key_mr + 1.0)),
        "bit_success_rate": float(key_bit),
        "word_hw_success_rate": float(word_hw_sr),
        "word_hw_corr_800": float(r_words),
        "word_hw_z_score": float(z_words),
        "tree44_full_keys_recovered": int(tree_full_ok),
        "tree44_full_success_rate": float(tree_full_ok / n_keys),
        "tree44_byte_top1_success_rate": float(tree_t1),
        "tree44_byte_top5_success_rate": float(tree_t5),
        "tree44_mean_byte_rank": float(tree_mr),
        "tree44_bit_success_rate": float(tree_bit),
        "free8_full_keys_recovered": int(free_full_ok),
        "free8_full_success_rate": float(free_full_ok / n_keys),
        "free8_byte_top1_success_rate": float(free_t1),
        "free8_byte_top5_success_rate": float(free_t5),
        "free8_mean_byte_rank": float(free_mr),
        "free8_bit_success_rate": float(free_bit),
    }


def _select_non_overlapping_log2_ticks(
    m_vals: list[int],
    min_log2_gap: float = 0.28,
) -> list[int]:
    if len(m_vals) <= 2:
        return list(m_vals)
    ticks = [m_vals[0]]
    for m in m_vals[1:-1]:
        if math.log2(m / ticks[-1]) >= min_log2_gap:
            ticks.append(m)
    if math.log2(m_vals[-1] / ticks[-1]) < min_log2_gap and len(ticks) > 1:
        ticks[-1] = m_vals[-1]
    elif m_vals[-1] != ticks[-1]:
        ticks.append(m_vals[-1])
    return ticks


def plot_success_rate_vs_traces(
    metrics_by_m: dict[str, Any],
    output_plot_path: Path,
) -> None:
    """Plot the Success Rate vs. Number of Traces (`M`) curve and save to `output_plot_path`."""
    m_vals = [v["M"] for v in metrics_by_m.values()]
    n_keys = int(next(iter(metrics_by_m.values()))["total_keys"]) if metrics_by_m else 0
    full_key_sr_pct = [v["full_key_success_rate"] * 100.0 for v in metrics_by_m.values()]
    top_n_key_sr_pct = [
        v.get("top_n_full_key_success_rate", v["full_key_success_rate"]) * 100.0
        for v in metrics_by_m.values()
    ]
    tree_full_sr_pct = [v["tree44_full_success_rate"] * 100.0 for v in metrics_by_m.values()]
    byte_top1_sr_pct = [v["byte_top1_success_rate"] * 100.0 for v in metrics_by_m.values()]
    byte_top5_sr_pct = [v["byte_top5_success_rate"] * 100.0 for v in metrics_by_m.values()]
    bit_sr_pct = [v["bit_success_rate"] * 100.0 for v in metrics_by_m.values()]
    word_hw_sr_pct = [v["word_hw_success_rate"] * 100.0 for v in metrics_by_m.values()]
    z_scores = [v["word_hw_z_score"] for v in metrics_by_m.values()]

    tick_vals = _select_non_overlapping_log2_ticks(m_vals)
    x_min = max(0.8, min(m_vals) * 0.85)
    x_max = max(m_vals) * 1.15

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0), dpi=150)

    ax1 = axes[0]
    ax1.plot(
        m_vals,
        full_key_sr_pct,
        marker="o",
        linewidth=2.4,
        color="#d62728",
        label="Top-1 Full 256-bit Key SR (%)",
    )
    ax1.plot(
        m_vals,
        top_n_key_sr_pct,
        marker="D",
        linewidth=2.1,
        linestyle="-.",
        color="#ff7f0e",
        label="Top-10 Full 256-bit Key SR (%)",
    )
    ax1.plot(
        m_vals,
        tree_full_sr_pct,
        marker="P",
        linewidth=2.0,
        color="#9467bd",
        label="Full 44-Byte S-Box Tree Recovery SR (%)",
    )
    ax1.plot(
        m_vals,
        byte_top1_sr_pct,
        marker="s",
        linewidth=2.0,
        color="#1f77b4",
        label="Key Byte Top-1 SR (%, random = 0.39%)",
    )
    ax1.plot(
        m_vals,
        byte_top5_sr_pct,
        marker="^",
        linewidth=1.8,
        linestyle="--",
        color="#2ca02c",
        label="Key Byte Top-5 SR (%, random = 1.95%)",
    )
    ax1.set_xlabel("Number of Traces per Key ($M$)", fontsize=11)
    ax1.set_ylabel("Success Rate (%)", fontsize=11)
    ax1.set_title(
        f"Blind `trace1 -> trace2` Key Recovery SR vs. Trace Budget ($M$, max={max(m_vals)})\n"
        f"(Holdout Split, {n_keys} Keys, Coupled 4-Mode Walsh + S-Box Tree BP)",
        fontsize=11.0,
        fontweight="bold",
    )
    ax1.set_xscale("log", base=2)
    ax1.set_xlim(x_min, x_max)
    ax1.set_xticks(tick_vals)
    ax1.set_xticklabels([str(m) for m in tick_vals], rotation=30)
    ax1.set_ylim(-2.0, 105.0)
    ax1.grid(True, which="both", linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", fontsize=9.0)

    ax2 = axes[1]
    l1 = ax2.plot(
        m_vals,
        word_hw_sr_pct,
        marker="D",
        linewidth=2.3,
        color="#9467bd",
        label="32-bit Key Word HW Distinguishing SR (%)",
    )
    l2 = ax2.plot(
        m_vals,
        bit_sr_pct,
        marker="o",
        linewidth=2.0,
        color="#ff7f0e",
        label="256-bit Key Bit Recovery SR (%)",
    )
    ax2.axhline(
        50.0, color="gray", linestyle=":", alpha=0.7, linewidth=1.2, label="Random Guess (50.0%)"
    )
    ax2.set_xlabel("Number of Traces per Key ($M$)", fontsize=11)
    ax2.set_ylabel("Recovery / Distinguishing SR (%)", fontsize=11)
    ax2.set_title(
        "Key Bit Recovery & Word-HW Distinguishing Significance ($z$)\n"
        f"vs. Number of Traces ($M$) Across {n_keys} Holdout Keys",
        fontsize=11.0,
        fontweight="bold",
    )
    ax2.set_xscale("log", base=2)
    ax2.set_xlim(x_min, x_max)
    ax2.set_xticks(tick_vals)
    ax2.set_xticklabels([str(m) for m in tick_vals], rotation=30)
    ax2.set_ylim(45.0, 102.0)
    ax2.grid(True, which="both", linestyle="--", alpha=0.35)

    ax2_r = ax2.twinx()
    l3 = ax2_r.plot(
        m_vals,
        z_scores,
        marker="*",
        markersize=8,
        linewidth=1.8,
        linestyle="-.",
        color="#2ca02c",
        label="Word-HW Significance ($z$-score $\\sigma$)",
    )
    ax2_r.set_ylabel("Statistical Significance ($z$-score $\\sigma$)", fontsize=11, color="#2ca02c")
    ax2_r.tick_params(axis="y", labelcolor="#2ca02c")
    ax2_r.set_ylim(min(-1.0, min(z_scores) - 0.5), max(4.5, max(z_scores) * 1.15))

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper left", fontsize=9.0)

    fig.tight_layout()
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot_path, dpi=150)
    plt.close(fig)


def _format_colored_key_text(
    key_hex: str,
    gt_bytes: np.ndarray,
    byte_top_k_candidates: list[list[dict[str, Any]]] | None = None,
) -> "Text":
    """Build a Rich `Text` object for a 32-byte key (64 hex chars) with per-byte match styling."""
    txt = Text()
    key_bytes = np.frombuffer(bytes.fromhex(key_hex), dtype=np.uint8)
    for b_idx in range(32):
        b_val = int(key_bytes[b_idx])
        b_hex = f"{b_val:02x}"
        gt_val = int(gt_bytes[b_idx])
        if b_val == gt_val:
            style = "bold bright_green"
        elif byte_top_k_candidates is not None and any(
            c["byte_int"] == gt_val for c in byte_top_k_candidates[b_idx]
        ):
            style = "bold yellow"
        else:
            style = "dim red"
        txt.append(b_hex, style=style)
    return txt


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the standalone key-recovery pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end key recovery on the holdout split of a sedpack dataset using "
            "Blind `trace1 -> trace2` Reconstruction & Coupled 4-Mode Walsh + S-Box Tree BP."
        )
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Optional positional path to the sedpack dataset directory.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="rp2350_aesctr_small_fixed",
        help="Path to the sedpack dataset directory (default: rp2350_aesctr_small_fixed).",
    )
    parser.add_argument(
        "--templates-path",
        type=str,
        default="trace2_walsh_tree_bp_templates.pkl",
        help="Path to pre-fitted Walsh+Tree-BP templates (default: trace2_walsh_tree_bp_templates.pkl).",
    )
    parser.add_argument(
        "--max-keys",
        type=int,
        default=None,
        help="Optional limit on number of holdout keys to evaluate.",
    )
    parser.add_argument(
        "--max-traces-per-key",
        type=int,
        default=None,
        help="Optional limit on traces per key (rounded to a multiple of 64).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top byte candidates per byte to evaluate/highlight (default: 5).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of Top-N full 256-bit key candidates per key to compute & highlight (default: 10).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: min(24, cpu_count)).",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="success_rate_vs_traces.png",
        help="Output path for the Success Rate vs. Number of Traces plot (default: success_rate_vs_traces.png).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="end_to_end_attack_results.json",
        help="Output path for JSON evaluation metrics (default: end_to_end_attack_results.json).",
    )
    parser.add_argument(
        "--text",
        "--force-text",
        "--no-rich",
        dest="force_text",
        action="store_true",
        help="Force plain-text output (disable `rich` progress bars and color tables).",
    )
    return parser.parse_args()


def validate_cli_args(args: argparse.Namespace) -> None:
    """Validate numeric CLI bounds before starting pipeline execution."""
    if args.max_keys is not None and args.max_keys < 1:
        raise ValueError(f"--max-keys must be >= 1, got {args.max_keys}.")
    if args.max_traces_per_key is not None and args.max_traces_per_key < 64:
        raise ValueError(
            f"--max-traces-per-key must be >= 64 (at least one 64-trace shard per key), "
            f"got {args.max_traces_per_key}."
        )
    if args.top_k < 1 or args.top_k > 256:
        raise ValueError(f"--top-k must be between 1 and 256, got {args.top_k}.")
    if args.top_n < 1:
        raise ValueError(f"--top-n must be >= 1, got {args.top_n}.")
    if args.workers is not None and args.workers < 1:
        raise ValueError(f"--workers must be >= 1, got {args.workers}.")


def _exit_with_cli_error(exc: Exception, use_rich: bool) -> None:
    """Render an actionable user-facing error message to stderr and exit with status 1."""
    if use_rich and RICH_AVAILABLE:
        err_console = Console(stderr=True, force_terminal=True)
        err_console.print(
            Panel(
                f"[bold red]{type(exc).__name__}:[/bold red] {exc}",
                title="[bold red]Pipeline Error[/bold red]",
                border_style="red",
                box=box.ROUNDED,
            )
        )
    else:
        print(f"[!] Error ({type(exc).__name__}): {exc}", file=sys.stderr)
    sys.exit(1)


def run_poi6_holdout_diagnostics(
    full_paths: list[str],
    t2_ref: np.ndarray,
    n_workers: int,
    console: Any,
) -> dict[str, Any]:
    """Run Welch `|t|` diagnostics on `trace1`-reconstructed `POI-6a/6b` probe shards."""
    poi6_probe_shards = full_paths[: min(8, len(full_paths))]
    with ProcessPoolExecutor(
        max_workers=min(n_workers, len(poi6_probe_shards)),
        initializer=_init_holdout_trace1_worker,
        initargs=(t2_ref,),
    ) as ex:
        poi6_probe_res = list(ex.map(_read_shard_holdout_poi6_probe_strict, poi6_probe_shards))
    poi6a_probe = np.concatenate([r[0] for r in poi6_probe_res], axis=0)
    poi6b_probe = np.concatenate([r[1] for r in poi6_probe_res], axis=0)
    bperm_probe = np.concatenate([r[2] for r in poi6_probe_res], axis=0)
    ks_probe = np.concatenate([r[3] for r in poi6_probe_res], axis=0)
    t_poi6a = _compute_poi6_welch_t(poi6a_probe, ks_probe[:, :16])
    t_poi6b = _compute_poi6_welch_t(poi6b_probe, ks_probe[:, 16:32])
    msg = (
        f"[*] Blind `trace1`-Reconstructed Holdout POI-6a/6b Diagnostics (N={len(ks_probe)} traces): "
        f"POI-6a Welch |t| = {t_poi6a:.2f}, POI-6b Welch |t| = {t_poi6b:.2f}, "
        f"CT_BPERM swap rate = {np.mean(bperm_probe) * 100:.1f}%"
    )
    if console is not None:
        console.print(f"    [dim]{msg}[/dim]")
    else:
        print(f"    {msg}")
    return {
        "num_probe_traces": int(len(ks_probe)),
        "bperm_swap_rate": round(float(np.mean(bperm_probe)), 4),
        "poi6a_max_abs_t": round(float(t_poi6a), 2),
        "poi6b_max_abs_t": round(float(t_poi6b), 2),
    }


def stream_holdout_and_extract_p2_features(
    full_paths: list[str],
    n_keys: int,
    traces_per_key: int,
    full_spk: int,
    trace_budgets: list[int],
    mdl: dict[str, Any],
    ops_fast: list[dict[str, Any]],
    perm_u: np.ndarray,
    psi_mats: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    t2_ref: np.ndarray,
    n_workers: int,
    console: Any,
) -> tuple[dict[int, dict[str, np.ndarray]], bool]:
    """Stream `holdout` `trace1` shards, calibrate target ZCA whitening if needed, and extract
    Pass-1/Pass-2 sufficient statistics across all `trace_budgets`.
    """
    eff_spk = traces_per_key // 64
    chunk_M = min(traces_per_key, 16384)
    chunk_spk = chunk_M // 64
    n_chunks = (eff_spk + chunk_spk - 1) // chunk_spk
    key_batch_size = max(1, min(n_keys, 32768 // max(64, chunk_M)))
    n_key_batches = (n_keys + key_batch_size - 1) // key_batch_size
    total_eval_traces = n_keys * traces_per_key

    p2_feats_by_m: dict[int, dict[str, np.ndarray]] = {
        m: {
            "D_pred": np.zeros((n_keys, 13, 4, 8), dtype=np.float32),
            "d0_g_proj_all": np.zeros((n_keys, 13, 8), dtype=np.float32),
            "d0_sr0_all": np.zeros((n_keys, 13, 8), dtype=np.float32),
            "d0_h0_all": np.zeros((n_keys, 13, 8), dtype=np.float32),
            "d0_h1_all": np.zeros((n_keys, 13, 8), dtype=np.float32),
            "D_bp_post": np.zeros((n_keys, 13, 4, 8), dtype=np.float32),
            "D_raw_p2": np.zeros((n_keys, 13, 4, 8), dtype=np.float32),
            "feat_bits_p2": np.zeros((n_keys, 13, 3, 8), dtype=np.float32),
            "D_k4_p2": np.zeros((n_keys, 13, 3, 8), dtype=np.float32),
            "feat_bc_p2": np.zeros((n_keys, 13, 3, 8), dtype=np.float32),
            "D_unb_p2": np.zeros((n_keys, 13, 4, 8), dtype=np.float64),
        }
        for m in trace_budgets
    }

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_holdout_trace1_worker,
        initargs=(t2_ref,),
    ) as ex:
        probe_spk = min(2, eff_spk)
        probe_shards = [
            full_paths[k_idx * full_spk + s] for k_idx in range(n_keys) for s in range(probe_spk)
        ]
        res_probe = list(ex.map(_read_shard_holdout_trace1_strict, probe_shards))
        ds_std = float(np.std([item[5] for item in res_probe]))
        del res_probe
        domain_matched = abs(ds_std - REF_TRACE2_STD) / REF_TRACE2_STD < 0.10

        if console is not None:
            console.print(
                f"[bold cyan][2/4][/bold cyan] Streaming holdout `trace1` shards (`trace1` ONLY -> `trace2` POIs) across "
                f"[bold]{n_keys}[/bold] keys x [bold]{traces_per_key}[/bold] traces/key "
                f"([bold]{total_eval_traces:,}[/bold] traces, {n_workers} workers)..."
            )
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]{task.description:<38}"),
                BarColumn(bar_width=36, complete_style="cyan", finished_style="bright_green"),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TextColumn("[bold yellow]{task.fields[rate]:>12}[/bold yellow]"),
                TimeElapsedColumn(),
                TextColumn("ETA"),
                TimeRemainingColumn(),
                console=console,
                refresh_per_second=10,
            )
            progress.start()
        else:
            progress = None
            print(
                f"\n[2/4] Streaming holdout `trace1` shards (`trace1` ONLY -> `trace2` POIs) "
                f"across {n_keys} keys x {traces_per_key} traces/key "
                f"({n_key_batches} key batch(es) x {n_chunks} trace chunk(s), {n_workers} workers)..."
            )

        try:
            if domain_matched:
                if progress is None:
                    print(
                        f"    [*] Reconstructed holdout `trace2` std={ds_std:.2f} ADC matches reference domain "
                        f"(~{REF_TRACE2_STD:.2f} ADC); using pre-profiled ZCA whitening."
                    )
                whiten_models: dict[str, tuple[np.ndarray, np.ndarray]] | None = mdl[
                    "whiten_models"
                ]
            elif n_key_batches == 1 and n_chunks == 1:
                if progress is None:
                    print(
                        f"    [*] Reconstructed holdout `trace2` std={ds_std:.2f} ADC differs from reference domain "
                        f"(~{REF_TRACE2_STD:.2f} ADC); fitting global unsupervised target-domain ZCA whitening "
                        f"in-place across {n_keys} keys x {traces_per_key} traces/key..."
                    )
                whiten_models = None
            else:
                w_spk = min(eff_spk, max(24, (768 + n_keys - 1) // n_keys))
                w_M = w_spk * 64
                if progress is None:
                    print(
                        f"    [*] Reconstructed holdout `trace2` std={ds_std:.2f} ADC differs from reference domain "
                        f"(~{REF_TRACE2_STD:.2f} ADC); fitting global unsupervised target-domain ZCA whitening "
                        f"across {n_keys} keys x {w_M} traces/key..."
                    )
                w_shards = [
                    full_paths[k_idx * full_spk + s]
                    for k_idx in range(n_keys)
                    for s in range(w_spk)
                ]
                t_w0 = time.time()
                if progress is not None:
                    w_task = progress.add_task(
                        "Target ZCA Whitening Calibration",
                        total=len(w_shards) * 64,
                        rate="0.0 tr/s",
                    )
                    res_w = []
                    for idx_sh, item in enumerate(
                        ex.map(_read_shard_holdout_trace1_strict, w_shards)
                    ):
                        res_w.append(item)
                        done_tr = (idx_sh + 1) * 64
                        rate_s = f"{done_tr / max(1e-6, time.time() - t_w0):.1f} tr/s"
                        progress.update(w_task, advance=64, rate=rate_s)
                else:
                    res_w = list(ex.map(_read_shard_holdout_trace1_strict, w_shards))

                whiten_models = {}
                for idx_w, (name_w, shape_w) in enumerate(
                    [
                        ("ha", (n_keys, w_M, 18)),
                        ("hb", (n_keys, w_M, 18)),
                        ("ra", (n_keys, w_M, 64, 30)),
                        ("rb", (n_keys, w_M, 64, 30)),
                        ("pre", (n_keys, w_M, 13, 56)),
                        ("tr", (n_keys, w_M, 13, 240)),
                    ]
                ):
                    arr_w = np.empty(shape_w, dtype=np.float32)
                    for ik in range(n_keys):
                        for s in range(w_spk):
                            arr_w[ik, 64 * s : 64 * (s + 1)] = res_w[ik * w_spk + s][idx_w]
                    whiten_models[name_w] = fit_whiten(arr_w, max_keys=max(32, n_keys))
                    del arr_w
                del res_w

            t_stream0 = time.time()
            stream_done_traces = 0
            stream_task = (
                progress.add_task(
                    "Streaming `trace1` -> `trace2` POIs",
                    total=total_eval_traces,
                    rate="0.0 tr/s",
                )
                if progress is not None
                else None
            )
            p2_task = (
                progress.add_task(
                    "Pass-1/2 4-Mode Walsh Extraction",
                    total=n_key_batches * len(trace_budgets),
                    rate="pending",
                )
                if progress is not None
                else None
            )

            for kb in range(n_key_batches):
                k_start = kb * key_batch_size
                k_end = min(n_keys, (kb + 1) * key_batch_size)
                cur_K = k_end - k_start

                stream_data = {
                    "d0_dir_sh": np.zeros((cur_K, eff_spk, 13, 8), dtype=np.float32),
                    "G_sh": np.zeros((cur_K, eff_spk, 13, 13, 8), dtype=np.float32),
                    "vals_i_sh": np.zeros((cur_K, eff_spk, 13, 4), dtype=np.float32),
                    "v_rem": np.zeros((cur_K, traces_per_key, 13, 4, 8), dtype=np.float32),
                    "s_r0": np.zeros((cur_K, traces_per_key, 13, 8), dtype=np.float32),
                    "s_r1": np.zeros((cur_K, traces_per_key, 13, 8), dtype=np.float32),
                    "z_r0": np.zeros((cur_K, traces_per_key, 13, 4, 8), dtype=np.float32),
                    "z_r1": np.zeros((cur_K, traces_per_key, 13, 4, 8), dtype=np.float32),
                    "psi1_raw": np.zeros((cur_K, traces_per_key, 13, 4), dtype=np.float32),
                    "psi2_raw": np.zeros((cur_K, traces_per_key, 13, 4), dtype=np.float32),
                    "psi3_raw": np.zeros((cur_K, traces_per_key, 13, 4), dtype=np.float32),
                }

                for c in range(n_chunks):
                    tc = time.time()
                    sh_start = c * chunk_spk
                    sh_end = min(eff_spk, (c + 1) * chunk_spk)
                    cur_spk = sh_end - sh_start
                    cur_M = cur_spk * 64
                    t_start_c = sh_start * 64
                    t_end_c = sh_end * 64

                    c_shards = []
                    for k_idx in range(k_start, k_end):
                        base = k_idx * full_spk
                        c_shards.extend(full_paths[base + sh_start : base + sh_end])

                    hdr_a = np.empty((cur_K, cur_M, 18), dtype=np.float32)
                    hdr_b = np.empty((cur_K, cur_M, 18), dtype=np.float32)
                    remap_a = np.empty((cur_K, cur_M, 64, 30), dtype=np.float32)
                    remap_b = np.empty((cur_K, cur_M, 64, 30), dtype=np.float32)
                    pre_sbox = np.empty((cur_K, cur_M, 13, 56), dtype=np.float32)
                    traces = np.empty((cur_K, cur_M, 13, 240), dtype=np.float32)

                    for idx_sh, item in enumerate(
                        ex.map(_read_shard_holdout_trace1_strict, c_shards)
                    ):
                        ik = idx_sh // cur_spk
                        s = idx_sh % cur_spk
                        hdr_a[ik, 64 * s : 64 * (s + 1)] = item[0]
                        hdr_b[ik, 64 * s : 64 * (s + 1)] = item[1]
                        remap_a[ik, 64 * s : 64 * (s + 1)] = item[2]
                        remap_b[ik, 64 * s : 64 * (s + 1)] = item[3]
                        pre_sbox[ik, 64 * s : 64 * (s + 1)] = item[4]
                        traces[ik, 64 * s : 64 * (s + 1)] = item[5]
                        stream_done_traces += 64
                        if progress is not None and stream_task is not None:
                            rate_s = (
                                f"{stream_done_traces / max(1e-6, time.time() - t_stream0):.1f} tr/s"
                            )
                            progress.update(
                                stream_task,
                                advance=64,
                                description=f"Streaming `trace1` (Keys {k_start}..{k_end - 1})",
                                rate=rate_s,
                            )

                    if whiten_models is None:
                        _fix_bimodal_trace_gain_in_place(
                            hdr_a, hdr_b, remap_a, remap_b, pre_sbox, traces
                        )
                        whiten_models = {
                            "ha": fit_whiten(hdr_a, max_keys=max(32, cur_K)),
                            "hb": fit_whiten(hdr_b, max_keys=max(32, cur_K)),
                            "ra": fit_whiten(remap_a, max_keys=max(32, cur_K)),
                            "rb": fit_whiten(remap_b, max_keys=max(32, cur_K)),
                            "pre": fit_whiten(pre_sbox, max_keys=max(32, cur_K)),
                            "tr": fit_whiten(traces, max_keys=max(32, cur_K)),
                        }

                    c_out = extract_chunk_features(
                        hdr_a,
                        hdr_b,
                        remap_a,
                        remap_b,
                        pre_sbox,
                        traces,
                        whiten_models,
                        ops_fast,
                        perm_u,
                        psi_mats,
                    )
                    del hdr_a, hdr_b, remap_a, remap_b, pre_sbox, traces
                    stream_data["d0_dir_sh"][:, sh_start:sh_end] = c_out["d0_dir_sh"]
                    stream_data["G_sh"][:, sh_start:sh_end] = c_out["G_sh"]
                    stream_data["vals_i_sh"][:, sh_start:sh_end] = c_out["vals_i_sh"]
                    for feat_name in (
                        "v_rem",
                        "s_r0",
                        "s_r1",
                        "z_r0",
                        "z_r1",
                        "psi1_raw",
                        "psi2_raw",
                        "psi3_raw",
                    ):
                        stream_data[feat_name][:, t_start_c:t_end_c] = c_out[feat_name]
                    del c_out
                    if progress is None:
                        print(
                            f"    [KeyBatch {kb + 1}/{n_key_batches} | Chunk {c + 1:2d}/{n_chunks}] "
                            f"Keys {k_start}..{k_end - 1}, M={t_end_c:5d}/{traces_per_key} traces/key "
                            f"in {time.time() - tc:.1f}s",
                            flush=True,
                        )

                for m in trace_budgets:
                    tm = time.time()
                    batch_p2 = extract_batch_p2_features_at_m(
                        stream_data, m, mdl, ops_fast, psi_mats
                    )
                    for k_feat, v_feat in batch_p2.items():
                        p2_feats_by_m[m][k_feat][k_start:k_end] = v_feat
                    if progress is not None and p2_task is not None:
                        progress.update(
                            p2_task,
                            advance=1,
                            description=f"Pass-1/2 Walsh Extract (M={m})",
                            rate=f"{time.time() - tm:.2f}s/m",
                        )
                    else:
                        print(
                            f"    [Pass-1/2 Extract | Keys {k_start}..{k_end - 1}] Completed M={m:5d} "
                            f"in {time.time() - tm:.1f}s",
                            flush=True,
                        )
                del stream_data
        finally:
            if progress is not None:
                progress.stop()

    return p2_feats_by_m, domain_matched


def solve_posteriors_across_budgets(
    p2_feats_by_m: dict[int, dict[str, np.ndarray]],
    trace_budgets: list[int],
    n_keys: int,
    mdl: dict[str, Any],
    domain_matched: bool,
    console: Any,
) -> tuple[dict[int, np.ndarray], float]:
    """Run Global Self-Calibrated Coupled 4-Mode Walsh + S-Box Tree BP across all trace budgets."""
    post_ll_by_m: dict[int, np.ndarray] = {
        m: np.zeros((n_keys, 13, 4, 256), dtype=np.float64) for m in trace_budgets
    }
    if console is not None:
        console.print(
            f"[bold cyan][3/4][/bold cyan] Running Global Self-Calibrated Coupled 4-Mode Walsh + S-Box Tree BP across "
            f"[bold]{len(trace_budgets)}[/bold] trace budgets ([bold]{trace_budgets[0]}..{trace_budgets[-1]}[/bold] traces/key)..."
        )
        bp_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description:<38}"),
            BarColumn(bar_width=36, complete_style="cyan", finished_style="bright_green"),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TextColumn("[bold yellow]{task.fields[rate]:>12}[/bold yellow]"),
            TimeElapsedColumn(),
            console=console,
            refresh_per_second=10,
        )
        bp_progress.start()
        bp_task = bp_progress.add_task(
            "Global Self-Calibrated Coupled BP",
            total=len(trace_budgets),
            rate="0.0s/m",
        )
    else:
        bp_progress = None
        bp_task = None
        print(
            f"\n[3/4] Running Global Self-Calibrated Coupled 4-Mode Walsh + S-Box Tree BP across "
            f"{len(trace_budgets)} trace budgets ({trace_budgets[0]}..{trace_budgets[-1]} traces/key)..."
        )

    t_bp0 = time.time()
    try:
        for m in trace_budgets:
            tm = time.time()
            post_ll_by_m[m] = solve_global_self_calibrated_bp_at_m(
                p2_feats_by_m[m], m, mdl, domain_matched=domain_matched
            )
            if bp_progress is not None and bp_task is not None:
                bp_progress.update(
                    bp_task,
                    advance=1,
                    description=f"Global Coupled BP Solver (M={m})",
                    rate=f"{time.time() - tm:.2f}s/m",
                )
            else:
                print(
                    f"    [Global BP Solver | All {n_keys} Keys] Completed M={m:5d} in {time.time() - tm:.1f}s",
                    flush=True,
                )
    finally:
        if bp_progress is not None:
            bp_progress.stop()
    return post_ll_by_m, time.time() - t_bp0


def evaluate_holdout_predictions(
    full_paths: list[str],
    n_keys: int,
    full_spk: int,
    n_workers: int,
    trace_budgets: list[int],
    post_ll_by_m: dict[int, np.ndarray],
    r_spine_final: np.ndarray,
    c_sat_final: np.ndarray,
    top_k: int,
    top_n: int,
    console: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, int]:
    """Load ground-truth `holdout` keys (`Attributes(6)`) strictly for evaluation and score predictions."""
    if console is not None:
        console.print(
            "[bold cyan][4/4][/bold cyan] Reading ground-truth holdout keys (`Attributes(6)`) strictly for evaluation..."
        )
    else:
        print("\n[4/4] Reading ground-truth holdout keys (`Attributes(6)`) strictly for evaluation...")

    first_shards = [full_paths[k_idx * full_spk] for k_idx in range(n_keys)]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        true_keys_bytes = np.array(
            list(ex.map(_read_shard_ground_truth_key_only, first_shards)), dtype=np.uint8
        )
    K_true, W_true = build_key_labels(true_keys_bytes)

    metrics_by_m: dict[str, Any] = {}
    for m in trace_budgets:
        m_res = compute_metrics_at_budget(post_ll_by_m[m], W_true, K_true, m)
        preds_m = decode_predicted_keys_from_posteriors(
            post_ll_by_m[m],
            np.zeros(n_keys, dtype=np.float64),
            np.zeros(n_keys, dtype=np.float64),
            top_k=top_k,
            top_n_keys=top_n,
        )
        top_n_hits_m = 0
        for k_idx, pm in enumerate(preds_m):
            gt_hex_k = true_keys_bytes[k_idx].tobytes().hex()
            if any(c["key_hex"] == gt_hex_k for c in pm["top_n_key_candidates"]):
                top_n_hits_m += 1
        m_res["top_n_full_keys_recovered"] = int(top_n_hits_m)
        m_res["top_n_full_key_success_rate"] = float(top_n_hits_m / max(1, n_keys))
        metrics_by_m[str(m)] = m_res

    final_m = trace_budgets[-1]
    predictions = decode_predicted_keys_from_posteriors(
        post_ll_by_m[final_m],
        r_spine_final,
        c_sat_final,
        top_k=top_k,
        top_n_keys=top_n,
    )
    verified_top_n_count = 0
    for k, pred in enumerate(predictions):
        gt_bytes = true_keys_bytes[k]
        gt_hex = gt_bytes.tobytes().hex()
        pred_bytes = np.frombuffer(bytes.fromhex(pred["predicted_key_hex"]), dtype=np.uint8)
        exact_match = bool(pred["predicted_key_hex"] == gt_hex)
        byte_matches = int(np.sum(pred_bytes == gt_bytes))
        top_k_byte_matches = int(
            sum(
                1
                for b_idx in range(32)
                if any(
                    c["byte_int"] == int(gt_bytes[b_idx])
                    for c in pred["byte_top_k_candidates"][b_idx]
                )
            )
        )
        matched_rank: int | None = None
        for cand in pred["top_n_key_candidates"]:
            cand_bytes = np.frombuffer(bytes.fromhex(cand["key_hex"]), dtype=np.uint8)
            cand["matching_bytes_out_of_32"] = int(np.sum(cand_bytes == gt_bytes))
            cand["exact_match"] = bool(cand["key_hex"] == gt_hex)
            if cand["exact_match"] and matched_rank is None:
                matched_rank = int(cand["rank"])

        pred["ground_truth_key_hex"] = gt_hex
        pred["exact_match"] = exact_match
        pred["matched_in_top_n_rank"] = matched_rank
        pred["matching_bytes_out_of_32"] = byte_matches
        pred["top_k_matching_bytes_out_of_32"] = top_k_byte_matches
        if matched_rank is not None:
            verified_top_n_count += 1

    return metrics_by_m, predictions, true_keys_bytes, verified_top_n_count


def render_results_rich(
    console: Any,
    dataset_name: str,
    metrics_by_m: dict[str, Any],
    predictions: list[dict[str, Any]],
    true_keys_bytes: np.ndarray,
    final_m: int,
    verified_top_n_count: int,
    top_k: int,
    top_n: int,
    t_start: float,
    dt_bp: float,
) -> None:
    """Render Rich tables and summary panel for trace budgets, predicted keys, and Top-N candidates."""
    final_res = metrics_by_m[str(final_m)]
    budget_table = Table(
        title=(
            f"[bold white]Trace Budget Evaluation (Blind `trace1 -> trace2` Reconstruction): "
            f"Success Rate vs. Number of Traces ({dataset_name})[/bold white]"
        ),
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        pad_edge=False,
        padding=(0, 1),
    )
    budget_table.add_column("Traces (M)", justify="right", style="bold white")
    budget_table.add_column("Top-1 Keys", justify="right")
    budget_table.add_column(f"Top-{top_n} Keys", justify="right")
    budget_table.add_column("Tree44 Full", justify="right")
    budget_table.add_column("Free8 Full", justify="right")
    budget_table.add_column("Key32 Top-1", justify="right")
    budget_table.add_column("Key32 Top-5", justify="right")
    budget_table.add_column("Key32 MR", justify="right")
    budget_table.add_column("Key32 Bit", justify="right")
    budget_table.add_column("Tree44 Top-1", justify="right")
    budget_table.add_column("Free8 Top-1", justify="right")

    for res in metrics_by_m.values():
        fk = res["full_keys_recovered"]
        tnk = res["top_n_full_keys_recovered"]
        tk = res["total_keys"]
        t44 = res["tree44_full_keys_recovered"]
        f8 = res["free8_full_keys_recovered"]
        fk_st = "bold bright_green" if fk == tk else ("bold yellow" if fk > 0 else "dim")
        tn_st = "bold bright_green" if tnk == tk else ("bold yellow" if tnk > 0 else "dim")
        t44_st = "bold bright_green" if t44 == tk else ("cyan" if t44 > 0 else "dim")
        budget_table.add_row(
            f"{res['M']:,d}",
            f"[{fk_st}]{fk:3d} / {tk:<3d}[/{fk_st}]",
            f"[{tn_st}]{tnk:3d} / {tk:<3d}[/{tn_st}]",
            f"[{t44_st}]{t44:3d} / {tk:<3d}[/{t44_st}]",
            f"{f8:3d} / {tk:<3d}",
            f"{res['byte_top1_success_rate'] * 100:6.2f}%",
            f"{res['byte_top5_success_rate'] * 100:6.2f}%",
            f"{res['mean_byte_rank']:6.2f}",
            f"{res['bit_success_rate'] * 100:6.2f}%",
            f"{res['tree44_byte_top1_success_rate'] * 100:6.2f}%",
            f"{res['free8_byte_top1_success_rate'] * 100:6.2f}%",
        )
    console.print()
    console.print(budget_table)

    legend = (
        "[bold bright_green]■ Exact Byte Match (Top-1)[/bold bright_green]   "
        f"[bold yellow]■ Correct Byte in Top-{top_k}[/bold yellow]   "
        f"[dim red]■ Outside Top-{top_k}[/dim red]"
    )
    main_table = Table(
        title=(
            f"[bold white]Predicted 256-Bit AES Master Keys ({dataset_name} — "
            f"split='holdout', M={final_m} traces/key)[/bold white]\n{legend}"
        ),
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        show_lines=False,
        pad_edge=False,
        padding=(0, 1),
    )
    main_table.add_column("Key", justify="right", style="bold white", min_width=3, no_wrap=True)
    main_table.add_column("Predicted 256-Bit Master Key (64 Hex Chars)", no_wrap=True, min_width=64)
    main_table.add_column("Top-1", justify="right", min_width=5, no_wrap=True)
    main_table.add_column(f"Top-{top_k}", justify="right", min_width=5, no_wrap=True)
    main_table.add_column("Tree44", justify="right", min_width=6, no_wrap=True)
    main_table.add_column("Key32", justify="right", min_width=6, no_wrap=True)
    main_table.add_column("Spine", justify="right", min_width=6, no_wrap=True)
    main_table.add_column("Status", justify="left", min_width=10, no_wrap=True)

    for pred in predictions:
        k_idx = pred["key_index"]
        conf = pred["confidence"]
        gt_b = true_keys_bytes[k_idx]
        colored_key = _format_colored_key_text(
            pred["predicted_key_hex"], gt_b, pred["byte_top_k_candidates"]
        )
        n_b = pred["matching_bytes_out_of_32"]
        n_tk = pred["top_k_matching_bytes_out_of_32"]
        b_style = "bold bright_green" if n_b == 32 else ("yellow" if n_b >= 24 else "white")
        tk_style = "bold bright_green" if n_tk == 32 else ("yellow" if n_tk >= 28 else "white")
        if pred["exact_match"]:
            st_cell = "[bold bright_green]✔ Rank 1[/bold bright_green]"
        elif pred["matched_in_top_n_rank"] is not None:
            st_cell = f"[bold yellow]★ Rank {pred['matched_in_top_n_rank']}[/bold yellow]"
        else:
            st_cell = f"[dim]{n_b:2d}/32[/dim]"
        main_table.add_row(
            f"{k_idx:d}",
            colored_key,
            f"[{b_style}]{n_b:2d}/32[/{b_style}]",
            f"[{tk_style}]{n_tk:2d}/32[/{tk_style}]",
            f"{conf['tree44_mean_byte_prob'] * 100:5.1f}%",
            f"{conf['key32_mean_byte_prob'] * 100:5.1f}%",
            f"{conf['spine_coherence_r']:+.3f}",
            st_cell,
        )

    console.print()
    console.print(main_table)

    non_top1 = [p for p in predictions if not p["exact_match"]]
    if non_top1:
        rescue_table = Table(
            title=(
                f"[bold white]Top-{top_n} Full-Key Candidates for Non-Rank-1 Keys "
                f"(Highlighted Bytes)[/bold white]"
            ),
            box=box.ROUNDED,
            header_style="bold yellow",
            pad_edge=False,
            padding=(0, 1),
        )
        rescue_table.add_column("Key", justify="right", style="bold white", min_width=3, no_wrap=True)
        rescue_table.add_column("Cand", justify="right", style="cyan", min_width=4, no_wrap=True)
        rescue_table.add_column(
            "Candidate 256-Bit Master Key (Highlighted Bytes)", no_wrap=True, min_width=64
        )
        rescue_table.add_column("Rel Prob", justify="right", min_width=8, no_wrap=True)
        rescue_table.add_column("Match", justify="right", min_width=14, no_wrap=True)

        for p in non_top1[:15]:
            k_idx = p["key_index"]
            gt_b = true_keys_bytes[k_idx]
            for cand in p["top_n_key_candidates"]:
                c_txt = _format_colored_key_text(cand["key_hex"], gt_b, p["byte_top_k_candidates"])
                m_str = (
                    "[bold bright_green]✔ 32/32 EXACT[/bold bright_green]"
                    if cand["exact_match"]
                    else f"{cand['matching_bytes_out_of_32']:2d}/32"
                )
                rescue_table.add_row(
                    f"{k_idx:d}",
                    f"#{cand['rank']}",
                    c_txt,
                    f"{cand['relative_prob']:.4f}",
                    m_str,
                )
        console.print()
        console.print(rescue_table)

    summary_grid = Table.grid(padding=(0, 3))
    summary_grid.add_column(style="bold cyan", justify="right")
    summary_grid.add_column(style="bold white")
    summary_grid.add_row(
        "Top-1 Exact 256-Bit Master Keys:",
        f"[bold bright_green]{final_res['full_keys_recovered']} / {final_res['total_keys']} "
        f"({final_res['full_key_success_rate'] * 100:.2f}%)[/bold bright_green]",
    )
    summary_grid.add_row(
        f"Top-{top_n} Exact 256-Bit Master Keys:",
        f"[bold yellow]{verified_top_n_count} / {final_res['total_keys']} "
        f"({100.0 * verified_top_n_count / max(1, final_res['total_keys']):.2f}%)[/bold yellow]",
    )
    summary_grid.add_row(
        "Full 44-Byte S-Box Trees (`Tree44`):",
        f"{final_res['tree44_full_keys_recovered']} / {final_res['total_keys']} "
        f"({final_res['tree44_full_success_rate'] * 100:.2f}%  |  Byte Top-1: {final_res['tree44_byte_top1_success_rate'] * 100:.2f}%, "
        f"Mean Rank: {final_res['tree44_mean_byte_rank']:.2f})",
    )
    summary_grid.add_row(
        "8 Free S-Box Bytes (`m=5,6`):",
        f"{final_res['free8_full_keys_recovered']} / {final_res['total_keys']} "
        f"({final_res['free8_full_success_rate'] * 100:.2f}%  |  Byte Top-1: {final_res['free8_byte_top1_success_rate'] * 100:.2f}%, "
        f"Top-5: {final_res['free8_byte_top5_success_rate'] * 100:.2f}%, Mean Rank: {final_res['free8_mean_byte_rank']:.2f})",
    )
    summary_grid.add_row(
        "32 Master Key Bytes Recovery:",
        f"Top-1: [bold]{final_res['byte_top1_success_rate'] * 100:.2f}%[/bold]  |  "
        f"Top-5: [bold]{final_res['byte_top5_success_rate'] * 100:.2f}%[/bold]  |  "
        f"Mean Rank: [bold]{final_res['mean_byte_rank']:.2f} / 255[/bold]  |  "
        f"Bit Acc: [bold]{final_res['bit_success_rate'] * 100:.2f}%[/bold]",
    )
    summary_grid.add_row(
        "Key Word HW Distinguishing Rate:",
        f"{final_res['word_hw_success_rate'] * 100:.2f}% (r = {final_res['word_hw_corr_800']:+.4f}, "
        f"z = {final_res['word_hw_z_score']:+.2f} sigma)",
    )
    summary_grid.add_row(
        "Total Elapsed Time:",
        f"{time.time() - t_start:.1f}s (BP decoding: {dt_bp:.1f}s)",
    )
    console.print()
    console.print(
        Panel(
            summary_grid,
            title=f"[bold bright_green]Final Blind `trace1 -> trace2` Holdout Key Recovery Summary (M={final_res['M']} traces/key)[/bold bright_green]",
            border_style="bright_green",
            box=box.ROUNDED,
        )
    )


def render_results_text(
    dataset_name: str,
    metrics_by_m: dict[str, Any],
    predictions: list[dict[str, Any]],
    final_m: int,
    verified_top_n_count: int,
    top_k: int,
    top_n: int,
) -> None:
    """Render plain-text tables and summary for trace budgets and predicted keys."""
    final_res = metrics_by_m[str(final_m)]
    print("\n" + "=" * 136)
    print(
        "TRACE BUDGET EVALUATION (BLIND `trace1 -> trace2` RECONSTRUCTION): "
        "SUCCESS RATE VS. NUMBER OF TRACES (HOLDOUT SPLIT)"
    )
    print("=" * 136)
    print(
        f"{'Traces (M)':>10s} | {'Top-1 Keys':>11s} | {f'Top-{top_n} Keys':>11s} | "
        f"{'Tree44 Full':>11s} | {'Free8 Full':>10s} | {'Key32 Top-1':>11s} | "
        f"{'Key32 Top-5':>11s} | {'Key32 MR':>8s} | {'Key32 Bit':>9s} | "
        f"{'Tree44 Top-1':>12s} | {'Free8 Top-1':>11s}"
    )
    print("-" * 136)
    for res in metrics_by_m.values():
        print(
            f"{res['M']:10d} | "
            f"{res['full_keys_recovered']:4d} / {res['total_keys']:<4d} | "
            f"{res['top_n_full_keys_recovered']:4d} / {res['total_keys']:<4d} | "
            f"{res['tree44_full_keys_recovered']:4d} / {res['total_keys']:<4d} | "
            f"{res['free8_full_keys_recovered']:4d} / {res['total_keys']:<3d} | "
            f"{res['byte_top1_success_rate'] * 100:10.2f}% | "
            f"{res['byte_top5_success_rate'] * 100:10.2f}% | "
            f"{res['mean_byte_rank']:8.2f} | "
            f"{res['bit_success_rate'] * 100:8.2f}% | "
            f"{res['tree44_byte_top1_success_rate'] * 100:11.2f}% | "
            f"{res['free8_byte_top1_success_rate'] * 100:10.2f}%"
        )
    print("=" * 136)
    print(
        f"\nPREDICTED 256-BIT AES MASTER KEYS ({dataset_name} — split='holdout', M={final_m} traces/key)"
    )
    print("-" * 136)
    print(
        f"{'Key #':>5s} | {'Predicted 256-Bit Master Key (64 Hex Chars)':<64s} | "
        f"{'Top-1':>6s} | {f'Top-{top_k}':>6s} | {'Tree44':>7s} | {'Key32':>7s} | {'Spine r':>7s} | Status"
    )
    print("-" * 136)
    for pred in predictions:
        k_idx = pred["key_index"]
        k_hex = pred["predicted_key_hex"]
        conf = pred["confidence"]
        if pred["exact_match"]:
            st = "EXACT (Rank 1)"
        elif pred["matched_in_top_n_rank"] is not None:
            st = f"EXACT (Top-{top_n} #{pred['matched_in_top_n_rank']})"
        else:
            st = f"{pred['matching_bytes_out_of_32']:2d}/32"
        print(
            f"{k_idx:5d} | {k_hex:<64s} | "
            f"{pred['matching_bytes_out_of_32']:2d}/32  | "
            f"{pred['top_k_matching_bytes_out_of_32']:2d}/32  | "
            f"{conf['tree44_mean_byte_prob'] * 100:6.1f}% | "
            f"{conf['key32_mean_byte_prob'] * 100:6.1f}% | "
            f"{conf['spine_coherence_r']:+7.3f} | {st}"
        )
    print("=" * 136)
    print(
        f"[*] Final Blind `trace1 -> trace2` Holdout Key Recovery Summary at M={final_res['M']} traces/key:\n"
        f"    -> Full 256-Bit Master Keys Recovered (Top-1): {final_res['full_keys_recovered']} / {final_res['total_keys']} "
        f"({final_res['full_key_success_rate'] * 100:.2f}%)\n"
        f"    -> Full 256-Bit Master Keys Recovered (Top-{top_n}): {verified_top_n_count} / {final_res['total_keys']} "
        f"({100.0 * verified_top_n_count / max(1, final_res['total_keys']):.2f}%)\n"
        f"    -> Full 44-Byte S-Box Trees Recovered:         {final_res['tree44_full_keys_recovered']} / {final_res['total_keys']} "
        f"({final_res['tree44_full_success_rate'] * 100:.2f}%, Byte Top-1: {final_res['tree44_byte_top1_success_rate'] * 100:.2f}%, "
        f"Mean Rank: {final_res['tree44_mean_byte_rank']:.2f})\n"
        f"    -> 8 Free S-Box Bytes (m=5,6) Recovery:        {final_res['free8_full_keys_recovered']} / {final_res['total_keys']} "
        f"({final_res['free8_full_success_rate'] * 100:.2f}%, Byte Top-1: {final_res['free8_byte_top1_success_rate'] * 100:.2f}%, "
        f"Top-5: {final_res['free8_byte_top5_success_rate'] * 100:.2f}%, Mean Rank: {final_res['free8_mean_byte_rank']:.2f})\n"
        f"    -> 32 Master Key Bytes Top-1 Recovery Rate:    {final_res['byte_top1_success_rate'] * 100:.2f}% "
        f"(Top-5: {final_res['byte_top5_success_rate'] * 100:.2f}%, Mean Rank: {final_res['mean_byte_rank']:.2f} / 255)\n"
        f"    -> 256-Bit Master Key Bit Recovery Rate:       {final_res['bit_success_rate'] * 100:.2f}%\n"
        f"    -> Key Word HW Distinguishing Rate:            {final_res['word_hw_success_rate'] * 100:.2f}% "
        f"(r = {final_res['word_hw_corr_800']:+.4f}, z = {final_res['word_hw_z_score']:+.2f} sigma)"
    )


def _run_pipeline(args: argparse.Namespace, console: Any) -> None:
    """Execute the validated standalone holdout key-recovery pipeline."""
    validate_cli_args(args)

    dataset_dir = Path(
        args.dataset if args.dataset is not None else args.dataset_path
    ).expanduser().resolve()
    templates_path = Path(args.templates_path).expanduser().resolve()
    output_plot_path = Path(args.output_plot).expanduser().resolve()
    output_json_path = Path(args.output_json).expanduser().resolve()

    # Validate required input files and dataset split metadata before rendering progress banners.
    t2_ref = load_canonical_t2_ref()
    full_paths = discover_split_shard_files(dataset_dir, "holdout")
    disk_total_traces = len(full_paths) * 64
    n_keys_disk, full_tpk = load_holdout_split_metadata(
        dataset_dir, n_total_traces=disk_total_traces
    )
    if not templates_path.exists():
        raise FileNotFoundError(
            f"Pre-profiled Walsh + S-Box Tree BP template file not found: {templates_path}. "
            "Provide a valid path via '--templates-path <file>'."
        )
    if not templates_path.is_file():
        raise IsADirectoryError(f"Template path is not a regular file: {templates_path}")

    n_keys = min(n_keys_disk, args.max_keys) if args.max_keys is not None else n_keys_disk
    traces_per_key = (
        min(full_tpk, (args.max_traces_per_key // 64) * 64)
        if args.max_traces_per_key is not None
        else full_tpk
    )
    full_spk = full_tpk // 64
    if full_spk < 1 or traces_per_key < 64:
        raise ValueError(
            f"Holdout split in {dataset_dir} requires at least 64 traces per key, got {traces_per_key}."
        )
    if len(full_paths) < n_keys * full_spk:
        raise ValueError(
            f"Holdout split in {dataset_dir} has {len(full_paths)} shard(s), "
            f"fewer than the {n_keys * full_spk} required for {n_keys} key(s) x {full_spk} shard(s)/key."
        )

    t_start = time.time()
    if console is not None:
        header_grid = Table.grid(padding=(0, 2))
        header_grid.add_column(style="bold cyan", justify="right")
        header_grid.add_column(style="white")
        header_grid.add_row("Dataset Path (Holdout `trace1`):", str(dataset_dir))
        header_grid.add_row(
            "Holdout Inputs Used:",
            "[bold green]trace1 (Attr 0), plaintext (Attr 2), ciphertext (Attr 3) ONLY[/bold green]",
        )
        header_grid.add_row("Walsh + Tree-BP Templates:", str(templates_path))
        header_grid.add_row(
            "Outputs:",
            f"Plot: [underline]{output_plot_path.name}[/underline] | JSON: [underline]{output_json_path.name}[/underline]",
        )
        console.print(
            Panel(
                header_grid,
                title="[bold white]RP2350 AES-256 CTR — End-to-End Blind `trace1 -> trace2` & Coupled 4-Mode Walsh + S-Box Tree BP Pipeline[/bold white]",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )
        console.print(
            f"[bold cyan][1/4][/bold cyan] Loading pre-profiled Walsh + S-Box Tree BP templates from [bold]{templates_path.name}[/bold]..."
        )
    else:
        print("=" * 100)
        print(
            "END-TO-END BLIND `trace1 -> trace2` RECONSTRUCTION + COUPLED 4-MODE WALSH & S-BOX TREE BP PIPELINE"
        )
        print("=" * 100)
        print(f"Dataset Path (Holdout `trace1`): {dataset_dir}")
        print(
            "Holdout Inputs Used:             trace1 (Attr 0), plaintext (Attr 2), ciphertext (Attr 3) ONLY"
        )
        print(f"Walsh + Tree-BP Templates:       {templates_path}")
        print(f"Output Plot:                     {output_plot_path}")
        print(f"Output JSON:                     {output_json_path}")
        print("=" * 100)
        print(
            f"\n[1/4] Loading pre-profiled Walsh + S-Box Tree BP templates from {templates_path.name}..."
        )

    mdl = load_walsh_tree_templates(templates_path)
    ops_fast, perm_u, psi_mats = prepare_operator_matrices(mdl)
    trace_budgets = build_trace_budget_schedule(traces_per_key)
    n_workers = args.workers or min(24, os.cpu_count() or 4)

    poi6_diag = run_poi6_holdout_diagnostics(full_paths, t2_ref, n_workers, console)

    p2_feats_by_m, domain_matched = stream_holdout_and_extract_p2_features(
        full_paths=full_paths,
        n_keys=n_keys,
        traces_per_key=traces_per_key,
        full_spk=full_spk,
        trace_budgets=trace_budgets,
        mdl=mdl,
        ops_fast=ops_fast,
        perm_u=perm_u,
        psi_mats=psi_mats,
        t2_ref=t2_ref,
        n_workers=n_workers,
        console=console,
    )

    post_ll_by_m, dt_bp = solve_posteriors_across_budgets(
        p2_feats_by_m=p2_feats_by_m,
        trace_budgets=trace_budgets,
        n_keys=n_keys,
        mdl=mdl,
        domain_matched=domain_matched,
        console=console,
    )

    final_m = trace_budgets[-1]
    r_spine_final, c_sat_final = _compute_unsupervised_spine_metrics(p2_feats_by_m[final_m])
    del p2_feats_by_m

    metrics_by_m, predictions, true_keys_bytes, verified_top_n_count = (
        evaluate_holdout_predictions(
            full_paths=full_paths,
            n_keys=n_keys,
            full_spk=full_spk,
            n_workers=n_workers,
            trace_budgets=trace_budgets,
            post_ll_by_m=post_ll_by_m,
            r_spine_final=r_spine_final,
            c_sat_final=c_sat_final,
            top_k=args.top_k,
            top_n=args.top_n,
            console=console,
        )
    )
    final_res = metrics_by_m[str(final_m)]

    if console is not None:
        render_results_rich(
            console=console,
            dataset_name=dataset_dir.name,
            metrics_by_m=metrics_by_m,
            predictions=predictions,
            true_keys_bytes=true_keys_bytes,
            final_m=final_m,
            verified_top_n_count=verified_top_n_count,
            top_k=args.top_k,
            top_n=args.top_n,
            t_start=t_start,
            dt_bp=dt_bp,
        )
    else:
        render_results_text(
            dataset_name=dataset_dir.name,
            metrics_by_m=metrics_by_m,
            predictions=predictions,
            final_m=final_m,
            verified_top_n_count=verified_top_n_count,
            top_k=args.top_k,
            top_n=args.top_n,
        )

    plot_success_rate_vs_traces(metrics_by_m, output_plot_path)
    summary_payload: dict[str, Any] = {
        "dataset_path": str(dataset_dir),
        "split_used": "holdout",
        "holdout_attributes_used": [
            "trace1 (Attr 0)",
            "plaintext (Attr 2)",
            "ciphertext (Attr 3)",
        ],
        "trace_source": "trace1 ONLY (standalone blind trace1 -> trace2 POI reconstruction)",
        "num_holdout_keys": int(n_keys),
        "max_traces_per_key": int(traces_per_key),
        "keys_recovered_at_max_traces": int(final_res["full_keys_recovered"]),
        "full_key_success_rate_at_max_traces": float(final_res["full_key_success_rate"]),
        "top_n_keys_recovered_at_max_traces": int(verified_top_n_count),
        "top_n_full_key_success_rate_at_max_traces": float(
            verified_top_n_count / max(1, final_res["total_keys"])
        ),
        "tree44_full_keys_recovered_at_max_traces": int(final_res["tree44_full_keys_recovered"]),
        "tree44_full_success_rate_at_max_traces": float(final_res["tree44_full_success_rate"]),
        "poi6_holdout_diagnostics": poi6_diag,
        "trace_budget_curve": metrics_by_m,
        "elapsed_seconds": round(time.time() - t_start, 2),
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    if console is not None:
        console.print(
            f"[bold green][+][/bold green] Saved plot to [underline]{output_plot_path.name}[/underline] "
            f"and JSON summary to [underline]{output_json_path.name}[/underline] "
            f"(total time: {time.time() - t_start:.1f}s)."
        )
    else:
        print(
            f"[*] Saved plot to {output_plot_path.name} and JSON summary to {output_json_path.name} "
            f"(total time: {time.time() - t_start:.1f}s)."
        )


def main() -> None:
    args = parse_args()
    use_rich = RICH_AVAILABLE and not args.force_text
    term_size = shutil.get_terminal_size((128, 40))
    console = (
        Console(
            force_terminal=True,
            width=max(128, term_size.columns),
            height=max(40, term_size.lines),
        )
        if use_rich
        else None
    )
    try:
        _run_pipeline(args, console)
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError, ValueError, RuntimeError) as exc:
        _exit_with_cli_error(exc, use_rich)


if __name__ == "__main__":
    main()
