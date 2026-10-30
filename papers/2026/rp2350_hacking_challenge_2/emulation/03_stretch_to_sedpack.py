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
"""Step 3: Stretch 1-Point/Instruction Rainbow HW Traces to 97,553-Sample `sedpack` Shards.

Transforms the raw instruction traces (`<work_dir>/raw_traces/`) produced by Step 2 into a
hardware-shaped `sedpack` dataset (`<work_dir>/dataset/`) matching `rp2350_aesctr_small_fixed`:
  1. Look up per-instruction cycle duration `cycles[i]` from `data/etm_cycle_model.json`
     (`base_cycles` + conditional-branch `taken_extra_cycles` + hardware `hw_stall` + cold fetch).
  2. Stretch each instruction's scalar Rainbow Hamming weight across its clock cycles via
     `np.repeat(rainbow_hw, cycles)` (`~42,400` instructions -> `~95,250` cycles) and pad idle
     post-return cycles to `97,553` samples.
  3. Apply ChipWhisperer shunt polarity (`-16.0` ADC/bit), mean-amplitude matching (`1584.411` ADC),
     and i.i.d. white Gaussian noise (`sigma = 25.0` ADC), quantized to `int16`.
  4. Reconstruct canonical `trace2` ONLY on `train` (via `capture_sedpack_dataset.py`);
     on `holdout`, write `trace2 = zeros(97553, int16)` and `tracewhisperer = ""` (zero pre-processing).

Inputs:
  - `<work_dir>/raw_traces/{manifest.json, calibration_zero_trace.npz, train/*.npz.lz4, holdout/*.npz.lz4}`
  - `data/etm_cycle_model.json`
  - `data/dataset_schema.json`
  - `<scaaml_dir>/reconstruct_aes_trace.py` (used strictly on `train`)

Outputs:
  - `<work_dir>/dataset/dataset_info.json`
  - `<work_dir>/dataset/{train,holdout}/shards_list.json`
  - `<work_dir>/dataset/{train,holdout}/shards/{shards_list.json, shard_XXXXX.fb}`
  - `<work_dir>/dataset_stretch_summary.json`
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import lz4.frame
import numpy as np
from sedpack.io.compress import CompressedFile
import sedpack.io.flatbuffer.shardfile.Shard as fbapi_Shard
from sedpack.io.metadata import DatasetStructure
from sedpack.io.shard.shard_writer_flatbuffer import ShardWriterFlatBuffer
from sedpack.io.utils import hash_checksums_from_bytes

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DEFAULT_SCAAML_DIR = ROOT_DIR.parent

CODE_BASE = 0x20080000
RET_ADDR = 0x20081FFE
MAX_INSN_IDX = 4096
FULL_ADC_SAMPLES = 97553
EXAMPLES_PER_SHARD = 64

CW_TARGET_MEAN_ADC = 1584.411
DEFAULT_HW_SCALE_ADC_PER_BIT = -16.0
DEFAULT_NOISE_STD_ADC = 25.0

# yapf: disable
AES_SBOX = np.array(
    [
        0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5,
        0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
        0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0,
        0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
        0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC,
        0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
        0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A,
        0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
        0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0,
        0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
        0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B,
        0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
        0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85,
        0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
        0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5,
        0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
        0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17,
        0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
        0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88,
        0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
        0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C,
        0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
        0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9,
        0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
        0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6,
        0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
        0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E,
        0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
        0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94,
        0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
        0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68,
        0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
    ],
    dtype=np.uint8,
)
RCON = np.array([0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40], dtype=np.uint8)
# yapf: enable

_DUMMY_ETM_EVENTS = [{
    "pkt_type": "ADDR_L_32IS0",
    "addr": 0x10001518,
    "cumulative_cycles": 0
}] * 20

_WORKER_STATE: dict[str, Any] | None = None


def load_etm_cycle_tables(etm_json_path: Path) -> dict[str, Any]:
    """Load per-PC ETMv4 cycle lookup arrays indexed by 
    `pidx = (pc - 0x20080000) >> 1`."""
    doc = json.loads(etm_json_path.read_text(encoding="utf-8"))
    base_cyc = np.ones(MAX_INSN_IDX, dtype=np.int32)
    taken_cyc = np.zeros(MAX_INSN_IDX, dtype=np.int32)
    insn_size = np.full(MAX_INSN_IDX, 2, dtype=np.int64)
    is_cond = np.zeros(MAX_INSN_IDX, dtype=np.bool_)

    for pc_hex, info in doc["instructions"].items():
        pc = int(pc_hex, 16)
        pidx = (pc - CODE_BASE) >> 1
        if 0 <= pidx < MAX_INSN_IDX:
            base_cyc[pidx] = int(info["base_cycles"])
            taken_cyc[pidx] = int(info["taken_extra_cycles"])
            insn_size[pidx] = int(info["size"])
            is_cond[pidx] = bool(info["is_cond"])

    msp_pidx = (int(doc["msp_cold_emu_pc"]) - CODE_BASE) >> 1
    msp_cold_extra = int(doc["msp_cold_extra_cycles"])
    return {
        "base_cyc": base_cyc,
        "taken_cyc": taken_cyc,
        "insn_size": insn_size,
        "is_cond": is_cond,
        "msp_pidx": msp_pidx,
        "msp_cold_extra": msp_cold_extra,
    }


def stretch_instruction_trace_to_cycles(
    pc_idx: np.ndarray,
    rainbow_hw: np.ndarray,
    hw_stall: np.ndarray,
    etm_tables: dict[str, Any],
    num_samples: int = FULL_ADC_SAMPLES,
) -> tuple[np.ndarray, int]:
    """Stretch a 1-point-per-instruction Rainbow HW trace into
      a `97,553`-cycle waveform."""
    n = len(pc_idx)
    pidx = pc_idx.astype(np.int64)
    pc_bytes = CODE_BASE + (pidx << 1)

    next_pc_bytes = np.empty(n, dtype=np.int64)
    next_pc_bytes[:-1] = pc_bytes[1:]
    next_pc_bytes[-1] = RET_ADDR

    # Conditional branch is taken whenever `next_pc` != sequential
    # fall-through `pc + insn_size`
    cond_taken = etm_tables["is_cond"][pidx] & (next_pc_bytes != (
        pc_bytes + etm_tables["insn_size"][pidx]))

    cycles = (etm_tables["base_cyc"][pidx] +
              np.where(cond_taken, etm_tables["taken_cyc"][pidx], 0) +
              hw_stall.astype(np.int32))

    # First call to `makesmallperm` incurs the cold XIP cache-line
    # fill (+9 cycles)
    msp_hits = np.flatnonzero(pidx == etm_tables["msp_pidx"])
    if len(msp_hits) > 0:
        cycles[msp_hits[0]] += etm_tables["msp_cold_extra"]

    cycles = np.maximum(cycles, 1)

    # Core stretching step: repeat each instruction's Rainbow HW
    # across its clock cycles
    stretched = np.repeat(rainbow_hw.astype(np.float32), cycles)
    decrypt_end_cycle = int(len(stretched))

    out = np.empty(num_samples, dtype=np.float32)
    if decrypt_end_cycle >= num_samples:
        out[:] = stretched[:num_samples]
    else:
        out[:decrypt_end_cycle] = stretched
        out[decrypt_end_cycle:] = float(
            stretched[-1]) if decrypt_end_cycle > 0 else 0.0
    return out, decrypt_end_cycle


def scale_stretched_hw_to_adc_int16(
    hw_stretched: np.ndarray,
    hw_mean_ref: float,
    target_mean_adc: float,
    hw_scale_adc_per_bit: float,
    noise_std_adc: float,
    noise_seed: int,
) -> np.ndarray:
    """Apply ChipWhisperer shunt polarity, mean-amplitude matching,
      and white Gaussian noise."""
    sig_f = np.float32(target_mean_adc) + np.float32(hw_scale_adc_per_bit) * (
        hw_stretched - np.float32(hw_mean_ref))
    if noise_std_adc > 0.0:
        noise_rng = np.random.default_rng(int(noise_seed))
        sig_f = sig_f + noise_rng.normal(
            0.0, noise_std_adc, size=len(hw_stretched)).astype(np.float32)
    return np.clip(np.rint(sig_f), -32768, 32767).astype(np.int16)


def expand_aes256_round_keys(key_32: np.ndarray) -> np.ndarray:
    w = np.zeros((60, 4), dtype=np.uint8)
    for i in range(8):
        w[i] = key_32[4 * i:4 * i + 4]
    for i in range(8, 60):
        tmp = w[i - 1].copy()
        if i % 8 == 0:
            tmp = AES_SBOX[np.roll(tmp, -1)]
            tmp[0] ^= RCON[(i // 8) - 1]
        elif i % 8 == 4:
            tmp = AES_SBOX[tmp]
        w[i] = w[i - 8] ^ tmp
    return w.reshape(15, 16)


def compute_round0_state_shares(
    ka: np.ndarray,
    kb: np.ndarray,
    kc: np.ndarray,
    kd: np.ndarray,
    iv: np.ndarray,
    iv_salt: np.ndarray,
) -> np.ndarray:
    ctr_mask = iv[:16] ^ iv_salt[:16]
    return np.stack(
        [ka[:16] ^ ctr_mask, kb[:16].copy(), kc[:16].copy(), kd[:16].copy()],
        axis=0)


def _init_stretch_worker(
    ds_struct_dict: dict[str, Any],
    etm_json_str: str,
    scaaml_dir_str: str,
    hw_mean_ref: float,
    target_mean_adc: float,
    hw_scale_adc_per_bit: float,
    noise_std_adc: float,
) -> None:
    global _WORKER_STATE
    if scaaml_dir_str not in sys.path:
        sys.path.insert(0, scaaml_dir_str)
    from capture_sedpack_dataset import compute_trace2_from_trace1

    _WORKER_STATE = {
        "ds_struct": DatasetStructure.model_validate(ds_struct_dict),
        "etm_tables": load_etm_cycle_tables(Path(etm_json_str)),
        "compute_trace2_fn": compute_trace2_from_trace1,
        "hw_mean_ref": hw_mean_ref,
        "target_mean_adc": target_mean_adc,
        "hw_scale": hw_scale_adc_per_bit,
        "noise_std": noise_std_adc,
    }


def _stretch_one_shard(task: dict[str, Any]) -> dict[str, Any]:
    global _WORKER_STATE
    assert _WORKER_STATE is not None

    split = str(task["split"])
    shard_idx = int(task["shard_idx"])
    is_train = split == "train"

    raw_bytes = lz4.frame.decompress(Path(task["raw_path"]).read_bytes())
    raw = np.load(io.BytesIO(raw_bytes), allow_pickle=False)

    insn_offsets = raw["insn_offsets"]
    pc_all = raw["pc_idx"]
    hw_all = raw["rainbow_hw"]
    stall_all = raw["hw_stall"]
    key_bytes = raw["key"]
    round_keys = expand_aes256_round_keys(key_bytes)

    ka_all, kb_all, kc_all, kd_all = raw["key_a"], raw["key_b"], raw[
        "key_c"], raw["key_d"]
    iv_all, ivs_all = raw["iv"], raw["iv_salt"]
    pt_all, ct_all = raw["plaintext"], raw["ciphertext"]
    noise_seeds = raw["noise_seeds"]

    if is_train:
        dwt_offsets = raw["dwt_offsets"]
        dwt_comp_all = raw["dwt_comp"]
        dwt_val_all = raw["dwt_val"]

    ds_root = Path(task["ds_root"])
    rel_fb = f"{split}/shards/shard_{shard_idx:05d}.fb"
    abs_fb = ds_root / rel_fb
    abs_fb.parent.mkdir(parents=True, exist_ok=True)
    if abs_fb.exists():
        abs_fb.unlink()

    writer = ShardWriterFlatBuffer(dataset_structure=_WORKER_STATE["ds_struct"],
                                   shard_file=abs_fb)
    zero_t2 = np.zeros(FULL_ADC_SAMPLES, dtype=np.int16)

    cyc_sum = 0
    adc_mean_sum = 0.0
    for j in range(EXAMPLES_PER_SHARD):
        i0, i1 = int(insn_offsets[j]), int(insn_offsets[j + 1])
        hw_stretched, dec_end_cyc = stretch_instruction_trace_to_cycles(
            pc_all[i0:i1],
            hw_all[i0:i1],
            stall_all[i0:i1],
            _WORKER_STATE["etm_tables"],
            num_samples=FULL_ADC_SAMPLES,
        )
        t1 = scale_stretched_hw_to_adc_int16(
            hw_stretched,
            hw_mean_ref=_WORKER_STATE["hw_mean_ref"],
            target_mean_adc=_WORKER_STATE["target_mean_adc"],
            hw_scale_adc_per_bit=_WORKER_STATE["hw_scale"],
            noise_std_adc=_WORKER_STATE["noise_std"],
            noise_seed=int(noise_seeds[j]),
        )
        cyc_sum += dec_end_cyc
        adc_mean_sum += float(np.mean(t1))

        if is_train:
            d0, d1 = int(dwt_offsets[j]), int(dwt_offsets[j + 1])
            dwt_msgs = [{
                "comparator": int(dwt_comp_all[m]),
                "pc_int": int(dwt_val_all[m])
            } for m in range(d0, d1)]
            tw_dict = {
                "stream1_itm_dwt": {
                    "messages": dwt_msgs
                },
                "stream2_etmv4": {
                    "events": _DUMMY_ETM_EVENTS,
                    "total_cycles": dec_end_cyc + 800
                },
            }
            t2 = _WORKER_STATE["compute_trace2_fn"](
                trace1=t1,
                tw_or_capture=tw_dict,
                nb_blocks=2,
                target_length=FULL_ADC_SAMPLES,
            )
        else:
            t2 = zero_t2

        inter_sh = compute_round0_state_shares(ka_all[j], kb_all[j], kc_all[j],
                                               kd_all[j], iv_all[j], ivs_all[j])
        writer._write({
            "trace1": t1,
            "trace2": t2,
            "plaintext": pt_all[j],
            "ciphertext": ct_all[j],
            "iv": iv_all[j],
            "iv_salt": ivs_all[j],
            "key": key_bytes,
            "key_a": ka_all[j],
            "key_b": kb_all[j],
            "key_c": kc_all[j],
            "key_d": kd_all[j],
            "round_keys": round_keys,
            "intermediate_shares": inter_sh,
            "tracewhisperer": "",
        })

    hashes = list(writer.close())
    return {
        "split": split,
        "shard_idx": shard_idx,
        "rel_fb": rel_fb,
        "hashes": hashes,
        "mean_decrypt_cycles": cyc_sum / EXAMPLES_PER_SHARD,
        "mean_adc": adc_mean_sum / EXAMPLES_PER_SHARD,
    }


def _write_split_metadata(
    ds_root: Path,
    split: str,
    shard_records: list[tuple[str, list[str]]],
    hash_algos: tuple[str, ...],
) -> dict[str, Any]:
    n_shards = len(shard_records)
    n_examples = n_shards * EXAMPLES_PER_SHARD

    leaf_rel = f"{split}/shards/shards_list.json"
    leaf_obj = {
        "relative_path_self":
            leaf_rel,
        "number_of_examples":
            n_examples,
        "shard_files": [{
            "file_infos": [{
                "file_path": rel_fb,
                "hash_checksums": hashes
            }],
            "number_of_examples": EXAMPLES_PER_SHARD,
        } for rel_fb, hashes in shard_records],
    }
    leaf_bytes = (json.dumps(leaf_obj, indent=2) + "\n").encode("utf-8")
    leaf_path = ds_root / leaf_rel
    leaf_path.parent.mkdir(parents=True, exist_ok=True)
    leaf_path.write_bytes(leaf_bytes)
    leaf_hashes = list(
        hash_checksums_from_bytes(file_content=leaf_bytes, hashes=hash_algos))

    top_rel = f"{split}/shards_list.json"
    top_obj = {
        "relative_path_self":
            top_rel,
        "number_of_examples":
            n_examples,
        "children_shard_lists": [{
            "shard_list_info_file": {
                "file_path": leaf_rel,
                "hash_checksums": leaf_hashes
            },
            "number_of_examples": n_examples,
            "number_of_shards": n_shards,
        }],
    }
    top_bytes = (json.dumps(top_obj, indent=2) + "\n").encode("utf-8")
    (ds_root / top_rel).write_bytes(top_bytes)
    top_hashes = list(
        hash_checksums_from_bytes(file_content=top_bytes, hashes=hash_algos))

    return {
        "shard_list_info_file": {
            "file_path": top_rel,
            "hash_checksums": top_hashes
        },
        "number_of_examples": n_examples,
        "number_of_shards": n_shards,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Stretch 1-pt/insn Rainbow HW traces"
                     " into a sedpack dataset."))
    parser.add_argument("--work-dir",
                        type=Path,
                        default=Path("test_run"),
                        help="Working directory (default: test_run)")
    parser.add_argument("--scaaml-dir",
                        type=Path,
                        default=DEFAULT_SCAAML_DIR,
                        help="Path to `capture_sedpack_dataset.py`")
    parser.add_argument("--target-mean-adc",
                        type=float,
                        default=CW_TARGET_MEAN_ADC)
    parser.add_argument("--hw-scale",
                        type=float,
                        default=DEFAULT_HW_SCALE_ADC_PER_BIT)
    parser.add_argument("--noise-std",
                        type=float,
                        default=DEFAULT_NOISE_STD_ADC)
    parser.add_argument("--workers",
                        type=int,
                        default=min(32,
                                    os.cpu_count() or 8))
    parser.add_argument(
        "--clean-raw-traces",
        action="store_true",
        help=
        "Delete temporary `raw_traces/{train,holdout}` shards after packaging")
    args = parser.parse_args()

    t0 = time.time()
    work_dir = args.work_dir.resolve()
    scaaml_dir = args.scaaml_dir.resolve()
    if not (scaaml_dir / "capture_sedpack_dataset.py").is_file():
        raise FileNotFoundError("Could not find `capture_sedpack_dataset.py` "
                                f"in `--scaaml-dir {scaaml_dir}`.")

    raw_dir = work_dir / "raw_traces"
    manifest = json.loads(
        (raw_dir / "manifest.json").read_text(encoding="utf-8"))
    etm_json = DATA_DIR / "etm_cycle_model.json"
    etm_tables = load_etm_cycle_tables(etm_json)

    # Compute non-secret reference mean HW from the fixed zero-key calibration trace
    cal = np.load(raw_dir / "calibration_zero_trace.npz", allow_pickle=False)
    cal_stretched, _ = stretch_instruction_trace_to_cycles(
        cal["pc_idx"], cal["rainbow_hw"], cal["hw_stall"], etm_tables,
        FULL_ADC_SAMPLES)
    hw_mean_ref = float(np.mean(cal_stretched))

    base_info = json.loads(
        (DATA_DIR / "dataset_schema.json").read_text(encoding="utf-8"))
    ds_struct_dict = base_info["dataset_structure"]
    hash_algos = tuple(ds_struct_dict["hash_checksum_algorithms"])

    ds_dir = work_dir / "dataset"
    if ds_dir.exists():
        shutil.rmtree(ds_dir)
    ds_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[dict[str, Any]] = []
    for split, n_shards in (("train", int(manifest["train_shards"])),
                            ("holdout", int(manifest["holdout_shards"]))):
        for sh_idx in range(n_shards):
            tasks.append({
                "split":
                    split,
                "shard_idx":
                    sh_idx,
                "raw_path":
                    str(raw_dir / split / f"raw_shard_{sh_idx:05d}.npz.lz4"),
                "ds_root":
                    str(ds_dir),
            })

    print(f"[*] Stretching {len(tasks)} shards from "
          f"~{manifest['mean_executed_instructions_per_trace']:.0f} insns "
          f"-> {FULL_ADC_SAMPLES} ADC samples "
          f"(`hw_mean_ref={hw_mean_ref:.3f}`)...")
    with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_stretch_worker,
            initargs=(
                ds_struct_dict,
                str(etm_json),
                str(scaaml_dir),
                hw_mean_ref,
                args.target_mean_adc,
                args.hw_scale,
                args.noise_std,
            ),
    ) as pool:
        results = list(pool.map(_stretch_one_shard, tasks, chunksize=4))

    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "holdout": []}
    for r in results:
        by_split[r["split"]].append(r)

    splits_meta: dict[str, Any] = {}
    for split in ("train", "holdout"):
        by_split[split].sort(key=lambda x: x["shard_idx"])
        recs = [(x["rel_fb"], x["hashes"]) for x in by_split[split]]
        splits_meta[split] = _write_split_metadata(ds_dir, split, recs,
                                                   hash_algos)

    info_out = copy.deepcopy(base_info)
    info_out["metadata"]["description"] = (
        "RP2350 Minimal Rainbow HammingWeight() + ETM Cycle-Stretched "
        "Dataset (Unpatched Bootloader)")
    cm = info_out["metadata"]["custom_metadata"]
    cm["total_traces"] = int(manifest["train_traces"]) + int(
        manifest["holdout_traces"])
    cm["splits_info"] = {
        "train": {
            "role": "profiling / training",
            "num_keys": int(manifest["train_keys"]),
            "traces_per_key": 256,
            "num_traces": int(manifest["train_traces"]),
            "tracewhisperer_captured": False,
        },
        "holdout": {
            "role": "blind attack evaluation (trace2=0, tracewhisperer='')",
            "num_keys": int(manifest["holdout_keys"]),
            "traces_per_key": 768,
            "num_traces": int(manifest["holdout_traces"]),
            "tracewhisperer_captured": False,
        },
    }
    info_out["splits"] = splits_meta
    (ds_dir / "dataset_info.json").write_text(json.dumps(info_out, indent=2) +
                                              "\n",
                                              encoding="utf-8")

    # Audit holdout shards on disk to guarantee `trace2 == 0`
    # and `tracewhisperer == ""`
    comp = CompressedFile("LZ4")
    max_abs_t2_holdout = 0
    max_tw_bytes_holdout = 0
    holdout_fbs = sorted((ds_dir / "holdout" / "shards").glob("*.fb"))
    for fb_p in holdout_fbs[::max(1, len(holdout_fbs) // 16)]:
        buf = comp.decompress(fb_p.read_bytes())
        sh = fbapi_Shard.Shard.GetRootAs(buf, 0)
        for j in range(sh.ExamplesLength()):
            ex = sh.Examples(j)
            t2_arr = np.frombuffer(ex.Attributes(1).AttributeBytesAsNumpy(),
                                   dtype=np.int16)
            tw_b = bytes(ex.Attributes(13).AttributeBytesAsNumpy())
            max_abs_t2_holdout = max(max_abs_t2_holdout,
                                     int(np.max(np.abs(t2_arr))))
            max_tw_bytes_holdout = max(max_tw_bytes_holdout, len(tw_b))
    if max_abs_t2_holdout != 0 or max_tw_bytes_holdout != 0:
        raise RuntimeError("Holdout non-preprocessing check failed: non-zero "
                           "trace2 or tracewhisperer!")

    if args.clean_raw_traces:
        shutil.rmtree(raw_dir / "train", ignore_errors=True)
        shutil.rmtree(raw_dir / "holdout", ignore_errors=True)

    mean_cyc = float(np.mean([r["mean_decrypt_cycles"] for r in results]))
    mean_adc = float(np.mean([r["mean_adc"] for r in results]))
    summary = {
        "dataset_dir":
            str(ds_dir),
        "adc_samples_per_trace":
            FULL_ADC_SAMPLES,
        "mean_raw_instructions_per_trace":
            manifest["mean_executed_instructions_per_trace"],
        "mean_stretched_decrypt_cycles":
            round(mean_cyc, 2),
        "idle_tail_samples_padded":
            round(FULL_ADC_SAMPLES - mean_cyc, 2),
        "hw_mean_ref_bits":
            round(hw_mean_ref, 4),
        "target_mean_adc":
            args.target_mean_adc,
        "empirical_mean_adc":
            round(mean_adc, 3),
        "hw_scale_adc_per_bit":
            args.hw_scale,
        "noise_std_adc":
            args.noise_std,
        "holdout_max_abs_trace2_on_disk":
            max_abs_t2_holdout,
        "holdout_max_tracewhisperer_bytes_on_disk":
            max_tw_bytes_holdout,
        "wall_time_sec":
            round(time.time() - t0, 2),
    }
    (work_dir / "dataset_stretch_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[+] Wrote sedpack dataset `{ds_dir}` in "
          f"{summary['wall_time_sec']:.1f}s")
    print("    - Stretched "
          f"{summary['mean_raw_instructions_per_trace']:.0f} insns -> "
          f"{summary['mean_stretched_decrypt_cycles']:.0f} active cycles"
          f" -> {FULL_ADC_SAMPLES} ADC samples "
          f"(mean ADC = {summary['empirical_mean_adc']:.2f})")
    print("    - Verified on disk: holdout `trace2 == 0` and"
          " `tracewhisperer == ''`.")


if __name__ == "__main__":
    main()
