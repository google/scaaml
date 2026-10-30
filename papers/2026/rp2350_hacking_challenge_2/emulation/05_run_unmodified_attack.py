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
"""Step 5: Stage & Run Unmodified `end_to_end_attack.py`.

Why staging in `<work_dir>/attack_run/` is required for zero modifications:
  - `end_to_end_attack.py` (`load_canonical_t2_ref()`, line 280) resolves `canonical_t2_ref.npy`
    via `Path(__file__).resolve().parent / "canonical_t2_ref.npy"` without a CLI flag.
  - To avoid touching a single file, this script copies the byte-exact
    `end_to_end_attack.py` (verifying `SHA-256` before and after execution) alongside
    `<work_dir>/profiling/canonical_t2_ref.npy` inside
    `<work_dir>/attack_run/`, and invokes it on `<work_dir>/dataset`.

Inputs:
  - `<scaaml_dir>/end_to_end_attack.py` (unmodified upstream contest attack script)
  - `<work_dir>/dataset/`               (from `03_stretch_to_sedpack.py`)
  - `<work_dir>/profiling/canonical_t2_ref.npy` & `trace2_walsh_tree_bp_templates.pkl` (from Step 4)

Temporary / Staged Files:
  - `<work_dir>/attack_run/end_to_end_attack.py` (byte-identical file copy, SHA-256 verified)
  - `<work_dir>/attack_run/canonical_t2_ref.npy` (copied from `<work_dir>/profiling/`)

Outputs:
  - `<work_dir>/attack_run/attack_results.json`  (trace-budget key recovery metrics)
  - `<work_dir>/attack_run/attack_plot.png`      (success-rate vs. trace budget plot)
  - `<work_dir>/attack_run/attack_stdout.log`    (full console output of `end_to_end_attack.py`)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_SCAAML_DIR = ROOT_DIR.parent
EXPECTED_ATTACK_PY_SHA256 = "c2ecf5e23eb5dc9aab590c24ae06ac84c7ef143f6919d688adc6eb40fdf175b1"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_unmodified_attack_script(p: Path) -> str:
    digest = sha256_file(p)
    if digest != EXPECTED_ATTACK_PY_SHA256:
        raise RuntimeError(f"Attack script integrity mismatch on {p}: "
                           f"sha256={digest} != {EXPECTED_ATTACK_PY_SHA256}")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run unmodified `end_to_end_attack.py` on emulated dataset."
    )
    parser.add_argument("--work-dir",
                        type=Path,
                        default=Path("test_run"),
                        help="Working directory (default: test_run)")
    parser.add_argument("--scaaml-dir",
                        type=Path,
                        default=DEFAULT_SCAAML_DIR,
                        help="Path to `end_to_end_attack.py`")
    parser.add_argument("--workers",
                        type=int,
                        default=min(48,
                                    os.cpu_count() or 8))
    parser.add_argument("--max-keys",
                        type=int,
                        default=None,
                        help="Optional cap on holdout keys evaluated")
    parser.add_argument("--max-traces-per-key",
                        type=int,
                        default=None,
                        help="Optional cap on traces per key")
    args = parser.parse_args()

    work_dir = args.work_dir.resolve()
    scaaml_dir = args.scaaml_dir.resolve()
    upstream_attack_py = scaaml_dir / "end_to_end_attack.py"
    if not upstream_attack_py.is_file():
        raise FileNotFoundError(f"Could not find `{upstream_attack_py}`.")

    verify_unmodified_attack_script(upstream_attack_py)

    ds_dir = work_dir / "dataset"
    prof_dir = work_dir / "profiling"
    ref_npy = prof_dir / "canonical_t2_ref.npy"
    tmpl_pkl = prof_dir / "trace2_walsh_tree_bp_templates.pkl"
    for req in (ds_dir / "dataset_info.json", ref_npy, tmpl_pkl):
        if not req.is_file():
            raise FileNotFoundError(
                f"Missing required input `{req}`. Run Steps 1-4 first.")

    run_dir = work_dir / "attack_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Regular file copy (NOT symlink!) so `Path(__file__).resolve().parent`
    # points to `run_dir`
    staged_attack_py = run_dir / "end_to_end_attack.py"
    shutil.copyfile(upstream_attack_py, staged_attack_py)
    shutil.copyfile(ref_npy, run_dir / "canonical_t2_ref.npy")
    verify_unmodified_attack_script(staged_attack_py)

    out_json = run_dir / "attack_results.json"
    out_plot = run_dir / "attack_plot.png"
    out_log = run_dir / "attack_stdout.log"

    cmd = [
        sys.executable,
        str(staged_attack_py),
        "--dataset-path",
        str(ds_dir),
        "--templates-path",
        str(tmpl_pkl),
        "--workers",
        str(args.workers),
        "--output-json",
        str(out_json),
        "--output-plot",
        str(out_plot),
        "--text",
    ]
    if args.max_keys is not None:
        cmd.extend(["--max-keys", str(args.max_keys)])
    if args.max_traces_per_key is not None:
        cmd.extend(["--max-traces-per-key", str(args.max_traces_per_key)])

    print("[*] Running unmodified `end_to_end_attack.py` "
          f"(SHA-256 verified = {EXPECTED_ATTACK_PY_SHA256[:16]}...)...")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    dt = time.time() - t0
    out_log.write_text(
        proc.stdout +
        ("\n--- STDERR ---\n" + proc.stderr if proc.stderr else ""),
        encoding="utf-8")
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + "\n" + proc.stderr + "\n")
        raise RuntimeError(
            f"end_to_end_attack.py exited with status {proc.returncode}")

    # Re-verify both upstream and staged scripts remain byte-for-byte unmodified
    verify_unmodified_attack_script(upstream_attack_py)
    verify_unmodified_attack_script(staged_attack_py)

    res = json.loads(out_json.read_text(encoding="utf-8"))
    print(
        f"[+] Completed blind holdout evaluation in {dt:.1f}s -> `{out_json}`")
    n_k = res.get("num_holdout_keys", "?")
    for budget_str, metrics in res.get("trace_budget_curve", {}).items():
        t1 = metrics["full_key_success_rate"] * 100.0
        tn = metrics["top_n_full_key_success_rate"] * 100.0
        tr44 = metrics["tree44_full_success_rate"] * 100.0
        b1 = metrics["byte_top1_success_rate"] * 100.0
        print(f"    M={int(budget_str):4d} traces/key ({n_k} keys): "
              f"Top-1={t1:5.1f}% | Top-10={tn:5.1f}% | "
              f"Tree44={tr44:5.1f}% | ByteTop1={b1:5.2f}%")


if __name__ == "__main__":
    main()
