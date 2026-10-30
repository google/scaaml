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
"""Step 2: Instruction-Level Bootloader Emulation & Standard Rainbow Hamming-Weight Tracing.

Clearly loads the compiled unmodified bootloader ELF (`<work_dir>/build/aes_bootloader.elf`)
via its ELF `PT_LOAD` segments and `.symtab` symbols, executes 2-block AES-256-CTR encryptions
on Unicorn Cortex-M33 + RP2350 MMIO (`RESETS`, `ROSC`, `TRNG`, `SHA256`), and records raw
1-point-per-instruction traces (`pc_idx`, `rainbow_hw`, `hw_stall`) WITHOUT cycle stretching
or ADC noise.

Inputs:
  - `<work_dir>/build/aes_bootloader.elf`  (from `01_build_bootloader.py`)
  - `src/rp2350_unicorn_tracer.c`          (minimal hook-free Cortex-M33 + RP2350 MMIO C tracer)

Temporary / Intermediate Outputs:
  - `<work_dir>/build/rp2350_unicorn_tracer.so`          (compiled shared library linked against `unicorn`)
  - `<work_dir>/raw_traces/calibration_zero_trace.npz`   (fixed non-secret all-zero trace for mean HW ref)
  - `<work_dir>/raw_traces/sample_instruction_trace.csv` (human-readable 1-row/insn trace of encryption #0)
  - `<work_dir>/raw_traces/train/raw_shard_XXXXX.npz.lz4`   (LZ4-compressed raw instruction shards + train DWT)
  - `<work_dir>/raw_traces/holdout/raw_shard_XXXXX.npz.lz4` (LZ4-compressed raw instruction shards, 0 DWT!)
  - `<work_dir>/raw_traces/manifest.json`                (metadata, ELF segments loaded, key disjointness proof)
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time
from typing import Any

from capstone import CS_ARCH_ARM, CS_MODE_MCLASS, CS_MODE_THUMB, Cs
from Crypto.Cipher import AES
from elftools.elf.elffile import ELFFile
import lz4.frame
import numpy as np
import unicorn

ROOT_DIR = Path(__file__).resolve().parent
SRC_C_PATH = ROOT_DIR / "src" / "rp2350_unicorn_tracer.c"

CODE_BASE = 0x20080000
CODE_SIZE = 0x00002000
MAX_INSN_IDX = 4096
MAX_EXEC_INSNS = 131072
MAX_DWT_EVENTS = 2048
EXAMPLES_PER_SHARD = 64

# Exact 17 ARM registers tracked by `rainbow_arm(trace_config=TraceConfig(register=HammingWeight()))`
RAINBOW_ARM_REG_INDEX: dict[str, int] = {
    "r0": 0,
    "r1": 1,
    "r2": 2,
    "r3": 3,
    "r4": 4,
    "r5": 5,
    "r6": 6,
    "r7": 7,
    "r8": 8,
    "r9": 9,
    "sb": 9,
    "r10": 10,
    "sl": 10,
    "r11": 11,
    "fp": 11,
    "r12": 12,
    "ip": 12,
    "sp": 13,
    "r13": 13,
    "lr": 14,
    "r14": 14,
    "pc": 15,
    "r15": 15,
    "cpsr": 16,
    "apsr": 16,
    "apsr_nzcvq": 16,
    "xpsr": 16,
}

_WORKER_CTX: dict[str, Any] | None = None


def compile_tracer_shared_lib(so_path: Path) -> Path:
    """Compile `src/rp2350_unicorn_tracer.c` against the active Python
      venv's `unicorn` library."""
    so_path.parent.mkdir(parents=True, exist_ok=True)
    if so_path.is_file() and so_path.stat().st_mtime >= SRC_C_PATH.stat(
    ).st_mtime:
        return so_path

    unicorn_pkg_dir = Path(unicorn.__file__).resolve().parent
    inc_dir = unicorn_pkg_dir / "include"
    lib_dir = unicorn_pkg_dir / "lib"
    libs = sorted(lib_dir.glob("libunicorn.so*"))
    if not libs:
        raise FileNotFoundError(
            f"Could not locate libunicorn.so* inside {lib_dir}")
    # Prefer versioned SONAME (`libunicorn.so.2`) when present
    lib_file = next((p for p in libs if p.name == "libunicorn.so.2"), libs[0])

    cmd = [
        "gcc",
        "-O3",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-shared",
        "-fPIC",
        f"-I{inc_dir}",
        str(SRC_C_PATH),
        str(lib_file),
        f"-Wl,-rpath,{lib_dir}",
        "-o",
        str(so_path),
    ]
    subprocess.check_call(cmd)
    return so_path


def load_bootloader_elf(elf_path: Path) -> dict[str, Any]:
    """Explicitly load `PT_LOAD` segments and `.symtab` symbols from
      `aes_bootloader.elf`."""
    if not elf_path.is_file():
        raise FileNotFoundError(
            f"Missing {elf_path}. Run `python 01_build_bootloader.py"
            " --work-dir {elf_path.parent.parent}` first.")

    elf_bytes = elf_path.read_bytes()
    sram_image = bytearray(CODE_SIZE)
    loaded_segments: list[dict[str, Any]] = []

    ef = ELFFile(io.BytesIO(elf_bytes))
    for seg_idx, seg in enumerate(ef.iter_segments()):
        if seg["p_type"] != "PT_LOAD" or seg["p_memsz"] == 0:
            continue
        vaddr = int(seg["p_vaddr"])
        filesz = int(seg["p_filesz"])
        memsz = int(seg["p_memsz"])
        if not (CODE_BASE <= vaddr and vaddr + memsz <= CODE_BASE + CODE_SIZE):
            raise RuntimeError(f"ELF PT_LOAD segment #{seg_idx} @ 0x{vaddr:08x}"
                               " outside SRAM window!")
        off = vaddr - CODE_BASE
        sram_image[off:off + filesz] = seg.data()
        loaded_segments.append({
            "segment_index": seg_idx,
            "vaddr": f"0x{vaddr:08x}",
            "filesz_bytes": filesz,
            "memsz_bytes": memsz,
        })

    symtab = ef.get_section_by_name(".symtab")
    symbols: dict[str, int] = {}
    all_markers: dict[str, int] = {}
    func_ranges: list[tuple[int, str]] = []
    for s in symtab.iter_symbols():
        val = int(s["st_value"]) & ~1
        if s.name:
            all_markers[s.name] = val
            if not s.name.startswith("$"):
                symbols[s.name] = val
                if CODE_BASE <= val < CODE_BASE + 0x1000:
                    func_ranges.append((val, s.name))
    func_ranges.sort()

    text_sec = ef.get_section_by_name(".text")
    text_addr = int(text_sec["sh_addr"])
    text_bytes = text_sec.data()
    code_end = all_markers.get("$d", text_addr + len(text_bytes))

    md = Cs(CS_ARCH_ARM, CS_MODE_THUMB | CS_MODE_MCLASS)
    md.detail = True

    is_udiv = np.zeros(MAX_INSN_IDX, dtype=np.uint8)
    reg_write_mask = np.zeros(MAX_INSN_IDX, dtype=np.uint32)
    disasm_map: dict[int, tuple[str, str, str, str]] = {}

    def lookup_func_name(pc: int) -> str:
        name = "unknown"
        for f_addr, f_name in func_ranges:
            if f_addr <= pc:
                name = f_name
            else:
                break
        return name

    cur = text_addr
    while cur < code_end:
        ins_list = list(
            md.disasm(text_bytes[cur - text_addr:code_end - text_addr],
                      cur,
                      count=1))
        if ins_list:
            ins = ins_list[0]
            pidx = (ins.address - CODE_BASE) >> 1
            if 0 <= pidx < MAX_INSN_IDX:
                if ins.mnemonic.startswith("udiv"):
                    is_udiv[pidx] = 1
                _, rw = ins.regs_access()
                mask = 0
                written_names: list[str] = []
                for reg_id in rw:
                    rname = ins.reg_name(reg_id).lower()
                    if rname in RAINBOW_ARM_REG_INDEX:
                        bit = RAINBOW_ARM_REG_INDEX[rname]
                        if not mask & (1 << bit):
                            mask |= 1 << bit
                            written_names.append(rname)
                reg_write_mask[pidx] = mask
                disasm_map[pidx] = (
                    lookup_func_name(ins.address),
                    ins.mnemonic,
                    ins.op_str,
                    "+".join(written_names) if written_names else "-",
                )
            cur += ins.size
        else:
            cur += 2

    return {
        "elf_sha256": hashlib.sha256(elf_bytes).hexdigest(),
        "sram_image": np.frombuffer(bytes(sram_image), dtype=np.uint8).copy(),
        "loaded_segments": loaded_segments,
        "symbols": symbols,
        "is_udiv": is_udiv,
        "reg_write_mask": reg_write_mask,
        "disasm_map": disasm_map,
    }


def pack_interleaved_4way_key(
    key_a: np.ndarray,
    key_b: np.ndarray,
    key_c: np.ndarray,
    key_d: np.ndarray,
) -> np.ndarray:
    """Pack 4x32-byte Boolean key shares into the 128-byte word-interleaved
    buffer expected by `decrypt`."""
    out = bytearray(128)
    for w in range(8):
        s = slice(4 * w, 4 * w + 4)
        base = 16 * w
        out[base:base + 4] = key_a[s].tobytes()
        out[base + 4:base + 8] = key_b[s].tobytes()
        out[base + 8:base + 12] = key_c[s].tobytes()
        out[base + 12:base + 16] = key_d[s].tobytes()
    return np.frombuffer(bytes(out), dtype=np.uint8).copy()


def split_key_into_4_shares(key_32: np.ndarray,
                            rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    ka = rng.integers(0, 256, size=32, dtype=np.uint8)
    kb = rng.integers(0, 256, size=32, dtype=np.uint8)
    kc = rng.integers(0, 256, size=32, dtype=np.uint8)
    kd = (key_32 ^ ka ^ kb ^ kc).astype(np.uint8)
    return ka, kb, kc, kd


def aes256_ctr_reference_rp2350(
    key_32: bytes,
    iv_16: bytes,
    iv_salt_16: bytes,
    pt_32: bytes,
) -> bytes:
    """Reference AES-256-CTR matching RP2350 `aes.S` 
    (`(iv ^ iv_salt) ^ big_endian_u32(block_idx)`)."""
    eff_iv = bytes(a ^ b for a, b in zip(iv_16, iv_salt_16))
    cipher = AES.new(key_32, AES.MODE_ECB)
    out = bytearray()
    for blk in range(len(pt_32) // 16):
        ctr = bytearray(eff_iv)
        be_blk = struct.pack(">I", blk)
        for i in range(4):
            ctr[12 + i] ^= be_blk[i]
        ks = cipher.encrypt(bytes(ctr))
        pt_blk = pt_32[16 * blk:16 * (blk + 1)]
        out.extend(a ^ b for a, b in zip(pt_blk, ks))
    return bytes(out)


class TracerHandle:
    """Thin ctypes wrapper around `rp2350_unicorn_tracer.so`."""

    def __init__(self, so_path: Path, elf_info: dict[str, Any]) -> None:
        self._lib = ctypes.CDLL(str(so_path))
        u8_p = ctypes.POINTER(ctypes.c_uint8)
        u16_p = ctypes.POINTER(ctypes.c_uint16)
        u32_p = ctypes.POINTER(ctypes.c_uint32)
        u64_p = ctypes.POINTER(ctypes.c_uint64)
        i32_p = ctypes.POINTER(ctypes.c_int)

        self._lib.rp2350_tracer_create.argtypes = [
            u8_p,
            ctypes.c_int,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            u8_p,
            u32_p,
        ]
        self._lib.rp2350_tracer_create.restype = ctypes.c_void_p

        self._lib.rp2350_tracer_run_one.argtypes = [
            ctypes.c_void_p,
            u8_p,
            u8_p,
            u8_p,
            u8_p,
            ctypes.c_uint32,
            u64_p,
            ctypes.c_int,
            u16_p,
            u16_p,
            u8_p,
            u8_p,
            u8_p,
            u32_p,
            i32_p,
        ]
        self._lib.rp2350_tracer_run_one.restype = ctypes.c_int

        self._lib.rp2350_tracer_destroy.argtypes = [ctypes.c_void_p]
        self._lib.rp2350_tracer_destroy.restype = None

        sram = elf_info["sram_image"]
        syms = elf_info["symbols"]
        self._ctx = self._lib.rp2350_tracer_create(
            sram.ctypes.data_as(u8_p),
            ctypes.c_int(len(sram)),
            ctypes.c_uint32(int(syms["decrypt"])),
            ctypes.c_uint32(int(syms["rstate_sha"])),
            ctypes.c_uint32(int(syms["jstate"])),
            ctypes.c_uint32(int(syms["rstate_lfsr"])),
            elf_info["is_udiv"].ctypes.data_as(u8_p),
            elf_info["reg_write_mask"].ctypes.data_as(u32_p),
        )
        if not self._ctx:
            raise RuntimeError("rp2350_tracer_create returned NULL")

        self.pc_idx_buf = np.empty(MAX_EXEC_INSNS, dtype=np.uint16)
        self.hw_buf = np.empty(MAX_EXEC_INSNS, dtype=np.uint16)
        self.stall_buf = np.empty(MAX_EXEC_INSNS, dtype=np.uint8)
        self.ct_buf = np.empty(32, dtype=np.uint8)
        self.dwt_comp_buf = np.empty(MAX_DWT_EVENTS, dtype=np.uint8)
        self.dwt_val_buf = np.empty(MAX_DWT_EVENTS, dtype=np.uint32)

    def run_one(
        self,
        ka: np.ndarray,
        kb: np.ndarray,
        kc: np.ndarray,
        kd: np.ndarray,
        iv_salt: np.ndarray,
        iv: np.ndarray,
        pt: np.ndarray,
        hw_rng_seed: np.ndarray,
        record_dwt: bool,
    ) -> dict[str, np.ndarray]:
        u8_p = ctypes.POINTER(ctypes.c_uint8)
        u16_p = ctypes.POINTER(ctypes.c_uint16)
        u32_p = ctypes.POINTER(ctypes.c_uint32)
        u64_p = ctypes.POINTER(ctypes.c_uint64)

        k_inter = pack_interleaved_4way_key(ka, kb, kc, kd)
        dwt_n = ctypes.c_int(0)
        n_insns = self._lib.rp2350_tracer_run_one(
            self._ctx,
            k_inter.ctypes.data_as(u8_p),
            np.ascontiguousarray(iv_salt, dtype=np.uint8).ctypes.data_as(u8_p),
            np.ascontiguousarray(iv, dtype=np.uint8).ctypes.data_as(u8_p),
            np.ascontiguousarray(pt, dtype=np.uint8).ctypes.data_as(u8_p),
            ctypes.c_uint32(2),
            np.ascontiguousarray(hw_rng_seed,
                                 dtype=np.uint64).ctypes.data_as(u64_p),
            ctypes.c_int(1 if record_dwt else 0),
            self.pc_idx_buf.ctypes.data_as(u16_p),
            self.hw_buf.ctypes.data_as(u16_p),
            self.stall_buf.ctypes.data_as(u8_p),
            self.ct_buf.ctypes.data_as(u8_p),
            self.dwt_comp_buf.ctypes.data_as(u8_p) if record_dwt else None,
            self.dwt_val_buf.ctypes.data_as(u32_p) if record_dwt else None,
            ctypes.byref(dwt_n) if record_dwt else None,
        )
        if n_insns < 0:
            raise RuntimeError(
                f"Unicorn execution failed with uc_err={-n_insns}")

        out: dict[str, np.ndarray] = {
            "pc_idx": self.pc_idx_buf[:n_insns].copy(),
            "rainbow_hw": self.hw_buf[:n_insns].copy(),
            "hw_stall": self.stall_buf[:n_insns].copy(),
            "ciphertext": self.ct_buf.copy(),
        }
        if record_dwt:
            nd = int(dwt_n.value)
            out["dwt_comp"] = self.dwt_comp_buf[:nd].copy()
            out["dwt_val"] = self.dwt_val_buf[:nd].copy()
        return out

    def close(self) -> None:
        if getattr(self, "_ctx", None):
            self._lib.rp2350_tracer_destroy(self._ctx)
            self._ctx = None

    def __del__(self) -> None:
        self.close()


def _init_worker(so_path_str: str, elf_path_str: str) -> None:
    global _WORKER_CTX
    elf_info = load_bootloader_elf(Path(elf_path_str))
    tracer = TracerHandle(Path(so_path_str), elf_info)
    _WORKER_CTX = {"tracer": tracer}


def _emulate_raw_shard(task: dict[str, Any]) -> dict[str, Any]:
    global _WORKER_CTX
    assert _WORKER_CTX is not None
    tracer: TracerHandle = _WORKER_CTX["tracer"]

    split = str(task["split"])
    shard_idx = int(task["shard_idx"])
    is_train = split == "train"
    key_bytes = np.frombuffer(bytes.fromhex(task["key_hex"]), dtype=np.uint8)
    rng = np.random.default_rng(int(task["seed_u64"]))

    pc_chunks: list[np.ndarray] = []
    hw_chunks: list[np.ndarray] = []
    stall_chunks: list[np.ndarray] = []
    insn_offsets = np.zeros(EXAMPLES_PER_SHARD + 1, dtype=np.int32)

    dwt_comp_chunks: list[np.ndarray] = []
    dwt_val_chunks: list[np.ndarray] = []
    dwt_offsets = np.zeros(EXAMPLES_PER_SHARD + 1, dtype=np.int32)

    ka_all = np.empty((EXAMPLES_PER_SHARD, 32), dtype=np.uint8)
    kb_all = np.empty((EXAMPLES_PER_SHARD, 32), dtype=np.uint8)
    kc_all = np.empty((EXAMPLES_PER_SHARD, 32), dtype=np.uint8)
    kd_all = np.empty((EXAMPLES_PER_SHARD, 32), dtype=np.uint8)
    iv_all = np.empty((EXAMPLES_PER_SHARD, 16), dtype=np.uint8)
    ivs_all = np.empty((EXAMPLES_PER_SHARD, 16), dtype=np.uint8)
    pt_all = np.empty((EXAMPLES_PER_SHARD, 32), dtype=np.uint8)
    ct_all = np.empty((EXAMPLES_PER_SHARD, 32), dtype=np.uint8)
    noise_seeds = np.empty(EXAMPLES_PER_SHARD, dtype=np.uint64)

    for j in range(EXAMPLES_PER_SHARD):
        ka, kb, kc, kd = split_key_into_4_shares(key_bytes, rng)
        iv = rng.integers(0, 256, size=16, dtype=np.uint8)
        iv_salt = rng.integers(0, 256, size=16, dtype=np.uint8)
        pt = rng.integers(0, 256, size=32, dtype=np.uint8)
        hw_seed = rng.integers(1, 2**63 - 1, size=4, dtype=np.uint64)
        noise_seeds[j] = rng.integers(1, 2**63 - 1, dtype=np.uint64)

        res = tracer.run_one(
            ka,
            kb,
            kc,
            kd,
            iv_salt,
            iv,
            pt,
            hw_rng_seed=hw_seed,
            record_dwt=is_train,  # Strictly disabled (`0`) on `holdout`!
        )
        pc_chunks.append(res["pc_idx"])
        hw_chunks.append(res["rainbow_hw"])
        stall_chunks.append(res["hw_stall"])
        insn_offsets[j + 1] = insn_offsets[j] + len(res["pc_idx"])

        ka_all[j], kb_all[j], kc_all[j], kd_all[j] = ka, kb, kc, kd
        iv_all[j], ivs_all[j], pt_all[j], ct_all[j] = iv, iv_salt, pt, res[
            "ciphertext"]

        if is_train:
            dwt_comp_chunks.append(res["dwt_comp"])
            dwt_val_chunks.append(res["dwt_val"])
            dwt_offsets[j + 1] = dwt_offsets[j] + len(res["dwt_comp"])

    arrays: dict[str, np.ndarray] = {
        "insn_offsets": insn_offsets,
        "pc_idx": np.concatenate(pc_chunks),
        "rainbow_hw": np.concatenate(hw_chunks),
        "hw_stall": np.concatenate(stall_chunks),
        "key": key_bytes,
        "key_a": ka_all,
        "key_b": kb_all,
        "key_c": kc_all,
        "key_d": kd_all,
        "iv": iv_all,
        "iv_salt": ivs_all,
        "plaintext": pt_all,
        "ciphertext": ct_all,
        "noise_seeds": noise_seeds,
    }
    if is_train:
        arrays["dwt_offsets"] = dwt_offsets
        arrays["dwt_comp"] = np.concatenate(dwt_comp_chunks)
        arrays["dwt_val"] = np.concatenate(dwt_val_chunks)

    bio = io.BytesIO()
    np.savez(bio, **arrays)
    compressed = lz4.frame.compress(bio.getvalue())

    out_path = Path(task["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(compressed)

    return {
        "split": split,
        "shard_idx": shard_idx,
        "rel_path": f"{split}/{out_path.name}",
        "mean_insns_per_trace": float(insn_offsets[-1]) / EXAMPLES_PER_SHARD,
        "compressed_bytes": len(compressed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emulate RP2350 bootloader instruction + Rainbow HW traces."
    )
    parser.add_argument("--work-dir",
                        type=Path,
                        default=Path("test_run"),
                        help="Working directory (default: test_run)")
    parser.add_argument(
        "--train-keys",
        type=int,
        default=256,
        help="Number of 256-bit keys in `train` (256 traces/key)")
    parser.add_argument(
        "--holdout-keys",
        type=int,
        default=100,
        help="Number of 256-bit keys in `holdout` (768 traces/key)")
    parser.add_argument("--seed",
                        type=int,
                        default=0x525032333530464E,
                        help="Deterministic master PRNG seed")
    parser.add_argument("--workers",
                        type=int,
                        default=min(32,
                                    os.cpu_count() or 8),
                        help="Parallel worker processes")
    args = parser.parse_args()

    t0 = time.time()
    work_dir = args.work_dir.resolve()
    elf_path = work_dir / "build" / "aes_bootloader.elf"
    so_path = work_dir / "build" / "rp2350_unicorn_tracer.so"

    compile_tracer_shared_lib(so_path)
    elf_info = load_bootloader_elf(elf_path)
    print(f"[+] Loaded bootloader ELF `{elf_path}` (SHA-256: "
          f"{elf_info['elf_sha256'][:16]}...)")
    for seg in elf_info["loaded_segments"]:
        print(
            f"    - ELF PT_LOAD #{seg['segment_index']}: vaddr={seg['vaddr']},"
            f" filesz={seg['filesz_bytes']}B, memsz={seg['memsz_bytes']}B")
    print("    - Entry symbol `decrypt` = "
          f"0x{elf_info['symbols']['decrypt']:08x}")

    raw_dir = work_dir / "raw_traces"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    (raw_dir / "train").mkdir(parents=True, exist_ok=True)
    (raw_dir / "holdout").mkdir(parents=True, exist_ok=True)

    # Single-trace sanity check vs `pycryptodome` AES-256-CTR + export
    # human-readable CSV & calibration trace
    tracer = TracerHandle(so_path, elf_info)
    z32 = np.zeros(32, dtype=np.uint8)
    z16 = np.zeros(16, dtype=np.uint8)
    cal = tracer.run_one(
        z32,
        z32,
        z32,
        z32,
        z16,
        z16,
        z32,
        hw_rng_seed=np.array([1, 2, 3, 4], dtype=np.uint64),
        record_dwt=False,
    )
    np.savez(
        raw_dir / "calibration_zero_trace.npz",
        pc_idx=cal["pc_idx"],
        rainbow_hw=cal["rainbow_hw"],
        hw_stall=cal["hw_stall"],
    )

    test_rng = np.random.default_rng(12345)
    t_key = test_rng.integers(0, 256, size=32, dtype=np.uint8)
    ka, kb, kc, kd = split_key_into_4_shares(t_key, test_rng)
    t_iv = test_rng.integers(0, 256, size=16, dtype=np.uint8)
    t_ivs = test_rng.integers(0, 256, size=16, dtype=np.uint8)
    t_pt = test_rng.integers(0, 256, size=32, dtype=np.uint8)
    sample = tracer.run_one(
        ka,
        kb,
        kc,
        kd,
        t_ivs,
        t_iv,
        t_pt,
        hw_rng_seed=test_rng.integers(1, 2**63 - 1, size=4, dtype=np.uint64),
        record_dwt=False,
    )
    tracer.close()

    expected_ct = aes256_ctr_reference_rp2350(t_key.tobytes(), t_iv.tobytes(),
                                              t_ivs.tobytes(), t_pt.tobytes())
    if sample["ciphertext"].tobytes() != expected_ct:
        raise RuntimeError(
            "AES-256-CTR output mismatch between emulated bootloader "
            "and PyCryptodome reference!")
    print(f"[+] Verified 2-block AES-256-CTR output matches PyCryptodome"
          f" (`{len(sample['pc_idx'])}` ARM instructions executed).")

    csv_path = raw_dir / "sample_instruction_trace.csv"
    disasm_map = elf_info["disasm_map"]
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "step", "pc_hex", "function", "mnemonic", "operands",
            "regs_written", "rainbow_hw", "hw_stall"
        ])
        for step_i in range(len(sample["pc_idx"])):
            pidx = int(sample["pc_idx"][step_i])
            pc = CODE_BASE + (pidx << 1)
            fn_name, mnem, ops, rwritten = disasm_map.get(
                pidx, ("unknown", "?", "", "-"))
            writer.writerow([
                step_i,
                f"0x{pc:08x}",
                fn_name,
                mnem,
                ops,
                rwritten,
                int(sample["rainbow_hw"][step_i]),
                int(sample["hw_stall"][step_i]),
            ])
    print("[+] Wrote human-readable 1-point-per-instruction "
          f"sample trace: {csv_path}")

    # Generate strictly disjoint 256-bit keys for `train` and `holdout`
    master_rng = np.random.default_rng(args.seed)
    total_keys = args.train_keys + args.holdout_keys
    unique_keys: list[bytes] = []
    seen: set[bytes] = {b"\x00" * 32}
    while len(unique_keys) < total_keys:
        kb_cand = master_rng.integers(0, 256, size=32, dtype=np.uint8).tobytes()
        if kb_cand not in seen:
            seen.add(kb_cand)
            unique_keys.append(kb_cand)

    train_keys = unique_keys[:args.train_keys]
    holdout_keys = unique_keys[args.train_keys:]
    assert set(train_keys).isdisjoint(holdout_keys)

    tasks: list[dict[str, Any]] = []
    for k_idx, kb_val in enumerate(train_keys):
        for s_off in range(4):
            sh_idx = 4 * k_idx + s_off
            tasks.append({
                "split":
                    "train",
                "shard_idx":
                    sh_idx,
                "key_hex":
                    kb_val.hex(),
                "seed_u64":
                    int(master_rng.integers(1, 2**63 - 1, dtype=np.uint64)),
                "out_path":
                    str(raw_dir / "train" / f"raw_shard_{sh_idx:05d}.npz.lz4"),
            })
    for k_idx, kb_val in enumerate(holdout_keys):
        for s_off in range(12):
            sh_idx = 12 * k_idx + s_off
            tasks.append({
                "split":
                    "holdout",
                "shard_idx":
                    sh_idx,
                "key_hex":
                    kb_val.hex(),
                "seed_u64":
                    int(master_rng.integers(1, 2**63 - 1, dtype=np.uint64)),
                "out_path":
                    str(raw_dir / "holdout" / f"raw_shard_{sh_idx:05d}.npz.lz4"
                       ),
            })

    print(f"[*] Emulating {len(tasks)} raw instruction shards "
          f"(train: {args.train_keys} keys x 256, holdout: "
          f"{args.holdout_keys} keys x 768) "
          f"across {args.workers} workers...")
    with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(str(so_path), str(elf_path)),
    ) as pool:
        results = list(pool.map(_emulate_raw_shard, tasks, chunksize=4))

    total_bytes = sum(r["compressed_bytes"] for r in results)
    mean_insns = float(np.mean([r["mean_insns_per_trace"] for r in results]))
    manifest = {
        "elf_path": str(elf_path),
        "elf_sha256": elf_info["elf_sha256"],
        "loaded_segments": elf_info["loaded_segments"],
        "train_keys": args.train_keys,
        "train_shards": 4 * args.train_keys,
        "train_traces": 256 * args.train_keys,
        "holdout_keys": args.holdout_keys,
        "holdout_shards": 12 * args.holdout_keys,
        "holdout_traces": 768 * args.holdout_keys,
        "keys_strictly_disjoint": True,
        "holdout_dwt_recorded": False,
        "mean_executed_instructions_per_trace": round(mean_insns, 2),
        "total_compressed_raw_bytes": total_bytes,
        "wall_time_sec": round(time.time() - t0, 2),
    }
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) +
                                           "\n",
                                           encoding="utf-8")
    print(
        "[+] Completed Stage 1 raw instruction emulation in "
        f"{manifest['wall_time_sec']:.1f}s "
        f"(mean {mean_insns:.1f} insns/trace, {total_bytes / 1e6:.1f} MB LZ4).")


if __name__ == "__main__":
    main()
