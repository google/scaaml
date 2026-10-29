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

"""
capture_sedpack_dataset.py
==========================
Standalone Sedpack Dataset Generation, Hardware Capture & Verification Script
for RP2350 Masked AES-256 CTR (`aes.S`, SimpleSerial2 target).

Instrument: ChipWhisperer Husky / Husky Plus (ADC power capture + CoreSight ETMv4 / DWT TraceWhisperer)
Platform: Native `sedpack` (FlatBuffers, LZ4 compression)

Dataset Specifications:
  1. Raw & Canonical Traces:
     - `trace1`: Full un-cut, un-postprocessed power waveforms from ChipWhisperer Husky
       encompassing all 6 identified POIs (~78,400+ samples, or calibrated length).
     - `trace2`: Canonical de-jittered and un-permuted power waveform computed at capture
       time from `trace1` and the TraceWhisperer stream (`SH_JITTER`, `jstate`/SHA-256 refills,
       `CT_BPERM`, `ST_VPERM`, and round-key vertical permutations removed).
  2. Supervision Labels & Metadata:
     - `plaintext`: (16 * nb_blocks,) uint8
     - `ciphertext`: (16 * nb_blocks,) uint8
     - `iv`: (16,) uint8
     - `iv_salt`: (16,) uint8
     - `key`: (32,) uint8 (AES-256 master key)
     - `key_a`, `key_b`, `key_c`, `key_d`: each (32,) uint8 (4-way Boolean shares)
     - `round_keys`: (15, 16) uint8 (15 AES-256 round keys)
     - `intermediate_shares`: (4, 16) uint8 (Round 0 state shares)
     - `tracewhisperer`: () string (JSON-encoded decoded TraceWhisperer DWT/ITM and ETMv4 streams)
  3. Splits:
     - `train`: Profiling split with random keys, random plaintexts, and configurable IV mode
       (TraceWhisperer enabled).
     - `val` (stored in `test` / symlinked as `val`): Validation split (default 10% of train,
       TraceWhisperer enabled).
     - `holdout`: Attack evaluation split using independent random keys (supports full or
       `--holdout-minimal` capture).
"""

from __future__ import annotations

import argparse
import atexit
from collections import Counter
import concurrent.futures
import contextlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import signal
import struct
import sys
import threading
import time
from typing import Any, Callable, Iterator

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from Crypto.Cipher import AES
import numpy as np
from sedpack.io import Attribute, Dataset, DatasetStructure, Metadata
from sedpack.io.types import HOLDOUT_SPLIT, TEST_SPLIT, TRAIN_SPLIT, SplitT

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_BAUDRATE = 230400
DEFAULT_CLKGEN_FREQ = 24_000_000.0

# AES S-Box & Rcon for AES-256 round-key expansion
AES_SBOX = [
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
]
RCON = [0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]

# Architectural & Firmware Constants (from aes.S / rp2350_hacking_challenge_2.elf)
LFSR_FEEDBACK_CONST = 0x1D872B41
MURMUR3_C1 = 0xCC9E2D51
MURMUR3_C2 = 0x1B873593
MURMUR3_C3 = 0xE6546B64
MURMUR3_C4 = 0x85EBCA6B
MURMUR3_C5 = 0xC2B2AE35
MURMUR3_LEN = 4
ADDR_DECRYPT_ENTRY = 0x10001518

# Precomputed Murmur3-32 swap_bit table for nb_blocks == 2 when k_m == 1 (pair_val == 3, r1 = 31 - m)
_CT_BPERM_NB2_SWAP_ON_K1: tuple[int, ...] = (
    0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1,
    0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0,
)

DATA_SYMBOLS = ("rstate_sha", "jstate", "rstate_lfsr")
FUNC_SYMBOLS = (
    "SubBytes",
    "MixColumns",
    "AddRoundKey",
    "Cipher",
    "AES128_ECB_indp_crypto",
    "mbedtls_aes_crypt_ecb",
    "mbedtls_internal_aes_encrypt",
    "mbedtls_internal_aes_decrypt",
    "decrypt",
    "gen_rand_sha",
    "gen_rand_sha_nonpres",
)


# ==============================================================================
# 1. Cryptographic Helpers: Key Expansion, Secret Sharing & Reference AES-CTR
# ==============================================================================

def expand_aes256_round_keys(key_bytes: bytes) -> np.ndarray:
    """Expand a 32-byte AES-256 master key into 15 round keys of 16 bytes `(15, 16)` uint8."""
    if len(key_bytes) != 32:
        raise ValueError(f"AES-256 key must be 32 bytes, got {len(key_bytes)} bytes.")
    w = [list(key_bytes[i * 4 : (i + 1) * 4]) for i in range(8)]
    for i in range(8, 60):
        temp = list(w[i - 1])
        if i % 8 == 0:
            temp = [
                AES_SBOX[temp[1]] ^ RCON[i // 8],
                AES_SBOX[temp[2]],
                AES_SBOX[temp[3]],
                AES_SBOX[temp[0]],
            ]
        elif i % 8 == 4:
            temp = [
                AES_SBOX[temp[0]],
                AES_SBOX[temp[1]],
                AES_SBOX[temp[2]],
                AES_SBOX[temp[3]],
            ]
        w.append([w[i - 8][b] ^ temp[b] for b in range(4)])

    round_keys = np.zeros((15, 16), dtype=np.uint8)
    for r in range(15):
        rk_bytes: list[int] = []
        for word_idx in range(4):
            rk_bytes.extend(w[r * 4 + word_idx])
        round_keys[r] = np.array(rk_bytes, dtype=np.uint8)
    return round_keys


def split_key_4way(key: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a 32-byte key into four 32-byte Boolean shares `(Sa, Sb, Sc, Sd)` with `Sa ^ Sb ^ Sc ^ Sd == key`."""
    if len(key) != 32:
        raise ValueError(f"AES-256 key must be 32 bytes, got {len(key)} bytes.")
    sa = np.frombuffer(os.urandom(32), dtype=np.uint8)
    sb = np.frombuffer(os.urandom(32), dtype=np.uint8)
    sc = np.frombuffer(os.urandom(32), dtype=np.uint8)
    k_arr = np.frombuffer(key, dtype=np.uint8)
    sd = sa ^ sb ^ sc ^ k_arr
    return sa, sb, sc, sd


def _concatenate_shares(a: bytes, b: bytes, c: bytes, d: bytes) -> bytes:
    """Interleave four 32-byte shares word-by-word into the 128-byte SimpleSerial `0x02 0x00` layout."""
    na = np.frombuffer(a, dtype=np.uint8).reshape(8, 4)
    nb = np.frombuffer(b, dtype=np.uint8).reshape(8, 4)
    nc = np.frombuffer(c, dtype=np.uint8).reshape(8, 4)
    nd = np.frombuffer(d, dtype=np.uint8).reshape(8, 4)
    return np.concatenate([na, nb, nc, nd], axis=1).reshape(-1).tobytes()


def compute_aes_ctr_output(
    key: bytes,
    iv: bytes,
    iv_salt: bytes,
    plaintext: bytes,
) -> bytes:
    """Compute reference AES-256 CTR encryption for the given key, IV, salt, and plaintext."""
    if len(key) != 32:
        raise ValueError(f"AES-256 key must be 32 bytes, got {len(key)} bytes.")
    if len(iv) < 16 or len(iv_salt) < 16:
        raise ValueError("IV and IV salt must each be at least 16 bytes.")
    if len(plaintext) % 16 != 0:
        raise ValueError(f"Plaintext length must be a multiple of 16 bytes, got {len(plaintext)} bytes.")
    init_ctr = np.frombuffer(iv[:16], dtype=np.uint8) ^ np.frombuffer(iv_salt[:16], dtype=np.uint8)
    cipher = AES.new(key, mode=AES.MODE_ECB)
    out = bytearray(plaintext)
    nb_blocks = len(out) // 16
    for blk in range(nb_blocks):
        ctr_bytes = blk.to_bytes(16, "big")
        ctr_np = init_ctr ^ np.frombuffer(ctr_bytes, dtype=np.uint8)
        keystream = cipher.encrypt(ctr_np.tobytes())
        for i in range(16):
            out[blk * 16 + i] ^= keystream[i]
    return bytes(out)


def compute_intermediate_shares(
    key_a: np.ndarray,
    key_b: np.ndarray,
    key_c: np.ndarray,
    key_d: np.ndarray,
    iv: np.ndarray,
    iv_salt: np.ndarray,
) -> np.ndarray:
    """Compute Round 0 initial 4-way state shares `(4, 16)` uint8."""
    ctr_mask = iv[:16] ^ iv_salt[:16]
    st_a = key_a[:16] ^ ctr_mask
    st_b = key_b[:16].copy()
    st_c = key_c[:16].copy()
    st_d = key_d[:16].copy()
    return np.stack([st_a, st_b, st_c, st_d], axis=0)


# ==============================================================================
# 2. ELF Symbol Lookup & CoreSight TPIU / ETMv4 Stream Decoders
# ==============================================================================

def _resolve_elf_path(elf_path: str | Path | None = None) -> Path:
    """Locate the RP2350 firmware ELF file from an explicit path or standard workspace locations."""
    candidates: list[Path] = []
    if elf_path is not None:
        raw = Path(elf_path).expanduser()
        candidates.append(raw.resolve())
        if not raw.is_absolute():
            candidates.append((SCRIPT_DIR / raw).resolve())

    candidates.extend([
        (SCRIPT_DIR / "firmware" / "build" / "rp2350_hacking_challenge_2.elf").resolve(),
        (SCRIPT_DIR / "build" / "rp2350_hacking_challenge_2.elf").resolve(),
    ])

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for cand in candidates:
        if cand not in seen:
            seen.add(cand)
            unique_candidates.append(cand)
            if cand.is_file():
                return cand

    checked_str = ", ".join(str(p) for p in unique_candidates)
    raise FileNotFoundError(
        f"RP2350 firmware ELF file not found. Checked: {checked_str} "
        "(pass --elf-path to specify its location)."
    )


def compute_dwt_addresses(elf_path: Path) -> tuple[dict[str, int], dict[str, int]]:
    """Parse the RP2350 ELF binary with `lief` and retrieve DWT data and function PC addresses."""
    try:
        import lief
    except ImportError as exc:
        raise ImportError(
            "Package 'lief' is required to parse RP2350 ELF symbols in hardware mode."
        ) from exc

    binary = lief.parse(str(elf_path))
    if not isinstance(binary, lief.ELF.Binary):
        raise ValueError(f"{elf_path} is not a valid ELF binary.")

    data_addrs: dict[str, int] = {}
    for sym_name in DATA_SYMBOLS:
        sym = binary.get_symtab_symbol(sym_name) or binary.get_symbol(sym_name)
        if sym is None or sym.value == 0:
            raise KeyError(f"Data symbol '{sym_name}' not found in {elf_path}")
        data_addrs[sym_name] = int(sym.value)

    func_addrs: dict[str, int] = {}
    for sym_name in FUNC_SYMBOLS:
        sym = binary.get_symtab_symbol(sym_name) or binary.get_symbol(sym_name)
        if sym is not None and sym.value != 0:
            # Clear Thumb bit (bit 0) for DWT PC comparator matching
            func_addrs[sym_name] = int(sym.value) & ~1

    if "decrypt" not in func_addrs:
        raise KeyError(f"Function symbol 'decrypt' not found in {elf_path}")

    return data_addrs, func_addrs


def decode_etm4_stream(s2_bytes: bytes, default_cc_threshold: int = 64) -> dict[str, Any]:
    """Decode ETMv4 byte stream (Stream 2) per ARM IHI0064H ETMv4 Architecture Specification.

    Supports RP2350 Cortex-M33 ETM-Teal (`TRCIDR0.COMMOPT=1`), including A-Sync, Trace Info
    (`INFO` and `CYCT`), Cycle Count Formats 1..3, Atom Formats 1..6, and Exact/Short/Long
    address packets with the 3-entry circular address history stack `q[0..2]`.
    """
    if not s2_bytes:
        return {
            "change_counts": {},
            "events": [],
            "addresses_seen": [],
            "total_eatoms": 0,
            "total_natoms": 0,
            "total_cycles": 0,
            "cc_threshold": default_cc_threshold,
            "unknown_bytes": 0,
        }

    idx_async = s2_bytes.find(b"\x00" * 11 + b"\x80")
    idx_trace_on = s2_bytes.find(b"\x04\x83")
    if idx_async == -1 or (0 <= idx_trace_on < idx_async):
        i = idx_trace_on if idx_trace_on >= 0 else 0
    else:
        i = idx_async

    n = len(s2_bytes)
    q = [0, 0, 0]
    current_addr = 0
    cc_threshold = default_cc_threshold

    change_counts: Counter[str] = Counter()
    events: list[dict[str, Any]] = []
    total_eatoms = 0
    total_natoms = 0
    total_cycles = 0
    addresses_seen: list[int] = []
    unknown_bytes = 0

    def push_addr(a: int) -> None:
        nonlocal current_addr
        q[2] = q[1]
        q[1] = q[0]
        q[0] = a & 0xFFFFFFFF
        current_addr = q[0]

    def record_event(
        changes: list[str],
        eatoms: int = 0,
        natoms: int = 0,
        disposition: int = 0,
        cycle_count: int = 0xFFFFFFFFFFFFFFFF,
        pkt_type: str = "",
    ) -> None:
        for ch in changes:
            change_counts[ch] += 1
        events.append({
            "changes": changes,
            "addr": current_addr,
            "eatoms": eatoms,
            "natoms": natoms,
            "disposition": disposition,
            "cycleCount": cycle_count,
            "cumulative_cycles": total_cycles,
            "pkt_type": pkt_type,
        })

    while i < n:
        c = s2_bytes[i]

        # 1. A-Sync (0x00 .. 0x00 0x80) or padding 0x00
        if c == 0x00:
            while i < n and s2_bytes[i] == 0x00:
                i += 1
            if i < n and s2_bytes[i] == 0x80:
                i += 1
            continue

        # 2. Trace Info (0x01)
        if c == 0x01:
            i += 1
            plctl = 0
            shift = 0
            while i < n:
                b = s2_bytes[i]
                i += 1
                plctl |= (b & 0x7F) << shift
                shift += 7
                if not (b & 0x80):
                    break
            if plctl & 1:  # INFO
                while i < n:
                    b = s2_bytes[i]
                    i += 1
                    if not (b & 0x80):
                        break
            if plctl & 2:  # KEY
                while i < n:
                    b = s2_bytes[i]
                    i += 1
                    if not (b & 0x80):
                        break
            if plctl & 4:  # SPEC
                while i < n:
                    b = s2_bytes[i]
                    i += 1
                    if not (b & 0x80):
                        break
            if plctl & 8:  # CYCT
                cyct = 0
                shift = 0
                while i < n:
                    b = s2_bytes[i]
                    i += 1
                    cyct |= (b & 0x7F) << shift
                    shift += 7
                    if not (b & 0x80):
                        break
                cc_threshold = cyct
            record_event(["EV_CH_TRACESTART"], pkt_type="TRACE_INFO")
            continue

        # 3. Trace On (0x04), Overflow (0x05), Discard (0x03)
        if c == 0x04:
            i += 1
            record_event(["EV_CH_TRACESTART"], pkt_type="TRACE_ON")
            continue
        if c == 0x05:
            i += 1
            record_event(["EV_CH_OVERFLOW"], pkt_type="OVERFLOW")
            continue
        if c == 0x03:
            i += 1
            record_event(["EV_CH_DISCARD"], pkt_type="DISCARD")
            continue

        # 4. Exception (0x06) and Exception Return (0x07)
        if c == 0x06:
            i += 1
            if i >= n:
                break
            b0 = s2_bytes[i]
            i += 1
            if b0 & 0x80:
                if i >= n:
                    break
                i += 1
            record_event(["EV_CH_EX_ENTRY"], pkt_type="EXCEPTION")
            continue
        if c == 0x07:
            i += 1
            record_event(["EV_CH_EX_EXIT"], pkt_type="EXCEPTION_RETURN")
            continue

        # 5. Cycle Count Format 2 (0x0c, 0x0d)
        if c in (0x0C, 0x0D):
            i += 1
            if i >= n:
                break
            b = s2_bytes[i]
            i += 1
            cyc = cc_threshold + (b & 0x0F)
            total_cycles += cyc
            record_event(["EV_CH_CYCLECOUNT"], cycle_count=cyc, pkt_type="CC_FMT2")
            continue

        # 6. Cycle Count Format 1 (0x0e, 0x0f) with COMMOPT=1
        if c in (0x0E, 0x0F):
            i += 1
            if c == 0x0F:
                record_event(
                    ["EV_CH_CYCLECOUNT"],
                    cycle_count=0xFFFFFFFFFFFFFFFF,
                    pkt_type="CC_FMT1_UNKNOWN",
                )
                continue
            val = 0
            shift = 0
            while i < n:
                b = s2_bytes[i]
                i += 1
                val |= (b & 0x7F) << shift
                shift += 7
                if not (b & 0x80):
                    break
            cyc = cc_threshold + val
            total_cycles += cyc
            record_event(["EV_CH_CYCLECOUNT"], cycle_count=cyc, pkt_type="CC_FMT1")
            continue

        # 7. Cycle Count Format 3 (0x10 .. 0x1f) with COMMOPT=1
        if 0x10 <= c <= 0x1F:
            i += 1
            cyc = cc_threshold + (c & 0x03)
            total_cycles += cyc
            record_event(["EV_CH_CYCLECOUNT"], cycle_count=cyc, pkt_type="CC_FMT3")
            continue

        # 8. Function Return (0x6d) / Data Sync Markers (0x20..0x2f) / Event (0x70..0x7f)
        if c == 0x6D:
            i += 1
            record_event(["EV_CH_FNRETURN"], pkt_type="FN_RETURN")
            continue
        if 0x20 <= c <= 0x27:
            i += 1
            record_event(["EV_CH_DATASYNC"], pkt_type="DATA_SYNC")
            continue
        if 0x28 <= c <= 0x2F:
            i += 1
            record_event(["EV_CH_UDATASYNC"], pkt_type="UDATA_SYNC")
            continue
        if c == 0x70:
            i += 1
            continue
        if 0x71 <= c <= 0x7F:
            i += 1
            ev_changes = [
                ev_name
                for bit_idx, ev_name in enumerate(
                    ("EV_CH_EVENT0", "EV_CH_EVENT1", "EV_CH_EVENT2", "EV_CH_EVENT3")
                )
                if c & (1 << bit_idx)
            ]
            record_event(ev_changes, pkt_type="EVENT")
            continue

        # 9. Exact Match Address (0x90 .. 0x92)
        if c in (0x90, 0x91, 0x92):
            i += 1
            push_addr(q[c & 0x03])
            addresses_seen.append(current_addr)
            record_event(["EV_CH_ADDRESS"], pkt_type="ADDR_EXACT")
            continue

        # 10. Short Address IS0 (0x95) or IS1 (0x96)
        if c in (0x95, 0x96):
            is0 = c == 0x95
            i += 1
            if i >= n:
                break
            b0 = s2_bytes[i]
            i += 1
            shift = 2 if is0 else 1
            addr = q[0]
            mask = ~(0x7F << shift) & 0xFFFFFFFF
            addr = (addr & mask) | ((b0 & 0x7F) << shift)
            shift += 7
            if b0 & 0x80:
                if i >= n:
                    break
                b1 = s2_bytes[i]
                i += 1
                mask2 = ~(0xFF << shift) & 0xFFFFFFFF
                addr = (addr & mask2) | (b1 << shift)
            push_addr(addr)
            addresses_seen.append(current_addr)
            record_event(["EV_CH_ADDRESS"], pkt_type="ADDR_SHORT")
            continue

        # 11. Long Address 32-bit IS0 (0x9a, 0x82) or IS1 (0x9b, 0x83)
        if c in (0x9A, 0x9B, 0x82, 0x83):
            with_ctx = c in (0x82, 0x83)
            is0 = c in (0x9A, 0x82)
            i += 1
            if i + 4 > n:
                break
            b0, b1, b2, b3 = s2_bytes[i : i + 4]
            i += 4
            if is0:
                addr = (
                    ((b0 & 0x7F) << 2)
                    | ((b1 & 0x7F) << 9)
                    | (b2 << 16)
                    | (b3 << 24)
                )
            else:
                addr = (
                    ((b0 & 0x7F) << 1)
                    | (b1 << 8)
                    | (b2 << 16)
                    | (b3 << 24)
                )
            step_changes = ["EV_CH_ADDRESS"]
            if with_ctx:
                if i < n:
                    i += 1
                step_changes.append("EV_CH_CONTEXTID")
            push_addr(addr)
            addresses_seen.append(current_addr)
            record_event(step_changes, pkt_type="ADDR_LONG32")
            continue

        # 12. Atom Formats 1..6 (0xc0 .. 0xff)
        if 0xC0 <= c <= 0xFF:
            i += 1
            if c in (0xF6, 0xF7):
                e = c & 1
                na = 1 - e
                disp = c & 1
            elif 0xD8 <= c <= 0xDB:
                e = (c & 1) + ((c >> 1) & 1)
                na = 2 - e
                disp = c & 0x03
            elif 0xF8 <= c <= 0xFF:
                e = (c & 1) + ((c >> 1) & 1) + ((c >> 2) & 1)
                na = 3 - e
                disp = c & 0x07
            elif 0xDC <= c <= 0xDF:
                pat = c & 0x03
                if pat == 0:
                    na, e, disp = 1, 3, 0x0E
                elif pat == 1:
                    na, e, disp = 4, 0, 0x00
                elif pat == 2:
                    na, e, disp = 2, 2, 0x0A
                else:
                    na, e, disp = 2, 2, 0x05
            elif c in (0xD5, 0xD6, 0xD7, 0xF5):
                pat = ((c >> 3) & 4) | (c & 2) | (c & 1)
                if pat == 5:
                    na, e, disp = 1, 4, 0x1E
                elif pat == 1:
                    na, e, disp = 5, 0, 0x00
                elif pat == 2:
                    na, e, disp = 3, 2, 0x0A
                else:
                    na, e, disp = 2, 3, 0x15
            else:
                count = (c & 0x1F) + 4
                if c & 0x20:
                    e = count - 1
                    na = 1
                    disp = (1 << e) - 1
                else:
                    e = count
                    na = 0
                    disp = (1 << e) - 1

            total_eatoms += e
            total_natoms += na
            record_event(
                ["EV_CH_ENATOMS"],
                eatoms=e,
                natoms=na,
                disposition=disp,
                pkt_type="ATOM",
            )
            continue

        unknown_bytes += 1
        i += 1

    return {
        "change_counts": dict(change_counts),
        "events": events,
        "addresses_seen": addresses_seen,
        "total_eatoms": total_eatoms,
        "total_natoms": total_natoms,
        "total_cycles": total_cycles,
        "cc_threshold": cc_threshold,
        "unknown_bytes": unknown_bytes,
    }


def decode_tpiu_with_frame_phase(
    raw_bytes: bytes,
    initial_stream: int = 2,
) -> tuple[Any, int]:
    """Decode raw parallel TPIU bytes using `TpiuDecoder`, finding the even halfword frame phase `k` in `{0..14}`."""
    from chipwhisperer.capture.trace.tpiu import TpiuDecoder
    import pyorb

    long_sync_bytes = (
        TpiuDecoder.LONG_SYNC
        if isinstance(TpiuDecoder.LONG_SYNC, (bytes, bytearray))
        else b"\xff\xff\xff\x7f"
    )
    idx_long = raw_bytes.find(long_sync_bytes)
    exact_k = -1
    if idx_long >= 0:
        prefix = raw_bytes[:idx_long]
        tmp_len = 0
        last_ff = False
        for b in prefix:
            if (tmp_len & 1) == 0:
                tmp_len += 1
                last_ff = b == 0xFF
            elif last_ff and b == 0x7F:
                tmp_len -= 1
                last_ff = False
            else:
                tmp_len += 1
                last_ff = False
        exact_k = (16 - (tmp_len % 16)) % 16

    candidates = list(range(0, 16, 2))
    if exact_k in candidates:
        candidates.remove(exact_k)
        candidates.insert(0, exact_k)

    best_k = 0
    best_dec = None
    best_score = (-1, -1, -1, -1, -1)

    for k in candidates:
        dec = TpiuDecoder(assume_synced=True)
        dec.current_stream = initial_stream
        dec.fifo.extend([0] * k)
        try:
            dec.accumulate(raw_bytes)
        except AssertionError:
            continue

        if k > 0 and initial_stream in dec.packets.streams:
            del dec.packets.streams[initial_stream][:k]

        stream_ids = set(dec.packets.get_stream_ids())
        valid_ids = stream_ids.issubset({1, 2})
        s2 = dec.packets.streams.get(2) or b""
        has_trace_on = (b"\x04\x83" in s2[:64]) or (b"\x00" * 11 + b"\x80" in s2[:64])
        itm_msgs = dec.get_itm_messages()
        invalid_itm = sum(1 for m in itm_msgs if isinstance(m, (pyorb.Empty, pyorb.swMsg)))
        valid_itm = sum(
            1
            for m in itm_msgs
            if isinstance(
                m,
                (
                    pyorb.watchMsg,
                    pyorb.wptMsg,
                    pyorb.nisyncMsg,
                    pyorb.pcSampleMsg,
                    pyorb.TSMsg,
                ),
            )
        )
        score = (
            int(valid_ids),
            int(invalid_itm == 0),
            int(k == exact_k),
            int(has_trace_on),
            valid_itm,
        )
        if score > best_score:
            best_score = score
            best_k = k
            best_dec = dec

        if valid_ids and invalid_itm == 0 and (
            k == exact_k or (exact_k < 0 and has_trace_on and valid_itm > 0)
        ):
            break

    if best_dec is None:
        raise RuntimeError("Failed to decode TPIU frame stream across all candidate halfword phases.")
    return best_dec, best_k


# ==============================================================================
# 3. Protected AES-256 CTR Trace Reconstruction (`trace1` -> `trace2`)
# ==============================================================================

def ror32(val: int, shift: int) -> int:
    """Rotate a 32-bit unsigned integer right by `shift` bits (mod 32)."""
    v = val & 0xFFFFFFFF
    s = shift & 31
    if s == 0:
        return v
    return ((v >> s) | ((v << (32 - s)) & 0xFFFFFFFF)) & 0xFFFFFFFF


def rev32(val: int) -> int:
    """ARM `rev` instruction: byte-reverse a 32-bit unsigned word."""
    v = val & 0xFFFFFFFF
    return (
        ((v & 0x000000FF) << 24)
        | ((v & 0x0000FF00) << 8)
        | ((v & 0x00FF0000) >> 8)
        | ((v & 0xFF000000) >> 24)
    )


def lfsr_step(state: int) -> int:
    """Single step of the RP2350 AES assembly 32-bit Galois LFSR (`steplfsr`)."""
    msb = (state >> 31) & 1
    nxt = (state << 1) & 0xFFFFFFFF
    if msb:
        nxt ^= LFSR_FEEDBACK_CONST
    return nxt


def _msg_val(m: dict[str, Any]) -> int | None:
    """Extract 32-bit integer value from a serialized DWT message (`wptMsg` or `watchMsg`)."""
    if "data_int" in m:
        return int(m["data_int"]) & 0xFFFFFFFF
    if "pc_int" in m:
        return int(m["pc_int"]) & 0xFFFFFFFF
    if "value_int" in m:
        return int(m["value_int"]) & 0xFFFFFFFF
    return None


def _extract_tw_dict(data: dict[str, Any] | str) -> dict[str, Any]:
    """Extract the `tracewhisperer` dictionary from a dict or JSON string."""
    if isinstance(data, str):
        parsed = json.loads(data.strip())
    elif isinstance(data, dict):
        parsed = data
    else:
        raise TypeError(f"Expected TraceWhisperer dict or JSON str, got {type(data).__name__}.")

    if not isinstance(parsed, dict):
        raise ValueError("Decoded TraceWhisperer JSON payload is not a JSON object.")

    if "tracewhisperer" in parsed and isinstance(parsed["tracewhisperer"], (dict, str)):
        inner = parsed["tracewhisperer"]
        tw_dict = json.loads(inner.strip()) if isinstance(inner, str) else inner
        if isinstance(tw_dict, dict):
            if "tracewhisperer" in tw_dict and isinstance(tw_dict["tracewhisperer"], dict):
                tw_dict = tw_dict["tracewhisperer"]
            if "stream1_itm_dwt" in tw_dict:
                return tw_dict

    if "stream1_itm_dwt" in parsed:
        return parsed

    raise KeyError("TraceWhisperer payload does not contain 'stream1_itm_dwt'.")


def _extract_post_decrypt_dwt_messages(
    tw: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    """Return `(messages_after_decrypt_entry, decrypt_msg_index)` from Stream 1."""
    msgs = tw["stream1_itm_dwt"]["messages"]
    for idx, m in enumerate(msgs):
        if m.get("comparator") == 3 or m.get("symbol") == "decrypt":
            return msgs[idx:], idx

    if len(msgs) >= 10:
        for idx, m in enumerate(msgs[:8]):
            val = _msg_val(m)
            if m.get("comparator") == 0 and val is not None and val != 0:
                return msgs[idx:], idx
            if val is not None and val != 0:
                break

    return msgs, 0


def infer_nb_blocks(tw_or_capture: dict[str, Any] | str) -> int:
    """Infer `nb_blocks` from `crypto.nb_blocks` or from Stream 1 `COMP2`/`COMP1` message counts."""
    if isinstance(tw_or_capture, dict):
        crypto = tw_or_capture.get("crypto")
        if isinstance(crypto, dict) and "nb_blocks" in crypto:
            return int(crypto["nb_blocks"])
    tw = _extract_tw_dict(tw_or_capture)
    post_msgs, _ = _extract_post_decrypt_dwt_messages(tw)
    comp2_count = sum(
        1 for m in post_msgs if m.get("comparator") == 2 and _msg_val(m) is not None
    )
    if comp2_count >= 52 and (comp2_count - 1) % 51 == 0:
        return (comp2_count - 1) // 51
    comp1_count = sum(
        1 for m in post_msgs if m.get("comparator") == 1 and _msg_val(m) is not None
    )
    if comp1_count >= 157 and (comp1_count - 142) % 15 == 0:
        return (comp1_count - 142) // 15
    return 1


def reconstruct_lfsr_intermediates(
    tw_or_capture: dict[str, Any] | str,
    nb_blocks: int | None = None,
) -> dict[str, Any]:
    """Recompute all observed and hidden LFSR intermediates across a protected AES-CTR run."""
    tw = _extract_tw_dict(tw_or_capture)
    if nb_blocks is None:
        nb_blocks = infer_nb_blocks(tw_or_capture if isinstance(tw_or_capture, dict) else tw)

    post_msgs, _ = _extract_post_decrypt_dwt_messages(tw)
    comp2_vals = [
        v
        for m in post_msgs
        if m.get("comparator") == 2 and (v := _msg_val(m)) is not None
    ]

    expected_comp2_len = 1 + 51 * nb_blocks
    if len(comp2_vals) < expected_comp2_len:
        raise ValueError(
            f"Expected at least {expected_comp2_len} post-decrypt COMP2 (rstate_lfsr) messages "
            f"for nb_blocks={nb_blocks}, got {len(comp2_vals)}."
        )

    init_seed = comp2_vals[0]
    lfsr_prev = init_seed
    rk_share_c = 0

    blocks_out: list[dict[str, Any]] = []
    all_savelfsr_verified = True

    for b in range(nb_blocks):
        base = 1 + 51 * b
        lfsr_reseeded = comp2_vals[base + 0]
        sha_reseed_delta = (lfsr_reseeded - lfsr_prev) & 0xFFFFFFFF

        chaff_refresh_states = comp2_vals[base + 1 : base + 13]
        cur = lfsr_reseeded
        chaff_steps_verified = True
        for st in chaff_refresh_states:
            cur = lfsr_step(cur)
            if cur != st:
                chaff_steps_verified = False

        cur = chaff_refresh_states[-1]
        inline_61_states: list[int] = []
        for _ in range(61):
            cur = lfsr_step(cur)
            inline_61_states.append(cur)

        rk_share_c_change = inline_61_states[0]
        rk_share_c = (rk_share_c ^ rk_share_c_change) & 0xFFFFFFFF

        roundkey_share_masks: list[dict[str, Any]] = []
        for r in range(15):
            words_int = inline_61_states[1 + 4 * r : 1 + 4 * (r + 1)]
            roundkey_share_masks.append({
                "round": r,
                "mask_words_int": words_int,
                "mask_words_hex": [f"0x{w:08x}" for w in words_int],
            })

        observed_savelfsr = comp2_vals[base + 13]
        savelfsr_verified = inline_61_states[-1] == observed_savelfsr
        if not savelfsr_verified:
            all_savelfsr_verified = False

        hvperm_raw = comp2_vals[base + 14 : base + 44]
        cur = observed_savelfsr
        hvperm_steps_verified = True
        for st in hvperm_raw:
            cur = lfsr_step(cur)
            if cur != st:
                hvperm_steps_verified = False

        roundkey_hvperms: list[dict[str, Any]] = []
        for r in range(15):
            shares_info = []
            for s in range(2):
                val = hvperm_raw[2 * r + s]
                vperm = (val >> 30) & 0x3
                ror_shifts = [(val >> (8 * i)) & 0x1F for i in range(4)]
                shares_info.append({
                    "share": "A" if s == 0 else "B",
                    "lfsr_int": val,
                    "lfsr_hex": f"0x{val:08x}",
                    "vperm": vperm,
                    "ror_shifts": ror_shifts,
                })
            roundkey_hvperms.append({
                "round": r,
                "share_a": shares_info[0],
                "share_b": shares_info[1],
            })

        statevperm_raw = comp2_vals[base + 44 : base + 51]
        cur = hvperm_raw[-1]
        statevperm_steps_verified = True
        for st in statevperm_raw:
            cur = lfsr_step(cur)
            if cur != st:
                statevperm_steps_verified = False

        cum_vperm = 0
        round_statevperm: list[dict[str, Any]] = []
        for r in range(14):
            before_vperm = cum_vperm
            if r < 7:
                val = statevperm_raw[r]
                delta = val & 0x3
                cum_vperm = (cum_vperm + delta) & 0x3
                round_statevperm.append({
                    "round": r,
                    "lfsr_int": val,
                    "lfsr_hex": f"0x{val:08x}",
                    "delta_vperm": delta,
                    "statevperm_before": before_vperm,
                    "statevperm_after": cum_vperm,
                })
            else:
                round_statevperm.append({
                    "round": r,
                    "lfsr_int": None,
                    "lfsr_hex": None,
                    "delta_vperm": 0,
                    "statevperm_before": before_vperm,
                    "statevperm_after": cum_vperm,
                })

        lfsr_prev = statevperm_raw[-1]

        blocks_out.append({
            "block_index": b,
            "lfsr_reseeded_int": lfsr_reseeded,
            "lfsr_reseeded_hex": f"0x{lfsr_reseeded:08x}",
            "sha_reseed_delta_int": sha_reseed_delta,
            "sha_reseed_delta_hex": f"0x{sha_reseed_delta:08x}",
            "chaff_refresh_states_int": chaff_refresh_states,
            "chaff_refresh_states_hex": [f"0x{x:08x}" for x in chaff_refresh_states],
            "chaff_steps_verified": chaff_steps_verified,
            "rk_share_c_change_int": rk_share_c_change,
            "rk_share_c_change_hex": f"0x{rk_share_c_change:08x}",
            "rk_share_c_cumulative_int": rk_share_c,
            "rk_share_c_cumulative_hex": f"0x{rk_share_c:08x}",
            "roundkey_share_masks": roundkey_share_masks,
            "savelfsr_int": observed_savelfsr,
            "savelfsr_hex": f"0x{observed_savelfsr:08x}",
            "savelfsr_verified": savelfsr_verified,
            "hvperm_steps_verified": hvperm_steps_verified,
            "roundkey_hvperms": roundkey_hvperms,
            "statevperm_steps_verified": statevperm_steps_verified,
            "round_statevperm": round_statevperm,
            "final_lfsr_state_int": lfsr_prev,
            "final_lfsr_state_hex": f"0x{lfsr_prev:08x}",
        })

    return {
        "nb_blocks": nb_blocks,
        "lfsr_feedback_polynomial_hex": f"0x{LFSR_FEEDBACK_CONST:08x}",
        "initial_seed_int": init_seed,
        "initial_seed_hex": f"0x{init_seed:08x}",
        "total_comp2_messages": len(comp2_vals),
        "all_savelfsr_verified": all_savelfsr_verified,
        "blocks": blocks_out,
    }


def reconstruct_murmur_intermediates(
    nb_blocks: int = 1,
    bperm_rand: list[int] | None = None,
    bperm_sha_words: list[int] | None = None,
) -> dict[str, Any]:
    """Recompute all 32 rounds of the constant-time `CT_BPERM` Murmur3-32 block permutation."""
    if nb_blocks < 1 or nb_blocks > 12:
        raise ValueError(f"nb_blocks must be in 1..12, got {nb_blocks}")

    if bperm_rand is not None:
        if len(bperm_rand) != 32:
            raise ValueError(f"bperm_rand must have 32 entries, got {len(bperm_rand)}")
        k_table = [int(x) % nb_blocks for x in bperm_rand]
        exact_k_known = True
    elif bperm_sha_words is not None:
        if len(bperm_sha_words) != 32:
            raise ValueError(f"bperm_sha_words must have 32 entries, got {len(bperm_sha_words)}")
        k_table = [((int(w) & 0xFFFFFFFF) * nb_blocks) >> 32 for w in bperm_sha_words]
        exact_k_known = True
    elif nb_blocks == 1:
        k_table = [0] * 32
        exact_k_known = True
    else:
        k_table = [0] * 32
        exact_k_known = False

    def _run_murmur_core(murmur_input: int) -> dict[str, Any]:
        s1 = (murmur_input * MURMUR3_C1) & 0xFFFFFFFF
        s2 = ror32(s1, 17)
        s3 = (s2 * MURMUR3_C2) & 0xFFFFFFFF
        s4 = ror32(s3, 19)
        s5 = (s4 * 5) & 0xFFFFFFFF
        s6 = (s5 + MURMUR3_C3) & 0xFFFFFFFF
        s7 = (s6 ^ MURMUR3_LEN) & 0xFFFFFFFF
        s8 = (s7 ^ (s7 >> 16)) & 0xFFFFFFFF
        s9 = (s8 * MURMUR3_C4) & 0xFFFFFFFF
        s10 = (s9 ^ (s9 >> 13)) & 0xFFFFFFFF
        s11 = (s10 * MURMUR3_C5) & 0xFFFFFFFF
        s12 = (s11 ^ (s11 >> 16)) & 0xFFFFFFFF
        swap_bit = (s12 >> 31) & 1
        return {
            "murmur_input_int": murmur_input,
            "murmur_input_hex": f"0x{murmur_input:08x}",
            "step1_mul_c1_hex": f"0x{s1:08x}",
            "step2_ror17_hex": f"0x{s2:08x}",
            "step3_mul_c2_hex": f"0x{s3:08x}",
            "step4_ror19_hex": f"0x{s4:08x}",
            "step5_mul5_hex": f"0x{s5:08x}",
            "step6_add_c3_hex": f"0x{s6:08x}",
            "step7_xor_len_hex": f"0x{s7:08x}",
            "step8_fmix_shr16_1_hex": f"0x{s8:08x}",
            "step9_fmix_mul_c4_hex": f"0x{s9:08x}",
            "step10_fmix_shr13_hex": f"0x{s10:08x}",
            "step11_fmix_mul_c5_hex": f"0x{s11:08x}",
            "step12_hash_out_int": s12,
            "step12_hash_out_hex": f"0x{s12:08x}",
            "swap_bit": swap_bit,
        }

    blocks_trace: list[dict[str, Any]] = []
    permutation: list[int] = []

    for block_counter in range(nb_blocks):
        i_cur = block_counter
        rounds_list: list[dict[str, Any]] = []
        for m in range(32):
            r1 = 31 - m
            k_val = k_table[m]
            j_val = (k_val - i_cur) % nb_blocks
            sum_ij = i_cur + j_val
            abs_diff_ij = abs(i_cur - j_val)
            pair_val = sum_ij * nb_blocks + abs_diff_ij
            murmur_in = (pair_val ^ (r1 << 27)) & 0xFFFFFFFF
            core = _run_murmur_core(murmur_in)
            i_next = j_val if core["swap_bit"] else i_cur
            rounds_list.append({
                "step_index": m,
                "r1_counter": r1,
                "k_bperm_rand": k_val,
                "i_before": i_cur,
                "j_partner": j_val,
                "sum_ij": sum_ij,
                "abs_diff_ij": abs_diff_ij,
                "pair_val": pair_val,
                **core,
                "i_after": i_next,
            })
            i_cur = i_next

        perm_idx = i_cur
        permutation.append(perm_idx)
        rev_ctr = rev32(perm_idx)
        blocks_trace.append({
            "block_counter": block_counter,
            "permuted_block_index": perm_idx,
            "processiv_rev32_counter_int": rev_ctr,
            "processiv_rev32_counter_hex": f"0x{rev_ctr:08x}",
            "rounds": rounds_list,
        })

    canonical_pair_table: list[dict[str, Any]] | None = None
    if nb_blocks > 1 and not exact_k_known:
        canonical_pair_table = []
        for m in range(32):
            r1 = 31 - m
            pair_entries = []
            for i_val in range(nb_blocks):
                for j_val in range(i_val, nb_blocks):
                    pair_val = (i_val + j_val) * nb_blocks + (j_val - i_val)
                    murmur_in = (pair_val ^ (r1 << 27)) & 0xFFFFFFFF
                    core = _run_murmur_core(murmur_in)
                    pair_entries.append({
                        "pair": [i_val, j_val],
                        "pair_val": pair_val,
                        **core,
                    })
            canonical_pair_table.append({
                "step_index": m,
                "r1_counter": r1,
                "canonical_pairs": pair_entries,
            })

    return {
        "nb_blocks": nb_blocks,
        "exact_bperm_rand_known": exact_k_known,
        "bperm_rand": k_table,
        "constants": {
            "C1": f"0x{MURMUR3_C1:08x}",
            "C2": f"0x{MURMUR3_C2:08x}",
            "C3": f"0x{MURMUR3_C3:08x}",
            "C4": f"0x{MURMUR3_C4:08x}",
            "C5": f"0x{MURMUR3_C5:08x}",
        },
        "permuted_block_indices": permutation,
        "blocks": blocks_trace,
        "canonical_pair_table": canonical_pair_table,
    }


def _describe_gen_rand_sha_call(call_idx: int, nb_blocks: int) -> dict[str, Any]:
    """Return structural metadata (`phase`, `caller`, `block_index`, `aes_round`) for `gen_rand_sha` call `call_idx`."""
    if call_idx == 0:
        return {
            "phase": "init_key_4way",
            "caller": "randomisechaff -> makesmallperm(12)",
            "block_index": None,
            "aes_round": None,
        }
    if 1 <= call_idx <= 24:
        word_idx = (call_idx - 1) // 3
        sub_idx = (call_idx - 1) % 3
        return {
            "phase": "init_key_4way",
            "caller": f"key_share_refresh(word={word_idx}, draw={sub_idx})",
            "block_index": None,
            "aes_round": None,
        }
    if call_idx in (25, 26):
        return {
            "phase": "init_key_4way",
            "caller": f"storeroundkey(round_key=0, share={'A' if call_idx == 25 else 'B'} -> makesmallperm(4))",
            "block_index": None,
            "aes_round": 0,
        }
    if call_idx in (27, 28):
        return {
            "phase": "init_key_4way",
            "caller": f"storeroundkey(round_key=1, share={'A' if call_idx == 27 else 'B'} -> makesmallperm(4))",
            "block_index": None,
            "aes_round": 1,
        }
    if 29 <= call_idx <= 106:
        grp = (call_idx - 29) // 6
        off = (call_idx - 29) % 6
        rk_num = grp + 2
        sub_names = [
            "init_key_sbox -> randomisechaff -> makesmallperm(12)",
            "init_key_sbox -> makesmallperm(4)",
            "init_key_sbox -> share_refresh(draw=0)",
            "init_key_sbox -> share_refresh(draw=1)",
            f"storeroundkey(round_key={rk_num}, share=A -> makesmallperm(4))",
            f"storeroundkey(round_key={rk_num}, share=B -> makesmallperm(4))",
        ]
        return {
            "phase": "init_key_4way_expansion",
            "caller": sub_names[off],
            "block_index": None,
            "aes_round": rk_num,
        }
    if 107 <= call_idx <= 138:
        m = call_idx - 107
        return {
            "phase": "ctr_crypt_s_bperm_init",
            "caller": f"bperm_rand[{m}] (r1={31 - m})",
            "block_index": None,
            "aes_round": None,
        }
    if call_idx == 139:
        return {
            "phase": "ctr_crypt_s_init",
            "caller": "randomisechaff -> makesmallperm(12)",
            "block_index": None,
            "aes_round": None,
        }

    final_call_idx = 140 + 15 * nb_blocks
    if 140 <= call_idx < final_call_idx:
        rel = call_idx - 140
        b = rel // 15
        sub = rel % 15
        if sub == 0:
            return {
                "phase": f"block_{b}_init",
                "caller": "refreshchaff_and_lfsr -> makesmallperm(12)",
                "block_index": b,
                "aes_round": None,
            }
        r = sub - 1
        return {
            "phase": f"block_{b}_round_{r}",
            "caller": f"map_sbox_s(round={r}) -> makeperm16",
            "block_index": b,
            "aes_round": r,
        }

    return {
        "phase": "decrypt_cleanup",
        "caller": "randomisechaff -> makesmallperm(12)",
        "block_index": None,
        "aes_round": None,
    }


def locate_jitter_in_adc_trace(
    tw_or_capture: dict[str, Any] | str,
    nb_blocks: int | None = None,
    adc_mul: int = 1,
    adc_offset: int = 0,
) -> dict[str, Any]:
    """Locate every `gen_rand_sha` jitter event and every AES round/operation in the ADC trace."""
    tw = _extract_tw_dict(tw_or_capture)
    if nb_blocks is None:
        nb_blocks = infer_nb_blocks(tw_or_capture)

    post_msgs, _ = _extract_post_decrypt_dwt_messages(tw)

    comp1_vals = [
        v
        for m in post_msgs
        if m.get("comparator") == 1 and (v := _msg_val(m)) is not None
    ]
    expected_calls = 141 + 15 * nb_blocks
    if len(comp1_vals) < expected_calls + 1:
        raise ValueError(
            f"Expected {expected_calls + 1} COMP1 (jstate) messages for nb_blocks={nb_blocks}, "
            f"got {len(comp1_vals)}."
        )

    all_comp0_vals = [
        v
        for m in post_msgs
        if m.get("comparator") == 0 and (v := _msg_val(m)) is not None
    ]
    total_sha_hw_refill_count = 0
    idx_c0 = 0
    while idx_c0 < len(all_comp0_vals):
        if all_comp0_vals[idx_c0] == 0x21:
            total_sha_hw_refill_count += 1
            idx_c0 += 2
        else:
            idx_c0 += 1

    s2_dict = tw.get("stream2_etmv4", {})
    existing_events = s2_dict.get("events")
    if (
        isinstance(existing_events, list)
        and existing_events
        and "pkt_type" in existing_events[0]
        and ("addr" in existing_events[0] or "addr_int" in existing_events[0])
    ):
        etm_events = existing_events
        etm_total_cycles = int(s2_dict.get("total_cycles", 0))
    else:
        s2_raw_hex = s2_dict.get("raw_hex", "")
        s2_bytes = bytes.fromhex(s2_raw_hex) if s2_raw_hex else b""
        etm_decoded = decode_etm4_stream(s2_bytes)
        etm_events = etm_decoded["events"]
        etm_total_cycles = int(etm_decoded["total_cycles"])

    cycles_by_addr: dict[int, list[int]] = {}
    for ev in etm_events:
        pkt_type = ev.get("pkt_type") or ""
        if pkt_type.startswith("ADDR"):
            if "EV_CH_CONTEXTID" in (ev.get("changes") or ()):
                continue
            addr_val = int(ev["addr"] if "addr" in ev else ev["addr_int"])
            cyc_val = int(ev["cumulative_cycles"])
            cycles_by_addr.setdefault(addr_val, []).append(cyc_val)

    c_decrypt = 0
    for cyc in cycles_by_addr.get(ADDR_DECRYPT_ENTRY, ()):
        if cyc >= 0:
            c_decrypt = cyc
            break

    def _cycle_to_adc_sample(cycle: int) -> int:
        return int(cycle * adc_mul - adc_offset)

    draw_n = 0
    call_k_cursor = 0
    cum_extra_cycles = 0
    standalone_refill_cycles: list[int] = []
    exact_call_records: list[dict[str, Any]] = []

    def _step_nonpres_at(can_ret_cycle: int) -> bool:
        nonlocal draw_n, cum_extra_cycles
        is_refill = (draw_n % 8) == 0
        draw_n += 1
        if is_refill:
            cum_extra_cycles += 188
            act_ret_cycle = c_decrypt + can_ret_cycle + cum_extra_cycles
            standalone_refill_cycles.append(act_ret_cycle)
        return is_refill

    def _step_sha_call_at(can_caller_ret_cycle: int) -> dict[str, Any]:
        nonlocal draw_n, call_k_cursor, cum_extra_cycles
        k_idx = call_k_cursor
        call_k_cursor += 1
        j_bef = comp1_vals[k_idx]
        j_aft = comp1_vals[k_idx + 1]
        j_bits = j_bef & 0x3
        u_div = 1 << (4 * j_bits)
        u_extra = j_bits
        j_refilled = (j_bef >> 2) == 0

        can_entry_cycle = can_caller_ret_cycle - 48
        act_entry_cycle = c_decrypt + can_entry_cycle + cum_extra_cycles
        act_udiv_cycle = act_entry_cycle + 17 + u_extra

        excise_ivs: list[tuple[int, int]] = []
        if u_extra > 0:
            excise_ivs.append((act_entry_cycle + 17, act_entry_cycle + 17 + u_extra))

        j_sha_refill = False
        j_extra = 0
        if j_refilled:
            j_sha_refill = (draw_n % 8) == 0
            draw_n += 1
            j_extra = 17
            j_branch_total = 17 + (188 if j_sha_refill else 0)
            j_start = act_entry_cycle + 20 + u_extra
            excise_ivs.append((j_start, j_start + j_branch_total))

        c_sha_refill = (draw_n % 8) == 0
        draw_n += 1
        sha_extra = 188 * (int(j_sha_refill) + int(c_sha_refill))
        tot_call_extra = u_extra + j_extra + sha_extra

        cum_extra_cycles += tot_call_extra
        act_caller_ret_cycle = c_decrypt + can_caller_ret_cycle + cum_extra_cycles
        act_15a6_cycle = act_caller_ret_cycle - 8
        if c_sha_refill:
            excise_ivs.append((act_15a6_cycle - 188, act_15a6_cycle))

        rec = {
            "call_index": k_idx,
            "j_before": j_bef,
            "j_after": j_aft,
            "jitter_bits": j_bits,
            "udiv_dividend": u_div,
            "udiv_extra": u_extra,
            "jstate_refilled": j_refilled,
            "jstate_extra": j_extra,
            "jstate_sha_hw_refill": j_sha_refill,
            "caller_sha_hw_refill": c_sha_refill,
            "sha_hw_extra": sha_extra,
            "total_call_extra": tot_call_extra,
            "can_entry": can_entry_cycle,
            "can_caller_ret": can_caller_ret_cycle,
            "act_entry": act_entry_cycle,
            "act_udiv": act_udiv_cycle,
            "act_return": act_15a6_cycle,
            "act_caller_ret": act_caller_ret_cycle,
            "excise_intervals": excise_ivs,
        }
        exact_call_records.append(rec)
        return rec

    # Phase 0: init_rstate (canonical [0, 12634])
    c_init_rstate = c_decrypt
    c_lut_mask = c_decrypt + 12634

    # Phase 1: lut_mask_init + remap
    for j_lut in range(64):
        _step_nonpres_at(12708 + 28 * j_lut)
    c_remap = c_decrypt + 14477 + cum_extra_cycles
    _step_nonpres_at(14500)
    _step_nonpres_at(16461)
    c_init_key = c_decrypt + 18426 + cum_extra_cycles

    # Phase 2: init_key_4way
    _step_sha_call_at(18522)
    for j_rc in range(12):
        _step_nonpres_at(18652 + 29 * j_rc)
    for g_ik in range(8):
        for s_ik in range(3):
            _step_sha_call_at(19037 + 161 * g_ik + 50 * s_ik)
    _step_nonpres_at(20290)
    for can_ret_hv in (20348, 20460, 20587, 20714):
        _step_sha_call_at(can_ret_hv)
    for m_ik in range(13):
        base_m_can = 1440 * m_ik
        for d_sb in range(4):
            _step_nonpres_at(20833 + base_m_can + 22 * d_sb)
        _step_sha_call_at(20976 + base_m_can)
        for j_rc in range(12):
            _step_nonpres_at(21097 + base_m_can + 29 * j_rc)
        _step_sha_call_at(21491 + base_m_can)
        _step_sha_call_at(21594 + base_m_can)
        _step_sha_call_at(21643 + base_m_can)
        _step_sha_call_at(22003 + base_m_can)
        _step_sha_call_at(22158 + base_m_can - (3 if m_ik % 2 == 1 else 0))
    c_ctr_crypt = c_decrypt + 39541 + cum_extra_cycles

    # Phase 3: ctr_crypt_s prologue
    for m_bp in range(32):
        _step_sha_call_at(39619 + 54 * m_bp)
    _step_sha_call_at(41363)
    for j_rc in range(12):
        _step_nonpres_at(41480 + 29 * j_rc)
    c_ctr_iv_setup = c_decrypt + 41813 + cum_extra_cycles
    for d_iv in range(4):
        _step_nonpres_at(41854 + 25 * d_iv)
    c_bperm_done = c_decrypt + 41943 + cum_extra_cycles

    # Phase 4: Per-block loop
    can_cursor = 41943
    exact_blocks_meta: list[dict[str, Any]] = []
    for b in range(nb_blocks):
        b_can_start = can_cursor
        b_act_start = c_decrypt + b_can_start + cum_extra_cycles

        _step_nonpres_at(b_can_start + 23)
        _step_sha_call_at(b_can_start + 96)
        can_cursor = b_can_start + 504
        c_ref_done_act = c_decrypt + can_cursor + cum_extra_cycles

        if b % 4 == 0:
            _step_nonpres_at(b_can_start + 538)
            _step_nonpres_at(b_can_start + 2499)
            can_cursor += 3963
        else:
            can_cursor += 4
        c_rk_shares_act = c_decrypt + can_cursor + cum_extra_cycles

        can_cursor += 1336
        c_hvperms_act = c_decrypt + can_cursor + cum_extra_cycles

        c_ct_bperm_act = c_decrypt + can_cursor + 1449 + cum_extra_cycles
        can_cursor += 2810
        c_proc_iv_act = c_decrypt + can_cursor + cum_extra_cycles

        piv_can_start = can_cursor
        for d_piv in range(6):
            _step_nonpres_at(piv_can_start + 33 + 25 * d_piv)
        c_proc_iv_20b4_act = c_decrypt + piv_can_start + 215 + cum_extra_cycles
        can_cursor = piv_can_start + 240

        exact_rounds_meta: list[dict[str, Any]] = []
        for r in range(14):
            r_can_start = can_cursor
            r_act_start = c_decrypt + r_can_start + cum_extra_cycles
            ak_len = 96 if r == 0 else 95
            c_20ce_can = r_can_start + ak_len
            c_20ce_act = c_decrypt + c_20ce_can + cum_extra_cycles

            mp1_refill = _step_nonpres_at(c_20ce_can + 50)
            call_rec_r = _step_sha_call_at(c_20ce_can + 152)
            c_post_perm_act = c_decrypt + c_20ce_can + 189 + cum_extra_cycles
            sb_refill = _step_nonpres_at(c_20ce_can + 210)
            c_20f8_can = c_20ce_can + 210
            c_20f8_act = c_decrypt + c_20f8_can + cum_extra_cycles

            sbox_mc_len = 617 if r < 7 else (539 if r < 13 else 433)
            can_cursor = c_20f8_can + sbox_mc_len
            r_act_end = c_decrypt + can_cursor + cum_extra_cycles

            exact_rounds_meta.append({
                "round": r,
                "ak_len": ak_len,
                "sbox_mc_len": sbox_mc_len,
                "base_round_dur": ak_len + 210 + sbox_mc_len,
                "c_r_start": r_act_start,
                "c_sbox_start": c_20ce_act,
                "c_post_perm": c_post_perm_act,
                "c_lookup_start": c_20f8_act,
                "c_r_end": r_act_end,
                "mp1_refill": mp1_refill,
                "sb_refill": sb_refill,
                "call_rec": call_rec_r,
            })

        c_r14_start_act = c_decrypt + can_cursor + cum_extra_cycles
        can_cursor += 96
        c_r14_end_act = c_decrypt + can_cursor + cum_extra_cycles
        can_cursor += 94
        c_dec_start_act = c_decrypt + can_cursor + cum_extra_cycles
        dec_start_len = 44 if b < (nb_blocks - 1) else 70
        can_cursor += dec_start_len
        b_act_end = c_decrypt + can_cursor + cum_extra_cycles

        exact_blocks_meta.append({
            "block_index": b,
            "b_act_start": b_act_start,
            "c_ref_done_act": c_ref_done_act,
            "c_rk_shares_act": c_rk_shares_act,
            "c_hvperms_act": c_hvperms_act,
            "c_ct_bperm_act": c_ct_bperm_act,
            "c_proc_iv_act": c_proc_iv_act,
            "c_proc_iv_20b4_act": c_proc_iv_20b4_act,
            "c_r14_start_act": c_r14_start_act,
            "c_r14_end_act": c_r14_end_act,
            "c_dec_start_act": c_dec_start_act,
            "dec_start_len": dec_start_len,
            "b_act_end": b_act_end,
            "rounds": exact_rounds_meta,
        })

    # Phase 5: decrypt_cleanup
    c_cleanup_start_can = can_cursor
    c_cleanup_start_act = c_decrypt + c_cleanup_start_can + cum_extra_cycles
    _step_sha_call_at(c_cleanup_start_can + 81)
    for j_rc in range(12):
        _step_nonpres_at(c_cleanup_start_can + 198 + 29 * j_rc)
    can_cursor = c_cleanup_start_can + 555
    c_decrypt_end_act = c_decrypt + can_cursor + cum_extra_cycles

    jitter_calls: list[dict[str, Any]] = []
    cumulative_udiv_jitter = 0
    cumulative_jstate_refill_cycles = 0
    cumulative_sha_hw_refill_cycles = 0

    for k in range(expected_calls):
        rec = exact_call_records[k]
        udiv_extra = int(rec["udiv_extra"])
        jstate_extra = int(rec["jstate_extra"])
        sha_hw_extra = int(rec["sha_hw_extra"])
        total_call_extra = int(rec["total_call_extra"])

        cumulative_udiv_jitter += udiv_extra
        cumulative_jstate_refill_cycles += jstate_extra
        cumulative_sha_hw_refill_cycles += sha_hw_extra
        cumulative_total_extra = (
            cumulative_udiv_jitter
            + cumulative_jstate_refill_cycles
            + cumulative_sha_hw_refill_cycles
        )

        cyc_entry = int(rec["act_entry"])
        cyc_udiv = int(rec["act_udiv"])
        cyc_return = int(rec["act_return"])

        meta = _describe_gen_rand_sha_call(k, nb_blocks)
        jitter_calls.append({
            "call_index": k,
            **meta,
            "jstate_before_int": rec["j_before"],
            "jstate_before_hex": f"0x{rec['j_before']:08x}",
            "jstate_after_int": rec["j_after"],
            "jstate_after_hex": f"0x{rec['j_after']:08x}",
            "jitter_bits": rec["jitter_bits"],
            "udiv_dividend_hex": f"0x{rec['udiv_dividend']:x}",
            "udiv_extra_cycles": udiv_extra,
            "jstate_refilled": rec["jstate_refilled"],
            "jstate_refill_extra_cycles": jstate_extra,
            "jstate_sha_hw_refill": rec["jstate_sha_hw_refill"],
            "caller_sha_hw_refill": rec["caller_sha_hw_refill"],
            "sha_hw_refill_extra_cycles": sha_hw_extra,
            "total_call_extra_cycles": total_call_extra,
            "cumulative_udiv_jitter_cycles": cumulative_udiv_jitter,
            "cumulative_total_extra_cycles": cumulative_total_extra,
            "cycle_entry": cyc_entry,
            "cycle_at_udiv": cyc_udiv,
            "cycle_at_return": cyc_return,
            "cycle_at_caller_return": int(rec["act_caller_ret"]),
            "excise_intervals": rec["excise_intervals"],
            "adc_sample_entry": _cycle_to_adc_sample(cyc_entry),
            "adc_sample_at_udiv": _cycle_to_adc_sample(cyc_udiv),
            "adc_sample_at_return": _cycle_to_adc_sample(cyc_return),
        })

    phases_summary = {
        "decrypt_entry": {"cycle": c_decrypt, "adc_sample": _cycle_to_adc_sample(c_decrypt)},
        "init_rstate": {"cycle": c_init_rstate, "adc_sample": _cycle_to_adc_sample(c_init_rstate)},
        "lut_mask_init": {"cycle": c_lut_mask, "adc_sample": _cycle_to_adc_sample(c_lut_mask)},
        "remap": {"cycle": c_remap, "adc_sample": _cycle_to_adc_sample(c_remap)},
        "init_key_4way": {"cycle": c_init_key, "adc_sample": _cycle_to_adc_sample(c_init_key)},
        "ctr_crypt_s_entry": {"cycle": c_ctr_crypt, "adc_sample": _cycle_to_adc_sample(c_ctr_crypt)},
        "ctr_crypt_s_iv_setup": {"cycle": c_ctr_iv_setup, "adc_sample": _cycle_to_adc_sample(c_ctr_iv_setup)},
        "ctr_crypt_s_mainloop": {"cycle": c_bperm_done, "adc_sample": _cycle_to_adc_sample(c_bperm_done)},
        "decrypt_cleanup": {"cycle": c_cleanup_start_act, "adc_sample": _cycle_to_adc_sample(c_cleanup_start_act)},
        "decrypt_end": {"cycle": c_decrypt_end_act, "adc_sample": _cycle_to_adc_sample(c_decrypt_end_act)},
    }

    blocks_timing: list[dict[str, Any]] = []
    for b in range(nb_blocks):
        bm = exact_blocks_meta[b]
        rounds_info: list[dict[str, Any]] = []
        for r in range(14):
            rm = bm["rounds"][r]
            call_k = 140 + 15 * b + 1 + r
            call_info = jitter_calls[call_k]

            c_start = int(rm["c_r_start"])
            c_sbox_start = int(rm["c_sbox_start"])
            c_post_perm = int(rm["c_post_perm"])
            c_sbox_lookup_exact = int(rm["c_lookup_start"])
            c_end = int(rm["c_r_end"])

            refill_locs: list[str] = []
            if rm["mp1_refill"]:
                refill_locs.append("makeperm16_draw1")
            if call_info["jstate_sha_hw_refill"]:
                refill_locs.append("makeperm16_jstate_draw")
            if call_info["caller_sha_hw_refill"]:
                refill_locs.append("makeperm16_draw2")
            if rm["sb_refill"]:
                refill_locs.append("sbox_mask_draw")

            round_sha_extra = 188 * len(refill_locs)
            round_total_extra = (
                call_info["udiv_extra_cycles"]
                + call_info["jstate_refill_extra_cycles"]
                + round_sha_extra
            )

            rounds_info.append({
                "round": r,
                "gen_rand_sha_call_index": call_k,
                "base_round_duration_cycles": int(rm["base_round_dur"]),
                "jitter_bits": call_info["jitter_bits"],
                "udiv_extra_cycles": call_info["udiv_extra_cycles"],
                "jstate_refilled": call_info["jstate_refilled"],
                "jstate_refill_extra_cycles": call_info["jstate_refill_extra_cycles"],
                "sha_hw_refill_triggered": bool(refill_locs),
                "sha_hw_refill_location": ",".join(refill_locs) if refill_locs else None,
                "sha_hw_refill_extra_cycles": round_sha_extra,
                "total_round_extra_cycles": round_total_extra,
                "cumulative_udiv_jitter_cycles": call_info["cumulative_udiv_jitter_cycles"],
                "cumulative_total_extra_cycles": call_info["cumulative_total_extra_cycles"],
                "cycle_round_start": c_start,
                "cycle_map_sbox_start": c_sbox_start,
                "cycle_makeperm16_udiv": call_info["cycle_at_udiv"],
                "cycle_makeperm16_return": call_info["cycle_at_return"],
                "cycle_post_makeperm16": c_post_perm,
                "cycle_sbox_lookup_start": c_sbox_lookup_exact,
                "cycle_round_end": c_end,
                "round_duration_cycles": c_end - c_start,
                "post_perm_sbox_mixcols_duration_cycles": c_end - c_sbox_lookup_exact,
                "adc_sample_round_start": _cycle_to_adc_sample(c_start),
                "adc_sample_map_sbox_start": _cycle_to_adc_sample(c_sbox_start),
                "adc_sample_makeperm16_udiv": call_info["adc_sample_at_udiv"],
                "adc_sample_sbox_lookup_start": _cycle_to_adc_sample(c_sbox_lookup_exact),
                "adc_sample_round_end": _cycle_to_adc_sample(c_end),
            })

        block_init_call = jitter_calls[140 + 15 * b]
        blocks_timing.append({
            "block_index": b,
            "block_init_gen_rand_sha_call_index": 140 + 15 * b,
            "block_init_jitter_bits": block_init_call["jitter_bits"],
            "block_init_extra_cycles": block_init_call["total_call_extra_cycles"],
            "cycle_refreshchaff_start": int(bm["b_act_start"]),
            "cycle_refreshchaff_end": int(bm["c_ref_done_act"]),
            "cycle_ref_roundkey_shares_start": int(bm["c_rk_shares_act"]),
            "cycle_ref_roundkey_hvperms_start": int(bm["c_hvperms_act"]),
            "cycle_ct_bperm_start": int(bm["c_ct_bperm_act"]),
            "cycle_process_iv_start": int(bm["c_proc_iv_act"]),
            "cycle_process_iv_addstatevperm_ret": int(bm["c_proc_iv_20b4_act"]),
            "cycle_round14_addrkey_start": int(bm["c_r14_start_act"]),
            "cycle_round14_addrkey_end": int(bm["c_r14_end_act"]),
            "cycle_decryption_start": int(bm["c_dec_start_act"]),
            "decryption_start_canonical_cycles": int(bm["dec_start_len"]),
            "cycle_block_done": int(bm["b_act_end"]),
            "adc_sample_refreshchaff_start": _cycle_to_adc_sample(int(bm["b_act_start"])),
            "adc_sample_process_iv_start": _cycle_to_adc_sample(int(bm["c_proc_iv_act"])),
            "adc_sample_decryption_start": _cycle_to_adc_sample(int(bm["c_dec_start_act"])),
            "adc_sample_block_done": _cycle_to_adc_sample(int(bm["b_act_end"])),
            "rounds": rounds_info,
        })

    total_all_sha_hw_refill_cycles = total_sha_hw_refill_count * 188
    return {
        "nb_blocks": nb_blocks,
        "adc_mul": adc_mul,
        "adc_offset": adc_offset,
        "total_etm_cycles": etm_total_cycles,
        "exact_decrypt_end_cycle": c_decrypt_end_act,
        "canonical_decrypt_cycles": can_cursor,
        "total_gen_rand_sha_calls": expected_calls,
        "total_udiv_jitter_cycles": cumulative_udiv_jitter,
        "total_jstate_refill_cycles": cumulative_jstate_refill_cycles,
        "total_sha_hw_refill_count": total_sha_hw_refill_count,
        "total_sha_hw_refill_cycles_in_gen_rand_sha": cumulative_sha_hw_refill_cycles,
        "total_sha_hw_refill_cycles": total_all_sha_hw_refill_cycles,
        "total_all_extra_cycles": (
            cumulative_udiv_jitter
            + cumulative_jstate_refill_cycles
            + total_all_sha_hw_refill_cycles
        ),
        "standalone_refill_cycles": standalone_refill_cycles,
        "phases": phases_summary,
        "blocks": blocks_timing,
        "gen_rand_sha_calls": jitter_calls,
    }


def _fit_canonical_segment(
    seg: np.ndarray,
    expected_samples: int,
    fill_val: int = 0,
) -> np.ndarray:
    """Pad or trim a 1D int16 ADC segment to an exact canonical sample length."""
    if expected_samples <= 0:
        return np.empty(0, dtype=np.int16)
    n = int(seg.shape[0])
    if n == expected_samples:
        return seg.astype(np.int16, copy=True)
    if n > expected_samples:
        return seg[:expected_samples].astype(np.int16, copy=True)
    if n == 0:
        return np.full(expected_samples, fill_val, dtype=np.int16)
    pad = np.full(expected_samples - n, int(seg[-1]), dtype=np.int16)
    return np.concatenate([seg.astype(np.int16, copy=False), pad])


def _remove_jitter_calls_in_window(
    trace1: np.ndarray,
    start_cycle: int,
    end_cycle: int,
    calls_in_window: list[dict[str, Any]],
    expected_canonical_cycles: int,
    adc_mul: int = 1,
    adc_offset: int = 0,
    extra_refill_cycles_list: list[int] | None = None,
) -> np.ndarray:
    """Remove `gen_rand_sha` jitter and SHA-256 refill windows inside `[start_cycle, end_cycle]`."""
    n_samples = int(trace1.shape[0])
    fill_val = int(trace1[-1]) if n_samples > 0 else 0
    expected_samples = expected_canonical_cycles * adc_mul

    def _c2s(cyc: int) -> int:
        return max(0, min(n_samples, cyc * adc_mul - adc_offset))

    intervals: list[tuple[int, int]] = []
    for call in calls_in_window:
        excise_ivs = call.get("excise_intervals")
        if excise_ivs:
            for iv_lo, iv_hi in excise_ivs:
                c_jit_start = max(start_cycle, int(iv_lo))
                c_jit_end = min(end_cycle, int(iv_hi))
                if c_jit_end > c_jit_start:
                    intervals.append((c_jit_start, c_jit_end))
            continue

        extra = int(call.get("total_call_extra_cycles", 0))
        if extra <= 0:
            continue
        c_ret = int(call["cycle_at_return"])
        c_udiv = int(call.get("cycle_at_udiv", c_ret - extra))
        c_jit_start = max(start_cycle, min(c_udiv + 4, c_ret - extra))
        c_jit_end = min(end_cycle, c_jit_start + extra)
        if c_jit_end > c_jit_start:
            intervals.append((c_jit_start, c_jit_end))

    if extra_refill_cycles_list:
        for c_ref in extra_refill_cycles_list:
            c_jit_start = max(start_cycle, int(c_ref) - 188)
            c_jit_end = min(end_cycle, int(c_ref))
            if c_jit_end > c_jit_start:
                intervals.append((c_jit_start, c_jit_end))

    intervals.sort(key=lambda iv: (iv[0], iv[1]))

    pieces: list[np.ndarray] = []
    cursor = start_cycle
    for iv_start, iv_end in intervals:
        s_cut = max(cursor, iv_start)
        e_cut = max(s_cut, iv_end)
        if s_cut > cursor:
            s0, s1 = _c2s(cursor), _c2s(s_cut)
            if s1 > s0:
                pieces.append(trace1[s0:s1])
        cursor = max(cursor, e_cut)

    if end_cycle > cursor:
        s0, s1 = _c2s(cursor), _c2s(end_cycle)
        if s1 > s0:
            pieces.append(trace1[s0:s1])

    merged = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int16)
    return _fit_canonical_segment(merged, expected_samples, fill_val=fill_val)


def _unpermute_4word_slots(
    seg: np.ndarray,
    offset_cycles: int,
    word_cycles: int,
    vperm: int,
    adc_mul: int = 1,
    slot_order: tuple[int, int, int, int] = (0, 1, 2, 3),
) -> None:
    """In-place un-permute 4 contiguous word slots at `offset_cycles` by circular shift `vperm` (`0..3`)."""
    vp = int(vperm) & 0x3
    if vp == 0:
        return
    base = offset_cycles * adc_mul
    w_len = word_cycles * adc_mul
    total_end = base + 4 * w_len
    if base < 0 or total_end > int(seg.shape[0]):
        return
    orig = seg[base:total_end].copy()
    for w in range(4):
        phys_word = (w + vp) & 0x3
        src_slot = slot_order.index(phys_word)
        dst_slot = slot_order.index(w)
        seg[base + dst_slot * w_len : base + (dst_slot + 1) * w_len] = (
            orig[src_slot * w_len : (src_slot + 1) * w_len]
        )


def _infer_ct_bperm_details_from_trace1(
    arr1: np.ndarray,
    blocks_timing: list[dict[str, Any]],
    adc_mul: int = 1,
    adc_offset: int = 0,
) -> dict[str, Any] | None:
    """Reconstruct the 32-step `CT_BPERM` trajectory (`bperm_rand`, `i_trajectory`, `permuted_block_indices`)
    from `trace1` for `nb_blocks == 2` using a 2-state Viterbi decoder.
    """
    if len(blocks_timing) != 2:
        return None
    n_samples = int(arr1.shape[0])

    def _c2s(cyc: int) -> int:
        return max(0, min(n_samples, cyc * adc_mul - adc_offset))

    c_bperm0 = int(blocks_timing[0]["cycle_ct_bperm_start"])
    c_bperm1 = int(blocks_timing[1]["cycle_ct_bperm_start"])
    loop_len = 32 * 42 * adc_mul
    s0 = _c2s(c_bperm0 + 15)
    s1 = _c2s(c_bperm1 + 15)
    if s0 + loop_len > n_samples or s1 + loop_len > n_samples:
        return None

    d = arr1[s1 : s1 + loop_len].astype(np.float64) - arr1[s0 : s0 + loop_len].astype(np.float64)
    period = 4 * adc_mul
    for r in range(period):
        d[r::period] -= float(np.mean(d[r::period]))

    w = d.reshape(32, 42, adc_mul).mean(axis=2) if adc_mul > 1 else d.reshape(32, 42)
    s_k0 = -np.sum(w[:, 3:8], axis=1)
    s_k1_a = np.sum(w[:, 9:13], axis=1)
    s_k1_b = -np.sum(w[:, 36:40], axis=1)

    cost0 = 0.0
    cost1 = 1e18
    back_i = np.zeros((32, 2), dtype=np.int8)
    back_k = np.zeros((32, 2), dtype=np.int8)

    for m in range(32):
        sk0 = float(s_k0[m])
        sk1a = float(s_k1_a[m])
        sk1b = float(s_k1_b[m])
        sw = _CT_BPERM_NB2_SWAP_ON_K1[m]

        e_i0_k0 = (sk0 - 410.0) ** 2 + sk1a**2 + sk1b**2
        e_i0_k1 = sk0**2 + (sk1a - 650.0) ** 2 + (sk1b - 350.0) ** 2
        e_i1_k0 = (sk0 + 410.0) ** 2 + sk1a**2 + sk1b**2
        e_i1_k1 = sk0**2 + (sk1a + 650.0) ** 2 + (sk1b + 350.0) ** 2

        if sw == 0:
            if e_i0_k0 <= e_i0_k1:
                new_cost0 = cost0 + e_i0_k0
                back_i[m, 0] = 0
                back_k[m, 0] = 0
            else:
                new_cost0 = cost0 + e_i0_k1
                back_i[m, 0] = 0
                back_k[m, 0] = 1

            if e_i1_k0 <= e_i1_k1:
                new_cost1 = cost1 + e_i1_k0
                back_i[m, 1] = 1
                back_k[m, 1] = 0
            else:
                new_cost1 = cost1 + e_i1_k1
                back_i[m, 1] = 1
                back_k[m, 1] = 1
        else:
            cand0_stay = cost0 + e_i0_k0
            cand0_swap = cost1 + e_i1_k1
            if cand0_stay <= cand0_swap:
                new_cost0 = cand0_stay
                back_i[m, 0] = 0
                back_k[m, 0] = 0
            else:
                new_cost0 = cand0_swap
                back_i[m, 0] = 1
                back_k[m, 0] = 1

            cand1_stay = cost1 + e_i1_k0
            cand1_swap = cost0 + e_i0_k1
            if cand1_stay <= cand1_swap:
                new_cost1 = cand1_stay
                back_i[m, 1] = 1
                back_k[m, 1] = 0
            else:
                new_cost1 = cand1_swap
                back_i[m, 1] = 0
                back_k[m, 1] = 1

        cost0, cost1 = new_cost0, new_cost1

    final_i = 0 if cost0 <= cost1 else 1
    k_seq = [0] * 32
    i_seq = [0] * 33
    i_seq[32] = final_i
    cur = final_i
    for m in range(31, -1, -1):
        k_seq[m] = int(back_k[m, cur])
        cur = int(back_i[m, cur])
        i_seq[m] = cur

    return {
        "permuted_block_indices": [final_i, 1 - final_i],
        "bperm_rand": k_seq,
        "i_trajectory": i_seq,
    }


def analyze_protected_aes_capture(
    tw_or_capture: dict[str, Any] | str,
    nb_blocks: int | None = None,
    adc_mul: int = 1,
    adc_offset: int = 0,
    adc_values: np.ndarray | None = None,
) -> dict[str, Any]:
    """Reconstruct LFSR intermediates, Murmur3-32 block permutation, and ADC jitter locations."""
    if nb_blocks is None:
        nb_blocks = infer_nb_blocks(tw_or_capture)

    lfsr_info = reconstruct_lfsr_intermediates(tw_or_capture, nb_blocks=nb_blocks)
    jitter_info = locate_jitter_in_adc_trace(
        tw_or_capture,
        nb_blocks=nb_blocks,
        adc_mul=adc_mul,
        adc_offset=adc_offset,
    )

    bperm_rand: list[int] | None = None
    inferred_from_trace1 = False
    if nb_blocks == 2 and adc_values is not None:
        arr1 = np.asarray(adc_values, dtype=np.int16).reshape(-1)
        bperm_details = _infer_ct_bperm_details_from_trace1(
            arr1,
            jitter_info["blocks"],
            adc_mul=adc_mul,
            adc_offset=adc_offset,
        )
        if bperm_details is not None:
            bperm_rand = bperm_details["bperm_rand"]
            inferred_from_trace1 = True

    murmur_info = reconstruct_murmur_intermediates(
        nb_blocks=nb_blocks,
        bperm_rand=bperm_rand,
    )
    murmur_info["inferred_from_trace1"] = inferred_from_trace1

    return {
        "nb_blocks": nb_blocks,
        "lfsr": lfsr_info,
        "murmur": murmur_info,
        "jitter": jitter_info,
    }


def compute_trace2_from_trace1(
    trace1: list[int] | np.ndarray,
    tw_or_capture: dict[str, Any] | str | None,
    nb_blocks: int | None = None,
    target_length: int | None = None,
    adc_mul: int = 1,
    adc_offset: int = 0,
) -> np.ndarray:
    """Compute `trace2` (canonical de-jittered & un-permuted power trace) from `trace1`."""
    arr1 = np.asarray(trace1, dtype=np.int16).reshape(-1)
    out_len = int(target_length) if target_length is not None else int(arr1.shape[0])
    if arr1.size == 0:
        return np.zeros(out_len, dtype=np.int16)

    fill_val = int(arr1[-1])
    if tw_or_capture is None or (isinstance(tw_or_capture, str) and not tw_or_capture.strip()):
        return _fit_canonical_segment(arr1, out_len, fill_val=fill_val)

    try:
        tw = _extract_tw_dict(tw_or_capture)
        post_msgs, _ = _extract_post_decrypt_dwt_messages(tw)
        etm_events = tw.get("stream2_etmv4", {}).get("events", [])
        etm_raw = tw.get("stream2_etmv4", {}).get("raw_hex", "")
    except Exception:
        return _fit_canonical_segment(arr1, out_len, fill_val=fill_val)

    if len(post_msgs) < 10 or (len(etm_events) < 20 and not etm_raw):
        return _fit_canonical_segment(arr1, out_len, fill_val=fill_val)

    try:
        existing_recon = tw.get("protected_aes_reconstruction") if isinstance(tw, dict) else None
        if (
            isinstance(existing_recon, dict)
            and isinstance(existing_recon.get("lfsr"), dict)
            and bool(existing_recon["lfsr"].get("all_savelfsr_verified", False))
            and isinstance(existing_recon.get("murmur"), dict)
            and isinstance(existing_recon.get("jitter"), dict)
            and isinstance(existing_recon["jitter"].get("phases"), dict)
            and "ctr_crypt_s_mainloop" in existing_recon["jitter"]["phases"]
            and isinstance(existing_recon["jitter"].get("gen_rand_sha_calls"), list)
            and len(existing_recon["jitter"]["gen_rand_sha_calls"]) > 0
            and "excise_intervals" in existing_recon["jitter"]["gen_rand_sha_calls"][0]
            and int(existing_recon["jitter"].get("adc_mul", 1)) == int(adc_mul)
            and int(existing_recon["jitter"].get("adc_offset", 0)) == int(adc_offset)
        ):
            lfsr_res = existing_recon["lfsr"]
            murmur_res = existing_recon["murmur"]
            jitter_res = existing_recon["jitter"]
            resolved_nb_blocks = int(lfsr_res["nb_blocks"])
        else:
            inferred_nb = infer_nb_blocks(tw_or_capture)
            use_nb = (
                int(nb_blocks)
                if (nb_blocks is not None and int(nb_blocks) >= 1)
                else (inferred_nb if inferred_nb >= 1 else 1)
            )
            lfsr_res = reconstruct_lfsr_intermediates(tw_or_capture, nb_blocks=use_nb)
            resolved_nb_blocks = int(lfsr_res["nb_blocks"])
            bperm_rand = tw_or_capture.get("bperm_rand") if isinstance(tw_or_capture, dict) else None
            bperm_sha_words = (
                tw_or_capture.get("bperm_sha_words") if isinstance(tw_or_capture, dict) else None
            )
            murmur_res = reconstruct_murmur_intermediates(
                nb_blocks=resolved_nb_blocks,
                bperm_rand=bperm_rand,
                bperm_sha_words=bperm_sha_words,
            )
            jitter_res = locate_jitter_in_adc_trace(
                tw_or_capture,
                nb_blocks=resolved_nb_blocks,
                adc_mul=adc_mul,
                adc_offset=adc_offset,
            )
    except Exception:
        return _fit_canonical_segment(arr1, out_len, fill_val=fill_val)

    all_calls = jitter_res["gen_rand_sha_calls"]
    standalone_refills = jitter_res.get("standalone_refill_cycles", [])
    phases = jitter_res["phases"]

    bperm_details: dict[str, Any] | None = None
    perm_indices = murmur_res.get("permuted_block_indices")
    if (
        not murmur_res.get("exact_bperm_rand_known", False)
        or not perm_indices
        or any(p is None for p in perm_indices)
    ):
        bperm_details = _infer_ct_bperm_details_from_trace1(
            arr1,
            jitter_res["blocks"],
            adc_mul=adc_mul,
            adc_offset=adc_offset,
        )
        if bperm_details is not None:
            perm_indices = bperm_details["permuted_block_indices"]
        else:
            perm_indices = list(range(resolved_nb_blocks))

    period = 4 * adc_mul
    if arr1.shape[0] >= period:
        arr1_f = arr1.astype(np.float64)
        overall_mean = float(np.mean(arr1_f))
        for r_phase in range(period):
            arr1_f[r_phase::period] -= float(np.mean(arr1_f[r_phase::period])) - overall_mean
        arr1 = np.clip(np.round(arr1_f), -32768, 32767).astype(np.int16)

    def _standalone_in(c_lo: int, c_hi: int) -> list[int]:
        return [c for c in standalone_refills if c_lo < c <= c_hi]

    c_decrypt = int(phases["decrypt_entry"]["cycle"])
    c_lut_mask = int(phases["lut_mask_init"]["cycle"])
    c_init_key = int(phases["init_key_4way"]["cycle"])
    c_ctr_crypt = int(phases["ctr_crypt_s_entry"]["cycle"])
    c_bperm_done = int(phases["ctr_crypt_s_mainloop"]["cycle"])

    seg_init_rstate = _remove_jitter_calls_in_window(
        arr1, c_decrypt, c_lut_mask, [], expected_canonical_cycles=12634, adc_mul=adc_mul, adc_offset=adc_offset
    )
    seg_lut_remap = _remove_jitter_calls_in_window(
        arr1,
        c_lut_mask,
        c_init_key,
        [],
        expected_canonical_cycles=5792,
        adc_mul=adc_mul,
        adc_offset=adc_offset,
        extra_refill_cycles_list=_standalone_in(c_lut_mask, c_init_key),
    )
    seg_init_key = _remove_jitter_calls_in_window(
        arr1,
        c_init_key,
        c_ctr_crypt,
        all_calls[0:107],
        expected_canonical_cycles=21115,
        adc_mul=adc_mul,
        adc_offset=adc_offset,
        extra_refill_cycles_list=_standalone_in(c_init_key, c_ctr_crypt),
    )
    seg_ctr_prologue = _remove_jitter_calls_in_window(
        arr1,
        c_ctr_crypt,
        c_bperm_done,
        all_calls[107:140],
        expected_canonical_cycles=2402,
        adc_mul=adc_mul,
        adc_offset=adc_offset,
        extra_refill_cycles_list=_standalone_in(c_ctr_crypt, c_bperm_done),
    )

    slot_remap_segments: dict[int, np.ndarray] = {}
    phys_block_bodies: list[np.ndarray] = []
    last_block_reset_trng_seg: np.ndarray = np.empty(0, dtype=np.int16)

    for b in range(resolved_nb_blocks):
        blk_timing = jitter_res["blocks"][b]
        blk_lfsr = lfsr_res["blocks"][b]
        b_start = int(blk_timing["cycle_refreshchaff_start"])
        c_ref_done = int(blk_timing["cycle_refreshchaff_end"])
        c_rk_shares_start = int(blk_timing["cycle_ref_roundkey_shares_start"])
        c_hvperms_start = int(blk_timing["cycle_ref_roundkey_hvperms_start"])
        c_proc_iv_start = int(blk_timing["cycle_process_iv_start"])
        c_round0_start = int(blk_timing["rounds"][0]["cycle_round_start"])
        c_r14_start = int(blk_timing["cycle_round14_addrkey_start"])
        c_r14_end = int(blk_timing["cycle_round14_addrkey_end"])
        c_dec_start = int(blk_timing["cycle_decryption_start"])
        b_end = int(blk_timing["cycle_block_done"])

        seg_refresh = _remove_jitter_calls_in_window(
            arr1,
            b_start,
            c_ref_done,
            [all_calls[140 + 15 * b]],
            expected_canonical_cycles=504,
            adc_mul=adc_mul,
            adc_offset=adc_offset,
            extra_refill_cycles_list=_standalone_in(b_start, c_ref_done),
        )

        remap_can_len = 3963 if b % 4 == 0 else 4
        slot_remap_segments[b] = _remove_jitter_calls_in_window(
            arr1,
            c_ref_done,
            c_rk_shares_start,
            [],
            expected_canonical_cycles=remap_can_len,
            adc_mul=adc_mul,
            adc_offset=adc_offset,
            extra_refill_cycles_list=_standalone_in(c_ref_done, c_rk_shares_start),
        )

        seg_rk_shares = _remove_jitter_calls_in_window(
            arr1,
            c_rk_shares_start,
            c_hvperms_start,
            [],
            expected_canonical_cycles=1336,
            adc_mul=adc_mul,
            adc_offset=adc_offset,
        )
        if b >= 1:
            prev_hv_steps = lfsr_res["blocks"][b - 1].get("roundkey_hvperms", [])
            for r_rk in range(min(15, len(prev_hv_steps))):
                vp_a_prev = int(prev_hv_steps[r_rk]["share_a"]["vperm"]) & 0x3
                _unpermute_4word_slots(
                    seg_rk_shares, offset_cycles=23 + r_rk * 87 + 12, word_cycles=12, vperm=vp_a_prev, adc_mul=adc_mul
                )
                _unpermute_4word_slots(
                    seg_rk_shares, offset_cycles=23 + r_rk * 87 + 64, word_cycles=3, vperm=vp_a_prev, adc_mul=adc_mul
                )

        seg_rk_hvperms = _remove_jitter_calls_in_window(
            arr1,
            c_hvperms_start,
            c_proc_iv_start,
            [],
            expected_canonical_cycles=2810,
            adc_mul=adc_mul,
            adc_offset=adc_offset,
        )
        hv_steps = blk_lfsr.get("roundkey_hvperms", [])
        for r_rk in range(min(15, len(hv_steps))):
            vp_a = int(hv_steps[r_rk]["share_a"]["vperm"]) & 0x3
            vp_b = int(hv_steps[r_rk]["share_b"]["vperm"]) & 0x3
            _unpermute_4word_slots(
                seg_rk_hvperms, offset_cycles=3 + (2 * r_rk) * 48 + 20, word_cycles=6, vperm=vp_a, adc_mul=adc_mul
            )
            _unpermute_4word_slots(
                seg_rk_hvperms, offset_cycles=3 + (2 * r_rk + 1) * 48 + 20, word_cycles=6, vperm=vp_b, adc_mul=adc_mul
            )

        seg_proc_iv = _remove_jitter_calls_in_window(
            arr1,
            c_proc_iv_start,
            c_round0_start,
            [],
            expected_canonical_cycles=240,
            adc_mul=adc_mul,
            adc_offset=adc_offset,
            extra_refill_cycles_list=_standalone_in(c_proc_iv_start, c_round0_start),
        )

        round_sv_list = blk_lfsr.get("round_statevperm", [])
        round_segments: list[np.ndarray] = []
        for r in range(14):
            rnd_t = blk_timing["rounds"][r]
            sv_info = round_sv_list[r] if r < len(round_sv_list) else {}
            sv_in = int(sv_info.get("statevperm_before", 0)) & 0x3
            sv_out = int(sv_info.get("statevperm_after", sv_in)) & 0x3

            c_r_start = int(rnd_t["cycle_round_start"])
            c_sbox_start = int(rnd_t["cycle_map_sbox_start"])
            c_lookup_start = int(rnd_t["cycle_sbox_lookup_start"])
            c_r_end = int(rnd_t["cycle_round_end"])

            ak_can_len = 96 if r == 0 else 95
            base_ak = 8 if r == 0 else 7
            seg_addrkey = _remove_jitter_calls_in_window(
                arr1, c_r_start, c_sbox_start, [], expected_canonical_cycles=ak_can_len, adc_mul=adc_mul, adc_offset=adc_offset
            )
            _unpermute_4word_slots(seg_addrkey, offset_cycles=base_ak + 14, word_cycles=7, vperm=sv_in, adc_mul=adc_mul)
            _unpermute_4word_slots(seg_addrkey, offset_cycles=base_ak + 54, word_cycles=8, vperm=sv_in, adc_mul=adc_mul)

            call_k = 140 + 15 * b + 1 + r
            seg_mp = _remove_jitter_calls_in_window(
                arr1,
                c_sbox_start,
                c_lookup_start,
                [all_calls[call_k]],
                expected_canonical_cycles=210,
                adc_mul=adc_mul,
                adc_offset=adc_offset,
                extra_refill_cycles_list=_standalone_in(c_sbox_start, c_lookup_start),
            )
            _unpermute_4word_slots(seg_mp, offset_cycles=6, word_cycles=4, vperm=sv_in, adc_mul=adc_mul)

            sbox_mc_len = 617 if r < 7 else (539 if r < 13 else 433)
            seg_sbox_mc = _remove_jitter_calls_in_window(
                arr1, c_lookup_start, c_r_end, [], expected_canonical_cycles=sbox_mc_len, adc_mul=adc_mul, adc_offset=adc_offset
            )
            _unpermute_4word_slots(
                seg_sbox_mc, offset_cycles=365, word_cycles=4, vperm=sv_in, adc_mul=adc_mul, slot_order=(0, 2, 3, 1)
            )

            if r < 7:
                _unpermute_4word_slots(seg_sbox_mc, offset_cycles=467, word_cycles=4, vperm=sv_in, adc_mul=adc_mul)
                _unpermute_4word_slots(seg_sbox_mc, offset_cycles=495, word_cycles=4, vperm=sv_in, adc_mul=adc_mul)

            if r < 13:
                mc_base = 520 if r < 7 else 442
                _unpermute_4word_slots(seg_sbox_mc, offset_cycles=mc_base + 2, word_cycles=7, vperm=sv_out, adc_mul=adc_mul)
                _unpermute_4word_slots(seg_sbox_mc, offset_cycles=mc_base + 35, word_cycles=7, vperm=sv_out, adc_mul=adc_mul)

            round_segments.append(np.concatenate([seg_addrkey, seg_mp, seg_sbox_mc]))

        sv_r14 = (
            int(round_sv_list[13].get("statevperm_after", 0)) & 0x3
            if len(round_sv_list) >= 14
            else 0
        )
        seg_r14_addrkey = _remove_jitter_calls_in_window(
            arr1, c_r14_start, c_r14_end, [], expected_canonical_cycles=96, adc_mul=adc_mul, adc_offset=adc_offset
        )
        _unpermute_4word_slots(seg_r14_addrkey, offset_cycles=22, word_cycles=7, vperm=sv_r14, adc_mul=adc_mul)
        _unpermute_4word_slots(seg_r14_addrkey, offset_cycles=62, word_cycles=8, vperm=sv_r14, adc_mul=adc_mul)

        seg_r14_post = _remove_jitter_calls_in_window(
            arr1, c_r14_end, c_dec_start, [], expected_canonical_cycles=94, adc_mul=adc_mul, adc_offset=adc_offset
        )
        _unpermute_4word_slots(seg_r14_post, offset_cycles=36, word_cycles=4, vperm=sv_r14, adc_mul=adc_mul)
        _unpermute_4word_slots(seg_r14_post, offset_cycles=64, word_cycles=4, vperm=sv_r14, adc_mul=adc_mul)

        dec_can_len = int(
            blk_timing.get("decryption_start_canonical_cycles", 70 if b == resolved_nb_blocks - 1 else 44)
        )
        seg_dec_full = _remove_jitter_calls_in_window(
            arr1, c_dec_start, b_end, [], expected_canonical_cycles=dec_can_len, adc_mul=adc_mul, adc_offset=adc_offset
        )
        seg_dec_body = seg_dec_full[: 44 * adc_mul]
        if b == resolved_nb_blocks - 1 and dec_can_len > 44:
            last_block_reset_trng_seg = seg_dec_full[44 * adc_mul :]

        phys_block_bodies.append(
            np.concatenate([
                seg_refresh,
                seg_rk_shares,
                seg_rk_hvperms,
                seg_proc_iv,
                *round_segments,
                seg_r14_addrkey,
                seg_r14_post,
                seg_dec_body,
            ])
        )

    i_trajectory: list[int] | None = None
    if bperm_details is not None:
        i_trajectory = bperm_details.get("i_trajectory")
    elif (
        resolved_nb_blocks == 2
        and murmur_res.get("exact_bperm_rand_known", False)
        and len(murmur_res.get("blocks", [])) == 2
        and len(murmur_res["blocks"][0].get("rounds", [])) == 32
    ):
        i_trajectory = [
            int(r["i_before"]) for r in murmur_res["blocks"][0]["rounds"]
        ] + [int(murmur_res["blocks"][0]["permuted_block_index"])]

    dec_tail_len = 12 * adc_mul
    prologue_tail = (
        phys_block_bodies[0][:dec_tail_len].copy()
        if resolved_nb_blocks > 0 and phys_block_bodies[0].size >= dec_tail_len
        else None
    )
    phys_dec_tails: list[np.ndarray | None] = []
    for phys_b in range(resolved_nb_blocks):
        if phys_b + 1 < resolved_nb_blocks and phys_block_bodies[phys_b + 1].size >= dec_tail_len:
            phys_dec_tails.append(phys_block_bodies[phys_b + 1][:dec_tail_len].copy())
        elif phys_b + 1 == resolved_nb_blocks and last_block_reset_trng_seg.size >= dec_tail_len:
            phys_dec_tails.append(last_block_reset_trng_seg[:dec_tail_len].copy())
        else:
            phys_dec_tails.append(None)

    canonical_bodies: list[np.ndarray] = [
        phys_block_bodies[slot_idx].copy() for slot_idx in range(resolved_nb_blocks)
    ]
    for phys_b, logical_p in enumerate(perm_indices):
        if isinstance(logical_p, int) and 0 <= logical_p < resolved_nb_blocks:
            canonical_bodies[logical_p] = phys_block_bodies[phys_b].copy()

    if prologue_tail is not None and resolved_nb_blocks > 0:
        canonical_bodies[0][:dec_tail_len] = prologue_tail
    for phys_b, logical_p in enumerate(perm_indices):
        tail_seg = phys_dec_tails[phys_b] if phys_b < len(phys_dec_tails) else None
        if tail_seg is None or not isinstance(logical_p, int):
            continue
        if 0 <= logical_p < resolved_nb_blocks - 1:
            next_body = canonical_bodies[logical_p + 1]
            if next_body.size >= dec_tail_len:
                next_body[:dec_tail_len] = tail_seg
        elif logical_p == resolved_nb_blocks - 1 and last_block_reset_trng_seg.size >= dec_tail_len:
            last_block_reset_trng_seg[:dec_tail_len] = tail_seg

    if resolved_nb_blocks == 2 and i_trajectory is not None and len(i_trajectory) == 33:
        ct_bperm_base = (504 + 1336 + 1449) * adc_mul
        p0_bperm = phys_block_bodies[0][ct_bperm_base : ct_bperm_base + 1361 * adc_mul]
        p1_bperm = phys_block_bodies[1][ct_bperm_base : ct_bperm_base + 1361 * adc_mul]
        phys_bperms = (p0_bperm, p1_bperm)

        can0_bperm = canonical_bodies[0][ct_bperm_base : ct_bperm_base + 1361 * adc_mul]
        can1_bperm = canonical_bodies[1][ct_bperm_base : ct_bperm_base + 1361 * adc_mul]

        pro_end = 15 * adc_mul
        can0_bperm[:pro_end] = p0_bperm[:pro_end]
        can1_bperm[:pro_end] = p1_bperm[:pro_end]

        step_len = 42 * adc_mul
        for m in range(32):
            st = pro_end + m * step_len
            en = st + step_len
            i_m = int(i_trajectory[m]) & 1
            can0_bperm[st:en] = phys_bperms[i_m][st:en]
            can1_bperm[st:en] = phys_bperms[1 - i_m][st:en]

        epi_start = 1359 * adc_mul
        final_i = int(i_trajectory[32]) & 1
        can0_bperm[epi_start:] = phys_bperms[final_i][epi_start:]
        can1_bperm[epi_start:] = phys_bperms[1 - final_i][epi_start:]

    ref_len = 504 * adc_mul
    canonical_block_segments: list[np.ndarray] = []
    for slot_idx in range(resolved_nb_blocks):
        body = canonical_bodies[slot_idx]
        remap_can_len = 3963 if slot_idx % 4 == 0 else 4
        remap_seg = slot_remap_segments.get(
            slot_idx,
            np.full(remap_can_len * adc_mul, fill_val, dtype=np.int16),
        )
        canonical_block_segments.append(
            np.concatenate([body[:ref_len], remap_seg, body[ref_len:]])
        )
    if last_block_reset_trng_seg.size > 0:
        canonical_block_segments.append(last_block_reset_trng_seg)

    c_last_block_done = int(phases["decrypt_cleanup"]["cycle"])
    c_decrypt_end = int(phases["decrypt_end"]["cycle"])
    cleanup_call_idx = 140 + 15 * resolved_nb_blocks
    cleanup_calls = [all_calls[cleanup_call_idx]] if cleanup_call_idx < len(all_calls) else []
    seg_cleanup = _remove_jitter_calls_in_window(
        arr1,
        c_last_block_done,
        c_decrypt_end,
        cleanup_calls,
        expected_canonical_cycles=555,
        adc_mul=adc_mul,
        adc_offset=adc_offset,
        extra_refill_cycles_list=_standalone_in(c_last_block_done, c_decrypt_end),
    )

    tail_start_sample = max(0, min(int(arr1.shape[0]), c_decrypt_end * adc_mul - adc_offset))
    seg_tail = arr1[tail_start_sample:]

    trace2_full = np.concatenate([
        seg_init_rstate,
        seg_lut_remap,
        seg_init_key,
        seg_ctr_prologue,
        *canonical_block_segments,
        seg_cleanup,
        seg_tail,
    ])
    return _fit_canonical_segment(trace2_full, out_len, fill_val=fill_val)


# ==============================================================================
# 4. Sedpack Schema Builder & Mock Trace Generator
# ==============================================================================

def build_sedpack_dataset_structure(
    num_samples: int = 78400,
    examples_per_shard: int = 64,
    compression: str = "LZ4",
    shard_file_type: str = "fb",
    trace_dtype: str = "int16",
    nb_blocks: int = 2,
) -> DatasetStructure:
    """Construct the 14-attribute `sedpack` `DatasetStructure` with full supervision labels."""
    attributes = [
        Attribute(
            name="trace1",
            shape=(num_samples,),
            dtype=trace_dtype,
            custom_metadata={
                "type": "power",
                "description": "Raw un-cut ChipWhisperer Husky ADC waveform encompassing all POIs",
                "unit": "adc_raw",
            },
        ),
        Attribute(
            name="trace2",
            shape=(num_samples,),
            dtype=trace_dtype,
            custom_metadata={
                "type": "power_canonical",
                "description": (
                    "Canonical de-jittered and un-permuted power waveform computed from trace1 "
                    "using TraceWhisperer DWT/ETMv4 streams"
                ),
                "unit": "adc_raw",
            },
        ),
        Attribute(
            name="plaintext",
            shape=(16 * nb_blocks,),
            dtype="uint8",
            custom_metadata={"max_val": 256, "description": "AES-CTR input plaintext block"},
        ),
        Attribute(
            name="ciphertext",
            shape=(16 * nb_blocks,),
            dtype="uint8",
            custom_metadata={"max_val": 256, "description": "AES-CTR output ciphertext block"},
        ),
        Attribute(
            name="iv",
            shape=(16,),
            dtype="uint8",
            custom_metadata={"max_val": 256, "description": "AES-CTR 16-byte initialization vector"},
        ),
        Attribute(
            name="iv_salt",
            shape=(16,),
            dtype="uint8",
            custom_metadata={"max_val": 256, "description": "16-byte random IV salt (XORed with IV)"},
        ),
        Attribute(
            name="key",
            shape=(32,),
            dtype="uint8",
            custom_metadata={"max_val": 256, "description": "AES-256 32-byte master key"},
        ),
        Attribute(
            name="key_a",
            shape=(32,),
            dtype="uint8",
            custom_metadata={"max_val": 256, "description": "4-way Boolean key share Sa"},
        ),
        Attribute(
            name="key_b",
            shape=(32,),
            dtype="uint8",
            custom_metadata={"max_val": 256, "description": "4-way Boolean key share Sb"},
        ),
        Attribute(
            name="key_c",
            shape=(32,),
            dtype="uint8",
            custom_metadata={"max_val": 256, "description": "4-way Boolean key share Sc"},
        ),
        Attribute(
            name="key_d",
            shape=(32,),
            dtype="uint8",
            custom_metadata={"max_val": 256, "description": "4-way Boolean key share Sd"},
        ),
        Attribute(
            name="round_keys",
            shape=(15, 16),
            dtype="uint8",
            custom_metadata={"max_val": 256, "description": "15 expanded AES-256 round keys"},
        ),
        Attribute(
            name="intermediate_shares",
            shape=(4, 16),
            dtype="uint8",
            custom_metadata={"max_val": 256, "description": "Round 0 4-way initial state shares"},
        ),
        Attribute(
            name="tracewhisperer",
            shape=(),
            dtype="str",
            custom_metadata={
                "type": "tracewhisperer_json",
                "description": "JSON-encoded CoreSight Stream 1 (DWT/ITM) and Stream 2 (ETMv4) packets",
            },
        ),
    ]

    return DatasetStructure(
        saved_data_description=attributes,
        examples_per_shard=examples_per_shard,
        compression=compression,
        shard_file_type=shard_file_type,
    )


def build_minimal_holdout_example(
    trace1: np.ndarray,
    key: bytes | np.ndarray,
    iv: bytes | np.ndarray,
    iv_salt: bytes | np.ndarray,
    num_samples: int,
    nb_blocks: int,
) -> dict[str, Any]:
    """Construct a minimal holdout example where ONLY `trace1`, `key`, `iv`, and `iv_salt`
    contain real values, while all other schema attributes are zero-filled for compatibility.
    """
    key_arr = (
        np.frombuffer(key, dtype=np.uint8)
        if isinstance(key, (bytes, bytearray, memoryview))
        else np.asarray(key, dtype=np.uint8)
    )
    iv_arr = (
        np.frombuffer(iv, dtype=np.uint8)
        if isinstance(iv, (bytes, bytearray, memoryview))
        else np.asarray(iv, dtype=np.uint8)
    )
    iv_salt_arr = (
        np.frombuffer(iv_salt, dtype=np.uint8)
        if isinstance(iv_salt, (bytes, bytearray, memoryview))
        else np.asarray(iv_salt, dtype=np.uint8)
    )
    pt_len = 16 * int(nb_blocks)
    return {
        "trace1": np.asarray(trace1, dtype=np.int16)[:num_samples],
        "trace2": np.zeros((num_samples,), dtype=np.int16),
        "plaintext": np.zeros((pt_len,), dtype=np.uint8),
        "ciphertext": np.zeros((pt_len,), dtype=np.uint8),
        "iv": iv_arr,
        "iv_salt": iv_salt_arr,
        "key": key_arr,
        "key_a": np.zeros((32,), dtype=np.uint8),
        "key_b": np.zeros((32,), dtype=np.uint8),
        "key_c": np.zeros((32,), dtype=np.uint8),
        "key_d": np.zeros((32,), dtype=np.uint8),
        "round_keys": np.zeros((15, 16), dtype=np.uint8),
        "intermediate_shares": np.zeros((4, 16), dtype=np.uint8),
        "tracewhisperer": "",
    }


def _normalize_iv_mode(iv_mode: str) -> str:
    """Normalize user-specified IV/IV-salt renewal mode into one of the 3 canonical modes."""
    m = (iv_mode or "per_trace").strip().lower().replace("-", "_")
    if m in ("fixed_per_dataset", "fixed_dataset", "dataset", "fixed", "1"):
        return "fixed_per_dataset"
    if m in ("per_key", "fixed_per_key", "key", "with_key", "renew_with_key", "2"):
        return "per_key"
    if m in ("per_trace", "random", "trace", "current", "default", "3"):
        return "per_trace"
    raise ValueError(
        f"Invalid iv_mode '{iv_mode}'. Must be one of: "
        "'fixed_per_dataset', 'per_key', or 'per_trace'."
    )


def generate_mock_example(
    num_samples: int = 78400,
    key_override: bytes | None = None,
    include_tw: bool = True,
    nb_blocks: int = 2,
    minimal_example: bool = False,
    iv_override: bytes | None = None,
    iv_salt_override: bytes | None = None,
) -> dict[str, Any]:
    """Generate a single mathematically consistent mock example for offline testing and verification."""
    key = key_override if key_override is not None else os.urandom(32)
    sa, sb, sc, sd = split_key_4way(key)

    plaintext = np.frombuffer(os.urandom(16 * nb_blocks), dtype=np.uint8)
    iv_bytes = iv_override if iv_override is not None else os.urandom(16)
    iv_salt_bytes = iv_salt_override if iv_salt_override is not None else os.urandom(16)
    if len(iv_bytes) != 16:
        raise ValueError(f"IV must be 16 bytes, got {len(iv_bytes)} bytes.")
    if len(iv_salt_bytes) != 16:
        raise ValueError(f"IV salt must be 16 bytes, got {len(iv_salt_bytes)} bytes.")
    iv = np.frombuffer(iv_bytes, dtype=np.uint8)
    iv_salt = np.frombuffer(iv_salt_bytes, dtype=np.uint8)

    intermediate_shares = compute_intermediate_shares(sa, sb, sc, sd, iv, iv_salt)

    rng = np.random.default_rng()
    noise = rng.normal(loc=0.0, scale=120.0, size=num_samples).astype(np.float32)
    baseline = np.sin(np.linspace(0, 100 * np.pi, num_samples, dtype=np.float32)) * 300.0

    # Inject slight Hamming-weight leakage at representative POI windows (when trace is long enough)
    if num_samples >= 6200:
        hw_sa = int(sa[0]).bit_count()
        noise[6000:6200] += float(hw_sa * 45.0)
    if num_samples >= 8600:
        hw_iv = int(iv[0] ^ iv_salt[0]).bit_count()
        noise[8400:8600] += float(hw_iv * 35.0)
    if num_samples >= 10300:
        sbox_in = int(
            intermediate_shares[0, 0]
            ^ intermediate_shares[1, 0]
            ^ intermediate_shares[2, 0]
            ^ intermediate_shares[3, 0]
        )
        noise[9900:10300] += float(AES_SBOX[sbox_in].bit_count() * 50.0)

    trace1 = np.clip(baseline + noise, -32768, 32767).astype(np.int16)

    if minimal_example:
        return build_minimal_holdout_example(
            trace1=trace1,
            key=key,
            iv=iv,
            iv_salt=iv_salt,
            num_samples=num_samples,
            nb_blocks=nb_blocks,
        )

    ct_bytes = compute_aes_ctr_output(
        key=key,
        iv=iv_bytes,
        iv_salt=iv_salt_bytes,
        plaintext=plaintext.tobytes(),
    )
    ciphertext = np.frombuffer(ct_bytes, dtype=np.uint8)
    round_keys = expand_aes256_round_keys(key)

    if include_tw:
        tw_payload = {
            "stream1_itm_dwt": {
                "messages": [
                    {"type": "data_trace", "comparator": 0, "value": int(sa[0]), "cycles": 6050},
                    {"type": "data_trace", "comparator": 1, "value": 0x42, "cycles": 6120},
                    {"type": "data_trace", "comparator": 2, "value": 0x1D872B41, "cycles": 6180},
                ]
            },
            "stream2_etmv4": {
                "events": [
                    {"type": "address", "address": 0x10001518, "cycles": 10},
                    {"type": "branch_target", "address": 0x100016C2, "cycles": 5990},
                    {"type": "cycle_count", "cycles": 64},
                ],
                "total_cycles": 76800 + 18500 * (nb_blocks - 1),
            },
        }
        tw_str = json.dumps(tw_payload)
    else:
        tw_str = ""

    trace2 = compute_trace2_from_trace1(
        trace1,
        tw_str,
        nb_blocks=nb_blocks,
        target_length=num_samples,
    )

    return {
        "trace1": trace1,
        "trace2": trace2,
        "plaintext": plaintext,
        "ciphertext": ciphertext,
        "iv": iv,
        "iv_salt": iv_salt,
        "key": np.frombuffer(key, dtype=np.uint8),
        "key_a": sa,
        "key_b": sb,
        "key_c": sc,
        "key_d": sd,
        "round_keys": round_keys,
        "intermediate_shares": intermediate_shares,
        "tracewhisperer": tw_str,
    }


# ==============================================================================
# 5. Hardware Acquisition Driver & Interrupt Recovery Guard
# ==============================================================================

class _HardwareInterruptGuard:
    """Defers SIGINT/SIGTERM/SIGHUP during ~40 ms atomic USB/UART transfers so the
    ChipWhisperer Husky SAM3U FastSMC interface and RP2350 SimpleSerial2 UART are
    never interrupted mid-packet, and guarantees clean hardware teardown on exit.
    """

    def __init__(self, cleanup_fn: Callable[[], None]) -> None:
        self._cleanup_fn = cleanup_fn
        self._in_atomic = False
        self._pending_signum: int | None = None
        self._installed = False
        self._prev_handlers: dict[int, Any] = {}
        self._atexit_registered = False

    def install(self) -> None:
        if not self._atexit_registered:
            atexit.register(self._safe_cleanup)
            self._atexit_registered = True
        if self._installed or threading.current_thread() is not threading.main_thread():
            return
        for sig_name in ("SIGINT", "SIGTERM", "SIGHUP"):
            sig = getattr(signal, sig_name, None)
            if sig is not None:
                try:
                    self._prev_handlers[sig] = signal.getsignal(sig)
                    signal.signal(sig, self._handle_signal)
                except Exception:
                    pass
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        for sig, prev in self._prev_handlers.items():
            try:
                signal.signal(sig, prev)
            except Exception:
                pass
        self._prev_handlers.clear()
        self._installed = False

    def _safe_cleanup(self) -> None:
        try:
            self._cleanup_fn()
        except Exception:
            pass

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        if self._in_atomic and self._pending_signum is None:
            self._pending_signum = signum
            return
        self._pending_signum = None
        raise KeyboardInterrupt(f"Interrupted by signal {signum}")

    @contextlib.contextmanager
    def atomic_hardware_op(self) -> Iterator[None]:
        if self._pending_signum is not None:
            sig = self._pending_signum
            self._pending_signum = None
            raise KeyboardInterrupt(f"Interrupted by signal {sig}")
        self._in_atomic = True
        try:
            yield
        finally:
            self._in_atomic = False
            if self._pending_signum is not None:
                sig = self._pending_signum
                self._pending_signum = None
                raise KeyboardInterrupt(f"Interrupted by signal {sig}")


class HardwareTraceCapturer:
    """Standalone ChipWhisperer Husky + TraceWhisperer + RP2350 capture driver for `aes_ctr_asm`."""

    def __init__(
        self,
        elf_path: str | Path | None = None,
        target_baudrate: int = DEFAULT_BAUDRATE,
        clkgen_freq: float = DEFAULT_CLKGEN_FREQ,
        nb_blocks: int = 2,
        warmup_captures_per_key: int = 10,
    ) -> None:
        self.elf_path: Path = _resolve_elf_path(elf_path)
        self.data_addrs, self.func_addrs = compute_dwt_addresses(self.elf_path)
        self.baudrate: int = int(target_baudrate)
        self.clkgen_freq: float = float(clkgen_freq)
        self.nb_blocks: int = max(1, int(nb_blocks))
        self.warmup_captures_per_key: int = max(0, int(warmup_captures_per_key))

        self.scope: Any = None
        self.target: Any = None
        self.connected: bool = False
        self.trace_enabled: bool = False
        self.trace_stall_cpu: bool = False
        self.trace_gate_on_trigger: bool = True
        self._last_trace_config_summary: dict[str, Any] | None = None
        self._last_warmed_key: bytes | None = None

        self.calibrated_samples: int = 78400 + 18800 * (self.nb_blocks - 1)
        self.calibrated_gain: float = 36.56
        self.conn_info: dict[str, Any] = {}
        self._interrupt_guard = _HardwareInterruptGuard(self.disconnect)

    def _ensure_connected(self) -> None:
        if not self.connected or self.scope is None or self.target is None:
            raise RuntimeError("Not connected to ChipWhisperer Husky scope/target. Call connect() first.")

    def set_adc_samples(self, num_samples: int) -> None:
        """Set calibrated ADC sample window length on both driver state and Husky scope."""
        self.calibrated_samples = int(num_samples)
        if self.scope is not None:
            self.scope.adc.samples = self.calibrated_samples

    def _set_trace_reg(self, reg_idx: int, val: int) -> None:
        self.target.send_cmd(ord("s"), 0x00, struct.pack(">BI", reg_idx, val & 0xFFFFFFFF))
        self.target.simpleserial_wait_ack(500)

    def _get_hw_max_samples(self) -> int:
        if self.scope is None:
            return 0
        max_s = getattr(self.scope.adc, "max_samples", None)
        if max_s is not None and int(max_s) > 0:
            return int(max_s)
        return int(getattr(self.scope.adc.oa, "hwMaxSamples", 0))

    def _read_target_frequencies(self) -> dict[str, int] | None:
        try:
            self.target.flush()
            self.target.send_cmd(ord("P"), 0x02, bytearray())
            raw = self.target.simpleserial_read("r", pay_len=36, timeout=500, ack=True)
            vals = struct.unpack("<9I", bytes(raw))
            names = (
                "xosc_khz",
                "sys_pll_khz",
                "usb_pll_khz",
                "rosc_khz",
                "clk_sys_khz",
                "clk_peri_khz",
                "clk_usb_khz",
                "clk_adc_khz",
                "clk_ref_khz",
            )
            return dict(zip(names, vals))
        except Exception:
            return None

    def _center_target_clock_phase(self) -> None:
        """Ensure RP2350 `pll_sys` 49-way post-divider is locked in a centered ADC phase `[1650..2300]`."""
        if (
            self.scope is None
            or self.target is None
            or abs(self.clkgen_freq - 24_000_000.0) >= 1000
        ):
            return

        saved_samples = int(self.scope.adc.samples)
        saved_gain_db = float(self.scope.gain.db)
        saved_offset = int(self.scope.adc.offset)
        try:
            self.scope.adc.offset = 0
            self.scope.adc.samples = min(4096, self._get_hw_max_samples() or 4096)
            self.scope.gain.db = 35.0
            probe_key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
            probe_pt = bytes.fromhex("6bc1bee22e409f96e93d7e117393172a")
            self.target.send_cmd(0x01, 0x02, bytearray(probe_key))
            self.target.simpleserial_wait_ack(500)

            def _probe_trigger() -> None:
                self.target.send_cmd(0x01, 0x01, bytearray(probe_pt))
                self.target.simpleserial_read("r", pay_len=16, timeout=1000, ack=True)

            _probe_trigger()

            relocked = False
            for attempt in range(10):
                self.scope.adc.errors = False
                self.scope.arm()
                _probe_trigger()
                if self.scope.capture():
                    break
                tr_int = self.scope.get_last_trace(as_int=True)
                if tr_int.size == 0:
                    break
                mean_v = float(tr_int.mean())
                ptp_v = int(tr_int.max()) - int(tr_int.min())
                if 1650.0 <= mean_v <= 2300.0 and ptp_v >= 900:
                    break
                if attempt + 1 < 10:
                    self.target.send_cmd(ord("P"), 0x10, bytearray())
                    self.target.simpleserial_wait_ack(500)
                    self.target.send_cmd(ord("P"), 0x11, bytearray())
                    self.target.simpleserial_wait_ack(500)
                    relocked = True
            if relocked and self.trace_enabled:
                time.sleep(0.05)
                self.scope.trace.resync()
        except Exception:
            pass
        finally:
            try:
                self.scope.adc.offset = saved_offset
                self.scope.adc.samples = saved_samples
                self.scope.gain.db = saved_gain_db
                self.scope.adc.errors = False
            except Exception:
                pass

    def _sync_target_clock(self, force: bool = False) -> dict[str, int] | None:
        """Synchronize RP2350 PLL/clocks with Husky `clkgen_freq` and return measured kHz."""
        if self.target is None:
            return None
        self.target.reset_comms()
        self.target.flush()
        if not force:
            existing = self._read_target_frequencies()
            if existing is not None and abs(existing.get("clk_sys_khz", 0) - 24000) <= 100:
                return existing

        if abs(self.clkgen_freq - 24_000_000.0) < 1000:
            self.target.send_cmd(ord("P"), 0x11, bytearray())
            self.target.simpleserial_wait_ack(500)
            time.sleep(0.05)
            self._center_target_clock_phase()
        elif abs(self.clkgen_freq - 12_000_000.0) < 1000:
            self.target.send_cmd(ord("P"), 0x13, bytearray())
            self.target.simpleserial_wait_ack(500)
            time.sleep(0.05)
        return self._read_target_frequencies()

    def _connect_scope_and_target(
        self,
        sn: str | None = None,
        force_bitstream: bool = False,
    ) -> dict[str, Any]:
        """Open ChipWhisperer Husky USB scope and SimpleSerial2 RP2350 target."""
        try:
            import chipwhisperer as cw
        except ImportError as exc:
            raise ImportError(
                "Package 'chipwhisperer' is required for --mode hardware."
            ) from exc

        if self.scope is not None:
            self._disconnect_hardware()

        # Patch OpenADCInterface._slurp_registers once so SAM3U always exits FastSMC mode cleanly
        try:
            from chipwhisperer.capture.scopes._OpenADCInterface import OpenADCInterface

            if not getattr(OpenADCInterface, "_fast_smc_patched", False):
                _orig_slurp = OpenADCInterface._slurp_registers

                def _patched_slurp_registers(oa_self: Any) -> None:
                    _orig_slurp(oa_self)
                    try:
                        oa_self.serial.set_smc_speed(0)
                    except Exception:
                        pass
                    try:
                        oa_self.setFastFIFORead(0)
                    except Exception:
                        pass

                OpenADCInterface._slurp_registers = _patched_slurp_registers
                OpenADCInterface._fast_smc_patched = True
        except Exception:
            pass

        try:
            self.scope = (
                cw.scope(sn=sn, force=force_bitstream)
                if sn
                else cw.scope(force=force_bitstream)
            )
        except Exception:
            if force_bitstream:
                raise
            import gc

            gc.collect()
            time.sleep(0.1)
            self.scope = cw.scope(sn=sn, force=True) if sn else cw.scope(force=True)

        try:
            self.scope.adc.oa.setFastSMC(0)
            self.scope.adc.oa._setMaxSamples()
        except Exception:
            pass

        if not force_bitstream and self._get_hw_max_samples() < 100000:
            resolved_sn = sn or getattr(self.scope, "sn", None)
            try:
                self.scope.dis()
            except Exception:
                pass
            self.scope = cw.scope(sn=resolved_sn, force=True) if resolved_sn else cw.scope(force=True)

        self.scope.default_setup(verbose=False)
        try:
            self.scope.io.cwe.setAVRISPMode(0)
        except Exception:
            pass
        try:
            self.scope._getNAEUSB().set_husky_tms_wr(0)
        except Exception:
            pass
        self.scope.userio.mode = "normal"
        self.scope.userio.direction = 0
        self.scope.userio.drive_data = 0
        self.scope.clock.clkgen_src = "system"
        self.scope.clock.clkgen_freq = self.clkgen_freq
        self.scope.clock.adc_mul = 1
        self.scope.io.hs2 = "clkgen"
        self.scope.io.tio1 = "serial_rx"
        self.scope.io.tio2 = "serial_tx"
        self.scope.io.target_pwr = True
        self.scope.io.nrst = "high_z"
        self.scope.trigger.triggers = "tio4"
        time.sleep(0.1)

        self.target = cw.target(self.scope, cw.targets.SimpleSerial2)
        self.target.baud = self.baudrate
        self.target.reset_comms()
        self.target.flush()
        self.connected = True

        self.reset_target(sync_clock_24mhz=True, reenable_trace=False)
        freqs = self._read_target_frequencies()
        self.trace_enabled = bool(self.scope.trace.enabled)

        return {
            "status": "connected",
            "scope_name": self.scope.get_name(),
            "scope_sn": self.scope.sn,
            "fw_version": getattr(self.scope, "fw_version_str", None),
            "cw_version": getattr(cw, "__version__", None),
            "is_husky_plus": bool(getattr(self.scope, "_is_husky_plus", False)),
            "baudrate": self.target.baud,
            "clkgen_freq_hz": self.scope.clock.clkgen_freq,
            "adc_freq_hz": self.scope.clock.adc_freq,
            "gain_db": self.scope.gain.db,
            "gain_mode": self.scope.gain.mode,
            "gain_raw": int(self.scope.gain.gain),
            "adc_samples": self.scope.adc.samples,
            "hw_max_samples": self._get_hw_max_samples(),
            "tracewhisperer_enabled": self.trace_enabled,
            "target_clocks_khz": freqs,
            "elf_path": str(self.elf_path),
        }

    def reset_target(
        self,
        sync_clock_24mhz: bool = True,
        reenable_trace: bool = True,
    ) -> None:
        """Pulse RP2350 nRST, wait for firmware boot, and restore clock & TraceWhisperer state."""
        self._ensure_connected()
        was_trace = self.trace_enabled

        self.scope.io.nrst = "low"
        time.sleep(0.05)
        self.scope.io.nrst = "high_z"
        time.sleep(0.58)

        self.target.reset_comms()
        self.target.flush()

        if sync_clock_24mhz:
            self._sync_target_clock(force=True)

        if was_trace and reenable_trace:
            self.set_tracewhisperer(
                enabled=True,
                stall_cpu=self.trace_stall_cpu,
                gate_etm_on_trigger=self.trace_gate_on_trigger,
            )
        else:
            self.trace_enabled = False

    def set_tracewhisperer(
        self,
        enabled: bool,
        stall_cpu: bool = False,
        gate_etm_on_trigger: bool = True,
    ) -> None:
        """Enable or disable TraceWhisperer 4-bit parallel TPIU trace capture."""
        self._ensure_connected()
        trace = self.scope.trace
        if not enabled:
            try:
                self.target.send_cmd(ord("T"), 0x00, bytearray())
                self.target.simpleserial_wait_ack(500)
                self.target.send_cmd(ord("T"), 0x04, bytearray([0]))
                self.target.simpleserial_wait_ack(500)
            except Exception:
                pass
            trace.enabled = False
            self.scope.userio.mode = "normal"
            self.scope.userio.direction = 0
            self.trace_enabled = False
            self.trace_stall_cpu = False
            return

        self._sync_target_clock(force=False)

        self.target.send_cmd(ord("e"), 0x00, bytearray())
        self.target.simpleserial_wait_ack(500)

        self.trace_gate_on_trigger = gate_etm_on_trigger
        self._set_trace_reg(20, 1 if gate_etm_on_trigger else 0)

        self.trace_stall_cpu = stall_cpu
        self.target.send_cmd(ord("T"), 0x04, bytearray([1 if stall_cpu else 0]))
        self.target.simpleserial_wait_ack(500)

        trace.target = self.target
        trace.enabled = True
        trace.trace_mode = "parallel"
        trace.trace_width = 4
        trace.clock.fe_clock_src = "target_clock"
        trace.capture.trigger_source = "firmware trigger"
        trace.capture.raw = True
        trace.capture.rules_enabled = []
        trace.capture.record_syncs = False
        trace.capture.mode = "while_trig"
        self.scope.trigger.triggers = "tio4"
        time.sleep(0.25)
        trace.resync()
        self.trace_enabled = True

    def _configure_trace_for_aes_ctr(self) -> dict[str, Any]:
        """Configure RP2350 DWT (`COMP0..COMP3`) & ETMv4 (`CCCTLR=64`) for `aes_ctr_asm` (`decrypt`)."""
        rstate_sha = self.data_addrs["rstate_sha"]
        jstate = self.data_addrs["jstate"]
        rstate_lfsr = self.data_addrs["rstate_lfsr"]
        decrypt_pc = self.func_addrs["decrypt"]

        self._set_trace_reg(20, 1 if self.trace_gate_on_trigger else 0)
        self.target.send_cmd(ord("T"), 0x00, bytearray())
        self.target.simpleserial_wait_ack(500)
        self.target.send_cmd(ord("T"), 0x04, bytearray([1 if self.trace_stall_cpu else 0]))
        self.target.simpleserial_wait_ack(500)

        self.target.send_cmd(ord("T"), 0x03, bytearray([0, 1, 64]))
        self.target.simpleserial_wait_ack(500)

        self.target.send_cmd(ord("T"), 0x01, struct.pack(">III", rstate_sha, jstate, rstate_lfsr))
        self.target.simpleserial_wait_ack(500)

        self._set_trace_reg(14, decrypt_pc)
        self._set_trace_reg(18, 0x00000422)  # DWT_FUNC_TRACE_PC_MATCH on COMP3

        mode_str = "DATA_WRITE_VALUE (0x82D)"
        config_summary: dict[str, Any] = {
            "elf_path": str(self.elf_path),
            "implementation": "aes_ctr_asm",
            "stall_cpu": self.trace_stall_cpu,
            "gate_etm_on_trigger": self.trace_gate_on_trigger,
            "etm_config": {"branch_broadcast": False, "cycle_counting": True, "ccctlr": 64},
            "comparators": {
                "COMP0": {"symbol": "rstate_sha", "address": f"0x{rstate_sha:08x}", "mode": mode_str},
                "COMP1": {"symbol": "jstate", "address": f"0x{jstate:08x}", "mode": mode_str},
                "COMP2": {"symbol": "rstate_lfsr", "address": f"0x{rstate_lfsr:08x}", "mode": mode_str},
                "COMP3": {"symbol": "decrypt", "address": f"0x{decrypt_pc:08x}", "mode": "PC_MATCH (0x422)"},
            },
        }

        # Warm up RP2350 XIP flash cache for decrypt() after trace register setup
        self.target.send_cmd(0x02, 0x30, bytearray([1]))
        self.target.simpleserial_read("r", pay_len=16, timeout=1000, ack=True)

        self._last_trace_config_summary = config_summary
        return config_summary

    def _prepare_and_trigger_aes_ctr(
        self,
        shared_key: bytes,
        iv: bytes,
        iv_salt: bytes,
        plaintext: bytes,
        nb_blocks: int,
    ) -> Callable[[], bytes]:
        """Send SimpleSerial2 key/IV/salt/data setup frames and return the `0x02 0x30` trigger callable."""
        self.target.send_cmd(0x02, 0x00, bytearray(shared_key))
        self.target.simpleserial_wait_ack(500)
        self.target.send_cmd(0x02, 0x10, bytearray(iv[:16]))
        self.target.simpleserial_wait_ack(500)
        self.target.send_cmd(0x02, 0x11, bytearray(iv_salt[:16]))
        self.target.simpleserial_wait_ack(500)
        self.target.send_cmd(0x02, 0x20, bytearray(plaintext[: 16 * nb_blocks]))
        self.target.simpleserial_wait_ack(500)

        def trigger_fn() -> bytes:
            self.target.send_cmd(0x02, 0x30, bytearray([nb_blocks]))
            resp = self.target.simpleserial_read(
                "r", pay_len=16 * nb_blocks, timeout=1500, ack=True
            )
            return bytes(resp)

        return trigger_fn

    def _calibrate(self, target_swing_ratio: float = 0.80) -> tuple[int, float]:
        """Compute and apply optimal `scope.adc.samples` and `scope.gain.db` for `aes_ctr_asm`."""
        self._ensure_connected()
        was_trace_enabled = bool(self.trace_enabled)
        eff_blocks = self.nb_blocks

        self._sync_target_clock(force=False)
        rng = np.random.default_rng(0x525032333530)

        def _prep_cal_vector() -> Callable[[], bytes]:
            shared_k = _concatenate_shares(
                rng.bytes(32), rng.bytes(32), rng.bytes(32), rng.bytes(32)
            )
            return self._prepare_and_trigger_aes_ctr(
                shared_key=shared_k,
                iv=rng.bytes(16),
                iv_salt=rng.bytes(16),
                plaintext=rng.bytes(16 * eff_blocks),
                nb_blocks=eff_blocks,
            )

        self._center_target_clock_phase()

        if not self.trace_enabled:
            try:
                self.target.send_cmd(ord("T"), 0x04, bytearray([0]))
                self.target.simpleserial_wait_ack(500)
            except Exception:
                pass
        else:
            self._configure_trace_for_aes_ctr()

        trigger_fn = _prep_cal_vector()
        for _ in range(3):
            _ = trigger_fn()

        hw_max = self._get_hw_max_samples()

        self.scope.adc.samples = min(5000, hw_max)
        self.scope.gain.db = 35.0
        self.scope.adc.errors = False
        if self.trace_enabled:
            self.scope.trace.resync()
            self.scope.trace.arm_trace()
        self.scope.arm()
        _ = trigger_fn()
        if self.scope.capture():
            cur_sn = getattr(self.scope, "sn", None)
            self._connect_scope_and_target(sn=cur_sn, force_bitstream=True)
            if was_trace_enabled:
                self.set_tracewhisperer(enabled=True)
                self._configure_trace_for_aes_ctr()
            trigger_fn = _prep_cal_vector()
            for _ in range(3):
                _ = trigger_fn()
            hw_max = self._get_hw_max_samples()
            self.scope.adc.samples = min(5000, hw_max)
            self.scope.gain.db = 35.0
            self.scope.adc.errors = False
            if self.trace_enabled:
                self.scope.trace.resync()
                self.scope.trace.arm_trace()
            self.scope.arm()
            _ = trigger_fn()
            if self.scope.capture():
                raise RuntimeError("Scope capture timed out waiting for trigger during calibration.")

        tr_probe = self.scope.get_last_trace(as_int=True)
        probe_mean = float(tr_probe.mean()) if tr_probe.size > 0 else 2048.0
        probe_ptp = (int(tr_probe.max()) - int(tr_probe.min())) if tr_probe.size > 0 else 0
        if (
            abs(self.clkgen_freq - 24_000_000.0) < 1000
            and (probe_mean < 1650.0 or probe_mean > 2300.0 or probe_ptp < 900)
        ):
            self._center_target_clock_phase()
            if self.trace_enabled:
                self._configure_trace_for_aes_ctr()
                self.scope.trace.resync()
            trigger_fn = _prep_cal_vector()
            for _ in range(3):
                _ = trigger_fn()
            self.scope.adc.samples = min(5000, hw_max)
            self.scope.gain.db = 35.0
            self.scope.adc.errors = False
            if self.trace_enabled:
                self.scope.trace.resync()
                self.scope.trace.arm_trace()
            self.scope.arm()
            _ = trigger_fn()
            self.scope.capture()

        trig_count = int(self.scope.adc.trig_count)
        offset = int(self.scope.adc.offset)
        presamples = int(self.scope.adc.presamples)

        jitter_margin = 1900 + 300 * (eff_blocks - 1)
        raw_optimal = trig_count + jitter_margin - offset + presamples
        if offset == 0:
            raw_optimal = max(raw_optimal, 78400 + 18800 * (eff_blocks - 1))
        optimal_samples = min(max(64, raw_optimal), hw_max)
        self.scope.adc.samples = optimal_samples

        lo_db = 25.0
        hi_db = 44.0
        best_db = 35.0
        target_excursion = max(512, min(1750, int(2048 * target_swing_ratio)))

        for _ in range(7):
            mid_db = (lo_db + hi_db) / 2.0
            self.scope.gain.db = mid_db
            self.scope.adc.errors = False
            self.scope.arm()
            _ = trigger_fn()
            self.scope.capture()

            trig_count = max(trig_count, int(self.scope.adc.trig_count))
            tr_int = self.scope.get_last_trace(as_int=True)
            min_v = int(tr_int.min())
            max_v = int(tr_int.max())
            excursion = max(max_v - 2048, 2048 - min_v)
            err_str = str(self.scope.adc.errors or "")
            clipped = ("clipped" in err_str.lower()) or max_v >= 3950 or min_v <= 140
            gain_low = "gain too low" in err_str.lower()

            if clipped or excursion > target_excursion:
                hi_db = mid_db
            elif gain_low or excursion < target_excursion:
                best_db = float(self.scope.gain.db)
                lo_db = mid_db
            else:
                best_db = float(self.scope.gain.db)
                break

        self.scope.gain.db = best_db
        if self.trace_enabled:
            self._configure_trace_for_aes_ctr()
            _ = trigger_fn()

        for _ in range(4):
            trigger_fn = _prep_cal_vector()
            self.scope.adc.errors = False
            if self.trace_enabled:
                self.scope.trace.resync()
                self.scope.trace.errors = 0
                self.scope.trace.arm_trace()
            self.scope.arm()
            _ = trigger_fn()
            self.scope.capture()
            trig_count = max(trig_count, int(self.scope.adc.trig_count))
            raw_optimal = trig_count + jitter_margin - offset + presamples
            if offset == 0:
                raw_optimal = max(raw_optimal, 78400 + 18800 * (eff_blocks - 1))
            optimal_samples = min(max(64, raw_optimal), hw_max)
            self.scope.adc.samples = optimal_samples
            tr_int = self.scope.get_last_trace(as_int=True)
            min_v = int(tr_int.min())
            max_v = int(tr_int.max())
            err_str = str(self.scope.adc.errors or "")
            if ("clipped" in err_str.lower()) or max_v >= 3900 or min_v <= 180:
                self.scope.gain.db = max(25.0, float(self.scope.gain.db) - 1.0)
            else:
                break

        return int(self.scope.adc.samples), round(float(self.scope.gain.db), 4)

    def _is_trace_clipped(self, trace_int_np: np.ndarray) -> tuple[bool, str]:
        adc_err_str = str(self.scope.adc.errors or "")
        min_val = int(trace_int_np.min()) if trace_int_np.size > 0 else 0
        max_val = int(trace_int_np.max()) if trace_int_np.size > 0 else 0
        if "clipped" in adc_err_str.lower():
            return True, f"scope.adc.errors={adc_err_str!r} (min={min_val}, max={max_val})"
        if trace_int_np.size > 0 and (min_val <= 0 or max_val >= 4095):
            return True, f"ADC rail reached (min={min_val}, max={max_val})"
        return False, ""

    def _read_tpiu_bytes_from_fifo(
        self,
        trace: Any,
        chunk_words: int = 8192,
    ) -> tuple[int, bytes]:
        """Read TraceWhisperer FIFO and extract raw TPIU bytes without per-entry list allocations."""
        if getattr(trace, "platform", None) != "Husky":
            raw_data = trace.read_capture_data()
            packets = trace.get_raw_trace_packets(raw_data, removesyncs=False, verbose=False)
            return len(raw_data), b"".join(bytes(p[1]) for p in packets)

        if trace.fifo_empty():
            raise RuntimeError("TraceWhisperer FIFO is empty after capture.")

        chunk_bytes = chunk_words * 4
        underflow_bit = 1 << int(trace.FE_FIFO_STAT_UNDERFLOW)
        cmd_stat = int(trace.FE_FIFO_CMD_STAT)
        out_chunks: list[bytes] = []
        raw_fifo_entries = 0

        while not trace.fifo_empty():
            raw_chunk = bytes(trace.fpga_read(trace.REG_SNIFF_FIFO_RD, chunk_bytes))
            arr = np.frombuffer(raw_chunk, dtype=np.uint8).reshape(-1, 4)
            uf_idx = np.nonzero(arr[:, 3] & underflow_bit)[0]
            if uf_idx.size > 0:
                arr = arr[: int(uf_idx[0])]
            raw_fifo_entries += int(arr.shape[0])
            mask = (arr[:, 3] & 0x3) == cmd_stat
            out_chunks.append(arr[mask, 2].tobytes())
            if uf_idx.size > 0:
                break

        return raw_fifo_entries, b"".join(out_chunks)

    def _read_raw_trace_fifo(self) -> tuple[int, bytes, Any]:
        trace = self.scope.trace
        fifo_errors = trace.errors
        raw_fifo_entries, raw_bytes = self._read_tpiu_bytes_from_fifo(trace, chunk_words=8192)
        if not fifo_errors:
            trace.errors = 0
        return raw_fifo_entries, raw_bytes, fifo_errors

    def _capture_single_raw(
        self,
        shared_key: bytes,
        iv: bytes,
        iv_salt: bytes,
        plaintext: bytes,
        max_clip_retries: int = 10,
    ) -> tuple[np.ndarray, bytes, dict[str, Any] | None]:
        """Execute one physical capture with automatic clipping retry and raw TraceWhisperer FIFO read."""
        self._ensure_connected()
        trace_config_summary: dict[str, Any] | None = None
        if self.trace_enabled or bool(self.scope.trace.enabled):
            self.trace_enabled = True
            if self._last_trace_config_summary is not None:
                trace_config_summary = self._last_trace_config_summary
            else:
                trace_config_summary = self._configure_trace_for_aes_ctr()

        clip_retries = 0
        max_attempts = max(1, int(max_clip_retries) + 1)
        trace_int_np: np.ndarray | None = None
        output_bytes = b""

        for attempt in range(max_attempts):
            trigger_fn = self._prepare_and_trigger_aes_ctr(
                shared_key=shared_key,
                iv=iv,
                iv_salt=iv_salt,
                plaintext=plaintext,
                nb_blocks=self.nb_blocks,
            )

            self.scope.adc.errors = False
            if self.trace_enabled:
                self.scope.trace.resync()
                self.scope.trace.errors = 0
                self.scope.trace.arm_trace()

            self.scope.arm()
            output_bytes = trigger_fn()
            if self.scope.capture():
                raise RuntimeError("ChipWhisperer Husky capture timed out waiting for trigger.")

            try:
                trace_int_np = self.scope.get_last_trace(as_int=True)
            except BaseException:
                try:
                    self.scope.adc.oa.setFastSMC(0)
                except Exception:
                    pass
                raise

            is_clipped, clip_reason = self._is_trace_clipped(trace_int_np)
            if not is_clipped:
                break

            if attempt + 1 < max_attempts:
                clip_retries += 1
                self.scope.adc.errors = False
                cur_gain = float(self.scope.gain.db)
                if cur_gain > -10.0:
                    self.scope.gain.db = max(-10.0, round(cur_gain - 0.5, 2))
            else:
                raise RuntimeError(
                    f"Captured ADC trace is still clipped after {clip_retries} retries "
                    f"with the same inputs ({clip_reason})."
                )

        if trace_int_np is None:
            raise RuntimeError("No ADC trace captured.")

        raw_trace_fifo: dict[str, Any] | None = None
        if self.trace_enabled and trace_config_summary is not None:
            raw_fifo_entries, raw_bytes, fifo_errors = self._read_raw_trace_fifo()
            adc_offset = int(self.scope.adc.offset) if self.scope is not None else 0
            raw_trace_fifo = {
                "raw_fifo_entries": raw_fifo_entries,
                "raw_bytes": raw_bytes,
                "fifo_errors": fifo_errors,
                "config_summary": trace_config_summary,
                "nb_blocks": self.nb_blocks,
                "adc_offset": adc_offset,
            }

        return trace_int_np, output_bytes, raw_trace_fifo

    def decode_raw_trace_bytes(
        self,
        raw_fifo_entries: int,
        raw_bytes: bytes,
        fifo_errors: Any,
        config_summary: dict[str, Any],
        nb_blocks: int | None = None,
        adc_offset: int = 0,
        adc_values: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Decode raw TPIU FIFO bytes into Stream 1 (ITM/DWT), Stream 2 (ETMv4), and AES reconstruction.

        Pure CPU work safe for execution in background worker threads (never touches `self.scope`).
        """
        if not raw_bytes:
            return {
                "configuration": config_summary,
                "raw_fifo_entries": raw_fifo_entries,
                "raw_tpiu_bytes": 0,
                "fifo_errors": fifo_errors,
                "stream1_itm_dwt": {"messages": [], "packet_counts": {}},
                "stream2_etmv4": {"events": [], "change_counts": {}},
            }

        import pyorb

        decoder, phase_k = decode_tpiu_with_frame_phase(raw_bytes, initial_stream=2)
        s1 = bytes(decoder.packets.get_stream(1) or b"")
        s2 = bytes(decoder.packets.get_stream(2) or b"")

        comp_symbols: dict[int, str] = {}
        for comp_key, info in config_summary.get("comparators", {}).items():
            if comp_key.startswith("COMP"):
                comp_symbols[int(comp_key[4:])] = info["symbol"]
        pc_to_sym = {addr: name for name, addr in self.func_addrs.items()}

        itm_msgs = decoder.get_itm_messages()
        type_counts = Counter(m.__class__.__name__ for m in itm_msgs)
        comp_counts = Counter(
            int(m.comp) for m in itm_msgs if isinstance(m, (pyorb.watchMsg, pyorb.wptMsg))
        )

        serialized_itm: list[dict[str, Any]] = []
        for m in itm_msgs:
            mtype = m.__class__.__name__
            item: dict[str, Any] = {"type": mtype}
            if isinstance(m, pyorb.watchMsg):
                comp = int(m.comp)
                data_val = int(m.data) & 0xFFFFFFFF
                item.update({
                    "comparator": comp,
                    "symbol": comp_symbols.get(comp),
                    "is_write": bool(m.isWrite),
                    "data_int": data_val,
                    "data_hex": f"0x{data_val:08x}",
                })
            elif isinstance(m, pyorb.wptMsg):
                comp = int(m.comp)
                pc_val = int(m.data) & 0xFFFFFFFF
                item.update({
                    "comparator": comp,
                    "symbol": comp_symbols.get(comp) or pc_to_sym.get(pc_val),
                    "pc_int": pc_val,
                    "pc_hex": f"0x{pc_val:08x}",
                })
            elif isinstance(m, pyorb.nisyncMsg):
                addr_val = int(m.addr) & 0xFFFFFFFF
                item.update({
                    "sync_type": int(m.type),
                    "addr_int": addr_val,
                    "addr_hex": f"0x{addr_val:08x}",
                    "symbol": pc_to_sym.get(addr_val),
                })
            elif isinstance(m, pyorb.pcSampleMsg):
                pc_val = int(m.pc) & 0xFFFFFFFF
                item.update({
                    "sleep": bool(m.sleep),
                    "pc_int": pc_val,
                    "pc_hex": f"0x{pc_val:08x}",
                    "symbol": pc_to_sym.get(pc_val),
                })
            elif isinstance(m, pyorb.TSMsg):
                item.update({
                    "time_inc": int(m.timeInc),
                    "time_status": int(getattr(m, "timeStatus", getattr(m, "rr", 0))),
                })
            elif isinstance(m, pyorb.swMsg):
                val = int(m.value) & 0xFFFFFFFF
                item.update({
                    "src_port": int(m.srcAddr),
                    "length": int(m.len),
                    "value_int": val,
                    "value_hex": f"0x{val:08x}",
                })
            serialized_itm.append(item)

        etm_res = decode_etm4_stream(s2) if s2 else {
            "change_counts": {},
            "events": [],
            "addresses_seen": [],
            "total_eatoms": 0,
            "total_natoms": 0,
            "total_cycles": 0,
        }
        uniq_branch_targets: list[str] = []
        for addr in etm_res["addresses_seen"]:
            hx = f"0x{addr:08x}"
            if not uniq_branch_targets or uniq_branch_targets[-1] != hx:
                uniq_branch_targets.append(hx)

        serialized_etm_events = []
        for ev in etm_res["events"]:
            addr_int = int(ev["addr"]) & 0xFFFFFFFF
            serialized_etm_events.append({
                "pkt_type": ev.get("pkt_type"),
                "changes": ev["changes"],
                "addr_int": addr_int,
                "addr_hex": f"0x{addr_int:08x}",
                "symbol": pc_to_sym.get(addr_int & ~1),
                "eatoms": int(ev["eatoms"]),
                "natoms": int(ev["natoms"]),
                "disposition": int(ev["disposition"]),
                "cycle_count": (
                    int(ev["cycleCount"])
                    if ev["cycleCount"] != 0xFFFFFFFFFFFFFFFF
                    else None
                ),
                "cumulative_cycles": int(ev.get("cumulative_cycles", 0)),
            })

        decoded_tw: dict[str, Any] = {
            "configuration": config_summary,
            "raw_fifo_entries": raw_fifo_entries,
            "raw_tpiu_bytes": len(raw_bytes),
            "fifo_errors": fifo_errors,
            "tpiu_frame_phase": phase_k,
            "tpiu_stream_ids": list(decoder.packets.get_stream_ids()),
            "stream1_itm_dwt": {
                "byte_length": len(s1),
                "raw_hex": s1.hex(),
                "total_messages": len(serialized_itm),
                "packet_counts_by_type": dict(type_counts),
                "dwt_hits_by_comparator": {
                    f"COMP{k} ({comp_symbols.get(k, 'unknown')})": v
                    for k, v in sorted(comp_counts.items())
                },
                "messages": serialized_itm,
            },
            "stream2_etmv4": {
                "byte_length": len(s2),
                "raw_hex": s2.hex(),
                "change_counts": etm_res["change_counts"],
                "total_eatoms": etm_res["total_eatoms"],
                "total_natoms": etm_res["total_natoms"],
                "total_cycles": etm_res["total_cycles"],
                "unique_branch_targets": uniq_branch_targets,
                "events": serialized_etm_events,
            },
        }

        if config_summary.get("implementation") == "aes_ctr_asm":
            try:
                decoded_tw["protected_aes_reconstruction"] = analyze_protected_aes_capture(
                    decoded_tw,
                    nb_blocks=nb_blocks,
                    adc_offset=int(adc_offset),
                    adc_values=adc_values,
                )
            except Exception as exc:
                decoded_tw["protected_aes_reconstruction_error"] = str(exc)

        return decoded_tw

    def connect(self) -> dict[str, Any]:
        """Connect to Husky scope and RP2350 target, and calibrate capture parameters."""
        self._interrupt_guard.install()
        with self._interrupt_guard.atomic_hardware_op():
            conn = self._connect_scope_and_target()
            self.conn_info = conn
            self._last_warmed_key = None

        print(
            f"[+] Connected to {conn['scope_name']} (serial: {conn['scope_sn']}, "
            f"FW: {conn.get('fw_version')}, CW: {conn.get('cw_version')}, "
            f"max_samples: {conn.get('hw_max_samples')})"
        )

        with self._interrupt_guard.atomic_hardware_op():
            self.set_tracewhisperer(enabled=True)
            try:
                opt_samples, opt_gain = self._calibrate()
                self.calibrated_samples = opt_samples
                self.calibrated_gain = opt_gain
                self.scope.adc.samples = self.calibrated_samples
                self.scope.gain.db = self.calibrated_gain
                print(
                    f"[+] Hardware calibrated: Samples={self.calibrated_samples}, "
                    f"Gain={self.calibrated_gain} dB"
                )
            except Exception as exc:
                print(
                    f"[!] Calibration warning: {exc}. Using defaults: "
                    f"samples={self.calibrated_samples}, gain={self.calibrated_gain} dB"
                )
                self.scope.adc.samples = self.calibrated_samples
                self.scope.gain.db = self.calibrated_gain

            self._configure_trace_for_aes_ctr()

        import gc

        gc.set_threshold(50000, 50, 50)
        return conn

    def warmup_key(
        self,
        key: bytes,
        include_tw: bool = True,
        num_warmup: int | None = None,
        iv_override: bytes | None = None,
        iv_salt_override: bytes | None = None,
    ) -> int:
        """Run unstored warmup captures for `key` to warm up the RP2350 16 KB XIP flash cache."""
        self._ensure_connected()
        n_warm = self.warmup_captures_per_key if num_warmup is None else max(0, int(num_warmup))
        if n_warm <= 0:
            self._last_warmed_key = bytes(key)
            return 0

        with self._interrupt_guard.atomic_hardware_op():
            if self.trace_enabled != include_tw:
                self.set_tracewhisperer(enabled=include_tw)
                if include_tw:
                    self._configure_trace_for_aes_ctr()
            if int(self.scope.adc.samples) != self.calibrated_samples:
                self.scope.adc.samples = self.calibrated_samples

        completed = 0
        for _ in range(n_warm):
            sa, sb, sc, sd = split_key_4way(key)
            shared_k = _concatenate_shares(sa.tobytes(), sb.tobytes(), sc.tobytes(), sd.tobytes())
            pt = os.urandom(16 * self.nb_blocks)
            iv = iv_override if iv_override is not None else os.urandom(16)
            iv_salt = iv_salt_override if iv_salt_override is not None else os.urandom(16)
            try:
                with self._interrupt_guard.atomic_hardware_op():
                    self._capture_single_raw(
                        shared_key=shared_k,
                        iv=iv,
                        iv_salt=iv_salt,
                        plaintext=pt,
                        max_clip_retries=3,
                    )
                completed += 1
            except KeyboardInterrupt:
                raise
            except Exception:
                try:
                    with self._interrupt_guard.atomic_hardware_op():
                        self.reset_target(sync_clock_24mhz=True, reenable_trace=include_tw)
                        if include_tw:
                            self._configure_trace_for_aes_ctr()
                        self.scope.adc.samples = self.calibrated_samples
                except KeyboardInterrupt:
                    raise
                except Exception:
                    pass
                time.sleep(0.05)

        self._last_warmed_key = bytes(key)
        return completed

    def capture_raw_bundle(
        self,
        key_override: bytes | None = None,
        include_tw: bool = True,
        max_retries: int = 5,
        minimal_example: bool = False,
        iv_override: bytes | None = None,
        iv_salt_override: bytes | None = None,
    ) -> dict[str, Any]:
        """Perform ONLY the physical USB/UART capture on the hardware thread."""
        self._ensure_connected()
        if minimal_example:
            include_tw = False

        with self._interrupt_guard.atomic_hardware_op():
            if self.trace_enabled != include_tw:
                self.set_tracewhisperer(enabled=include_tw)
                if include_tw:
                    self._configure_trace_for_aes_ctr()
            if int(self.scope.adc.samples) != self.calibrated_samples:
                self.scope.adc.samples = self.calibrated_samples

        key = key_override if key_override is not None else os.urandom(32)
        if self.warmup_captures_per_key > 0 and self._last_warmed_key != key:
            self.warmup_key(
                key,
                include_tw=include_tw,
                iv_override=iv_override,
                iv_salt_override=iv_salt_override,
            )

        sa, sb, sc, sd = split_key_4way(key)
        shared_k = _concatenate_shares(sa.tobytes(), sb.tobytes(), sc.tobytes(), sd.tobytes())
        pt = os.urandom(16 * self.nb_blocks)
        iv = iv_override if iv_override is not None else os.urandom(16)
        iv_salt = iv_salt_override if iv_salt_override is not None else os.urandom(16)
        if len(iv) != 16:
            raise ValueError(f"IV must be 16 bytes, got {len(iv)} bytes.")
        if len(iv_salt) != 16:
            raise ValueError(f"IV salt must be 16 bytes, got {len(iv_salt)} bytes.")

        last_err: Exception | None = None
        raw_adc: np.ndarray | None = None
        ct_bytes: bytes = b""
        raw_trace_fifo: dict[str, Any] | None = None

        for attempt in range(max(1, max_retries)):
            try:
                with self._interrupt_guard.atomic_hardware_op():
                    raw_adc, ct_bytes, raw_trace_fifo = self._capture_single_raw(
                        shared_key=shared_k,
                        iv=iv,
                        iv_salt=iv_salt,
                        plaintext=pt,
                        max_clip_retries=10,
                    )

                if raw_adc.size < self.calibrated_samples:
                    raise RuntimeError(
                        f"Incomplete ADC trace: received {raw_adc.size} samples, "
                        f"expected {self.calibrated_samples}"
                    )
                break
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                last_err = exc
                print(
                    f"[!] Capture attempt {attempt + 1}/{max_retries} failed ({exc}); "
                    "resetting target..."
                )
                try:
                    with self._interrupt_guard.atomic_hardware_op():
                        self.reset_target(sync_clock_24mhz=True, reenable_trace=include_tw)
                        if include_tw:
                            self._configure_trace_for_aes_ctr()
                        self.scope.adc.samples = self.calibrated_samples
                    if self.warmup_captures_per_key > 0:
                        self.warmup_key(
                            key,
                            include_tw=include_tw,
                            iv_override=iv_override,
                            iv_salt_override=iv_salt_override,
                        )
                except KeyboardInterrupt:
                    raise
                except Exception:
                    pass
                time.sleep(0.05)
        else:
            raise RuntimeError(
                f"Hardware trace capture failed after {max_retries} attempts: {last_err}"
            ) from last_err

        if raw_adc is None:
            raise RuntimeError("Hardware capture failed to produce an ADC waveform.")

        return {
            "raw_adc": raw_adc,
            "calibrated_samples": self.calibrated_samples,
            "nb_blocks": self.nb_blocks,
            "include_tw": include_tw,
            "minimal_example": bool(minimal_example),
            "key": key,
            "sa": sa,
            "sb": sb,
            "sc": sc,
            "sd": sd,
            "pt": pt,
            "iv": iv,
            "iv_salt": iv_salt,
            "ct_bytes": ct_bytes,
            "raw_trace_fifo": raw_trace_fifo,
        }

    def postprocess_raw_bundle(self, raw_bundle: dict[str, Any]) -> dict[str, Any]:
        """Perform all post-capture CPU computations on `raw_bundle` in a background thread."""
        raw_adc: np.ndarray = raw_bundle["raw_adc"]
        calibrated_samples: int = int(raw_bundle["calibrated_samples"])
        nb_blocks: int = int(raw_bundle["nb_blocks"])
        include_tw: bool = bool(raw_bundle["include_tw"])
        minimal_example: bool = bool(raw_bundle.get("minimal_example", False))
        key: bytes = raw_bundle["key"]
        sa: np.ndarray = raw_bundle["sa"]
        sb: np.ndarray = raw_bundle["sb"]
        sc: np.ndarray = raw_bundle["sc"]
        sd: np.ndarray = raw_bundle["sd"]
        pt: bytes = raw_bundle["pt"]
        iv: bytes = raw_bundle["iv"]
        iv_salt: bytes = raw_bundle["iv_salt"]
        ct_bytes: bytes = raw_bundle["ct_bytes"]

        expected_ct = compute_aes_ctr_output(key=key, iv=iv, iv_salt=iv_salt, plaintext=pt)
        if ct_bytes != expected_ct:
            raise RuntimeError(
                f"Target ciphertext mismatch: got {ct_bytes.hex()}, expected {expected_ct.hex()}"
            )

        trace1 = raw_adc[:calibrated_samples].astype(np.int16)

        if minimal_example:
            return build_minimal_holdout_example(
                trace1=trace1,
                key=key,
                iv=iv,
                iv_salt=iv_salt,
                num_samples=calibrated_samples,
                nb_blocks=nb_blocks,
            )

        round_keys = expand_aes256_round_keys(key)
        intermediate_shares = compute_intermediate_shares(
            sa,
            sb,
            sc,
            sd,
            np.frombuffer(iv, dtype=np.uint8),
            np.frombuffer(iv_salt, dtype=np.uint8),
        )

        tw_dict: dict[str, Any] | None = None
        tw_str = ""
        if include_tw:
            raw_fifo = raw_bundle.get("raw_trace_fifo")
            if isinstance(raw_fifo, dict):
                tw_dict = self.decode_raw_trace_bytes(
                    raw_fifo_entries=int(raw_fifo["raw_fifo_entries"]),
                    raw_bytes=raw_fifo["raw_bytes"],
                    fifo_errors=raw_fifo["fifo_errors"],
                    config_summary=raw_fifo["config_summary"],
                    nb_blocks=int(raw_fifo.get("nb_blocks", nb_blocks)),
                    adc_offset=int(raw_fifo.get("adc_offset", 0)),
                    adc_values=trace1,
                )
                tw_str = json.dumps(tw_dict)

        trace2 = compute_trace2_from_trace1(
            trace1,
            tw_dict if tw_dict is not None else tw_str,
            nb_blocks=nb_blocks,
            target_length=calibrated_samples,
        )

        return {
            "trace1": trace1,
            "trace2": trace2,
            "plaintext": np.frombuffer(pt, dtype=np.uint8),
            "ciphertext": np.frombuffer(ct_bytes, dtype=np.uint8),
            "iv": np.frombuffer(iv, dtype=np.uint8),
            "iv_salt": np.frombuffer(iv_salt, dtype=np.uint8),
            "key": np.frombuffer(key, dtype=np.uint8),
            "key_a": sa,
            "key_b": sb,
            "key_c": sc,
            "key_d": sd,
            "round_keys": round_keys,
            "intermediate_shares": intermediate_shares,
            "tracewhisperer": tw_str,
        }

    def _disconnect_hardware(self) -> None:
        if self.scope is None and self.target is None:
            self.connected = False
            return

        try:
            if self.scope is not None:
                try:
                    self.scope.adc.oa.setFastSMC(0)
                except Exception:
                    pass
                try:
                    self.scope.sc.arm(False)
                except Exception:
                    pass
                try:
                    self.scope.adc.errors = False
                except Exception:
                    pass

            if self.target is not None:
                target_ok = False
                try:
                    self.target.reset_comms()
                    self.target.flush()
                    self.target.send_cmd(ord("T"), 0x00, bytearray())
                    self.target.simpleserial_wait_ack(250)
                    self.target.send_cmd(ord("T"), 0x04, bytearray([0]))
                    self.target.simpleserial_wait_ack(250)
                    target_ok = True
                except Exception:
                    target_ok = False

                if not target_ok and self.scope is not None:
                    try:
                        self.scope.io.nrst = "low"
                        time.sleep(0.05)
                        self.scope.io.nrst = "high_z"
                        time.sleep(0.58)
                        self.target.reset_comms()
                        self.target.flush()
                        self._sync_target_clock(force=True)
                    except Exception:
                        pass
        finally:
            if self.target is not None:
                try:
                    self.target.dis()
                except Exception:
                    pass
                self.target = None

            if self.scope is not None:
                try:
                    if not self.scope.trace.fifo_empty():
                        try:
                            self._read_tpiu_bytes_from_fifo(self.scope.trace, chunk_words=8192)
                        except Exception:
                            pass
                    self.scope.trace.errors = 0
                except Exception:
                    pass
                try:
                    self.scope.trace.enabled = False
                    self.scope.userio.mode = "normal"
                    self.scope.userio.direction = 0
                    self.scope.userio.drive_data = 0
                    if self.scope.LA.enabled:
                        self.scope.LA.enabled = False
                except Exception:
                    pass
                try:
                    self.scope.dis()
                except Exception:
                    pass
                self.scope = None

            self.trace_enabled = False
            self.connected = False

    def disconnect(self) -> None:
        """Safely disconnect scope, disarm ADC/TraceWhisperer, and restore normal USERIO pins."""
        try:
            self._disconnect_hardware()
        finally:
            self._interrupt_guard.uninstall()


# ==============================================================================
# 6. Core Dataset Generation Pipeline & Unified Progress Reporting
# ==============================================================================

class _AsyncCapturePipeline:
    """Offloads post-capture CPU computation and `dataset_filler.write_example` calls to
    background threads while preserving strict FIFO example ordering per split.
    """

    def __init__(
        self,
        dataset_filler: Any,
        hw_capturer: HardwareTraceCapturer | None,
        num_samples: int,
        nb_blocks: int,
        max_pending: int = 32,
        postprocess_workers: int = 2,
    ) -> None:
        self.dataset_filler = dataset_filler
        self.hw_capturer = hw_capturer
        self.num_samples = int(num_samples)
        self.nb_blocks = int(nb_blocks)
        self._write_queue: queue.Queue[tuple[concurrent.futures.Future[dict[str, Any]], SplitT] | None] = (
            queue.Queue(maxsize=max(2, int(max_pending)))
        )
        self._postprocess_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, int(postprocess_workers)),
            thread_name_prefix="sedpack-postproc",
        )
        self._error_lock = threading.Lock()
        self._first_error: BaseException | None = None
        self._stop_event = threading.Event()
        self._closed = False
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="sedpack-dataset-writer",
            daemon=True,
        )
        self._writer_thread.start()

    def _record_error(self, exc: BaseException) -> None:
        with self._error_lock:
            if self._first_error is None:
                self._first_error = exc
        self._stop_event.set()

    def _check_error(self) -> None:
        with self._error_lock:
            err = self._first_error
        if err is not None:
            raise RuntimeError(f"Background post-processing / dataset writer failed: {err}") from err

    def _writer_loop(self) -> None:
        while True:
            item = self._write_queue.get()
            if item is None:
                self._write_queue.task_done()
                break
            fut, split = item
            try:
                if not self._stop_event.is_set():
                    ex = fut.result()
                    self.dataset_filler.write_example(values=ex, split=split)
            except BaseException as exc:
                self._record_error(exc)
            finally:
                self._write_queue.task_done()

    def capture_and_enqueue(
        self,
        key_override: bytes,
        include_tw: bool,
        split: SplitT,
        minimal_example: bool = False,
        iv_override: bytes | None = None,
        iv_salt_override: bytes | None = None,
    ) -> None:
        """Capture raw trace on the calling thread and enqueue post-processing + writing."""
        self._check_error()
        if self.hw_capturer is not None:
            raw_bundle = self.hw_capturer.capture_raw_bundle(
                key_override=key_override,
                include_tw=include_tw,
                minimal_example=minimal_example,
                iv_override=iv_override,
                iv_salt_override=iv_salt_override,
            )
            fut = self._postprocess_executor.submit(
                self.hw_capturer.postprocess_raw_bundle,
                raw_bundle,
            )
        else:
            fut = self._postprocess_executor.submit(
                generate_mock_example,
                self.num_samples,
                key_override,
                include_tw,
                self.nb_blocks,
                minimal_example,
                iv_override,
                iv_salt_override,
            )

        while True:
            self._check_error()
            try:
                self._write_queue.put((fut, split), timeout=0.1)
                break
            except queue.Full:
                continue

    def flush(self) -> None:
        """Wait for all currently queued post-processing and dataset write tasks to finish."""
        self._write_queue.join()
        self._check_error()

    def close(self, abort: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        if abort:
            self._stop_event.set()
            while True:
                try:
                    item = self._write_queue.get_nowait()
                    if item is not None:
                        item[0].cancel()
                    self._write_queue.task_done()
                except queue.Empty:
                    break
            self._write_queue.put(None)
            self._writer_thread.join(timeout=2.0)
            self._postprocess_executor.shutdown(wait=False, cancel_futures=True)
        else:
            try:
                self._write_queue.join()
            finally:
                self._write_queue.put(None)
                self._writer_thread.join(timeout=10.0)
                self._postprocess_executor.shutdown(wait=True)
            self._check_error()

    def __enter__(self) -> _AsyncCapturePipeline:
        return self

    def __exit__(self, exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        try:
            self.close(abort=exc_type is not None)
        finally:
            if exc_type is not None and self.hw_capturer is not None:
                try:
                    self.hw_capturer.disconnect()
                except Exception:
                    pass


def _should_use_progress_bar(progress_mode: str = "auto") -> bool:
    """Return True when running in an interactive user terminal, False for agents/pipes."""
    mode = (progress_mode or "auto").strip().lower()
    if mode == "bar":
        return True
    if mode == "text":
        return False

    if not sys.stdout.isatty():
        return False

    term = os.environ.get("TERM", "").strip().lower()
    if term in ("", "dumb"):
        return False

    agent_env_vars = (
        "ANTIGRAVITY_AGENT",
        "ANTIGRAVITY_SOURCE_METADATA",
        "CLAUDECODE",
        "CLAUDE_CODE",
        "GEMINI_CLI",
        "CURSOR_AGENT",
        "CODEX_CI",
        "AIDER",
        "CI",
    )
    return not any(os.environ.get(var) for var in agent_env_vars)


class _CaptureProgressReporter:
    """Unified progress reporter supporting Rich multi-bar, `tqdm` fallback, and plain-text logging."""

    def __init__(
        self,
        *,
        use_bar: bool,
        progress_mode: str,
        append: bool,
        total_traces: int,
        num_train: int,
        num_train_keys: int,
        traces_per_train_key: int,
        num_val: int,
        total_holdout: int,
        num_holdout_keys: int,
        traces_per_holdout_key: int,
        holdout_minimal: bool,
        examples_per_shard: int,
        t_start: float,
    ) -> None:
        self.append = append
        self.total_traces = total_traces
        self.num_train = num_train
        self.num_train_keys = num_train_keys
        self.traces_per_train_key = traces_per_train_key
        self.num_val = num_val
        self.total_holdout = total_holdout
        self.num_holdout_keys = num_holdout_keys
        self.traces_per_holdout_key = traces_per_holdout_key
        self.holdout_minimal = holdout_minimal
        self.examples_per_shard = max(1, examples_per_shard)
        self.t_start = t_start

        self.captured_count = 0
        self._t_split = t_start
        self._split_done = 0

        self._backend = "text"
        self._stack = contextlib.ExitStack()
        self._rich_progress: Any = None
        self._rich_tasks: dict[str, Any] = {}
        self._tqdm_cls: Any = None
        self._tqdm_total: Any = None
        self._tqdm_split: Any = None

        if use_bar:
            try:
                from rich.console import Console
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

                console = Console(force_terminal=True if progress_mode == "bar" else None)
                self._rich_progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[bold cyan]{task.description:<26}"),
                    BarColumn(bar_width=None, complete_style="cyan", finished_style="green"),
                    MofNCompleteColumn(),
                    TaskProgressColumn(),
                    TextColumn("[bold yellow]{task.fields[rate]:>10}[/bold yellow]"),
                    TimeElapsedColumn(),
                    TextColumn("ETA"),
                    TimeRemainingColumn(),
                    console=console,
                    refresh_per_second=10,
                    expand=True,
                )
                self._backend = "rich"
            except ImportError:
                try:
                    from tqdm import tqdm

                    self._tqdm_cls = tqdm
                    self._backend = "tqdm"
                except ImportError:
                    self._backend = "text"

    def __enter__(self) -> _CaptureProgressReporter:
        if self._backend == "rich":
            self._stack.enter_context(self._rich_progress)
            self._rich_tasks["overall"] = self._rich_progress.add_task(
                "Overall Append" if self.append else "Overall Dataset",
                total=self.total_traces,
                rate="0.00 tr/s",
            )
            if self.num_train > 0:
                self._rich_tasks["train"] = self._rich_progress.add_task(
                    f"Train Split (0/{self.num_train_keys} keys)",
                    total=self.num_train,
                    rate="0.00 tr/s",
                )
            if self.num_val > 0:
                self._rich_tasks["val"] = self._rich_progress.add_task(
                    "Validation Split (TW=ON)",
                    total=self.num_val,
                    rate="pending",
                )
            if self.total_holdout > 0:
                self._rich_tasks["holdout"] = self._rich_progress.add_task(
                    f"Holdout Split (0/{self.num_holdout_keys} keys)",
                    total=self.total_holdout,
                    rate="pending",
                )
        elif self._backend == "tqdm":
            overall_desc = "Overall Append" if self.append else "Overall Dataset"
            self._tqdm_total = self._stack.enter_context(
                self._tqdm_cls(total=self.total_traces, desc=overall_desc, unit="tr")
            )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._close_tqdm_split()
        self._stack.__exit__(exc_type, exc_val, exc_tb)

    def _close_tqdm_split(self) -> None:
        if self._tqdm_split is not None:
            self._tqdm_split.close()
            self._tqdm_split = None

    @contextlib.contextmanager
    def split_scope(self, split_id: str) -> Iterator[None]:
        self._t_split = time.monotonic()
        self._split_done = 0
        action_verb = "Appending to" if self.append else "Capturing"

        if self._backend == "tqdm":
            if split_id == "train":
                desc, total = "Train Split (TW=ON)", self.num_train
            elif split_id == "val":
                desc, total = "Validation Split (TW=ON)", self.num_val
            else:
                desc = "Holdout Split (Minimal)" if self.holdout_minimal else "Holdout Split (TW=ON)"
                total = self.total_holdout
            self._tqdm_split = self._tqdm_cls(total=total, desc=desc, unit="tr", leave=True)
        elif self._backend == "text":
            if split_id == "train":
                print(
                    f"\n[+] {action_verb} Training Split ('train', {self.num_train_keys} keys x "
                    f"{self.traces_per_train_key} tr/key = {self.num_train} traces)..."
                )
            elif split_id == "val":
                print(f"\n[+] {action_verb} Validation Split ('test' / val, {self.num_val} traces)...")
            elif split_id == "holdout":
                ho_tag = " [MINIMAL: trace1, key, iv, iv_salt ONLY]" if self.holdout_minimal else ""
                print(
                    f"\n[+] {action_verb} Holdout Split ('holdout', {self.num_holdout_keys} keys x "
                    f"{self.traces_per_holdout_key} tr/key = {self.total_holdout} traces){ho_tag}..."
                )

        try:
            yield
        finally:
            self._close_tqdm_split()

    def set_key(self, split_id: str, key_num: int, total_keys: int) -> None:
        if self._backend == "rich" and split_id in self._rich_tasks:
            label = "Train Split" if split_id == "train" else "Holdout Split"
            self._rich_progress.update(
                self._rich_tasks[split_id],
                description=f"{label} (key {key_num}/{total_keys})",
            )
        elif self._backend == "tqdm" and self._tqdm_split is not None:
            self._tqdm_split.set_postfix_str(f"key {key_num}/{total_keys}")

    def advance(self, split_id: str, key_num: int | None = None) -> None:
        self.captured_count += 1
        self._split_done += 1
        now = time.monotonic()
        overall_rate_val = self.captured_count / max(1e-6, now - self.t_start)

        if self._backend == "rich":
            overall_rate = f"{overall_rate_val:.2f} tr/s"
            split_rate = f"{self._split_done / max(1e-6, now - self._t_split):.2f} tr/s"
            self._rich_progress.update(self._rich_tasks["overall"], advance=1, rate=overall_rate)
            if split_id in self._rich_tasks:
                self._rich_progress.update(self._rich_tasks[split_id], advance=1, rate=split_rate)
        elif self._backend == "tqdm":
            if self._tqdm_split is not None:
                self._tqdm_split.update(1)
            if self._tqdm_total is not None:
                self._tqdm_total.update(1)
        else:
            if split_id == "train":
                if (
                    self._split_done % self.examples_per_shard == 0
                    or self._split_done == self.num_train
                ):
                    print(
                        f"  -> Train: key {key_num}/{self.num_train_keys} "
                        f"({self._split_done}/{self.num_train} traces captured, {overall_rate_val:.2f} tr/s)"
                    )
            elif split_id == "val":
                if (
                    self._split_done % self.examples_per_shard == 0
                    or self._split_done == self.num_val
                ):
                    print(
                        f"  -> Val: {self._split_done}/{self.num_val} traces captured "
                        f"({overall_rate_val:.2f} tr/s)"
                    )
            elif split_id == "holdout" and key_num is not None:
                if self._split_done % max(1, self.traces_per_holdout_key) == 0:
                    step = max(1, self.num_holdout_keys // 10)
                    if key_num % step == 0 or key_num == self.num_holdout_keys:
                        print(
                            f"  -> Holdout: key {key_num}/{self.num_holdout_keys} complete "
                            f"({self._split_done}/{self.total_holdout} traces, {overall_rate_val:.2f} tr/s)"
                        )


def generate_sedpack_dataset(
    dataset_path: str | Path,
    num_train_keys: int = 100,
    traces_per_train_key: int = 64,
    val_ratio: float = 0.10,
    num_holdout_keys: int = 100,
    traces_per_holdout_key: int = 768,
    mode: str = "mock",
    num_samples: int = 78400,
    examples_per_shard: int = 64,
    compression: str = "LZ4",
    shard_file_type: str = "fb",
    overwrite: bool = True,
    append: bool = False,
    progress_mode: str = "auto",
    nb_blocks: int = 2,
    warmup_captures_per_key: int = 10,
    holdout_minimal: bool = False,
    iv_mode: str = "per_trace",
    elf_path: str | Path | None = None,
) -> Path:
    """Generate or append to the RP2350 AES-256 CTR `sedpack` profiling and evaluation dataset."""
    if mode not in ("mock", "hardware"):
        raise ValueError(f"Unsupported acquisition mode '{mode}'. Expected 'mock' or 'hardware'.")

    nb_blocks = int(nb_blocks)
    if not (1 <= nb_blocks <= 12):
        raise ValueError(f"nb_blocks must be between 1 and 12 (got {nb_blocks}).")

    num_train_keys = int(num_train_keys)
    traces_per_train_key = int(traces_per_train_key)
    num_holdout_keys = int(num_holdout_keys)
    traces_per_holdout_key = int(traces_per_holdout_key)
    num_samples = int(num_samples)
    examples_per_shard = int(examples_per_shard)
    val_ratio = float(val_ratio)
    warmup_captures_per_key = max(0, int(warmup_captures_per_key))
    holdout_minimal = bool(holdout_minimal)
    canonical_iv_mode = _normalize_iv_mode(iv_mode)

    if num_train_keys < 0 or traces_per_train_key < 0:
        raise ValueError("Training key and trace-per-key counts must be non-negative.")
    if num_holdout_keys < 0 or traces_per_holdout_key < 0:
        raise ValueError("Holdout key and trace-per-key counts must be non-negative.")
    if val_ratio < 0.0:
        raise ValueError(f"val_ratio must be non-negative (got {val_ratio}).")
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive (got {num_samples}).")
    if examples_per_shard <= 0:
        raise ValueError(f"examples_per_shard must be positive (got {examples_per_shard}).")

    num_train = num_train_keys * traces_per_train_key
    num_val = int(math.ceil(num_train * val_ratio)) if num_train > 0 and val_ratio > 0 else 0
    traces_per_val_key = max(1, traces_per_train_key if traces_per_train_key > 0 else examples_per_shard)
    total_holdout = num_holdout_keys * traces_per_holdout_key
    total_traces = num_train + num_val + total_holdout
    if total_traces <= 0:
        raise ValueError("Total traces to capture must be positive (got 0).")

    ds_dir = Path(dataset_path).expanduser().resolve()
    use_progress_bar = _should_use_progress_bar(progress_mode)

    existing_dataset: Dataset | None = None
    existing_counts: dict[str, int] = {}
    dataset_iv: bytes | None = None
    dataset_iv_salt: bytes | None = None

    if append:
        if not (ds_dir / "dataset_info.json").is_file():
            raise FileNotFoundError(
                f"Cannot append: existing sedpack dataset_info.json not found in {ds_dir}"
            )
        existing_dataset = Dataset(ds_dir)
        for attr in existing_dataset.dataset_structure.saved_data_description:
            if attr.name == "trace1" and len(attr.shape) == 1:
                num_samples = int(attr.shape[0])
            elif attr.name == "plaintext" and len(attr.shape) == 1:
                nb_blocks = max(1, int(attr.shape[0]) // 16)
        examples_per_shard = int(existing_dataset.dataset_structure.examples_per_shard)
        compression = str(existing_dataset.dataset_structure.compression)
        shard_file_type = str(existing_dataset.dataset_structure.shard_file_type)
        existing_counts = {
            sname: int(sinfo.number_of_examples)
            for sname, sinfo in existing_dataset.dataset_info.splits.items()
        }
        cmeta_existing = existing_dataset.dataset_info.metadata.custom_metadata
        if isinstance(cmeta_existing, dict):
            if not holdout_minimal:
                ho_meta = cmeta_existing.get("splits_info", {}).get(HOLDOUT_SPLIT, {})
                if isinstance(ho_meta, dict) and bool(ho_meta.get("minimal_holdout", False)):
                    holdout_minimal = True
            if iv_mode == "per_trace" and isinstance(cmeta_existing.get("iv_mode"), str):
                canonical_iv_mode = _normalize_iv_mode(cmeta_existing["iv_mode"])
            if canonical_iv_mode == "fixed_per_dataset":
                iv_hex = cmeta_existing.get("dataset_iv_hex")
                salt_hex = cmeta_existing.get("dataset_iv_salt_hex")
                if (
                    isinstance(iv_hex, str)
                    and len(iv_hex) == 32
                    and isinstance(salt_hex, str)
                    and len(salt_hex) == 32
                ):
                    dataset_iv = bytes.fromhex(iv_hex)
                    dataset_iv_salt = bytes.fromhex(salt_hex)
        if canonical_iv_mode == "fixed_per_dataset" and (dataset_iv is None or dataset_iv_salt is None):
            for sp_candidate in (TRAIN_SPLIT, TEST_SPLIT, HOLDOUT_SPLIT):
                if existing_counts.get(sp_candidate, 0) > 0:
                    first_ex = next(
                        iter(existing_dataset.as_numpy_iterator(split=sp_candidate, repeat=False, shuffle=0)),
                        None,
                    )
                    if first_ex is not None:
                        dataset_iv = np.asarray(first_ex["iv"], dtype=np.uint8).tobytes()
                        dataset_iv_salt = np.asarray(first_ex["iv_salt"], dtype=np.uint8).tobytes()
                        break
    else:
        if ds_dir.exists() and not overwrite:
            raise FileExistsError(
                f"Dataset directory already exists: {ds_dir} (pass --overwrite or --append)."
            )
        if mode == "mock" and num_samples == 78400 and nb_blocks > 1:
            num_samples = 78400 + 18800 * (nb_blocks - 1)

    if canonical_iv_mode == "fixed_per_dataset" and (dataset_iv is None or dataset_iv_salt is None):
        dataset_iv = os.urandom(16)
        dataset_iv_salt = os.urandom(16)

    def _next_key_and_iv() -> tuple[bytes, bytes | None, bytes | None]:
        k = os.urandom(32)
        if canonical_iv_mode == "fixed_per_dataset":
            return k, dataset_iv, dataset_iv_salt
        if canonical_iv_mode == "per_key":
            return k, os.urandom(16), os.urandom(16)
        return k, None, None

    # Validate plaintexts-per-key vs. shard size alignment BEFORE touching disk or hardware
    if (
        num_train_keys > 0
        and traces_per_train_key > 0
        and (traces_per_train_key % examples_per_shard != 0)
    ):
        print(
            f"[!] WARNING: Train split plaintexts per key ({traces_per_train_key}) is not a "
            f"multiple of the shard size ({examples_per_shard}). "
            f"A shard may end up containing more than 1 key."
        )

    if (
        num_holdout_keys > 0
        and traces_per_holdout_key > 0
        and (traces_per_holdout_key % examples_per_shard != 0)
    ):
        raise ValueError(
            f"Holdout split plaintexts per key ({traces_per_holdout_key}) is not a multiple "
            f"of the shard size ({examples_per_shard}). A holdout shard must not contain more than 1 key."
        )

    iv_mode_desc_map = {
        "fixed_per_dataset": (
            f"FIXED FOR WHOLE DATASET (iv={dataset_iv.hex() if dataset_iv else ''}, "
            f"iv_salt={dataset_iv_salt.hex() if dataset_iv_salt else ''})"
        ),
        "per_key": "RENEWED PER KEY (fixed for all traces of the same key)",
        "per_trace": "RENEWED PER TRACE (random iv & iv_salt on every trace)",
    }
    iv_mode_desc = iv_mode_desc_map[canonical_iv_mode]
    holdout_mode_desc = (
        "MINIMAL: trace1, key, iv, iv_salt ONLY (others null-filled, TW OFF)"
        if holdout_minimal
        else "FULL: all 14 attributes populated (TraceWhisperer ON)"
    )

    print("=" * 70)
    print("SEDPACK DATASET APPEND CONFIGURATION" if append else "SEDPACK DATASET GENERATION CONFIGURATION")
    print("=" * 70)
    print(f"[*] Dataset Path:            {ds_dir}")
    print(f"[*] Acquisition Mode:        {mode.upper()}{' (APPEND)' if append else ''}")
    print(f"[*] Plaintext Blocks:        {nb_blocks} ({16 * nb_blocks} bytes)")
    print(f"[*] ADC Samples per Trace:   {num_samples} (encompassing all 6 POIs)")
    print(f"[*] Warmup Captures / Key:   {warmup_captures_per_key} (XIP flash cache warmup, not stored)")
    print(f"[*] Shard Configuration:     {examples_per_shard} ex/shard, {compression}, format={shard_file_type}")
    print(f"[*] IV / IV-Salt Mode:       {iv_mode_desc}")
    print(f"[*] Holdout Attribute Mode:  {holdout_mode_desc}")
    if append:
        prev_tr = existing_counts.get(TRAIN_SPLIT, 0)
        prev_val = existing_counts.get(TEST_SPLIT, 0)
        prev_ho = existing_counts.get(HOLDOUT_SPLIT, 0)
        print(f"[*] Existing Split Counts:   train={prev_tr:,}, val(test)={prev_val:,}, holdout={prev_ho:,}")
        print(
            f"[*] Appending to Training:   +{num_train:,} traces "
            f"({num_train_keys} keys x {traces_per_train_key} plaintexts/key, TraceWhisperer ON)"
        )
        print(f"[*] Appending to Validation: +{num_val:,} traces ({val_ratio * 100:.1f}% of train, TraceWhisperer ON)")
        print(
            f"[*] Appending to Holdout:    +{total_holdout:,} traces "
            f"({num_holdout_keys} keys x {traces_per_holdout_key} plaintexts/key, "
            f"{'Minimal/TW OFF' if holdout_minimal else 'TraceWhisperer ON'})"
        )
        print(f"[*] Traces to Capture Now:   {total_traces:,} traces")
    else:
        print(
            f"[*] Training Split:          {num_train:,} traces "
            f"({num_train_keys} keys x {traces_per_train_key} plaintexts/key, TraceWhisperer ON)"
        )
        print(f"[*] Validation Split:        {num_val:,} traces ({val_ratio * 100:.1f}% of train, TraceWhisperer ON)")
        print(
            f"[*] Holdout Evaluation:      {total_holdout:,} traces "
            f"({num_holdout_keys} keys x {traces_per_holdout_key} plaintexts/key, "
            f"{'Minimal/TW OFF' if holdout_minimal else 'TraceWhisperer ON'})"
        )
        print(f"[*] Total Dataset Size:      {total_traces:,} traces")
    print("=" * 70)

    if not append and overwrite and ds_dir.exists():
        print(f"[*] Overwrite requested: removing existing directory {ds_dir}...")
        shutil.rmtree(ds_dir)

    hw_capturer: HardwareTraceCapturer | None = None
    t_start = time.monotonic()
    captured_count = 0

    try:
        if mode == "hardware":
            hw_capturer = HardwareTraceCapturer(
                elf_path=elf_path,
                nb_blocks=nb_blocks,
                warmup_captures_per_key=warmup_captures_per_key,
            )
            hw_capturer.connect()
            if append or (num_samples > 0 and num_samples != 78400):
                if hw_capturer.calibrated_samples != num_samples:
                    print(
                        f"[*] Setting scope ADC samples ({hw_capturer.calibrated_samples} -> {num_samples}) "
                        f"to match requested dataset sample count."
                    )
                hw_capturer.set_adc_samples(num_samples)
            else:
                num_samples = hw_capturer.calibrated_samples

        if append:
            if existing_dataset is None:
                raise RuntimeError("Internal error: existing_dataset was not loaded in append mode.")
            dataset = existing_dataset
        else:
            scope_meta: dict[str, Any] = {}
            if hw_capturer is not None and hw_capturer.conn_info:
                scope_meta = {
                    "name": hw_capturer.conn_info.get("scope_name"),
                    "sn": hw_capturer.conn_info.get("scope_sn"),
                    "fw_version": hw_capturer.conn_info.get("fw_version"),
                    "cw_version": hw_capturer.conn_info.get("cw_version"),
                    "is_husky_plus": hw_capturer.conn_info.get("is_husky_plus"),
                    "hw_max_samples": hw_capturer.conn_info.get("hw_max_samples"),
                    "gain_db": hw_capturer.calibrated_gain,
                    "adc_samples": num_samples,
                }
            custom_meta: dict[str, Any] = {
                "target": "RP2350 Hacking Challenge 2",
                "implementation": "aes_ctr_asm",
                "nb_blocks": nb_blocks,
                "warmup_captures_per_key": warmup_captures_per_key,
                "iv_mode": canonical_iv_mode,
                "total_traces": total_traces,
                "adc_samples": num_samples,
                "scope": scope_meta,
                "splits_info": {
                    "train": {
                        "role": "profiling / training",
                        "num_keys": num_train_keys,
                        "traces_per_key": traces_per_train_key,
                        "num_traces": num_train,
                        "key_mode": "fixed_per_key_group",
                        "iv_mode": canonical_iv_mode,
                        "tracewhisperer_captured": True,
                    },
                    "test": {
                        "role": "validation (10% of train)",
                        "alias": ["val", "validation"],
                        "num_traces": num_val,
                        "key_mode": "random",
                        "iv_mode": canonical_iv_mode,
                        "tracewhisperer_captured": True,
                    },
                    "holdout": {
                        "role": "attack evaluation",
                        "num_keys": num_holdout_keys,
                        "traces_per_key": traces_per_holdout_key,
                        "num_traces": total_holdout,
                        "key_mode": "fixed_per_key_group",
                        "iv_mode": canonical_iv_mode,
                        "tracewhisperer_captured": not holdout_minimal,
                        "minimal_holdout": holdout_minimal,
                        "saved_attributes": (
                            ["trace1", "key", "iv", "iv_salt"]
                            if holdout_minimal
                            else [
                                "trace1", "trace2", "plaintext", "ciphertext", "iv", "iv_salt",
                                "key", "key_a", "key_b", "key_c", "key_d",
                                "round_keys", "intermediate_shares", "tracewhisperer",
                            ]
                        ),
                    },
                },
                "poi_analysis_budgets": {
                    "mangard_cpa_theoretical": 164,
                    "safety_margin_factor": 2.5,
                    "recommended_attack_traces_per_key": traces_per_holdout_key,
                },
            }
            if canonical_iv_mode == "fixed_per_dataset" and dataset_iv is not None and dataset_iv_salt is not None:
                custom_meta["dataset_iv_hex"] = dataset_iv.hex()
                custom_meta["dataset_iv_salt_hex"] = dataset_iv_salt.hex()

            metadata = Metadata(
                description="RP2350 Masked AES-256 CTR Side-Channel Profiling & Evaluation Dataset",
                dataset_license="https://creativecommons.org/licenses/by/4.0/",
                custom_metadata=custom_meta,
            )

            structure = build_sedpack_dataset_structure(
                num_samples=num_samples,
                examples_per_shard=examples_per_shard,
                compression=compression,
                shard_file_type=shard_file_type,
                trace_dtype="int16",
                nb_blocks=nb_blocks,
            )

            dataset = Dataset.create(path=ds_dir, metadata=metadata, dataset_structure=structure)

        t_start = time.monotonic()

        filler = dataset.filler(concurrency=2)
        filler._concurrent_pool.shutdown(wait=False)
        thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="sedpack-shard-io"
        )
        filler._concurrent_pool = thread_pool
        filler._dataset_filler_context._concurrent_pool = thread_pool

        with (
            filler as dataset_filler,
            _AsyncCapturePipeline(
                dataset_filler=dataset_filler,
                hw_capturer=hw_capturer,
                num_samples=num_samples,
                nb_blocks=nb_blocks,
            ) as async_pipeline,
            _CaptureProgressReporter(
                use_bar=use_progress_bar,
                progress_mode=progress_mode,
                append=append,
                total_traces=total_traces,
                num_train=num_train,
                num_train_keys=num_train_keys,
                traces_per_train_key=traces_per_train_key,
                num_val=num_val,
                total_holdout=total_holdout,
                num_holdout_keys=num_holdout_keys,
                traces_per_holdout_key=traces_per_holdout_key,
                holdout_minimal=holdout_minimal,
                examples_per_shard=examples_per_shard,
                t_start=t_start,
            ) as reporter,
        ):
            # Split A: Training Split
            if num_train > 0:
                with reporter.split_scope("train"):
                    for k_idx in range(num_train_keys):
                        train_key, key_iv, key_iv_salt = _next_key_and_iv()
                        reporter.set_key("train", k_idx + 1, num_train_keys)
                        for _ in range(traces_per_train_key):
                            async_pipeline.capture_and_enqueue(
                                key_override=train_key,
                                include_tw=True,
                                split=TRAIN_SPLIT,
                                iv_override=key_iv,
                                iv_salt_override=key_iv_salt,
                            )
                            reporter.advance("train", key_num=k_idx + 1)
                    async_pipeline.flush()

            # Split B: Validation Split (stored in sedpack 'test' split)
            if num_val > 0:
                with reporter.split_scope("val"):
                    val_key, key_iv, key_iv_salt = _next_key_and_iv()
                    for i in range(num_val):
                        if i > 0 and (i % traces_per_val_key == 0):
                            val_key, key_iv, key_iv_salt = _next_key_and_iv()
                        async_pipeline.capture_and_enqueue(
                            key_override=val_key,
                            include_tw=True,
                            split=TEST_SPLIT,
                            iv_override=key_iv,
                            iv_salt_override=key_iv_salt,
                        )
                        reporter.advance("val")
                    async_pipeline.flush()

            # Split C: Holdout Evaluation Split
            if total_holdout > 0:
                with reporter.split_scope("holdout"):
                    for k_idx in range(num_holdout_keys):
                        target_key, key_iv, key_iv_salt = _next_key_and_iv()
                        reporter.set_key("holdout", k_idx + 1, num_holdout_keys)
                        for _ in range(traces_per_holdout_key):
                            async_pipeline.capture_and_enqueue(
                                key_override=target_key,
                                include_tw=not holdout_minimal,
                                split=HOLDOUT_SPLIT,
                                minimal_example=holdout_minimal,
                                iv_override=key_iv,
                                iv_salt_override=key_iv_salt,
                            )
                            reporter.advance("holdout", key_num=k_idx + 1)
                    async_pipeline.flush()

            captured_count = reporter.captured_count

        # Synchronize custom_metadata split counts with updated dataset_info.splits on append
        if append:
            cmeta = dataset._dataset_info.metadata.custom_metadata
            if isinstance(cmeta, dict):
                updated_splits = dataset.dataset_info.splits
                new_total = sum(int(s.number_of_examples) for s in updated_splits.values())
                cmeta["total_traces"] = new_total
                cmeta["iv_mode"] = canonical_iv_mode
                if canonical_iv_mode == "fixed_per_dataset" and dataset_iv is not None and dataset_iv_salt is not None:
                    cmeta["dataset_iv_hex"] = dataset_iv.hex()
                    cmeta["dataset_iv_salt_hex"] = dataset_iv_salt.hex()
                s_info = cmeta.get("splits_info")
                if isinstance(s_info, dict):
                    if isinstance(s_info.get(TRAIN_SPLIT), dict):
                        prev_keys = s_info[TRAIN_SPLIT].get("num_keys")
                        if isinstance(prev_keys, int):
                            s_info[TRAIN_SPLIT]["num_keys"] = prev_keys + num_train_keys
                    if num_holdout_keys > 0 and isinstance(s_info.get(HOLDOUT_SPLIT), dict):
                        prev_h_keys = s_info[HOLDOUT_SPLIT].get("num_keys")
                        if isinstance(prev_h_keys, int):
                            s_info[HOLDOUT_SPLIT]["num_keys"] = prev_h_keys + num_holdout_keys
                        if holdout_minimal:
                            s_info[HOLDOUT_SPLIT]["minimal_holdout"] = True
                            s_info[HOLDOUT_SPLIT]["tracewhisperer_captured"] = False
                            s_info[HOLDOUT_SPLIT]["saved_attributes"] = ["trace1", "key", "iv", "iv_salt"]
                    for sname in (TRAIN_SPLIT, TEST_SPLIT, HOLDOUT_SPLIT):
                        if sname in updated_splits and isinstance(s_info.get(sname), dict):
                            s_info[sname]["num_traces"] = int(updated_splits[sname].number_of_examples)
                            s_info[sname]["iv_mode"] = canonical_iv_mode
                dataset.write_config()
    finally:
        if hw_capturer is not None:
            hw_capturer.disconnect()

    val_symlink = ds_dir / "val"
    if not val_symlink.exists():
        try:
            val_symlink.symlink_to("test")
        except OSError:
            pass

    elapsed = time.monotonic() - t_start
    print("\n" + "=" * 70)
    action_done = "DATASET APPEND COMPLETE" if append else "DATASET GENERATION COMPLETE"
    print(
        f"[+] {action_done}: {captured_count:,} traces in {elapsed:.1f} s "
        f"({captured_count / max(1e-6, elapsed):.2f} tr/s)"
    )
    print(f"[+] Output Directory: {ds_dir}")
    print("=" * 70)

    return ds_dir


# ==============================================================================
# 7. Dataset Inspection & Verification
# ==============================================================================

def inspect_sedpack_dataset(
    dataset_path: str | Path,
    max_check_examples: int = 10,
) -> dict[str, Any]:
    """Inspect and mathematically verify an existing `sedpack` dataset directory."""
    ds_dir = Path(dataset_path).expanduser().resolve()
    print("=" * 70)
    print(f"INSPECTING SEDPACK DATASET: {ds_dir}")
    print("=" * 70)

    if not ds_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {ds_dir}")
    if not (ds_dir / "dataset_info.json").is_file():
        raise FileNotFoundError(f"Missing dataset_info.json in {ds_dir}")

    dataset = Dataset(ds_dir)

    attr_map = {attr.name: attr for attr in dataset.dataset_structure.saved_data_description}
    print(f"[*] Registered Attributes ({len(attr_map)}):")
    for name, attr in attr_map.items():
        print(f"    - {name:<22} shape={str(attr.shape):<10} dtype={attr.dtype}")

    required_attrs = (
        "trace1", "trace2", "plaintext", "ciphertext", "iv", "iv_salt",
        "key", "key_a", "key_b", "key_c", "key_d",
        "round_keys", "intermediate_shares", "tracewhisperer",
    )
    for req in required_attrs:
        if req not in attr_map:
            raise ValueError(f"Dataset schema is missing required attribute '{req}'.")

    info = dataset.dataset_info
    print("\n[*] Dataset Splits in dataset_info.json:")
    split_stats: dict[str, dict[str, int]] = {}
    for sname, sinfo in info.splits.items():
        n_ex = int(sinfo.number_of_examples)
        n_sh = int(sinfo.number_of_shards)
        split_stats[sname] = {"examples": n_ex, "shards": n_sh}
        print(f"    - Split '{sname}': {n_ex:,} examples across {n_sh:,} shards")

    if not info.splits:
        raise ValueError(f"Dataset at {ds_dir} contains no splits.")

    for req_split in (TRAIN_SPLIT, TEST_SPLIT, HOLDOUT_SPLIT):
        if req_split not in info.splits:
            raise ValueError(f"Dataset must contain a '{req_split}' split.")

    cmeta = info.metadata.custom_metadata if isinstance(info.metadata.custom_metadata, dict) else {}
    ho_meta = (
        cmeta.get("splits_info", {}).get(HOLDOUT_SPLIT, {})
        if isinstance(cmeta.get("splits_info"), dict)
        else {}
    )
    holdout_is_minimal_meta = bool(ho_meta.get("minimal_holdout", False)) if isinstance(ho_meta, dict) else False
    recorded_iv_mode = (
        _normalize_iv_mode(str(cmeta["iv_mode"]))
        if isinstance(cmeta.get("iv_mode"), str)
        else None
    )
    if recorded_iv_mode is not None:
        print(f"\n[*] Recorded IV / IV-Salt Mode: '{recorded_iv_mode}'")

    dataset_first_iv: bytes | None = (
        bytes.fromhex(cmeta["dataset_iv_hex"])
        if isinstance(cmeta.get("dataset_iv_hex"), str) and len(cmeta["dataset_iv_hex"]) == 32
        else None
    )
    dataset_first_salt: bytes | None = (
        bytes.fromhex(cmeta["dataset_iv_salt_hex"])
        if isinstance(cmeta.get("dataset_iv_salt_hex"), str) and len(cmeta["dataset_iv_salt_hex"]) == 32
        else None
    )
    seen_key_to_iv_salt: dict[bytes, tuple[bytes, bytes]] = {}

    print("\n[*] Verifying Example Decodability & Cryptographic Integrity:")
    for split_key in (TRAIN_SPLIT, TEST_SPLIT, HOLDOUT_SPLIT):
        it = dataset.as_numpy_iterator(split=split_key, repeat=False, shuffle=0)
        checked = 0
        minimal_checked = 0
        pt_size = 0

        for ex in it:
            if pt_size == 0:
                pt_size = int(ex["plaintext"].shape[0])
                if pt_size <= 0 or pt_size % 16 != 0:
                    raise ValueError(
                        f"Plaintext must be a positive multiple of 16 bytes, got {pt_size} bytes."
                    )

            if ex["trace1"].ndim != 1 or ex["trace2"].ndim != 1:
                raise ValueError(f"trace1 and trace2 must be 1D arrays in split '{split_key}'.")
            if ex["trace2"].shape != ex["trace1"].shape or ex["trace2"].dtype != ex["trace1"].dtype:
                raise ValueError(
                    f"trace2 shape/dtype ({ex['trace2'].shape}, {ex['trace2'].dtype}) "
                    f"must match trace1 ({ex['trace1'].shape}, {ex['trace1'].dtype}) in split '{split_key}'."
                )
            if ex["key"].shape != (32,):
                raise ValueError(f"key must have shape (32,), got {ex['key'].shape}.")
            if ex["iv"].shape != (16,) or ex["iv_salt"].shape != (16,):
                raise ValueError("iv and iv_salt must each have shape (16,).")
            if ex["plaintext"].shape != (pt_size,) or ex["ciphertext"].shape != (pt_size,):
                raise ValueError(f"plaintext and ciphertext must have shape ({pt_size},).")
            if ex["round_keys"].shape != (15, 16):
                raise ValueError(f"round_keys must have shape (15, 16), got {ex['round_keys'].shape}.")
            if ex["intermediate_shares"].shape != (4, 16):
                raise ValueError(
                    f"intermediate_shares must have shape (4, 16), got {ex['intermediate_shares'].shape}."
                )

            iv_b = ex["iv"].tobytes()
            salt_b = ex["iv_salt"].tobytes()
            key_b = ex["key"].tobytes()
            if recorded_iv_mode == "fixed_per_dataset":
                if dataset_first_iv is None or dataset_first_salt is None:
                    dataset_first_iv = iv_b
                    dataset_first_salt = salt_b
                if iv_b != dataset_first_iv:
                    raise ValueError(
                        f"IV mismatch in split '{split_key}' under iv_mode='fixed_per_dataset': "
                        f"got {iv_b.hex()}, expected {dataset_first_iv.hex()}"
                    )
                if salt_b != dataset_first_salt:
                    raise ValueError(
                        f"IV salt mismatch in split '{split_key}' under iv_mode='fixed_per_dataset': "
                        f"got {salt_b.hex()}, expected {dataset_first_salt.hex()}"
                    )
            elif recorded_iv_mode == "per_key":
                prev_pair = seen_key_to_iv_salt.get(key_b)
                if prev_pair is None:
                    seen_key_to_iv_salt[key_b] = (iv_b, salt_b)
                elif (iv_b, salt_b) != prev_pair:
                    raise ValueError(
                        f"IV/IV-salt changed within the same key in split '{split_key}' "
                        f"under iv_mode='per_key'."
                    )

            is_minimal_ex = (
                split_key == HOLDOUT_SPLIT
                and (
                    holdout_is_minimal_meta
                    or (
                        not np.any(ex["round_keys"])
                        and not np.any(ex["key_a"])
                        and not np.any(ex["trace2"])
                    )
                )
            )

            if is_minimal_ex:
                for populated_attr in ("trace1", "key", "iv", "iv_salt"):
                    if not np.any(ex[populated_attr]):
                        raise ValueError(
                            f"Minimal holdout attribute '{populated_attr}' must not be all zeros."
                        )
                for null_attr in (
                    "trace2",
                    "plaintext",
                    "ciphertext",
                    "key_a",
                    "key_b",
                    "key_c",
                    "key_d",
                    "round_keys",
                    "intermediate_shares",
                ):
                    if np.any(ex[null_attr]):
                        raise ValueError(
                            f"Minimal holdout attribute '{null_attr}' must be zero-filled."
                        )
                if ex["tracewhisperer"] != "":
                    raise ValueError("Minimal holdout attribute 'tracewhisperer' must be empty.")
                minimal_checked += 1
            else:
                k_rec = ex["key_a"] ^ ex["key_b"] ^ ex["key_c"] ^ ex["key_d"]
                if not np.array_equal(k_rec, ex["key"]):
                    raise ValueError(f"Key share recombination failed in split '{split_key}'.")

                exp_rks = expand_aes256_round_keys(key_b)
                if not np.array_equal(exp_rks, ex["round_keys"]):
                    raise ValueError(f"Round key expansion mismatch in split '{split_key}'.")

                expected_ct = compute_aes_ctr_output(
                    key=key_b,
                    iv=iv_b,
                    iv_salt=salt_b,
                    plaintext=ex["plaintext"].tobytes(),
                )
                if not np.array_equal(np.frombuffer(expected_ct, dtype=np.uint8), ex["ciphertext"]):
                    raise ValueError(f"Ciphertext verification failed in split '{split_key}'.")

                st_shares = ex["intermediate_shares"]
                rec_st = st_shares[0] ^ st_shares[1] ^ st_shares[2] ^ st_shares[3]
                expected_st = ex["round_keys"][0] ^ (ex["iv"] ^ ex["iv_salt"])
                if not np.array_equal(rec_st, expected_st):
                    raise ValueError(f"State share recombination mismatch in split '{split_key}'.")

                tw_val = ex["tracewhisperer"]
                if tw_val:
                    tw_dict = json.loads(tw_val)
                    if not isinstance(tw_dict, dict):
                        raise ValueError(f"tracewhisperer must be a JSON object in split '{split_key}'.")
                    expected_t2 = compute_trace2_from_trace1(
                        ex["trace1"],
                        tw_dict,
                        nb_blocks=pt_size // 16,
                        target_length=int(ex["trace1"].shape[0]),
                    )
                    if not np.array_equal(expected_t2, ex["trace2"]):
                        raise ValueError(f"trace2 reconstruction mismatch in split '{split_key}'.")

            checked += 1
            if checked >= max_check_examples:
                break

        mode_note = (
            " (minimal holdout: trace1, key, iv, iv_salt populated; others null-filled)"
            if minimal_checked > 0
            else ""
        )
        print(
            f"    [+] Split '{split_key}': successfully verified {checked} examples "
            f"with zero errors{mode_note}."
        )

    print("\n" + "=" * 70)
    print("[+] ALL SEDPACK DATASET CHECKS PASSED SUCCESSFULLY (ZERO ERRORS)")
    print("=" * 70)

    return {
        "status": "verified",
        "dataset_path": str(ds_dir),
        "splits": split_stats,
        "attributes": list(attr_map.keys()),
    }


# ==============================================================================
# 8. CLI Entry Point
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture & build RP2350 AES-256 CTR Sedpack Dataset with raw traces and full labels."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/tmp/rp2350_sedpack_dataset",
        help="Local directory path for the sedpack dataset.",
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "hardware"],
        default="mock",
        help="Acquisition mode: 'mock' (offline synthetic) or 'hardware' (physical ChipWhisperer Husky).",
    )
    parser.add_argument(
        "--elf-path",
        type=str,
        default=None,
        help="Optional path to the RP2350 firmware ELF file (auto-detected in hardware mode if omitted).",
    )
    parser.add_argument(
        "--train-keys",
        type=int,
        default=100,
        help="Number of independent random keys in training split (default: 100).",
    )
    parser.add_argument(
        "--traces-per-train-key",
        "--plaintexts-per-train-key",
        dest="traces_per_train_key",
        type=int,
        default=64,
        help="Random plaintexts (traces) per key in training split (default: 64).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Validation split ratio relative to train (default: 0.10 = 10%%).",
    )
    parser.add_argument(
        "--holdout-keys",
        type=int,
        default=100,
        help="Number of independent random keys in holdout split (default: 100).",
    )
    parser.add_argument(
        "--traces-per-holdout-key",
        "--plaintexts-per-holdout-key",
        dest="traces_per_holdout_key",
        type=int,
        default=768,
        help="Attack traces (plaintexts) per holdout key (default: 768, must be a multiple of --examples-per-shard).",
    )
    parser.add_argument(
        "--holdout-minimal",
        "--holdout-only-trace1",
        dest="holdout_minimal",
        action="store_true",
        default=False,
        help=(
            "Save only `trace1`, `key`, `iv`, and `iv_salt` on the `holdout` split, "
            "keeping all other attributes zero-filled for compatibility."
        ),
    )
    parser.add_argument(
        "--iv-mode",
        type=_normalize_iv_mode,
        default="per_trace",
        metavar="{fixed_per_dataset,per_key,per_trace}",
        help=(
            "Renewal behavior for `iv` and `iv_salt`: "
            "'fixed_per_dataset' ('fixed'/'dataset'/'1'), "
            "'per_key' ('key'/'2'), or "
            "'per_trace' ('random'/'current'/'3', default)."
        ),
    )
    parser.add_argument(
        "--iv-fixed-dataset",
        "--fixed-iv",
        dest="iv_mode",
        action="store_const",
        const="fixed_per_dataset",
        help="Shortcut for `--iv-mode fixed_per_dataset`.",
    )
    parser.add_argument(
        "--iv-per-key",
        dest="iv_mode",
        action="store_const",
        const="per_key",
        help="Shortcut for `--iv-mode per_key`.",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=2,
        help="Size of plaintext/ciphertext in 16-byte blocks (default: 2).",
    )
    parser.add_argument(
        "--warmup-captures",
        type=int,
        default=10,
        help="Number of unstored warmup captures per new key on hardware (default: 10).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=78400,
        help="ADC sample count per trace covering all POIs (default: 78400).",
    )
    parser.add_argument(
        "--examples-per-shard",
        type=int,
        default=64,
        help="Examples per shard file (default: 64).",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="LZ4",
        help="Shard compression: 'LZ4' or 'GZIP' (default: LZ4).",
    )
    parser.add_argument(
        "--mini-test",
        action="store_true",
        help="Generate a mini dataset (4 keys x 16 tr = 64 train, 7 val, 2 keys x 16 tr = 32 holdout).",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect and verify the dataset at `--dataset-path`.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing dataset directory if it exists (default: True; ignored with --append).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=False,
        help="Append new traces to an existing dataset.",
    )
    parser.add_argument(
        "--progress",
        choices=["auto", "bar", "text"],
        default="auto",
        help="Progress display mode: 'auto', 'bar', or 'text' (default: auto).",
    )

    args = parser.parse_args()

    try:
        if args.inspect:
            inspect_sedpack_dataset(args.dataset_path)
            return

        ex_per_sh = 16 if args.mini_test else args.examples_per_shard
        n_train_keys = 4 if args.mini_test else args.train_keys
        tr_per_train_k = 16 if args.mini_test else args.traces_per_train_key
        n_h_keys = 2 if args.mini_test else args.holdout_keys
        tr_per_k = 16 if args.mini_test else args.traces_per_holdout_key

        out_path = generate_sedpack_dataset(
            dataset_path=args.dataset_path,
            num_train_keys=n_train_keys,
            traces_per_train_key=tr_per_train_k,
            val_ratio=args.val_ratio,
            num_holdout_keys=n_h_keys,
            traces_per_holdout_key=tr_per_k,
            mode=args.mode,
            num_samples=args.samples,
            examples_per_shard=ex_per_sh,
            compression=args.compression,
            overwrite=args.overwrite,
            append=args.append,
            progress_mode=args.progress,
            nb_blocks=args.blocks,
            warmup_captures_per_key=args.warmup_captures,
            holdout_minimal=args.holdout_minimal,
            iv_mode=args.iv_mode,
            elf_path=args.elf_path,
        )

        inspect_sedpack_dataset(out_path)
    except KeyboardInterrupt:
        print("\n[!] Interrupted.", file=sys.stderr)
        sys.exit(130)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, KeyError, ImportError) as exc:
        print(f"[!] Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
