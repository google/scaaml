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
"""Step 1: Compile Unmodified RP2350 Bootloader (`aes.S`) & Verify Against `picotool`.

Inputs:
  - `bootloader/aes.S`                     (unmodified RP2350 encrypted bootloader assembly)
  - `bootloader/config.h`                  (unmodified bootloader countermeasure config)
  - `bootloader/aes.ld`                    (SRAM linker script: `.text` @ `0x20080000`, `.scratch_y` @ `0x20081000`)
  - `bootloader/include/hardware/*`        (unmodified `rcp.h`, `platform_defs.h`, RP2350 MMIO reg offsets)
  - `bootloader/picotool_enc_bootloader.elf` (upstream `raspberrypi/picotool` reference ELF)
  - `data/etm_cycle_model.json`            (hardware CoreSight ETMv4 per-PC instruction timing table)

Outputs:
  - `<work_dir>/build/aes_bootloader.elf`           (compiled ARMv8-M Cortex-M33 ELF loaded by Step 2)
  - `<work_dir>/build/bootloader_verification.json` (SHA-256 & instruction equivalence verification report)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from capstone import CS_ARCH_ARM, CS_MODE_MCLASS, CS_MODE_THUMB, Cs
from elftools.elf.elffile import ELFFile

ROOT_DIR = Path(__file__).resolve().parent
BOOTLOADER_DIR = ROOT_DIR / "bootloader"
DATA_DIR = ROOT_DIR / "data"

EXPECTED_SOURCE_SHA256: dict[str, str] = {
    "aes.S":
        "714b5a30601a7c34cb3e2acdbefe9282fa51e113bd6c99c43c7a8c1002de5641",
    "config.h":
        "82e3bf63b4ade7912e5d6f3db5418dfcf6d2831a74a2022e4f093a1cd5d0e71d",
    "include/hardware/rcp.h":
        "7c084f0925315b0571131de9651582282510aa2fe92aad67755cf09c2c228206",
}

AES_FUNCTION_SPANS: list[tuple[str, str]] = [
    ("init_rstate", "decrypt"),
    ("decrypt", "reset_sha_trng"),
    ("reset_sha_trng", "gen_rand_sha"),
    ("gen_rand_sha", "gen_rand_sha_nonpres"),
    ("gen_rand_sha_nonpres", "gen_rand_lfsr_nonpres"),
    ("gen_rand_lfsr_nonpres", "makesmallperm"),
    ("makesmallperm", "makeperm16"),
    ("makeperm16", "remap"),
    ("remap", "remap_1"),
    ("remap_1", "addstatevperm"),
    ("addstatevperm", "conjshareC"),
    ("conjshareC", "mix_cols_s"),
    ("mix_cols_s", "randomisechaff"),
    ("randomisechaff", "refreshchaff_and_lfsr"),
    ("refreshchaff_and_lfsr", "init_key_sbox"),
    ("init_key_sbox", "storeroundkey"),
    ("storeroundkey", "init_key_4way"),
    ("init_key_4way", "addrkey_s"),
    ("addrkey_s", "ctr_crypt_s"),
    ("ctr_crypt_s", "decryption_end"),
]

CORE_CRYPTO_NO_HARDENING_DIFF_FUNCS: set[str] = {
    "reset_sha_trng",
    "makesmallperm",
    "makeperm16",
    "remap",
    "remap_1",
    "addstatevperm",
    "conjshareC",
    "mix_cols_s",
    "randomisechaff",
    "init_key_sbox",
    "storeroundkey",
    "addrkey_s",
}

SCRATCH_Y_SYMBOLS: list[str] = [
    "workspace_start",
    "chaff",
    "rkey_s",
    "rkey4way",
    "bperm_rand",
    "permscratch",
    "perm16",
    "fourway",
    "shareA",
    "shareB",
    "shareC",
    "statevperm",
    "RKshareC",
    "RKshareCchange",
    "IV0",
    "lut_a",
    "lut_a_map",
    "lut_b",
    "lut_b_map",
    "rstate_all_start",
    "rstate_sha",
    "jstate",
    "rstate_lfsr",
    "rstate_count",
    "rstate_all_end",
    "murmur3_constants",
    "scratch_y_end",
]

_HEX_TARGET_RE = re.compile(r"#0x([0-9a-fA-F]+)")
_PC_LOAD_RE = re.compile(r"\[pc,\s*#(-?0x[0-9a-fA-F]+|-?\d+)\]")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_elf_sections_and_symbols(
    elf_bytes: bytes,
) -> tuple[dict[int, bytes], dict[str, int], dict[int, str], set[int]]:
    ef = ELFFile(io.BytesIO(elf_bytes))
    mem: dict[int, bytes] = {}
    for sec in ef.iter_sections():
        if (sec["sh_flags"] & 0x2) and sec["sh_size"] > 0:
            mem[int(sec["sh_addr"])] = sec.data()

    symtab = ef.get_section_by_name(".symtab")
    syms: dict[str, int] = {}
    addr_to_sym: dict[int, str] = {}
    data_addrs: set[int] = set()
    code_addrs: set[int] = set()
    for s in symtab.iter_symbols():
        val = int(s["st_value"]) & ~1
        if s.name == "$d":
            data_addrs.add(val)
        elif s.name == "$t":
            code_addrs.add(val)
        elif s.name and not s.name.startswith("$"):
            syms[s.name] = val
            addr_to_sym.setdefault(val, s.name)

    markers = sorted([(a, "t") for a in code_addrs] +
                     [(a, "d") for a in data_addrs])
    in_data: set[int] = set()
    for idx, (addr, kind) in enumerate(markers):
        if kind == "d":
            end_addr = markers[idx +
                               1][0] if idx + 1 < len(markers) else addr + 256
            for b_addr in range(addr, end_addr, 2):
                in_data.add(b_addr)
    return mem, syms, addr_to_sym, in_data


def _read_mem(mem: dict[int, bytes], addr: int, size: int) -> bytes:
    for base, buf in mem.items():
        if base <= addr and addr + size <= base + len(buf):
            return buf[addr - base:addr - base + size]
    return b""


def _disasm_span(
    md: Cs,
    mem: dict[int, bytes],
    in_data: set[int],
    start_addr: int,
    end_addr: int,
) -> list[tuple[int, str, str]]:
    out: list[tuple[int, str, str]] = []
    cur = start_addr
    while cur < end_addr:
        if cur in in_data:
            cur += 2
            continue
        chunk = _read_mem(mem, cur, min(4, end_addr - cur))
        insns = list(md.disasm(chunk, cur, count=1))
        if not insns:
            cur += 2
            continue
        ins = insns[0]
        out.append((ins.address, ins.mnemonic, ins.op_str))
        cur += ins.size
    return out


def _normalize_instructions(
    insns: list[tuple[int, str, str]],
    mem: dict[int, bytes],
    addr_to_sym: dict[int, str],
    strip_rcp_coprocessor: bool,
) -> list[str]:
    resolved: list[tuple[int, str, str, int | None]] = []
    for addr, mnem, op in insns:
        if strip_rcp_coprocessor and op.startswith("p7,"):
            continue
        lit_val = None
        m_pc = _PC_LOAD_RE.search(op)
        if m_pc and mnem.startswith("ldr"):
            disp = int(m_pc.group(1), 0)
            lit_addr = ((addr + 4) & ~3) + disp
            raw = _read_mem(mem, lit_addr, 4)
            lit_val = int.from_bytes(raw, "little") if len(raw) == 4 else None
        resolved.append((addr, mnem, op, lit_val))

    while resolved and resolved[-1][1] == "nop":
        resolved.pop()

    func_start = insns[0][0] if insns else 0
    norm: list[str] = []
    for addr, mnem, op, lit_val in resolved:
        mn_base = mnem.replace(".w", "")
        if lit_val is not None:
            reg_part = op.split(",")[0].strip()
            sym_name = addr_to_sym.get(lit_val)
            norm.append(f"{mn_base} {reg_part}, ={sym_name}" if sym_name else
                        f"{mn_base} {reg_part}, =0x{lit_val:08x}")
            continue
        if mn_base in {
                "b",
                "bl",
                "beq",
                "bne",
                "bpl",
                "bmi",
                "bcc",
                "blo",
                "bcs",
                "bhs",
                "bhi",
                "bls",
                "bge",
                "blt",
                "bgt",
                "ble",
                "cbz",
                "cbnz",
        }:
            m_t = _HEX_TARGET_RE.search(op)
            if m_t:
                tgt = int(m_t.group(1), 16)
                if tgt in addr_to_sym and tgt != func_start:
                    norm.append(f"{mn_base} {addr_to_sym[tgt]}")
                else:
                    norm.append(
                        f"{mn_base} {'backward' if tgt < addr else 'forward'}")
                continue
        norm.append(f"{mn_base} {op}")
    return norm


def build_and_verify_bootloader(work_dir: Path) -> dict[str, Any]:
    # 1. Verify unmodified upstream bootloader source checksums
    source_hashes: dict[str, str] = {}
    for rel_path, expected_sha in EXPECTED_SOURCE_SHA256.items():
        actual_sha = sha256_bytes((BOOTLOADER_DIR / rel_path).read_bytes())
        if actual_sha != expected_sha:
            raise RuntimeError(f"Bootloader source {rel_path} modified! "
                               f"sha256={actual_sha} != {expected_sha}")
        source_hashes[rel_path] = actual_sha

    build_dir = work_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    elf_path = build_dir / "aes_bootloader.elf"

    # 2. Compile unmodified `bootloader/aes.S` with contest firmware
    # defines (`HARDENING=0`)
    cmd = [
        "arm-none-eabi-gcc",
        "-mcpu=cortex-m33",
        "-mthumb",
        "-mfloat-abi=soft",
        "-nostdlib",
        "-nostartfiles",
        "-ffreestanding",
        "-g",
        f"-I{BOOTLOADER_DIR / 'include'}",
        f"-I{BOOTLOADER_DIR}",
        "-DRC_COUNT=0",
        "-DHARDENING=0",
        "-DDOUBLE_HARDENING=0",
        "-DINLINE_REF_ROUNDKEY_SHARES_S=1",
        "-DINLINE_REF_ROUNDKEY_HVPERMS_S=1",
        "-DINLINE_SHIFT_ROWS_S=1",
        "-DINLINE_MAP_SBOX_S=1",
        "-DCALLER_INIT_RCP_COUNT=1",
        "-DST_VPERM=1",
        "-DCT_BPERM=1",
        "-DRK_ROR=1",
        "-DSH_JITTER=1",
        "-DCK_JITTER=1",
        "-DRC_CANARY=1",
        f"-T{BOOTLOADER_DIR / 'aes.ld'}",
        "-o",
        str(elf_path),
        str(BOOTLOADER_DIR / "aes.S"),
    ]
    subprocess.check_call(cmd)

    # 3. Verify assembly & SRAM layout equivalence against `bootloader/picotool_enc_bootloader.elf`
    md = Cs(CS_ARCH_ARM, CS_MODE_THUMB | CS_MODE_MCLASS)
    our_bytes = elf_path.read_bytes()
    ref_elf_path = BOOTLOADER_DIR / "picotool_enc_bootloader.elf"
    ref_bytes = ref_elf_path.read_bytes()

    m_ou, s_ou, a2s_ou, d_ou = load_elf_sections_and_symbols(our_bytes)
    m_ref, s_ref, a2s_ref, d_ref = load_elf_sections_and_symbols(ref_bytes)

    core_matched = 0
    core_total_insns = 0
    per_function: dict[str, Any] = {}
    for fn, nxt in AES_FUNCTION_SPANS:
        nxt_ref = "_entry_point" if fn == "reset_sha_trng" else nxt
        end_ref = (0x20080044 +
                   0x0F98) if fn == "ctr_crypt_s" else s_ref[nxt_ref]
        end_ou = s_ou[nxt] + (2 if nxt == "decryption_end" else 0)

        ins_ref = _disasm_span(md, m_ref, d_ref, s_ref[fn], end_ref)
        ins_ou = _disasm_span(md, m_ou, d_ou, s_ou[fn], end_ou)

        norm_ref = _normalize_instructions(ins_ref,
                                           m_ref,
                                           a2s_ref,
                                           strip_rcp_coprocessor=True)
        norm_ou = _normalize_instructions(ins_ou,
                                          m_ou,
                                          a2s_ou,
                                          strip_rcp_coprocessor=True)
        exact_match = norm_ref == norm_ou

        if fn in CORE_CRYPTO_NO_HARDENING_DIFF_FUNCS:
            if not exact_match:
                raise RuntimeError(
                    "Assembly mismatch vs picotool in core crypto "
                    f"function `{fn}`!")
            core_matched += 1
            core_total_insns += len(norm_ou)

        per_function[fn] = {
            "our_insns": len(ins_ou),
            "picotool_insns": len(ins_ref),
            "exact_match_modulo_rcp_canaries": exact_match,
        }

    for sym in SCRATCH_Y_SYMBOLS:
        if s_ou.get(sym) != s_ref.get(sym):
            raise RuntimeError(f".scratch_y symbol mismatch for `{sym}`: "
                               f"ours=0x{s_ou.get(sym, 0):x} vs "
                               f"picotool=0x{s_ref.get(sym, 0):x}")

    # 4. Verify 100% of ETMv4 cycle model entries match `aes_bootloader.elf`
    etm_doc = json.loads(
        (DATA_DIR / "etm_cycle_model.json").read_text(encoding="utf-8"))
    etm_matched = 0
    for pc_hex, einfo in etm_doc["instructions"].items():
        pc = int(pc_hex, 16)
        sz = int(einfo["size"])
        ins_list = list(md.disasm(_read_mem(m_ou, pc, sz), pc, count=1))
        if not ins_list or ins_list[0].size != sz or ins_list[
                0].mnemonic != einfo["mnem"]:
            raise RuntimeError(f"ETMv4 instruction mismatch at {pc_hex}!")
        etm_matched += 1

    report = {
        "verified":
            True,
        "compiled_elf":
            str(elf_path),
        "compiled_elf_sha256":
            sha256_bytes(our_bytes),
        "source_sha256":
            source_hashes,
        "picotool_reference_elf":
            str(ref_elf_path),
        "picotool_reference_elf_sha256":
            sha256_bytes(ref_bytes),
        "core_crypto_shuffling_functions_matched":
            f"{core_matched}/{len(CORE_CRYPTO_NO_HARDENING_DIFF_FUNCS)}",
        "core_crypto_shuffling_instructions_verified":
            core_total_insns,
        "scratch_y_sram_symbols_matched":
            f"{len(SCRATCH_Y_SYMBOLS)}/{len(SCRATCH_Y_SYMBOLS)}",
        "etm_cycle_model_instructions_matched":
            f"{etm_matched}/{len(etm_doc['instructions'])}",
        "per_function":
            per_function,
    }
    out_json = build_dir / "bootloader_verification.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile & verify unmodified RP2350 bootloader ELF.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("test_run"),
        help=("Working directory for build, intermediate, and"
              " output files (default: test_run)"),
    )
    args = parser.parse_args()
    report = build_and_verify_bootloader(args.work_dir.resolve())
    print(f"[+] Compiled bootloader ELF: {report['compiled_elf']}")
    print(f"    - ELF SHA-256:              {report['compiled_elf_sha256']}")
    print("    - Core crypto funcs match:  "
          f"{report['core_crypto_shuffling_functions_matched']}"
          f" ({report['core_crypto_shuffling_instructions_verified']}"
          " insns vs picotool)")
    print("    - .scratch_y SRAM symbols:  "
          f"{report['scratch_y_sram_symbols_matched']}"
          " identical")
    print("    - ETMv4 cycle model match:  "
          f"{report['etm_cycle_model_instructions_matched']} instructions")


if __name__ == "__main__":
    main()
