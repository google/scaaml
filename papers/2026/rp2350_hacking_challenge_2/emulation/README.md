# RP2350 Bootloader Side-Channel Emulator Submission

Minimal, self-contained RP2350 Cortex-M33 + MMIO side-channel emulator that
compiles and loads the unmodified RP2350 encrypted bootloader
(`bootloader/aes.S`), generates standard Rainbow `HammingWeight()` traces
stretched into a hardware-shaped `sedpack` dataset (`97,553` `int16`
samples/trace), and runs the **unmodified** `end_to_end_attack.py`.

For the technical explanation of how 1-point-per-instruction Hamming-weight
traces are stretched to `97,553` ADC samples and a threats-to-validity analysis
of the ETMv4 per-address cycle calibration, see [`EMULATOR_ETM_CALIBRATION.md`](EMULATOR_ETM_CALIBRATION.md).

---

## 1. Prerequisites & Installation

System compilers required on `PATH`:

-   `gcc` (builds the Unicorn C tracer shared library)
-   `arm-none-eabi-gcc` (compiles unmodified `bootloader/aes.S` into `aes_bootloader.elf`)

Create a fresh Python virtual environment and install `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Usage (5-Step Modular Workflow)

Set `SCAAML_RP2350` to the directory where `end_to_end_attack.py` and other files
are and choose an output working directory (e.g. `test_run`):

```bash
SCAAML_RP2350=`pwd`/../
WORK_DIR=test_run

# Step 1: Compile unmodified bootloader/aes.S -> <WORK_DIR>/build/aes_bootloader.elf
#         and verify assembly & SRAM layout equivalence against picotool's enc_bootloader.elf
python3 01_build_bootloader.py --work-dir "$WORK_DIR"

# Step 2: Load aes_bootloader.elf PT_LOAD segments into Unicorn Cortex-M33 + RP2350 MMIO
#         and record raw 1-point-per-instruction Rainbow HammingWeight() traces
#         (Use --holdout-keys 100 for full contest scale, or --holdout-keys 12 for a fast test run)
python3 02_emulate_instructions.py --work-dir "$WORK_DIR" --train-keys 256 --holdout-keys 12

# Step 3: Stretch 1-point/instruction traces to 97,553-sample int16 ADC waveforms
#         and write the sedpack dataset (holdout has trace2=0 and tracewhisperer="")
python3 03_stretch_to_sedpack.py --work-dir "$WORK_DIR" --scaaml-dir "$SCAAML_RP2350"

# Step 4: Fit canonical_t2_ref.npy & Walsh/Tree-BP templates strictly on <WORK_DIR>/dataset/train (0 holdout)
python3 04_fit_train_templates.py --work-dir "$WORK_DIR" --scaaml-dir "$SCAAML_RP2350"

# Step 5: Stage byte-identical end_to_end_attack.py (SHA-256 verified) + run blind holdout recovery
python3 05_run_unmodified_attack.py --work-dir "$WORK_DIR" --scaaml-dir "$SCAAML_RP2350"
```

---

## 3. Complete Inventory of Input, Temporary/Intermediate, and Output Files

### A. Static Input Files

| Path | Role |
| --- | --- |
| `requirements.txt` | Python package dependencies (`numpy`, `matplotlib`, `sedpack`, `unicorn`, `capstone`, `pyelftools`, `pycryptodome`, `lz4`, `rich`). |
| `bootloader/aes.S` | Unmodified RP2350 encrypted bootloader assembly (`SHA-256: 714b5a30...`). |
| `bootloader/config.h` | Unmodified countermeasure configuration header (`SHA-256: 82e3bf63...`). |
| `bootloader/aes.ld` | Linker script placing `.text` at `0x20080000` and `.scratch_y.aes` at `0x20081000`. |
| `bootloader/include/hardware/*` | Unmodified `rcp.h` (`SHA-256: 7c084f09...`), `platform_defs.h`, and RP2350 MMIO register offset headers (`regs/*.h`). |
| `bootloader/picotool_enc_bootloader.elf` | Official upstream Raspberry Pi `picotool` reference ELF (`SHA-256: 9ce3c424...`) used by Step 1 to verify assembly equivalence. |
| `src/rp2350_unicorn_tracer.c` | Minimal hook-free Cortex-M33 + RP2350 MMIO (`RESETS`, `ROSC`, `TRNG`, `SHA256`) C tracer emitting raw 1-point-per-instruction `(pc_idx, rainbow_hw, hw_stall)`. |
| `data/etm_cycle_model.json` | Hardware CoreSight ETMv4 per-PC cycle timing table (`1,153` instructions; see [`EMULATOR_ETM_CALIBRATION.md`](EMULATOR_ETM_CALIBRATION.md)). |
| `data/dataset_schema.json` | `sedpack` `dataset_info.json` attribute schema matching `rp2350_aesctr_small_fixed` (`14` attributes, `97,553` `int16` samples). |
| `<SCAAML_RP2350>/reconstruct_aes_trace.py` | alignment helper (used strictly on `train` in Step 3 to populate `trace2`). |
| `<SCAAML_RP2350>/end_to_end_attack.py` | Unmodified contest attack script (`SHA-256: c2ecf5e2...`) executed on `holdout` in Step 5. |

### B. Temporary & Intermediate Files (`<WORK_DIR>/`)

| Path | Produced By | Consumed By | Description |
| --- | --- | --- | --- |
| `<WORK_DIR>/build/aes_bootloader.elf` | Step 1 | Step 2 | Compiled ARMv8-M Cortex-M33 bootloader ELF whose `PT_LOAD` segments (`0x20080000`, `0x20081000`) and `.symtab` symbols are loaded into Unicorn. |
| `<WORK_DIR>/build/bootloader_verification.json` | Step 1 | Audit | SHA-256 and per-function assembly equivalence report vs. `picotool_enc_bootloader.elf`. |
| `<WORK_DIR>/build/rp2350_unicorn_tracer.so` | Step 2 | Step 2 | Compiled C shared library linked dynamically against the active venv's `libunicorn.so.2`. |
| `<WORK_DIR>/raw_traces/calibration_zero_trace.npz` | Step 2 | Step 3 | Single non-secret all-zero instruction trace (`pc_idx`, `rainbow_hw`, `hw_stall`) used to compute reference mean HW ($\mu_{\text{HW,ref}}$). |
| `<WORK_DIR>/raw_traces/sample_instruction_trace.csv` | Step 2 | Human inspection | Readable CSV of ~`66,600` executed ARM instructions for 1 encryption (`step, pc_hex, function, mnemonic, operands, regs_written, rainbow_hw, hw_stall`). |
| `<WORK_DIR>/raw_traces/train/raw_shard_XXXXX.npz.lz4` | Step 2 | Step 3 | Temporary LZ4-compressed NumPy bundles of 64 un-stretched instruction traces + `train`-only DWT watchpoint events (removable via `--clean-raw-traces`). |
| `<WORK_DIR>/raw_traces/holdout/raw_shard_XXXXX.npz.lz4` | Step 2 | Step 3 | Temporary LZ4-compressed NumPy bundles of 64 un-stretched instruction traces with DWT disabled (`record_dwt=0`, removable via `--clean-raw-traces`). |
| `<WORK_DIR>/raw_traces/manifest.json` | Step 2 | Step 3 | Split counts, ELF segment map, key-disjointness flag, and instruction count statistics. |
| `<WORK_DIR>/attack_run/end_to_end_attack.py` | Step 5 | Step 5 | Temporary byte-for-byte copy of `<SCAAML_RP2350>/end_to_end_attack.py` (`SHA-256` verified) staged next to `canonical_t2_ref.npy`. |
| `<WORK_DIR>/attack_run/canonical_t2_ref.npy` | Step 5 | Step 5 | Copy of `<WORK_DIR>/profiling/canonical_t2_ref.npy` resolved by `end_to_end_attack.py` from `Path(__file__).resolve().parent`. |

### C. Final Dataset, Profiling, & Attack Output Files (`<WORK_DIR>/`)

| Path | Produced By | Description |
| --- | --- | --- |
| `<WORK_DIR>/dataset/dataset_info.json` | Step 3 | Root `sedpack` dataset descriptor (`train` + `holdout` splits, `97,553` `int16` samples). |
| `<WORK_DIR>/dataset/{train,holdout}/shards_list.json` | Step 3 | Top-level `sedpack` split shard index with SHA-256 checksums. |
| `<WORK_DIR>/dataset/{train,holdout}/shards/shards_list.json` | Step 3 | Leaf `sedpack` shard manifest listing every `shard_XXXXX.fb` file and checksum. |
| `<WORK_DIR>/dataset/{train,holdout}/shards/shard_XXXXX.fb` | Step 3 | LZ4-compressed FlatBuffer `sedpack` shards (`64` traces/shard; `holdout` has `trace2=0` and `tracewhisperer=""`). |
| `<WORK_DIR>/dataset_stretch_summary.json` | Step 3 | Summary of instruction-to-cycle stretching, empirical ADC mean, and on-disk `holdout` zero-`trace2` audit. |
| `<WORK_DIR>/profiling/canonical_t2_ref.npy` | Step 4 | Mean waveform (`float32[97553]`) of the first 512 `train` `trace2` examples (`0` `holdout` traces). |
| `<WORK_DIR>/profiling/trace2_walsh_tree_bp_templates.pkl` | Step 4 | Coupled 4-mode Walsh + S-Box Tree-BP templates fitted strictly on `train` (`0` `holdout` traces). |
| `<WORK_DIR>/profiling/profiling_summary.json` | Step 4 | Out-of-fold bit correlations (`r_oof_bits`) and `train`-only provenance summary. |
| `<WORK_DIR>/attack_run/attack_results.json` | Step 5 | Full 256-bit blind `holdout` key-recovery metrics across trace budgets (`M = 64..768`) from unmodified `end_to_end_attack.py`. |
| `<WORK_DIR>/attack_run/attack_plot.png` | Step 5 | Key-recovery vs. trace-budget plot generated by `end_to_end_attack.py`. |
| `<WORK_DIR>/attack_run/attack_stdout.log` | Step 5 | Complete stdout/stderr transcript of `end_to_end_attack.py`. |
