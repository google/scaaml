# Instruction-Trace Stretching & ETMv4 Timing Calibration (`EMULATOR_ETM_CALIBRATION.md`)

This document explains how raw 1-point-per-instruction Hamming-weight traces
from `02_emulate_instructions.py` are stretched into `97,553`-sample
ChipWhisperer-shaped power traces in `03_stretch_to_sedpack.py`, how
`data/etm_cycle_model.json` was calibrated from real RP2350 CoreSight ETMv4
hardware captures, and the **threats to validity** of using a per-address timing
table rather than a context-free per-opcode cycle model.

---

## 1. How Instruction Traces Are Stretched (`02` $\rightarrow$ `03`)

### Stage 1: Raw 1-Point-per-Instruction Emulation (`02_emulate_instructions.py`)

`src/rp2350_unicorn_tracer.c` executes the unmodified RP2350 bootloader ELF
(`aes_bootloader.elf`) on an ARMv8-M Cortex-M33 Unicorn core with RP2350 MMIO
peripherals (`RESETS`, `ROSC`, `TRNG`, `SHA256`). It does **no** cycle
stretching and adds **no** noise. Instead, for each executed ARM instruction
$i \in \{0, \dots, N_{\text{insn}}-1\}$ ($N_{\text{insn}} \approx 66,598$ for a
2-block AES-256-CTR call), it emits one tuple:

1.  `pc_idx[i]` $= (\text{PC}_i - \text{0x20080000}) \gg 1$ — static instruction
    halfword index;
2.  `rainbow_hw[i]` $= \sum_{r \in \text{regs\_written}(i)} \operatorname{popcount}(\text{reg}_i[r])$
     — standard Rainbow `HammingWeight()` register writeback leakage (computed
     over the 17 ARM registers `r0..r12, sp, lr, pc, cpsr` identified by
     Capstone `regs_access()`);
3.  `hw_stall[i]` $= \Delta c_{\text{udiv}}(i) + \Delta c_{\text{APB}}(i)$ —
    **dynamic** runtime bus/execution wait states:
    -   `udiv r2, r3, r2` inside `gen_rand_sha`:
        $\Delta c_{\text{udiv}} = \lfloor (31 - \operatorname{ clz }(\texttt{r3})) / 4 \rfloor \in \{0, 1, 2, 3\}$
        data-dependent Cortex-M33 divider cycles (the firmware's `SH_JITTER` /
        `CK_JITTER` countermeasure);
    -   APB peripheral polling stalls on `TRNG_BUSY` (`+1` cycle on final poll
        of first 33 EHR blocks) and `SHA256_CSR` (`+8` APB bridge wait cycles per
        busy poll).

### Stage 2: Cycle Stretching & Amplitude Matching (`03_stretch_to_sedpack.py`)

Given `(pc_idx, rainbow_hw, hw_stall)` of length
$N_{\text{insn}} \approx 66,598$, `stretch_instruction_trace_to_cycles()`
converts the instruction sequence into a `97,553`-sample `int16` ADC trace in
four steps:

1.  **Per-instruction cycle count $c_i \ge 1$**:
    $$c_i = \text{base\_cycles}[\text{pc}_i] + \mathbb{I}[\text{cond\_taken}_i] \cdot \text{taken\_extra\_cycles}[\text{pc}_i] + \text{hw\_stall}[i] + \Delta c_{\text{cold}}(i)$$
    where $\text{cond\_taken}_i \iff \text{is\_cond}[\text{pc}_i] \land (\text{PC}_{i+1} \neq \text{PC}_i + \text{size}[\text{pc}_i])$,
    and $\Delta c_{\text{cold}}(i) = +9$ on the first invocation of
    `makesmallperm` ($0$ thereafter).
2.  **Piecewise-constant cycle stretching (`np.repeat`)**:
    Each instruction's scalar Hamming weight `rainbow_hw[i]` is held constant
    across its $c_i$ clock cycles (`stretched = np.repeat(rainbow_hw, c)`),
    expanding $\sim 66,598$ instructions into $\sum_i c_i \approx 95,239$
    active CPU  clock cycles (at 1 ADC sample = 1 CPU clock cycle at
    $24\,\text{MHz}$ synchronous ChipWhisperer Husky clocking).
3.  **Post-return idle tail padding**:
    After `decrypt` executes `bx lr` (`~95,239` cycles), the remaining
    $\sim 2,314$ samples up to the fixed hardware capture window
    (`97,553` samples) hold the final idle register Hamming weight.
4.  **ChipWhisperer shunt polarity, mean-amplitude matching, and white Gaussian noise**:
    $$\text{trace1}[t] = \operatorname{round}\!\Big(1584.411 + (-16.0)\cdot\big(\text{stretched}[t] - \mu_{\text{HW,ref}}\big) + \varepsilon_t\Big), \qquad \varepsilon_t \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, 25.0^2)$$
    where $1584.411$ matches the empirical DC mean of the real ChipWhisperer
    Husky dataset (`rp2350_aesctr_small_fixed/train`), $-16.0\,\text{ADC/bit}$
    models the negative-going resistive shunt voltage drop on higher current
    draw, and  $\mu_{\text{HW,ref}} \approx 9.379$ bits is calibrated once on
    an all-zero dummy run (zero secret dependence).

---

## 2. How `data/etm_cycle_model.json` Was Built

Rather than assigning a single context-free cycle count per ARM mnemonic (e.g.,
*"every `ldr` = 2 cycles"*), `data/etm_cycle_model.json` stores
`(base_cycles,taken_extra_cycles)` for each of the `1,153` static instruction
addresses (`0x20080000 .. 0x20080f12`) in `aes_bootloader.elf`.

It was constructed by constrained integer calibration
(`build_etm_timing_model.py`) combining three sources:

1.  **Cortex-M33 Pipeline Prior (`prior_base`)**:
    -   ALU / shifts / bitops / `str`: `1` cycle;
    -   `ldr` / `ldrb` / `ldrh`: `1` cycle when pipelined with an immediately
       following load/store, `2` cycles otherwise;
    -   `ldm` / `stm` / `push` / `pop` of $N$ registers: `1 + N` cycles (`2 + N`
       if writing `pc`, `1` for 1-reg `push`/`stm`);
    -   `b` / `bx` / `bl` / base `udiv`: `2` cycles; conditional branch: `1` cycle
       untaken, `+1` extra cycle taken.
2.  **Real RP2350 CoreSight ETMv4 Basic-Block Cycle Counts (`stream2_etmv4`)**:
    -   In the hardware dataset (`rp2350_aesctr_small_fixed/train`), ChipWhisperer
        TraceWhisperer captured CoreSight ETMv4 (`EV_CH_ADDRESS`,
        `EV_CH_ENATOMS`, `EV_CH_CYCLECOUNT`) alongside DWT watchpoints
        (`stream1_itm_dwt`).
    -   Across `decrypt(..., nb_blocks=2)`, ETMv4 pins down `>1,000` zero-variance
        basic-block cycle intervals totaling exactly
        **`81,253` canonical non-wait cycles** (e.g., `remap_1` = `30`
        cycles/word $\times 64$, `init_key_sbox` = `1,440` canonical
        cycles/word with `22`-cycle `pre_sbox` and `50`-cycle `sbox_win`
        byte loops, `CT_BPERM` = `42` cycles/Murmur3 round, `+17` cycles
        per `jstate` refill, `+188` cycles per SHA-256 hardware refill).
3.  **Minimum-$L_1$ Integer Adjustment Within Basic Blocks**:
    -   Whenever a basic block's ETMv4 hardware cycle total
        $\sum_{i \in \text{BB}} c_i$ differed from the sum of `prior_base`
        across that block, integer adjustments were solved to match every
        ETMv4 basic-block checkpoint with **zero cumulative cycle drift**
        across all `95,239` cycles.

---

## 3. Why a Pure "Cycles-per-Opcode" Model Drifts by Thousands of Cycles

Comparing `data/etm_cycle_model.json` against the Cortex-M33 pipelined opcode
prior across all `1,153` static instructions shows:

| Agreement Metric (`1,153` Static Instructions) | Count | Share |
| --- | ---: | ---: |
| Exact match with pipelined Cortex-M33 opcode prior ($\Delta = 0$) | `988` | **85.7%** |
| Within $\pm 1$ cycle of pipelined opcode prior ($\|\Delta\| \le 1$) | `1,103` | **95.7%** |
| Deviations with $\|\Delta\| \ge 2$ cycles (XIP flash / fetch alignment / BB boundary) | `50` | **4.3%** |

Why do those `14.3%` deviations exist on real RP2350 hardware, and what happens
if we replace `etm_cycle_model.json` with a pure opcode-only table?

1.  **Contest Hardware Executed `aes.S` from XIP QSPI Flash (`0x100014E8`),**
    **Not Zero-Wait-State SRAM**:
    -   In upstream `picotool`, `aes.S` is placed in `.scratch_x.aes`
        (`0x20080000`, zero-wait-state SRAM).
    -   However, in the contest capture firmware (`firmware/CMakeLists.txt`,
        line 71), `#pico_set_binary_type(rp2350_hacking_challenge_2 no_flash)`
        was commented out, linking `aes.S` into `.text` in
        **XIP QSPI Flash (`0x100014E8 .. 0x100023FC`)**.
    -   Consequently, the first time execution reaches a cold 8-byte XIP cache
        line or reads a literal pool in flash, the RP2350 stalls for QSPI serial
        flash refill (`+23` cycles at `0x20080000` `init_rstate` entry, `+21`
        cycles at `0x20080398`, `+9` cycles on first `makesmallperm`, `+6`
        cycles on flash literal loads `0x2008062e` / `0x20080b16`). Once cached
        in the 16 KB XIP cache, subsequent loop iterations run at 1-cycle cached
        speed. Whether a cache line hits or misses is
        **address- and control-flow-history-dependent**, impossible to express
        as `f(opcode)` alone.
2.  **32-Bit Thumb-2 Fetch Alignment & Neighboring-Instruction Pipeline Coupling**:
    -   On Cortex-M33, 32-bit Thumb-2 instructions (`size = 4`) straddling a
        32-bit word boundary (`PC % 4 == 2`) vs. word-aligned (`PC % 4 == 0`),
        as well as load-use interlocks and write-buffer stalls, depend on
        **instruction address parity and surrounding instructions** (for
        instance, even in real hardware, odd vs. even `m` iterations in
        `init_key_sbox` differ by `3` cycles!).
3.  **Cumulative Drift Breaks `end_to_end_attack.py`'s Hardcoded Cycle Landmarks**:
    Evaluating all three timing models on the exact same `66,608`-instruction
    execution trace (`test_run/raw_traces/sample_instruction_trace.csv`) yields:

   | Landmark Function (First Entry) | Executed Insn # | ETMv4 Model Cycle | ARM TRM Opcode Model (`ldr/str=2`) | Pipelined Opcode Model (`str=1, ldr-ldr=1`) |
   | --- | ---: | ---: | ---: | ---: |
   | `decrypt` | `0` | `0` | `0` (`+0`) | `0` (`+0`) |
   | `remap` | `9,959` | `15,985` | `17,302` (**`+1,317`**) | `16,143` (**`+158`**) |
   | `init_key_sbox` (`m=0`) | `15,257` | `23,516` | `25,387` (**`+1,871`**) | `23,478` (`-38`) |
   | `ctr_crypt_s` (after `m=0..12`) | `31,897` | `49,407` | `56,119` (**`+6,712`**) | `50,353` (**`+946`**) |
   | `rounds_s_mainloop` (Block 0) | `40,743` | `62,069` | `69,377` (**`+7,308`**) | `62,332` (**`+263`**) |
   | `decryption_end` (Block 0) | `51,630` | `75,705` | `86,036` (**`+10,331`**) | `77,040` (**`+1,335`**) |
   | Return (`bx lr`, end of Block 1) | `66,608` | **`95,237`** | **`108,794`** (**`+13,557`**, overflows `97,553`!) | **`97,214`** (**`+1,977`**) |

   Because the unmodified submission `end_to_end_attack.py` hardcodes the exact
   XIP-linked loop strides and search windows (`remap` `30` cycles/word
   $\times 64$, `init_key_sbox` `1,440` cycles/word and `50` cycles/S-box byte,
   `CT_BPERM` `42` cycles/Murmur3 round), an unpipelined opcode model overflows
   the `97,553`-sample capture buffer completely (truncating Block 1), while
   even a pipelined opcode model accumulates `+984` cycles of drift across the
   13 `init_key_sbox` words (`~75.7` extra cycles/word), desynchronizing
   `end_to_end_attack.py`'s cross-correlation and fixed-stride POI slicers.

---

## 4. Threats to Validity: Is Per-Address Calibration "Over-Fit" to the Bootloader?

**Yes — with respect to binary portability and microarchitectural generality.**
However, **no — with respect to secret-key leakage or artificially helping the attack.**
Both aspects must be stated transparently when interpreting results from this emulator:

### A. Genuine Threats to External / Construct Validity (Where It *Is* Over-Fit)

1.  **Non-Portability Across Recompilations or Link Layouts**:
    -   Because `data/etm_cycle_model.json` indexes `base_cycles` by static PC
        (`0x20080000 + offset`) rather than simulating the underlying RP2350 XIP
        cache + Cortex-M33 prefetch buffer mechanistically,
        **modifying `aes.S`, changing compiler flags (e.g., enabling**
        **`HARDENING=1`), inserting a single 2-byte `nop`, or linking `aes.S` into**
        **`.scratch_x` SRAM invalidates the cycle table**.
    -   Note that `end_to_end_attack.py` shares this exact same limitation: its
        hardcoded cycle constants (`12634`, `14500`, `20833 + 1440*m`,
        `39619 + 54*m`) are likewise over-fit to the contest's specific
        XIP-linked binary (`CMakeLists.txt` with `HARDENING=0` and
        `no_flash` disabled).
2.  **Sub-Basic-Block Cycle Attribution Ambiguity**:
    -   CoreSight ETMv4 observes exact elapsed cycles **per basic block**
        (between branches), not per individual instruction inside a straight-line
        block. For example, inside `gen_rand_sha_nonpres`'s prologue
        (`0x2008021c`), ETMv4 constrains the *sum* of cycles of the basic
        block, not whether a stall cycle occurred on `push` vs. a neighboring instruction.
    -   Consequently, within a straight-line basic block of $5\text{--}15$
        instructions, a register's Hamming weight can be shifted by
        $\pm 1\text{--}3$ clock cycles relative to physical silicon, and
        stretched over $c_i$ cycles if an instruction absorbed an XIP stall.
        Because `04_fit_train_templates.py` trains the linear/quadratic Walsh
        templates on `train` traces from this same timing model, internal
        consistency between `train` and `holdout` is preserved,
        **cross-domain transfer** (applying real-hardware `.pkl` templates
        directly to emulated `holdout` traces without re-profiling `train`)
        would suffer alignment mismatch inside basic blocks.

### B. Why Internal Side-Channel Validity Is Preserved (Not Cheating)

1.  **Zero Secret or Key Dependence in `etm_cycle_model.json`**:
    -   The lookup table is a pure static function of instruction address and
        control-flow branch outcome $( \text{PC}_i, \text{cond\_taken}_i )$.
        In `aes.S`, control-flow branches depend only on loop counters and
        public/PRNG refill counters — **never** on the secret key, Boolean
        key shares (`key_a..d`), IV, or plaintext.
2.  **All Countermeasure Timing Jitter Is Mechanistic, Not Tabulated**:
    -   Every dynamic timing countermeasure that `end_to_end_attack.py`'s
        blind resynchronizer must overcome on `holdout` — random `0..3` cycle
        `udiv` jitter (`SH_JITTER` / `CK_JITTER`) across `171` `gen_rand_sha`
        calls, random `+17`-cycle `jstate` refills, random `+188`-cycle
        hardware SHA-256 compression stalls, and random 2-block `CT_BPERM`
        permutation — is generated **live** by executing the unpatched ARM
        instructions and hardware MMIO state machines driven by fresh 256-bit
        `xoshiro256**` entropy per trace.
3.  **No Artificial Leakage Amplification on XIP-Stalled Instructions**:
    -   Checkpoints with large cold XIP stalls (`0x20080000` `+23` cycles in
        `init_rstate`, `0x20080398` `+21` cycles, `makesmallperm` `+9` cycles)
        occur exclusively during non-secret initialization/PRNG setup or
        zero-constant initialization (`mov r2, #0`), never on secret-key-share
        instructions. Within the attacked `remap_1` (`30` cyc/word) and
        `init_key_sbox` (`50` cyc/byte) hot loops, XIP cache hits prevail and
        instruction cycle counts match standard cached Cortex-M33 pipeline
        timing (`1–2` cycles/insn).

### C. Recommendation for a Fully Generalizable (Non-Overfit) Future Emulator

To eliminate per-address calibration altogether so the emulator generalizes to
arbitrary RP2350 binaries (including `picotool`'s `HARDENING=1` SRAM build),
`etm_cycle_model.json` should be replaced by a stateful microarchitectural
timing model implementing:

1.  **Cortex-M33 3-stage pipeline + 32-bit prefetch buffer** (tracking `PC % 4`
    16/32-bit fetch alignment, load-store pipelining, and load-use stalls from
    opcode + operand registers); and
2.  **RP2350 XIP cache + QSPI SSI model** (16 KB 2-way set-associative cache
    with 8-byte lines when executing from `0x10000000..0x13ffffff`, vs.
    0-wait-state AHB5 access when executing from `.scratch_x` SRAM `0x20080000`).
