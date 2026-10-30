/*
 * Copyright (C) 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Minimal, Hook-Free Cortex-M33 + RP2350 Hardware Emulator with Standard Rainbow
 * HammingWeight() Instruction Leakage (`rp2350_unicorn_tracer.c`).
 *
 * Stage 1 (Pure Instruction-Level Emulation):
 *   - Executes the unmodified RP2350 encrypted bootloader ELF (`aes_bootloader.elf`)
 *     on an ARMv8-M Cortex-M33 Thumb-2 Unicorn core with faithful RP2350 MMIO
 *     peripherals (`RESETS`, `ROSC`, `TRNG`, `SHA256`), where the CPU natively
 *     polls `TRNG_BUSY` and `SHA256_CSR.SUM_VLD`.
 *   - Zero RNG shortcuts/mocks and zero PC/address-based leakage-suppression hooks.
 *   - Emits 1 data point per executed instruction `i`:
 *       1. `pc_idx[i]`     = `(pc - 0x20080000) >> 1`
 *       2. `rainbow_hw[i]` = `sum_{r in regs_written(i)} popcount(reg[r])`
 *       3. `hw_stall[i]`   = `udiv_extra + apb_stall` (data/peripheral bus wait states)
 *
 * Cycle stretching to 97,553 ADC samples and `sedpack` dataset packaging are performed
 * separately in `03_stretch_to_sedpack.py` so the transformation from 1-point-per-instruction
 * Hamming-weight trace to hardware-shaped power trace is completely transparent.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unicorn/unicorn.h>

#define CODE_BASE       0x20080000u
#define CODE_SIZE       0x00002000u
#define DATA_BASE       0x20000000u
#define DATA_SIZE       0x00001000u
#define PERIPH_BASE     0x40020000u
#define PERIPH_SIZE     0x00100000u
#define RET_ADDR        0x20081FFEu

#define RESETS_BASE     0x40020000u
#define ROSC_BASE       0x400e8000u
#define TRNG_BASE       0x400f0000u
#define SHA256_BASE     0x400f8000u

#define MAX_INSN_IDX    4096
#define MAX_EXEC_INSNS  131072
#define MAX_DWT_EVENTS  2048

/* 17 ARM registers tracked by rainbow_arm: r0..r12, sp, lr, pc, cpsr */
#define NUM_ARM_REGS    17

typedef struct {
    uc_engine *uc;
    uc_hook hh_code;
    uc_hook hh_mem;

    uint8_t zero_mem[0x1000] __attribute__((aligned(64)));
    uint8_t code_init[CODE_SIZE] __attribute__((aligned(64)));
    uint8_t code_mem[CODE_SIZE] __attribute__((aligned(64)));
    uint8_t data_mem[DATA_SIZE] __attribute__((aligned(64)));
    uint8_t periph_mem[PERIPH_SIZE] __attribute__((aligned(64)));

    /* Per-PC Capstone metadata (`is_udiv` + `regs_written` bitmask) */
    uint8_t  is_udiv[MAX_INSN_IDX];
    uint32_t reg_write_mask[MAX_INSN_IDX];
    uint32_t decrypt_entry_pc;

    /* Physical entropy source state (xoshiro256**) backing TRNG & ROSC hardware */
    uint64_t hw_rng_s[4];

    /* RP2350 hardware peripheral registers */
    uint32_t resets_reg;
    uint32_t rosc_ctrl, rosc_random, rosc_dormant, rosc_div, rosc_phase, rosc_status;

    uint32_t trng_imr, trng_isr, trng_icr, trng_config, trng_rnd_en, trng_sample_cnt1, trng_debug;
    uint32_t trng_ehr[6];
    int      trng_busy_polls_left;
    int      trng_finish_after_insn;
    int      trng_sample_index;

    uint32_t sha_csr;
    uint32_t sha_h[8];
    uint32_t sha_wbuf[16];
    int      sha_wlen;
    int      sha_busy_polls_left;
    int      sha_finish_after_insn;

    /* CPU register snapshot buffer */
    int      reg_ids[NUM_ARM_REGS];
    void    *reg_ptrs[NUM_ARM_REGS];
    uint32_t cur_regs[NUM_ARM_REGS];

    /* Instruction execution tracking (1 entry per executed instruction) */
    int      has_prev;
    uint32_t prev_pc;
    int      prev_udiv_extra;
    int      prev_apb_stall;
    int      insn_count;

    uint16_t pc_idx_buf[MAX_EXEC_INSNS];
    uint16_t rainbow_hw_buf[MAX_EXEC_INSNS];
    uint8_t  hw_stall_buf[MAX_EXEC_INSNS];

    /* Optional DWT watchpoints (enabled ONLY for `train` ground-truth alignment; 0 on `holdout`) */
    int      record_dwt;
    uint32_t dwt_addr0; /* rstate_sha */
    uint32_t dwt_addr1; /* jstate */
    uint32_t dwt_addr2; /* rstate_lfsr */
    uint8_t  dwt_comp[MAX_DWT_EVENTS];
    uint32_t dwt_val[MAX_DWT_EVENTS];
    int      dwt_count;
} Rp2350UnicornTracer;

static inline uint64_t rotl64(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t xoshiro_step(uint64_t s[4]) {
    const uint64_t result = rotl64(s[1] * 5ULL, 7) * 9ULL;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl64(s[3], 45);
    return result;
}

static inline void write_periph_u32(Rp2350UnicornTracer *ctx, uint32_t addr, uint32_t val) {
    if (addr >= PERIPH_BASE && addr + 4u <= PERIPH_BASE + PERIPH_SIZE) {
        memcpy(ctx->periph_mem + (addr - PERIPH_BASE), &val, 4);
    }
}

static inline uint32_t rosc_step_lfsr(Rp2350UnicornTracer *ctx) {
    uint32_t s = ctx->rosc_random;
    if (s == 0u) s = 0x3f04b16du;
    uint32_t msb = (s >> 31) & 1u;
    s = (s << 1) ^ (msb ? 0x3f04b16du : 0u);
    s ^= (uint32_t)(xoshiro_step(ctx->hw_rng_s) & 1u);
    if (s == 0u) s = 1u;
    ctx->rosc_random = s;
    write_periph_u32(ctx, ROSC_BASE + 0x0cu, s);
    write_periph_u32(ctx, ROSC_BASE + 0x20u, s & 1u);
    return s;
}

static void trng_start_sampling(Rp2350UnicornTracer *ctx) {
    ctx->trng_busy_polls_left = 37;
    ctx->trng_finish_after_insn = 0;
    ctx->trng_isr &= ~1u;
    write_periph_u32(ctx, TRNG_BASE + 0x104u, ctx->trng_isr);
    write_periph_u32(ctx, TRNG_BASE + 0x110u, 0u); /* TRNG_VALID = 0 */
    write_periph_u32(ctx, TRNG_BASE + 0x1b8u, 1u); /* TRNG_BUSY  = 1 */
}

static void trng_latch_ehr(Rp2350UnicornTracer *ctx) {
    for (int i = 0; i < 6; i++) {
        uint32_t v = (uint32_t)(xoshiro_step(ctx->hw_rng_s) >> 32);
        ctx->trng_ehr[i] = v;
        write_periph_u32(ctx, TRNG_BASE + 0x114u + 4u * (uint32_t)i, v);
    }
    ctx->trng_busy_polls_left = 0;
    ctx->trng_finish_after_insn = 0;
    ctx->trng_isr |= 1u;
    write_periph_u32(ctx, TRNG_BASE + 0x104u, ctx->trng_isr);
    write_periph_u32(ctx, TRNG_BASE + 0x110u, 1u); /* TRNG_VALID = 1 */
    write_periph_u32(ctx, TRNG_BASE + 0x1b8u, 0u); /* TRNG_BUSY  = 0 */
    ctx->trng_sample_index++;
}

static void trng_reset(Rp2350UnicornTracer *ctx) {
    ctx->trng_imr = 0x0000000fu;
    ctx->trng_isr = 0u;
    ctx->trng_icr = 0u;
    ctx->trng_config = 0u;
    ctx->trng_rnd_en = 0u;
    ctx->trng_sample_cnt1 = 0x0000ffffu;
    ctx->trng_debug = 0u;
    ctx->trng_busy_polls_left = 0;
    ctx->trng_finish_after_insn = 0;
    for (int i = 0; i < 6; i++) {
        ctx->trng_ehr[i] = 0u;
        write_periph_u32(ctx, TRNG_BASE + 0x114u + 4u * (uint32_t)i, 0u);
    }
    write_periph_u32(ctx, TRNG_BASE + 0x100u, ctx->trng_imr);
    write_periph_u32(ctx, TRNG_BASE + 0x104u, ctx->trng_isr);
    write_periph_u32(ctx, TRNG_BASE + 0x108u, 0u);
    write_periph_u32(ctx, TRNG_BASE + 0x10cu, ctx->trng_config);
    write_periph_u32(ctx, TRNG_BASE + 0x110u, 0u);
    write_periph_u32(ctx, TRNG_BASE + 0x12cu, ctx->trng_rnd_en);
    write_periph_u32(ctx, TRNG_BASE + 0x130u, ctx->trng_sample_cnt1);
    write_periph_u32(ctx, TRNG_BASE + 0x138u, ctx->trng_debug);
    write_periph_u32(ctx, TRNG_BASE + 0x140u, 0u);
    write_periph_u32(ctx, TRNG_BASE + 0x1b8u, 0u);
}

static const uint32_t SHA256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

static inline uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

static void sha256_sync_regs(Rp2350UnicornTracer *ctx) {
    write_periph_u32(ctx, SHA256_BASE + 0x00u, ctx->sha_csr);
    for (int i = 0; i < 8; i++) {
        write_periph_u32(ctx, SHA256_BASE + 0x08u + 4u * (uint32_t)i, ctx->sha_h[i]);
    }
}

static void sha256_init_state(Rp2350UnicornTracer *ctx) {
    ctx->sha_h[0] = 0x6a09e667u;
    ctx->sha_h[1] = 0xbb67ae85u;
    ctx->sha_h[2] = 0x3c6ef372u;
    ctx->sha_h[3] = 0xa54ff53au;
    ctx->sha_h[4] = 0x510e527fu;
    ctx->sha_h[5] = 0x9b05688cu;
    ctx->sha_h[6] = 0x1f83d9abu;
    ctx->sha_h[7] = 0x5be0cd19u;
}

static void sha256_reset(Rp2350UnicornTracer *ctx) {
    ctx->sha_csr = 0x00001206u;
    ctx->sha_wlen = 0;
    ctx->sha_busy_polls_left = 0;
    ctx->sha_finish_after_insn = 0;
    sha256_init_state(ctx);
    sha256_sync_regs(ctx);
}

static void sha256_compress_block(Rp2350UnicornTracer *ctx) {
    uint32_t W[64];
    int bswap = (ctx->sha_csr & 0x00001000u) != 0;
    for (int i = 0; i < 16; i++) {
        uint32_t w = ctx->sha_wbuf[i];
        W[i] = bswap ? w : __builtin_bswap32(w);
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr32(W[i - 15], 7) ^ rotr32(W[i - 15], 18) ^ (W[i - 15] >> 3);
        uint32_t s1 = rotr32(W[i - 2], 17) ^ rotr32(W[i - 2], 19) ^ (W[i - 2] >> 10);
        W[i] = W[i - 16] + s0 + W[i - 7] + s1;
    }
    uint32_t a = ctx->sha_h[0], b = ctx->sha_h[1], c = ctx->sha_h[2], d = ctx->sha_h[3];
    uint32_t e = ctx->sha_h[4], f = ctx->sha_h[5], g = ctx->sha_h[6], h = ctx->sha_h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + S1 + ch + SHA256_K[i] + W[i];
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;
        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }
    ctx->sha_h[0] += a; ctx->sha_h[1] += b; ctx->sha_h[2] += c; ctx->sha_h[3] += d;
    ctx->sha_h[4] += e; ctx->sha_h[5] += f; ctx->sha_h[6] += g; ctx->sha_h[7] += h;
    ctx->sha_wlen = 0;

    ctx->sha_busy_polls_left = 8;
    ctx->sha_finish_after_insn = 0;
    ctx->sha_csr &= ~0x00000004u; /* SUM_VLD = 0 until hardware compression finishes */
}

static void periph_reset_all(Rp2350UnicornTracer *ctx) {
    memset(ctx->periph_mem, 0, PERIPH_SIZE);
    ctx->resets_reg = 0xffffffffu;
    write_periph_u32(ctx, RESETS_BASE + 0x00u, ctx->resets_reg);

    ctx->rosc_ctrl = 0x00000aa0u;
    ctx->rosc_random = (uint32_t)xoshiro_step(ctx->hw_rng_s) | 1u;
    ctx->rosc_dormant = 0x77616b65u;
    ctx->rosc_div = 0xaa20u;
    ctx->rosc_phase = 0x00000008u;
    ctx->rosc_status = 0x80001000u;
    write_periph_u32(ctx, ROSC_BASE + 0x00u, ctx->rosc_ctrl);
    write_periph_u32(ctx, ROSC_BASE + 0x0cu, ctx->rosc_random);
    write_periph_u32(ctx, ROSC_BASE + 0x10u, ctx->rosc_dormant);
    write_periph_u32(ctx, ROSC_BASE + 0x14u, ctx->rosc_div);
    write_periph_u32(ctx, ROSC_BASE + 0x18u, ctx->rosc_phase);
    write_periph_u32(ctx, ROSC_BASE + 0x1cu, ctx->rosc_status);
    write_periph_u32(ctx, ROSC_BASE + 0x20u, ctx->rosc_random & 1u);

    ctx->trng_sample_index = 0;
    trng_reset(ctx);
    sha256_reset(ctx);
}

static void periph_handle_write(Rp2350UnicornTracer *ctx, uint32_t addr, uint32_t val) {
    if (addr >= TRNG_BASE && addr < TRNG_BASE + 0x1000u) {
        uint32_t off = addr - TRNG_BASE;
        if (off == 0x100u) {
            ctx->trng_imr = val;
            write_periph_u32(ctx, addr, val);
        } else if (off == 0x108u) {
            ctx->trng_icr = val;
            ctx->trng_isr &= ~val;
            write_periph_u32(ctx, TRNG_BASE + 0x104u, ctx->trng_isr);
            write_periph_u32(ctx, addr, 0u);
            if ((val & 1u) && (ctx->trng_rnd_en & 1u)) {
                trng_start_sampling(ctx);
            }
        } else if (off == 0x10cu) {
            ctx->trng_config = val;
            write_periph_u32(ctx, addr, val);
        } else if (off == 0x12cu) {
            uint32_t prev_en = ctx->trng_rnd_en & 1u;
            ctx->trng_rnd_en = val;
            write_periph_u32(ctx, addr, val);
            if ((val & 1u) && !prev_en) {
                trng_start_sampling(ctx);
            } else if (!(val & 1u)) {
                ctx->trng_busy_polls_left = 0;
                ctx->trng_finish_after_insn = 0;
                write_periph_u32(ctx, TRNG_BASE + 0x1b8u, 0u);
            }
        } else if (off == 0x130u) {
            ctx->trng_sample_cnt1 = val;
            write_periph_u32(ctx, addr, val);
        } else if (off == 0x138u) {
            ctx->trng_debug = val;
            write_periph_u32(ctx, addr, val);
        } else if (off == 0x140u) {
            if (val & 1u) trng_reset(ctx);
            write_periph_u32(ctx, addr, 0u);
        }
    } else if (addr >= SHA256_BASE && addr < SHA256_BASE + 0x1000u) {
        uint32_t off = addr - SHA256_BASE;
        if (off == 0x00u) {
            if (val & 1u) {
                sha256_init_state(ctx);
                ctx->sha_wlen = 0;
                ctx->sha_busy_polls_left = 0;
                ctx->sha_finish_after_insn = 0;
                ctx->sha_csr = (val & ~1u) | 0x00000006u;
            } else {
                ctx->sha_csr = (ctx->sha_csr & ~0x1300u) | (val & 0x1300u);
            }
            sha256_sync_regs(ctx);
        } else if (off == 0x04u) {
            ctx->sha_csr &= ~0x00000004u;
            if (ctx->sha_wlen < 16) {
                ctx->sha_wbuf[ctx->sha_wlen++] = val;
            }
            if (ctx->sha_wlen >= 16) {
                sha256_compress_block(ctx);
            }
            write_periph_u32(ctx, addr, 0u);
            sha256_sync_regs(ctx);
        }
    } else if (addr >= ROSC_BASE && addr < ROSC_BASE + 0x1000u) {
        uint32_t off = addr - ROSC_BASE;
        if (off == 0x0cu) {
            ctx->rosc_random = val ? val : 0x3f04b16du;
            write_periph_u32(ctx, addr, ctx->rosc_random);
            write_periph_u32(ctx, ROSC_BASE + 0x20u, ctx->rosc_random & 1u);
        } else if (off <= 0x18u) {
            write_periph_u32(ctx, addr, val);
        }
    } else if (addr == RESETS_BASE) {
        if ((ctx->resets_reg & 0x00020000u) && !(val & 0x00020000u)) {
            sha256_reset(ctx);
        }
        if ((ctx->resets_reg & 0x02000000u) && !(val & 0x02000000u)) {
            trng_reset(ctx);
        }
        ctx->resets_reg = val;
        write_periph_u32(ctx, addr, val);
    }
}

static void periph_handle_read(Rp2350UnicornTracer *ctx, uint32_t addr) {
    if (addr >= TRNG_BASE && addr < TRNG_BASE + 0x1000u) {
        uint32_t off = addr - TRNG_BASE;
        if (off == 0x1b8u) {
            if (ctx->trng_busy_polls_left > 0) {
                ctx->trng_busy_polls_left--;
                if (ctx->trng_busy_polls_left == 0) {
                    ctx->trng_finish_after_insn = 1;
                    if (ctx->trng_sample_index < 33) {
                        ctx->prev_apb_stall += 1;
                    }
                }
                if (ctx->sha_busy_polls_left > 0) {
                    ctx->sha_busy_polls_left = 0;
                    ctx->sha_csr |= 0x00000004u;
                    write_periph_u32(ctx, SHA256_BASE + 0x00u, ctx->sha_csr);
                }
            }
        } else if (off == 0x128u) {
            if (ctx->trng_rnd_en & 1u) {
                ctx->trng_isr &= ~1u;
                write_periph_u32(ctx, TRNG_BASE + 0x104u, ctx->trng_isr);
                write_periph_u32(ctx, TRNG_BASE + 0x110u, 0u);
            }
        }
    } else if (addr >= SHA256_BASE && addr < SHA256_BASE + 0x1000u) {
        uint32_t off = addr - SHA256_BASE;
        if (off == 0x00u) {
            if (ctx->sha_busy_polls_left > 0) {
                ctx->sha_busy_polls_left--;
                ctx->prev_apb_stall += 8;
                if (ctx->sha_busy_polls_left == 0) {
                    ctx->sha_finish_after_insn = 1;
                }
            }
        }
    } else if (addr >= ROSC_BASE && addr < ROSC_BASE + 0x1000u) {
        uint32_t off = addr - ROSC_BASE;
        if (off == 0x0cu || off == 0x20u) {
            rosc_step_lfsr(ctx);
        }
    }
}

static void finalize_prev_insn(Rp2350UnicornTracer *ctx) {
    uint32_t ppc = ctx->prev_pc;
    if (ppc < CODE_BASE || ppc >= CODE_BASE + CODE_SIZE) {
        ctx->has_prev = 0;
        ctx->prev_apb_stall = 0;
        return;
    }

    uint32_t pidx = (ppc - CODE_BASE) >> 1;
    uint32_t mask = ctx->reg_write_mask[pidx];
    int hw = 0;
    if (mask != 0u) {
        uc_reg_read_batch(ctx->uc, ctx->reg_ids, ctx->reg_ptrs, NUM_ARM_REGS);
        while (mask) {
            int r = __builtin_ctz(mask);
            hw += __builtin_popcount(ctx->cur_regs[r]);
            mask &= (mask - 1u);
        }
    }

    if (ctx->insn_count < MAX_EXEC_INSNS) {
        int idx = ctx->insn_count++;
        ctx->pc_idx_buf[idx] = (uint16_t)pidx;
        ctx->rainbow_hw_buf[idx] = (uint16_t)hw;
        ctx->hw_stall_buf[idx] = (uint8_t)(ctx->prev_udiv_extra + ctx->prev_apb_stall);
    }

    ctx->prev_apb_stall = 0;
    ctx->has_prev = 0;

    if (ctx->trng_finish_after_insn) {
        trng_latch_ehr(ctx);
    }
    if (ctx->sha_finish_after_insn) {
        ctx->sha_finish_after_insn = 0;
        ctx->sha_csr |= 0x00000004u;
        write_periph_u32(ctx, SHA256_BASE + 0x00u, ctx->sha_csr);
    }
}

static void mem_hook_cb(
    uc_engine *uc,
    uc_mem_type type,
    uint64_t address,
    int size,
    int64_t value,
    void *user_data
) {
    (void)uc;
    Rp2350UnicornTracer *ctx = (Rp2350UnicornTracer *)user_data;
    uint32_t addr = (uint32_t)address;

    if (addr >= PERIPH_BASE && addr < PERIPH_BASE + PERIPH_SIZE) {
        if (type == UC_MEM_WRITE) {
            periph_handle_write(ctx, addr, (uint32_t)value);
        } else if (type == UC_MEM_READ) {
            periph_handle_read(ctx, addr);
        }
    }

    if (type == UC_MEM_WRITE && ctx->record_dwt && ctx->dwt_count < MAX_DWT_EVENTS) {
        uint32_t vmask = (size >= 4) ? 0xFFFFFFFFu : ((1u << (8 * size)) - 1u);
        uint32_t v = ((uint32_t)value) & vmask;
        if (addr == ctx->dwt_addr0) {
            ctx->dwt_comp[ctx->dwt_count] = 0;
            ctx->dwt_val[ctx->dwt_count++] = v;
        } else if (addr == ctx->dwt_addr1) {
            ctx->dwt_comp[ctx->dwt_count] = 1;
            ctx->dwt_val[ctx->dwt_count++] = v;
        } else if (addr == ctx->dwt_addr2) {
            ctx->dwt_comp[ctx->dwt_count] = 2;
            ctx->dwt_val[ctx->dwt_count++] = v;
        }
    }
}

static void code_hook_cb(uc_engine *uc, uint64_t address, uint32_t size, void *user_data) {
    (void)size;
    Rp2350UnicornTracer *ctx = (Rp2350UnicornTracer *)user_data;
    uint32_t pc = (uint32_t)address;

    if (ctx->has_prev) {
        finalize_prev_insn(ctx);
    }
    if (pc == RET_ADDR) {
        uc_emu_stop(uc);
        return;
    }

    ctx->prev_pc = pc;
    ctx->has_prev = 1;
    ctx->prev_apb_stall = 0;

    if (pc >= CODE_BASE && pc < CODE_BASE + CODE_SIZE) {
        uint32_t pidx = (pc - CODE_BASE) >> 1;
        if (ctx->is_udiv[pidx]) {
            uint32_t r3_v = 0u;
            uc_reg_read(uc, UC_ARM_REG_R3, &r3_v);
            ctx->prev_udiv_extra = (r3_v <= 1u) ? 0 : ((31 - __builtin_clz(r3_v)) >> 2);
        } else {
            ctx->prev_udiv_extra = 0;
        }
    } else {
        ctx->prev_udiv_extra = 0;
    }
}

Rp2350UnicornTracer *rp2350_tracer_create(
    const uint8_t *sram_image,
    int sram_image_len,
    uint32_t decrypt_entry_pc,
    uint32_t dwt_addr0,
    uint32_t dwt_addr1,
    uint32_t dwt_addr2,
    const uint8_t *is_udiv,
    const uint32_t *reg_write_mask
) {
    Rp2350UnicornTracer *ctx = (Rp2350UnicornTracer *)calloc(1, sizeof(Rp2350UnicornTracer));
    if (!ctx) return NULL;

    if (uc_open(UC_ARCH_ARM, UC_MODE_THUMB | UC_MODE_MCLASS, &ctx->uc) != UC_ERR_OK) {
        free(ctx);
        return NULL;
    }

    memset(ctx->code_init, 0, CODE_SIZE);
    if (sram_image_len > (int)CODE_SIZE) sram_image_len = (int)CODE_SIZE;
    memcpy(ctx->code_init, sram_image, (size_t)sram_image_len);

    uc_mem_map_ptr(ctx->uc, 0x00000000u, 0x1000u, UC_PROT_ALL, ctx->zero_mem);
    uc_mem_map_ptr(ctx->uc, CODE_BASE, CODE_SIZE, UC_PROT_ALL, ctx->code_mem);
    uc_mem_map_ptr(ctx->uc, DATA_BASE, DATA_SIZE, UC_PROT_ALL, ctx->data_mem);
    uc_mem_map_ptr(ctx->uc, PERIPH_BASE, PERIPH_SIZE, UC_PROT_ALL, ctx->periph_mem);

    ctx->decrypt_entry_pc = decrypt_entry_pc;
    ctx->dwt_addr0 = dwt_addr0;
    ctx->dwt_addr1 = dwt_addr1;
    ctx->dwt_addr2 = dwt_addr2;

    memcpy(ctx->is_udiv, is_udiv, MAX_INSN_IDX);
    memcpy(ctx->reg_write_mask, reg_write_mask, MAX_INSN_IDX * sizeof(uint32_t));

    for (int r = 0; r < 13; r++) {
        ctx->reg_ids[r] = UC_ARM_REG_R0 + r;
    }
    ctx->reg_ids[13] = UC_ARM_REG_SP;
    ctx->reg_ids[14] = UC_ARM_REG_LR;
    ctx->reg_ids[15] = UC_ARM_REG_PC;
    ctx->reg_ids[16] = UC_ARM_REG_CPSR;
    for (int r = 0; r < NUM_ARM_REGS; r++) {
        ctx->reg_ptrs[r] = &ctx->cur_regs[r];
    }

    uc_hook_add(ctx->uc, &ctx->hh_mem, UC_HOOK_MEM_READ | UC_HOOK_MEM_WRITE, (void *)mem_hook_cb, ctx, 1, 0);
    uc_hook_add(ctx->uc, &ctx->hh_code, UC_HOOK_CODE, (void *)code_hook_cb, ctx, CODE_BASE, CODE_BASE + CODE_SIZE);

    return ctx;
}

int rp2350_tracer_run_one(
    Rp2350UnicornTracer *ctx,
    const uint8_t *key_4way_128,
    const uint8_t *iv_salt_16,
    const uint8_t *iv_16,
    const uint8_t *pt_32,
    uint32_t nb_blocks,
    const uint64_t *rng_seed_4,
    int record_dwt,
    uint16_t *out_pc_idx,
    uint16_t *out_rainbow_hw,
    uint8_t *out_hw_stall,
    uint8_t *out_ct_32,
    uint8_t *out_dwt_comp,
    uint32_t *out_dwt_val,
    int *out_dwt_count
) {
    if (!ctx) return -1;

    memcpy(ctx->hw_rng_s, rng_seed_4, 4 * sizeof(uint64_t));
    if ((ctx->hw_rng_s[0] | ctx->hw_rng_s[1] | ctx->hw_rng_s[2] | ctx->hw_rng_s[3]) == 0ULL) {
        ctx->hw_rng_s[0] = 0x9e3779b97f4a7c15ULL;
    }

    memcpy(ctx->code_mem, ctx->code_init, CODE_SIZE);
    memset(ctx->data_mem, 0, DATA_SIZE);
    memcpy(ctx->data_mem + 0x000u, key_4way_128, 128);
    memcpy(ctx->data_mem + 0x100u, iv_salt_16, 16);
    memcpy(ctx->data_mem + 0x120u, iv_16, 16);
    memcpy(ctx->data_mem + 0x200u, pt_32, 32);

    uint32_t sp = (CODE_BASE + CODE_SIZE) - 16u;
    memcpy(ctx->code_mem + (sp - CODE_BASE), &nb_blocks, 4);
    periph_reset_all(ctx);

    uint32_t zero = 0u;
    for (int r = UC_ARM_REG_R4; r <= UC_ARM_REG_R12; r++) {
        uc_reg_write(ctx->uc, r, &zero);
    }
    uint32_t r0 = DATA_BASE + 0x000u;
    uint32_t r1 = DATA_BASE + 0x100u;
    uint32_t r2 = DATA_BASE + 0x120u;
    uint32_t r3 = DATA_BASE + 0x200u;
    uint32_t lr = RET_ADDR | 1u;
    uint32_t xpsr = 0x01000000u;
    uc_reg_write(ctx->uc, UC_ARM_REG_R0, &r0);
    uc_reg_write(ctx->uc, UC_ARM_REG_R1, &r1);
    uc_reg_write(ctx->uc, UC_ARM_REG_R2, &r2);
    uc_reg_write(ctx->uc, UC_ARM_REG_R3, &r3);
    uc_reg_write(ctx->uc, UC_ARM_REG_SP, &sp);
    uc_reg_write(ctx->uc, UC_ARM_REG_LR, &lr);
    uc_reg_write(ctx->uc, UC_ARM_REG_APSR_NZCVQG, &zero);
    uc_reg_write(ctx->uc, UC_ARM_REG_XPSR_G, &zero);
    uc_reg_write(ctx->uc, UC_ARM_REG_CPSR, &xpsr);
    uc_reg_write(ctx->uc, UC_ARM_REG_XPSR, &xpsr);

    ctx->has_prev = 0;
    ctx->prev_pc = 0;
    ctx->prev_udiv_extra = 0;
    ctx->prev_apb_stall = 0;
    ctx->insn_count = 0;
    ctx->record_dwt = record_dwt;
    ctx->dwt_count = 0;

    uc_err err = uc_emu_start(ctx->uc, ctx->decrypt_entry_pc | 1u, RET_ADDR, 0, 0);
    if (ctx->has_prev) {
        finalize_prev_insn(ctx);
    }
    if (err != UC_ERR_OK) {
        return -(int)err;
    }

    int n = ctx->insn_count;
    if (n >= MAX_EXEC_INSNS) {
        return -999; /* Buffer overflow guard: never silently truncate instruction trace */
    }
    memcpy(out_pc_idx, ctx->pc_idx_buf, (size_t)n * sizeof(uint16_t));
    memcpy(out_rainbow_hw, ctx->rainbow_hw_buf, (size_t)n * sizeof(uint16_t));
    memcpy(out_hw_stall, ctx->hw_stall_buf, (size_t)n * sizeof(uint8_t));
    memcpy(out_ct_32, ctx->data_mem + 0x200u, 32);

    if (record_dwt && out_dwt_comp && out_dwt_val && out_dwt_count) {
        *out_dwt_count = ctx->dwt_count;
        memcpy(out_dwt_comp, ctx->dwt_comp, (size_t)ctx->dwt_count);
        memcpy(out_dwt_val, ctx->dwt_val, (size_t)ctx->dwt_count * sizeof(uint32_t));
    }
    return n;
}

void rp2350_tracer_destroy(Rp2350UnicornTracer *ctx) {
    if (!ctx) return;
    if (ctx->uc) {
        uc_close(ctx->uc);
    }
    free(ctx);
}
