#include <hardware/platform_defs.h>
#include <pico.h>
#include <pico/types.h>
#define PICO_STDIO_USB_CONNECT_WAIT_TIMEOUT_MS 100

#include "boot/picoboot_constants.h"
#include "pico/bootrom.h"
#include "pico/status_led.h"
#include "pico/stdlib.h"
#include <pico/stdio.h>
#include <pico/stdio_uart.h>
#include <pico/stdio_usb.h>
#include <stdbool.h>
#include <stdint.h>

#include <mbedtls/aes.h>

#include "hardware/adc.h"
#include "hardware/clocks.h"
#include "hardware/pll.h"
#include "hardware/regs/addressmap.h"
#include "hardware/resets.h"
#include "hardware/structs/clocks.h"
#include "hardware/structs/pll.h"
#include "hardware/ticks.h"

// Cortex-M33
#define TPIU_BASE (CORESIGHT_TPIU_BASE) /*!< 0x40148000 */
#define ETM_VERSION 4
#include "arm_etm.h"
#define TSGEN ((TSGEN_Type *)CORESIGHT_TIMESTAMP_GEN_BASE)
#define ATB_FUNNEL ((ATB_FUNNEL_Type *)CORESIGHT_ATB_FUNNEL_BASE)

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hal.h"
#include "simpleserial.h"
#include "tinyaes.h"

#ifndef UART_BAUD_RATE
#define UARUART_BAUD_RATE 38400
#endif

// Define PIN numbers
#define TRIGGER 16

char scratch_backup[2048];

// Code for ETM tracing support
volatile bool pcsamp_enable = 0;
volatile bool etm_gate_on_trigger = 0;

uint8_t setreg(uint8_t x, uint32_t val) {
  // Must match capture/trace/TraceWhisperer.py:
  // 0:  DWT->CTRL
  // 1:  DWT->COMP0
  // 2:  DWT->COMP1
  // 3:  ETM->CR (ETMv3) / ETM->CONFIGR (ETMv4: BB=bit3, CCI=bit4)
  // 4:  ETM->TESSEICR (ETMv3) / ETM->CCCTLR (ETMv4: cycle count threshold >= 4)
  // 5:  ETM->TEEVR (ETMv3) / ETM->EVENTCTLR0 (ETMv4)
  // 6:  ETM->TECR1 (ETMv3) / ETM->VICTLR (ETMv4)
  // 7:  ETM->TRACEIDR
  // 8:  TPI->ACPR
  // 9:  TPI->SPPR
  // 10: TPI->FFCR
  // 11: TPI->CSPSR
  // 12: ITM->TCR
  // 13: DWT->COMP2
  // 14: DWT->COMP3
  // 15: DWT->FUNCTION0
  // 16: DWT->FUNCTION1
  // 17: DWT->FUNCTION2
  // 18: DWT->FUNCTION3
  // 19: ETM->STALLCTLR (ETMv4)
  // 20: etm_gate_on_trigger (0 = ETM always in TraceMode, 1 = ETM active only between trigger_high/trigger_low)
  switch (x) {
  case 0:
    DWT->CTRL = val;
    break;
  case 1:
    DWT->COMP0 = val;
    break;
  case 2:
    DWT->COMP1 = val;
    break;
  case 3:
    ETM_SetupMode();
#if ETM_VERSION == 3
    ETM->CR = val;
#elif ETM_VERSION == 4
    ETM->CONFIGR = val;
#endif
    if (!etm_gate_on_trigger) {
      ETM_TraceMode();
    }
    break;
  case 4:
    ETM_SetupMode();
#if ETM_VERSION == 3
    ETM->TESSEICR = val;
#elif ETM_VERSION == 4
    ETM->CCCTLR = val;
#endif
    if (!etm_gate_on_trigger) {
      ETM_TraceMode();
    }
    break;
  case 5:
    ETM_SetupMode();
#if ETM_VERSION == 3
    ETM->TEEVR = val;
#elif ETM_VERSION == 4
    ETM->EVENTCTLR0 = val;
#endif
    if (!etm_gate_on_trigger) {
      ETM_TraceMode();
    }
    break;
  case 6:
    ETM_SetupMode();
#if ETM_VERSION == 3
    ETM->TECR1 = val;
#elif ETM_VERSION == 4
    ETM->VICTLR = val;
#endif
    if (!etm_gate_on_trigger) {
      ETM_TraceMode();
    }
    break;
  case 7:
    ETM_SetupMode();
    ETM->TRACEIDR = val;
    if (!etm_gate_on_trigger) {
      ETM_TraceMode();
    }
    break;
  case 8:
    TPI->ACPR = val;
    break;
  case 9:
    TPI->SPPR = val;
    break;
  case 10:
    TPI->FFCR = val;
    break;
  case 11:
    TPI->CSPSR = val;
    break;
  case 12:
    ITM->TCR = val;
    break;
  case 13:
    DWT->COMP2 = val;
    break;
  case 14:
    DWT->COMP3 = val;
    break;
  case 15:
    DWT->FUNCTION0 = val;
    break;
  case 16:
    DWT->FUNCTION1 = val;
    break;
  case 17:
    DWT->FUNCTION2 = val;
    break;
  case 18:
    DWT->FUNCTION3 = val;
    break;
  case 19:
#if ETM_VERSION == 4
    ETM_SetupMode();
    ETM->STALLCTLR = val;
    if (!etm_gate_on_trigger) {
      ETM_TraceMode();
    }
#endif
    break;
  case 20:
    etm_gate_on_trigger = (val != 0);
    if (etm_gate_on_trigger) {
      ETM_SetupMode();
    } else {
      ETM_TraceMode();
    }
    break;
  default:
    return 0x06;
  }
  return 0x00;
}

uint8_t getreg(uint8_t x, uint32_t *val) {
  if (val == NULL) {
    return 0x10;
  }
  switch (x) {
  case 0:
    *val = DWT->CTRL;
    break;
  case 1:
    *val = DWT->COMP0;
    break;
  case 2:
    *val = DWT->COMP1;
    break;
  case 3:
#if ETM_VERSION == 3
    *val = ETM->CR;
#elif ETM_VERSION == 4
    *val = ETM->CONFIGR;
#endif
    break;
  case 4:
#if ETM_VERSION == 3
    *val = ETM->TESSEICR;
#elif ETM_VERSION == 4
    *val = ETM->CCCTLR;
#endif
    break;
  case 5:
#if ETM_VERSION == 3
    *val = ETM->TEEVR;
#elif ETM_VERSION == 4
    *val = ETM->EVENTCTLR0;
#endif
    break;
  case 6:
#if ETM_VERSION == 3
    *val = ETM->TECR1;
#elif ETM_VERSION == 4
    *val = ETM->VICTLR;
#endif
    break;
  case 7:
    *val = ETM->TRACEIDR;
    break;
  case 8:
    *val = TPI->ACPR;
    break;
  case 9:
    *val = TPI->SPPR;
    break;
  case 10:
    *val = TPI->FFCR;
    break;
  case 11:
    *val = TPI->CSPSR;
    break;
  case 12:
    *val = ITM->TCR;
    break;
  case 13:
    *val = DWT->COMP2;
    break;
  case 14:
    *val = DWT->COMP3;
    break;
  case 15:
    *val = DWT->FUNCTION0;
    break;
  case 16:
    *val = DWT->FUNCTION1;
    break;
  case 17:
    *val = DWT->FUNCTION2;
    break;
  case 18:
    *val = DWT->FUNCTION3;
    break;
  case 19:
#if ETM_VERSION == 4
    *val = ETM->STALLCTLR;
#else
    *val = 0;
#endif
    break;
  case 20:
#if ETM_VERSION == 4
    *val = (uint32_t)etm_gate_on_trigger | ((ETM->STATR & 0x3U) << 8) |
           ((ETM->IDR3 & ETM4_CCCTLR_THRESHOLD_Msk) << 16);
#else
    *val = (uint32_t)etm_gate_on_trigger;
#endif
    break;
  default:
    return 0x06;
  }
  return 0x00;
}

void set_pcsample_params(bool enable, bool cyctap, uint8_t postinit,
                         uint8_t postreset) {
  pcsamp_enable = enable;
  // must clear everything before updating postinit field:
  DWT->CTRL = 0;
  // then update postinit:
  DWT->CTRL = ((postinit & 0x0f) << DWT_CTRL_POSTINIT_Pos);
  // then update the reset, but don't turn on PC sampling yet;
  // that will be handled in trigger_high
  DWT->CTRL |= ((cyctap & 1) << DWT_CTRL_CYCTAP_Pos) |
               ((postreset & 0x0f) << DWT_CTRL_POSTPRESET_Pos) |
               (1 << DWT_CTRL_SYNCTAP_Pos) | (1 << DWT_CTRL_CYCCNTENA_Pos);
}

// DWT FUNCTION helper macros for ARMv8-M (Cortex-M33)
// Note: ARMv8-M (Cortex-M33) changed MATCH[3:0] encodings compared to ARMv7-M!
// MATCH[3:0] in ARMv8-M:
//   0b0010 (2)  = Instruction Address
//   0b0011 (3)  = Instruction Address Limit
//   0b0100 (4)  = Data Address, Reads or Writes (Match/PC packet only)
//   0b0101 (5)  = Data Address, Only Writes (Match/PC packet only)
//   0b0110 (6)  = Data Address, Only Reads (Match/PC packet only)
//   0b0111 (7)  = Data Address Limit (Debug/Trigger only; does not support Data Trace)
//   0b1100 (12) = Data Address with Value, Reads or Writes (Data Value packet)
//   0b1101 (13) = Data Address with Value, Only Writes (Data Value packet)
//   0b1110 (14) = Data Address with Value, Only Reads (Data Value packet)
// ACTION[5:4] in ARMv8-M:
//   0b00   (0) = Trigger only
//   0b01   (1) = Generate Debug event (Watchpoint)
//   0b10   (2) = Generate Data Trace Match / Data Value packet
//   0b11   (3) = Generate Data Trace PC Value packet (and Data Value packet if MATCH[3]==1)
// DATAVSIZE[11:10]:
//   0b00   (0) = 1 byte
//   0b01   (1) = 2 bytes (halfword)
//   0b10   (2) = 4 bytes (word)
#define DWT_MATCH_PC              2UL
#define DWT_MATCH_DATA_RW         4UL
#define DWT_MATCH_DATA_WRITE      5UL
#define DWT_MATCH_DATA_READ       6UL
#define DWT_MATCH_DATA_ADDR_LIMIT 7UL
#define DWT_MATCH_DATA_RW_VAL     12UL
#define DWT_MATCH_DATA_WRITE_VAL  13UL
#define DWT_MATCH_DATA_READ_VAL   14UL
#define DWT_ACTION_TRIGGER_ONLY   0UL
#define DWT_ACTION_TRACE_VALUE    2UL
#define DWT_ACTION_TRACE_VAL_PC   3UL
#define DWT_DATAVSIZE_HALFWORD    1UL
#define DWT_DATAVSIZE_WORD        2UL

#define DWT_FUNC_TRACE_PC_MATCH                                                \
  ((DWT_MATCH_PC << DWT_FUNCTION_MATCH_Pos) |                                  \
   (DWT_DATAVSIZE_HALFWORD << DWT_FUNCTION_DATAVSIZE_Pos) |                    \
   (DWT_ACTION_TRACE_VALUE << DWT_FUNCTION_ACTION_Pos))

#define DWT_FUNC_TRACE_WRITE_WORD                                              \
  ((DWT_MATCH_DATA_WRITE_VAL << DWT_FUNCTION_MATCH_Pos) |                      \
   (DWT_DATAVSIZE_WORD << DWT_FUNCTION_DATAVSIZE_Pos) |                        \
   (DWT_ACTION_TRACE_VALUE << DWT_FUNCTION_ACTION_Pos))

#define DWT_FUNC_TRACE_WRITE_WORD_PC                                           \
  ((DWT_MATCH_DATA_WRITE_VAL << DWT_FUNCTION_MATCH_Pos) |                      \
   (DWT_DATAVSIZE_WORD << DWT_FUNCTION_DATAVSIZE_Pos) |                        \
   (DWT_ACTION_TRACE_VAL_PC << DWT_FUNCTION_ACTION_Pos))

void enable_trace(void) {
  DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk; // Enable DWT, PMU and ITM features

  // Configure TPI
  if (TPI->LSR & 1) {
    TPI->LAR = ETM_LAR_KEY;
  }
  if ((TPI->DEVID & (1U << 17)) == 0) {
    TPI->ACPR = 0; // SWO trace baud rate = cpu clock / (ACPR+1)
  }
  if ((TPI->DEVID & TPIU_DEVID_PTINVALID_Msk) == 0) {
    TPI->SPPR = 0;               // parallel trace mode
    TPI->CSPSR = (1 << (4 - 1)); // 4 trace lanes
  } else {
    if (TPI->DEVID & TPIU_DEVID_NRZVALID_Msk) {
      TPI->SPPR = 1; // SWO NRZ
    } else {
      if (TPI->DEVID & TPIU_DEVID_MANCVALID_Msk) {
        TPI->SPPR = 2; // SWO Manchester
      } else {
        // Invalid. panic?
      }
    }
  }
  TPI->FFCR = TPIU_FFCR_TrigIn_Msk     // Flush
              | TPIU_FFCR_EnFCont_Msk; // Continuous framing*/
  //| TPIU_FFCR_EnFTC_Msk; // packet framing enabled

  // Configure timestamp generator
  TSGEN->CNTCR = 0;  // stop
  TSGEN->CNTCVL = 0; // reset counter
  TSGEN->CNTCVU = 0; // reset counter
  TSGEN->CNTCR = 3;  // start

  // Configure funnel
  if (ATB_FUNNEL->LSR & 1) {
    ATB_FUNNEL->LAR = ETM_LAR_KEY;
  }
  ATB_FUNNEL->FUNNELCONTROL = 3; // enable both ARM cores to go though funnel

  // Configure ITM (STALLENA disabled by default to avoid distorting power traces;
  // can be toggled via SimpleSerial 'T' scmd=0x04)
  if (ITM->LSR & 1) {
    ITM->LAR = ETM_LAR_KEY;
  }
  ITM->TCR = (1 << ITM_TCR_TRACEBUSID_Pos) // Trace bus ID for TPIU
             | (0 << ITM_TCR_STALLENA_Pos) // Do not stall processor by default
             | (1 << ITM_TCR_DWTENA_Pos)   // Enable events from DWT
             | (0 << ITM_TCR_SYNCENA_Pos)  // Disable sync packets!
             | (1 << ITM_TCR_ITMENA_Pos);  // Main enable for ITM
  ITM->TER = 0xFFFFFFFF;                   // Enable all stimulus ports
  ITM->TPR = 0x00000000;                   // allow unpriviledged access

  // Configure DWT:
  if (DWT->LSR & 1) {
    DWT->LAR = ETM_LAR_KEY;
  }
  DWT->CTRL =
      (0xf << DWT_CTRL_POSTINIT_Pos); // countdown counter for PC sampling, must
                                      // be written before enabling PC sampling
  DWT->CTRL |=
      (1 << DWT_CTRL_CYCTAP_Pos) // Prescaler for PC sampling: 0 = x32, 1 = x512
      | (8 << DWT_CTRL_POSTPRESET_Pos) // PC sampling postscaler
      | (0 << DWT_CTRL_PCSAMPLENA_Pos) // disable PC sampling
      | (1 << DWT_CTRL_SYNCTAP_Pos)    // sync packets every 16M cycles
      | (0 << DWT_CTRL_EXCTRCENA_Pos)  // disable exception trace
      | (1 << DWT_CTRL_CYCCNTENA_Pos); // enable cycle counter

  // Configure ETM:
  ETM->OSLAR = 0;
  if (ETM->LSR & 1) {
    ETM->LAR = ETM_LAR_KEY;
  }
  ETM_SetupMode();
#if ETM_VERSION == 3
  ETM->CR = ETM_CR_ETMEN; // Enable ETM output port
  ETM->TRACEIDR = 1;      // Trace bus ID for TPIU
  ETM->FFLR = 0;          // Stall processor when FIFO is full
  ETM->TEEVR =
      0x000150a0; // EmbeddedICE comparator 0 or 1 (DWT->COMP0 or DWT->COMP1)
  ETM->TESSEICR =
      0xf; // set EmbeddedICE watchpoint 0 as a TraceEnable start resource.
  ETM->TECR1 = 0; // tracing is unaffected by the trace start/stop logic
#elif ETM_VERSION == 4
  // Enable Branch Broadcasting (BB, bit 3) and Cycle Counting in Instruction
  // trace (CCI, bit 4) to trace SH_JITTER and RC_JITTER delays:
  ETM->CONFIGR = (1UL << ETM4_CONFIGR_BB_Pos)    // Branch broadcasting
                 | (1UL << ETM4_CONFIGR_CCI_Pos) // Cycle count instruction trace
                 | (0UL << ETM4_CONFIGR_TS_Pos)  // Global timestamping disabled
                 | 1UL;                          // Bit 0 (RES1 on Cortex-M33 ETM)

  // Map Resource Selector 2 -> EXTIN[2] (DWT->COMP2 match)
  // Map Resource Selector 3 -> EXTIN[3] (DWT->COMP3 match)
  ETM->RSCTLR[2] = ETM4_RSC_GROUP_EXTIN | ((1UL << 2) << ETM4_RSC_SELECT_Pos);
  ETM->RSCTLR[3] = ETM4_RSC_GROUP_EXTIN | ((1UL << 3) << ETM4_RSC_SELECT_Pos);

  // Route RSCTLR[2] to ETM Event 0 (SEL0=2) and RSCTLR[3] to ETM Event 1 (SEL1=3)
  ETM->EVENTCTLR0 = (2UL << 0) | (3UL << 8);
  // Disabled until PC watchpoints on COMP2/COMP3 are configured via SimpleSerial 'T'
  ETM->EVENTCTLR1 = 0;

  // Disable processor stalling by default so power traces match real hardware
  // timing (can be enabled via SimpleSerial 'T' scmd=0x04 if lossless FIFO trace is needed)
  ETM->STALLCTLR = (0UL << ETM4_STALLCTLR_ISTALL_Pos);

  // Note: ETM->SYNCPR is read-only (fixed at 0x0A = 1024 bytes) on Cortex-M33

  // ATB Trace Bus ID for ETM = 2 (multiplexed by TPIU with ITM/DWT ID = 1)
  ETM->TRACEIDR = 2;

  // Set Cycle Count threshold to the minimum supported by hardware
  // (TRCIDR3.CCITMIN = 4 cycles on Cortex-M33). Programming 0 (< CCITMIN)
  // suppresses cycle count packets per ARM ETMv4 architecture spec.
  uint32_t ccitmin = ETM->IDR3 & ETM4_CCCTLR_THRESHOLD_Msk;
  ETM->CCCTLR = (ccitmin != 0U) ? ccitmin : 4U;

  ETM->TSCTLR = 0; // Disable timestamp events

  // ViewInst Control:
  //   SSSTATUS (bit 9) = 1 (Start/Stop logic started)
  //   EVENT (bits [7:0]) = 0x01 (Resource 1: architectural Always-True resource)
  //   EXLEVEL_S / EXLEVEL_NS = 0 (Trace all privilege modes)
  ETM->VICTLR = (1UL << ETM4_VICTLR_SSSTATUS_Pos) | 1UL;
  ETM->VIIECTLR = 0;   // No address range filtering (trace all addresses)
  ETM->VISSCTLR = 0;   // No start/stop points for ViewInst
  ETM->VIPCSSCTLR = 0; // No PE comparator control for ViewInst
#endif
  if (!etm_gate_on_trigger) {
    ETM_TraceMode();
  }
}

void ITM_Print(int port, const char *p) {
  if ((ITM->TCR & ITM_TCR_ITMENA_Msk) && (ITM->TER & (1UL << port))) {
    while (*p) {
      while (ITM->PORT[port].u32 == 0) {
      }
      ITM->PORT[port].u8 = *p++;
    }
    printf("ITM alive\n");
  } else {
    printf("Couldn't print!\n");
  }
}

void setup_etm_gpio() {
  // Configure GP1 to GP5 for TRACE
  for (uint pin = 1; pin <= 5; ++pin) {
    gpio_init(pin);
    gpio_set_dir(pin, GPIO_IN);
    gpio_pull_down(pin);
    gpio_set_drive_strength(pin, GPIO_DRIVE_STRENGTH_12MA);
    gpio_set_slew_rate(pin, GPIO_SLEW_RATE_FAST);
    gpio_set_function(pin, GPIO_FUNC_CORESIGHT_TRACE);
  }
  enable_trace();
}
// End of code for ETM tracing support

void led_on() { status_led_set_state(true); }

void led_off() { status_led_set_state(false); }

void led2(bool on_off) { gpio_put(22, on_off == true ? 1 : 0); }

void led3(bool on_off) { gpio_put(23, on_off == true ? 1 : 0); }

void trigger() {
  gpio_put(TRIGGER, 1);
  __asm("NOP; NOP; NOP; NOP; NOP; NOP; NOP; NOP; ");
  gpio_put(TRIGGER, 0);
}

static inline void trigger_high(void) {
  if (pcsamp_enable == 1) {
    DWT->CTRL |= (1 << DWT_CTRL_PCSAMPLENA_Pos); // enable PC sampling
  }
  if (etm_gate_on_trigger) {
    ETM_TraceMode();
  }
  gpio_put(TRIGGER, 1);
}

static inline void trigger_low(void) {
  gpio_put(TRIGGER, 0);
  if (etm_gate_on_trigger) {
    ETM_SetupMode();
  }
  DWT->CTRL &= ~(1 << DWT_CTRL_PCSAMPLENA_Pos); // disable PC sampling
  TPI->FFCR |= 0x0400;                          // Manual flush
  while (TPI->FFSR & 1) {
    // wait for flush
  }
}

// HAL
void putch(int c) { stdio_putchar_raw(c); }
char getch(void) { return (uint8_t)(stdio_getchar()); }
void led_ok(void) {
  led_on();
  led2(false);
}
void led_error(void) {
  led_off();
  led2(true);
}
// End of HAL

#define AES128_KEY_SIZE 16
#define AES256_KEY_SIZE 32
#define AES_BLOCK_SIZE 16
#define NBLOCKS 12

mbedtls_aes_context mbed_ctx;
uint8_t mbed_aes_key[AES128_KEY_SIZE];
uint8_t mbed_block[AES_BLOCK_SIZE];

// Default key will decode to 00000...
// Use keytool.py to generate a new key
unsigned char key4way[4 * AES256_KEY_SIZE] =
    "\x6c\x31\x10\x89\x36\x54\x06\x49\xb8\x3b\xc5\x4b\xe2\x5e\xd3\x8b\x7a\xc9"
    "\x40\x76\xa9\x83\xac\x10\x70\xf3\x77\xe8\xa3\xb9\x9b\x8e\x81\x4f\xe5\xf5"
    "\x80\x8d\x1c\xa7\x0e\xbd\xf7\x0d\x0f\x7f\x0e\x5f\xaa\x0b\xee\xc6\x93\xf7"
    "\x79\xfc\x52\x5f\x6d\xb8\x6b\xa3\xfa\x82\x5b\xf0\xef\x65\xfd\x70\xb2\x31"
    "\x87\x6b\x54\x85\x21\xeb\x09\xd1\x17\x5c\xfd\x1c\x35\x6d\x44\x60\x71\xd1"
    "\xcc\xbf\x53\xe0\x75\xc3\x8b\x1f\xd4\xbf\x4b\x99\x45\xc7\x01\x3a\x2f\x06"
    "\xc1\xbc\xbe\x7e\xc4\xf3\xcc\x93\x42\x6a\xdf\x21\x3a\xb2\xf8\x92\xbc\x2b"
    "\xeb\x20";

// iv, iv_salt, data all initialized to 0
unsigned char iv[AES_BLOCK_SIZE] = {0};
unsigned char iv_salt[AES_BLOCK_SIZE] = {0};
unsigned char aes_data[AES_BLOCK_SIZE * NBLOCKS] = {0};
typedef void (*aes_ctr_fn)(uint8_t *key4way, uint8_t *IV_OTPsalt,
                           uint8_t *IV_public,
                           uint8_t (*buf)[AES_BLOCK_SIZE * NBLOCKS], int nblk);

extern void decrypt(uint8_t *key4way, uint8_t *IV_OTPsalt, uint8_t *IV_public,
                    uint8_t (*buf)[AES_BLOCK_SIZE * NBLOCKS], int nblk);

extern void mbedtls_decrypt(uint8_t *key4way, uint8_t *IV_OTPsalt,
                            uint8_t *IV_public,
                            uint8_t (*buf)[AES_BLOCK_SIZE * NBLOCKS], int nblk);

struct g_clocks_khz {
  uint32_t xosc;
  uint32_t sys_pll;
  uint32_t usb_pll;
  uint32_t rosc;
  uint32_t clk_sys;
  uint32_t clk_peri;
  uint32_t clk_usb;
  uint32_t clk_adc;
  uint32_t clk_rtc;
} g_clocks_khz = {0};

void measure_frequencies(void) {
  g_clocks_khz.sys_pll =
      frequency_count_khz(CLOCKS_FC0_SRC_VALUE_PLL_SYS_CLKSRC_PRIMARY);
  g_clocks_khz.usb_pll =
      frequency_count_khz(CLOCKS_FC0_SRC_VALUE_PLL_USB_CLKSRC_PRIMARY);
  g_clocks_khz.rosc = frequency_count_khz(CLOCKS_FC0_SRC_VALUE_ROSC_CLKSRC);
  g_clocks_khz.clk_sys = frequency_count_khz(CLOCKS_FC0_SRC_VALUE_CLK_SYS);
  g_clocks_khz.clk_peri = frequency_count_khz(CLOCKS_FC0_SRC_VALUE_CLK_PERI);
  g_clocks_khz.clk_usb = frequency_count_khz(CLOCKS_FC0_SRC_VALUE_CLK_USB);
  g_clocks_khz.clk_adc = frequency_count_khz(CLOCKS_FC0_SRC_VALUE_CLK_ADC);
#ifdef CLOCKS_FC0_SRC_VALUE_CLK_RTC
  g_clocks_khz.clk_rtc = frequency_count_khz(CLOCKS_FC0_SRC_VALUE_CLK_RTC);
#else
  g_clocks_khz.clk_rtc = 0;
#endif
}

void platform_init(void) {
  static bool stdio_initialized = false;
  led3(true);

  if (!stdio_initialized) {
    // Initialize UART and USB stdio drivers once at boot
    stdio_uart_init_full(uart0, UART_BAUD_RATE, 14, 15);
    stdio_set_translate_crlf(&stdio_uart, false);

    if (gpio_get(24)) {
      // USB is plugged
      if (stdio_usb_init()) {
        stdio_set_translate_crlf(&stdio_usb, false);
      }
    }
    stdio_initialized = true;
  } else {
    // When called after a dynamic clock change, only update the UART baudrate
    // divisor for the new clk_peri without resetting UART or tearing down USB
    uart_set_baudrate(uart0, UART_BAUD_RATE);
  }

  measure_frequencies();
  led3(false);
}

static void configure_pll_xtal(PLL pll, uint32_t xtal_mhz, uint32_t refdiv,
                               uint32_t vco_freq, uint32_t post_div1,
                               uint32_t post_div2) {
  uint32_t ref_freq = (xtal_mhz * MHZ) / refdiv;
  uint32_t fbdiv = vco_freq / ref_freq;
  uint32_t pdiv = (post_div1 << PLL_PRIM_POSTDIV1_LSB) |
                  (post_div2 << PLL_PRIM_POSTDIV2_LSB);

  // Do not reset or disrupt the PLL if it is already locked at the target config
  if ((pll->cs & PLL_CS_LOCK_BITS) &&
      (refdiv == (pll->cs & PLL_CS_REFDIV_BITS)) &&
      (fbdiv == (pll->fbdiv_int & PLL_FBDIV_INT_BITS)) &&
      (pdiv == (pll->prim & (PLL_PRIM_POSTDIV1_BITS | PLL_PRIM_POSTDIV2_BITS)))) {
    return;
  }

  reset_unreset_block_num_wait_blocking(PLL_RESET_NUM(pll));

  pll->cs = refdiv;
  pll->fbdiv_int = fbdiv;

  hw_clear_bits(&pll->pwr, PLL_PWR_PD_BITS | PLL_PWR_VCOPD_BITS);
  while (!(pll->cs & PLL_CS_LOCK_BITS)) {
    tight_loop_contents();
  }

  pll->prim = pdiv;
  hw_clear_bits(&pll->pwr, PLL_PWR_POSTDIVPD_BITS);
}

uint8_t set_clock(uint32_t target_clk_mhz, uint32_t xtal_mhz) {
  uint32_t sys_refdiv = 1;
  uint32_t sys_vco_freq = 0;
  uint32_t sys_postdiv1 = 1;
  uint32_t sys_postdiv2 = 1;
  uint32_t usb_refdiv = 1;

  if (xtal_mhz == 12) {
    usb_refdiv = 1;
  } else if (xtal_mhz == 24) {
    usb_refdiv = 2;
  } else {
    return 0x41;
  }

  switch (target_clk_mhz) {
  case 24:
    // 1176 MHz / (7 * 7) = 24 MHz
    sys_refdiv = (xtal_mhz == 24) ? 1 : 1;
    sys_vco_freq = 1176 * MHZ;
    sys_postdiv1 = 7;
    sys_postdiv2 = 7;
    break;

  case 48:
    // 1200 MHz / (5 * 5) = 48 MHz
    sys_refdiv = usb_refdiv;
    sys_vco_freq = 1200 * MHZ;
    sys_postdiv1 = 5;
    sys_postdiv2 = 5;
    break;

  case 125:
    // 1500 MHz / (6 * 2) = 125 MHz
    sys_refdiv = usb_refdiv;
    sys_vco_freq = 1500 * MHZ;
    sys_postdiv1 = 6;
    sys_postdiv2 = 2;
    break;

  case 150:
    // 1500 MHz / (5 * 2) = 150 MHz
    sys_refdiv = usb_refdiv;
    sys_vco_freq = 1500 * MHZ;
    sys_postdiv1 = 5;
    sys_postdiv2 = 2;
    break;

  default:
    return 0x43;
  }

  // 1. Wait for any ongoing UART transmission to finish before changing clocks
  uart_tx_wait_blocking(uart0);

  // 2. Configure clk_ref to run directly from XOSC at (xtal_mhz * MHZ)
  clock_configure_undivided(clk_ref, CLOCKS_CLK_REF_CTRL_SRC_VALUE_XOSC_CLKSRC,
                            0, xtal_mhz * MHZ);

  // 3. Switch clk_sys to the glitchless clk_ref (XOSC) BEFORE resetting pll_sys!
  //    (Resetting pll_sys while clk_sys is still sourced from pll_sys stops the CPU clock.)
  clock_configure_undivided(clk_sys, CLOCKS_CLK_SYS_CTRL_SRC_VALUE_CLK_REF,
                            0, xtal_mhz * MHZ);

  // 4. Re-lock pll_sys (and pll_usb if needed, without disrupting it if already locked at 48MHz)
  configure_pll_xtal(pll_sys, xtal_mhz, sys_refdiv, sys_vco_freq, sys_postdiv1,
                     sys_postdiv2);
  configure_pll_xtal(pll_usb, xtal_mhz, usb_refdiv, 1200 * MHZ, 5, 5);

  // 5. Switch clk_sys back to pll_sys via the glitchless mux
  clock_configure_undivided(clk_sys,
                            CLOCKS_CLK_SYS_CTRL_SRC_VALUE_CLKSRC_CLK_SYS_AUX,
                            CLOCKS_CLK_SYS_CTRL_AUXSRC_VALUE_CLKSRC_PLL_SYS,
                            target_clk_mhz * MHZ);

  // 6. Ensure clk_usb and clk_adc remain at 48MHz from pll_usb
  clock_configure_undivided(clk_usb, 0,
                            CLOCKS_CLK_USB_CTRL_AUXSRC_VALUE_CLKSRC_PLL_USB,
                            48 * MHZ);
  clock_configure_undivided(clk_adc, 0,
                            CLOCKS_CLK_ADC_CTRL_AUXSRC_VALUE_CLKSRC_PLL_USB,
                            48 * MHZ);

  // 7. Update clk_peri (UART clock) and clk_hstx to the new clk_sys frequency
  clock_configure_undivided(clk_peri, 0,
                            CLOCKS_CLK_PERI_CTRL_AUXSRC_VALUE_CLK_SYS,
                            target_clk_mhz * MHZ);
#if HAS_HSTX
  clock_configure_undivided(clk_hstx, 0,
                            CLOCKS_CLK_HSTX_CTRL_AUXSRC_VALUE_CLK_SYS,
                            target_clk_mhz * MHZ);
#endif

  // 8. Update all 1MHz hardware tick generators (TIMER0/1, PROC0/1, WATCHDOG)
  //    which divide clk_ref (xtal_mhz) down to 1MHz
  for (int i = 0; i < (int)TICK_COUNT; ++i) {
    tick_start((tick_gen_num_t)i, xtal_mhz);
  }

  g_clocks_khz.xosc = xtal_mhz * 1000;

  // 9. Adjust UART baudrate divisor for the new clk_peri and re-measure clocks
  platform_init();
  return SS_ERR_OK;
}

// ETM/DWT/ITM commands ('T')
// Subcommands (scmd):
//   0x00: Disable all DWT comparators (COMP0..COMP3) and ETM event packets
//   0x01: Set data write value watchpoints (DWT_FUNC_TRACE_WRITE_WORD = 0x82D):
//         - dlen == 4:  COMP0 = addr0
//         - dlen == 8:  COMP0 = start, COMP1 = end (or if end == start + 0x14,
//                       sets COMP0 = rstate_sha, COMP1 = jstate, COMP2 = rstate_lfsr)
//         - dlen == 12: COMP0 = addr0, COMP1 = addr1, COMP2 = addr2
//   0x02: Set PC function entry trace watchpoints on the 3rd & 4th DWT comparators
//         (DWT->COMP2 and DWT->COMP3) and emit both DWT PC match packets (ATB ID 1)
//         and ETMv4 Event 0 / Event 1 packets (ATB ID 2):
//         - dlen == 0: Default to TinyAES SubBytes (COMP2) and MixColumns (COMP3)
//         - dlen == 4: Set COMP2 to pc0 (big-endian) and disable COMP3
//         - dlen == 8: Set COMP2 to pc0 (big-endian) and COMP3 to pc1 (big-endian)
//   0x03: Configure ETMv4 instruction trace mode (Branch Broadcasting & Cycle Counting)
//         (dlen == 2, 3, or 4: data[0]=enable_bb, data[1]=enable_cci,
//          optional data[2]=ccctlr, optional data[3]=enable_stall)
//   0x04: Configure CPU stall on trace FIFO full (to choose between undistorted
//         power traces vs. lossless trace packets):
//         - dlen == 1: data[0] sets both ETM->STALLCTLR.ISTALL and ITM->TCR.STALLENA (0=no stall, 1=stall)
//         - dlen == 2: data[0] sets ETM->STALLCTLR.ISTALL (0/1), data[1] sets ITM->TCR.STALLENA (0/1)
uint8_t handle_etm_trace(uint8_t cmd, uint8_t scmd, uint8_t dlen, uint8_t *data) {
  uint8_t res = 0x10 + SS_ERR_CMD;

  switch (scmd) {
    case 0x00:
      // Disable all DWT comparators and ETM event packets
      DWT->FUNCTION0 = 0;
      DWT->FUNCTION1 = 0;
      DWT->FUNCTION2 = 0;
      DWT->FUNCTION3 = 0;
#if ETM_VERSION == 4
      ETM_SetupMode();
      ETM->EVENTCTLR1 = 0;
      if (!etm_gate_on_trigger) {
        ETM_TraceMode();
      }
#endif
      res = SS_ERR_OK;
      break;

    case 0x01:
      // Configure DWT comparators for Data Trace Data Value on writes (0x0000082D).
      // Note: ARMv8-M DWT does not support Data Trace (ACTION=2,3) on Address Limit
      // comparators (MATCH=7), so each watched address is assigned its own comparator.
      if (dlen == 4 || dlen == 8 || dlen == 12) {
        uint32_t addr0 = ((uint32_t)data[0] << 24)
                         | ((uint32_t)data[1] << 16)
                         | ((uint32_t)data[2] << 8)
                         | (uint32_t)data[3];
        DWT->FUNCTION0 = 0;
        DWT->COMP0 = addr0;
        DWT->FUNCTION0 = DWT_FUNC_TRACE_WRITE_WORD;

        if (dlen == 4) {
          DWT->FUNCTION1 = 0;
        } else if (dlen == 8) {
          uint32_t addr1 = ((uint32_t)data[4] << 24)
                           | ((uint32_t)data[5] << 16)
                           | ((uint32_t)data[6] << 8)
                           | (uint32_t)data[7];
          DWT->FUNCTION1 = 0;
          if (addr1 == addr0 + 0x14U) {
            // [rstate_sha .. rstate_lfsr]: watch rstate_sha (COMP0), jstate (COMP1), rstate_lfsr (COMP2)
            DWT->COMP1 = addr0 + 0x10U;
            DWT->FUNCTION1 = DWT_FUNC_TRACE_WRITE_WORD;
            DWT->FUNCTION2 = 0;
            DWT->COMP2 = addr1;
            DWT->FUNCTION2 = DWT_FUNC_TRACE_WRITE_WORD;
          } else {
            DWT->COMP1 = addr1;
            DWT->FUNCTION1 = DWT_FUNC_TRACE_WRITE_WORD;
          }
        } else if (dlen == 12) {
          uint32_t addr1 = ((uint32_t)data[4] << 24)
                           | ((uint32_t)data[5] << 16)
                           | ((uint32_t)data[6] << 8)
                           | (uint32_t)data[7];
          uint32_t addr2 = ((uint32_t)data[8] << 24)
                           | ((uint32_t)data[9] << 16)
                           | ((uint32_t)data[10] << 8)
                           | (uint32_t)data[11];
          DWT->FUNCTION1 = 0;
          DWT->COMP1 = addr1;
          DWT->FUNCTION1 = DWT_FUNC_TRACE_WRITE_WORD;
          DWT->FUNCTION2 = 0;
          DWT->COMP2 = addr2;
          DWT->FUNCTION2 = DWT_FUNC_TRACE_WRITE_WORD;
        }
        res = SS_ERR_OK;
      } else {
        res = 0x10 + SS_ERR_LEN;
      }
      break;

    case 0x02: {
      // Configure 3rd & 4th comparators (DWT->COMP2 and DWT->COMP3) for PC
      // function entry trace events (TinyAES SubBytes/MixColumns by default,
      // or custom PC addresses passed in data)
      uint32_t pc0 = 0;
      uint32_t pc1 = 0;
      uint32_t event_mask = 0;

      if (dlen == 0) {
        pc0 = (uint32_t)(SubBytes) & 0xFFFFFFFEU;
        pc1 = (uint32_t)(MixColumns) & 0xFFFFFFFEU;
        event_mask = 0x3U; // Enable ETM Event 0 (COMP2) and Event 1 (COMP3)
      } else if (dlen == 4) {
        pc0 = (((uint32_t)data[0] << 24) | ((uint32_t)data[1] << 16) |
               ((uint32_t)data[2] << 8) | (uint32_t)data[3]) &
              0xFFFFFFFEU;
        event_mask = 0x1U; // Enable ETM Event 0 (COMP2)
      } else if (dlen == 8) {
        pc0 = (((uint32_t)data[0] << 24) | ((uint32_t)data[1] << 16) |
               ((uint32_t)data[2] << 8) | (uint32_t)data[3]) &
              0xFFFFFFFEU;
        pc1 = (((uint32_t)data[4] << 24) | ((uint32_t)data[5] << 16) |
               ((uint32_t)data[6] << 8) | (uint32_t)data[7]) &
              0xFFFFFFFEU;
        event_mask = (pc0 ? 0x1U : 0U) | (pc1 ? 0x2U : 0U);
      } else {
        res = 0x10 + SS_ERR_LEN;
        break;
      }

      DWT->FUNCTION2 = 0;
      DWT->FUNCTION3 = 0;

      if (event_mask & 0x1U) {
        DWT->COMP2 = pc0;
        DWT->FUNCTION2 = DWT_FUNC_TRACE_PC_MATCH;
      }
      if (event_mask & 0x2U) {
        DWT->COMP3 = pc1;
        DWT->FUNCTION3 = DWT_FUNC_TRACE_PC_MATCH;
      }

#if ETM_VERSION == 4
      ETM_SetupMode();
      // RSCTLR[2] = EXTIN[2] (DWT->COMP2), RSCTLR[3] = EXTIN[3] (DWT->COMP3)
      ETM->RSCTLR[2] = ETM4_RSC_GROUP_EXTIN | ((1UL << 2) << ETM4_RSC_SELECT_Pos);
      ETM->RSCTLR[3] = ETM4_RSC_GROUP_EXTIN | ((1UL << 3) << ETM4_RSC_SELECT_Pos);
      ETM->EVENTCTLR0 = (2UL << 0) | (3UL << 8);
      // Emit ETMv4 instruction trace Event 0 / Event 1 packets when COMP2 / COMP3 match
      ETM->EVENTCTLR1 = event_mask;
      if (!etm_gate_on_trigger) {
        ETM_TraceMode();
      }
#endif
      res = SS_ERR_OK;
      break;
    }

    case 0x03:
      // Configure ETMv4 Branch Broadcasting (BB) and Cycle Counting (CCI)
      if (dlen >= 2 && dlen <= 4) {
#if ETM_VERSION == 4
        uint32_t bb = (data[0] != 0) ? (1UL << ETM4_CONFIGR_BB_Pos) : 0UL;
        uint32_t cci = (data[1] != 0) ? (1UL << ETM4_CONFIGR_CCI_Pos) : 0UL;
        uint32_t ccitmin = ETM->IDR3 & ETM4_CCCTLR_THRESHOLD_Msk;
        if (ccitmin == 0U) {
          ccitmin = 4U;
        }
        uint32_t threshold = (dlen >= 3 && data[2] >= ccitmin) ? data[2] : ccitmin;

        ETM_SetupMode();
        ETM->CONFIGR = bb | cci | 1UL;
        ETM->CCCTLR = threshold;
        if (dlen == 4) {
          ETM->STALLCTLR = (data[3] != 0) ? (1UL << ETM4_STALLCTLR_ISTALL_Pos) : 0UL;
          if (data[3] != 0) {
            ITM->TCR |= (1UL << ITM_TCR_STALLENA_Pos);
          } else {
            ITM->TCR &= ~(1UL << ITM_TCR_STALLENA_Pos);
          }
        }
        if (!etm_gate_on_trigger) {
          ETM_TraceMode();
        }
#endif
        res = SS_ERR_OK;
      } else {
        res = 0x10 + SS_ERR_LEN;
      }
      break;

    case 0x04:
      // Configure CPU stalling when ETM / ITM trace FIFOs are full:
      //   dlen == 1: data[0] toggles both ETM->STALLCTLR.ISTALL and ITM->TCR.STALLENA
      //   dlen == 2: data[0] toggles ETM->STALLCTLR.ISTALL, data[1] toggles ITM->TCR.STALLENA
      if (dlen == 1 || dlen == 2) {
        bool etm_stall = (data[0] != 0);
        bool itm_stall = (dlen == 2) ? (data[1] != 0) : etm_stall;

#if ETM_VERSION == 4
        ETM_SetupMode();
        ETM->STALLCTLR = etm_stall ? (1UL << ETM4_STALLCTLR_ISTALL_Pos) : 0UL;
        if (!etm_gate_on_trigger) {
          ETM_TraceMode();
        }
#endif
        if (itm_stall) {
          ITM->TCR |= (1UL << ITM_TCR_STALLENA_Pos);
        } else {
          ITM->TCR &= ~(1UL << ITM_TCR_STALLENA_Pos);
        }
        res = SS_ERR_OK;
      } else {
        res = 0x10 + SS_ERR_LEN;
      }
      break;
  }
  return res;
}


// TinyAES commands
uint8_t handle_aes128_ecb(uint8_t cmd, uint8_t scmd, uint8_t dlen,
                          uint8_t *data) {
  uint8_t res = 0x10 + SS_ERR_CMD;
  int ret;

  switch (scmd) {
  case 0x01:
    // TinyAES encrypt
    if (dlen == AES_BLOCK_SIZE) {
      trigger_high();
      AES128_ECB_indp_crypto(data);
      trigger_low();
      res = SS_ERR_OK;
      simpleserial_put('r', dlen, data);
    } else {
      res = 0x10 + SS_ERR_LEN;
    }
    break;

  case 0x02:
    // TinyAES set key
    if (dlen == AES128_KEY_SIZE) {
      AES128_ECB_indp_setkey(data);
      res = SS_ERR_OK;
    } else {
      res = 0x10 + SS_ERR_LEN;
    }
    break;

  case 0x11:
    // MbedTLS AES128 encrypt
    if (dlen == AES_BLOCK_SIZE) {
      trigger_high();
      ret = mbedtls_aes_crypt_ecb(&mbed_ctx, MBEDTLS_AES_ENCRYPT, data,
                                  mbed_block);
      trigger_low();
      if (ret != 0) {
        res = 0x20 + (ret & 0xff);
      } else {
        res = SS_ERR_OK;
      }
      simpleserial_put('r', dlen, mbed_block);
    } else {
      res = 0x10 + SS_ERR_LEN;
    }
    break;

  case 0x12:
    // MbedTLS set key
    if (dlen == AES128_KEY_SIZE) {
      mbedtls_aes_setkey_enc(&mbed_ctx, data, 8 * AES128_KEY_SIZE);
      res = SS_ERR_OK;
    } else {
      res = 0x10 + SS_ERR_LEN;
    }
  }
  return res;
}

// AES-CTR commands
uint8_t handle_aes_ctr(uint8_t cmd, uint8_t scmd, uint8_t dlen, uint8_t *data) {
  uint32_t scratch_addr = 0x20081000U;
  uint32_t *scratch_ptr = (uint32_t *)scratch_addr;

  uint8_t res = 0x10 + SS_ERR_CMD;
  uint8_t nb_blocks = 1;

  switch (scmd) {
  case 0x00:
    // RP2350 masked AES - full key
    if (dlen == sizeof(key4way)) {
      memcpy(key4way, data, dlen);
      res = SS_ERR_OK;
    } else {
      res = 0x10 + SS_ERR_LEN;
    }
    break;

  case 0x01:
  case 0x02:
  case 0x03:
  case 0x04:
    // RP2350 masked AES - 1 key share at a time
    if (dlen == AES256_KEY_SIZE) {
      memcpy(key4way + (AES256_KEY_SIZE * (scmd - 1)), data, dlen);
      res = SS_ERR_OK;
    } else {
      res = 0x10 + SS_ERR_LEN;
    }
    break;

  case 0x10:
    // Set IV
    if (dlen == sizeof(iv)) {
      memcpy(iv, data, dlen);
      res = SS_ERR_OK;
    } else {
      res = 0x10 + SS_ERR_LEN;
    }
    break;

  case 0x11:
    // set IV salt
    if (dlen == sizeof(iv_salt)) {
      memcpy(iv_salt, data, dlen);
      res = SS_ERR_OK;
    } else {
      res = 0x10 + SS_ERR_LEN;
    }
    break;

  case 0x20:
    // set data
    if (dlen <= sizeof(aes_data)) {
      memcpy(aes_data, data, dlen);
      res = SS_ERR_OK;
    } else {
      res = 0x10 + SS_ERR_LEN;
    }
    break;

  case 0x30:
    // RP2350 decrypt
    if (dlen > 1) {
      res = 0x10 + SS_ERR_LEN;
    } else {
      if (dlen == 1) {
        nb_blocks = data[0];
        if (nb_blocks > NBLOCKS) {
          nb_blocks = NBLOCKS;
        }
      }

      memcpy(scratch_ptr, scratch_backup, sizeof(scratch_backup));
      trigger_high();
      decrypt(key4way, iv_salt, iv, &aes_data, nb_blocks);
      trigger_low();
      res = SS_ERR_OK;
    }
    simpleserial_put('r', AES_BLOCK_SIZE * nb_blocks, aes_data);
    break;

  case 0x40:
    // mbedTLS decrypt
    if (dlen > 1) {
      res = 0x10 + SS_ERR_LEN;
    } else {
      if (dlen == 1) {
        nb_blocks = data[0];
        if (nb_blocks > NBLOCKS) {
          nb_blocks = NBLOCKS;
        }
      }

      memcpy(scratch_ptr, scratch_backup, sizeof(scratch_backup));
      trigger_high();
      mbedtls_decrypt(key4way, iv_salt, iv, &aes_data, nb_blocks);
      trigger_low();
      res = SS_ERR_OK;
    }
    simpleserial_put('r', AES_BLOCK_SIZE * nb_blocks, aes_data);
    break;
  }

  return res;
}

// ETM tracing commands
uint8_t enable_tracing(uint8_t cmd, uint8_t scmd, uint8_t dlen, uint8_t *data) {
  enable_trace();
  return SS_ERR_OK;
}

uint8_t trace_set_reg(uint8_t cmd, uint8_t scmd, uint8_t dlen, uint8_t *data) {
  if (dlen != 5) {
    return 0x10 + SS_ERR_LEN;
  }
  uint32_t reg_val =
      (data[1] << 24) | (data[2] << 16) | (data[3] << 8) | (data[4]);
  uint8_t res = setreg(data[0], reg_val);
  if (res != 0) {
    return 0x10 + res;
  }
  return SS_ERR_OK;
}

uint8_t trace_get_reg(uint8_t cmd, uint8_t scmd, uint8_t dlen, uint8_t *data) {
  unsigned char etm_reg[4] = {0};

  if (dlen != 5) {
    simpleserial_put('r', sizeof(etm_reg), etm_reg);
    return 0x10 + SS_ERR_LEN;
  }
  uint32_t reg_val = 0;
  uint8_t res = getreg(data[0], &reg_val);
  etm_reg[0] = (reg_val >> 24) & 0xff;
  etm_reg[1] = (reg_val >> 16) & 0xff;
  etm_reg[2] = (reg_val >> 8) & 0xff;
  etm_reg[3] = (reg_val) & 0xff;
  simpleserial_put('r', 4, etm_reg);
  if (res != 0) {
    return 0x10 + res;
  }
  return SS_ERR_OK;
}

uint8_t trace_set_pcsample_params(uint8_t cmd, uint8_t scmd, uint8_t dlen,
                                  uint8_t *data) {
  uint8_t res = 0;
  if (dlen != 4) {
    simpleserial_put('r', 4, data);
    res = 0x10 + SS_ERR_LEN;
  } else {
    set_pcsample_params((data[0] & 1) == 1, /* bool enable */
                        (data[1] & 1) == 1, /* bool cyctap */
                        data[2] /* uint8_t postinit */,
                        data[3] /* uint8_t postreset */);
  }
  simpleserial_put('r', 4, data);
  return res;
}

uint8_t trace_test_itm(uint8_t cmd, uint8_t scmd, uint8_t dlen, uint8_t *data) {
  if (dlen != 1) {
    return 0x10 + SS_ERR_LEN;
  }
  ITM_Print(data[0], "ITM alive!\n");
  return SS_ERR_OK;
}

uint8_t trace_fw_info(uint8_t cmd, uint8_t scmd, uint8_t dlen, uint8_t *data) {
  measure_frequencies();
  printf("Chipwhisperer RP2350 rp2350_hacking_challenge_2-%s-trace, compiled "
         "%s, %s (SYS_CLK=%dMHz - XOSC=%dMHz - SYS_PLL=%dMHz - ROSC=%dMHz - "
         "PERI=%dMHz)\n",
         FIRMWARE_CONFIG_NAME,
         __DATE__, __TIME__, g_clocks_khz.clk_sys / 1000,
         g_clocks_khz.xosc / 1000, g_clocks_khz.sys_pll / 1000,
         g_clocks_khz.rosc / 1000, g_clocks_khz.clk_peri / 1000);
  return SS_ERR_OK;
}

// Platform commands
uint8_t handle_platform(uint8_t cmd, uint8_t scmd, uint8_t dlen,
                        uint8_t *data) {
  uint16_t temp = 0;
  uint8_t temp_be[2] = {0};

  switch (scmd) {
  case 0x00:
    // reboot in bootloader
    rom_reboot(REBOOT2_FLAG_REBOOT_TYPE_BOOTSEL, 1000, 0, 0);
    break;

  case 0x01:
    // Get board temperature
    temp = adc_read();
    temp_be[0] = (temp >> 8) & 0xff;
    temp_be[1] = temp & 0xff;
    simpleserial_put('r', sizeof(temp_be), temp_be);
    break;

  case 0x02:
    // Get measured frequencies
    measure_frequencies();
    simpleserial_put('r', sizeof(g_clocks_khz), (uint8_t *)&g_clocks_khz);
    break;

  case 0x10:
    // Set clock frequency to default ones except XTAL
    /*
        XOSC_HZ=24000000
        PLL_SYS_REFDIV=2
        PLL_SYS_VCO_FREQ_HZ=1500000000
        PLL_SYS_POSTDIV1=5
        PLL_SYS_POSTDIV2=2
        SYS_CLK_HZ=150000000
    */
    return set_clock(150, 24);

  case 0x11:
    // Set clock frequency to synchronous 24MHz
    return set_clock(24, 24);

  case 0x12:
    // Use PICO2 dev board defaults, i.e. 12MHz XTAL, 150MHz sysclk
    return set_clock(150, 12);

  case 0x13:
    // 12MHz XTAL, 24MHz sysclk
    return set_clock(24, 12);

  case 0x14:
    // Custom preset: data[0] = target_clk_mhz (24, 48, 125, 150), data[1] = xtal_mhz (12, 24)
    if (dlen != 2) {
      return 0x10 + SS_ERR_LEN;
    }
    return set_clock(data[0], data[1]);

  default:
    return 0x10 + SS_ERR_CMD;
  }
  return SS_ERR_OK;
}

int main() {
  g_clocks_khz.xosc = XOSC_HZ / 1000;

  // USB plugged GPIO
  gpio_init(24);
  gpio_set_dir(24, GPIO_IN);
  gpio_set_input_enabled(24, true);

  // LED2
  gpio_init(22);
  gpio_set_dir(22, GPIO_OUT);
  led2(false);
  // LED3
  gpio_init(23);
  gpio_set_dir(23, GPIO_OUT);
  led3(true);

  adc_init();
  adc_set_temp_sensor_enabled(true);
  adc_select_input(4);

  sleep_ms(500);

  simpleserial_init();
  bool simpleserial_error = false;
  // AES commands
  simpleserial_error |= simpleserial_addcmd(0x01, 0, handle_aes128_ecb);
  simpleserial_error |= simpleserial_addcmd(0x02, 0, handle_aes_ctr);
  // Platform commands
  simpleserial_error |= simpleserial_addcmd('P', 0, handle_platform);
  // ETM Tracing - SimpleSerial v1 compatible
  simpleserial_error |= simpleserial_addcmd('e', 0, enable_tracing);
  simpleserial_error |= simpleserial_addcmd('s', 4, trace_set_reg);
  simpleserial_error |= simpleserial_addcmd('g', 4, trace_get_reg);
  simpleserial_error |= simpleserial_addcmd('c', 4, trace_set_pcsample_params);
  simpleserial_error |= simpleserial_addcmd('t', 1, trace_test_itm);
  simpleserial_error |= simpleserial_addcmd('i', 0, trace_fw_info);
  // ETM Tracing commands specific to this firmware
  simpleserial_error |= simpleserial_addcmd('T', 0, handle_etm_trace);

  platform_init();

  gpio_init(TRIGGER);
  gpio_set_dir(TRIGGER, GPIO_OUT);
  gpio_put(TRIGGER, 0);

  setup_etm_gpio();

  status_led_init();
  led_on();
  led3(simpleserial_error);

  // Backup scratch-memory
  uint32_t scratch_addr = 0x20081000U;
  uint32_t *scratch_ptr = (uint32_t *)scratch_addr;
  memcpy(scratch_backup, scratch_ptr, sizeof(scratch_backup));

  while (true) {
    simpleserial_get();
  }
}
