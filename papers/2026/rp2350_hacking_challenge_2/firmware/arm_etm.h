/* 2015 Petteri Aimonen <jpa@git.mail.kapsi.fi>
 * Public domain. */
/* clang-format off */
#ifndef ARM_ETM_H
#define ARM_ETM_H

#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
  #define   __I     volatile             /*!< Defines 'read only' permissions */
#else
  #define   __I     volatile const       /*!< Defines 'read only' permissions */
#endif
#define     __O     volatile             /*!< Defines 'write only' permissions */
#define     __IO    volatile             /*!< Defines 'read / write' permissions */

/* following defines should be used for structure members */
#define     __IM     volatile const      /*! Defines 'read only' structure member permissions */
#define     __OM     volatile            /*! Defines 'write only' structure member permissions */
#define     __IOM    volatile            /*! Defines 'read / write' structure member permissions */


/**
  \brief  Structure type to access the Instrumentation Trace Macrocell Register (ITM).
 */
typedef struct
{
  __OM  union
  {
    __OM  uint8_t    u8;                 /*!< Offset: 0x000 ( /W)  Stimulus Port 8-bit */
    __OM  uint16_t   u16;                /*!< Offset: 0x000 ( /W)  Stimulus Port 16-bit */
    __OM  uint32_t   u32;                /*!< Offset: 0x000 ( /W)  Stimulus Port 32-bit */
  }  PORT [32U];                         /*!< Offset: 0x000 ( /W)  Stimulus Port Registers */
        uint32_t RESERVED0[864U];
  __IOM uint32_t TER;                    /*!< Offset: 0xE00 (R/W)  Trace Enable Registers */
        uint32_t RESERVED1[15U];
  __IOM uint32_t TPR;                    /*!< Offset: 0xE40 (R/W)  Trace Privilege Register */
        uint32_t RESERVED2[15U];
  __IOM uint32_t TCR;                    /*!< Offset: 0xE80 (R/W)  Trace Control Register */
        uint32_t RESERVED3[27U];
  __IM  uint32_t ITREAD;                 /*!< Offset: 0xEF0 (R/ )  Integration Read Register */
        uint32_t RESERVED4[1U];
  __OM  uint32_t ITWRITE;                /*!< Offset: 0xEF8 ( /W)  Integration Write Register */
        uint32_t RESERVED5[1U];
  __IOM uint32_t ITCTRL;                 /*!< Offset: 0xF00 (R/W)  Integration Mode Control Register */
        uint32_t RESERVED6[43U];
  __IOM uint32_t LAR;                    /*!< Offset: 0xFB0 (R/W)  Lock Access Register */
  __IM  uint32_t LSR;                    /*!< Offset: 0xFB4 (R/ )  Lock Status Register */
        uint32_t RESERVED7[1U];
  __IM  uint32_t DEVARCH;                /*!< Offset: 0xFBC (R/ )  Device Architecture Register */
        uint32_t RESERVED8[3U];
  __IM  uint32_t DEVTYPE;                /*!< Offset: 0xFCC (R/ )  Device Type Register */
  __IM  uint32_t PIDR4;                  /*!< Offset: 0xFD0 (R/ )  Peripheral Identifier Register */
  __IM  uint32_t PIDR5;                  /*!< Offset: 0xFD4 (R/ )  Peripheral Identifier Register */
  __IM  uint32_t PIDR6;                  /*!< Offset: 0xFD8 (R/ )  Peripheral Identifier Register */
  __IM  uint32_t PIDR7;                  /*!< Offset: 0xFDC (R/ )  Peripheral Identifier Register */
  __IM  uint32_t PIDR0;                  /*!< Offset: 0xFE0 (R/ )  Peripheral Identifier Register */
  __IM  uint32_t PIDR1;                  /*!< Offset: 0xFE4 (R/ )  Peripheral Identifier Register */
  __IM  uint32_t PIDR2;                  /*!< Offset: 0xFE8 (R/ )  Peripheral Identifier Register */
  __IM  uint32_t PIDR3;                  /*!< Offset: 0xFEC (R/ )  Peripheral Identifier Register */
  __IM  uint32_t CIDR0;                  /*!< Offset: 0xFF0 (R/ )  Component Identifier Register */
  __IM  uint32_t CIDR1;                  /*!< Offset: 0xFF4 (R/ )  Component Identifier Register */
  __IM  uint32_t CIDR2;                  /*!< Offset: 0xFF8 (R/ )  Component Identifier Register */
  __IM  uint32_t CIDR3;                  /*!< Offset: 0xFFC (R/ )  Component Identifier Register */
} ITM_Type;

/** \brief ITM Stimulus Port Register Definitions */
#define ITM_STIM_DISABLED_Pos               1U                                            /*!< ITM STIM: DISABLED Position */
#define ITM_STIM_DISABLED_Msk              (1UL << ITM_STIM_DISABLED_Pos)                 /*!< ITM STIM: DISABLED Mask */

#define ITM_STIM_FIFOREADY_Pos              0U                                            /*!< ITM STIM: FIFOREADY Position */
#define ITM_STIM_FIFOREADY_Msk             (1UL /*<< ITM_STIM_FIFOREADY_Pos*/)            /*!< ITM STIM: FIFOREADY Mask */

/** \brief ITM Trace Privilege Register Definitions */
#define ITM_TPR_PRIVMASK_Pos                0U                                            /*!< ITM TPR: PRIVMASK Position */
#define ITM_TPR_PRIVMASK_Msk               (0xFUL /*<< ITM_TPR_PRIVMASK_Pos*/)            /*!< ITM TPR: PRIVMASK Mask */

/** \brief ITM Trace Control Register Definitions */
#define ITM_TCR_BUSY_Pos                   23U                                            /*!< ITM TCR: BUSY Position */
#define ITM_TCR_BUSY_Msk                   (1UL << ITM_TCR_BUSY_Pos)                      /*!< ITM TCR: BUSY Mask */

#define ITM_TCR_TRACEBUSID_Pos             16U                                            /*!< ITM TCR: ATBID Position */
#define ITM_TCR_TRACEBUSID_Msk             (0x7FUL << ITM_TCR_TRACEBUSID_Pos)             /*!< ITM TCR: ATBID Mask */

#define ITM_TCR_GTSFREQ_Pos                10U                                            /*!< ITM TCR: Global timestamp frequency Position */
#define ITM_TCR_GTSFREQ_Msk                (3UL << ITM_TCR_GTSFREQ_Pos)                   /*!< ITM TCR: Global timestamp frequency Mask */

#define ITM_TCR_TSPRESCALE_Pos              8U                                            /*!< ITM TCR: TSPRESCALE Position */
#define ITM_TCR_TSPRESCALE_Msk             (3UL << ITM_TCR_TSPRESCALE_Pos)                /*!< ITM TCR: TSPRESCALE Mask */

#define ITM_TCR_STALLENA_Pos                5U                                            /*!< ITM TCR: STALLENA Position */
#define ITM_TCR_STALLENA_Msk               (1UL << ITM_TCR_STALLENA_Pos)                  /*!< ITM TCR: STALLENA Mask */

#define ITM_TCR_SWOENA_Pos                  4U                                            /*!< ITM TCR: SWOENA Position */
#define ITM_TCR_SWOENA_Msk                 (1UL << ITM_TCR_SWOENA_Pos)                    /*!< ITM TCR: SWOENA Mask */

#define ITM_TCR_DWTENA_Pos                  3U                                            /*!< ITM TCR: DWTENA Position */
#define ITM_TCR_DWTENA_Msk                 (1UL << ITM_TCR_DWTENA_Pos)                    /*!< ITM TCR: DWTENA Mask */

#define ITM_TCR_SYNCENA_Pos                 2U                                            /*!< ITM TCR: SYNCENA Position */
#define ITM_TCR_SYNCENA_Msk                (1UL << ITM_TCR_SYNCENA_Pos)                   /*!< ITM TCR: SYNCENA Mask */

#define ITM_TCR_TSENA_Pos                   1U                                            /*!< ITM TCR: TSENA Position */
#define ITM_TCR_TSENA_Msk                  (1UL << ITM_TCR_TSENA_Pos)                     /*!< ITM TCR: TSENA Mask */

#define ITM_TCR_ITMENA_Pos                  0U                                            /*!< ITM TCR: ITM Enable bit Position */
#define ITM_TCR_ITMENA_Msk                 (1UL /*<< ITM_TCR_ITMENA_Pos*/)                /*!< ITM TCR: ITM Enable bit Mask */

/** \brief ITM Integration Read Register Definitions */
#define ITM_ITREAD_AFVALID_Pos              1U                                            /*!< ITM ITREAD: AFVALID Position */
#define ITM_ITREAD_AFVALID_Msk             (1UL << ITM_ITREAD_AFVALID_Pos)                /*!< ITM ITREAD: AFVALID Mask */

#define ITM_ITREAD_ATREADY_Pos              0U                                            /*!< ITM ITREAD: ATREADY Position */
#define ITM_ITREAD_ATREADY_Msk             (1UL /*<< ITM_ITREAD_ATREADY_Pos*/)            /*!< ITM ITREAD: ATREADY Mask */

/** \brief ITM Integration Write Register Definitions */
#define ITM_ITWRITE_AFVALID_Pos             1U                                            /*!< ITM ITWRITE: AFVALID Position */
#define ITM_ITWRITE_AFVALID_Msk            (1UL << ITM_ITWRITE_AFVALID_Pos)               /*!< ITM ITWRITE: AFVALID Mask */

#define ITM_ITWRITE_ATREADY_Pos             0U                                            /*!< ITM ITWRITE: ATREADY Position */
#define ITM_ITWRITE_ATREADY_Msk            (1UL /*<< ITM_ITWRITE_ATREADY_Pos*/)           /*!< ITM ITWRITE: ATREADY Mask */

/** \brief ITM Integration Mode Control Register Definitions */
#define ITM_ITCTRL_IME_Pos                  0U                                            /*!< ITM ITCTRL: IME Position */
#define ITM_ITCTRL_IME_Msk                 (1UL /*<< ITM_ITCTRL_IME_Pos*/)                /*!< ITM ITCTRL: IME Mask */

/*@}*/ /* end of group CMSIS_ITM */


/**
  \ingroup  CMSIS_core_register
  \defgroup CMSIS_DWT     Data Watchpoint and Trace (DWT)
  \brief    Type definitions for the Data Watchpoint and Trace (DWT)
  @{
 */

/**
  \brief  Structure type to access the Data Watchpoint and Trace Register (DWT).
 */
typedef struct
{
  __IOM uint32_t CTRL;                   /*!< Offset: 0x000 (R/W)  Control Register */
  __IOM uint32_t CYCCNT;                 /*!< Offset: 0x004 (R/W)  Cycle Count Register */
  __IOM uint32_t CPICNT;                 /*!< Offset: 0x008 (R/W)  CPI Count Register */
  __IOM uint32_t EXCCNT;                 /*!< Offset: 0x00C (R/W)  Exception Overhead Count Register */
  __IOM uint32_t SLEEPCNT;               /*!< Offset: 0x010 (R/W)  Sleep Count Register */
  __IOM uint32_t LSUCNT;                 /*!< Offset: 0x014 (R/W)  LSU Count Register */
  __IOM uint32_t FOLDCNT;                /*!< Offset: 0x018 (R/W)  Folded-instruction Count Register */
  __IM  uint32_t PCSR;                   /*!< Offset: 0x01C (R/ )  Program Counter Sample Register */
  __IOM uint32_t COMP0;                  /*!< Offset: 0x020 (R/W)  Comparator Register 0 */
        uint32_t RESERVED1[1U];
  __IOM uint32_t FUNCTION0;              /*!< Offset: 0x028 (R/W)  Function Register 0 */
  __IOM uint32_t VMASK0;                 /*!< Offset: 0x02C (R/W)  Comparator Value Mask Register 0 */
  __IOM uint32_t COMP1;                  /*!< Offset: 0x030 (R/W)  Comparator Register 1 */
        uint32_t RESERVED2[1U];
  __IOM uint32_t FUNCTION1;              /*!< Offset: 0x038 (R/W)  Function Register 1 */
  __IOM uint32_t VMASK1;                 /*!< Offset: 0x03C (R/W)  Comparator Value Mask Register 1 */
  __IOM uint32_t COMP2;                  /*!< Offset: 0x040 (R/W)  Comparator Register 2 */
        uint32_t RESERVED3[1U];
  __IOM uint32_t FUNCTION2;              /*!< Offset: 0x048 (R/W)  Function Register 2 */
  __IOM uint32_t VMASK2;                 /*!< Offset: 0x04C (R/W)  Comparator Value Mask Register 2 */
  __IOM uint32_t COMP3;                  /*!< Offset: 0x050 (R/W)  Comparator Register 3 */
        uint32_t RESERVED4[1U];
  __IOM uint32_t FUNCTION3;              /*!< Offset: 0x058 (R/W)  Function Register 3 */
  __IOM uint32_t VMASK3;                 /*!< Offset: 0x05C (R/W)  Comparator Value Mask Register 3 */
        uint32_t RESERVED5[980U];
  __IOM uint32_t LAR;                    /*!< Offset: 0xFB0 (R/W)  Software Lock Access Register */
  __IOM uint32_t LSR;                    /*!< Offset: 0xFB4 (R/W)  Software Lock Status Register */
        uint32_t RESERVED6[1U];
  __IM  uint32_t DEVARCH;                /*!< Offset: 0xFBC (R/ )  Device Type Architecture Register */
        uint32_t RESERVED7[3U];
  __IM  uint32_t DEVTYPE;                /*!< Offset: 0xFCC (R/ )  Device Type Identifier Register */
  __IM  uint32_t PIDR4;                  /*!< Offset: 0xFD0 (R/ )  Peripheral Identifier Register */
  __IM  uint32_t PIDR5;                  /*!< Offset: 0xFD4 (R/ )  Peripheral Identifier Register */
  __IM  uint32_t PIDR6;                  /*!< Offset: 0xFD8 (R/ )  Peripheral Identifier Register */
  __IM  uint32_t PIDR7;                  /*!< Offset: 0xFDC (R/ )  Peripheral Identifier Register */
  __IM  uint32_t PIDR0;                  /*!< Offset: 0xFE0 (R/ )  Peripheral Identifier Register */
  __IM  uint32_t PIDR1;                  /*!< Offset: 0xFE4 (R/ )  Peripheral Identifier Register */
  __IM  uint32_t PIDR2;                  /*!< Offset: 0xFE8 (R/ )  Peripheral Identifier Register */
  __IM  uint32_t PIDR3;                  /*!< Offset: 0xFEC (R/ )  Peripheral Identifier Register */
  __IM  uint32_t CIDR0;                  /*!< Offset: 0xFF0 (R/ )  Component Identifier Register */
  __IM  uint32_t CIDR1;                  /*!< Offset: 0xFF4 (R/ )  Component Identifier Register */
  __IM  uint32_t CIDR2;                  /*!< Offset: 0xFF8 (R/ )  Component Identifier Register */
  __IM  uint32_t CIDR3;                  /*!< Offset: 0xFFC (R/ )  Component Identifier Register */
} DWT_Type;

/** \brief DWT Control Register Definitions */
#define DWT_CTRL_NUMCOMP_Pos               28U                                         /*!< DWT CTRL: NUMCOMP Position */
#define DWT_CTRL_NUMCOMP_Msk               (0xFUL << DWT_CTRL_NUMCOMP_Pos)             /*!< DWT CTRL: NUMCOMP Mask */

#define DWT_CTRL_NOTRCPKT_Pos              27U                                         /*!< DWT CTRL: NOTRCPKT Position */
#define DWT_CTRL_NOTRCPKT_Msk              (1UL << DWT_CTRL_NOTRCPKT_Pos)              /*!< DWT CTRL: NOTRCPKT Mask */

#define DWT_CTRL_NOEXTTRIG_Pos             26U                                         /*!< DWT CTRL: NOEXTTRIG Position */
#define DWT_CTRL_NOEXTTRIG_Msk             (1UL << DWT_CTRL_NOEXTTRIG_Pos)             /*!< DWT CTRL: NOEXTTRIG Mask */

#define DWT_CTRL_NOCYCCNT_Pos              25U                                         /*!< DWT CTRL: NOCYCCNT Position */
#define DWT_CTRL_NOCYCCNT_Msk              (1UL << DWT_CTRL_NOCYCCNT_Pos)              /*!< DWT CTRL: NOCYCCNT Mask */

#define DWT_CTRL_NOPRFCNT_Pos              24U                                         /*!< DWT CTRL: NOPRFCNT Position */
#define DWT_CTRL_NOPRFCNT_Msk              (1UL << DWT_CTRL_NOPRFCNT_Pos)              /*!< DWT CTRL: NOPRFCNT Mask */

#define DWT_CTRL_CYCDISS_Pos               23U                                         /*!< DWT CTRL: CYCDISS Position */
#define DWT_CTRL_CYCDISS_Msk               (1UL << DWT_CTRL_CYCDISS_Pos)               /*!< DWT CTRL: CYCDISS Mask */

#define DWT_CTRL_CYCEVTENA_Pos             22U                                         /*!< DWT CTRL: CYCEVTENA Position */
#define DWT_CTRL_CYCEVTENA_Msk             (1UL << DWT_CTRL_CYCEVTENA_Pos)             /*!< DWT CTRL: CYCEVTENA Mask */

#define DWT_CTRL_FOLDEVTENA_Pos            21U                                         /*!< DWT CTRL: FOLDEVTENA Position */
#define DWT_CTRL_FOLDEVTENA_Msk            (1UL << DWT_CTRL_FOLDEVTENA_Pos)            /*!< DWT CTRL: FOLDEVTENA Mask */

#define DWT_CTRL_LSUEVTENA_Pos             20U                                         /*!< DWT CTRL: LSUEVTENA Position */
#define DWT_CTRL_LSUEVTENA_Msk             (1UL << DWT_CTRL_LSUEVTENA_Pos)             /*!< DWT CTRL: LSUEVTENA Mask */

#define DWT_CTRL_SLEEPEVTENA_Pos           19U                                         /*!< DWT CTRL: SLEEPEVTENA Position */
#define DWT_CTRL_SLEEPEVTENA_Msk           (1UL << DWT_CTRL_SLEEPEVTENA_Pos)           /*!< DWT CTRL: SLEEPEVTENA Mask */

#define DWT_CTRL_EXCEVTENA_Pos             18U                                         /*!< DWT CTRL: EXCEVTENA Position */
#define DWT_CTRL_EXCEVTENA_Msk             (1UL << DWT_CTRL_EXCEVTENA_Pos)             /*!< DWT CTRL: EXCEVTENA Mask */

#define DWT_CTRL_CPIEVTENA_Pos             17U                                         /*!< DWT CTRL: CPIEVTENA Position */
#define DWT_CTRL_CPIEVTENA_Msk             (1UL << DWT_CTRL_CPIEVTENA_Pos)             /*!< DWT CTRL: CPIEVTENA Mask */

#define DWT_CTRL_EXCTRCENA_Pos             16U                                         /*!< DWT CTRL: EXCTRCENA Position */
#define DWT_CTRL_EXCTRCENA_Msk             (1UL << DWT_CTRL_EXCTRCENA_Pos)             /*!< DWT CTRL: EXCTRCENA Mask */

#define DWT_CTRL_PCSAMPLENA_Pos            12U                                         /*!< DWT CTRL: PCSAMPLENA Position */
#define DWT_CTRL_PCSAMPLENA_Msk            (1UL << DWT_CTRL_PCSAMPLENA_Pos)            /*!< DWT CTRL: PCSAMPLENA Mask */

#define DWT_CTRL_SYNCTAP_Pos               10U                                         /*!< DWT CTRL: SYNCTAP Position */
#define DWT_CTRL_SYNCTAP_Msk               (0x3UL << DWT_CTRL_SYNCTAP_Pos)             /*!< DWT CTRL: SYNCTAP Mask */

#define DWT_CTRL_CYCTAP_Pos                 9U                                         /*!< DWT CTRL: CYCTAP Position */
#define DWT_CTRL_CYCTAP_Msk                (1UL << DWT_CTRL_CYCTAP_Pos)                /*!< DWT CTRL: CYCTAP Mask */

#define DWT_CTRL_POSTINIT_Pos               5U                                         /*!< DWT CTRL: POSTINIT Position */
#define DWT_CTRL_POSTINIT_Msk              (0xFUL << DWT_CTRL_POSTINIT_Pos)            /*!< DWT CTRL: POSTINIT Mask */

#define DWT_CTRL_POSTPRESET_Pos             1U                                         /*!< DWT CTRL: POSTPRESET Position */
#define DWT_CTRL_POSTPRESET_Msk            (0xFUL << DWT_CTRL_POSTPRESET_Pos)          /*!< DWT CTRL: POSTPRESET Mask */

#define DWT_CTRL_CYCCNTENA_Pos              0U                                         /*!< DWT CTRL: CYCCNTENA Position */
#define DWT_CTRL_CYCCNTENA_Msk             (1UL /*<< DWT_CTRL_CYCCNTENA_Pos*/)         /*!< DWT CTRL: CYCCNTENA Mask */

/** \brief DWT CPI Count Register Definitions */
#define DWT_CPICNT_CPICNT_Pos               0U                                         /*!< DWT CPICNT: CPICNT Position */
#define DWT_CPICNT_CPICNT_Msk              (0xFFUL /*<< DWT_CPICNT_CPICNT_Pos*/)       /*!< DWT CPICNT: CPICNT Mask */

/** \brief DWT Exception Overhead Count Register Definitions */
#define DWT_EXCCNT_EXCCNT_Pos               0U                                         /*!< DWT EXCCNT: EXCCNT Position */
#define DWT_EXCCNT_EXCCNT_Msk              (0xFFUL /*<< DWT_EXCCNT_EXCCNT_Pos*/)       /*!< DWT EXCCNT: EXCCNT Mask */

/** \brief DWT Sleep Count Register Definitions */
#define DWT_SLEEPCNT_SLEEPCNT_Pos           0U                                         /*!< DWT SLEEPCNT: SLEEPCNT Position */
#define DWT_SLEEPCNT_SLEEPCNT_Msk          (0xFFUL /*<< DWT_SLEEPCNT_SLEEPCNT_Pos*/)   /*!< DWT SLEEPCNT: SLEEPCNT Mask */

/** \brief DWT LSU Count Register Definitions */
#define DWT_LSUCNT_LSUCNT_Pos               0U                                         /*!< DWT LSUCNT: LSUCNT Position */
#define DWT_LSUCNT_LSUCNT_Msk              (0xFFUL /*<< DWT_LSUCNT_LSUCNT_Pos*/)       /*!< DWT LSUCNT: LSUCNT Mask */

/** \brief DWT Folded-instruction Count Register Definitions */
#define DWT_FOLDCNT_FOLDCNT_Pos             0U                                         /*!< DWT FOLDCNT: FOLDCNT Position */
#define DWT_FOLDCNT_FOLDCNT_Msk            (0xFFUL /*<< DWT_FOLDCNT_FOLDCNT_Pos*/)     /*!< DWT FOLDCNT: FOLDCNT Mask */

/** \brief DWT Comparator Function Register Definitions */
#define DWT_FUNCTION_ID_Pos                27U                                         /*!< DWT FUNCTION: ID Position */
#define DWT_FUNCTION_ID_Msk                (0x1FUL << DWT_FUNCTION_ID_Pos)             /*!< DWT FUNCTION: ID Mask */

#define DWT_FUNCTION_MATCHED_Pos           24U                                         /*!< DWT FUNCTION: MATCHED Position */
#define DWT_FUNCTION_MATCHED_Msk           (1UL << DWT_FUNCTION_MATCHED_Pos)           /*!< DWT FUNCTION: MATCHED Mask */

#define DWT_FUNCTION_DATAVSIZE_Pos         10U                                         /*!< DWT FUNCTION: DATAVSIZE Position */
#define DWT_FUNCTION_DATAVSIZE_Msk         (0x3UL << DWT_FUNCTION_DATAVSIZE_Pos)       /*!< DWT FUNCTION: DATAVSIZE Mask */

#define DWT_FUNCTION_ACTION_Pos             4U                                         /*!< DWT FUNCTION: ACTION Position */
#define DWT_FUNCTION_ACTION_Msk            (0x3UL << DWT_FUNCTION_ACTION_Pos)          /*!< DWT FUNCTION: ACTION Mask */

#define DWT_FUNCTION_MATCH_Pos              0U                                         /*!< DWT FUNCTION: MATCH Position */
#define DWT_FUNCTION_MATCH_Msk             (0xFUL /*<< DWT_FUNCTION_MATCH_Pos*/)       /*!< DWT FUNCTION: MATCH Mask */

/*@}*/ /* end of group CMSIS_DWT */


/**
  \brief  Structure type to access the Debug Control Block Registers (DCB).
 */
typedef struct
{
  __IOM uint32_t DHCSR;                  /*!< Offset: 0x000 (R/W)  Debug Halting Control and Status Register */
  __OM  uint32_t DCRSR;                  /*!< Offset: 0x004 ( /W)  Debug Core Register Selector Register */
  __IOM uint32_t DCRDR;                  /*!< Offset: 0x008 (R/W)  Debug Core Register Data Register */
  __IOM uint32_t DEMCR;                  /*!< Offset: 0x00C (R/W)  Debug Exception and Monitor Control Register */
  __IOM uint32_t DSCEMCR;                /*!< Offset: 0x010 (R/W)  Debug Set Clear Exception and Monitor Control Register */
  __IOM uint32_t DAUTHCTRL;              /*!< Offset: 0x014 (R/W)  Debug Authentication Control Register */
  __IOM uint32_t DSCSR;                  /*!< Offset: 0x018 (R/W)  Debug Security Control and Status Register */
} DCB_Type;

/** \brief DCB Debug Halting Control and Status Register Definitions */
#define DCB_DHCSR_DBGKEY_Pos               16U                                            /*!< DCB DHCSR: Debug key Position */
#define DCB_DHCSR_DBGKEY_Msk               (0xFFFFUL << DCB_DHCSR_DBGKEY_Pos)             /*!< DCB DHCSR: Debug key Mask */

#define DCB_DHCSR_S_RESTART_ST_Pos         26U                                            /*!< DCB DHCSR: Restart sticky status Position */
#define DCB_DHCSR_S_RESTART_ST_Msk         (1UL << DCB_DHCSR_S_RESTART_ST_Pos)            /*!< DCB DHCSR: Restart sticky status Mask */

#define DCB_DHCSR_S_RESET_ST_Pos           25U                                            /*!< DCB DHCSR: Reset sticky status Position */
#define DCB_DHCSR_S_RESET_ST_Msk           (1UL << DCB_DHCSR_S_RESET_ST_Pos)              /*!< DCB DHCSR: Reset sticky status Mask */

#define DCB_DHCSR_S_RETIRE_ST_Pos          24U                                            /*!< DCB DHCSR: Retire sticky status Position */
#define DCB_DHCSR_S_RETIRE_ST_Msk          (1UL << DCB_DHCSR_S_RETIRE_ST_Pos)             /*!< DCB DHCSR: Retire sticky status Mask */

#define DCB_DHCSR_S_SDE_Pos                20U                                            /*!< DCB DHCSR: Secure debug enabled Position */
#define DCB_DHCSR_S_SDE_Msk                (1UL << DCB_DHCSR_S_SDE_Pos)                   /*!< DCB DHCSR: Secure debug enabled Mask */

#define DCB_DHCSR_S_LOCKUP_Pos             19U                                            /*!< DCB DHCSR: Lockup status Position */
#define DCB_DHCSR_S_LOCKUP_Msk             (1UL << DCB_DHCSR_S_LOCKUP_Pos)                /*!< DCB DHCSR: Lockup status Mask */

#define DCB_DHCSR_S_SLEEP_Pos              18U                                            /*!< DCB DHCSR: Sleeping status Position */
#define DCB_DHCSR_S_SLEEP_Msk              (1UL << DCB_DHCSR_S_SLEEP_Pos)                 /*!< DCB DHCSR: Sleeping status Mask */

#define DCB_DHCSR_S_HALT_Pos               17U                                            /*!< DCB DHCSR: Halted status Position */
#define DCB_DHCSR_S_HALT_Msk               (1UL << DCB_DHCSR_S_HALT_Pos)                  /*!< DCB DHCSR: Halted status Mask */

#define DCB_DHCSR_S_REGRDY_Pos             16U                                            /*!< DCB DHCSR: Register ready status Position */
#define DCB_DHCSR_S_REGRDY_Msk             (1UL << DCB_DHCSR_S_REGRDY_Pos)                /*!< DCB DHCSR: Register ready status Mask */

#define DCB_DHCSR_C_SNAPSTALL_Pos           5U                                            /*!< DCB DHCSR: Snap stall control Position */
#define DCB_DHCSR_C_SNAPSTALL_Msk          (1UL << DCB_DHCSR_C_SNAPSTALL_Pos)             /*!< DCB DHCSR: Snap stall control Mask */

#define DCB_DHCSR_C_MASKINTS_Pos            3U                                            /*!< DCB DHCSR: Mask interrupts control Position */
#define DCB_DHCSR_C_MASKINTS_Msk           (1UL << DCB_DHCSR_C_MASKINTS_Pos)              /*!< DCB DHCSR: Mask interrupts control Mask */

#define DCB_DHCSR_C_STEP_Pos                2U                                            /*!< DCB DHCSR: Step control Position */
#define DCB_DHCSR_C_STEP_Msk               (1UL << DCB_DHCSR_C_STEP_Pos)                  /*!< DCB DHCSR: Step control Mask */

#define DCB_DHCSR_C_HALT_Pos                1U                                            /*!< DCB DHCSR: Halt control Position */
#define DCB_DHCSR_C_HALT_Msk               (1UL << DCB_DHCSR_C_HALT_Pos)                  /*!< DCB DHCSR: Halt control Mask */

#define DCB_DHCSR_C_DEBUGEN_Pos             0U                                            /*!< DCB DHCSR: Debug enable control Position */
#define DCB_DHCSR_C_DEBUGEN_Msk            (1UL /*<< DCB_DHCSR_C_DEBUGEN_Pos*/)           /*!< DCB DHCSR: Debug enable control Mask */

/** \brief DCB Debug Core Register Selector Register Definitions */
#define DCB_DCRSR_REGWnR_Pos               16U                                            /*!< DCB DCRSR: Register write/not-read Position */
#define DCB_DCRSR_REGWnR_Msk               (1UL << DCB_DCRSR_REGWnR_Pos)                  /*!< DCB DCRSR: Register write/not-read Mask */

#define DCB_DCRSR_REGSEL_Pos                0U                                            /*!< DCB DCRSR: Register selector Position */
#define DCB_DCRSR_REGSEL_Msk               (0x7FUL /*<< DCB_DCRSR_REGSEL_Pos*/)           /*!< DCB DCRSR: Register selector Mask */

/** \brief DCB Debug Core Register Data Register Definitions */
#define DCB_DCRDR_DBGTMP_Pos                0U                                            /*!< DCB DCRDR: Data temporary buffer Position */
#define DCB_DCRDR_DBGTMP_Msk               (0xFFFFFFFFUL /*<< DCB_DCRDR_DBGTMP_Pos*/)     /*!< DCB DCRDR: Data temporary buffer Mask */

/** \brief DCB Debug Exception and Monitor Control Register Definitions */
#define DCB_DEMCR_TRCENA_Pos               24U                                            /*!< DCB DEMCR: Trace enable Position */
#define DCB_DEMCR_TRCENA_Msk               (1UL << DCB_DEMCR_TRCENA_Pos)                  /*!< DCB DEMCR: Trace enable Mask */

#define DCB_DEMCR_MONPRKEY_Pos             23U                                            /*!< DCB DEMCR: Monitor pend req key Position */
#define DCB_DEMCR_MONPRKEY_Msk             (1UL << DCB_DEMCR_MONPRKEY_Pos)                /*!< DCB DEMCR: Monitor pend req key Mask */

#define DCB_DEMCR_UMON_EN_Pos              21U                                            /*!< DCB DEMCR: Unprivileged monitor enable Position */
#define DCB_DEMCR_UMON_EN_Msk              (1UL << DCB_DEMCR_UMON_EN_Pos)                 /*!< DCB DEMCR: Unprivileged monitor enable Mask */

#define DCB_DEMCR_SDME_Pos                 20U                                            /*!< DCB DEMCR: Secure DebugMonitor enable Position */
#define DCB_DEMCR_SDME_Msk                 (1UL << DCB_DEMCR_SDME_Pos)                    /*!< DCB DEMCR: Secure DebugMonitor enable Mask */

#define DCB_DEMCR_MON_REQ_Pos              19U                                            /*!< DCB DEMCR: Monitor request Position */
#define DCB_DEMCR_MON_REQ_Msk              (1UL << DCB_DEMCR_MON_REQ_Pos)                 /*!< DCB DEMCR: Monitor request Mask */

#define DCB_DEMCR_MON_STEP_Pos             18U                                            /*!< DCB DEMCR: Monitor step Position */
#define DCB_DEMCR_MON_STEP_Msk             (1UL << DCB_DEMCR_MON_STEP_Pos)                /*!< DCB DEMCR: Monitor step Mask */

#define DCB_DEMCR_MON_PEND_Pos             17U                                            /*!< DCB DEMCR: Monitor pend Position */
#define DCB_DEMCR_MON_PEND_Msk             (1UL << DCB_DEMCR_MON_PEND_Pos)                /*!< DCB DEMCR: Monitor pend Mask */

#define DCB_DEMCR_MON_EN_Pos               16U                                            /*!< DCB DEMCR: Monitor enable Position */
#define DCB_DEMCR_MON_EN_Msk               (1UL << DCB_DEMCR_MON_EN_Pos)                  /*!< DCB DEMCR: Monitor enable Mask */

#define DCB_DEMCR_VC_SFERR_Pos             11U                                            /*!< DCB DEMCR: Vector Catch SecureFault Position */
#define DCB_DEMCR_VC_SFERR_Msk             (1UL << DCB_DEMCR_VC_SFERR_Pos)                /*!< DCB DEMCR: Vector Catch SecureFault Mask */

#define DCB_DEMCR_VC_HARDERR_Pos           10U                                            /*!< DCB DEMCR: Vector Catch HardFault errors Position */
#define DCB_DEMCR_VC_HARDERR_Msk           (1UL << DCB_DEMCR_VC_HARDERR_Pos)              /*!< DCB DEMCR: Vector Catch HardFault errors Mask */

#define DCB_DEMCR_VC_INTERR_Pos             9U                                            /*!< DCB DEMCR: Vector Catch interrupt errors Position */
#define DCB_DEMCR_VC_INTERR_Msk            (1UL << DCB_DEMCR_VC_INTERR_Pos)               /*!< DCB DEMCR: Vector Catch interrupt errors Mask */

#define DCB_DEMCR_VC_BUSERR_Pos             8U                                            /*!< DCB DEMCR: Vector Catch BusFault errors Position */
#define DCB_DEMCR_VC_BUSERR_Msk            (1UL << DCB_DEMCR_VC_BUSERR_Pos)               /*!< DCB DEMCR: Vector Catch BusFault errors Mask */

#define DCB_DEMCR_VC_STATERR_Pos            7U                                            /*!< DCB DEMCR: Vector Catch state errors Position */
#define DCB_DEMCR_VC_STATERR_Msk           (1UL << DCB_DEMCR_VC_STATERR_Pos)              /*!< DCB DEMCR: Vector Catch state errors Mask */

#define DCB_DEMCR_VC_CHKERR_Pos             6U                                            /*!< DCB DEMCR: Vector Catch check errors Position */
#define DCB_DEMCR_VC_CHKERR_Msk            (1UL << DCB_DEMCR_VC_CHKERR_Pos)               /*!< DCB DEMCR: Vector Catch check errors Mask */

#define DCB_DEMCR_VC_NOCPERR_Pos            5U                                            /*!< DCB DEMCR: Vector Catch NOCP errors Position */
#define DCB_DEMCR_VC_NOCPERR_Msk           (1UL << DCB_DEMCR_VC_NOCPERR_Pos)              /*!< DCB DEMCR: Vector Catch NOCP errors Mask */

#define DCB_DEMCR_VC_MMERR_Pos              4U                                            /*!< DCB DEMCR: Vector Catch MemManage errors Position */
#define DCB_DEMCR_VC_MMERR_Msk             (1UL << DCB_DEMCR_VC_MMERR_Pos)                /*!< DCB DEMCR: Vector Catch MemManage errors Mask */

#define DCB_DEMCR_VC_CORERESET_Pos          0U                                            /*!< DCB DEMCR: Vector Catch Core reset Position */
#define DCB_DEMCR_VC_CORERESET_Msk         (1UL /*<< DCB_DEMCR_VC_CORERESET_Pos*/)        /*!< DCB DEMCR: Vector Catch Core reset Mask */

/** \brief DCB Debug Authentication Control Register Definitions */
#define DCB_DAUTHCTRL_INTSPNIDEN_Pos        3U                                            /*!< DCB DAUTHCTRL: Internal Secure non-invasive debug enable Position */
#define DCB_DAUTHCTRL_INTSPNIDEN_Msk       (1UL << DCB_DAUTHCTRL_INTSPNIDEN_Pos)          /*!< DCB DAUTHCTRL: Internal Secure non-invasive debug enable Mask */

#define DCB_DAUTHCTRL_SPNIDENSEL_Pos        2U                                            /*!< DCB DAUTHCTRL: Secure non-invasive debug enable select Position */
#define DCB_DAUTHCTRL_SPNIDENSEL_Msk       (1UL << DCB_DAUTHCTRL_SPNIDENSEL_Pos)          /*!< DCB DAUTHCTRL: Secure non-invasive debug enable select Mask */

#define DCB_DAUTHCTRL_INTSPIDEN_Pos         1U                                            /*!< DCB DAUTHCTRL: Internal Secure invasive debug enable Position */
#define DCB_DAUTHCTRL_INTSPIDEN_Msk        (1UL << DCB_DAUTHCTRL_INTSPIDEN_Pos)           /*!< DCB DAUTHCTRL: Internal Secure invasive debug enable Mask */

#define DCB_DAUTHCTRL_SPIDENSEL_Pos         0U                                            /*!< DCB DAUTHCTRL: Secure invasive debug enable select Position */
#define DCB_DAUTHCTRL_SPIDENSEL_Msk        (1UL /*<< DCB_DAUTHCTRL_SPIDENSEL_Pos*/)       /*!< DCB DAUTHCTRL: Secure invasive debug enable select Mask */

/** \brief DCB Debug Security Control and Status Register Definitions */
#define DCB_DSCSR_CDSKEY_Pos               17U                                            /*!< DCB DSCSR: CDS write-enable key Position */
#define DCB_DSCSR_CDSKEY_Msk               (1UL << DCB_DSCSR_CDSKEY_Pos)                  /*!< DCB DSCSR: CDS write-enable key Mask */

#define DCB_DSCSR_CDS_Pos                  16U                                            /*!< DCB DSCSR: Current domain Secure Position */
#define DCB_DSCSR_CDS_Msk                  (1UL << DCB_DSCSR_CDS_Pos)                     /*!< DCB DSCSR: Current domain Secure Mask */

#define DCB_DSCSR_SBRSEL_Pos                1U                                            /*!< DCB DSCSR: Secure banked register select Position */
#define DCB_DSCSR_SBRSEL_Msk               (1UL << DCB_DSCSR_SBRSEL_Pos)                  /*!< DCB DSCSR: Secure banked register select Mask */

#define DCB_DSCSR_SBRSELEN_Pos              0U                                            /*!< DCB DSCSR: Secure banked register select enable Position */
#define DCB_DSCSR_SBRSELEN_Msk             (1UL /*<< DCB_DSCSR_SBRSELEN_Pos*/)            /*!< DCB DSCSR: Secure banked register select enable Mask */

/*@} end of group CMSIS_DCB */


/**
  \ingroup  CMSIS_core_register
  \defgroup CMSIS_TPIU     Trace Port Interface Unit (TPIU)
  \brief    Type definitions for the Trace Port Interface Unit (TPIU)
  @{
 */

/**
  \brief  Structure type to access the Trace Port Interface Unit Register (TPIU).
 */
typedef struct
{
  __IM  uint32_t SSPSR;                  /*!< Offset: 0x000 (R/ )  Supported Parallel Port Size Register */
  __IOM uint32_t CSPSR;                  /*!< Offset: 0x004 (R/W)  Current Parallel Port Size Register */
        uint32_t RESERVED0[2U];
  __IOM uint32_t ACPR;                   /*!< Offset: 0x010 (R/W)  Asynchronous Clock Prescaler Register */
        uint32_t RESERVED1[55U];
  __IOM uint32_t SPPR;                   /*!< Offset: 0x0F0 (R/W)  Selected Pin Protocol Register */
        uint32_t RESERVED2[131U];
  __IM  uint32_t FFSR;                   /*!< Offset: 0x300 (R/ )  Formatter and Flush Status Register */
  __IOM uint32_t FFCR;                   /*!< Offset: 0x304 (R/W)  Formatter and Flush Control Register */
  __IOM uint32_t PSCR;                   /*!< Offset: 0x308 (R/W)  Periodic Synchronization Control Register */
        uint32_t RESERVED3[759U];
  __IM  uint32_t TRIGGER;                /*!< Offset: 0xEE8 (R/ )  TRIGGER Register */
  __IM  uint32_t ITFTTD0;                /*!< Offset: 0xEEC (R/ )  Integration Test FIFO Test Data 0 Register */
  __IOM uint32_t ITATBCTR2;              /*!< Offset: 0xEF0 (R/W)  Integration Test ATB Control Register 2 */
        uint32_t RESERVED4[1U];
  __IM  uint32_t ITATBCTR0;              /*!< Offset: 0xEF8 (R/ )  Integration Test ATB Control Register 0 */
  __IM  uint32_t ITFTTD1;                /*!< Offset: 0xEFC (R/ )  Integration Test FIFO Test Data 1 Register */
  __IOM uint32_t ITCTRL;                 /*!< Offset: 0xF00 (R/W)  Integration Mode Control */
        uint32_t RESERVED5[39U];
  __IOM uint32_t CLAIMSET;               /*!< Offset: 0xFA0 (R/W)  Claim tag set */
  __IOM uint32_t CLAIMCLR;               /*!< Offset: 0xFA4 (R/W)  Claim tag clear */
        uint32_t RESERVED7[2U];
  __IOM uint32_t LAR;                    /*!< Offset: 0xFB0 (R/W)  Lock Access Register */
  __IM  uint32_t LSR;                    /*!< Offset: 0xFB4 (R/ )  Lock Status Register */
        uint32_t RESERVED8[4U];
  __IM  uint32_t DEVID;                  /*!< Offset: 0xFC8 (R/ )  Device Configuration Register */
  __IM  uint32_t DEVTYPE;                /*!< Offset: 0xFCC (R/ )  Device Type Identifier Register */
  __IM  uint32_t PIDR4;                  /*!< Offset: 0xFD0 (R/ )  Peripheral Identifier Register */
  __IM  uint32_t PIDR5;                  /*!< Offset: 0xFD4 (R/ )  Peripheral Identifier Register */
  __IM  uint32_t PIDR6;                  /*!< Offset: 0xFD8 (R/ )  Peripheral Identifier Register */
  __IM  uint32_t PIDR7;                  /*!< Offset: 0xFDC (R/ )  Peripheral Identifier Register */
  __IM  uint32_t PIDR0;                  /*!< Offset: 0xFE0 (R/ )  Peripheral Identifier Register */
  __IM  uint32_t PIDR1;                  /*!< Offset: 0xFE4 (R/ )  Peripheral Identifier Register */
  __IM  uint32_t PIDR2;                  /*!< Offset: 0xFE8 (R/ )  Peripheral Identifier Register */
  __IM  uint32_t PIDR3;                  /*!< Offset: 0xFEC (R/ )  Peripheral Identifier Register */
  __IM  uint32_t CIDR0;                  /*!< Offset: 0xFF0 (R/ )  Component Identifier Register */
  __IM  uint32_t CIDR1;                  /*!< Offset: 0xFF4 (R/ )  Component Identifier Register */
  __IM  uint32_t CIDR2;                  /*!< Offset: 0xFF8 (R/ )  Component Identifier Register */
  __IM  uint32_t CIDR3;                  /*!< Offset: 0xFFC (R/ )  Component Identifier Register */
} TPIU_Type;

/** \brief TPIU Asynchronous Clock Prescaler Register Definitions */
#define TPIU_ACPR_PRESCALER_Pos             0U                                         /*!< TPIU ACPR: PRESCALER Position */
#define TPIU_ACPR_PRESCALER_Msk            (0x1FFFUL /*<< TPIU_ACPR_PRESCALER_Pos*/)   /*!< TPIU ACPR: PRESCALER Mask */

/** \brief TPIU Selected Pin Protocol Register Definitions */
#define TPIU_SPPR_TXMODE_Pos                0U                                         /*!< TPIU SPPR: TXMODE Position */
#define TPIU_SPPR_TXMODE_Msk               (0x3UL /*<< TPIU_SPPR_TXMODE_Pos*/)         /*!< TPIU SPPR: TXMODE Mask */

/** \brief TPIU Formatter and Flush Status Register Definitions */
#define TPIU_FFSR_FtNonStop_Pos             3U                                         /*!< TPIU FFSR: FtNonStop Position */
#define TPIU_FFSR_FtNonStop_Msk            (1UL << TPIU_FFSR_FtNonStop_Pos)            /*!< TPIU FFSR: FtNonStop Mask */

#define TPIU_FFSR_TCPresent_Pos             2U                                         /*!< TPIU FFSR: TCPresent Position */
#define TPIU_FFSR_TCPresent_Msk            (1UL << TPIU_FFSR_TCPresent_Pos)            /*!< TPIU FFSR: TCPresent Mask */

#define TPIU_FFSR_FtStopped_Pos             1U                                         /*!< TPIU FFSR: FtStopped Position */
#define TPIU_FFSR_FtStopped_Msk            (1UL << TPIU_FFSR_FtStopped_Pos)            /*!< TPIU FFSR: FtStopped Mask */

#define TPIU_FFSR_FlInProg_Pos              0U                                         /*!< TPIU FFSR: FlInProg Position */
#define TPIU_FFSR_FlInProg_Msk             (1UL /*<< TPIU_FFSR_FlInProg_Pos*/)         /*!< TPIU FFSR: FlInProg Mask */

/** \brief TPIU Formatter and Flush Control Register Definitions */
#define TPIU_FFCR_TrigIn_Pos                8U                                         /*!< TPIU FFCR: TrigIn Position */
#define TPIU_FFCR_TrigIn_Msk               (1UL << TPIU_FFCR_TrigIn_Pos)               /*!< TPIU FFCR: TrigIn Mask */

#define TPIU_FFCR_FOnMan_Pos                6U                                         /*!< TPIU FFCR: FOnMan Position */
#define TPIU_FFCR_FOnMan_Msk               (1UL << TPIU_FFCR_FOnMan_Pos)               /*!< TPIU FFCR: FOnMan Mask */

#define TPIU_FFCR_EnFCont_Pos               1U                                         /*!< TPIU FFCR: EnFCont Position */
#define TPIU_FFCR_EnFCont_Msk              (1UL << TPIU_FFCR_EnFCont_Pos)              /*!< TPIU FFCR: EnFCont Mask */

#define TPIU_FFCR_EnFTC_Pos                 0U
#define TPIU_FFCR_EnFTC_Msk                (1UL /*<< TPIU_FFCR_EnFTC_Pos*/)

/** \brief TPIU Periodic Synchronization Control Register Definitions */
#define TPIU_PSCR_PSCount_Pos               0U                                         /*!< TPIU PSCR: PSCount Position */
#define TPIU_PSCR_PSCount_Msk              (0x1FUL /*<< TPIU_PSCR_PSCount_Pos*/)       /*!< TPIU PSCR: TPSCount Mask */

/** \brief TPIU TRIGGER Register Definitions */
#define TPIU_TRIGGER_TRIGGER_Pos            0U                                         /*!< TPIU TRIGGER: TRIGGER Position */
#define TPIU_TRIGGER_TRIGGER_Msk           (1UL /*<< TPIU_TRIGGER_TRIGGER_Pos*/)       /*!< TPIU TRIGGER: TRIGGER Mask */

/** \brief TPIU Integration Test FIFO Test Data 0 Register Definitions */
#define TPIU_ITFTTD0_ATB_IF2_ATVALID_Pos   29U                                         /*!< TPIU ITFTTD0: ATB Interface 2 ATVALIDPosition */
#define TPIU_ITFTTD0_ATB_IF2_ATVALID_Msk   (0x3UL << TPIU_ITFTTD0_ATB_IF2_ATVALID_Pos) /*!< TPIU ITFTTD0: ATB Interface 2 ATVALID Mask */

#define TPIU_ITFTTD0_ATB_IF2_bytecount_Pos 27U                                         /*!< TPIU ITFTTD0: ATB Interface 2 byte count Position */
#define TPIU_ITFTTD0_ATB_IF2_bytecount_Msk (0x3UL << TPIU_ITFTTD0_ATB_IF2_bytecount_Pos) /*!< TPIU ITFTTD0: ATB Interface 2 byte count Mask */

#define TPIU_ITFTTD0_ATB_IF1_ATVALID_Pos   26U                                         /*!< TPIU ITFTTD0: ATB Interface 1 ATVALID Position */
#define TPIU_ITFTTD0_ATB_IF1_ATVALID_Msk   (0x3UL << TPIU_ITFTTD0_ATB_IF1_ATVALID_Pos) /*!< TPIU ITFTTD0: ATB Interface 1 ATVALID Mask */

#define TPIU_ITFTTD0_ATB_IF1_bytecount_Pos 24U                                         /*!< TPIU ITFTTD0: ATB Interface 1 byte count Position */
#define TPIU_ITFTTD0_ATB_IF1_bytecount_Msk (0x3UL << TPIU_ITFTTD0_ATB_IF1_bytecount_Pos) /*!< TPIU ITFTTD0: ATB Interface 1 byte countt Mask */

#define TPIU_ITFTTD0_ATB_IF1_data2_Pos     16U                                         /*!< TPIU ITFTTD0: ATB Interface 1 data2 Position */
#define TPIU_ITFTTD0_ATB_IF1_data2_Msk     (0xFFUL << TPIU_ITFTTD0_ATB_IF1_data1_Pos)  /*!< TPIU ITFTTD0: ATB Interface 1 data2 Mask */

#define TPIU_ITFTTD0_ATB_IF1_data1_Pos      8U                                         /*!< TPIU ITFTTD0: ATB Interface 1 data1 Position */
#define TPIU_ITFTTD0_ATB_IF1_data1_Msk     (0xFFUL << TPIU_ITFTTD0_ATB_IF1_data1_Pos)  /*!< TPIU ITFTTD0: ATB Interface 1 data1 Mask */

#define TPIU_ITFTTD0_ATB_IF1_data0_Pos      0U                                         /*!< TPIU ITFTTD0: ATB Interface 1 data0 Position */
#define TPIU_ITFTTD0_ATB_IF1_data0_Msk     (0xFFUL /*<< TPIU_ITFTTD0_ATB_IF1_data0_Pos*/) /*!< TPIU ITFTTD0: ATB Interface 1 data0 Mask */

/** \brief TPIU Integration Test ATB Control Register 2 Register Definitions */
#define TPIU_ITATBCTR2_AFVALID2S_Pos        1U                                         /*!< TPIU ITATBCTR2: AFVALID2S Position */
#define TPIU_ITATBCTR2_AFVALID2S_Msk       (1UL << TPIU_ITATBCTR2_AFVALID2S_Pos)       /*!< TPIU ITATBCTR2: AFVALID2SS Mask */

#define TPIU_ITATBCTR2_AFVALID1S_Pos        1U                                         /*!< TPIU ITATBCTR2: AFVALID1S Position */
#define TPIU_ITATBCTR2_AFVALID1S_Msk       (1UL << TPIU_ITATBCTR2_AFVALID1S_Pos)       /*!< TPIU ITATBCTR2: AFVALID1SS Mask */

#define TPIU_ITATBCTR2_ATREADY2S_Pos        0U                                         /*!< TPIU ITATBCTR2: ATREADY2S Position */
#define TPIU_ITATBCTR2_ATREADY2S_Msk       (1UL /*<< TPIU_ITATBCTR2_ATREADY2S_Pos*/)   /*!< TPIU ITATBCTR2: ATREADY2S Mask */

#define TPIU_ITATBCTR2_ATREADY1S_Pos        0U                                         /*!< TPIU ITATBCTR2: ATREADY1S Position */
#define TPIU_ITATBCTR2_ATREADY1S_Msk       (1UL /*<< TPIU_ITATBCTR2_ATREADY1S_Pos*/)   /*!< TPIU ITATBCTR2: ATREADY1S Mask */

/** \brief TPIU Integration Test FIFO Test Data 1 Register Definitions */
#define TPIU_ITFTTD1_ATB_IF2_ATVALID_Pos   29U                                         /*!< TPIU ITFTTD1: ATB Interface 2 ATVALID Position */
#define TPIU_ITFTTD1_ATB_IF2_ATVALID_Msk   (0x3UL << TPIU_ITFTTD1_ATB_IF2_ATVALID_Pos) /*!< TPIU ITFTTD1: ATB Interface 2 ATVALID Mask */

#define TPIU_ITFTTD1_ATB_IF2_bytecount_Pos 27U                                         /*!< TPIU ITFTTD1: ATB Interface 2 byte count Position */
#define TPIU_ITFTTD1_ATB_IF2_bytecount_Msk (0x3UL << TPIU_ITFTTD1_ATB_IF2_bytecount_Pos) /*!< TPIU ITFTTD1: ATB Interface 2 byte count Mask */

#define TPIU_ITFTTD1_ATB_IF1_ATVALID_Pos   26U                                         /*!< TPIU ITFTTD1: ATB Interface 1 ATVALID Position */
#define TPIU_ITFTTD1_ATB_IF1_ATVALID_Msk   (0x3UL << TPIU_ITFTTD1_ATB_IF1_ATVALID_Pos) /*!< TPIU ITFTTD1: ATB Interface 1 ATVALID Mask */

#define TPIU_ITFTTD1_ATB_IF1_bytecount_Pos 24U                                         /*!< TPIU ITFTTD1: ATB Interface 1 byte count Position */
#define TPIU_ITFTTD1_ATB_IF1_bytecount_Msk (0x3UL << TPIU_ITFTTD1_ATB_IF1_bytecount_Pos) /*!< TPIU ITFTTD1: ATB Interface 1 byte countt Mask */

#define TPIU_ITFTTD1_ATB_IF2_data2_Pos     16U                                         /*!< TPIU ITFTTD1: ATB Interface 2 data2 Position */
#define TPIU_ITFTTD1_ATB_IF2_data2_Msk     (0xFFUL << TPIU_ITFTTD1_ATB_IF2_data1_Pos)  /*!< TPIU ITFTTD1: ATB Interface 2 data2 Mask */

#define TPIU_ITFTTD1_ATB_IF2_data1_Pos      8U                                         /*!< TPIU ITFTTD1: ATB Interface 2 data1 Position */
#define TPIU_ITFTTD1_ATB_IF2_data1_Msk     (0xFFUL << TPIU_ITFTTD1_ATB_IF2_data1_Pos)  /*!< TPIU ITFTTD1: ATB Interface 2 data1 Mask */

#define TPIU_ITFTTD1_ATB_IF2_data0_Pos      0U                                         /*!< TPIU ITFTTD1: ATB Interface 2 data0 Position */
#define TPIU_ITFTTD1_ATB_IF2_data0_Msk     (0xFFUL /*<< TPIU_ITFTTD1_ATB_IF2_data0_Pos*/) /*!< TPIU ITFTTD1: ATB Interface 2 data0 Mask */

/** \brief TPIU Integration Test ATB Control Register 0 Definitions */
#define TPIU_ITATBCTR0_AFVALID2S_Pos        1U                                         /*!< TPIU ITATBCTR0: AFVALID2S Position */
#define TPIU_ITATBCTR0_AFVALID2S_Msk       (1UL << TPIU_ITATBCTR0_AFVALID2S_Pos)       /*!< TPIU ITATBCTR0: AFVALID2SS Mask */

#define TPIU_ITATBCTR0_AFVALID1S_Pos        1U                                         /*!< TPIU ITATBCTR0: AFVALID1S Position */
#define TPIU_ITATBCTR0_AFVALID1S_Msk       (1UL << TPIU_ITATBCTR0_AFVALID1S_Pos)       /*!< TPIU ITATBCTR0: AFVALID1SS Mask */

#define TPIU_ITATBCTR0_ATREADY2S_Pos        0U                                         /*!< TPIU ITATBCTR0: ATREADY2S Position */
#define TPIU_ITATBCTR0_ATREADY2S_Msk       (1UL /*<< TPIU_ITATBCTR0_ATREADY2S_Pos*/)   /*!< TPIU ITATBCTR0: ATREADY2S Mask */

#define TPIU_ITATBCTR0_ATREADY1S_Pos        0U                                         /*!< TPIU ITATBCTR0: ATREADY1S Position */
#define TPIU_ITATBCTR0_ATREADY1S_Msk       (1UL /*<< TPIU_ITATBCTR0_ATREADY1S_Pos*/)   /*!< TPIU ITATBCTR0: ATREADY1S Mask */

/** \brief TPIU Integration Mode Control Register Definitions */
#define TPIU_ITCTRL_Mode_Pos                0U                                         /*!< TPIU ITCTRL: Mode Position */
#define TPIU_ITCTRL_Mode_Msk               (0x3UL /*<< TPIU_ITCTRL_Mode_Pos*/)         /*!< TPIU ITCTRL: Mode Mask */

/** \brief TPIU DEVID Register Definitions */
#define TPIU_DEVID_CPPT_Pos                17U                                         /*!< TPIU DEVID: CPPT Position */
#define TPIU_DEVID_CPPT_Msk                (1UL << TPIU_DEVID_CPPT_Pos)                /*!< TPIU DEVID: CPPT Mask */
#define TPIU_DEVID_NRZVALID_Pos            11U                                         /*!< TPIU DEVID: NRZVALID Position */
#define TPIU_DEVID_NRZVALID_Msk            (1UL << TPIU_DEVID_NRZVALID_Pos)            /*!< TPIU DEVID: NRZVALID Mask */

#define TPIU_DEVID_MANCVALID_Pos           10U                                         /*!< TPIU DEVID: MANCVALID Position */
#define TPIU_DEVID_MANCVALID_Msk           (1UL << TPIU_DEVID_MANCVALID_Pos)           /*!< TPIU DEVID: MANCVALID Mask */

#define TPIU_DEVID_PTINVALID_Pos            9U                                         /*!< TPIU DEVID: PTINVALID Position */
#define TPIU_DEVID_PTINVALID_Msk           (1UL << TPIU_DEVID_PTINVALID_Pos)           /*!< TPIU DEVID: PTINVALID Mask */

#define TPIU_DEVID_FIFOSZ_Pos               6U                                         /*!< TPIU DEVID: FIFOSZ Position */
#define TPIU_DEVID_FIFOSZ_Msk              (0x7UL << TPIU_DEVID_FIFOSZ_Pos)            /*!< TPIU DEVID: FIFOSZ Mask */

#define TPIU_DEVID_NrTraceInput_Pos         0U                                         /*!< TPIU DEVID: NrTraceInput Position */
#define TPIU_DEVID_NrTraceInput_Msk        (0x3FUL /*<< TPIU_DEVID_NrTraceInput_Pos*/) /*!< TPIU DEVID: NrTraceInput Mask */

/** \brief TPIU DEVTYPE Register Definitions */
#define TPIU_DEVTYPE_SubType_Pos            4U                                         /*!< TPIU DEVTYPE: SubType Position */
#define TPIU_DEVTYPE_SubType_Msk           (0xFUL /*<< TPIU_DEVTYPE_SubType_Pos*/)     /*!< TPIU DEVTYPE: SubType Mask */

#define TPIU_DEVTYPE_MajorType_Pos          0U                                         /*!< TPIU DEVTYPE: MajorType Position */
#define TPIU_DEVTYPE_MajorType_Msk         (0xFUL << TPIU_DEVTYPE_MajorType_Pos)       /*!< TPIU DEVTYPE: MajorType Mask */

/*@}*/ /* end of group CMSIS_TPIU */


/* ETM Peripheral Register definitions.
 * See here for register documentation:
 * http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.ihi0014q/Chdfiagc.html
 *
 * Not all features are supported on Cortex-M3, see here for details:
 * http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.ddi0337i/CHDBGEED.html
 *
 * The ETM peripheral has a lot of registers, but these are the main ones:
 * - LAR:       Allow write access to other ETM registers
 * - CR:        Enable/disable tracing
 * - TRIGGER:   Select tracing trigger event
 * - SR:        Current status
 * - TECR1      Select areas of code where to enable trace
 * - TECR2      Select comparator for trace enable
 * - TEEVR      Select event for trace enable
 */
typedef struct
{
    __IO uint32_t CR;           /*  Offset: 0x000 (R/W)  Main Control Register */
    __IO uint32_t CCR;          /*  Offset: 0x004 (R/ )  Configuration Code Register */
    __IO uint32_t TRIGGER;      /*  Offset: 0x008 (R/W)  Trigger Event */
    __IO uint32_t ASICCR;       /*  Offset: 0x00C (R/W)  ASIC Control Register */
    __IO uint32_t SR;           /*  Offset: 0x010 (R/W)  ETM Status Register */
    __IO uint32_t SCR;          /*  Offset: 0x014 (R/ )  System Configuration Register */
    __IO uint32_t TSSCR;        /*  Offset: 0x018 (R/W)  TraceEnable Start/Stop Control Register */
    __IO uint32_t TECR2;        /*  Offset: 0x01C (R/W)  TraceEnable Control 2 */
    __IO uint32_t TEEVR;        /*  Offset: 0x020 (R/W)  TraceEnable Event Register */
    __IO uint32_t TECR1;        /*  Offset: 0x024 (R/W)  TraceEnable Control 1 */
    __IO uint32_t FFRR;         /*  Offset: 0x028 (R/W)  FIFOFULL Region Register */
    __IO uint32_t FFLR;         /*  Offset: 0x02C (R/W)  FIFOFULL Level Register */
    __IO uint32_t VDEVR;        /*  Offset: 0x030 (R/W)  ViewData Event Register */
    __IO uint32_t VDCR1;        /*  Offset: 0x034 (R/W)  ViewData Control 1 */
    __IO uint32_t VDCR2;        /*  Offset: 0x038 (R/W)  ViewData Control 2 */
    __IO uint32_t VDCR3;        /*  Offset: 0x03C (R/W)  ViewData Control 3 */
    __IO uint32_t ACVR[16];     /*  Offset: 0x040 (R/W)  Address Comparator Value Registers */
    __IO uint32_t ACTR[16];     /*  Offset: 0x080 (R/W)  Address Comparator Access Type Registers */
    __IO uint32_t DCVR[16];     /*  Offset: 0x0C0 (R/W)  Data Comparator Value Registers */
    __IO uint32_t DCMR[16];     /*  Offset: 0x100 (R/W)  Data Comparator Mask Registers */
    __IO uint32_t CNTRLDVR[4];  /*  Offset: 0x140 (R/W)  Counter Reload Value Registers */
    __IO uint32_t CNTENR[4];    /*  Offset: 0x150 (R/W)  Counter Enable Registers */
    __IO uint32_t CNTRLDEVR[4]; /*  Offset: 0x160 (R/W)  Counter Reload Event Registers */
    __IO uint32_t CNTVR[4];     /*  Offset: 0x170 (R/W)  Counter Value Registers */
    __IO uint32_t SQabEVR[6];   /*  Offset: 0x180 (R/W)  Sequencer State Transition Event Registers */
    __IO uint32_t RESERVED0;
    __IO uint32_t SQR;          /*  Offset: 0x19C (R/W)  Current Sequencer State Register */
    __IO uint32_t EXTOUTEVR[4]; /*  Offset: 0x1A0 (R/W)  External Output Event Registers */
    __IO uint32_t CIDCVR[3];    /*  Offset: 0x1B0 (R/W)  Context ID Comparator Value Registers */
    __IO uint32_t CIDCMR;       /*  Offset: 0x1BC (R/W)  Context ID Comparator Mask Register */
    __IO uint32_t IMPL[8];      /*  Offset: 0x1C0 (R/W)  Implementation specific registers */
    __IO uint32_t SYNCFR;       /*  Offset: 0x1E0 (R/W)  Synchronization Frequency Register */
    __IO uint32_t IDR;          /*  Offset: 0x1E4 (R/ )  ETM ID Register */
    __IO uint32_t CCER;         /*  Offset: 0x1E8 (R/ )  Configuration Code Extension Register */
    __IO uint32_t EXTINSELR;    /*  Offset: 0x1EC (R/W)  Extended External Input Selection Register */
    __IO uint32_t TESSEICR;     /*  Offset: 0x1F0 (R/W)  TraceEnable Start/Stop EmbeddedICE Control Register */
    __IO uint32_t EIBCR;        /*  Offset: 0x1F4 (R/W)  EmbeddedICE Behavior Control Register */
    __IO uint32_t TSEVR;        /*  Offset: 0x1F8 (R/W)  Timestamp Event Register, ETMv3.5 */
    __IO uint32_t AUXCR;        /*  Offset: 0x1FC (R/W)  Auxiliary Control Register, ETMv3.5 */
    __IO uint32_t TRACEIDR;     /*  Offset: 0x200 (R/W)  CoreSight Trace ID Register */
    __IO uint32_t RESERVED1;
    __IO uint32_t IDR2;         /*  Offset: 0x208 (R/ )  ETM ID Register 2 */
    __IO uint32_t RESERVED2[13];
    __IO uint32_t VMIDCVR;      /*  Offset: 0x240 (R/ )  VMID Comparator Value Register, ETMv3.5 */
    __IO uint32_t RESERVED3[47];
    __IO uint32_t OSLAR;        /*  Offset: 0x300 (R/W)  OS Lock Access Register */
    __IO uint32_t OSLSR;        /*  Offset: 0x304 (R/ )  OS Lock Status Register */
    __IO uint32_t OSSRR;        /*  Offset: 0x308 (R/W)  OS Save and Restore Register */
    __IO uint32_t RESERVED4;
    __IO uint32_t PDCR;         /*  Offset: 0x310 (R/W)  Power Down Control Register, ETMv3.5 */
    __IO uint32_t PDSR;         /*  Offset: 0x314 (R/ )  Device Power-Down Status Register */
    __IO uint32_t RESERVED5[762];
    __IO uint32_t ITCTRL;       /*  Offset: 0xF00 (R/ )  Integration Mode Control Register */
    __IO uint32_t RESERVED6[39];
    __IO uint32_t CLAIMSET;     /*  Offset: 0xFA0 (R/ )  Claim Tag Set Register */
    __IO uint32_t CLAIMCLR;     /*  Offset: 0xFA4 (R/ )  Claim Tag Clear Register */
    __IO uint32_t RESERVED7[2];
    __IO uint32_t LAR;          /*  Offset: 0xFB0 (R/ )  Lock Access Register */
    __IO uint32_t LSR;          /*  Offset: 0xFB4 (R/ )  Lock Status Register */
    __IO uint32_t AUTHSTATUS;   /*  Offset: 0xFB8 (R/ )  Authentication Status Register */
    __IO uint32_t RESERVED8[3];
    __IO uint32_t DEVID;        /*  Offset: 0xFC8 (R/ )  CoreSight Device Configuration Register */
    __IO uint32_t DEVTYPE;      /*  Offset: 0xFCC (R/ )  CoreSight Device Type Register */
    __IO uint32_t PIDR4;        /*  Offset: 0xFD0 (R/ )  Peripheral ID4 */
    __IO uint32_t PIDR5;        /*  Offset: 0xFD4 (R/ )  Peripheral ID5 */
    __IO uint32_t PIDR6;        /*  Offset: 0xFD8 (R/ )  Peripheral ID6 */
    __IO uint32_t PIDR7;        /*  Offset: 0xFDC (R/ )  Peripheral ID7 */
    __IO uint32_t PIDR0;        /*  Offset: 0xFE0 (R/ )  Peripheral ID0 */
    __IO uint32_t PIDR1;        /*  Offset: 0xFE4 (R/ )  Peripheral ID1 */
    __IO uint32_t PIDR2;        /*  Offset: 0xFE8 (R/ )  Peripheral ID2 */
    __IO uint32_t PIDR3;        /*  Offset: 0xFEC (R/ )  Peripheral ID3 */
    __IO uint32_t CIDR0;        /*  Offset: 0xFF0 (R/ )  Component ID0 */
    __IO uint32_t CIDR1;        /*  Offset: 0xFF4 (R/ )  Component ID1 */
    __IO uint32_t CIDR2;        /*  Offset: 0xFF8 (R/ )  Component ID2 */
    __IO uint32_t CIDR3;        /*  Offset: 0xFFC (R/ )  Component ID3 */
} ETMv3_Type;

#define ETM_CR_POWERDOWN                0x00000001
#define ETM_CR_MONITORCPRT              0x00000002
#define ETM_CR_TRACE_DATA               0x00000004
#define ETM_CR_TRACE_ADDR               0x00000008
#define ETM_CR_PORTSIZE_1BIT            0x00200000
#define ETM_CR_PORTSIZE_2BIT            0x00200010
#define ETM_CR_PORTSIZE_4BIT            0x00000000
#define ETM_CR_PORTSIZE_8BIT            0x00000010
#define ETM_CR_PORTSIZE_16BIT           0x00000020
#define ETM_CR_STALL_PROCESSOR          0x00000080
#define ETM_CR_BRANCH_OUTPUT            0x00000100
#define ETM_CR_DEBUGREQ                 0x00000200
#define ETM_CR_PROGRAMMING              0x00000400
#define ETM_CR_ETMEN                    0x00000800
#define ETM_CR_CYCLETRACE               0x00001000
#define ETM_CR_CONTEXTID_8BIT           0x00004000
#define ETM_CR_CONTEXTID_16BIT          0x00008000
#define ETM_CR_CONTEXTID_32BIT          0x0000C000
#define ETM_CR_CONTEXTID_8BIT           0x00004000
#define ETM_CR_PORTMODE_ONCHIP          0x00000000
#define ETM_CR_PORTMODE_2_1             0x00010000
#define ETM_CR_PORTMODE_IMPL            0x00030000
#define ETM_CR_PORTMODE_1_1             0x00002000
#define ETM_CR_PORTMODE_1_2             0x00022000
#define ETM_CR_PORTMODE_1_3             0x00012000
#define ETM_CR_PORTMODE_1_4             0x00032000
#define ETM_CR_SUPPRESS_DATA            0x00040000
#define ETM_CR_FILTER_CPRT              0x00080000
#define ETM_CR_DATA_ONLY                0x00100000
#define ETM_CR_BLOCK_DEBUGGER           0x00400000
#define ETM_CR_BLOCK_SOFTWARE           0x00800000
#define ETM_CR_ACCESS                   0x01000000
#define ETM_CR_PROCSEL_Pos              25
#define ETM_CR_TIMESTAMP                0x10000000
#define ETM_CR_VMID                     0x40000000

#define ETM_SR_PROGSTATUS               0x00000002
#define ETM_SR_TRIGSTATUS               0x00000008

#define ETM_TECR1_EXCLUDE               0x01000000
#define ETM_TECR1_TSSEN                 0x02000000

#define ETM_FFRR_EXCLUDE                0x01000000

#define ETM_LAR_KEY                     0xC5ACCE55

typedef struct {
  __IOM uint64_t H;
  __IOM uint64_t L;
} uint128_t;

typedef struct {
        uint32_t RESERVED0[1U];
  __IOM uint32_t PRGCTLR;         /*!< Offset: 0x004 (R/W)  Programming Control Register */
  __IOM uint32_t PROCSELR;        /*!< Offset: 0x008 (R/W)  PE Select Control Register */
  __IM  uint32_t STATR;           /*!< Offset: 0x00C (R/ )  Trace Status Register */
  __IOM uint32_t CONFIGR;         /*!< Offset: 0x010 (R/W)  Trace Configuration Register */
        uint32_t RESERVED2[1U];
  __IOM uint32_t AUXCTLR;         /*!< Offset: 0x018  (R/W)  Auxiliary Control Register */
        uint32_t RESERVED3[1U];
  __IOM uint32_t EVENTCTLR0;      /*!< Offset: 0x020 (R/W)  Event Control 0 Register */
  __IOM uint32_t EVENTCTLR1;      /*!< Offset: 0x024 (R/W)  Event Control 1 Register */
        uint32_t RESERVED4[1U];
  __IOM uint32_t STALLCTLR;       /*!< Offset: 0x02C (R/W)  Stall Control Register */
  __IOM uint32_t TSCTLR;          /*!< Offset: 0x030 (R/W)  Global Timestamp Control Register */
  __IM  uint32_t SYNCPR;          /*!< Offset: 0x034 (R/ )  Synchronization Period Register */
  __IOM uint32_t CCCTLR;          /*<! Offset: 0x038 (R/W)  Cycle Count Control Register */
  __IOM uint32_t BBCTLR;          /*<! Offset: 0x03C (R/W)  Branch Broadcast Control Register */
  __IOM uint32_t TRACEIDR;        /*<! Offset: 0x040 (R/W)  Trace ID Register */
  __IOM uint32_t CQCTLR;          /*<! Offset: 0x044 (R/W)  Q Element Control Register */
        uint32_t RESERVED6[14U];
  __IOM uint32_t VICTLR;          /*<! Offset: 0x080 (R/W)  ViewInst Main Control Register */
  __IOM uint32_t VIIECTLR;        /*<! Offset: 0x084 (R/W)  ViewInst Include/Exclude Control Register */
  __IOM uint32_t VISSCTLR;        /*<! Offset: 0x088 (R/W)  ViewInst Start/Stop Control Register */
  __IOM uint32_t VIPCSSCTLR;      /*<! Offset: 0x08C (R/W)  ViewInst Start/Stop PE Comparator Control Register */
        uint32_t RESERVED7[4U];
  __IOM uint32_t VDCTLR;          /*<! Offset: 0x0A0 (R/W)  ViewData Main Control Register */
  __IOM uint32_t VDSACCTLR;       /*<! Offset: 0x0A4 (R/W)  ViewData Include/Exclude Single Address Comparator Control Register */
  __IOM uint32_t VDARCCTLR;       /*<! Offset: 0x0A8 (R/W)  ViewData Include/Exclude Address Range Comparator Control Register */
        uint32_t RESERVED8[21U];
  __IOM uint32_t SEQEVR[3U];      /*<! Offset: 0x100 (R/W)  Sequencer State Transition Control Register 0-2 */
        uint32_t RESERVED9[3U];
  __IOM uint32_t SEQRSTEVR;       /*<! Offset: 0x118 (R/W)  Sequencer Reset Control Register */
  __IOM uint32_t SEQSTR;          /*<! Offset: 0x11C (R/W)  Sequencer State Register */
  __IOM uint32_t EXTINSELR;       /*<! Offset: 0x120 (R/W)  External Input Select Register */
        uint32_t RESERVED10[7U];
  __IOM uint32_t CNTRLDVR[4U];    /*<! Offset: 0x140 (R/W)  Counter Reload Value Register 0-3 */
  __IOM uint32_t CNTCTLR[4U];     /*<! Offset: 0x150 (R/W)  Counter Control Register 0-3 */
  __IOM uint32_t CNTVR[4U];       /*<! Offset: 0x160 (R/W)  Counter Value Register 0-3 */
        uint32_t RESERVED11[4U];
  __IM  uint32_t IDR8;            /*<! Offset: 0x180 (R/ )  ID Register 8 */
  __IM  uint32_t IDR9;            /*<! Offset: 0x184 (R/ )  ID Register 9 */
  __IM  uint32_t IDR10;           /*<! Offset: 0x188 (R/ )  ID Register 10 */
  __IM  uint32_t IDR11;           /*<! Offset: 0x18C (R/ )  ID Register 11 */
  __IM  uint32_t IDR12;           /*<! Offset: 0x190 (R/ )  ID Register 12 */
  __IM  uint32_t IDR13;           /*<! Offset: 0x194 (R/ )  ID Register 13 */
        uint32_t RESERVED12[10U];
  __IOM uint32_t IMSPECR[8U];     /*<! Offset: 0x1C0 (R/W)  Implementation Specific Register 0-7 */
  __IM  uint32_t IDR0;            /*<! Offset: 0x1E0 (R/ )  ID Register 0 */
  __IM  uint32_t IDR1;            /*<! Offset: 0x1E4 (R/ )  ID Register 1 */
  __IM  uint32_t IDR2;            /*<! Offset: 0x1E8 (R/ )  ID Register 2 */
  __IM  uint32_t IDR3;            /*<! Offset: 0x1EC (R/ )  ID Register 3 */
  __IM  uint32_t IDR4;            /*<! Offset: 0x1F0 (R/ )  ID Register 4 */
  __IM  uint32_t IDR5;            /*<! Offset: 0x1F4 (R/ )  ID Register 5 */
  __IM  uint32_t IDR6;            /*<! Offset: 0x1F8 (R/ )  ID Register 6 */
  __IM  uint32_t IDR7;            /*<! Offset: 0x1FC (R/ )  ID Register 7 */
  __IOM uint32_t RSCTLR[32U];     /*<! Offset: 0x200 (R/W)  Resource Selection Register 0-31. 0 and 1 are reserved. */
  __IOM uint32_t SSCCR[8U];       /*<! Offset: 0x280 (R/W)  Single-shot Comparator Control Register 0-7 */
  __IOM uint32_t SSCSR[8U];       /*<! Offset: 0x2A0 (R/W)  Single-shot Comparator Status Register 0-7 */
  __IOM uint32_t SSPCICR[8U];     /*<! Offset: 0x2C0 (R/W)  Single-shot Comparator PE Input Control Register 0-7 */
        uint32_t RESERVED13[8U];
  __IO  uint32_t OSLAR;           /*<! Offset: 0x300 ( /W)  OS Lock Access Register */
  __IM  uint32_t OSLSR;           /*<! Offset: 0x304 (R/ )  OS Lock Status Register */
        uint32_t RESERVED14[2U];
  __IOM uint32_t PDCR;            /*<! Offset: 0x310 (R/W)  Power Down Control Register */
  __IM  uint32_t PDSR;            /*<! Offset: 0x314 (R/ )  Power Down Status Register */
        uint32_t RESERVED15[58U];
  __IOM uint64_t ACVR[16U];       /*<! Offset: 0x400 (R/W)  Address Comparator Value Register 0-15 */
  __IOM uint64_t ACATR[16U];      /*<! Offset: 0x480 (R/W)  Address Comparator Access Type Register 0-15 */
  __IOM uint128_t DVCVR[8U];      /*<! Offset: 0x500 (R/W)  Data Value Comparator Value Register 0-7 */
  __IOM uint128_t DVCMR[8U];      /*<! Offset: 0x580 (R/W)  Data Value Comparator Mask Register 0-7 */
  __IOM uint64_t CIDVR[8U];       /*<! Offset: 0x600 (R/W)  Context identifier Comparator Value Register 0-7 */
  __IOM uint64_t VMIDCVR[8U];     /*<! Offset: 0x640 (R/W)  Virtual Context Identifier Comparator Value Register 0-7 */
  __IOM uint32_t CIDCCTLR0;       /*<! Offset: 0x680 (R/W)  Context Identifier Comparator Control Register 0 */
  __IOM uint32_t CIDCCTLR1;       /*<! Offset: 0x684 (R/W)  Context Identifier Comparator Control Register 1 */
  __IOM uint32_t VMIDCCTLR0;      /*<! Offset: 0x688 (R/W)  Virtual Context Identifier Comparator Control Register 0 */
  __IOM uint32_t VMIDCCTLR1;      /*<! Offset: 0x68C (R/W)  Virtual Context Identifier Comparator Control Register 1 */
        uint32_t RESERVED16[533U];
  __IOM uint32_t ITATBIDR;        /*<! Offset: 0xEE4 (R/W)  Integration ATB Identification Register */
        uint32_t RESERVED17[3U];
  __IM  uint32_t ITIATBINR;       /*<! Offset: 0xEF4 (R/ )  Integration Instruction ATB In Register */
        uint32_t RESERVED18[1U];
  __IOM uint32_t ITIATBOUTR;      /*<! Offset: 0xEFC (R/W)  Integration Instruction ATB Out Register */
  __IOM uint32_t ITCTRL;          /*<! Offset: 0xF00 (R/W)  Integration Mode Control Register */
        uint32_t RESERVED19[39U];
  __IOM uint32_t CLAIMSET;        /*<! Offset: 0xFA0 (R/W)  Claim Tag Set Register */
  __IOM uint32_t CLAIMCLR;        /*<! Offset: 0xFA4 (R/W)  Claim Tag Clear Register */
  __IM  uint32_t DEVAFF0;         /*<! Offset: 0xFA8 (R/ )  Device Affinity Register 0 */
  __IM  uint32_t DEVAFF1;         /*<! Offset: 0xFAC (R/ )  Device Affinity Register 1 */
  __IOM uint32_t LAR;             /*<! Offset: 0xFB0 (R/W)  Software Lock Acccess Register */
  __IOM uint32_t LSR;             /*<! Offset: 0xFB4 (R/W)  Software Lock Status Register */
  __IM  uint32_t AUTHSTATUS;      /*<! Offset: 0xFB8 (R/W)  Authentication Status Register */
  __IM  uint32_t DEVARCH;         /*<! Offset: 0xFBC (R/W)  Device Architecture Register */
  __IM  uint32_t DEVID2;          /*<! Offset: 0xFC0 (R/ )  CoreSight Device Configuration Register 2 */
  __IM  uint32_t DEVID1;          /*<! Offset: 0xFC0 (R/ )  CoreSight Device Configuration Register 1 */
  __IM  uint32_t DEVID0;          /*<! Offset: 0xFC8 (R/ )  CoreSight Device Configuration Register 0 */
  __IM  uint32_t DEVTYPE;         /*<! Offset: 0xFCC (R/ )  CoreSight Device Type Register */
  __IM  uint32_t PIDR4;           /*<! Offset: 0xFD0 (R/ )  Peripheral ID4 */
  __IM  uint32_t PIDR5;           /*<! Offset: 0xFD4 (R/ )  Peripheral ID5 */
  __IM  uint32_t PIDR6;           /*<! Offset: 0xFD8 (R/ )  Peripheral ID6 */
  __IM  uint32_t PIDR7;           /*<! Offset: 0xFDC (R/ )  Peripheral ID7 */
  __IM  uint32_t PIDR0;           /*<! Offset: 0xFE0 (R/ )  Peripheral ID0 */
  __IM  uint32_t PIDR1;           /*<! Offset: 0xFE4 (R/ )  Peripheral ID1 */
  __IM  uint32_t PIDR2;           /*<! Offset: 0xFE8 (R/ )  Peripheral ID2 */
  __IM  uint32_t PIDR3;           /*<! Offset: 0xFEC (R/ )  Peripheral ID3 */
  __IM  uint32_t CIDR0;           /*<! Offset: 0xFF0 (R/ )  Component ID0 */
  __IM  uint32_t CIDR1;           /*<! Offset: 0xFF4 (R/ )  Component ID1 */
  __IM  uint32_t CIDR2;           /*<! Offset: 0xFF8 (R/ )  Component ID2 */
  __IM  uint32_t CIDR3;           /*<! Offset: 0xFFC (R/ )  Component ID3 */
} ETMv4_Type;

#define ETM4_CCCTLR_THRESHOLD_Pos 0U
#define ETM4_CCCTLR_THRESHOLD_Msk (0x0FFFUL /*<< ETM4_CCCTLR_THRESHOLD_Pos*/)

#define ETM4_CONFIGR_BB_Pos 3U
#define ETM4_CONFIGR_BB_Mask (1UL << ETM4_CONFIGR_BB_Pos)
#define ETM4_CONFIGR_CCI_Pos 4U
#define ETM4_CONFIGR_CCI_Mask (1UL << ETM4_CONFIGR_CCI_Pos)
#define ETM4_CONFIGR_COND_Pos 8U
#define ETM4_CONFIGR_COND_Mask (7UL << ETM4_CONFIGR_COND_Pos)
#define ETM4_CONFIGR_COND_LOAD (1UL << ETM4_CONFIGR_COND_Pos)
#define ETM4_CONFIGR_COND_STORE (2UL << ETM4_CONFIGR_COND_Pos)
#define ETM4_CONFIGR_COND_LOADSTORE (3UL << ETM4_CONFIGR_COND_Pos)
#define ETM4_CONFIGR_COND_ALL (ETM4_CONFIGR_COND_Mask)
#define ETM4_CONFIGR_TS_Pos 11U
#define ETM4_CONFIGR_TS_Mask (1UL << ETM4_CONFIGR_TS_Pos)
#define ETM4_CONFIGR_RS_Pos 12U
#define ETM4_CONFIGR_RS_Mask (1UL << ETM4_CONFIGR_RS_Pos)

#define ETM4_STALLCTLR_LEVEL_Pos 0U
#define ETM4_STALLCTLR_LEVEL_Mask (0xFUL /*<< ETM4_STALLCTLR_LEVEL_Pos*/)
#define ETM4_STALLCTLR_ISTALL_Pos 8U
#define ETM4_STALLCTLR_ISTALL_Mask (1UL << ETM4_STALLCTLR_ISTALL_Pos)

#define ETM4_VICTLR_EVENT_Pos 0U
#define ETM4_VICTLR_EVENT_Mask (0xFFUL /*<< ETM4_VICTLR_EVENT_Pos*/)
#define ETM4_VICTLR_SSSTATUS_Pos 9U
#define ETM4_VICTLR_SSSTATUS_Mask (1UL << ETM4_VICTLR_SSSTATUS_Pos)
#define ETM4_VICTLR_TRCRESET_Pos 10U
#define ETM4_VICTLR_TRCRESET_Mask (1UL << ETM4_VICTLR_TRCRESET_Pos)
#define ETM4_VICTLR_TRCERR_Pos 11U
#define ETM4_VICTLR_TRCERR_Mask (1UL << ETM4_VICTLR_TRERRT_Pos)

#define ETM4_EVENTCTLR1_LPOVERRIDE_Pos 12U
#define ETM4_EVENTCTLR1_LPOVERRIDE_Mask (1UL << ETM4_EVENTCTLR1_LPOVERRIDE_Pos)
#define ETM4_EVENTCTLR1_ATB_Pos 11U
#define ETM4_EVENTCTLR1_ATB_Mask (1UL << ETM4_EVENTCTLR1_ATB_Pos)
#define ETM4_EVENTCTLR1_INSTEN_Pos 0U
#define ETM4_EVENTCTLR1_INSTEN_Mask (0xFUL /*<< ETM4_EVENTCTLR1_INSTEN_Pos*/)

#define ETM4_IDR0_BB_Pos 5U
#define ETM4_IDR0_BB_Mask (1UL << ETM4_IDR0_BB_Pos)
#define ETM4_IDR0_COND_Pos 6U
#define ETM4_IDR0_COND_Mask (1UL << ETM4_IDR0_COND_Pos)
#define ETM4_IDR0_CCI_Pos 7U
#define ETM4_IDR0_CCI_Mask (1UL << ETM4_IDR0_CCI_Pos)
#define ETM4_IDR0_RETSTACK_Pos 8U
#define ETM4_IDR0_RETSTACK_Mask (1UL << ETM4_IDR0_RETSTACK_Pos)
#define ETM4_IDR0_NUMEVENT_Pos 10U
#define ETM4_IDR0_NUMEVENT_Mask (3UL << ETM4_IDR0_NUMEVENT_Pos)

#define ETM4_RSC_PAIRINV_Pos 21U
#define ETM4_RSC_PAIRINV_Mask (1UL << ETM4_RSC_PAIRINV_Pos)
#define ETM4_RSC_INV_Pos 20U
#define ETM4_RSV_INV_Mask (1UL << ETM4_RSV_INV_Pos)
#define ETM4_RSC_GROUP_Pos 16U
#define ETM4_RSC_GROUP_Mask (0xFUL << ETM4_RSC_GROUP_Pos)
#define ETM4_RSC_GROUP_EXTIN (0UL << ETM4_RSC_GROUP_Pos)
#define ETM4_RSC_GROUP_PECOMP (1UL << ETM4_RSC_GROUP_Pos)
#define ETM4_RSC_GROUP_COUNTER (2UL << ETM4_RSC_GROUP_Pos)
#define ETM4_RSC_GROUP_SSHOT (3UL << ETM4_RSC_GROUP_Pos)
#define ETM4_RSC_GROUP_SADDRCOMP (4UL << ETM4_RSC_GROUP_Pos)

#define ETM4_RSC_SELECT_Pos 0U
#define ETM4_RSC_SELECT_Mask (0xFFFFUL /*<< ETM4_RSC_SELECT_Pos*/)

typedef struct {
  __IOM uint32_t CNTCR;           /*!< Offset: 0x000 (R/W)  Counter Control Register */
  __IM  uint32_t CNTSR;           /*!< Offset: 0x004 (R/ )  Counter Status Register */
  __IOM uint32_t CNTCVL;          /*!< Offset: 0x008 (R/W)  Current value of Counter[31:0] */
  __IOM uint32_t CNTCVU;          /*!< Offset: 0x00C (R/W)  Current value of Counter[63:32] */
        uint32_t RESERVED0[4U];
  __IOM uint32_t CNTFID0;         /*!< Offset: 0x020 (R/W)  Base Frequency ID Register */
        uint32_t RESERVED1[949U];
  __IM  uint32_t ISTAT;           /*!< Offset: 0xEF8 (R/ )  Integration Test Status Register */
        uint32_t RESERVED2[1U];
  __IOM uint32_t ITCTRL;          /*!< Offset: 0xF00 (R/W)  Integration Mode Control Register */
        uint32_t RESERVED3[51U];
  __IM  uint32_t PIDR4;           /*!< Offset: 0xFD0 (R/ )  PIDR4 */
  __IM  uint32_t PIDR5;           /*!< Offset: 0xFD4 (R/ )  PIDR5 */
  __IM  uint32_t PIDR6;           /*!< Offset: 0xFD8 (R/ )  PIDR6 */
  __IM  uint32_t PIDR7;           /*!< Offset: 0xFDC (R/ )  PIDR7 */
  __IM  uint32_t PIDR0;           /*!< Offset: 0xFE0 (R/ )  PIDR0 */
  __IM  uint32_t PIDR1;           /*!< Offset: 0xFE4 (R/ )  PIDR1 */
  __IM  uint32_t PIDR2;           /*!< Offset: 0xFE8 (R/ )  PIDR2 */
  __IM  uint32_t PIDR3;           /*!< Offset: 0xFEC (R/ )  PIDR3 */
  __IM  uint32_t CIDR0;           /*!< Offset: 0xFF0 (R/ )  CIDR0 */
  __IM  uint32_t CIDR1;           /*!< Offset: 0xFF4 (R/ )  CIDR1 */
  __IM  uint32_t CIDR2;           /*!< Offset: 0xFF8 (R/ )  CIDR2 */
  __IM  uint32_t CIDR3;           /*!< Offset: 0xFFC (R/ )  CIDR3 */
} TSGEN_Type;

typedef struct {
  __IOM uint32_t FUNNELCONTROL;     /*!< Offset: 0x000 (R/W)  Funnel Control Register */
  __IOM uint32_t PIORITYCONTROL;    /*!< Offset: 0x004 (R/W)  Priority Control Register */
        uint32_t RESERVED0[953U];
  __IOM uint32_t ITATBDATA0;        /*!< Offset: 0xEEC (R/W)  Integration Test Data Register */
  __IOM uint32_t ITATBCTR3;         /*!< Offset: 0xEF0 (R/W)  Integration Test Control Register 3 */
  __IOM uint32_t ITATBCTR2;         /*!< Offset: 0xEF4 (R/W)  Integration Test Control Register 2 */
  __IOM uint32_t ITATBCTR1;         /*!< Offset: 0xEF8 (R/W)  Integration Test Control Register 1 */
  __IOM uint32_t ITATBCTR0;         /*!< Offset: 0xEFC (R/W)  Integration Test Control Register 0 */
  __IOM uint32_t ITCTRL;            /*!< Offset: 0xF00 (R/W)  Integration Mode Control Register */
        uint32_t RESERVED1[39U];
  __IOM uint32_t CLAIMSET;          /*!< Offset: 0xFA0 (R/W)  Claim Tag Set Register */
  __IOM uint32_t CLAIMCLR;          /*!< Offset: 0xFA4 (R/W)  Claim Tag Clear Register */
  __IM  uint32_t DEVAFF0;           /*!< Offset: 0xFA8 (R/ )  Device Affinity Register 0 */
  __IM  uint32_t DEVAFF1;           /*!< Offset: 0xFAC (R/ )  Device Affinity Register 1 */
  __IOM uint32_t LAR;               /*!< Offset: 0xFB0 (R/W)  Software Lock Access Register */
  __IM  uint32_t LSR;               /*!< Offset: 0xFB4 (R/ )  Software Lock Status Register */
  __IM  uint32_t AUTHSTATUS;        /*!< Offset: 0xFB8 (R/ )  Authentication Status Register */
  __IM  uint32_t DEVARCH;           /*!< Offset: 0xFBC (R/ )  Device Architecture Register */
  __IM  uint32_t DEVID2;            /*!< Offset: 0xFC0 (R/ )  Device Configuration Register 2 */
  __IM  uint32_t DEVID1;            /*!< Offset: 0xFC4 (R/ )  Device Configuration Register 1 */
  __IM  uint32_t DEVID;             /*!< Offset: 0xFC8 (R/ )  Device Configuration Register */
  __IM  uint32_t DEVTYPE;           /*!< Offset: 0xFCC (R/ )  Device Type Identifier Register */
  __IM  uint32_t PIDR4;             /*!< Offset: 0xFD0 (R/ )  PIDR4 */
  __IM  uint32_t PIDR5;             /*!< Offset: 0xFD4 (R/ )  PIDR5 */
  __IM  uint32_t PIDR6;             /*!< Offset: 0xFD8 (R/ )  PIDR6 */
  __IM  uint32_t PIDR7;             /*!< Offset: 0xFDC (R/ )  PIDR7 */
  __IM  uint32_t PIDR0;             /*!< Offset: 0xFE0 (R/ )  PIDR0 */
  __IM  uint32_t PIDR1;             /*!< Offset: 0xFE4 (R/ )  PIDR1 */
  __IM  uint32_t PIDR2;             /*!< Offset: 0xFE8 (R/ )  PIDR2 */
  __IM  uint32_t PIDR3;             /*!< Offset: 0xFEC (R/ )  PIDR3 */
  __IM  uint32_t CIDR0;             /*!< Offset: 0xFF0 (R/ )  CIDR0 */
  __IM  uint32_t CIDR1;             /*!< Offset: 0xFF4 (R/ )  CIDR1 */
  __IM  uint32_t CIDR2;             /*!< Offset: 0xFF8 (R/ )  CIDR2 */
  __IM  uint32_t CIDR3;             /*!< Offset: 0xFFC (R/ )  CIDR3 */
} ATB_FUNNEL_Type;

#define STATIC_ASSERT(COND,MSG) typedef char static_assertion_##MSG[(!!(COND))*2-1]
// token pasting madness:
#define COMPILE_TIME_ASSERT3(X,L) STATIC_ASSERT(X,static_assertion_at_line_##L)
#define COMPILE_TIME_ASSERT2(X,L) COMPILE_TIME_ASSERT3(X,L)
#define COMPILE_TIME_ASSERT(X)    COMPILE_TIME_ASSERT2(X,__LINE__)

// Sanitiy checks at compile time that the strucs are well defined
COMPILE_TIME_ASSERT(sizeof(ETMv3_Type) == 0x1000);
COMPILE_TIME_ASSERT(sizeof(ETMv4_Type) == 0x1000);
COMPILE_TIME_ASSERT(sizeof(ATB_FUNNEL_Type) == 0x1000);
COMPILE_TIME_ASSERT(sizeof(TSGEN_Type) == 0x1000);
COMPILE_TIME_ASSERT(sizeof(ITM_Type) == 0x1000);
COMPILE_TIME_ASSERT(sizeof(DWT_Type) == 0x1000);
COMPILE_TIME_ASSERT(sizeof(TPIU_Type) == 0x1000);


#ifndef ITM_BASE
#define ITM_BASE 0xe0000000
#endif
#ifndef DWT_BASE
#define DWT_BASE ((ITM_BASE) + 0x1000UL)
#endif
#ifndef SCB_BASE
#define SCB_BASE ((ITM_BASE) + 0x0ED00UL)
#endif
#ifndef  DCB_BASE
#define DCB_BASE ((ITM_BASE) + 0x0EDF0UL)
#endif
#ifndef DIB_BASE
#define DIB_BASE ((ITM_BASE) + 0x0EFB0UL)
#endif
#ifndef ETM_BASE
#define ETM_BASE ((ITM_BASE) + 0x41000UL)
#endif
#ifndef TPIU_BASE
#define TPIU_BASE ((ITM_BASE) + 0x40000UL)
#endif

#if ETM_VERSION == 3
#define ETM ((ETMv3_Type*)ETM_BASE)
#define ETM_TraceMode() ETM->CR &= ~ETM_CR_PROGRAMMING
#define ETM_SetupMode() ETM->CR |= ETM_CR_PROGRAMMING

#elif ETM_VERSION == 4
#define ETM ((ETMv4_Type *)ETM_BASE)
#define ETM_TraceMode() do { ETM->PRGCTLR = 1; while ((ETM->STATR & 1) != 0) {} } while (0)
#define ETM_SetupMode() do { ETM->PRGCTLR = 0; while ((ETM->STATR & 1) == 0) {} } while (0)

#else
#error "ETM_VERSION must be defined to a supported version"
#endif

#define ITM ((ITM_Type *)ITM_BASE)    /*!< ITM configuration struct */
#define DWT ((DWT_Type *)DWT_BASE)    /*!< DWT configuration struct */
#define TPI ((TPIU_Type *)TPIU_BASE)  /*!< TPIU configuration struct */
#define DCB ((DCB_Type *)DCB_BASE)    /*!< DCB configuration struct */

#endif
