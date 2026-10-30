/*
 * Minimal RCP stub for emulation - just defines macros as NOPs
 * since RC_CANARY=0 and HARDENING=0
 */
#ifndef _HARDWARE_RCP_H
#define _HARDWARE_RCP_H

#ifdef __ASSEMBLER__

/* All RCP macros are NOPs since HARDENING=0 */

.macro rcp_bvalid r
.endm

.macro rcp_bvalid_nodelay r
.endm

.macro rcp_btrue r
.endm

.macro rcp_btrue_nodelay r
.endm

.macro rcp_bfalse r
.endm

.macro rcp_bfalse_nodelay r
.endm

.macro rcp_b2valid b0, b1
.endm

.macro rcp_b2valid_nodelay b0, b1
.endm

.macro rcp_b2and b0, b1
.endm

.macro rcp_b2and_nodelay b0, b1
.endm

.macro rcp_b2or b0, b1
.endm

.macro rcp_b2or_nodelay b0, b1
.endm

.macro rcp_bxorvalid b, mask
.endm

.macro rcp_bxorvalid_nodelay b, mask
.endm

.macro rcp_bxortrue b, mask
.endm

.macro rcp_bxortrue_nodelay b, mask
.endm

.macro rcp_bxorfalse b, mask
.endm

.macro rcp_bxorfalse_nodelay b, mask
.endm

.macro rcp_ivalid x, parity
.endm

.macro rcp_ivalid_nodelay x, parity
.endm

.macro rcp_iequal x, y
.endm

.macro rcp_iequal_nodelay x, y
.endm

.macro rcp_count_set cnt
.endm

.macro rcp_count_set_nodelay cnt
.endm

.macro rcp_count_check cnt
.endm

.macro rcp_count_check_nodelay cnt
.endm

.macro rcp_canary_get x, tag
.endm

.macro rcp_canary_get_nodelay x, tag
.endm

.macro rcp_canary_check x, tag
.endm

.macro rcp_canary_check_nodelay x, tag
.endm

.macro rcp_panic
    /* Just an infinite loop in emulation */
    b .
.endm

#endif /* __ASSEMBLER__ */

#endif /* _HARDWARE_RCP_H */
