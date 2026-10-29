cmake_minimum_required(VERSION 3.13)
# initialize the SDK based on PICO_SDK_PATH
# note: this must happen before project()
set(PICO_SDK_FETCH_FROM_GIT 1)
set(PICO_PLATFORM rp2350)
set(PICO_BOARD pico2)
include(pico_sdk_import.cmake)

project(rp2350_hacking_challenge_2)

pico_sdk_init()

add_executable(rp2350_hacking_challenge_2)

set(PICO_FLASH_SIZE_BYTES 4194304)
set(PICO_RP2350_A2_SUPPORTED 1)

target_compile_definitions(rp2350_hacking_challenge_2 PRIVATE
        # Define a custom external clock so that the CPU can run at the same speed
        XOSC_HZ=24000000

        # Make sure that USB still works at 48MHz
        PLL_USB_REFDIV=2
        PLL_USB_VCO_FREQ_HZ=1200000000
        PLL_USB_POSTDIV1=5
        PLL_USB_POSTDIV2=5

        # Slow down the CPU to be synchronized with the external clock
        PLL_SYS_REFDIV=1
        PLL_SYS_VCO_FREQ_HZ=1176000000
        PLL_SYS_POSTDIV1=7
        PLL_SYS_POSTDIV2=7
        SYS_CLK_HZ=24000000
        PICO_XOSC_STARTUP_DELAY_MULTIPLIER=64

        # Board specifics
        PICO_STDIO_USB=1
        PICO_STDIO_UART=1

        # Simpleserial v2
        SS_VER=SS_VER_2_1
        UART_BAUD_RATE=230400
)

target_compile_definitions(rp2350_hacking_challenge_2 PRIVATE
	FIRMWARE_CONFIG_NAME="HARD"
        # The following are to reduce the size of the binary
        RC_COUNT=0
        HARDENING=0
        DOUBLE_HARDENING=0

        INLINE_REF_ROUNDKEY_SHARES_S=1
        INLINE_REF_ROUNDKEY_HVPERMS_S=1
        INLINE_SHIFT_ROWS_S=1
        INLINE_MAP_SBOX_S=1

        CALLER_INIT_RCP_COUNT=1
)

target_include_directories(rp2350_hacking_challenge_2 PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/config_hard
)
target_sources(rp2350_hacking_challenge_2 PRIVATE 
        rp2350_hacking_challenge_2.c
        aes.S
        tinyaes.c
        mbedtls_aes.c
        simpleserial.c
)
#pico_set_binary_type(rp2350_hacking_challenge_2 no_flash)
target_link_libraries(rp2350_hacking_challenge_2 PRIVATE
        pico_stdlib
        pico_stdio_uart
        pico_status_led
        pico_mbedtls
        hardware_pio
        hardware_adc
)

pico_add_extra_outputs(rp2350_hacking_challenge_2)

# enable usb output and uart output for stdio
pico_enable_stdio_usb(rp2350_hacking_challenge_2 1)
pico_enable_stdio_uart(rp2350_hacking_challenge_2 1)

######
# Add additional firmware with a relaxed AES CTR
######
add_executable(rp2350_hacking_challenge_2_relaxed)

target_compile_definitions(rp2350_hacking_challenge_2_relaxed PRIVATE
        # Define a custom external clock so that the CPU can run at the same speed
        XOSC_HZ=24000000

        # Make sure that USB still works at 48MHz
        PLL_USB_REFDIV=2
        PLL_USB_VCO_FREQ_HZ=1200000000
        PLL_USB_POSTDIV1=5
        PLL_USB_POSTDIV2=5

        # Slow down the CPU to be synchronized with the external clock
        PLL_SYS_REFDIV=1
        PLL_SYS_VCO_FREQ_HZ=1176000000
        PLL_SYS_POSTDIV1=7
        PLL_SYS_POSTDIV2=7
        SYS_CLK_HZ=24000000
        PICO_XOSC_STARTUP_DELAY_MULTIPLIER=64

        # Board specifics
        PICO_STDIO_USB=1
        PICO_STDIO_UART=1

        # Simpleserial v2
        SS_VER=SS_VER_2_1
        UART_BAUD_RATE=230400
)

target_compile_definitions(rp2350_hacking_challenge_2_relaxed PRIVATE
        FIRMWARE_CONFIG_NAME="RELAXED"
        # The following are to reduce the size of the binary
        RC_COUNT=0
        HARDENING=0
        DOUBLE_HARDENING=0

        INLINE_REF_ROUNDKEY_SHARES_S=1
        INLINE_REF_ROUNDKEY_HVPERMS_S=1
        INLINE_SHIFT_ROWS_S=1
        INLINE_MAP_SBOX_S=1

        CALLER_INIT_RCP_COUNT=1
)

target_include_directories(rp2350_hacking_challenge_2_relaxed PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/config_relaxed
)
target_sources(rp2350_hacking_challenge_2_relaxed PRIVATE 
        rp2350_hacking_challenge_2.c
        aes.S
        tinyaes.c
        mbedtls_aes.c
        simpleserial.c
)
#pico_set_binary_type(rp2350_hacking_challenge_2_relaxed no_flash)
target_link_libraries(rp2350_hacking_challenge_2_relaxed PRIVATE
        pico_stdlib
        pico_stdio_uart
        pico_status_led
        pico_mbedtls
        hardware_pio
        hardware_adc
)

pico_add_extra_outputs(rp2350_hacking_challenge_2_relaxed)

# enable usb output and uart output for stdio
pico_enable_stdio_usb(rp2350_hacking_challenge_2_relaxed 1)
pico_enable_stdio_uart(rp2350_hacking_challenge_2_relaxed 1)

