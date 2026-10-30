
MEMORY
{
    /* Code goes in SRAM scratch X */
    CODE (rwx) : ORIGIN = 0x20080000, LENGTH = 4K
    /* Data/workspace in SRAM scratch Y */
    DATA (rw) : ORIGIN = 0x20081000, LENGTH = 4K
    /* Stack at the end */
    STACK (rw) : ORIGIN = 0x20081C00, LENGTH = 1K
}

SECTIONS
{
    .text : {
        *(.text)
        *(.text.*)
    } > CODE

    .data : {
        *(.data)
        *(.data.*)
    } > DATA

    .scratch_y.aes : {
        *(.scratch_y.aes)
    } > DATA

    .bss : {
        *(.bss)
        *(.bss.*)
    } > DATA
    
    /* Provide stack pointer */
    __stack_top = ORIGIN(STACK) + LENGTH(STACK);
}

ENTRY(decrypt)
