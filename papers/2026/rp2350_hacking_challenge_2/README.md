# RP2350 Hacking Challenge 2

This directory contains our solution to the RP2350 Hacking Challenge 2 which
was happened from July 2025 until October 2026.

This provides:

-   Source code for the firmware we wrote under [`firmware/`](firmware/)
-   Pre-compiled firmware under [`firmware/build`](firmware/build/) for both the
    [original configuration](firmware/config_hard/config.h) with countermeasures
    enabled and the [relaxed one](firmware/config_relaxed/config.h) that was
    allowed starting February 2026
-   [`capture_sedpack_dataset.py`](capture_sedpack_dataset.py) python script
    to capture a dataset using `sedpack` and Chipwhisperer Husky.
-   [`end_to_end_attack.py`](end_to_end_attack.py) python script to recover
    AES keys from a given dataset.
-   [`canonical_t2_ref.npy`](canonical_t2_ref.npy) and
    [`trace2_walsh_tree_bp_templates.pkl`](trace2_walsh_tree_bp_templates.pkl)
    which have been computed to match the firmware will countermeasures enabled
    [rp2350_hacking_challenge_2](firmware/build/rp2350_hacking_challenge_2.elf)
-   Unicorn based emulation tooling under [`emulation/`](emulation/)


**Important**: If you compile the firmware yourself, you will need to recalculate
some memory address for the dataset capture script. These are needed to properly
configure ETMv4 and DWT units if you want to capture `train` and `test` splits of
the dataset.
