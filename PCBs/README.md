# PCBs

Here we publish gerber files and/or schematics of PCBs we developed.

They all are released under a Creative Commons Attribution 4.0
[CC-BY-4.0 license](https://creativecommons.org/licenses/by/4.0/)

## List of PCBs

### CW313 to JTrace

This adapter board connects to the CW313 board over the JTAG and TRACE
connectors and provides the standard Samtech 20-pin connector to connect
to a Segger J-Trace or to an Orbtrace device.

**Changelog**:

-   v0.2: added a solder bridge on VTref

### CW313 Trace to Husky

This adapter board connects to the CW313 board over the JTAG and TRACE
connectors and provides a HE-10 20-pin connector that matches the pinout for
JTAG and TRACE on a ChipWhisperer&reg; Husky/HuskyPlus.

A standard ribbon cable such as the one shipped with ChipWhisperer&reg; devices
can be used.

### Husky to Segger

This small adapter board connects to a ChipWhisperer&reg; Husky/HuskyPlus 20-pin
connector and provides the standard Samtech 20-pin connector used for ARM
Trace and SWD.

**Changelog**:

-   v0.2: added a solder bridge (normally open) on VTref
