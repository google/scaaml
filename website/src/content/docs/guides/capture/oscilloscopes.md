---
title: Oscilloscope Support
description: Oscilloscope Support by SCAAML
---

SCAAML has support for context managers for the following oscilloscopes:

-   CW
-   LeCroy TODO
-   PicoScope&reg; 6424E

## Scope Context Manager vs ScopeTemplate

### ScopeTemplate

All of our custom code regarding scopes follows the
`scaaml.capture.scope.scope_template.ScopeTemplate` protocol. Thus it can be
passed seamlessly to `cw.capture_trace` or used on it's own.

```python
class ScopeTemplate(Protocol):
    """A base class for scope objects that can be passed as a scope to
    chipwhisperer API (such as Pico6424E)."""

    def con(self, sn: str | None = None) -> bool:
        """Connect to the attached hardware. Trying to keep compatibility with
        `cw.capture.scopes.OpenADC` and being able to pass as `scope` argument
        to `cw.capture_trace`.

        Args:

          sn (str | None): The serial number of the scope.

        Returns: True if the connection was successful, False otherwise.
        """

    def dis(self) -> bool:
        """Disconnect.

        Returns: True if the disconnection was successful, False otherwise.
        """

    def arm(self) -> None:
        """Setup scope to begin capture when triggered."""

    def capture(self, poll_done: bool = False) -> bool:
        """Capture trace (must be armed first). Same signature as
        cw.capture.scopes.OpenADC.

        Args:

          poll_done (bool): Poll if the capture has finished. Not supported
          everywhere.

        Returns: True if the capture timed out, False if it did not.
        """

    def get_last_trace(self, as_int: bool = False) -> ScopeTraceType:
        """Return the last trace. Same signature as
        `cw.capture.scopes.OpenADC.get_last_trace`.

        Args:

          as_int (bool): Scope dependent. Could be either not implemented or
          checked to be False.
      """

    def __str__(self) -> str:
        """Return string representation of this object."""
```

The main advantage of this is that we can keep compatibility with ChipWhisperer
scopes (without them inheriting from our scope base class or vice versa).

These are the more low-level classes:

-   `scaaml.capture.scope.ps6424e.Pico6424E`
-   `scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationVisa`
-   `scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationSocket`

### Context Managers

Moreover there is also context manager support which initializes, connects, and
disconnects the oscilloscopes safely even in the case of a crash. These classes
implement the `scaaml.capture.scope.scope_base.AbstractSScope`:

-   `scaaml.capture.scope.lecroy.lecroy.LeCroy`
-   `scaaml.capture.scope.picoscope.PicoScope`
-   `scaaml.capture.scope.cw_scope.CWScope`
-   `scaaml.capture.scope.cw_scope.DefaultCWScope`

The difference between `CWScope` and `DefaultCWScope` is that we use the
`DefaultCWScope` just to do a default setup and get a `ChipWhisperer` target
using `cw.target(scope_manager.scope)` while using another scope to take
measurements. On the other hand with `CWScope` we use the ChipWhisperer
built-in oscilloscope.

## PicoScope&reg;

So far we support PicoScope&reg; 6424E. The implementation uses
[picosdk-python-wrappers](https://github.com/picotech/picosdk-python-wrappers)
and needs `libps6000a` (see
[https://www.picotech.com/downloads](https://www.picotech.com/downloads)).
There is support for:

-   analog or digital trigger (MSO pod)
-   setting resolution (where supported)

For detailed documentation of the used API see:
[https://www.picotech.com/download/manuals/picoscope-6000-series-a-api-programmers-guide.pdf](https://www.picotech.com/download/manuals/picoscope-6000-series-a-api-programmers-guide.pdf).

For quick setup one leverage the official GUI which can be downloaded from
[https://www.picotech.com/downloads](https://www.picotech.com/downloads). This
setup is likely to works well even over your favourite remote desktop solution.

```python
import chipwhisperer as cw

from scaaml.capture.scope import PicoScope


# Communication with the device under test.
cw_scope = cw.scope()  # To get the cw.target
scope.default_setup()
target = cw.target(cw_scope)

with PicoScope(
        samples=5_000,
        sample_rate=7_000_000,  # Hz
        offset=0,  # pre-trigger samples
        trace_channel="A",
        trace_probe_range=0.5,  # V
        trace_coupling="AC",  # DC, DC50
        trace_attenuation="1:1",
        trace_bw_limit="PICO_BW_FULL",
        trace_ignore_overflow=False,
        trigger_channel="PORT0",
        trigger_pin=0,  # MSO pod pin
        trigger_hysteresis="PICO_NORMAL_100MV",
        trigger_range=5.0,  # V
        trigger_level=1.0,  # V
        trigger_coupling="DC",
        resolution="PICO_DR_8BIT",
) as oscilloscope:
    trace = cw.capture_trace(oscilloscope.scope, target, pt, key)
```

## LeCroy

Communication with the oscilloscope is supported by both:

-   LXI protocol over TCP using
    [PyVISA](https://pyvisa.readthedocs.io/en/latest/) when the oscilloscope is
    set "Utilities > Utilities Setup > Remote" to LXI.
-   Python socket over TCP/IP using VICP when the oscilloscopes is set
    "Utilities > Utilities Setup > Remote" to TCPIP.

Our choice of protocol is usually the TCPIP for it has automatic recovery
properties for long lasting capture campaigns. This being said the LXI protocol
has been used to capture the
[GPAM](https://cdn.teledynelecroy.com/files/manuals/automation_command_ref_manual_wr.pdf)
ECC datasets and is more tested.

```python
import chipwhisperer as cw

from scaaml.capture.scope import LeCroy


# Communication with the device under test.
cw_scope = cw.scope()  # To get the cw.target
scope.default_setup()
target = cw.target(cw_scope)

# Just an example, notice the {trace_channel} wildcards.
scope_setup_commands = [
    { "command": "C1:TRACE OFF", "query": "C1:TRACE?" },
    { "command": "C2:TRACE OFF", "query": "C2:TRACE?" },
    { "command": "C3:TRACE OFF", "query": "C3:TRACE?" },
    { "command": "C4:TRACE OFF", "query": "C4:TRACE?" },
    { "command": "{trace_channel}:TRACE ON" },
    {
        "command": "MEMORY_SIZE 1e+9",
        "query": "MEMORY_SIZE?"
    },
    {
        "command": "{trace_channel}:VOLT_DIV 0.5V",
        "query": "{trace_channel}:VOLT_DIV?"
    },
    {
        "command": "{trace_channel}:COUPLING D1M",
        "query": "{trace_channel}:COUPLING?"
    },
    {
        "command": "BANDWIDTH_LIMIT {trace_channel},OFF",
        "query": "BANDWIDTH_LIMIT?"
    },
    {
        "command": "TIME_DIV 1MS",
        "query": "TIME_DIV?"
    },
    {
        "method": "set_trig_delay",
        "kwargs": {"divs_left": -4.9},
        "query": "TRIG_DELAY?"
    },

    { "command": "TRIG_SELECT EDGE,SR,{trigger_line},HT,OFF", "query": "TRIG_SELECT?" },
    { "query": "VBS? 'Return=app.Utility.Remote.Interface'" },
    { "command": "VBS 'app.Acquisition.Horizontal.SmartMemory = \"SetMaximumMemory\"'" },
    { "query": "VBS? 'Return=app.LogicAnalyzer.Digital1.LineNames'" },
    { "command": "WAVEFORM_SETUP SP,1,NP,500000000,FP,0,SN,0", "query": "WAVEFORM_SETUP?" }
]

with LeCroy(
        samples=5_000,
        offset=0,
        ip_address="192.168.0.1",  # Change to your device.
        trace_channel="C1",
        #trigger_channel="C1",
        trigger_channel="DIGITAL1",
        trigger_line="D1",
        communication_timeout=1.0,  # [s]
        trigger_timeout=1.0,  # [s]
        scope_setup_commands=scope_setup_commands,
        communication_class_name="LeCroyCommunicationSocket",
) as oscilloscope:
    trace = cw.capture_trace(oscilloscope.scope, target, pt, key)
```

### Low-level Controls

We provide wrappers over automation API.

-   First read the getting started
    [https://cdn.teledynelecroy.com/files/manuals/wavepro-hd-gsg-eng.pdf](https://cdn.teledynelecroy.com/files/manuals/wavepro-hd-gsg-eng.pdf).
-   The manual
    [https://www.teledynelecroy.com/doc/docview.aspx?id=14960](https://www.teledynelecroy.com/doc/docview.aspx?id=14960).
-   For detailed documentation of the API see the official documentation
    [https://cdn.teledynelecroy.com/files/manuals/maui-remote-control-and-automation-manual.pdf](https://cdn.teledynelecroy.com/files/manuals/maui-remote-control-and-automation-manual.pdf).
-   And automation commmand reference manual (using VBS API)
    [https://cdn.teledynelecroy.com/files/manuals/automation_command_ref_manual_wr.pdf](https://cdn.teledynelecroy.com/files/manuals/automation_command_ref_manual_wr.pdf).

### Additional Features of the LeCroy API

#### Print Screen

When controlling an oscilloscope remotely it can be very convenient to just
take a screenshot (control what is captured by setting `capture_area` to one of
`"FULLSCREEN"`, `"DSOWINDOW"`, `"GRIDAREAONLY"`):

```python
oscilloscope.scope.print_screen(
    file_path="screenshot.png",
    capture_area="FULLSCREEN",
)
```

Or more generally one can call `LeCroyScope.retrieve_file` or
`LeCroyScope.delete_file` which are used to implement `print_screen` but can be
useful in other contexts.
