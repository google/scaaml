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
-   And automation command reference manual (using VBS API)
    [https://cdn.teledynelecroy.com/files/manuals/automation_command_ref_manual_wr.pdf](https://cdn.teledynelecroy.com/files/manuals/automation_command_ref_manual_wr.pdf).

### LeCroy Tips and Tricks

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

#### Testing Digital Trigger

We failed to find documentation for parsing `.digXML` or `XMLdig`. To ensure we
return what we should we connected both the trigger and trace channel to the
calibration output of the oscilloscope and tested several variations of
settings.

```python
"""Create tests for the LeCroy oscilloscope. Connect both the trace channel and
a digital line to the calibration output of your oscilloscope. Both trigger and
trace waveforms should be the same.
"""
import itertools
from pathlib import Path
import random
import time

import matplotlib.pyplot as plt
import numpy as np

from scaaml.capture.scope import LeCroy


def plot_trace(trace, trigger, trace_fig_name):
    plt.clf()
    plt.plot(trace)
    plt.plot(trigger * np.max(trace) * 0.8)
    plt.savefig(trace_fig_name)


def main():
    max_difference: int = 0

    scope_setup_commands = [
        { "command": "{trace_channel}:TRACE ON" },
        {
            "command": "MEMORY_SIZE 1e+9",
            "query": "MEMORY_SIZE?"
        },
        # Calibration output settings.
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
        # Digital trigger:
        { "command": "TRIG_SELECT EDGE,SR,{trigger_line},HT,OFF", "query": "TRIG_SELECT?" },
    ]

    # Define possible values for each of the parameters. This will take a while
    # to try.
    parameters = {
        "samples": [5_432],
        "offset": [3_141],
        "trig_delay": [2.3],
        "SP": [7_123],
        "time_div": ["0.5MS"],
    }
    print(f"Total {np.prod([len(v) for v in parameters.values()])} experiments")

    # Iterate dictionaries with all combinations of the parameters.
    named_parameters = [[(name, value) for value in values] for name, values in parameters.items()]
    experiments = list(map(dict, itertools.product(*named_parameters)))
    random.shuffle(experiments)

    # Do at most a couple experiments:
    experiments = experiments[:5]

    with LeCroy(
         samples=10_000_000,
         offset=0,
         ip_address="192.168.0.1",  # Change to your device.
         trace_channel="C3",
         trigger_channel="DIGITAL1",
         trigger_line="D2",
         communication_timeout=10.0,
         trigger_timeout=10.0,
         scope_setup_commands=scope_setup_commands,
         communication_class_name="LeCroyCommunicationSocket",
    ) as oscilloscope:
        #print(oscilloscope.scope._scope_communication.query(f"TEMPLATE?"))
        #print(oscilloscope.scope._scope_communication.query(f"*IDN?"))
        #return

        for values in experiments:
            print(f">>> {values = }")

            # Postpone failing after a screenshot
            fail = ""

            # Current setup
            # Since we are changing the offset and number of samples by a
            # command the oscilloscope object will log an error that these
            # values are of (but the traces are still parsed correctly).
            oscilloscope.scope._run_command({
                "command": f"WAVEFORM_SETUP SP,{values['SP']},NP,{values['samples']},FP,{values['offset']},SN,0",
                "query": "WAVEFORM_SETUP?",
            })
            time.sleep(1)
            oscilloscope.scope._run_command({
                "command": f"TIME_DIV {values['time_div']}",
                "query": "TIME_DIV?",
            })
            time.sleep(1)
            oscilloscope.scope._run_command({
                "method": "set_trig_delay",
                "kwargs": {"divs_left": values["trig_delay"]},
                "query": "TRIG_DELAY?",
            })
            time.sleep(1)

            # Capture a calibration wave
            oscilloscope._scope.arm()
            assert not oscilloscope._scope.capture()
            trace = oscilloscope.scope.get_last_trace()
            trigger = oscilloscope.scope.get_last_trigger_trace()

            # Plot what we got
            plot_trace(trace, trigger, "trace_lecroy.png")
            oscilloscope.print_screen(
                file_path="lecroy.png",
                capture_area="FULLSCREEN",
            )

            if len(trigger) != len(trace):
                fail = f"Different length {len(trigger) = :_} {len(trace) = :_}"
                print(fail)
                trigger = trigger[:len(trace)]

            # Make sure the trigger and trace are roughly the same.
            trace_high = trace > 0.5
            max_difference = max(
                int(np.sum(np.logical_xor(trace_high, trigger))),
                max_difference,
            )
            print(f"{max_difference = }")
            if difference > values['samples'] / 500:
                fail = f"Too different trigger {difference = :_}"
                print(fail)

            if len(trace) != values["samples"]:
                fail = f"Wrong trace length {len(trace) = :_} {values['samples'] = :_}"
                print(fail)

            if fail:
                raise ValueError(f"{values}: {fail = }")

        print("All tests passed!")
        print(f"{max_difference = }")


if __name__ == "__main__":
    main()
```

## TODO

A list of features currently not supported. Beware that these lists are
probably lacking a lot of features.

### PicoScope&reg; TODO

-   Just one model is supported.

### LeCroy TODO

-   Segmented capture is currently not supported.
-   Compression for digital trace data (they can be rather large but well
    compressible).
