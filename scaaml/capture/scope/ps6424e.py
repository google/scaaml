"""This code is modified from the ChipWhisperer project:
http://www.github.com/newaetech/chipwhisperer"""

from __future__ import absolute_import

import ctypes
from decimal import Decimal, ROUND_HALF_DOWN
import time
import traceback
from typing import OrderedDict

from chipwhisperer.capture.api.cwcommon import ChipWhispererCommonInterface
from chipwhisperer.common.utils import util
import numpy as np
from picosdk.ps6000a import ps6000a as ps
from picosdk.PicoDeviceEnums import picoEnum
from picosdk.functions import adc2mV, assert_pico_ok, mV2adc
from picosdk.errors import PicoSDKCtypesError


def assert_ok(status):
    """Check assert_pico_ok and if it raises change PicoSDKCtypesError to
    IOError."""
    try:
        assert_pico_ok(status)
    except PicoSDKCtypesError as e:
        raise IOError(e)


class CaptureSettings(object):
    _name = "Capture Setting"

    CHANNEL_COUPLINGS = {
        "DC50": picoEnum.PICO_COUPLING["PICO_DC_50OHM"],
        "DC": picoEnum.PICO_COUPLING["PICO_DC"],
        "AC": picoEnum.PICO_COUPLING["PICO_AC"],
    }
    CHANNELS = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "External": 4,
        "MaxChannels": 4,
        "TriggerAux": 5
    }
    CHANNEL_RANGE = [
        {
            "rangeV": 20E-3,
            "apivalue": 1,
            "rangeStr": "20 mV"
        },
        {
            "rangeV": 50E-3,
            "apivalue": 2,
            "rangeStr": "50 mV"
        },
        {
            "rangeV": 100E-3,
            "apivalue": 3,
            "rangeStr": "100 mV"
        },
        {
            "rangeV": 200E-3,
            "apivalue": 4,
            "rangeStr": "200 mV"
        },
        {
            "rangeV": 500E-3,
            "apivalue": 5,
            "rangeStr": "500 mV"
        },
        {
            "rangeV": 1.0,
            "apivalue": 6,
            "rangeStr": "1 V"
        },
        {
            "rangeV": 2.0,
            "apivalue": 7,
            "rangeStr": "2 V"
        },
        {
            "rangeV": 5.0,
            "apivalue": 8,
            "rangeStr": "5 V"
        },
        {
            "rangeV": 10.0,
            "apivalue": 9,
            "rangeStr": "10 V"
        },
        {
            "rangeV": 20.0,
            "apivalue": 10,
            "rangeStr": "20 V"
        },
    ]

    ATTENUATION = {
        "1:1": 1,
        "1:10": 10,
    }
    REV_ATTENUATION = {1: "1:1", 10: "1:10"}

    def __init__(self):
        self._couplings = {}
        self._rev_couplings = {}
        for name, val in self.CHANNEL_COUPLINGS.items():
            self._couplings[name] = val
            self._rev_couplings[val] = name
        # channels
        self._chList = {}
        self._rev_chList = {}
        for t in self.CHANNELS:
            if self.CHANNELS[t] < self.CHANNELS["MaxChannels"]:
                self._chList[t] = self.CHANNELS[t]
                self._rev_chList[self.CHANNELS[t]] = t
        # ranges
        self._chRange = {}
        self._chRangeList = []
        self._chRangeApiValue = {}
        for key in self.CHANNEL_RANGE:
            self._chRange[key["rangeV"]] = key["rangeStr"]
            self._chRangeList.append(key["rangeV"])
            self._chRangeApiValue[key["rangeV"]] = key["apivalue"]
        self._chRangeList.sort()

        self._channel = 0
        self._probe_attenuation = 1
        self._coupling = self._couplings["AC"]
        self._range: float = 5.0

    @property
    def ps_api_channel(self):
        """Channel for PicoScope API."""
        channel_enums = {
            0: picoEnum.PICO_CHANNEL["PICO_CHANNEL_A"],
            1: picoEnum.PICO_CHANNEL["PICO_CHANNEL_B"],
            2: picoEnum.PICO_CHANNEL["PICO_CHANNEL_C"],
            3: picoEnum.PICO_CHANNEL["PICO_CHANNEL_D"],
        }
        # Ensure that we did not forget any channel
        assert len(channel_enums) == self.CHANNELS["MaxChannels"]
        return channel_enums[self._channel]

    @property
    def channel(self):
        return self._rev_chList[self._channel]

    @channel.setter
    def channel(self, val):
        if val not in self._chList:
            raise ValueError("Unknown channel")
        self._channel = self._chList[val]

    @property
    def probe_attenuation(self):
        return self.REV_ATTENUATION[self._probe_attenuation]

    @probe_attenuation.setter
    def probe_attenuation(self, val):
        if val not in self.ATTENUATION:
            raise ValueError("Unsupported value")
        self._probe_attenuation = self.ATTENUATION[val]

    @property
    def coupling(self):
        return self._rev_couplings[self._coupling]

    @coupling.setter
    def coupling(self, val):
        if val not in self._couplings:
            raise ValueError("Unsupported value")
        self._coupling = self._couplings[val]

    @property
    def ps_api_range(self):
        """Range value for PicoScope API."""
        return self._chRangeApiValue[self._range]

    @property
    def range(self):
        """Human readable range voltage string."""
        return self._chRange[self._range]

    @range.setter
    def range(self, val):
        if not isinstance(val, float):
            raise ValueError("Unsupported value (should be float)")

        # Find the smallest supported range that is higher than val
        for r in self._chRangeList:
            if val <= r:
                self._range = r
                return
        raise ValueError(f"Unsupported value (too large), got {val}, maximum "
                         f"is {self._chRangeList[-1]}")

    def _dict_repr(self):
        ret = OrderedDict()
        ret["channel"] = self.channel
        ret["range"] = self.range
        ret["probe_attenuation"] = self.probe_attenuation
        ret["coupling"] = self.coupling
        return ret

    def __repr__(self):
        return util.dict_to_str(self._dict_repr())

    def __str__(self):
        return self.__repr__()


class TriggerSettings(CaptureSettings):
    _name = "Trigger Setting"

    THRESHOLD_DIRECTION = {
        "Above":
            picoEnum.PICO_THRESHOLD_DIRECTION["PICO_ABOVE"],
        "Below":
            picoEnum.PICO_THRESHOLD_DIRECTION["PICO_BELOW"],
        "Rising":
            picoEnum.PICO_THRESHOLD_DIRECTION["PICO_RISING"],
        "Falling":
            picoEnum.PICO_THRESHOLD_DIRECTION["PICO_FALLING"],
        "RiseOrFall":
            picoEnum.PICO_THRESHOLD_DIRECTION["PICO_RISING_OR_FALLING"],
    }

    def __init__(self):
        CaptureSettings.__init__(self)
        self._trigDir = {}
        self._rev_trigDir = {}
        for name, val in self.THRESHOLD_DIRECTION.items():
            self._trigDir[name] = val
            self._rev_trigDir[val] = name

        self._channel = 1
        self._range = 5.0
        self._coupling = self._couplings["DC"]
        self._trigger_direction = self._trigDir["Rising"]
        self._trigger_level = 2  # V

    @property
    def ps_api_trigger_direction(self):
        """Trigger direction compatible with PicoScope API."""
        return self._trigger_direction

    @property
    def ps_api_trigger_level(self) -> int:
        """Trigger level compatible with PicoScope simple trigger API. Returns
        trigger level in mV.
        """
        # From V to mV and convert to integer
        return int(1_000 * self._trigger_level)

    @property
    def trigger_level(self) -> float:
        """Return trigger level value in V."""
        return self._trigger_level

    @trigger_level.setter
    def trigger_level(self, val: float) -> None:
        """Set trigger level in V.

        Args:
          val (float): The level in V at which to trigger.
        """
        self._trigger_level = val

    @property
    def trigger_direction(self):
        return self._rev_trigDir[self._trigger_direction]

    @trigger_direction.setter
    def trigger_direction(self, val):
        if val not in self._trigDir:
            raise ValueError("Unupported value")
        self._trigger_direction = self._trigDir[val]

    def _dict_repr(self):
        ret = OrderedDict()
        ret["channel"] = self.channel
        ret["range"] = self.range
        ret["probe_attenuation"] = self.probe_attenuation
        ret["coupling"] = self.coupling
        ret["trigger_level"] = self.trigger_level
        ret["trigger_direction"] = self.trigger_direction
        return ret


class Pico6424E(ChipWhispererCommonInterface):
    _name = "Picoscope 6424E series 6000a (picosdk)"
    _NUM_CHANNELS = 4  # Number of analog channels
    #  Resolutions 8bit and 10bit work, but 12bit does not seem to be working
    #  (PICO_CHANNEL_COMBINATION_NOT_VALID_IN_THIS_RESOLUTION)
    _RESOLUTION = picoEnum.PICO_DEVICE_RESOLUTION["PICO_DR_10BIT"]

    DOWNSAMPLING_RATIO = 1

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.ps_handle = ctypes.c_int16()

        self.trace = CaptureSettings()
        self.trigger = TriggerSettings()
        self._sample_length = 500
        self._sample_offset = 0

        # Sample rate settings.
        self._sample_rate = 1E6
        # Set timebase (seconds per sample)
        self._timebase = Pico6424E._get_timebase(self._sample_rate)

        # Trace and trigger buffer, _buffers[0] is the trace buffer,
        # _buffers[1] is the trigger buffer.
        self._buffers = [[], []]

        self.connectStatus = False  # Connected status for cw
        self._maxADC = ctypes.c_int16()  # To get mV values

    @staticmethod
    def _get_timebase(sample_rate: float):
        """Return timebase for PicoScope API.

        Args:
          sample_rate (float): Samples per second (in Hz).

        Returns: Timebase (seconds per sample) representated as ctypes.c_uint32
          value for use in ps6000aRunBlock.
        """
        # Handle too large sample_rate
        if sample_rate > 5e9:
            raise ValueError("This scope support at most 5GHz sample_rate.")

        # From PicoScope API manual:
        # https://www.picotech.com/download/manuals/picoscope-6000-series-a-api-programmers-guide.pdf
        # n<5       2**timebase / 5_000_000_000
        # n>4       (timebase - 4) / 156_250_000
        # timebase  time
        # 0         200ps
        # 1         400ps
        # 2         800ps
        # 3         1.6ns
        # 4         3.2ns
        # 5         6.4ns
        # ...
        # 2**32-1   6.87s
        s_per_sample = 1 / Decimal(sample_rate)  # avoid floating point errors

        if s_per_sample >= Decimal("6.4e-9"):
            # Compute for the large timebase
            timebase = (156_250_000 * s_per_sample) + 4
            # Round carefully
            timebase = timebase.to_integral_exact(rounding=ROUND_HALF_DOWN)
            timebase = int(timebase)
            return ctypes.c_uint32(timebase)

        # timebase should be <= 4
        smallest_timebase = Decimal("0.2e-9")  # 200ps
        for i in range(4, -1, -1):
            if s_per_sample >= (2**i) * smallest_timebase:
                return ctypes.c_uint32(i)

    def con(self, sn=None):
        try:
            # Open the scope and get the corresponding handle self.ps_handle.
            # resolution 8, 10, 12 bit
            assert_ok(
                ps.ps6000aOpenUnit(
                    ctypes.byref(self.ps_handle),  # handle
                    None,  # serial, open the first scope found
                    self._RESOLUTION,  # resolution
                ))
            # ps6000aOpenUnit could return an indication of a needed firmware
            # update, but picosdk.constants.PICO_STATUS raises KeyError on
            # PICO_FIRMWARE_UPDATE_REQUIRED_TO_USE_DEVICE_WITH_THIS_DRIVER.

            # Get analog to digital converter limits.
            minADC = ctypes.c_int16()
            assert_ok(
                ps.ps6000aGetAdcLimits(
                    self.ps_handle,  # handle
                    self._RESOLUTION,  # resolution
                    ctypes.byref(minADC),  # minADC
                    ctypes.byref(self._maxADC),  # maxADC
                ))

            # Set channels and trigger.
            self._set_channels()
            self.connectStatus = True
            return True
        except Exception as exc:
            # Print stack traceback.
            traceback.print_exc()

            # Disconnect the scope if the exception was raised during setting
            # channels.
            self.dis()
            return False

    def dis(self):
        if self.ps_handle.value > 0:
            # Check that the scope is connected
            assert self.connectStatus

            # Interrupt data capture
            assert_ok(ps.ps6000aStop(self.ps_handle))

            # Close the connection to the PicoScope.
            assert_ok(ps.ps6000aCloseUnit(self.ps_handle))
            # set the handle value to zero
            self.ps_handle.value = 0

        self.connectStatus = False
        # ScopeTemplate expects True to be returned.
        return True

    def arm(self):
        """Prepare the scope for capturing."""
        # Check if this scope is connected.
        if self.connectStatus is False:
            raise Exception(
                f"Scope {self._name} is not connected. Connect it first.")

        # Run the capture block
        timeIndisposedMs = ctypes.c_double(0)
        assert_ok(
            ps.ps6000aRunBlock(
                self.ps_handle,  # handle
                self.DOWNSAMPLING_RATIO *
                self._sample_offset,  # Pre-trigger samples
                self.DOWNSAMPLING_RATIO *
                self._sample_length,  # Post-trigger samples
                self._timebase,  # timebase
                ctypes.byref(timeIndisposedMs),
                0,  # segmentIndex
                None,  # lpReady callback
                None,  # pParameter
            ))

    def capture(self, poll_done: bool = False) -> bool:
        """Capture one trace and return True if timeout has happened
        (possible capture failure).

        Args:
          poll_done: Not supported in PicoScope, but a part of API.

        Raises: IOError if unknown failure.

        Returns: True if timeout happened, False otherwise.
        """
        # Wait until the result is ready
        ready = ctypes.c_int16(0)
        check = ctypes.c_int16(0)
        while ready.value == check.value:
            ps.ps6000aIsReady(self.ps_handle, ctypes.byref(ready))

        # Retrieve the values
        overflow = ctypes.c_int16()
        max_samples = ctypes.c_int32(self._total_samples)
        assert_ok(
            ps.ps6000aGetValues(
                self.ps_handle,  # handle
                0,  # Start index
                ctypes.byref(max_samples),  # number of processed samples
                self.DOWNSAMPLING_RATIO,  # Downsample ratio
                self._downsampling_mode,  # Downsampling mode
                0,  # Segment index
                ctypes.byref(overflow),  # overflow
            ))

        # Convert memory buffers into mV values and save them into
        # self._buffers[0] is the trace and self._buffers[1] is the trigger
        self._buffers[0] = adc2mV(
            self._trace_buffer,
            self.trace.ps_api_range,  # range
            self._maxADC)[:max_samples.value]
        self._buffers[1] = adc2mV(
            self._trigger_buffer,
            self.trigger.ps_api_range,  # range
            self._maxADC)[:max_samples.value]

        # print("Overflow: {}".format(overflow.value))
        # print("Samples: {}".format(len(self._trace_buffer)))
        # print("Max samples: {}".format(max_samples.value))
        return (overflow.value >> self.trace.ps_api_channel) & 1 == 1

    def get_last_trace(self, as_int: bool = False) -> np.ndarray:
        """Return a copy of the last trace.

        Args:
          as_int (bool): Not supported, part of CW API. Return integer
            representation of the trace. Defaults to False meaning return
            np.array of dtype np.float32.

        Returns: np array representing the last captured trace.
        """
        if as_int:
            msg = "Returning trace as integers is not implemented."
            raise NotImplemented(msg)

        return np.array(self._buffers[0][:], dtype=np.float32)

    getLastTrace = util.camel_case_deprecated(get_last_trace)

    def get_last_trigger_trace(self) -> np.ndarray:
        """Return a copy of the last trigger trace."""
        return np.array(self._buffers[1][:])

    def _set_channels(self):
        """Setup channels, buffers, and trigger."""
        if self.ps_handle.value <= 0:
            # No opened PicoScope handle
            msg = f"Scope handle is {self.ps_handle.value}"
            assert not self.connectStatus, msg
            return

        # Turn off all analog channels
        for c in range(self._NUM_CHANNELS):
            assert_ok(ps.ps6000aSetChannelOff(
                self.ps_handle,
                c,  # channel
            ))
        # Set MSO pods off
        assert_ok(
            ps.ps6000aSetDigitalPortOff(
                self.ps_handle,  # handle
                picoEnum.PICO_CHANNEL["PICO_PORT0"],  # port
            ))
        assert_ok(
            ps.ps6000aSetDigitalPortOff(
                self.ps_handle,  # handle
                picoEnum.PICO_CHANNEL["PICO_PORT1"],  # port
            ))

        # Trace channel configuration
        assert_ok(
            ps.ps6000aSetChannelOn(
                self.ps_handle,  # handle
                self.trace.ps_api_channel,  # channel
                self.trace._coupling,  # coupling
                self.trace.ps_api_range,  # range
                0,  # analogue offset
                picoEnum.PICO_BANDWIDTH_LIMITER["PICO_BW_FULL"],  # bandwidth
            ))

        # Trigger channel configuration
        assert_ok(
            ps.ps6000aSetChannelOn(
                self.ps_handle,  # handle
                self.trigger.ps_api_channel,  # channel
                self.trigger._coupling,  # coupling
                self.trigger.ps_api_range,  # range
                0,  # analogue offset
                picoEnum.PICO_BANDWIDTH_LIMITER["PICO_BW_FULL"],  # bandwidth
            ))

        # TODO(kralka): implement MSO pod trigger

        # Set simple trigger
        assert_ok(
            ps.ps6000aSetSimpleTrigger(
                self.ps_handle,  # handle
                1,  # enable=1 (for disable set to 0)
                self.trigger.ps_api_channel,  # channel
                self.trigger.ps_api_trigger_level,  # threshold in mV
                self.trigger.ps_api_trigger_direction,  # direction
                0,  # delay = 0 s
                100_000_000,  # autoTriggerMicroSeconds = 100s
            ))

        # Set data buffers
        self._total_samples = self.DOWNSAMPLING_RATIO * (self._sample_length +
                                                         self._sample_offset)
        self._trace_buffer = (ctypes.c_int16 * self._total_samples)()
        self._trigger_buffer = (ctypes.c_int16 * self._total_samples)()
        # If we use hardware downsampling, use averaging.
        if self.DOWNSAMPLING_RATIO > 1:
            self._downsampling_mode = picoEnum.PICO_RATIO_MODE[
                "PICO_RATIO_MODE_AVERAGE"]
            # "PICO_RATIO_MODE_DECIMATE" could also be an option.
        else:
            self._downsampling_mode = picoEnum.PICO_RATIO_MODE[
                "PICO_RATIO_MODE_RAW"]
        data_type = picoEnum.PICO_DATA_TYPE["PICO_INT16_T"]
        waveform = 0
        # Set trace buffer
        action = picoEnum.PICO_ACTION["PICO_CLEAR_ALL"] | picoEnum.PICO_ACTION[
            "PICO_ADD"]
        assert_ok(
            ps.ps6000aSetDataBuffer(
                self.ps_handle,  # handle
                self.trace.ps_api_channel,  # channel
                ctypes.byref(self._trace_buffer),  # buffer
                self._total_samples,  # samples
                data_type,  # data type
                waveform,  # waveform (the segment index)
                self._downsampling_mode,  # Downsampling mode
                action,  # buffer action
            ))
        # Set trigger buffer
        # Note that there is no PICO_CLEAR_ALL action, as that would remove
        # the trace buffer.
        assert_ok(
            ps.ps6000aSetDataBuffer(
                self.ps_handle,  # handle
                self.trigger.ps_api_channel,  # channel
                ctypes.byref(self._trigger_buffer),  # buffer
                self._total_samples,  # samples
                data_type,  # data type
                waveform,  # waveform (the segment index)
                self._downsampling_mode,  # Downsampling mode
                picoEnum.PICO_ACTION["PICO_ADD"],  # buffer action
            ))

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, val):
        self._sample_rate = val
        self._timebase = Pico6424E._get_timebase(self._sample_rate)

    @property
    def sample_length(self):
        return self._sample_length

    @sample_length.setter
    def sample_length(self, val):
        self._sample_length = val
        self._set_channels()

    @property
    def sample_offset(self):
        return self._sample_offset

    @sample_offset.setter
    def sample_offset(self, val):
        self._sample_offset = val
        self._set_channels()

    def _dict_repr(self):
        ret = OrderedDict()
        ret["trace"] = self.trace._dict_repr()
        ret["trigger"] = self.trigger._dict_repr()
        ret["sample_rate"] = self.sample_rate
        ret["sample_length"] = self.sample_length
        ret["sample_offset"] = self.sample_offset
        return ret

    def __repr__(self) -> str:
        """Return device name, connected status and dict representation as
        multi-line string."""
        connected = "Connected" if self.connectStatus else "Not connected"
        dict_repr = util.dict_to_str(self._dict_repr())
        return f"{self._name} device {connected}\n{dict_repr}"

    def __str__(self) -> str:
        return self.__repr__()
