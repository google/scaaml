"""This code is modified from the ChipWhisperer project:
http://www.github.com/newaetech/chipwhisperer"""

from __future__ import absolute_import

import ctypes
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_DOWN
import time
import traceback
from typing import Any, Dict, Optional, OrderedDict, Union

from chipwhisperer.common.utils import util
import numpy as np
from picosdk.constants import make_enum
from picosdk.ps6000a import ps6000a as ps
from picosdk.PicoDeviceEnums import picoEnum
from picosdk.PicoDeviceStructs import picoStruct
from picosdk.functions import adc2mV, assert_pico_ok
from picosdk.errors import PicoSDKCtypesError

from scaaml.capture.scope.scope_template import ScopeTemplate, ScopeTraceType, ScopeTriggerTraceType


@dataclass
class ChannelRange:
    """API values for channel range."""
    range_v: float
    api_value: int
    range_str: str


def assert_ok(status: int) -> None:  # pragma: no cover
    """Check assert_pico_ok and if it raises change PicoSDKCtypesError to
    IOError."""
    try:
        assert_pico_ok(status)
    except PicoSDKCtypesError as error:
        raise IOError from error


# Workaround until PR #43 in
# https://github.com/picotech/picosdk-python-wrappers/ is merged
PICO_PORT_DIGITAL_CHANNEL = make_enum([  # pylint: disable=invalid-name
    "PICO_PORT_DIGITAL_CHANNEL0",
    "PICO_PORT_DIGITAL_CHANNEL1",
    "PICO_PORT_DIGITAL_CHANNEL2",
    "PICO_PORT_DIGITAL_CHANNEL3",
    "PICO_PORT_DIGITAL_CHANNEL4",
    "PICO_PORT_DIGITAL_CHANNEL5",
    "PICO_PORT_DIGITAL_CHANNEL6",
    "PICO_PORT_DIGITAL_CHANNEL7",
])


class CaptureSettings:
    """Channel settings."""
    _name = "Capture Setting"

    CHANNEL_COUPLINGS: Dict[str, int] = {
        "DC50": picoEnum.PICO_COUPLING["PICO_DC_50OHM"],
        "DC": picoEnum.PICO_COUPLING["PICO_DC"],
        "AC": picoEnum.PICO_COUPLING["PICO_AC"],
    }
    CHANNELS: Dict[str, int] = {
        "A": picoEnum.PICO_CHANNEL["PICO_CHANNEL_A"],
        "B": picoEnum.PICO_CHANNEL["PICO_CHANNEL_B"],
        "C": picoEnum.PICO_CHANNEL["PICO_CHANNEL_C"],
        "D": picoEnum.PICO_CHANNEL["PICO_CHANNEL_D"],
        "PORT0": picoEnum.PICO_CHANNEL["PICO_PORT0"],
        "PORT1": picoEnum.PICO_CHANNEL["PICO_PORT1"],
        "External": 4,
        "TriggerAux": 5
    }
    CHANNEL_RANGE: list[ChannelRange] = [
        ChannelRange(
            range_v=20E-3,
            api_value=1,
            range_str="20 mV",
        ),
        ChannelRange(
            range_v=50E-3,
            api_value=2,
            range_str="50 mV",
        ),
        ChannelRange(
            range_v=100E-3,
            api_value=3,
            range_str="100 mV",
        ),
        ChannelRange(
            range_v=200E-3,
            api_value=4,
            range_str="200 mV",
        ),
        ChannelRange(
            range_v=500E-3,
            api_value=5,
            range_str="500 mV",
        ),
        ChannelRange(
            range_v=1.0,
            api_value=6,
            range_str="1 V",
        ),
        ChannelRange(
            range_v=2.0,
            api_value=7,
            range_str="2 V",
        ),
        ChannelRange(
            range_v=5.0,
            api_value=8,
            range_str="5 V",
        ),
        ChannelRange(
            range_v=10.0,
            api_value=9,
            range_str="10 V",
        ),
        ChannelRange(
            range_v=20.0,
            api_value=10,
            range_str="20 V",
        ),
    ]

    ATTENUATION = {
        "1:1": 1,
        "1:10": 10,
    }
    REV_ATTENUATION = {1: "1:1", 10: "1:10"}

    def __init__(self) -> None:
        self._couplings: Dict[str, int] = {}
        self._rev_couplings: Dict[int, str] = {}
        for name, val in self.CHANNEL_COUPLINGS.items():
            self._couplings[name] = val
            self._rev_couplings[val] = name
        # channels
        self._ch_list: Dict[str, int] = {}
        self._rev_ch_list: Dict[int, str] = {}
        for channel_name, channel_id in self.CHANNELS.items():
            self._ch_list[channel_name] = channel_id
            self._rev_ch_list[channel_id] = channel_name
        # ranges
        self._ch_range: Dict[float, str] = {}
        self._ch_range_list: list[float] = []
        self._ch_range_api_value: Dict[float, int] = {}
        for key in self.CHANNEL_RANGE:
            self._ch_range[key.range_v] = key.range_str
            self._ch_range_list.append(key.range_v)

            self._ch_range_api_value[key.range_v] = key.api_value

        self._ch_range_list.sort()

        self._channel = 0
        self._port_pin: Optional[int] = None
        self._probe_attenuation = 1
        self._coupling = self._couplings["AC"]
        self._range: float = 5.0
        self.bw_limit: str = "PICO_BW_FULL"

    @property
    def ps_api_channel(self) -> int:
        """Channel for PicoScope API."""
        return self._channel

    @property
    def channel(self) -> str:
        return self._rev_ch_list[self._channel]

    @channel.setter
    def channel(self, val: str) -> None:
        if val not in self._ch_list:
            raise ValueError(f"Unknown channel {val} not in {self._ch_list}")
        self._channel = self._ch_list[val]

        if self.is_digital:
            if self._port_pin is None:
                # Provide a sensible default if there is no value.
                self._port_pin = 0
        else:
            # Analog channels do not have a pin.
            self._port_pin = None

    @property
    def port_pin(self) -> Optional[int]:
        """Number of pin of the digital port."""
        return self._port_pin

    @port_pin.setter
    def port_pin(self, val: Optional[int]) -> None:
        # If the channel is digital port is not None.
        if self.is_digital and val is None:
            raise ValueError(f"Using digital port {self.channel} cannot set "
                             f"the pin to None.")

        # If the channel is analog port is None.
        if not self.is_digital and val is not None:
            raise ValueError(f"Using analog channel {self.channel} pin must "
                             f"be None, not {val}.")

        self._port_pin = val

    @property
    def probe_attenuation(self) -> str:
        return self.REV_ATTENUATION[self._probe_attenuation]

    @probe_attenuation.setter
    def probe_attenuation(self, val: str) -> None:
        if val not in self.ATTENUATION:
            raise ValueError(f"Unsupported value {val} not in "
                             f"{self.ATTENUATION}")
        self._probe_attenuation = self.ATTENUATION[val]

    @property
    def ps_api_coupling(self) -> int:
        return self._coupling

    @property
    def coupling(self) -> str:
        return self._rev_couplings[self._coupling]

    @coupling.setter
    def coupling(self, val: str) -> None:
        if val not in self._couplings:
            raise ValueError("Unsupported value")
        self._coupling = self._couplings[val]

    @property
    def ps_api_range(self) -> float:
        """Range value for PicoScope API."""
        return self._ch_range_api_value[self._range]

    @property
    def range(self) -> str:
        """Human readable range voltage string."""
        return self._ch_range[self._range]

    @range.setter
    def range(self, val: float) -> None:
        if not isinstance(val, float):
            raise ValueError("Unsupported value (should be float)")

        # Find the smallest supported range that is higher than val
        for r in self._ch_range_list:
            if val <= r:
                self._range = r
                return
        raise ValueError(f"Unsupported value (too large), got {val}, maximum "
                         f"is {self._ch_range_list[-1]}")

    def _dict_repr(self) -> Dict[str, Any]:
        """Human readable representation as a key value dictionary."""
        ret: OrderedDict[str, Any] = OrderedDict()
        ret["channel"] = self.channel
        ret["range"] = self.range
        ret["probe_attenuation"] = self.probe_attenuation
        ret["coupling"] = self.coupling
        ret["port_pin"] = self._port_pin
        ret["bandwidth_limit"] = self.bw_limit
        return ret

    def dict_repr(self) -> Dict[str, Any]:
        """Public dictionary representation."""
        return self._dict_repr()

    def __repr__(self) -> str:
        val = util.dict_to_str(
            self._dict_repr())  # type: ignore[no-untyped-call]
        return str(val)

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def is_digital(self) -> bool:
        """An MSO Pod can be connected as trigger, but for trace it does not
        make sense.
        """
        return False


class TriggerSettings(CaptureSettings):
    """Trigger channel settings."""
    _name = "Trigger Setting"

    THRESHOLD_DIRECTION: Dict[str, int] = {
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

    def __init__(self) -> None:
        super().__init__()
        self._trig_dir: Dict[str, int] = {}
        self._rev_trig_dir: Dict[int, str] = {}
        for name, val in self.THRESHOLD_DIRECTION.items():
            self._trig_dir[name] = val
            self._rev_trig_dir[val] = name

        self._channel: int = picoEnum.PICO_CHANNEL["PICO_PORT0"]
        self._port_pin: Optional[int] = 0
        self._range: float = 5.0
        self._coupling: int = self._couplings["DC"]
        self._trigger_direction: int = self._trig_dir["Rising"]
        self._trigger_level: float = 2.0  # V
        # PICO_VERY_HIGH_400MV
        # PICO_HIGH_200MV
        # PICO_NORMAL_100MV
        # PICO_LOW_50MV
        self.hysteresis: Optional[str] = "PICO_NORMAL_100MV"

    @property
    def ps_api_trigger_direction(self) -> int:
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
    def trigger_direction(self) -> str:
        return self._rev_trig_dir[self._trigger_direction]

    @trigger_direction.setter
    def trigger_direction(self, val: str) -> None:
        if val not in self._trig_dir:
            raise ValueError("Unsupported value")
        self._trigger_direction = self._trig_dir[val]

    def _dict_repr(self) -> Dict[str, Any]:
        """Human readable representation as a key value dictionary."""
        config = super()._dict_repr()
        config.update({
            "trigger_level": self.trigger_level,
            "trigger_direction": self.trigger_direction,
            "hysteresis": self.hysteresis,
        })

        # Remove specific for analog / digital trigger.
        if self.is_digital:
            # Remove safely when keys not present
            config.pop("trigger_range", None)
            config.pop("trigger_coupling", None)

        return config

    @property
    def is_digital(self) -> bool:
        """Return if this channel is a digital channel (PORT0 or PORT1)."""
        return self.ps_api_channel in [
            picoEnum.PICO_CHANNEL["PICO_PORT0"],
            picoEnum.PICO_CHANNEL["PICO_PORT1"],
        ]


class Pico6424E(ScopeTemplate):
    """Class that interacts with the Picoscope 6424E oscilloscope."""
    _name = "Picoscope 6424E series 6000a (picosdk)"
    _NUM_CHANNELS: int = 4  # Number of analog channels
    #  Resolutions 8bit and 10bit work, but 12bit does not seem to be working
    #  (PICO_CHANNEL_COMBINATION_NOT_VALID_IN_THIS_RESOLUTION)
    _RESOLUTION: int = picoEnum.PICO_DEVICE_RESOLUTION["PICO_DR_10BIT"]

    DOWNSAMPLING_RATIO: int = 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args  # unused
        del kwargs  # unused
        self.ps_handle: ctypes.c_int16 = ctypes.c_int16()

        self.trace = CaptureSettings()
        self.trigger = TriggerSettings()
        self._sample_length: int = 500
        self._sample_offset: int = 0

        # Sample rate settings.
        self._sample_rate: float = 1E6
        # Set timebase (seconds per sample)
        self._timebase = Pico6424E._get_timebase(self._sample_rate)

        # Trace and trigger buffer, _buffers[0] is the trace buffer,
        # _buffers[1] is the trigger buffer.
        self._buffer_trace: list[float] = []
        self._buffer_trigger: Union[list[float], list[int]] = []

        # Part of cw API
        self.connectStatus: bool = False  # Connected status for cw  # pylint: disable=C0103
        self._max_adc: ctypes.c_int16 = ctypes.c_int16()  # To get mV values

        # Ignore signal overflow during capture.
        self.ignore_overflow: bool = False

        # If we use hardware downsampling, use averaging.
        # "PICO_RATIO_MODE_DECIMATE" could also be an option.
        self._downsampling_mode: int = picoEnum.PICO_RATIO_MODE[
            "PICO_RATIO_MODE_RAW"]
        if self.DOWNSAMPLING_RATIO > 1:
            self._downsampling_mode = picoEnum.PICO_RATIO_MODE[
                "PICO_RATIO_MODE_AVERAGE"]

    @staticmethod
    def _get_timebase(sample_rate: float) -> ctypes.c_uint32:
        """Return timebase for PicoScope API.

        Args:
          sample_rate (float): Samples per second (in Hz).

        Returns: Timebase (seconds per sample) represented as ctypes.c_uint32
          value for use in ps6000aRunBlock.
        """
        # Handle too large sample_rate
        if sample_rate > 5e9:
            raise ValueError("This scope supports at most 5GHz sample_rate.")

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
            return ctypes.c_uint32(int(timebase))

        # timebase should be <= 4
        smallest_timebase = Decimal("0.2e-9")  # 200ps
        for i in range(4, -1, -1):
            if s_per_sample >= (2**i) * smallest_timebase:
                return ctypes.c_uint32(i)
        raise ValueError("Couldn't find a supported timebase")

    def con(self, sn: Optional[str] = None) -> bool:  # pragma: no cover
        del sn  # unused
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
            assert_ok(
                ps.ps6000aGetAdcLimits(
                    self.ps_handle,  # handle
                    self._RESOLUTION,  # resolution
                    ctypes.byref(ctypes.c_int16()),  # minADC
                    ctypes.byref(self._max_adc),  # maxADC
                ))

            # Set channels and trigger.
            self._set_channels()
            self.connectStatus = True
            return True
        except Exception:  # pylint: disable=W0703
            # Whatever happened call disconnect.

            # Print stack traceback.
            traceback.print_exc()

            # Disconnect the scope if the exception was raised during setting
            # channels.
            self.dis()
            return False

    def dis(self) -> bool:  # pragma: no cover
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

    def arm(self) -> None:  # pragma: no cover
        """Prepare the scope for capturing."""
        # Check if this scope is connected.
        if self.connectStatus is False:
            raise ConnectionError(
                f"Scope {self._name} is not connected. Connect it first.")

        # Run the capture block
        assert_ok(
            ps.ps6000aRunBlock(
                self.ps_handle,  # handle
                self.DOWNSAMPLING_RATIO *
                self._sample_offset,  # Pre-trigger samples
                self.DOWNSAMPLING_RATIO *
                self._sample_length,  # Post-trigger samples
                self._timebase,  # timebase
                ctypes.byref(ctypes.c_double(0)),  # timeIndisposedMs
                0,  # segmentIndex
                None,  # lpReady callback
                None,  # pParameter
            ))

    def capture(self, poll_done: bool = False) -> bool:  # pragma: no cover
        """Capture one trace and return True if timeout has happened
        (possible capture failure).

        Args:
          poll_done: Not supported in PicoScope, but a part of API.

        Raises: IOError if unknown failure.

        Returns: True if the trace needs to be recaptured due to timeout or
          trace overflow (if ignore_overflow is set to False). False otherwise.
        """
        del poll_done  # unused

        # Wait until the result is ready
        ready: ctypes.c_int16 = ctypes.c_int16(0)
        check: ctypes.c_int16 = ctypes.c_int16(0)
        start_waiting = time.time()  # s from start of the epoch
        while ready.value == check.value:
            ps.ps6000aIsReady(self.ps_handle, ctypes.byref(ready))
            # Check for timeout
            time_now = time.time()
            if time_now - start_waiting > 100:  # [s]
                # Stop current capture
                assert_ok(ps.ps6000aStop(self.ps_handle))
                # Indicate timeout
                return True

        # Retrieve the values
        overflow: ctypes.c_int16 = ctypes.c_int16()
        max_samples: ctypes.c_int32 = ctypes.c_int32(self._total_samples)
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
        self._buffer_trace = adc2mV(
            self._trace_buffer,
            self.trace.ps_api_range,  # range
            self._max_adc)[:max_samples.value]
        if self.trigger.is_digital:
            self._buffer_trigger = self._trigger_buffer[:max_samples.value]
        else:
            self._buffer_trigger = adc2mV(
                self._trigger_buffer,
                self.trigger.ps_api_range,  # range
                self._max_adc)[:max_samples.value]

        # print("Overflow: {}".format(overflow.value))
        # print("Samples: {}".format(len(self._trace_buffer)))
        # print("Max samples: {}".format(max_samples.value))
        if self.ignore_overflow:
            return False
        else:
            return (overflow.value >> self.trace.ps_api_channel) & 1 == 1

    def get_last_trace(self, as_int: bool = False) -> ScopeTraceType:
        """Return a copy of the last trace.

        Args:
          as_int (bool): Not supported, part of CW API. Return integer
            representation of the trace. Defaults to False meaning return
            np.array of dtype np.float32.

        Returns: np array representing the last captured trace in mV unless
        `as_int` is set to `True` in which case it returns the ADC values
        directly.
        """
        if as_int:
            return np.array(self._trace_buffer[:], dtype=np.int32)

        return np.array(self._buffer_trace[:], dtype=np.float32)

    def get_last_trigger_trace(self) -> ScopeTriggerTraceType:
        """Return a copy of the last trigger trace.

        Returns: np array representing the trigger trace. If the trigger is
          digital returns a boolean array, else returns a float32 array.
        """
        # Digital trigger.
        if self.trigger.is_digital:
            bits = np.array(self._buffer_trigger[:], dtype=np.int16)
            port_pin = self.trigger.port_pin
            assert port_pin is not None
            return np.array((bits >> port_pin) & 1, dtype=np.bool_)

        # Analog trigger.
        return np.array(self._buffer_trigger[:], dtype=np.float32)

    def _set_channel_on(
            self, channel_info: CaptureSettings) -> None:  # pragma: no cover
        """Turn on a single channel.

        Args:
          channel_info (CaptureSettings): Either the trace or the trigger
            channel.
        """
        if channel_info.is_digital:
            assert isinstance(channel_info, TriggerSettings)
            # Turn on a digital port.
            # Logic level needs to be set individually for each digital
            # channel/pin in the port.
            pins = 8
            # A list of threshold voltages (one for each port pin), used to
            # distinguish 0 and 1 states. Range -32_767 (-5V) to 32_767 (5V).
            logic_threshold_level = (ctypes.c_int16 * pins)(0)
            for i in range(pins):
                logic_threshold_level[i] = int(
                    (32_767 / 5) * channel_info.trigger_level)
            logic_threshold_level_length = len(logic_threshold_level)
            # Hysteresis of the digital channel.
            ps_api_hysteresis = picoEnum.PICO_DIGITAL_PORT_HYSTERESIS[
                channel_info.hysteresis]
            assert_ok(
                ps.ps6000aSetDigitalPortOn(
                    self.ps_handle,  # handle
                    channel_info.ps_api_channel,  # port
                    ctypes.byref(logic_threshold_level),
                    logic_threshold_level_length,
                    ps_api_hysteresis,
                ))
        else:
            # Turn on an analog channel.
            assert_ok(
                ps.ps6000aSetChannelOn(
                    self.ps_handle,  # handle
                    channel_info.ps_api_channel,  # channel
                    channel_info.ps_api_coupling,  # coupling
                    channel_info.ps_api_range,  # range
                    0,  # analogue offset
                    picoEnum.PICO_BANDWIDTH_LIMITER[
                        channel_info.bw_limit],  # bandwidth
                ))

    def _set_digital_trigger(self) -> None:  # pragma: no cover
        """Set a trigger on a digital channel.
        """
        # Make sure we are using a digital trigger
        assert self.trigger.is_digital

        # Set Pico channel conditions
        conditions = (picoStruct.PICO_CONDITION * 1)()
        conditions[0] = picoStruct.PICO_CONDITION(
            self.trigger.ps_api_channel,  # PICO_CHANNEL source
            picoEnum.PICO_TRIGGER_STATE[
                "PICO_CONDITION_TRUE"],  # PICO_TRIGGER_STATE condition
        )
        assert_ok(
            ps.ps6000aSetTriggerChannelConditions(
                self.ps_handle,  # handle
                ctypes.byref(conditions),  # * conditions
                len(conditions),  # n_conditions
                3,  # action PICO_CLEAR_ALL = 1  PICO_ADD = 2
            ))

        # Set trigger digital port properties
        digital_channels: int = 8
        # PICO_DIGITAL_DONT_CARE:
        #    channel has no effect on trigger
        # PICO_DIGITAL_DIRECTION_LOW:
        #    channel must be low to trigger
        # PICO_DIGITAL_DIRECTION_HIGH:
        #    channel must be high to trigger
        # PICO_DIGITAL_DIRECTION_RISING:
        #    channel must transition from low to high to trigger
        # PICO_DIGITAL_DIRECTION_FALLING:
        #    channel must transition from high to low to trigger
        # PICO_DIGITAL_DIRECTION_RISING_OR_FALLING:
        #    any transition on channel causes a trigger
        directions = (picoStruct.DIGITAL_CHANNEL_DIRECTIONS *
                      digital_channels)()
        for i in range(digital_channels):
            # Ignore all pins except for the trigger pin.
            trigger_direction: str = "PICO_DIGITAL_DONT_CARE"
            if i == self.trigger.port_pin:
                trigger_direction = "PICO_DIGITAL_DIRECTION_RISING"

            directions[i] = picoStruct.DIGITAL_CHANNEL_DIRECTIONS(
                PICO_PORT_DIGITAL_CHANNEL[
                    f"PICO_PORT_DIGITAL_CHANNEL{i}"],  # channel
                picoEnum.PICO_DIGITAL_DIRECTION[trigger_direction],
            )

        assert_ok(
            ps.ps6000aSetTriggerDigitalPortProperties(
                self.ps_handle,  # handle
                self.trigger.ps_api_channel,  # port
                ctypes.byref(directions),  # directions
                len(directions),  # n_directions
            ))

    def _set_channels(self) -> None:  # pragma: no cover
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
        for port in [
                picoEnum.PICO_CHANNEL["PICO_PORT0"],
                picoEnum.PICO_CHANNEL["PICO_PORT1"],
        ]:
            assert_ok(
                ps.ps6000aSetDigitalPortOff(
                    self.ps_handle,  # handle
                    port,  # port
                ))

        # Turn on trace and trigger channels.
        self._set_channel_on(channel_info=self.trace)
        self._set_channel_on(channel_info=self.trigger)

        # Set a trigger.
        if self.trigger.is_digital:
            # Set a digital trigger.
            self._set_digital_trigger()
        else:
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
    def sample_rate(self) -> float:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, val: float) -> None:
        self._sample_rate = val
        self._timebase = Pico6424E._get_timebase(self._sample_rate)

    @property
    def sample_length(self) -> int:
        return self._sample_length

    @sample_length.setter
    def sample_length(self, val: int) -> None:
        self._sample_length = val
        self._set_channels()

    @property
    def sample_offset(self) -> int:
        return self._sample_offset

    @sample_offset.setter
    def sample_offset(self, val: int) -> None:
        self._sample_offset = val
        self._set_channels()

    def _dict_repr(self) -> Dict[str, Any]:
        """Human readable representation as a key value dictionary."""
        ret: OrderedDict[str, Any] = OrderedDict()
        ret["trace"] = self.trace.dict_repr()
        ret["trigger"] = self.trigger.dict_repr()
        ret["sample_rate"] = self.sample_rate
        ret["sample_length"] = self.sample_length
        ret["sample_offset"] = self.sample_offset
        ret["ignore_overflow"] = self.ignore_overflow
        return ret

    def __repr__(self) -> str:
        """Return device name, connected status and dict representation as
        multi-line string."""
        connected = "Connected" if self.connectStatus else "Not connected"
        dict_repr = util.dict_to_str(
            self._dict_repr())  # type: ignore[no-untyped-call]
        return f"{self._name} device {connected}\n{dict_repr}"

    def __str__(self) -> str:
        return self.__repr__()
