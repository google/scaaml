# Copyright 2023-2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Parse data coming from the LeCroy oscilloscope.
"""

import dataclasses
import datetime
import struct
from typing import Any, Dict, Tuple, Union, cast

import numpy as np
import numpy.typing as npt

from scaaml.capture.scope.scope_template import ScopeTraceType


class LecroyWaveform:
    """Parse waveform response.
    """

    # This is a sha256 of the template definition string. Ideally one would
    # first ask for the template format using the command "TMPL?", parse it,
    # and then parse the waveform.
    # https://github.com/google/scaaml/issues/130
    #
    # Workaround check that hash of the template has not changed:
    #   hashlib.sha256(scope.query("TMPL?").encode("utf8")).hexdigest()
    SUPPORTED_PROTOCOL_TEMPLATE_SHA = "0ad362abcbe5b15ada8410f22f65434240f" \
                                      "1206c7ba4b88847d422b0ee55096b"

    def __init__(self, raw_data: bytes) -> None:
        """Initialize the parser. Use the `parse` method with the `get_wave1`.

        Args:
          raw_data (iterable): Iterable of bytes.
        """
        self.raw_data = bytearray(raw_data)
        self.d: Dict[str, Any] = {}
        self._order = "<"
        self._unit = "b"
        self._ofs = 0
        self._wave_description = WaveDesc()
        self.parse()

    def get_string(self, max_size: int = 16) -> str:
        """Parse a string from self.raw_data.
        """
        val = struct.unpack_from(f"{self._order:s}{max_size:d}s", self.raw_data,
                                 self._ofs)[0]
        self._ofs += max_size
        return cast(bytes, val).strip(b"\x00").decode("ascii")

    def get_int8(self) -> int:
        """Parse int8 from self.raw_data.
        """
        self._ofs += 1
        vals = struct.unpack_from("b", self.raw_data, self._ofs - 1)
        return cast(int, vals[0])

    def get_int16(self) -> int:
        """Parse int16 from self.raw_data.
        """
        self._ofs += 2
        vals = struct.unpack_from(f"{self._order:s}h", self.raw_data,
                                  self._ofs - 2)
        return cast(int, vals[0])

    def get_int32(self) -> int:
        """Parse int32 from self.raw_data.
        """
        self._ofs += 4
        vals = struct.unpack_from(f"{self._order:s}l", self.raw_data,
                                  self._ofs - 4)
        return cast(int, vals[0])

    def get_float(self) -> float:
        """Parse float from self.raw_data.
        """
        self._ofs += 4
        vals = struct.unpack_from(f"{self._order:s}f", self.raw_data,
                                  self._ofs - 4)
        return cast(float, vals[0])

    def get_double(self) -> float:
        """Parse double from self.raw_data.
        """
        self._ofs += 8
        vals = struct.unpack_from(f"{self._order:s}d", self.raw_data,
                                  self._ofs - 8)
        return cast(float, vals[0])

    def unpack_wave(self, amount: int) -> Tuple[int]:
        """Parse the whole waveform from self.raw_data. Increments the offset
        `self._ofs` by the number of parsed bytes.

        Args:
          amount (int): Number of units (depending on self._unit).

        Returns: A tuple of `amount` values.
        """
        fmt = f"{self._order}{amount}{self._unit}"
        size = struct.calcsize(fmt)
        val = struct.unpack_from(fmt, self.raw_data, self._ofs)
        self._ofs += size
        return cast(Tuple[int], val)

    def get_timestamp(self) -> datetime.datetime:
        """Parse timestamp from self.raw_data.
        """
        vals = struct.unpack_from(f"{self._order:s}d4b2h", self.raw_data,
                                  self._ofs)
        self._ofs += 16
        if all(int(v) == 0 for v in vals):
            return datetime.datetime.fromtimestamp(0)
        return datetime.datetime(
            vals[5],  # year
            vals[4],  # month
            vals[3],  # day
            vals[2],  # hours
            vals[1],  # minutes
            int(vals[0]),  # seconds
            int(1000 * (vals[0] - (int(vals[0])))),  # microseconds
        )

    def get_raw_wave1(self) -> npt.NDArray[np.uint16]:
        """Get raw (whole, all samples, no gain or offset compensation) wave.
        """
        wave: npt.NDArray[np.uint16] = self.d.get("wave1",
                                                  np.array([], dtype=np.uint16))
        return wave

    def get_wave1(self,
                  first_sample: int = 0,
                  length: int = -1) -> ScopeTraceType:
        """Get wave.
        """
        if length == 0:
            return np.array([], dtype=np.float32)
        wave: ScopeTraceType = np.array(self.get_raw_wave1(), dtype=np.float32)
        gain = self._wave_description.vertical_gain
        offset = self._wave_description.vertical_offset
        scaled_wave: ScopeTraceType = wave * gain - offset
        end = scaled_wave.size
        if length > 0:
            end = min(end, length + first_sample)
        return scaled_wave[first_sample:end]

    def __repr__(self) -> str:
        """Return human readable representation of self."""
        return {
            **self.d,
            **dataclasses.asdict(self._wave_description),
        }.__repr__()

    def parse(self) -> None:
        """Parse answer from LeCroy.
        """
        # BEGINNING OF WAVE DESCRIPTION BLOCK

        # Header
        header = self.get_string(16)
        assert header == "WAVEDESC"

        # Template name
        self._wave_description.template_name = self.get_string(16)

        # Communication type
        self._wave_description.comm_type = WaveDesc.translate(
            "comm_type", self.get_int16())
        self._unit = self._wave_description.comm_type

        # Byte order
        self._wave_description.comm_order = WaveDesc.translate(
            "comm_order", self.get_int16())
        self._order = self._wave_description.comm_order

        # Wave description length
        self._wave_description.wavedesc_len = self.get_int32()
        assert self._wave_description.wavedesc_len >= 346

        # User-text length
        self._wave_description.user_text_len = self.get_int32()

        _ = self.get_int32()  # reserved
        self._wave_description.trigtime_len = self.get_int32()
        self._wave_description.ristime_len = self.get_int32()
        _ = self.get_int32()  # reserved

        # Length of wave1
        self._wave_description.wave1_len = self.get_int32()

        # Length of wave2
        self._wave_description.wave2_len = self.get_int32()

        _ = self.get_int32()  # reserved
        _ = self.get_int32()  # reserved
        self._wave_description.instrument_name = self.get_string(16)
        self._wave_description.instrument_number = self.get_int32()
        self._wave_description.trace_label = self.get_string(16)
        _ = self.get_int16()  # reserved
        _ = self.get_int16()  # reserved
        self._wave_description.wave_array_count = self.get_int32()
        self._wave_description.points_per_screen = self.get_int32()
        self._wave_description.first_valid_pnt = self.get_int32()
        self._wave_description.last_valid_pnt = self.get_int32()
        self._wave_description.first_point = self.get_int32()
        self._wave_description.sparsing_factor = self.get_int32()
        self._wave_description.segment_index = self.get_int32()
        self._wave_description.subarray_count = self.get_int32()
        self._wave_description.sweeps_per_acq = self.get_int32()
        self._wave_description.points_per_pair = self.get_int16()
        self._wave_description.pair_offsets = self.get_int16()
        self._wave_description.vertical_gain = self.get_float()
        self._wave_description.vertical_offset = self.get_float()
        self._wave_description.max_value = self.get_float()
        self._wave_description.min_value = self.get_float()
        self._wave_description.nominal_bits = self.get_int16()
        self._wave_description.nom_subarray_count = self.get_int16()
        self._wave_description.horizontal_interval = self.get_float()
        self._wave_description.horizontal_offset = self.get_double()
        self._wave_description.pixel_offset = self.get_double()
        self._wave_description.vertical_unit = self.get_string(48)
        self._wave_description.horizontal_unit = self.get_string(48)
        self._wave_description.horizontal_uncertainty = self.get_float()
        self._wave_description.trigger_time = self.get_timestamp()
        self._wave_description.acq_duration = self.get_float()
        self._wave_description.record_type = WaveDesc.translate(
            "record_type", self.get_int16())
        self._wave_description.processing_done = WaveDesc.translate(
            "processing_done", self.get_int16())
        _ = self.get_int16()  # reserved
        self._wave_description.ris_sweeps = self.get_int16()

        # Timebase
        self._wave_description.timebase = WaveDesc.translate(
            "timebase", self.get_int16())

        # Vertical coupling
        self._wave_description.vertical_coupling = WaveDesc.translate(
            "vertical_coupling", self.get_int16())

        self._wave_description.probe_att = self.get_float()

        # Fixed vertical gain
        self._wave_description.fixed_vertical_gain = WaveDesc.translate(
            "fixed_vertical_gain", self.get_int16())

        # Bandwidth limit on/off
        self._wave_description.bandwidth_limit = WaveDesc.translate(
            "bandwidth_limit", self.get_int16())

        self._wave_description.vertical_vernier = self.get_float()
        self._wave_description.acq_vertical_offset = self.get_float()

        # Wave source
        self._wave_description.wave_source = WaveDesc.translate(
            "wave_source", self.get_int16())

        # END OF WAVE DESCRIPTION BLOCK

        self._ofs = self._wave_description.wavedesc_len

        # Block length
        block_len = self._wave_description.user_text_len
        if block_len > 0:
            assert block_len <= 160
            self.d["user_text"] = self.get_string(block_len)

        # Unused
        block_len = self._wave_description.trigtime_len
        if block_len > 0:
            # trigtime array
            self.d["trigtime"] = []
            self._ofs += block_len

        # Unused
        block_len = self._wave_description.ristime_len
        if block_len > 0:
            # random-interleaved-sampling array
            self.d["ristime"] = []
            self._ofs += block_len

        # Receive wave1
        block_len = self._wave_description.wave1_len
        if block_len > 0:
            self.d["wave1"] = np.array(
                self.unpack_wave(self._wave_description.wave_array_count))

        # Receive wave2
        block_len = self._wave_description.wave2_len
        if block_len > 0:
            self.d["wave2"] = np.array(
                self.unpack_wave(self._wave_description.wave_array_count))
        assert self._ofs == len(self.raw_data)


@dataclasses.dataclass
class WaveDesc:
    """Wave description fields.
    """
    template_name: str = ""
    comm_type: str = "b"  # Default in LecroyWaveform
    comm_order: str = "<"  # Default in LecroyWaveform
    wavedesc_len: int = 0
    user_text_len: int = 0
    trigtime_len: int = 0
    ristime_len: int = 0
    wave1_len: int = 0
    wave2_len: int = 0
    instrument_name: str = ""
    instrument_number: int = 0
    trace_label: str = ""
    wave_array_count: int = 0
    points_per_screen: int = 0
    first_valid_pnt: int = 0
    last_valid_pnt: int = 0
    first_point: int = 0
    sparsing_factor: int = 0
    segment_index: int = 0
    subarray_count: int = 0
    sweeps_per_acq: int = 0
    points_per_pair: int = 0
    pair_offsets: int = 0
    vertical_gain: float = 1.0  # Default value
    vertical_offset: float = 0.0  # Default value
    max_value: float = 0.0
    min_value: float = 0.0
    nominal_bits: int = 0
    nom_subarray_count: int = 0
    horizontal_interval: float = 0.0
    horizontal_offset: float = 0.0
    pixel_offset: float = 0.0
    vertical_unit: str = ""
    horizontal_unit: str = ""
    horizontal_uncertainty: float = 0.0
    trigger_time: datetime.datetime = datetime.datetime.fromtimestamp(0)
    acq_duration: float = 0.0
    record_type: str = ""
    processing_done: str = ""
    ris_sweeps: int = 0
    timebase: str = ""
    vertical_coupling: str = ""
    probe_att: float = 0.0
    fixed_vertical_gain: str = ""
    bandwidth_limit: str = ""
    vertical_vernier: float = 0.0
    acq_vertical_offset: float = 0.0
    wave_source: str = ""

    @staticmethod
    def translate(attr_name: str, value: Union[int, str]) -> str:
        """Helper method for setters. Enums have defined values which are a set
        of integers -- keys of the `translation` dictionary. Human readable
        value is a string -- values of the `translation` dictionary. This
        function takes an integer and returns its translation to string. If
        provided with a string it returns that.

        Raises: ValueError if `value` is a string which is not in
          `translation.keys()` or if `value` is an integer which is not in
          `translation.keys()` or neither of those types.
        """
        translation = {
            "comm_type": {
                0: "b",
                1: "h",
            },
            "comm_order": {
                0: ">",
                1: "<",
            },
            "record_type": {
                0: "single_sweep",
                1: "interleaved",
                2: "histogram",
                3: "graph",
                4: "filter_coefficient",
                5: "complex",
                6: "extrema",
                7: "sequence_obsolete",
                8: "centered_RIS",
                9: "peak_detect",
            },
            "processing_done": {
                0: "no_processing",
                1: "fir_filter",
                2: "interpolated",
                3: "sparsed",
                4: "autoscaled",
                5: "no_result",
                6: "rolling",
                7: "cumulative",
            },
            "timebase": {  # units are /div
                0: "1 ps",
                1: "2 ps",
                2: "5 ps",
                3: "10 ps",
                4: "20 ps",
                5: "50 ps",
                6: "100 ps",
                7: "200 ps",
                8: "500 ps",
                9: "1 ns",
                10: "2 ns",
                11: "5 ns",
                12: "10 ns",
                13: "20 ns",
                14: "50 ns",
                15: "100 ns",
                16: "200 ns",
                17: "500 ns",
                18: "1 us",
                19: "2 us",
                20: "5 us",
                21: "10 us",
                22: "20 us",
                23: "50 us",
                24: "100 us",
                25: "200 us",
                26: "500 us",
                27: "1 ms",
                28: "2 ms",
                29: "5 ms",
                30: "10 ms",
                31: "20 ms",
                32: "50 ms",
                33: "100 ms",
                34: "200 ms",
                35: "500 ms",
                36: "1 s",
                37: "2 s",
                38: "5 s",
                39: "10 s",
                40: "20 s",
                41: "50 s",
                42: "100 s",
                43: "200 s",
                44: "500 s",
                45: "1 ks",
                46: "2 ks",
                47: "5 ks",
                100: "external",
            },
            "vertical_coupling": {  # Translated to scope commands
                0: "D50",  # "DC_50_Ohms",
                1: "GND",  # "ground",
                2: "D1M",  # "DC_1MOhm",
                3: "GND",  # "ground",
                4: "A1M",  # "AC_1MOhm",
            },
            "fixed_vertical_gain": {
                0: "1 uV/div",
                1: "2 uV/div",
                2: "5 uV/div",
                3: "10 uV/div",
                4: "20 uV/div",
                5: "50 uV/div",
                6: "100 uV/div",
                7: "200 uV/div",
                8: "500 uV/div",
                9: "1 mV/div",
                10: "2 mV/div",
                11: "5 mV/div",
                12: "10 mV/div",
                13: "20 mV/div",
                14: "50 mV/div",
                15: "100 mV/div",
                16: "200 mV/div",
                17: "500 mV/div",
                18: "1 V/div",
                19: "2 V/div",
                20: "5 V/div",
                21: "10 V/div",
                22: "20 V/div",
                23: "50 V/div",
                24: "100 V/div",
                25: "200 V/div",
                26: "500 V/div",
                27: "1 kV/div",
            },
            "bandwidth_limit": {
                0: "off",
                1: "on",
            },
            "wave_source": {
                0: "channel_1",
                1: "channel_2",
                2: "channel_3",
                3: "channel_4",
                9: "unknown",
            },
        }[attr_name]

        if isinstance(value, str):  # Assigning string value
            # Make sure it is an allowed value
            if value not in translation.values():
                raise ValueError(f"Wrong {attr_name}, {value} not in "
                                 f"{translation.values()}")
            return value

        elif isinstance(value, int):  # Assigning int value
            # Make sure it is a valid enum
            if value not in translation.keys():
                raise ValueError(
                    f"Wrong {attr_name}, {value} not in {translation.keys()}")
            return translation[value]

        else:
            raise ValueError(
                f"{value} is of type {type(value)}, expected either int or str."
            )
