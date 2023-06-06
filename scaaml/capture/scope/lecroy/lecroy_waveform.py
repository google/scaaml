# Copyright 2023 Google LLC
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
"""Parse data comming from the LeCroy oscilloscope.
"""

import datetime
import struct

import numpy as np


class LecroyWaveform:
    """Parse waveform response.
    """

    def __init__(self, raw_data):
        """Initialize the parser. Use the `parse` method with the `get_wave1`.

        Args:
          raw_data (iterable): Iterable of bytes.
        """
        self.raw_data = bytearray(raw_data)
        self.d = {}
        self._order = "<"
        self._unit = "b"
        self._ofs = 0
        self.parse()

    def get_string(self, max_size=16) -> str:
        """Parse a string from self.raw_data.
        """
        val = struct.unpack_from(f"{self._order:s}{max_size:d}s", self.raw_data,
                                 self._ofs)[0]
        self._ofs += max_size
        return val.strip(b"\x00").decode("ascii")

    def get_int8(self) -> int:
        """Parse int8 from self.raw_data.
        """
        self._ofs += 1
        return struct.unpack_from("b", self.raw_data, self._ofs - 1)[0]

    def get_int16(self) -> int:
        """Parse int16 from self.raw_data.
        """
        self._ofs += 2
        return struct.unpack_from(f"{self._order:s}h", self.raw_data,
                                  self._ofs - 2)[0]

    def get_int32(self) -> int:
        """Parse int32 from self.raw_data.
        """
        self._ofs += 4
        return struct.unpack_from(f"{self._order:s}l", self.raw_data,
                                  self._ofs - 4)[0]

    def get_float(self) -> float:
        """Parse float from self.raw_data.
        """
        self._ofs += 4
        return struct.unpack_from(f"{self._order:s}f", self.raw_data,
                                  self._ofs - 4)[0]

    def get_double(self) -> float:
        """Parse double from self.raw_data.
        """
        self._ofs += 8
        return struct.unpack_from(f"{self._order:s}d", self.raw_data,
                                  self._ofs - 8)[0]

    def get_unit(self):
        """Parse a unit from self.raw_data.
        """
        fmt = f"{self._order}{self._unit}"
        size = struct.calcsize(fmt)
        val = struct.unpack_from(fmt, self.raw_data, self._ofs)[0]
        self._ofs += size
        return val

    def get_timestamp(self) -> datetime.datetime:
        """Parse timestamp from self.raw_data.
        """
        vals = struct.unpack_from(f"{self._order:s}d4b2h", self.raw_data,
                                  self._ofs)
        self._ofs += 16
        if all(int(v) == 0 for v in vals):
            return None
        return datetime.datetime(
            vals[5],  # year
            vals[4],  # month
            vals[3],  # day
            vals[2],  # hours
            vals[1],  # minutes
            int(vals[0]),  # seconds
            int(1000 * (vals[0] - (int(vals[0])))),  # microseconds
        )

    def get_raw_wave1(self):
        """Get raw (whole, all samples, no gain or offset compensation) wave.
        """
        return self.d.get("wave1", np.array([]))

    def get_wave1(self, first_sample=0, length=-1):
        """Get wave.
        """
        if length == 0:
            return np.array([], dtype=np.float32)
        wave = np.array(self.get_raw_wave1(), dtype=np.float32)
        gain = self.d.get("vertical_gain", 1.0)
        offset = self.d.get("vertical_offset", 0.0)
        scaled_wave = wave * gain - offset
        end = scaled_wave.size
        if length > 0:
            end = min(end, length + first_sample)
        return scaled_wave[first_sample:end]

    def __repr__(self):
        """Return human readable representation of self."""
        return self.d.__repr__()

    def parse(self):
        """Parse answer from LeCroy.
        """
        # Header
        header = self.get_string(16)
        assert header == "WAVEDESC"

        # Template name
        self.d["template_name"] = self.get_string(16)

        # Communication type
        comm_type = self.get_int16()
        self.d["comm_type"] = {0: "b", 1: "h"}.get(comm_type)
        self._unit = self.d["comm_type"]

        # Byte order
        comm_order = self.get_int16()
        self.d["comm_order"] = {0: ">", 1: "<"}.get(comm_order)
        self._order = self.d["comm_order"]

        # Wave description length
        self.d["wavedesc_len"] = self.get_int32()
        assert self.d["wavedesc_len"] >= 346

        # User-text length
        self.d["usertext_len"] = self.get_int32()

        # NOP
        self.get_int32()  # Reserved

        self.d["trigtime_len"] = self.get_int32()

        self.d["ristime_len"] = self.get_int32()

        # NOP
        self.get_int32()  # Reserved

        # Length of wave1
        self.d["wave1_len"] = self.get_int32()

        # Length of wave2
        self.d["wave2_len"] = self.get_int32()

        # NOP
        self.get_int32()  # Reserved
        # NOP
        self.get_int32()  # Reserved

        self.d["instrument_name"] = self.get_string(16)
        self.d["instrument_number"] = self.get_int32()
        self.d["trace_label"] = self.get_string(16)
        self.get_int16()  # Reserved
        self.get_int16()  # Reserved
        self.d["wave_array_count"] = self.get_int32()
        self.d["pnts_per_screen"] = self.get_int32()
        self.d["first_valid_pnt"] = self.get_int32()
        self.d["last_valid_pnt"] = self.get_int32()
        self.d["first_point"] = self.get_int32()
        self.d["sparsing_factor"] = self.get_int32()
        self.d["segment_index"] = self.get_int32()
        self.d["subarray_count"] = self.get_int32()
        self.d["sweeps_per_acq"] = self.get_int32()
        self.d["points_per_pair"] = self.get_int16()
        self.d["pair_offsets"] = self.get_int16()
        self.d["vertical_gain"] = self.get_float()
        self.d["vertical_offset"] = self.get_float()
        self.d["max_value"] = self.get_float()
        self.d["min_value"] = self.get_float()
        self.d["nominal_bits"] = self.get_int16()
        self.d["nom_subarray_count"] = self.get_int16()
        self.d["horiz_interval"] = self.get_float()
        self.d["horiz_offset"] = self.get_double()
        self.d["pixel_offset"] = self.get_double()
        self.d["vertunit"] = self.get_string(48)
        self.d["horunit"] = self.get_string(48)
        self.d["horiz_uncertainty"] = self.get_float()
        self.d["trigger_time"] = self.get_timestamp()
        self.d["acq_duration"] = self.get_float()

        # Record type
        self.d["record_type"] = {
            0: "single_sweep",
            1: "interleaved",
            2: "histogram",
            3: "graph",
            4: "filter_coefficient",
            5: "complex",
            6: "extrema",
            7: "sequence_obsolete",
            8: "centered_RIS",
            9: "peak_detect"
        }.get(self.get_int16())

        # Processing done
        self.d["processing_done"] = {
            0: "no_processing",
            1: "fir_filter",
            2: "interpolated",
            3: "sparsed",
            4: "autoscaled",
            5: "no_result",
            6: "rolling",
            7: "cumulative"
        }.get(self.get_int16())

        self.get_int16()  # Reserved
        self.d["ris_sweeps"] = self.get_int16()

        # Timebase
        self.d["timebase"] = {
            0: "1 ps/div",
            1: "2 ps/div",
            2: "5 ps/div",
            3: "10 ps/div",
            4: "20 ps/div",
            5: "50 ps/div",
            6: "100 ps/div",
            7: "200 ps/div",
            8: "500 ps/div",
            9: "1 ns/div",
            10: "2 ns/div",
            11: "5 ns/div",
            12: "10 ns/div",
            13: "20 ns/div",
            14: "50 ns/div",
            15: "100 ns/div",
            16: "200 ns/div",
            17: "500 ns/div",
            18: "1 us/div",
            19: "2 us/div",
            20: "5 us/div",
            21: "10 us/div",
            22: "20 us/div",
            23: "50 us/div",
            24: "100 us/div",
            25: "200 us/div",
            26: "500 us/div",
            27: "1 ms/div",
            28: "2 ms/div",
            29: "5 ms/div",
            30: "10 ms/div",
            31: "20 ms/div",
            32: "50 ms/div",
            33: "100 ms/div",
            34: "200 ms/div",
            35: "500 ms/div",
            36: "1 s/div",
            37: "2 s/div",
            38: "5 s/div",
            39: "10 s/div",
            40: "20 s/div",
            41: "50 s/div",
            42: "100 s/div",
            43: "200 s/div",
            44: "500 s/div",
            45: "1 ks/div",
            46: "2 ks/div",
            47: "5 ks/div",
            100: "external"
        }.get(self.get_int16())

        # Vertical coupling
        self.d["vert_coupling"] = {
            0: "DC_50_Ohms",
            1: "ground",
            2: "DC_1MOhm",
            3: "ground",
            4: "AC_1MOhm"
        }.get(self.get_int16())

        self.d["probe_att"] = self.get_float()

        # Fixed vertical gain
        self.d["fixed_vert_gain"] = {
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
            27: "1 kV/div"
        }.get(self.get_int16())

        # Bandwidth limit on/off
        self.d["bandwidth_limit"] = {0: "off", 1: "on"}.get(self.get_int16())

        self.d["vertical_vernier"] = self.get_float()
        self.d["acq_vert_offset"] = self.get_float()

        # Wave source
        self.d["wave_source"] = {
            0: "channel_1",
            1: "channel_2",
            2: "channel_3",
            3: "channel_4",
            9: "unknown"
        }.get(self.get_int16())

        self._ofs = self.d["wavedesc_len"]

        # Block length
        block_len = self.d["usertext_len"]
        if block_len > 0:
            assert block_len <= 160
            self.d["usertext"] = self.get_string(block_len)

        # Unused
        block_len = self.d["trigtime_len"]
        if block_len > 0:
            # trigtime array
            self.d["trigtime"] = []
            self._ofs += block_len

        # Unused
        block_len = self.d["ristime_len"]
        if block_len > 0:
            # random-interleaved-sampling array
            self.d["ristime"] = []
            self._ofs += block_len

        # Receive wave1
        block_len = self.d["wave1_len"]
        if block_len > 0:
            tmp = []
            for _ in range(self.d["wave_array_count"]):
                tmp.append(self.get_unit())
            self.d["wave1"] = np.array(tmp)

        # Receive wave2
        block_len = self.d["wave2_len"]
        if block_len > 0:
            tmp = []
            for _ in range(self.d["wave_array_count"]):
                tmp.append(self.get_unit())
            self.d["wave2"] = np.array(tmp)
        assert self._ofs == len(self.raw_data)
