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

import base64
import copy
import dataclasses
import datetime
import struct
from typing import Any, Dict, Tuple, Union, cast
import xml.etree.ElementTree as ET

import numpy as np
import numpy.typing as npt

from scaaml.capture.scope.scope_template import ScopeTraceType


class LecroyWaveform:
    """Parse waveform response.
    """
    # Template changes from 2_2 to 2_3:
    # - <292> 2_2 RESERVED3: word
    # - <292> 2_3 HORIZ_UNCERTAINTY: float ; uncertainty from one acquisition
    #   to the ; next, of the horizontal offset in seconds

    # This is a sha256 of the template definition string. Ideally one would
    # first ask for the template format using the command "TMPL?", parse it,
    # and then parse the waveform.
    # https://github.com/google/scaaml/issues/130
    #
    # Workaround check that hash of the template has not changed:
    #   hashlib.sha256(scope.query("TMPL?").encode("utf8")).hexdigest()
    SUPPORTED_PROTOCOL_TEMPLATE_SHAS: tuple[str, ...] = (
        # Probably LECROY_2_3 older release
        "0ad362abcbe5b15ada8410f22f65434240f1206c7ba4b88847d422b0ee55096b",
        # LECROY_2_3 ; Software Release 8.1.0, 98/09/29.
        "3ccd866b3f2eb4c0d7d3743c016ce1c6e8d1b24c3e9ca2dda66b110f79caac3d",
    )

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

    @property
    def wave_description(self) -> "WaveDesc":
        """Return a copy of the wave description.
        """
        return copy.deepcopy(self._wave_description)

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
        wave: npt.NDArray[np.uint16] = self.d.get(
            "wave1",
            np.array([], dtype=np.uint16),
        )
        return wave

    def get_wave1(
        self,
        first_sample: int = 0,
        length: int = -1,
    ) -> ScopeTraceType:
        """Get wave.
        """
        if length == 0:
            return np.array([], dtype=np.float32)
        wave: ScopeTraceType = np.array(self.get_raw_wave1(), dtype=np.float32)
        gain = self._wave_description.vertical_gain
        offset = self._wave_description.vertical_offset
        scaled_wave: ScopeTraceType = np.array(wave * gain - offset,
                                               dtype=np.float32)
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

        # <0> Header
        header = self.get_string(16)
        assert header == "WAVEDESC"

        # <16> Template name
        self._wave_description.template_name = self.get_string(16)

        # <32> Communication type
        self._wave_description.comm_type = WaveDesc.translate(
            "comm_type", self.get_int16())
        self._unit = self._wave_description.comm_type

        # <34> Byte order
        self._wave_description.comm_order = WaveDesc.translate(
            "comm_order", self.get_int16())
        self._order = self._wave_description.comm_order

        # <36> Wave description length
        self._wave_description.wavedesc_len = self.get_int32()
        assert self._wave_description.wavedesc_len >= 346

        # <40> User-text length
        self._wave_description.user_text_len = self.get_int32()

        # <44> RES_DESC1 (reserved?)
        _ = self.get_int32()
        # <48>
        self._wave_description.trigtime_len = self.get_int32()
        # <52>
        self._wave_description.ristime_len = self.get_int32()
        # <56> RES_ARRAY1 (reserved)
        _ = self.get_int32()

        # <60> Length of wave1
        self._wave_description.wave1_len = self.get_int32()

        # <64> Length of wave2
        self._wave_description.wave2_len = self.get_int32()

        _ = self.get_int32()  # <68> reserved
        _ = self.get_int32()  # <72> reserved
        self._wave_description.instrument_name = self.get_string(16)  # <76>
        self._wave_description.instrument_number = self.get_int32()  # <92>
        self._wave_description.trace_label = self.get_string(16)  # <96>
        _ = self.get_int16()  # <112> reserved
        _ = self.get_int16()  # <114> reserved
        self._wave_description.wave_array_count = self.get_int32()  # <116>
        self._wave_description.points_per_screen = self.get_int32()  # <120>
        self._wave_description.first_valid_pnt = self.get_int32()  # <124>
        self._wave_description.last_valid_pnt = self.get_int32()  # <128>
        self._wave_description.first_point = self.get_int32()  # <132>
        self._wave_description.sparsing_factor = self.get_int32()  # <136>
        self._wave_description.segment_index = self.get_int32()  # <140>
        self._wave_description.subarray_count = self.get_int32()  # <144>
        self._wave_description.sweeps_per_acq = self.get_int32()  # <148>
        self._wave_description.points_per_pair = self.get_int16()  # <152>
        self._wave_description.pair_offsets = self.get_int16()  # <154>
        self._wave_description.vertical_gain = self.get_float()  # <156>
        self._wave_description.vertical_offset = self.get_float()  # <160>
        self._wave_description.max_value = self.get_float()  # <164>
        self._wave_description.min_value = self.get_float()  # <168>
        self._wave_description.nominal_bits = self.get_int16()  # <172>
        self._wave_description.nom_subarray_count = self.get_int16()  # <174>
        self._wave_description.horizontal_interval = self.get_float()  # <176>
        self._wave_description.horizontal_offset = self.get_double()  # <180>
        self._wave_description.pixel_offset = self.get_double()  # <188>
        self._wave_description.vertical_unit = self.get_string(48)  # <196>
        self._wave_description.horizontal_unit = self.get_string(48)  # <244>
        # <292>
        self._wave_description.horizontal_uncertainty = self.get_float()
        # Time settings
        self._wave_description.trigger_time = self.get_timestamp()  # <296>
        self._wave_description.acq_duration = self.get_float()  # <312>
        self._wave_description.record_type = WaveDesc.translate(  # <316>
            "record_type",
            self.get_int16(),
        )
        self._wave_description.processing_done = WaveDesc.translate(  # <318>
            "processing_done",
            self.get_int16(),
        )
        _ = self.get_int16()  # <320> RESERVED5
        self._wave_description.ris_sweeps = self.get_int16()  # <322>

        # <324> Timebase
        self._wave_description.timebase = WaveDesc.translate(
            "timebase",
            self.get_int16(),
        )

        # <326> Vertical coupling
        self._wave_description.vertical_coupling = WaveDesc.translate(
            "vertical_coupling", self.get_int16())

        # <328>
        self._wave_description.probe_att = self.get_float()

        # <332> Fixed vertical gain
        self._wave_description.fixed_vertical_gain = WaveDesc.translate(
            "fixed_vertical_gain",
            self.get_int16(),
        )

        # <334> Bandwidth limit on/off
        self._wave_description.bandwidth_limit = WaveDesc.translate(
            "bandwidth_limit",
            self.get_int16(),
        )

        self._wave_description.vertical_vernier = self.get_float()  # <336>
        self._wave_description.acq_vertical_offset = self.get_float()  # <340>

        # <344> Wave source
        self._wave_description.wave_source = WaveDesc.translate(
            "wave_source",
            self.get_int16(),
        )

        # END OF WAVE DESCRIPTION BLOCK (total len 346)

        # USERTEXT block at most 160 bytes long
        # Block length
        block_len = self._wave_description.user_text_len
        if block_len > 0:
            assert block_len <= 160
            self.d["user_text"] = self.get_string(block_len)

        # TRIGTIME array
        block_len = self._wave_description.trigtime_len
        if block_len > 0:
            self.d["trigtime"] = []
            self._ofs += block_len

        # RISTIME array
        block_len = self._wave_description.ristime_len
        if block_len > 0:
            # random-interleaved-sampling array
            self.d["ristime"] = []
            self._ofs += block_len

        # DATA_ARRAY_1
        # Receive wave1
        block_len = self._wave_description.wave1_len
        if block_len > 0:
            self.d["wave1"] = np.array(
                self.unpack_wave(self._wave_description.wave_array_count))

        # DATA_ARRAY_2
        # Receive wave2
        block_len = self._wave_description.wave2_len
        if block_len > 0:
            self.d["wave2"] = np.array(
                self.unpack_wave(self._wave_description.wave_array_count))

        # SIMPLE: Array

        # DUAL: Array

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


class DigitalChannelWaveform:
    """Parsing of digital bus traces based on XML data (*.digXML or XMLdig
    files).

    Attributes:

        generic_info (dict[str, str]): Information about the oscilloscope and
        bus:
        -   InstrumentId: Name, type, serial number, software version.
        -   SerialNumber: The serial number of the oscilloscope.
        -   FirmwareVersion: The firmware version (verbose with build).
        -   Source: Which digital bus ("DIGITAL1", ..., "DIGITAL4").

        horizontal_unit (str): Probably "S".

        horizontal_resolution (float): Time resolution of the samples (in
        seconds). (This explanation is a hypothesis.) See
        app.Acquisition.C1.Out.Result.HorizontalResolution.

        horizontal_per_step (float): Time between two sample points in seconds.
        This is reported. Also the reciprocal of sampling rate (sampling_rate
        in samples per second is: 1 / horizontal_per_step).

        sampling_rate (float): Computed samples per second (1 /
        horizontal_per_step).

        trace_infos (dict[str, DigitalChannelWaveform.Line]): A dictionary of
        line sub_bus_name (the line name -- "D0", "D1", ..., "D15" if there are
        16 lines) to `DigitalChannelWaveform.Line` information. Only visible
        lines are present.

        traces (dict[str, npt.NDArray[bool]]): A dictionary of line
        sub_bus_name (the line name -- "D0", "D1", ..., "D15" if there are 16
        lines) to trace. The line names are user-specified and default to
        "DATA.0", "DATA.1", ... Only visible lines are present.
    """

    def __init__(self, xml_data: str) -> None:
        """Parse the digital channel.

        Args:

          xml_data (str): The XML data. See
          `LeCroyCommunication._get_xml_dig_data`. This should have the root
          element `LeCroyXStreamDSOdata`.
        """
        # the root should have a Header and Data.
        root: ET.Element = ET.fromstring(xml_data)
        assert root.tag == "LeCroyXStreamDSOdata"
        assert [child.tag for child in root] == ["Header", "Data"]

        header: ET.Element = root[0]

        # Parse the header.
        self.generic_info: dict[str, str] = {
            "InstrumentId": "",
            "SerialNumber": "",
            "FirmwareVersion": "",
            "Source": "",
        }
        # Information about the digital result.
        digital_result_info: ET.Element | None = None
        for child in header:
            if child.tag in self.generic_info:
                assert child.text is not None
                self.generic_info[child.tag] = child.text
                continue

            if child.tag == "ILecDigitalResult":
                digital_result_info = child
                continue

            raise ValueError(f"Unknown tag in the Header: {child.tag}")

        # Parse ILecDigitalResult
        assert digital_result_info is not None, "ILecDigitalResult not found"
        self.horizontal_unit: str = "S"
        line_infos: list[DigitalChannelWaveform.Line] = []
        self.selected_lines_mask: list[bool] = []
        # The number of bits in a single trace.
        self.num_samples: int
        # Number of selected lines.
        self.num_lines: int

        # Reciprocal of sampling rate (sampling_rate in samples per second is:
        # 1 / horizontal_per_step).
        self.horizontal_per_step: float
        self.horizontal_resolution: float

        for child in digital_result_info:
            match child.tag:
                case ("Times" | "Details" | "Status" | "FrameInfo" |
                      "LabelForState"):
                    pass
                case "StatusDescription":
                    assert child.text is not None
                    assert child.text.startswith(
                        "internal:NumInputLines=16|internal:SelectedLines=")
                    assert child.text.endswith("|internal:DisplayMode=Lines")
                    for c in child.text[49:49 + 16]:
                        match c:
                            case "0":
                                self.selected_lines_mask.append(False)
                            case "1":
                                self.selected_lines_mask.append(True)
                            case _:
                                raise ValueError(
                                    "Unexpected character in StatusDescription"
                                    f" {child.text}")
                case "BusInfo":
                    for bus_info in child:
                        match bus_info.tag:
                            case "Name" | "HasSymbolic":
                                pass
                            case "NumLevels":
                                assert bus_info.text is not None
                                assert int(
                                    bus_info.text
                                ) == 2, "Wrong number of binary levels"
                            case "NumLines":
                                assert bus_info.text is not None
                                self.num_lines = int(bus_info.text)
                case "LineInfo":
                    line_infos = [
                        DigitalChannelWaveform.Line.parse(line_element)
                        for line_element in child
                    ]
                case "HorUnit":
                    assert child.text is not None
                    self.horizontal_unit = child.text
                case "HorScale":
                    for scale_info in child:
                        match scale_info.tag:
                            case "UniformInterval" | "ClockSynchronized":
                                assert scale_info.text == "true"
                            case "HorStart":
                                pass
                            case "HorPerStep":
                                assert scale_info.text is not None
                                self.horizontal_per_step = float(
                                    scale_info.text)
                            case "HorResolution":
                                assert scale_info.text is not None
                                self.horizontal_resolution = float(
                                    scale_info.text)
                            case _:
                                raise ValueError(
                                    f"Unknown HorScale {scale_info.tag}")
                case "NumSamples":
                    if child.text:
                        self.num_samples = int(child.text)
                    else:
                        # child.text could be None, default
                        self.num_samples = 0
                case _:
                    raise ValueError(f"Unknown ILecDigitalResult: {child.tag}")

        # Check consistency of selected lines:
        # - their number is the same
        assert len(line_infos) == sum(
            self.selected_lines_mask), "Number of selected lines does not match"
        # - their indexes are correct (no Line is repeated)
        for line in line_infos:
            assert self.selected_lines_mask[
                line.weight], "Line selection index inconsistency"
        # - their number is expected
        assert len(line_infos) == self.num_lines

        # We do not parse the data since those are just the binary values.
        # data: ET.Element = root[1]
        #    <Segment>
        #      <Id />
        #      <HorizontalDelta />
        #      <BinaryData>BASE64DATA</BinaryData>
        #    </Segment>

        # Find the binary data (BinaryData is present)
        list_binary_data = root.findall(".//BinaryData")
        assert len(list_binary_data) == 1
        assert list_binary_data[0].text is not None
        binary_data: str = list_binary_data[0].text
        decoded = base64.b64decode(binary_data)

        # Split the array into line traces.
        np_trigger = np.frombuffer(decoded, dtype=bool)
        self.trace_infos: dict[str, DigitalChannelWaveform.Line] = {}
        self.traces: dict[str, npt.NDArray[np.bool_]] = {}
        for line in line_infos:
            self.trace_infos[line.sub_bus_name] = line
            begin: int = line.index * self.num_samples
            end: int = begin + self.num_samples
            self.traces[line.sub_bus_name] = np.copy(np_trigger[begin:end])

    @property
    def sampling_rate(self) -> float:
        """Return computed sampling rate (in samples per second) as a
        reciprocal of self.horizontal_per_step.
        """
        return 1 / self.horizontal_per_step

    @dataclasses.dataclass
    class Line:
        """Digital line description.

        Attributes:

          index (int): The digital results is a two dimensional array. This
          indexes into that array when only a subset of lines are visible
          (beware that the format is undocumented, this is just a hypothesis).

          name (str): User-definable name. Defaults to "DATA.0", "DATA.1", ...

          sub_bus_name (str): The line name ("D0", "D1", ..., "D15" if there
          are 16 lines).

          weight (int): The number of the line.
        """
        index: int
        name: str
        sub_bus_name: str
        weight: int

        @staticmethod
        def parse(line_element: ET.Element) -> "DigitalChannelWaveform.Line":
            assert len(line_element) == 4, "Expecting 4 attributes"
            line = DigitalChannelWaveform.Line(
                index=0,
                name="",
                sub_bus_name="",
                weight=0,
            )

            for child in line_element:
                if child.text is None:
                    continue
                match child.tag:
                    case "Idx":
                        line.index = int(child.text)
                    case "Name":
                        line.name = child.text
                    case "SubBusName":
                        line.sub_bus_name = child.text
                    case "Weight":
                        line.weight = int(child.text)

            return line
