# Copyright 2022 Google LLC
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
"""Test Lecroy."""

from unittest.mock import call, Mock, patch
import uuid

import scaaml
from scaaml.capture.scope import LeCroy
from scaaml.capture.scope.lecroy.lecroy import LeCroyScope


@patch.object(
    scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationVisa,
    "close")
@patch.object(
    scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationVisa,
    "query")
@patch.object(
    scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationVisa,
    "write")
@patch.object(
    scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationVisa,
    "connect")
def test_get_identity_info(mock_connect, mock_write, mock_query, mock_close):
    model = "WAVEMASTER"
    serial_number = "WM01234"
    firmware_level = "1.2.3"
    mock_query.return_value = ",".join([
        "LECROY",
        model,
        serial_number,
        firmware_level,
    ])

    with LeCroy(samples=10,
                offset=0,
                ip_address="127.0.0.1",
                trace_channel="C1",
                trigger_channel="C2",
                communication_timeout=1.0,
                scope_setup_commands=[],
                trigger_timeout=0.1) as lecroy:

        assert lecroy.scope.get_identity_info() == {
            "lecroy_model": model,
            "lecroy_serial_number": serial_number,
            "lecroy_firmware_level": firmware_level,
        }


PRE_COMMANDS = [
    call({
        "command": "COMM_HEADER OFF",
    }),
    call({
        "command": "COMM_FORMAT DEF9,WORD,BIN",
        "query": "COMM_FORMAT?",
    }),
    call({
        "command": "TRMD SINGLE",
    }),
    call({
        "command": "AUTO_CALIBRATE OFF",
    }),
    call({
        "command": "OFFSET 0",
    }),
]
POST_COMMANDS = [
    call({"command": "STOP"}),
]


@patch.object(
    scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationVisa,
    "close")
@patch.object(
    scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationVisa,
    "query")
@patch.object(
    scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationVisa,
    "write")
@patch.object(
    scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationVisa,
    "connect")
@patch.object(scaaml.capture.scope.lecroy.lecroy.LeCroyScope, "_run_command")
def test_scope_setup_commands_empty(mock_run_command, mock_connect, mock_write,
                                    mock_query, mock_close):
    with LeCroy(samples=10,
                offset=0,
                ip_address="127.0.0.1",
                trace_channel="C1",
                trigger_channel="C2",
                communication_timeout=1.0,
                scope_setup_commands=[],
                trigger_timeout=0.1) as _:
        assert mock_run_command.call_args_list == PRE_COMMANDS + POST_COMMANDS


@patch.object(
    scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationVisa,
    "close")
@patch.object(
    scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationVisa,
    "query")
@patch.object(
    scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationVisa,
    "write")
@patch.object(
    scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationVisa,
    "connect")
@patch.object(scaaml.capture.scope.lecroy.lecroy.LeCroyScope, "_run_command")
def test_scope_setup_commands_non_empty(mock_run_command, mock_connect,
                                        mock_write, mock_query, mock_close):
    scope_setup_commands = [uuid.uuid4() for _ in range(4)]
    with LeCroy(
            samples=10,
            offset=0,
            ip_address="127.0.0.1",
            trace_channel="C1",
            trigger_channel="C2",
            communication_timeout=1.0,
            scope_setup_commands=scope_setup_commands,  # type: ignore[arg-type]
            trigger_timeout=0.1) as _:
        assert mock_run_command.call_args_list == PRE_COMMANDS + list(
            map(call, scope_setup_commands)) + POST_COMMANDS


@patch.object(
    scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationVisa,
    "close")
@patch.object(
    scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationVisa,
    "query")
@patch.object(
    scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationVisa,
    "write")
@patch.object(
    scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationVisa,
    "connect")
def test_run_command_empty(mock_connect, mock_write, mock_query, mock_close):
    scope = LeCroyScope(
        samples=42,
        offset=0,
        ip_address="127.0.0.1",
        trace_channel="C1",
        trigger_channel="C2",
        communication_timeout=5.0,
        trigger_timeout=1.0,
        communication_class_name="LeCroyCommunicationVisa",
        scope_setup_commands=[],
    )
    scope.con()

    scope._run_command({})


@patch.object(
    scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationVisa,
    "close")
@patch.object(
    scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationVisa,
    "query")
@patch.object(
    scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationVisa,
    "write")
@patch.object(
    scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationVisa,
    "connect")
def test_run_command_command_then_query(mock_connect, mock_write, mock_query,
                                        mock_close):
    scope = LeCroyScope(
        samples=42,
        offset=0,
        ip_address="127.0.0.1",
        trace_channel="C1",
        trigger_channel="C2",
        communication_timeout=5.0,
        trigger_timeout=1.0,
        communication_class_name="LeCroyCommunicationVisa",
        scope_setup_commands=[],
    )
    scope.con()

    def query_not_called(*args, **kwargs):
        assert mock_query.call_args_list == [call("COMM_FORMAT?")]

    mock_write.side_effect = query_not_called

    scope._run_command({
        "command": "command_a",
        "query": "query_question",
    })

    mock_write.assert_called_with("command_a")
    mock_query.assert_called_with("query_question")


@patch.object(
    scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationVisa,
    "close")
@patch.object(
    scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationVisa,
    "query")
@patch.object(
    scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationVisa,
    "write")
@patch.object(
    scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationVisa,
    "connect")
def test_run_command_command_query_updates_scope_answers(
        mock_connect, mock_write, mock_query, mock_close):
    query_response = "query_return_value"
    mock_query.side_effect = [query_response + str(i) for i in range(100)]
    scope = LeCroyScope(
        samples=42,
        offset=0,
        ip_address="127.0.0.1",
        trace_channel="C1",
        trigger_channel="C2",
        communication_timeout=5.0,
        trigger_timeout=1.0,
        communication_class_name="LeCroyCommunicationVisa",
        scope_setup_commands=[],
    )
    scope.con()

    scope._run_command({
        "query": "query_question",
    })

    assert scope.get_scope_answers() == {
        "COMM_FORMAT?": query_response + "0",
        "query_question": query_response + "1",
    }


@patch.object(
    scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationVisa,
    "close")
@patch.object(
    scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationVisa,
    "query")
@patch.object(
    scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationVisa,
    "write")
@patch.object(
    scaaml.capture.scope.lecroy.lecroy_communication.LeCroyCommunicationVisa,
    "connect")
@patch.object(scaaml.capture.scope.lecroy.lecroy.LeCroyScope, "set_trig_delay")
def test_run_command_method(mock_set_trig_delay, mock_connect, mock_write,
                            mock_query, mock_close):
    scope = LeCroyScope(
        samples=42,
        offset=0,
        ip_address="127.0.0.1",
        trace_channel="C1",
        trigger_channel="C2",
        communication_timeout=5.0,
        trigger_timeout=1.0,
        communication_class_name="LeCroyCommunicationVisa",
        scope_setup_commands=[],
    )
    scope.con()

    kwargs = {"divs_left": Mock()}
    scope._run_command({
        "method": "set_trig_delay",
        "kwargs": kwargs,
    })

    mock_set_trig_delay.assert_called_once_with(**kwargs)
