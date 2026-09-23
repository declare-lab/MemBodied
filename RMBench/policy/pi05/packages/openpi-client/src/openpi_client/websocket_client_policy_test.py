from unittest import mock

import pytest

from openpi_client import websocket_client_policy


def test_reset_sends_command_and_consumes_reply():
    client = object.__new__(websocket_client_policy.WebsocketClientPolicy)
    client._packer = mock.Mock()
    client._packer.pack.return_value = b"reset command"
    client._ws = mock.Mock()
    client._ws.recv.return_value = b"reset reply"
    client.reset()
    client._packer.pack.assert_called_once_with({"_command": "reset"})
    client._ws.send.assert_called_once_with(b"reset command")
    client._ws.recv.assert_called_once_with()


def test_reset_surfaces_server_error():
    client = object.__new__(websocket_client_policy.WebsocketClientPolicy)
    client._packer = mock.Mock()
    client._ws = mock.Mock()
    client._ws.recv.return_value = "reset failed"
    with pytest.raises(RuntimeError, match="reset failed"):
        client.reset()
