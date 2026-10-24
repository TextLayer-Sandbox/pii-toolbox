import socket

import pytest


def _blocked(*args, **kwargs):
    raise RuntimeError("network blocked: mark the test with @pytest.mark.network to allow it")


@pytest.fixture(autouse=True)
def block_network(request, monkeypatch):
    """Keeps every test off the network unless it is marked network, so no text can leave the machine."""
    if request.node.get_closest_marker("network") is None:
        monkeypatch.setattr(socket.socket, "connect", _blocked)
        monkeypatch.setattr(socket, "create_connection", _blocked)
        monkeypatch.setattr(socket, "getaddrinfo", _blocked)
