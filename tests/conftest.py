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


@pytest.fixture(autouse=True)
def tmp_sandbox(tmp_path, monkeypatch):
    """Points home, cache and data directories into the test's temporary folder so nothing outside it is touched."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / ".cache"))
    monkeypatch.setenv("PII_DATA_DIR", str(home / "data"))
