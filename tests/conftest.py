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
def tmp_sandbox(request, tmp_path_factory, monkeypatch):
    """Points home, cache, data and model directories into a temporary folder so nothing outside it is touched.

    Tests marked data or model keep the data and model directories the caller set, since they read real downloads.
    """
    home = tmp_path_factory.mktemp("sandbox_home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / ".cache"))
    needs_real_downloads = (
        request.node.get_closest_marker("data") is not None
        or request.node.get_closest_marker("model") is not None
    )
    if not needs_real_downloads:
        monkeypatch.setenv("PII_DATA_DIR", str(home / "data"))
        monkeypatch.setenv("PII_MODEL_DIR", str(home / "models"))
