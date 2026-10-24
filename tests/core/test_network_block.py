import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

CONFTEST = Path(__file__).parent.parent / "conftest.py"


def test_unmarked_test_cannot_connect():
    with pytest.raises(RuntimeError, match="network blocked"):
        socket.create_connection(("127.0.0.1", 9))


def test_unmarked_test_cannot_resolve_names():
    with pytest.raises(RuntimeError, match="network blocked"):
        socket.getaddrinfo("localhost", 9)


def test_unmarked_test_cannot_use_socket_connect():
    with socket.socket() as sock:
        with pytest.raises(RuntimeError, match="network blocked"):
            sock.connect(("127.0.0.1", 9))


def test_network_marked_test_reaches_the_socket_layer(tmp_path):
    # The marked case runs in a child pytest because the default options deselect network tests here.
    shutil.copy(CONFTEST, tmp_path / "conftest.py")
    (tmp_path / "test_inner.py").write_text(
        "import socket\n"
        "import pytest\n"
        "\n"
        "@pytest.mark.network\n"
        "def test_attempt():\n"
        "    with pytest.raises(ConnectionRefusedError):\n"
        "        socket.create_connection(('127.0.0.1', 9))\n"
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(tmp_path / "test_inner.py"), "-q",
         "-o", "addopts=", "-m", "network", "-p", "no:cacheprovider", "--rootdir", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_home_points_inside_tmp_path(tmp_path):
    assert Path(os.environ["HOME"]).is_relative_to(tmp_path)
    assert Path(os.environ["XDG_CACHE_HOME"]).is_relative_to(tmp_path)
    assert Path(os.environ["PII_DATA_DIR"]).is_relative_to(tmp_path)
