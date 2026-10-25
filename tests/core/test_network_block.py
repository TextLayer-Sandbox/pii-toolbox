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


def test_sandbox_directories_are_inside_the_pytest_temp_root_and_apart_from_tmp_path(tmp_path, tmp_path_factory):
    root = tmp_path_factory.getbasetemp()
    for name in ("HOME", "XDG_CACHE_HOME", "PII_DATA_DIR", "PII_MODEL_DIR"):
        path = Path(os.environ[name])
        assert path.is_relative_to(root)
        assert not path.is_relative_to(tmp_path)


def test_data_and_model_marked_tests_keep_the_callers_directories(tmp_path):
    shutil.copy(CONFTEST, tmp_path / "conftest.py")
    (tmp_path / "test_inner.py").write_text(
        "import os\n"
        "import pytest\n"
        "\n"
        "@pytest.mark.data\n"
        "def test_data():\n"
        "    assert os.environ['PII_DATA_DIR'] == os.environ['EXPECTED_DATA_DIR']\n"
        "\n"
        "@pytest.mark.model\n"
        "def test_model():\n"
        "    assert os.environ['PII_MODEL_DIR'] == os.environ['EXPECTED_MODEL_DIR']\n"
    )
    callers_data_dir = str(tmp_path / "callers_data")
    callers_model_dir = str(tmp_path / "callers_models")
    env = {
        **os.environ,
        "PII_DATA_DIR": callers_data_dir,
        "PII_MODEL_DIR": callers_model_dir,
        "EXPECTED_DATA_DIR": callers_data_dir,
        "EXPECTED_MODEL_DIR": callers_model_dir,
    }
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(tmp_path / "test_inner.py"), "-q",
         "-o", "addopts=", "-m", "data or model", "-p", "no:cacheprovider", "--rootdir", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
