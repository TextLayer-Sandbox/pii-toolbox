import socket

import pytest


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


@pytest.mark.network
def test_network_marked_test_reaches_the_socket_layer():
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("127.0.0.1", 9))
