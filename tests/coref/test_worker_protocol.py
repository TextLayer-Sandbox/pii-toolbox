"""Tests of the co-reference worker protocol using the pattern-based stub worker."""

import json
import sys

import pytest
import zmq

from pii_toolbox.coref.nickname_step import NicknameConfig
from pii_toolbox.coref import resolver as resolver_module
from pii_toolbox.coref.resolver import WORKER_DIR, CorefWorkerError, MaverickResolver, stub_worker_command
from pii_toolbox.core import Document, PiiType, Span

PASSAGE = "David Parker joined in May. Dave was glad, and Mr. Parker thanked the team. Anna Lee waved."
DOCUMENT = Document(id="d1", text=PASSAGE, language="en")
MAX_STRING_LENGTH = 20
MIN_TEXT_FRAGMENT = 4


READY_THEN_SILENT_WORKER = """
import json, sys, time, zmq
socket = zmq.Context().socket(zmq.ROUTER)
socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
socket.bind(sys.argv[1])
while True:
    try:
        socket.send_multipart([b"pii-coref-resolver", json.dumps({"ready": True}).encode()])
        break
    except zmq.ZMQError:
        time.sleep(0.05)
time.sleep(60)
"""

DELAYED_STUB_WORKER = """
import runpy, sys, time
time.sleep(float(sys.argv[1]))
sys.argv = [sys.argv[2], "--endpoint", sys.argv[3], "--stub"]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


@pytest.fixture
def runtime_dir(tmp_path_factory):
    # A short name keeps the ipc socket path under the operating system's socket path limit.
    return tmp_path_factory.mktemp("c")


def person(needle: str) -> Span:
    start = PASSAGE.index(needle)
    return Span(start, start + len(needle), PiiType.PERSON, "PERSON", ("fake",), 0.9)


def walk(value):
    yield value
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item)


def assert_positions_only(reply: dict) -> None:
    for value in walk(reply):
        if isinstance(value, str):
            assert len(value) <= MAX_STRING_LENGTH
            assert not (len(value) >= MIN_TEXT_FRAGMENT and value in PASSAGE)


def test_worker_reply_holds_positions_only(runtime_dir):
    with MaverickResolver(stub_worker_command, reply_timeout_seconds=20, runtime_dir=runtime_dir) as resolver:
        assert resolver._context is not None
        socket = resolver._context.socket(zmq.DEALER)
        socket.connect(resolver._endpoint)
        names = [[0, 12]]
        socket.send(json.dumps({"id": 7, "text": PASSAGE, "names": names}).encode())
        assert socket.poll(20000)
        reply = json.loads(socket.recv())
        socket.close(linger=0)
    assert reply["id"] == 7
    assert reply["clusters"]
    assert_positions_only(reply)
    for cluster in reply["clusters"]:
        for start, end in cluster:
            assert isinstance(start, int) and isinstance(end, int)


def test_resolver_returns_group_of_name_forms(runtime_dir):
    with MaverickResolver(stub_worker_command, reply_timeout_seconds=20, runtime_dir=runtime_dir) as resolver:
        groups = resolver.resolve(DOCUMENT, [person("David Parker")])
    assert resolver.name == "maverick"
    assert resolver.languages == frozenset({"en"})
    (group,) = groups
    texts = [PASSAGE[span.start : span.end] for span in group.spans]
    assert texts == ["David Parker", "Mr. Parker"]


def test_detected_nickname_joins_the_group_through_the_nickname_list(runtime_dir):
    with MaverickResolver(stub_worker_command, reply_timeout_seconds=20, runtime_dir=runtime_dir) as resolver:
        (group,) = resolver.resolve(DOCUMENT, [person("David Parker"), person("Dave")])
    assert [PASSAGE[span.start : span.end] for span in group.spans] == ["David Parker", "Dave", "Mr. Parker"]


def test_detected_nickname_stays_unlinked_with_the_list_off(runtime_dir):
    resolver = MaverickResolver(
        stub_worker_command, reply_timeout_seconds=20, runtime_dir=runtime_dir, nickname_config=NicknameConfig(None)
    )
    with resolver:
        (group,) = resolver.resolve(DOCUMENT, [person("David Parker"), person("Dave")])
    assert [PASSAGE[span.start : span.end] for span in group.spans] == ["David Parker", "Mr. Parker"]


def test_killed_worker_is_restarted_and_request_succeeds(runtime_dir):
    with MaverickResolver(stub_worker_command, reply_timeout_seconds=20, runtime_dir=runtime_dir) as resolver:
        assert resolver.resolve(DOCUMENT, [person("David Parker")])
        first = resolver.worker_process
        assert first is not None
        first.kill()
        first.wait()
        groups = resolver.resolve(DOCUMENT, [person("David Parker")])
        restarted = resolver.worker_process
        assert restarted is not None
        assert restarted.pid != first.pid
        assert restarted.poll() is None
    assert len(groups) == 1


def test_unresponsive_worker_raises_after_capped_retries(runtime_dir):
    launches = []

    def silent_worker(endpoint: str) -> list[str]:
        launches.append(endpoint)
        return [sys.executable, "-c", READY_THEN_SILENT_WORKER, endpoint]

    with MaverickResolver(silent_worker, reply_timeout_seconds=0.3, max_retries=2, runtime_dir=runtime_dir) as resolver:
        with pytest.raises(CorefWorkerError):
            resolver.resolve(DOCUMENT, [person("David Parker")])
    assert len(launches) == 3


def test_resolve_before_start_raises(runtime_dir):
    resolver = MaverickResolver(stub_worker_command, runtime_dir=runtime_dir)
    with pytest.raises(CorefWorkerError):
        resolver.resolve(DOCUMENT, [person("David Parker")])


def test_no_person_spans_needs_no_worker(runtime_dir):
    resolver = MaverickResolver(stub_worker_command, runtime_dir=runtime_dir)
    assert resolver.resolve(DOCUMENT, []) == []


def test_slow_model_load_does_not_use_up_the_reply_timeout(runtime_dir):
    def slow_loading_worker(endpoint: str) -> list[str]:
        return [sys.executable, "-c", DELAYED_STUB_WORKER, "1.5", str(WORKER_DIR / "server.py"), endpoint]

    with MaverickResolver(slow_loading_worker, reply_timeout_seconds=0.5, max_retries=0, runtime_dir=runtime_dir) as resolver:
        groups = resolver.resolve(DOCUMENT, [person("David Parker")])
    assert len(groups) == 1


def test_start_raises_when_worker_exits_before_ready(runtime_dir):
    def crashing_worker(endpoint: str) -> list[str]:
        return [sys.executable, "-c", "raise SystemExit(3)"]

    resolver = MaverickResolver(crashing_worker, runtime_dir=runtime_dir)
    with pytest.raises(CorefWorkerError, match="exited with code 3 before it was ready"):
        resolver.start()
    assert resolver.worker_process is None


def test_start_raises_when_worker_is_not_ready_in_time(runtime_dir, monkeypatch):
    monkeypatch.setattr(resolver_module, "WORKER_STARTUP_TIMEOUT_SECONDS", 0.3)

    def never_ready_worker(endpoint: str) -> list[str]:
        return [sys.executable, "-c", "import time; time.sleep(60)"]

    resolver = MaverickResolver(never_ready_worker, runtime_dir=runtime_dir)
    with pytest.raises(CorefWorkerError, match="not ready within"):
        resolver.start()
    assert resolver.worker_process is None


def test_failed_spawn_leaves_resolver_not_started_and_start_can_retry(runtime_dir):
    def missing_executable(endpoint: str) -> list[str]:
        return ["/nonexistent/pii-coref-worker"]

    resolver = MaverickResolver(missing_executable, runtime_dir=runtime_dir)
    with pytest.raises(FileNotFoundError):
        resolver.start()
    assert resolver.worker_process is None
    assert resolver._context is None
    assert resolver._socket is None
    with pytest.raises(FileNotFoundError):
        resolver.start()
    with pytest.raises(CorefWorkerError, match="not started"):
        resolver.resolve(DOCUMENT, [person("David Parker")])
