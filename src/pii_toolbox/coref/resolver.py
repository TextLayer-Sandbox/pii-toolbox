"""Identity resolver that talks to a co-reference worker process over a ZeroMQ socket."""

import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import TracebackType
from typing import Self

import zmq

from pii_toolbox.coref.groups import RESOLVER_NAME, build_groups
from pii_toolbox.coref.nickname_step import NicknameConfig, apply_nickname_step
from pii_toolbox.core import Document, PersonMentions, IdentityResolver, PiiType, Span
from pii_toolbox.core.model_dir import default_model_dir

WORKER_DIR = Path(__file__).parent / "worker"
ENCODER_CACHE_DIRECTORY = "huggingface_cache"
DEFAULT_REPLY_TIMEOUT_SECONDS = 120.0
WORKER_STARTUP_TIMEOUT_SECONDS = 600.0
RESOLVER_IDENTITY = b"pii-coref-resolver"
DEFAULT_MAX_RETRIES = 2
_POLL_SLICE_MILLISECONDS = 100

WorkerCommand = Callable[[str], Sequence[str]]
"""Builds the argument list that launches a worker bound to the given ipc:// endpoint."""


class CorefWorkerError(RuntimeError):
    """The worker did not answer after every allowed restart, or was used before it was started."""


def maverick_worker_command(model_dir: Path) -> WorkerCommand:
    """Builds the command that launches the model-backed worker in its own uv environment, reading models from `model_dir`."""

    def command(endpoint: str) -> list[str]:
        return ["uv", "run", "--project", str(WORKER_DIR), "python", str(WORKER_DIR / "server.py"), "--endpoint", endpoint, "--model-dir", str(model_dir)]

    return command


def stub_worker_command(endpoint: str) -> list[str]:
    """Launches the pattern-based stub worker with this interpreter; no model is needed."""
    return [sys.executable, str(WORKER_DIR / "server.py"), "--endpoint", endpoint, "--stub"]


class MaverickResolver(IdentityResolver):
    """Groups mentions of one person by asking a co-reference worker process."""

    languages = frozenset({"en"})

    def __init__(
        self,
        worker_command: WorkerCommand | None = None,
        reply_timeout_seconds: float = DEFAULT_REPLY_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        runtime_dir: Path | None = None,
        nickname_config: NicknameConfig = NicknameConfig(),
        model_dir: Path | None = None,
    ) -> None:
        self._model_dir = model_dir or default_model_dir()
        self._worker_command = worker_command or maverick_worker_command(self._model_dir)
        self._reply_timeout_seconds = reply_timeout_seconds
        self._max_retries = max_retries
        self._runtime_dir = runtime_dir
        self._nicknames = nickname_config.load()
        self._own_runtime_dir: tempfile.TemporaryDirectory | None = None
        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        self._process: subprocess.Popen | None = None
        self._endpoint = ""
        self._request_ids = itertools.count(1)

    @property
    def name(self) -> str:
        return RESOLVER_NAME

    @property
    def worker_process(self) -> subprocess.Popen | None:
        return self._process

    def start(self) -> None:
        if self._context is not None:
            return
        if self._runtime_dir is None:
            self._own_runtime_dir = tempfile.TemporaryDirectory(prefix="pii-coref-")
            self._runtime_dir = Path(self._own_runtime_dir.name)
        self._endpoint = f"ipc://{self._runtime_dir / 'coref.ipc'}"
        self._context = zmq.Context()
        try:
            self._launch()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._stop_worker()
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None
        if self._own_runtime_dir is not None:
            self._own_runtime_dir.cleanup()
            self._own_runtime_dir = None
            self._runtime_dir = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        self.close()

    def resolve(self, document: Document, spans: Sequence[Span]) -> list[PersonMentions]:
        person_spans = [span for span in spans if span.pii_type is PiiType.PERSON]
        if not person_spans:
            return []
        clusters = self._request_clusters(document.text, [(span.start, span.end) for span in person_spans])
        groups = build_groups(document, spans, clusters)
        return apply_nickname_step(document, spans, groups, self._nicknames)

    def _launch(self) -> None:
        """Spawns the worker on a fresh socket and blocks until it reports ready; raises if it exits or the startup timeout passes."""
        if self._socket is not None:
            self._socket.close(linger=0)
        assert self._context is not None
        socket = self._context.socket(zmq.DEALER)
        socket.setsockopt(zmq.IDENTITY, RESOLVER_IDENTITY)
        socket.connect(self._endpoint)
        self._socket = socket
        environment = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_HUB_CACHE": str(self._model_dir / ENCODER_CACHE_DIRECTORY)}
        self._process = subprocess.Popen(list(self._worker_command(self._endpoint)), env=environment)
        deadline = time.monotonic() + WORKER_STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if socket.poll(_POLL_SLICE_MILLISECONDS):
                if json.loads(socket.recv()).get("ready") is True:
                    return
                continue
            if self._process.poll() is not None:
                raise CorefWorkerError(f"the co-reference worker exited with code {self._process.returncode} before it was ready")
        raise CorefWorkerError(f"the co-reference worker was not ready within {WORKER_STARTUP_TIMEOUT_SECONDS:g} seconds")

    def _stop_worker(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._process = None

    def _restart_worker(self) -> None:
        self._stop_worker()
        self._launch()

    def _request_clusters(self, text: str, name_positions: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
        if self._context is None:
            raise CorefWorkerError("the co-reference worker is not started; call start() first")
        assert self._socket is not None
        request_id = next(self._request_ids)
        payload = json.dumps({"id": request_id, "text": text, "names": name_positions}).encode()
        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                self._restart_worker()
            self._socket.send(payload)
            reply = self._await_reply(request_id)
            if reply is not None:
                return [[(start, end) for start, end in cluster] for cluster in reply["clusters"]]
        raise CorefWorkerError(f"no reply from the co-reference worker after {self._max_retries} restarts")

    def _await_reply(self, request_id: int) -> dict | None:
        assert self._socket is not None
        deadline = time.monotonic() + self._reply_timeout_seconds
        while time.monotonic() < deadline:
            if self._socket.poll(_POLL_SLICE_MILLISECONDS):
                reply = json.loads(self._socket.recv())
                if reply.get("id") == request_id:
                    return reply
                continue
            if self._process is not None and self._process.poll() is not None:
                return None
        return None
