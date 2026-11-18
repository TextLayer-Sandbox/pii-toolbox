"""Identity resolver backed by a co-reference worker process.

The worker runs in its own uv environment and is reached over a ZeroMQ DEALER socket on a
local ipc:// endpoint. start() waits for a {"ready": true} message before returning. Requests
carry an id and replies are matched by it. A reply that does not arrive in time restarts the
worker and resends the request (Lazy Pirate), up to a capped number of retries. Replies hold character positions only; the resolver turns them into
entity groups.
"""

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
from pii_toolbox.evaluation.model_manifest import default_model_dir

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
    """Builds the command that launches the real worker in its own uv environment, reading models from `model_dir`."""

    def command(endpoint: str) -> list[str]:
        return ["uv", "run", "--project", str(WORKER_DIR), "python", str(WORKER_DIR / "server.py"), "--endpoint", endpoint, "--model-dir", str(model_dir)]

    return command


def stub_worker_command(endpoint: str) -> list[str]:
    """Launches the pattern-based stub worker with this interpreter; no model is needed."""
    return [sys.executable, str(WORKER_DIR / "server.py"), "--endpoint", endpoint, "--stub"]


class MaverickResolver(IdentityResolver):
    """Groups mentions of one person by asking a co-reference worker process.

    Detected names the model left unlinked then join a group through the nickname list chosen by
    `nickname_config`. The real worker reads its models from `model_dir` (the default model
    directory when omitted); a custom `worker_command` replaces the real worker.
    Starting the worker is explicit: call start() or use the resolver as a context manager.
    """

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
        self._launch()
        try:
            self._await_ready()
        except CorefWorkerError:
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
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.IDENTITY, RESOLVER_IDENTITY)
        self._socket.connect(self._endpoint)
        environment = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_HUB_CACHE": str(self._model_dir / ENCODER_CACHE_DIRECTORY)}
        self._process = subprocess.Popen(list(self._worker_command(self._endpoint)), env=environment)

    def _await_ready(self) -> None:
        """Blocks until a ready message arrives; raises if the process exits or the startup timeout passes."""
        deadline = time.monotonic() + WORKER_STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self._socket.poll(_POLL_SLICE_MILLISECONDS):
                if json.loads(self._socket.recv()).get("ready") is True:
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
