"""Durable native identity and one active writer per research author.

The research kernel still owns outcomes, leases, budgets and inbox delivery.
This sidecar owns only the provider binding and its observable event stream.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import socket
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from outerloop.harness import ClaudeCodeHarness, CodexHarness, Harness, SessionResult
from outerloop.session_stream import native_id

log = logging.getLogger(__name__)


class ControlError(RuntimeError):
    """An identity, ownership or persistence condition prevents author execution."""


@dataclass(frozen=True)
class Binding:
    version: int
    generation: int
    backend: str
    workspace: str
    home: str
    session_id: str = ""
    status: str = "launching"
    hostname: str = ""
    pid: int = 0
    released: bool = False


def read_binding(directory: Path) -> Binding | None:
    """Read the canonical binding without creating files (legacy runs have none)."""
    path = directory / "session-control" / "binding.json"
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        if (path.parent / "initialized").exists() or any(path.parent.glob("events-*.jsonl")):
            raise ControlError("author binding disappeared after initialization") from None
        return None
    try:
        with os.fdopen(fd) as stream:
            raw = json.load(stream)
        binding = Binding(**raw)
        if (
            binding.version != 1
            or type(binding.generation) is not int
            or binding.generation < 1
            or binding.backend not in ("codex", "claude-code")
            or binding.status not in ("launching", "running", "finished", "failed")
            or type(binding.pid) is not int
            or binding.pid < 0
            or type(binding.released) is not bool
            or (binding.released and binding.status not in ("finished", "failed"))
            or not all(
                isinstance(v, str)
                for v in (binding.workspace, binding.home, binding.hostname, binding.session_id)
            )
        ):
            raise ValueError("invalid binding fields")
        if binding.session_id:
            native_id(binding.session_id)
        return binding
    except (ValueError, TypeError, KeyError) as exc:
        raise ControlError("invalid persisted author binding") from exc


def projected_session_id(directory: Path, fallback: str) -> str:
    """Preserve legacy record readability; admission separately fails closed."""
    try:
        binding = read_binding(directory)
    except (ControlError, OSError):
        log.warning("unreadable author binding in %s; author admission will refuse it", directory)
        return fallback
    return binding.session_id if binding and binding.session_id else fallback


def read_events(directory: Path, generation: int) -> list[dict[str, Any]]:
    """Public journal reader. A writer's incomplete last line is not yet visible."""
    if type(generation) is not int or generation < 1:
        raise ValueError("invalid generation")
    path = directory / "session-control" / f"events-{generation}.jsonl"
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return []
    with os.fdopen(fd, "rb") as stream:
        data = stream.read()
    return [json.loads(line) for line in data.split(b"\n")[:-1] if line]


def _group_alive(binding: Binding) -> bool:
    if binding.released:
        return False  # The prior controller recorded completed group termination.
    if binding.hostname != socket.gethostname():
        raise ControlError("prior author belongs to another host; liveness is unresolved")
    if not binding.pid:
        if binding.status in ("finished", "failed"):
            return False  # A known pre-spawn failure is retryable.
        raise ControlError("prior author launch has no recorded process; liveness is unresolved")
    try:
        os.killpg(binding.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError as exc:
        raise ControlError("prior author process group is inaccessible") from exc
    return True


class SessionStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.path = directory / "session-control"
        self.path.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise ControlError("author control directory is a symlink")
        os.chmod(self.path, 0o700)
        self.sequence = 0

    @contextlib.contextmanager
    def _lock(self, name: str, *, blocking: bool = True) -> Iterator[None]:
        fd = os.open(self.path / name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            except BlockingIOError as exc:
                raise ControlError("an author controller already owns this run") from exc
            yield
        finally:
            os.close(fd)

    def execution(self) -> contextlib.AbstractContextManager[None]:
        return self._lock("execution.lock", blocking=False)

    def _write(self, binding: Binding) -> None:
        # A missing binding after first admission must never look like a legacy
        # run. Write the sentinel first; an interrupted first save fails closed.
        marker_fd = os.open(
            self.path / "initialized", os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600
        )
        try:
            os.fsync(marker_fd)
        finally:
            os.close(marker_fd)
        tmp = self.path / f".binding-{uuid.uuid4().hex}.tmp"
        try:
            fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "w") as stream:
                json.dump(asdict(binding), stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, self.path / "binding.json")
            dir_fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        finally:
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()

    def _current(self, generation: int) -> Binding:
        binding = read_binding(self.directory)
        if binding is None or binding.generation != generation:
            raise ControlError("stale author generation")
        return binding

    def _ensure_live(self) -> None:
        record_path = self.directory / "state.json"
        if record_path.exists() and json.loads(record_path.read_text()).get("state") == "ended":
            raise ControlError("an ended research run cannot start or steer an author")

    def begin(self, backend: str, workspace: Path, resume: str | None) -> Binding:
        with self._lock("state.lock"):
            self._ensure_live()
            workspace = workspace.resolve()
            home = workspace.parent / f"{workspace.name}-home"
            old = read_binding(self.directory)
            if old is not None:
                if (old.backend, old.workspace, old.home) != (backend, str(workspace), str(home)):
                    raise ControlError("author backend or native home changed")
                if _group_alive(old):
                    raise ControlError("prior author process group is still alive")
                if resume and old.session_id and resume != old.session_id:
                    raise ControlError("requested resume conflicts with durable binding")
                resume = old.session_id or resume
            if resume:
                native_id(resume)
                if not home.is_dir() or home.is_symlink():
                    raise ControlError("bound author home is missing or unsafe")
            binding = Binding(
                version=1,
                generation=(old.generation + 1 if old else 1),
                backend=backend,
                workspace=str(workspace),
                home=str(home),
                session_id=resume or "",
                hostname=socket.gethostname(),
            )
            self._write(binding)
            self.sequence = 0
            return binding

    def started(self, generation: int, pid: int) -> None:
        with self._lock("state.lock"):
            self._ensure_live()
            current = self._current(generation)
            if current.status != "launching" or pid <= 0:
                raise ControlError("invalid author process start")
            self._write(replace(current, status="running", pid=pid))

    def bind(self, generation: int, identity: str) -> None:
        with self._lock("state.lock"):
            self._bind(self._current(generation), identity)

    def _bind(self, current: Binding, identity: str) -> None:
        self._ensure_live()
        identity = native_id(identity)
        if current.session_id == identity:
            return  # A replay never rewrites the durable snapshot.
        if current.session_id or current.status not in ("launching", "running"):
            raise ControlError("native session identity conflicts with durable binding")
        self._write(replace(current, session_id=identity))

    def publish(self, generation: int, event: dict[str, Any]) -> None:
        with self._lock("state.lock"):
            self._ensure_live()
            current = self._current(generation)
            if current.status != "running":
                raise ControlError("author event arrived outside its active turn")
            if current.backend == "codex" and event.get("type") == "thread.started":
                self._bind(current, native_id(event.get("thread_id")))
            if current.backend == "claude-code" and (
                event.get("type") == "result"
                or (event.get("type") == "system" and event.get("subtype") == "init")
            ):
                self._bind(current, native_id(event.get("session_id")))
            # The identity is durable before consumers can see its event.
            self.sequence += 1
            frame = {"generation": generation, "sequence": self.sequence, "event": event}
            fd = os.open(
                self.path / f"events-{generation}.jsonl",
                os.O_CREAT | os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW,
                0o600,
            )
            with os.fdopen(fd, "w") as stream:
                stream.write(json.dumps(frame) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def finish(self, generation: int, failed: bool, *, released: bool = False) -> None:
        with self._lock("state.lock"):
            current = self._current(generation)
            self._write(
                replace(current, status="failed" if failed else "finished", released=released)
            )


@dataclass
class ControlledAuthor:
    harness: ClaudeCodeHarness | CodexHarness
    directory: Path
    supports_resume = True

    def run(
        self, brief_text: str, workspace: Path, resume_session_id: str | None = None
    ) -> SessionResult:
        result: SessionResult | None = None
        try:
            store = SessionStore(self.directory)
            with store.execution():
                backend = "codex" if isinstance(self.harness, CodexHarness) else "claude-code"
                binding = store.begin(backend, workspace, resume_session_id)
                observed = replace(
                    self.harness,
                    on_start=lambda pid: store.started(binding.generation, pid),
                    on_event=lambda event: store.publish(binding.generation, event),
                )
                result = observed.run(brief_text, workspace, binding.session_id or None)
                store.finish(
                    binding.generation,
                    result.is_error,
                    released=result.stop_reason != "cleanup-error",
                )
                identity = projected_session_id(self.directory, result.session_id)
                return replace(result, session_id=identity)
        except Exception as exc:
            # Admission and disk failures must flow through the kernel's ordinary
            # failed-session policy, never evaluation or inbox acknowledgement.
            identity = projected_session_id(self.directory, resume_session_id or "")
            return SessionResult(
                stop_reason="control-error",
                is_error=True,
                cost_usd=result.cost_usd if result else 0,
                num_turns=result.num_turns if result else 0,
                session_id=identity,
                final_text="",
                transcript_path=result.transcript_path if result else "",
                error_detail=(
                    f"author control refused execution: {exc}"
                    if isinstance(exc, ControlError)
                    else f"author control failed ({type(exc).__name__})"
                ),
            )


def controlled_author(harness: Harness, directory: Path) -> Harness:
    """Scope control to real native authors; preserve test doubles and other backends."""
    if isinstance(harness, (ClaudeCodeHarness, CodexHarness)):
        return ControlledAuthor(harness, directory)
    return harness
