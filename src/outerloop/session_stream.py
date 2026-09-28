"""Bounded, single-reader JSONL transport for observed author sessions."""

from __future__ import annotations

import contextlib
import json
import os
import selectors
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

EventSink = Callable[[dict[str, Any]], None]
StartSink = Callable[[int], None]
MAX_FRAME = 4 * 1024 * 1024
MAX_OUTPUT = 64 * 1024 * 1024
STDERR_TAIL = 1024 * 1024


@dataclass
class StreamResult:
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    detail: str = ""
    session_id: str = ""
    completed: bool = False
    result: dict[str, Any] | None = None


def native_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError("invalid native session identity")
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError("invalid native session identity")
    return value


def communicate_events(
    process: subprocess.Popen[str],
    prompt: str,
    timeout_s: float,
    backend: str,
    secret: str,
    on_event: EventSink,
    on_start: StartSink | None,
) -> StreamResult:
    """Drain both pipes while delivering complete frames before process exit.

    Own every pipe until closed; never race communicate() against another reader.
    On any transport/observer error, kill the process group before returning. An
    escaped descendant cannot keep this reader alive past its deadline.
    """
    result = StreamResult()
    output = bytearray()
    errors = bytearray()
    pending = bytearray()
    deadline = time.monotonic() + timeout_s
    drain_deadline: float | None = None

    def scrub(value: Any) -> Any:
        if isinstance(value, str):
            return value.replace(secret, "[REDACTED]") if secret else value
        if isinstance(value, list):
            return [scrub(item) for item in value]
        if isinstance(value, dict):
            return {scrub(key): scrub(item) for key, item in value.items()}
        return value

    def frame(raw: bytes) -> None:
        if time.monotonic() >= deadline:
            raise subprocess.TimeoutExpired(process.args, timeout_s)
        try:
            text = raw.decode("utf-8")
            event = scrub(json.loads(text))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return  # Provider diagnostics can interleave JSONL.
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            return
        if backend == "claude-code" and event["type"] == "result" and result.completed:
            raise ValueError("provider emitted more than one terminal result")
        identity = None
        if backend == "codex" and event["type"] == "thread.started":
            identity = native_id(event.get("thread_id"))
        if backend == "claude-code" and (
            event["type"] == "result"
            or (event["type"] == "system" and event.get("subtype") == "init")
        ):
            identity = native_id(event.get("session_id"))
        if identity is not None and result.session_id and result.session_id != identity:
            raise ValueError("native session identity changed during a turn")
        on_event(event)
        if identity is not None:
            result.session_id = identity
        if backend == "codex" and event["type"] == "turn.completed":
            result.completed = True
        if backend == "claude-code" and event["type"] == "result":
            result.completed = True
            result.result = event

    try:
        with selectors.DefaultSelector() as selector:
            assert process.stdin is not None
            assert process.stdout is not None
            assert process.stderr is not None
            for pipe, name in ((process.stdout, "out"), (process.stderr, "err")):
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, name)
            payload = memoryview(prompt.encode("utf-8"))
            os.set_blocking(process.stdin.fileno(), False)
            if payload:
                selector.register(process.stdin, selectors.EVENT_WRITE, "in")
            else:
                process.stdin.close()
            if on_start is not None:
                on_start(process.pid)
            while selector.get_map() or process.poll() is None:
                if process.poll() is not None:
                    if drain_deadline is None:
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                        drain_deadline = time.monotonic() + 0.25
                    if time.monotonic() >= drain_deadline:
                        break  # An escaped descendant may still hold a pipe.
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(process.args, timeout_s)
                for key, _ in selector.select(min(remaining, 0.05)):
                    if key.data == "in":
                        try:
                            sent = os.write(key.fd, payload[:65536])
                            payload = payload[sent:]
                        except BrokenPipeError:
                            payload = memoryview(b"")
                        if not payload:
                            selector.unregister(key.fileobj)
                            process.stdin.close()
                        continue
                    try:
                        chunk = os.read(key.fd, 65536)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        if key.data == "out" and pending:
                            frame(bytes(pending))
                            pending.clear()
                        continue
                    if key.data == "err":
                        errors.extend(chunk)
                        del errors[:-STDERR_TAIL]
                        continue
                    output.extend(chunk)
                    if len(output) > MAX_OUTPUT:
                        raise ValueError("provider output exceeded 64 MiB")
                    pending.extend(chunk)
                    while b"\n" in pending:
                        line, _, rest = pending.partition(b"\n")
                        pending = bytearray(rest)
                        if len(line) > MAX_FRAME:
                            raise ValueError("provider frame exceeded 4 MiB")
                        frame(bytes(line))
                    if len(pending) > MAX_FRAME:
                        raise ValueError("provider frame exceeded 4 MiB")
            process.wait(timeout=max(0.001, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        result.error = "timeout"
        result.detail = f"session hit its {timeout_s}s walltime and was killed"
    except Exception as exc:
        result.error = "control-error"
        # Observer exceptions can contain provider text or credentials.
        result.detail = f"session transport or observer failed ({type(exc).__name__})"
    finally:
        # Local tool descendants must not outlive a completed author either.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            result.error = "cleanup-error"
            result.detail = "could not terminate author process group"
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            result.error = "cleanup-error"
            result.detail = "author process did not exit after termination"
        for final_pipe in (process.stdin, process.stdout, process.stderr):
            if final_pipe is not None:
                with contextlib.suppress(OSError):
                    final_pipe.close()
        lines = []
        for output_line in output.decode("utf-8", errors="replace").splitlines():
            try:
                lines.append(json.dumps(scrub(json.loads(output_line))))
            except (ValueError, RecursionError):
                lines.append(scrub(output_line))
        result.stdout = "\n".join(lines)
        result.stderr = errors.decode("utf-8", errors="replace")
    return result
