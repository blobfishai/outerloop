"""Native CLI workers behind the existing Harness protocol.

This is a local research adapter, not the unattended author/panel deployment.
Profiles select subscription, API or Vertex authentication explicitly; credentials
never enter prompts or persisted plans. A persisted unfinished turn is refused
on restart: absence of its controller is not evidence that its provider stopped.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import selectors
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from outerloop.harness import SESSION_ENV_ALLOWLIST, SessionResult, VertexConfig, redact

SCHEMA = 1
MAX_LINE_BYTES = 2 * 1024 * 1024
ACTIVE = {"starting", "running"}


class ResearchError(ValueError):
    """A rejected operation that must not start a provider."""


def kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        # macOS can return EPERM for a group containing only reparented zombies
        # after its leader died. Do not confuse that with a live, unkillable
        # worker: verify the group before accepting cleanup.
        check = subprocess.run(
            ["/bin/ps", "-axo", "pgid=,stat="],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        rows = [line.split() for line in check.stdout.splitlines()]
        if check.returncode or any(
            len(row) != 2 or (row[0] == str(pid) and not row[1].startswith("Z")) for row in rows
        ):
            raise ResearchError("could not confirm native process-group cleanup") from None


def private_dir(path: Path) -> None:
    if path.is_symlink():
        raise ResearchError(f"refusing symlink directory: {path}")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)


def save_json(path: Path, value: Any) -> None:
    """Atomic, private, durable replacement, including the directory entry."""
    fd, name = tempfile.mkstemp(prefix=".record-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(name)


def load_state(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ResearchError(f"refusing symlink record: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ResearchError(f"cannot read record: {path}") from exc
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ResearchError(f"unsupported research record: {path}")
    return value


@contextlib.contextmanager
def exclusive(path: Path, *, wait: bool = False) -> Iterator[int]:
    """A stable flock inode; never unlink it, including after release."""
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise ResearchError(f"resource is already in use: {path}") from exc
        yield fd
    finally:
        os.close(fd)


@dataclass(frozen=True)
class SubscriptionProfile:
    backend: str
    directory: Path
    binary: str
    model: str
    auth_mode: str = "subscription"
    api_key_file: str = ""
    workspace_id: str = ""
    vertex: VertexConfig | None = None

    def __post_init__(self) -> None:
        if self.backend not in {"claude", "codex"}:
            raise ResearchError("subscription backend must be claude or codex")
        if not self.model.strip() or not self.binary.strip():
            raise ResearchError("an explicit model and CLI binary are required")
        path = self.directory.expanduser().resolve(strict=True)
        if not path.is_dir():
            raise ResearchError("subscription profile must be an existing native login directory")
        object.__setattr__(self, "directory", path)
        if self.auth_mode not in {"subscription", "api-key", "vertex"}:
            raise ResearchError("unsupported explicit native authentication mode")
        if self.auth_mode == "vertex":
            if self.backend != "claude" or self.vertex is None:
                raise ResearchError(
                    "Vertex authentication requires Claude and explicit coordinates"
                )
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,199}", self.vertex.project):
                raise ResearchError("invalid Vertex project")
            if not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", self.vertex.region):
                raise ResearchError("invalid Vertex region")
            if self.vertex.adc_file:
                adc = Path(self.vertex.adc_file).expanduser().resolve(strict=True)
                if not adc.is_file():
                    raise ResearchError("Vertex credential path must be a regular file")
                object.__setattr__(self, "vertex", replace(self.vertex, adc_file=str(adc)))
        elif self.vertex is not None:
            raise ResearchError("Vertex coordinates require explicit Vertex authentication")
        if self.api_key_file:
            if self.auth_mode != "api-key":
                raise ResearchError("an API credential file requires explicit API authentication")
            credential = Path(self.api_key_file).expanduser().resolve(strict=True)
            if not credential.is_file() or credential.stat().st_mode & 0o077:
                raise ResearchError("native API credential file must be private")
            object.__setattr__(self, "api_key_file", str(credential))
        if self.backend == "claude" and self.auth_mode == "api-key" and not self.api_key_file:
            raise ResearchError("Claude API authentication requires a private credential file")
        if self.workspace_id and (
            self.backend != "claude"
            or self.auth_mode != "api-key"
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,199}", self.workspace_id)
        ):
            raise ResearchError(
                "workspace header requires Claude API authentication and a valid ID"
            )

    def authentication_identity(self) -> dict:
        # Keep the legacy identity byte-compatible for subscription conversations.
        if self.auth_mode == "subscription":
            return {}
        value: dict = {"auth_mode": self.auth_mode}
        if self.api_key_file:
            value["api_key_file"] = self.api_key_file
        if self.workspace_id:
            value["workspace_id"] = self.workspace_id
        if self.vertex:
            value["vertex"] = asdict(self.vertex)
        return value

    def environment(self, home: Path) -> dict[str, str]:
        env = {key: os.environ[key] for key in SESSION_ENV_ALLOWLIST if key in os.environ}
        env["HOME"] = str(home)
        env["CUDA_VISIBLE_DEVICES"] = ""
        if self.backend == "claude":
            # On macOS, explicitly setting CLAUDE_CONFIG_DIR changes the native
            # keychain service, even when it names ~/.claude. Default-profile
            # login therefore needs the original native home and USER context.
            # Custom profiles keep their explicit directory and isolated home.
            if "USER" in os.environ:
                env["USER"] = os.environ["USER"]
            if self.directory == (Path.home() / ".claude").resolve():
                env["HOME"] = str(Path.home())
            else:
                env["CLAUDE_CONFIG_DIR"] = str(self.directory)
            env.update(
                CLAUDE_CODE_DISABLE_BACKGROUND_TASKS="1",
                CLAUDE_CODE_DISABLE_ADVISOR_TOOL="1",
            )
        else:
            env["CODEX_HOME"] = str(self.directory)
        if self.auth_mode == "api-key" and self.api_key_file:
            credential = Path(self.api_key_file)
            if not credential.is_file() or credential.stat().st_mode & 0o077:
                raise ResearchError("native API credential file must remain private")
            key = credential.read_text().strip()
            if not key or any(c in key for c in "\r\n"):
                raise ResearchError("native API credential is missing or malformed")
            env["ANTHROPIC_API_KEY" if self.backend == "claude" else "OPENAI_API_KEY"] = key
            if self.workspace_id:
                env["ANTHROPIC_CUSTOM_HEADERS"] = "anthropic-workspace-id: " + self.workspace_id
        if self.auth_mode == "vertex":
            assert self.vertex is not None
            env.update(self.vertex.env())
        return env

    def auth_command(self) -> list[str]:
        if self.backend == "claude":
            return [self.binary, *claude_isolation(self), "auth", "status", "--json"]
        method = "api" if self.auth_mode == "api-key" else "chatgpt"
        return [self.binary, "-c", f'forced_login_method="{method}"', "login", "status"]

    def check_auth(self, home: Path) -> None:
        """Use the native CLI; do not expose account metadata or credential files."""
        process = subprocess.Popen(
            self.auth_command(),
            env=self.environment(home),
            cwd=home,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=30)
        finally:
            kill_group(process.pid)
            process.wait(timeout=5)
        if self.backend == "claude":
            try:
                auth = json.loads(stdout)
            except ValueError:
                auth = {}
            ok = isinstance(auth, dict) and auth.get("loggedIn") is True
            if self.auth_mode == "vertex":
                ok = ok and auth.get("apiProvider") == "vertex"
            else:
                expected = "api_key" if self.auth_mode == "api-key" else "claude.ai"
                ok = (
                    ok
                    and auth.get("authMethod") == expected
                    and auth.get("apiProvider") == "firstParty"
                )
        else:
            expected = (
                "Logged in using API key"
                if self.auth_mode == "api-key"
                else "Logged in using ChatGPT"
            )
            surface = (stdout + stderr).replace("using an API key", "using API key")
            ok = expected in surface
        if process.returncode or not ok:
            label = (
                "subscription login"
                if self.auth_mode == "subscription"
                else self.auth_mode + " authentication"
            )
            raise ResearchError(f"{self.backend}: {label} required in selected profile")


CLAUDE_ISOLATION = [
    "--safe-mode",
    "--restricted",
    "--strict-mcp-config",
    "--mcp-config",
    '{"mcpServers":{}}',
    "--settings",
    '{"forceLoginMethod":"claudeai"}',
]


def claude_isolation(profile: SubscriptionProfile) -> list[str]:
    if profile.auth_mode == "subscription":
        return CLAUDE_ISOLATION
    # Retain isolation without forcing a subscription over an explicit API/ADC mode.
    return [*CLAUDE_ISOLATION[:-2], "--settings", "{}"]


def native_command(
    profile: SubscriptionProfile,
    workspace: Path,
    session_id: str,
    tools: str,
    max_turns: int,
) -> list[str]:
    """Use persistent native sessions, explicit maximum effort, and read-only tools."""
    if profile.backend == "claude":
        allowed = "Read,Glob,Grep,WebSearch,WebFetch" if tools == "read" else ""
        return [
            profile.binary,
            "--print",
            *claude_isolation(profile),
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--model",
            profile.model,
            "--effort",
            "max",
            "--max-turns",
            str(max_turns),
            "--tools",
            allowed,
            "--allowedTools",
            allowed,
            "--permission-mode",
            "dontAsk",
            "--disable-slash-commands",
            *(["--resume", session_id] if session_id else []),
        ]
    config = {
        "forced_login_method": "api" if profile.auth_mode == "api-key" else "chatgpt",
        "model_provider": "openai",
        "model_reasoning_effort": "max",
        "approval_policy": "never",
        "web_search": "live" if tools == "read" else "disabled",
        "features.shell_tool": False,
        "features.unified_exec": False,
        "features.view_image": False,
        "features.image_generation": False,
        "features.multi_agent": False,
        "features.apps": False,
        "apps._default.enabled": False,
        "features.remote_plugin": False,
        "features.code_mode": False,
        "skills.include_instructions": False,
        "orchestrator.skills.enabled": False,
        "project_doc_max_bytes": 0,
        # Recent Codex routes native web tools through this host even when
        # code mode is off. Shell/execution and extension tools stay disabled.
        "features.code_mode_host": tools == "read",
        "features.browser_use": False,
        "features.computer_use": False,
        "features.hooks": False,
        "features.plugins": False,
        "features.skill_search": False,
        "features.goals": False,
        "features.sleep_tool": False,
        "features.tool_suggest": False,
        "features.multi_agent_v2": False,
        "features.collaboration_modes": False,
        "features.default_mode_request_user_input": False,
    }
    argv = [profile.binary, "exec"]
    if session_id:
        argv += ["resume", session_id]
    else:
        argv += ["--sandbox", "read-only", "--cd", str(workspace)]
    argv += [
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--json",
        "--model",
        profile.model,
    ]
    for key, value in config.items():
        argv += ["-c", f"{key}={json.dumps(value)}"]
    return [*argv, "-"]


@dataclass
class SubscriptionHarness:
    """One persisted conversation; one active turn per profile and workspace.

    Compatible with Harness.run. The local research command uses this explicitly;
    legacy author and judge runners retain their existing authentication and gates.
    """

    profile: SubscriptionProfile
    state_dir: Path
    timeout_s: float = 900
    idle_timeout_s: float = 300
    tools: str = "read"
    max_turns: int = 24
    cancel_event: threading.Event | None = None
    supports_resume = True

    def run(
        self,
        brief_text: str,
        workspace: Path,
        resume_session_id: str | None = None,
    ) -> SessionResult:
        if not 0 < self.timeout_s <= 5400 or not 0 < self.idle_timeout_s <= self.timeout_s:
            raise ResearchError("require 0 < idle timeout <= walltime <= 5400 seconds")
        if self.tools not in {"none", "read"} or not 1 <= self.max_turns <= 120:
            raise ResearchError("invalid tools or turn budget")
        workspace = workspace.resolve(strict=True)
        private_dir(self.state_dir)
        self.state_dir = self.state_dir.resolve()
        if self.state_dir.is_relative_to(workspace):
            raise ResearchError("research records must be outside the worker workspace")
        home = self.state_dir / "home"
        private_dir(home)
        with contextlib.ExitStack() as stack:
            # Bind the same account profile across roots/processes; aliases resolve
            # to the same directory. The provider inherits locks so a dead caller
            # cannot immediately admit another worker over a still-running CLI.
            locks = [
                stack.enter_context(exclusive(path))
                for path in (
                    self.state_dir / "controller.lock",
                    self.profile.directory / ".outerloop-research.lock",
                    workspace.parent / f".{workspace.name}.outerloop-research.lock",
                )
            ]
            return self._run_locked(brief_text, workspace, resume_session_id, home, locks)

    def _run_locked(
        self,
        brief: str,
        workspace: Path,
        resume_id: str | None,
        home: Path,
        locks: list[int],
    ) -> SessionResult:
        record_path = self.state_dir / "state.json"
        identity = {
            "backend": self.profile.backend,
            "profile": str(self.profile.directory),
            "binary": self.profile.binary,
            "model": self.profile.model,
            "workspace": str(workspace),
            "tools": self.tools,
            **self.profile.authentication_identity(),
        }
        previous = load_state(record_path) if record_path.exists() else None
        if previous is not None:
            if previous.get("identity") != identity:
                raise ResearchError(
                    "conversation binding changed; use its original profile/model/workspace"
                )
            if previous.get("status") in ACTIVE:
                raise ResearchError(
                    "previous turn is unfinished; inspect its process before recovery"
                )
            if not resume_id or resume_id != previous.get("session_id"):
                raise ResearchError("existing conversation requires its exact native session ID")
        elif resume_id:
            raise ResearchError("cannot resume a session without a matching durable binding")

        # Authenticate before accepting a new turn. A failure preserves the last
        # completed result, so fixing login cannot erase recoverable history.
        self.profile.check_auth(home)
        # Resolve credentials before accepting a turn or allocating liveness FDs.
        # A key removed/changed after auth must preserve the completed conversation.
        native_environment = self.profile.environment(home)
        generation = int(previous["generation"]) + 1 if previous else 1
        turn_dir = self.state_dir / f"turn-{generation:04d}"
        turn_dir.mkdir(mode=0o700)  # refuse an existing receipt, never overwrite
        record: dict[str, Any] = {
            "schema": SCHEMA,
            "identity": identity,
            "generation": generation,
            "session_id": resume_id or "",
            "status": "starting",
            "pid": None,
            "started_at": time.time(),
            "prompt_sha256": hashlib.sha256(brief.encode()).hexdigest(),
            "timeout_s": self.timeout_s,
            "idle_timeout_s": self.idle_timeout_s,
            "max_turns": self.max_turns,
            "last_event_at": None,
            "event_count": 0,
        }

        def persist() -> None:
            save_json(turn_dir / "state.json", record)
            save_json(record_path, record)

        persist()
        argv = native_command(self.profile, workspace, resume_id or "", self.tools, self.max_turns)
        watch_read, watch_write = os.pipe()
        done_read, done_write = os.pipe()
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name("_subscription_process.py")),
                    str(watch_read),
                    str(done_write),
                    ",".join(str(fd) for fd in locks),
                    *argv,
                ],
                cwd=workspace,
                env=native_environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                pass_fds=(*locks, watch_read, done_write),
            )
        except OSError:
            os.close(watch_write)
            os.close(done_read)
            record.update(status="spawn-error", finished_at=time.time())
            persist()
            raise ResearchError("could not start the configured native CLI") from None
        finally:
            os.close(watch_read)
            os.close(done_write)
        # Cleanup covers all exceptions after spawn, including a failed disk write.
        try:
            record.update(status="running", pid=process.pid)
            persist()
            secrets = tuple(
                native_environment.get(k, "") for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
            )
            result = self._stream(process, brief, record, turn_dir, persist, done_read, secrets)
        finally:
            os.close(watch_write)  # supervisor observes EOF and kills the native group
            os.close(done_read)
            kill_group(process.pid)
            process.wait(timeout=10)
            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()
        record.update(status=result.stop_reason, finished_at=time.time(), result=asdict(result))
        persist()
        return result

    def _stream(
        self, process, brief, record, turn_dir, persist, done_read, secrets=()
    ) -> SessionResult:
        started = last_activity = time.monotonic()
        last_checkpoint = started
        buffers: dict[str, bytes] = {"stdout": b"", "stderr": b""}
        remaining = memoryview(brief.encode())
        terminal = False
        detail = ""
        stop = ""
        final = ""
        turns = 0
        native_returncode: int | None = None
        provider_cost: float | None = None
        completion_bytes = b""
        events_path = turn_dir / "events.jsonl"
        event_fd = os.open(events_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        native_path = turn_dir / "native-events.jsonl"
        try:
            native_fd = os.open(native_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except BaseException:
            os.close(event_fd)
            raise
        native_digest = hashlib.sha256()
        native_bytes = 0

        def event(line: bytes) -> None:
            nonlocal terminal, detail, final, turns, last_checkpoint, native_bytes, provider_cost
            try:
                data = json.loads(line)
            except (ValueError, UnicodeError):
                return
            if not isinstance(data, dict):
                return
            original = (redact(line.decode(), secrets) + "\n").encode()
            if native_bytes + len(original) > 64 * 1024 * 1024:
                raise ResearchError("private native transcript exceeds size limit")
            remaining_event = memoryview(original)
            while remaining_event:
                written = os.write(native_fd, remaining_event)
                remaining_event = remaining_event[written:]
            native_digest.update(original)
            native_bytes += len(original)
            kind = data.get("type", "")
            native_id = ""
            if self.profile.backend == "claude":
                if kind in {"system", "result"}:
                    native_id = data.get("session_id", "")
                if kind == "result":
                    terminal = data.get("subtype") == "success" and not data.get("is_error")
                    if not terminal:
                        detail = "provider did not report a successful result"
                    final = redact(str(data.get("result") or ""), secrets)
                    turns = data.get("num_turns", 0)
                    cost = data.get("total_cost_usd")
                    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                        if not math.isfinite(cost) or cost < 0:
                            raise ResearchError("invalid provider cost receipt")
                        provider_cost = float(cost)
            else:
                if kind == "thread.started":
                    native_id = data.get("thread_id", "")
                if kind in {"error", "turn.failed"}:
                    detail = "provider reported a failed turn"
                if kind == "turn.completed":
                    terminal = True
                    turns += 1
                item = data.get("item")
                if (
                    kind == "item.completed"
                    and isinstance(item, dict)
                    and item.get("type") == "agent_message"
                ):
                    final = redact(str(item.get("text") or ""), secrets)
            if native_id:
                if not isinstance(native_id, str) or len(native_id) > 200:
                    raise ResearchError("invalid native session identity")
                if record["session_id"] and native_id != record["session_id"]:
                    raise ResearchError("provider changed the bound native session identity")
                record["session_id"] = native_id
            record["event_count"] += 1
            record["last_event_at"] = time.time()
            # Only normalized metadata in the event journal; no auth responses,
            # provider configuration, raw prompts, or partial chain of thought.
            observed = {
                "type": kind,
                "at": record["last_event_at"],
                "session_id": record["session_id"],
                "generation": record["generation"],
            }
            os.write(event_fd, (json.dumps(observed) + "\n").encode())
            if (
                native_id
                or kind in {"result", "turn.completed", "turn.failed"}
                or time.monotonic() - last_checkpoint >= 1
            ):
                os.fsync(event_fd)
                os.fsync(native_fd)
                persist()  # native identity is durable while the turn is still live
                last_checkpoint = time.monotonic()

        try:
            with selectors.DefaultSelector() as selector:
                os.set_blocking(done_read, False)
                selector.register(done_read, selectors.EVENT_READ, "completion")
                for name in ("stdout", "stderr", "stdin"):
                    pipe = getattr(process, name)
                    os.set_blocking(pipe.fileno(), False)
                    selector.register(
                        pipe,
                        selectors.EVENT_WRITE if name == "stdin" else selectors.EVENT_READ,
                        name,
                    )
                while selector.get_map():
                    if self.cancel_event is not None and self.cancel_event.is_set():
                        stop, detail = "cancelled", "controller cancelled the turn"
                        break
                    if process.poll() is not None:
                        # A completed/crashed supervisor can leave descendants
                        # holding its pipes. Kill the known group before draining.
                        kill_group(process.pid)
                    now = time.monotonic()
                    if now - started >= self.timeout_s:
                        stop, detail = "timeout", "turn exceeded its walltime; process group killed"
                        break
                    if now - last_activity >= self.idle_timeout_s:
                        stop, detail = (
                            "idle-timeout",
                            "provider stream went idle; process group killed",
                        )
                        break
                    for key, _ in selector.select(timeout=min(0.1, self.timeout_s)):
                        name = key.data
                        if name == "stdin":
                            try:
                                if remaining:
                                    size = os.write(key.fd, remaining[:65536])
                                    remaining = remaining[size:]
                            except BrokenPipeError:
                                remaining = memoryview(b"")
                            if not remaining:
                                selector.unregister(key.fileobj)
                                process.stdin.close()
                            continue
                        chunk = os.read(key.fd, 65536)
                        if name == "completion":
                            completion_bytes += chunk
                            if len(completion_bytes) > 32:
                                raise ResearchError("invalid native completion receipt")
                            if b"\n" in completion_bytes:
                                try:
                                    native_returncode = int(completion_bytes.strip())
                                except ValueError:
                                    raise ResearchError(
                                        "invalid native completion receipt"
                                    ) from None
                                record["provider_returncode"] = native_returncode
                                selector.unregister(key.fileobj)
                                kill_group(process.pid)
                            elif not chunk:
                                selector.unregister(key.fileobj)
                            continue
                        if not chunk:
                            selector.unregister(key.fileobj)
                            if name == "stdout" and buffers[name].strip():
                                event(buffers[name])
                            continue
                        if name == "stderr":
                            # Diagnostics are not progress. Bound memory and avoid
                            # exposing account identifiers in the public result.
                            buffers[name] = (buffers[name] + chunk)[-4096:]
                            continue
                        last_activity = time.monotonic()
                        buffers[name] += chunk
                        while b"\n" in buffers[name]:
                            line, buffers[name] = buffers[name].split(b"\n", 1)
                            if len(line) > MAX_LINE_BYTES:
                                raise ResearchError("provider event exceeds size limit")
                            event(line)
                        if len(buffers[name]) > MAX_LINE_BYTES:
                            raise ResearchError("provider event exceeds size limit")
                if not stop:
                    try:
                        process.wait(
                            timeout=max(0.001, self.timeout_s - (time.monotonic() - started))
                        )
                    except subprocess.TimeoutExpired:
                        stop, detail = "timeout", "provider closed streams but did not exit"
                    else:
                        if native_returncode is None:
                            detail = "supervisor exited without a native completion receipt"
                        elif native_returncode:
                            detail = f"provider exited with status {native_returncode}"
        except ResearchError as exc:
            stop, detail = "protocol-error", str(exc)
        finally:
            try:
                os.fsync(event_fd)
                os.fsync(native_fd)
            finally:
                os.close(event_fd)
                os.close(native_fd)
            record.update(
                native_transcript_path=str(native_path),
                native_transcript_sha256=native_digest.hexdigest(),
                native_transcript_bytes=native_bytes,
                provider_cost_usd=provider_cost,
                provider_cost_status="reported" if provider_cost is not None else "unavailable",
            )
        if not stop:
            stop = (
                "completed"
                if terminal and record["session_id"] and final and not detail
                else "incomplete"
            )
        return SessionResult(
            stop_reason=stop,
            is_error=stop != "completed",
            cost_usd=provider_cost or 0.0,
            num_turns=turns if isinstance(turns, int) else 0,
            session_id=record["session_id"],
            final_text=final,
            transcript_path=str(events_path),
            error_detail=detail
            or (
                "missing successful terminal event, identity, or final answer"
                if stop != "completed"
                else ""
            ),
        )
