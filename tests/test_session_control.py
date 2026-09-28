"""Exercise native control with real pipe, process and filesystem boundaries."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from outerloop.harness import ClaudeCodeHarness, CodexHarness
from outerloop.role_runner import run_role
from outerloop.roles import author_spec
from outerloop.runstate import ENDED, RUNNING, STUCK, RunRecord, load_record, save_record
from outerloop.session_control import (
    ControlError,
    SessionStore,
    controlled_author,
    read_binding,
    read_events,
)

FIXTURE = Path(__file__).parent / "fixtures" / "native_author.py"


def required_binding(directory):
    value = read_binding(directory)
    assert value is not None
    return value


def prepare(tmp_path, backend, mode="normal", timeout=2):
    directory = tmp_path / "runs" / "r1"
    workspace = directory / "ws"
    workspace.mkdir(parents=True)
    binary = tmp_path / "provider"
    binary.write_text(f"#!{sys.executable}\n" + FIXTURE.read_text())
    binary.chmod(0o700)
    (workspace / "fixture.json").write_text(json.dumps({"backend": backend, "mode": mode}))
    harness_type = CodexHarness if backend == "codex" else ClaudeCodeHarness
    harness = harness_type(
        api_key="fixture-secret", model="fixture", binary=str(binary), timeout_s=timeout
    )
    record = RunRecord(run_id="r1", target="owner/repo", task_title="fixture", state=RUNNING)
    save_record(tmp_path, record, now=1)
    return directory, workspace, harness, record


def wait_for(predicate, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.005)
    raise AssertionError("observation watchdog expired")


def stopped(pid):
    if not pid:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    # A zombie cannot execute or retain credentials; init owns its final reap.
    status = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return status.stdout.strip().startswith("Z") or not status.stdout.strip()


@pytest.mark.parametrize("backend", ["codex", "claude-code"])
def test_live_journal_and_stale_record(tmp_path, backend):
    directory, workspace, harness, old = prepare(tmp_path, backend, "barrier")
    with ThreadPoolExecutor() as executor:
        future = executor.submit(
            run_role, author_spec(), controlled_author(harness, directory), "brief", workspace
        )
        try:
            events = wait_for(
                lambda: [e for e in read_events(directory, 1) if e["event"].get("seq") == 7]
            )
            assert events[0]["event"]["text"] == "résumé"
            assert not future.done()
            assert required_binding(directory).session_id == "fixture-native-session"
            save_record(tmp_path, old, now=2)
            assert load_record(tmp_path, "r1").resume_session_id == "fixture-native-session"
            assert (directory / "session-control" / "binding.json").stat().st_mode & 0o777 == 0o600
        finally:
            (workspace / "release").touch()
        result = future.result(timeout=4)
    assert result.ok
    assert result.session.final_text == "native counter 1"
    assert required_binding(directory).status == "finished"


@pytest.mark.parametrize("backend", ["codex", "claude-code"])
def test_timeout_cleanup_and_new_process_resume(tmp_path, backend):
    directory, workspace, harness, _ = prepare(tmp_path, backend, "timeout", timeout=0.5)
    result = controlled_author(harness, directory).run("brief", workspace)
    assert result.is_error and result.stop_reason == "timeout"
    assert result.session_id == "fixture-native-session"
    started = json.loads((workspace / "started.json").read_text())
    wait_for(lambda: stopped(started["pid"]) and stopped(started["child"]))
    (workspace / "fixture.json").write_text(json.dumps({"backend": backend}))
    script = """
import json,sys
from pathlib import Path
from dataclasses import asdict
from outerloop.harness import CodexHarness, ClaudeCodeHarness
from outerloop.session_control import controlled_author
from outerloop.runstate import load_record
root, backend, binary = sys.argv[1:]
root = Path(root)
r = load_record(root, "r1")
h = (CodexHarness if backend == "codex" else ClaudeCodeHarness)(
    api_key="fixture-secret", model="fixture", binary=binary, timeout_s=2)
d = root / "runs" / "r1"
print(json.dumps(asdict(controlled_author(h, d).run("wake", d / "ws", r.resume_session_id))))
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), backend, harness.binary],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert child.returncode == 0, child.stderr
    resumed = json.loads(child.stdout)
    assert not resumed["is_error"], resumed
    assert resumed["final_text"] == "native counter 2"
    started2 = json.loads((workspace / "started.json").read_text())
    assert started2["resume"] == started["resume"] or started2["resume"] == result.session_id
    assert started2["home"] == started["home"] and started2["counter"] == 2


@pytest.mark.parametrize("backend", ["codex", "claude-code"])
@pytest.mark.parametrize(
    "mode,ok",
    [
        ("fragmented", True),
        ("malformed", True),
        ("stderr", True),
        ("closed-stdin", True),
        ("missing-id", False),
        ("nonzero", False),
        ("conflict", False),
        ("no-newline", False),
    ],
)
def test_protocol_edges(tmp_path, backend, mode, ok):
    directory, workspace, harness, _ = prepare(
        tmp_path, backend, mode, timeout=0.5 if mode == "no-newline" else 3
    )
    result = controlled_author(harness, directory).run("brief", workspace)
    assert result.is_error is not ok, asdict(result)
    if mode in ("missing-id", "no-newline"):
        assert not required_binding(directory).session_id


@pytest.mark.parametrize("backend", ["codex", "claude-code"])
def test_observer_failure_kills_group(tmp_path, backend, monkeypatch):
    directory, workspace, harness, _ = prepare(tmp_path, backend, "observer")
    original = SessionStore.publish

    def fail(self, generation, event):
        if event["type"] == "fixture.progress":
            raise OSError("simulated disk failure fixture-secret")
        original(self, generation, event)

    monkeypatch.setattr(SessionStore, "publish", fail)
    result = controlled_author(harness, directory).run("brief", workspace)
    assert result.is_error and result.stop_reason == "control-error"
    assert "fixture-secret" not in result.error_detail
    started = json.loads((workspace / "started.json").read_text())
    wait_for(lambda: stopped(started["pid"]) and stopped(started["child"]))
    assert result.session_id == "fixture-native-session"


def test_generation_identity_and_unresolved_launch(tmp_path):
    directory, workspace, _, _ = prepare(tmp_path, "codex")
    store = SessionStore(directory)
    binding = store.begin("codex", workspace, None)
    store.bind(binding.generation, "native-id")
    before = (store.path / "binding.json").stat().st_mtime_ns
    store.bind(binding.generation, "native-id")
    assert (store.path / "binding.json").stat().st_mtime_ns == before
    with pytest.raises(ControlError, match="conflicts"):
        store.bind(binding.generation, "different-id")
    with pytest.raises(ControlError, match="stale"):
        store.bind(binding.generation + 1, "native-id")
    with pytest.raises(ControlError, match="unresolved"):
        store.begin("codex", workspace, "native-id")
    store._write(replace(required_binding(directory), hostname=socket.gethostname() + "-other"))
    with pytest.raises(ControlError, match="another host"):
        store.begin("codex", workspace, "native-id")


def test_legacy_and_terminal_record(tmp_path):
    directory, workspace, harness, old = prepare(tmp_path, "codex")
    assert read_binding(directory) is None
    assert load_record(tmp_path, "r1").resume_session_id == ""
    assert not controlled_author(harness, directory).run("brief", workspace).is_error
    save_record(tmp_path, replace(old, state=ENDED, ending=STUCK), now=3)
    save_record(tmp_path, old, now=4)
    result = controlled_author(harness, directory).run("late wake", workspace)
    assert result.is_error
    assert load_record(tmp_path, "r1").state == ENDED
    assert json.loads((workspace / "started.json").read_text())["counter"] == 1


@pytest.mark.parametrize("corrupt", ['{"version":100}', "[]", "{broken"])
def test_corrupt_binding_refuses_author_but_keeps_record_readable(tmp_path, corrupt):
    directory, workspace, harness, _ = prepare(tmp_path, "codex")
    store = SessionStore(directory)
    (store.path / "binding.json").write_text(corrupt)
    assert load_record(tmp_path, "r1").run_id == "r1"
    assert controlled_author(harness, directory).run("brief", workspace).is_error
    assert not (workspace / "started.json").exists()


@pytest.mark.parametrize("backend", ["codex", "claude-code"])
def test_completed_cli_reaps_pipe_holding_child(tmp_path, backend):
    directory, workspace, harness, _ = prepare(tmp_path, backend, "completed-child")
    result = controlled_author(harness, directory).run("brief", workspace)
    assert not result.is_error, result
    started = json.loads((workspace / "started.json").read_text())
    wait_for(lambda: stopped(started["child"]))
    assert required_binding(directory).released


def test_released_turn_can_move_hosts_and_missing_binding_cannot_restart(tmp_path):
    directory, workspace, harness, _ = prepare(tmp_path, "codex")
    assert not controlled_author(harness, directory).run("brief", workspace).is_error
    store = SessionStore(directory)
    store._write(replace(required_binding(directory), hostname="other-host"))
    assert not controlled_author(harness, directory).run("wake", workspace).is_error
    (store.path / "binding.json").unlink()
    assert controlled_author(harness, directory).run("wake", workspace).is_error
    assert json.loads((workspace / "started.json").read_text())["counter"] == 2


def test_first_claude_result_cannot_be_replaced(tmp_path):
    directory, workspace, harness, _ = prepare(tmp_path, "claude-code", "duplicate-result")
    result = controlled_author(harness, directory).run("brief", workspace)
    assert result.is_error and result.cost_usd == 9


@pytest.mark.parametrize("backend", ["codex", "claude-code"])
def test_decoded_secrets_are_redacted_before_publication(tmp_path, backend):
    directory, workspace, harness, _ = prepare(tmp_path, backend, "escaped-secret")
    result = controlled_author(harness, directory).run("brief", workspace)
    assert not result.is_error
    events = read_events(directory, 1)
    secret_event = next(e["event"] for e in events if e["event"]["type"] == "fixture.secret")
    assert secret_event["text"] == "[REDACTED]"
    transcript = Path(result.transcript_path).read_text()
    assert "fixture-secret" not in transcript
    assert "fixture-secret" not in json.dumps(events)


def test_ending_during_login_prevents_prompt_delivery(tmp_path, monkeypatch):
    directory, workspace, harness, record = prepare(tmp_path, "codex")

    def end_during_login(self, home):
        save_record(tmp_path, replace(record, state=ENDED, ending=STUCK), now=2)
        return None

    monkeypatch.setattr(CodexHarness, "_login", end_during_login)
    result = controlled_author(harness, directory).run("must not deliver", workspace)
    assert result.is_error
    assert not (workspace / "received-prompt.json").exists()
    assert load_record(tmp_path, "r1").state == ENDED
    assert required_binding(directory).session_id == ""
