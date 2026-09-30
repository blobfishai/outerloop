"""Actual subprocesses, native JSONL contracts, and adversarial lifecycle cases."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from outerloop.harness import VertexConfig
from outerloop.research_cli import Plan, Worker, resume, run_plan
from outerloop.subscription import (
    ResearchError,
    SubscriptionHarness,
    SubscriptionProfile,
    exclusive,
    load_state,
    native_command,
    save_json,
)

SESSION = "cbb18db5-69e0-4f25-bb51-a08e510bdb04"
FAKE = r"""
import json, os, pathlib, signal, subprocess, sys, time

backend = "codex" if "CODEX_HOME" in os.environ else "claude"
profile = pathlib.Path(os.environ.get("CODEX_HOME") or os.environ["CLAUDE_CONFIG_DIR"])
options = json.loads((profile / "fixture.json").read_text())
if "status" in sys.argv:
    if backend == "codex":
        print("Logged in using API key" if options.get("bad_auth") else "Logged in using ChatGPT")
    else:
        print(
            json.dumps(
                {
                    "loggedIn": True,
                    "authMethod": "api_key" if options.get("bad_auth") else "claude.ai",
                    "apiProvider": "vertex" if options.get("vertex") else "firstParty",
                }
            )
        )
    sys.exit(0)
prompt = sys.stdin.read()
(profile / "native.pid").write_text(str(os.getpid()))
with (profile / "calls.jsonl").open("a") as f:
    f.write(
        json.dumps(
            {"argv": sys.argv[1:], "env": dict(os.environ), "prompt": prompt, "at": time.time()}
        )
        + "\n"
    )
sid = "cbb18db5-69e0-4f25-bb51-a08e510bdb04"
if "resume" in sys.argv or "--resume" in sys.argv:
    assert (profile / "native-history").read_text() == sid
else:
    (profile / "native-history").write_text(sid)


def emit(value):
    print(json.dumps(value), flush=True)


if options.get("child"):
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
        ]
    )
    (profile / "child.pid").write_text(str(child.pid))
emit(
    {"type": "thread.started", "thread_id": sid}
    if backend == "codex"
    else {"type": "system", "subtype": "init", "session_id": sid}
)
if options.get("rich_events"):
    emit({"type": "assistant", "message": {"content": [{
        "type": "tool_use", "id": "fixture-tool", "name": "Read",
        "input": {"file_path": "fixture.txt"},
    }]}})
    emit({"type": "stream_event", "event": {"delta": {
        "text": os.environ.get("ANTHROPIC_API_KEY", "fixture-text"),
    }}})
if options.get("conflict"):
    emit(
        {"type": "thread.started", "thread_id": "other"}
        if backend == "codex"
        else {"type": "system", "session_id": "other"}
    )
if options.get("oversize"):
    print("x" * (2 * 1024 * 1024 + 1), flush=True)
if options.get("hang"):
    time.sleep(60)
until = time.monotonic() + options.get("delay", 0.05)
while time.monotonic() < until:
    if options.get("heartbeat"):
        emit({"type": "stream_event"})
    time.sleep(0.025)
if options.get("failure"):
    emit({"type": "turn.failed", "error": {"message": "fixture"}})
if not options.get("missing_terminal"):
    if backend == "codex":
        emit(
            {"type": "item.completed", "item": {"type": "agent_message", "text": "verified answer"}}
        )
        emit({"type": "turn.completed"})
    else:
        emit(
            {
                "type": "result",
                "subtype": "success",
                "session_id": sid,
                "is_error": False,
                "num_turns": 1,
                "result": "verified answer",
                **({"total_cost_usd": options["cost"]} if "cost" in options else {}),
            }
        )
sys.exit(options.get("exit", 0))
"""


def fixture_profile(tmp_path: Path, backend="codex", **options) -> SubscriptionProfile:
    tmp_path.mkdir(parents=True, exist_ok=True)
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "fixture.json").write_text(json.dumps(options))
    binary = tmp_path / "native-cli"
    binary.write_text(f"#!{sys.executable}\n" + FAKE)
    binary.chmod(0o700)
    return SubscriptionProfile(backend, profile, str(binary), "fixture-model")


def harness(tmp_path: Path, backend="codex", timeout=2, idle=1, **options):
    profile = fixture_profile(tmp_path, backend, **options)
    ws = tmp_path / "workspace"
    ws.mkdir()
    return SubscriptionHarness(profile, tmp_path / "state", timeout, idle), ws


@pytest.mark.parametrize("backend", ["claude", "codex"])
def test_subscription_environment_native_resume_and_private_receipts(
    tmp_path, monkeypatch, backend
):
    for key in [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "GH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "CODEX_API_KEY",
    ]:
        monkeypatch.setenv(key, "should-not-reach-worker")
    adapter, ws = harness(tmp_path, backend)
    first = adapter.run("remember this research goal", ws)
    second = adapter.run("continue", ws, first.session_id)
    assert not first.is_error and not second.is_error
    assert first.session_id == second.session_id == SESSION
    calls = [
        json.loads(s) for s in (adapter.profile.directory / "calls.jsonl").read_text().splitlines()
    ]
    assert len(calls) == 2
    assert "should-not-reach-worker" not in json.dumps(calls)
    assert all("remember this research goal" not in c["argv"] for c in calls)
    assert ("--resume" if backend == "claude" else "resume") in calls[1]["argv"]
    state = load_state(tmp_path / "state" / "state.json")
    assert state["generation"] == 2
    assert (tmp_path / "state" / "turn-0001" / "state.json").is_file()
    for path in (tmp_path / "state").rglob("*.json*"):
        assert path.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize("backend", ["claude", "codex"])
@pytest.mark.parametrize("options", [dict(missing_terminal=True), dict(exit=7)])
def test_exit_or_partial_stream_cannot_claim_success(tmp_path, backend, options):
    adapter, ws = harness(tmp_path, backend, **options)
    result = adapter.run("brief", ws)
    assert result.is_error and result.stop_reason == "incomplete"
    assert result.session_id == SESSION


@pytest.mark.parametrize("backend", ["claude", "codex"])
def test_id_is_durable_before_completion_and_retained_on_timeout(tmp_path, backend):
    adapter, ws = harness(tmp_path, backend, timeout=0.7, idle=0.4, hang=True)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(adapter.run, "brief", ws)
        deadline = time.monotonic() + 3
        observed = None
        while time.monotonic() < deadline:
            state_path = tmp_path / "state" / "state.json"
            if state_path.exists():
                observed = load_state(state_path)
                if observed.get("session_id"):
                    break
            time.sleep(0.01)
        assert observed is not None
        assert observed["session_id"] == SESSION
        assert observed["status"] == "running" and not future.done()
        result = future.result(timeout=3)
    assert result.stop_reason == "idle-timeout" and result.session_id == SESSION


def test_partial_activity_uses_walltime_not_idle_timeout(tmp_path):
    adapter, ws = harness(tmp_path, timeout=1.5, idle=1, delay=3, heartbeat=True)
    result = adapter.run("brief", ws)
    assert result.stop_reason == "timeout"
    assert result.session_id == SESSION


@pytest.mark.parametrize("options", [dict(conflict=True), dict(oversize=True)])
def test_protocol_conflicts_are_refused(tmp_path, options):
    adapter, ws = harness(tmp_path, **options)
    result = adapter.run("brief", ws)
    assert result.stop_reason == "protocol-error"
    assert result.session_id == SESSION


def process_alive(pid: int) -> bool:
    result = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return bool(result.stdout.strip()) and not result.stdout.strip().startswith("Z")


@pytest.mark.parametrize("index", range(20))
def test_timeout_kills_term_ignoring_descendants(tmp_path, index):
    # The test must reach child creation to exercise descendant cleanup, even
    # when a busy host takes over 300ms to start its two Python processes.
    adapter, ws = harness(tmp_path, timeout=3, idle=1, child=True, hang=True)
    result = adapter.run("brief", ws)
    assert result.stop_reason == "idle-timeout"
    pid = int((adapter.profile.directory / "child.pid").read_text())
    # A killed descendant may remain a zombie until init reaps it, but cannot write.
    assert not process_alive(pid)


def test_normal_exit_also_cleans_owned_background_children(tmp_path):
    adapter, ws = harness(tmp_path, child=True)
    result = adapter.run("brief", ws)
    assert not result.is_error
    assert not process_alive(int((adapter.profile.directory / "child.pid").read_text()))


def test_bad_auth_starts_no_model(tmp_path):
    adapter, ws = harness(tmp_path, bad_auth=True)
    with pytest.raises(ResearchError, match="subscription login required"):
        adapter.run("brief", ws)
    assert not (adapter.profile.directory / "calls.jsonl").exists()


@pytest.mark.parametrize("backend", ["claude", "codex"])
def test_explicit_api_mode_resumes_and_does_not_store_the_credential(tmp_path, backend):
    original = fixture_profile(tmp_path, backend, bad_auth=True, rich_events=True)
    key = tmp_path / "api-key"
    key.write_text("fixture-api-key-never-a-real-credential")
    key.chmod(0o600)
    profile = SubscriptionProfile(
        backend,
        original.directory,
        original.binary,
        original.model,
        auth_mode="api-key",
        api_key_file=str(key),
        workspace_id="fixture-workspace" if backend == "claude" else "",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = SubscriptionHarness(profile, tmp_path / "state", timeout_s=4, idle_timeout_s=3)
    first = adapter.run("first fixture request", workspace)
    second = adapter.run("continued fixture request", workspace, first.session_id)
    assert not first.is_error and not second.is_error
    state = load_state(tmp_path / "state/state.json")
    assert state["generation"] == 2
    assert state["identity"]["auth_mode"] == "api-key"
    assert state["native_transcript_bytes"] > 0
    import hashlib

    transcript = Path(state["native_transcript_path"])
    assert hashlib.sha256(transcript.read_bytes()).hexdigest() == state["native_transcript_sha256"]
    assert '"tool_use"' in transcript.read_text()
    for path in (tmp_path / "state").rglob("*.json*"):
        assert key.read_text() not in path.read_text()
        assert path.stat().st_mode & 0o077 == 0
    if backend == "claude":
        assert "[redacted]" in transcript.read_text()
        assert (
            profile.environment(tmp_path)["ANTHROPIC_CUSTOM_HEADERS"]
            == "anthropic-workspace-id: fixture-workspace"
        )
        assert "forceLoginMethod" not in " ".join(native_command(profile, workspace, "", "none", 2))
    else:
        assert 'forced_login_method="api"' in native_command(profile, workspace, "", "none", 2)


def test_vertex_coordinates_are_bound_and_cannot_change_on_resume(tmp_path):
    original = fixture_profile(tmp_path, "claude", vertex=True)
    profile = SubscriptionProfile(
        "claude",
        original.directory,
        original.binary,
        original.model,
        auth_mode="vertex",
        vertex=VertexConfig(project="fixture-project", region="global"),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = SubscriptionHarness(profile, tmp_path / "state", timeout_s=4, idle_timeout_s=3)
    result = adapter.run("fixture", workspace)
    assert not result.is_error
    assert profile.environment(tmp_path)["CLAUDE_CODE_USE_VERTEX"] == "1"
    adapter.profile = SubscriptionProfile(
        "claude",
        original.directory,
        original.binary,
        original.model,
        auth_mode="vertex",
        vertex=VertexConfig(project="different-project", region="global"),
    )
    with pytest.raises(ResearchError, match="binding changed"):
        adapter.run("continue", workspace, result.session_id)


def test_credential_change_after_auth_preserves_completed_turn_and_allocates_no_pipes(
    tmp_path, monkeypatch
):
    original = fixture_profile(tmp_path, "claude", bad_auth=True)
    key = tmp_path / "api-key"
    key.write_text("fixture-key")
    key.chmod(0o600)
    profile = SubscriptionProfile(
        "claude",
        original.directory,
        original.binary,
        original.model,
        auth_mode="api-key",
        api_key_file=str(key),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = SubscriptionHarness(profile, tmp_path / "state", timeout_s=4, idle_timeout_s=3)
    first = adapter.run("first fixture request", workspace)
    previous = (tmp_path / "state/state.json").read_bytes()
    original_environment = SubscriptionProfile.environment
    environment_calls = 0

    def change_after_auth(selected, home):
        nonlocal environment_calls
        env = original_environment(selected, home)
        environment_calls += 1
        if environment_calls == 1:
            key.chmod(0o644)
        return env

    pipe_calls = []
    original_pipe = os.pipe

    def count_pipe():
        pipe_calls.append(True)
        return original_pipe()

    monkeypatch.setattr(SubscriptionProfile, "environment", change_after_auth)
    monkeypatch.setattr(os, "pipe", count_pipe)
    with pytest.raises(ResearchError, match="credential file must remain private"):
        adapter.run("rejected continuation", workspace, first.session_id)
    # Auth subprocess creates its own pipes; no additional liveness pipes exist.
    auth_pipe_count = len(pipe_calls)
    assert auth_pipe_count == 3
    assert (tmp_path / "state/state.json").read_bytes() == previous
    assert not (tmp_path / "state/turn-0002").exists()
    assert len((profile.directory / "calls.jsonl").read_text().splitlines()) == 1
    key.chmod(0o600)
    monkeypatch.setattr(SubscriptionProfile, "environment", original_environment)
    second = adapter.run("recovered continuation", workspace, first.session_id)
    assert not second.is_error
    assert load_state(tmp_path / "state/state.json")["generation"] == 2


@pytest.mark.parametrize("filename", ["./adc.json", "~/adc.json"])
def test_vertex_relative_adc_is_resolved_before_auth_and_frozen_plan(
    tmp_path, monkeypatch, filename
):
    original = fixture_profile(tmp_path / "fixture", "claude", vertex=True)
    adc = tmp_path / "adc.json"
    adc.write_text("fixture-adc-not-an-actual-credential")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    plan = Plan(
        goal="fixture Vertex paths",
        tools="none",
        timeout_s=4,
        idle_timeout_s=3,
        workers=[
            Worker(
                id="researcher",
                backend="claude",
                profile=str(original.directory),
                binary=original.binary,
                model=original.model,
                prompt="fixture request",
                auth_mode="vertex",
                vertex={"project": "fixture-project", "adc_file": filename},
            )
        ],
    )
    root = tmp_path / "research"
    assert run_plan(plan, root, 1)["complete"]
    saved = load_state(root / "plan.json")
    assert saved["plan"]["workers"][0]["vertex"]["adc_file"] == str(adc)
    call = json.loads((original.directory / "calls.jsonl").read_text())
    assert call["env"]["GOOGLE_APPLICATION_CREDENTIALS"] == str(adc)
    assert "fixture-adc-not-an-actual-credential" not in json.dumps(saved)
    assert run_plan(Plan.model_validate(saved["plan"]), root, 1)["complete"]
    assert len((original.directory / "calls.jsonl").read_text().splitlines()) == 1


def test_legacy_subscription_plan_remains_observation_only_after_upgrade(tmp_path):
    plan = plan_fixture(tmp_path)
    root = tmp_path / "research"
    run_plan(plan, root, 2)
    saved = load_state(root / "plan.json")
    assert all("auth_mode" not in w for w in saved["plan"]["workers"])
    legacy_plan = Plan.model_validate(saved["plan"])
    assert run_plan(legacy_plan, root, 2)["complete"]
    assert all(
        len((tmp_path / b / "profile/calls.jsonl").read_text().splitlines()) == 1
        for b in ["codex", "claude"]
    )


def test_api_mode_rejects_world_readable_credentials_and_header_injection(tmp_path):
    original = fixture_profile(tmp_path, "claude")
    key = tmp_path / "key"
    key.write_text("fixture-api-key")
    key.chmod(0o644)
    with pytest.raises(ResearchError, match="private"):
        SubscriptionProfile(
            "claude",
            original.directory,
            original.binary,
            original.model,
            auth_mode="api-key",
            api_key_file=str(key),
        )
    key.chmod(0o600)
    with pytest.raises(ResearchError, match="workspace header"):
        SubscriptionProfile(
            "claude",
            original.directory,
            original.binary,
            original.model,
            auth_mode="api-key",
            api_key_file=str(key),
            workspace_id="fixture\nInjected: header",
        )


def test_provider_cost_is_reported_and_missing_cost_is_not_called_free(tmp_path):
    adapter, workspace = harness(tmp_path / "reported", "claude", cost=1.25)
    result = adapter.run("fixture", workspace)
    assert result.cost_usd == 1.25
    assert load_state(tmp_path / "reported/state/state.json")["provider_cost_status"] == "reported"
    adapter, workspace = harness(tmp_path / "unknown", "claude")
    adapter.run("fixture", workspace)
    state = load_state(tmp_path / "unknown/state/state.json")
    assert state["provider_cost_usd"] is None
    assert state["provider_cost_status"] == "unavailable"


def test_profile_and_workspace_exclusion(tmp_path):
    adapter, ws = harness(tmp_path)
    for lock in [
        adapter.profile.directory / ".outerloop-research.lock",
        ws.parent / f".{ws.name}.outerloop-research.lock",
    ]:
        with exclusive(lock), pytest.raises(ResearchError, match="already in use"):
            adapter.run("brief", ws)
    assert not (adapter.profile.directory / "calls.jsonl").exists()


def test_crash_record_and_changed_binding_refuse_relaunch(tmp_path):
    adapter, ws = harness(tmp_path)
    result = adapter.run("brief", ws)
    state = load_state(tmp_path / "state" / "state.json")
    state["status"] = "running"
    save_json(tmp_path / "state" / "state.json", state)
    with pytest.raises(ResearchError, match="unfinished"):
        adapter.run("brief", ws, result.session_id)
    state["status"] = "completed"
    state["identity"]["model"] = "different-model"
    save_json(tmp_path / "state" / "state.json", state)
    with pytest.raises(ResearchError, match="binding changed"):
        adapter.run("brief", ws, result.session_id)
    assert len((adapter.profile.directory / "calls.jsonl").read_text().splitlines()) == 1


def plan_fixture(tmp_path: Path, **options) -> Plan:
    profiles = [fixture_profile(tmp_path / b, b, **options) for b in ("claude", "codex")]
    return Plan(
        goal="Compare two algorithms",
        timeout_s=3,
        idle_timeout_s=2,
        workers=[
            Worker(
                id=p.backend,
                backend=p.backend,
                profile=str(p.directory),
                binary=p.binary,
                model=p.model,
                prompt=f"Analyze {p.backend} assignment",
            )
            for p in profiles
        ],
    )


def test_goal_concurrency_retry_and_native_followup(tmp_path):
    plan = plan_fixture(tmp_path, delay=0.2)
    root = tmp_path / "research"
    result = run_plan(plan, root, 2)
    assert result["complete"]
    a, b = result["workers"]
    assert max(a["started_at"], b["started_at"]) < min(a["finished_at"], b["finished_at"])
    assert run_plan(plan, root, 2) == result  # no second model call
    followup = resume(root, "codex", "check the result", "verify-1")
    assert followup["complete"]
    assert resume(root, "codex", "check the result", "verify-1") == followup
    with pytest.raises(ResearchError, match="different prompt"):
        resume(root, "codex", "other question", "verify-1")
    calls = (tmp_path / "codex" / "profile" / "calls.jsonl").read_text().splitlines()
    assert len(calls) == 2


def test_same_account_alias_and_unknown_fields_fail_before_execution(tmp_path):
    plan = plan_fixture(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(Path(plan.workers[0].profile), target_is_directory=True)
    plan.workers[1].profile = str(alias)
    with pytest.raises(ResearchError, match="distinct native profile"):
        run_plan(plan, tmp_path / "research", 2)
    assert not (tmp_path / "research" / "plan.json").exists()
    with pytest.raises(ValueError):
        Plan.model_validate({"goal": "x", "workers": [], "api_key": "not-supported"})


@pytest.mark.parametrize("tools", ["none", "read"])
def test_fresh_and_resume_commands_keep_auth_and_tool_policy(tmp_path, tools):
    for backend in ("claude", "codex"):
        profile = fixture_profile(tmp_path / backend, backend)
        for sid in ("", SESSION):
            args = native_command(profile, tmp_path, sid, tools, 3)
            assert "--bare" not in args and "--ephemeral" not in args
            assert "--no-session-persistence" not in args
            if backend == "codex":
                assert 'forced_login_method="chatgpt"' in args
                assert 'model_reasoning_effort="max"' in args
                assert f"web_search={json.dumps('live' if tools == 'read' else 'disabled')}" in args
                assert f"features.code_mode_host={json.dumps(tools == 'read')}" in args
                assert "features.code_mode=false" in args
                assert "features.shell_tool=false" in args
                assert "features.unified_exec=false" in args
                assert "features.image_generation=false" in args
                assert "--sandbox" not in args if sid else "--sandbox" in args
            else:
                expected = "Read,Glob,Grep,WebSearch,WebFetch" if tools == "read" else ""
                assert args[args.index("--tools") + 1] == expected
                assert args[args.index("--effort") + 1] == "max"


@pytest.mark.parametrize("index", range(20))
def test_controller_death_preserves_uncertain_state_and_refuses_replay(tmp_path, index):
    adapter, ws = harness(tmp_path, hang=True)
    driver = tmp_path / "driver.py"
    driver.write_text(
        "from pathlib import Path\n"
        "from outerloop.subscription import SubscriptionProfile, SubscriptionHarness\n"
        f"p=SubscriptionProfile('codex', Path({str(adapter.profile.directory)!r}), "
        f"{adapter.profile.binary!r}, 'fixture-model')\n"
        f"h=SubscriptionHarness(p, Path({str(adapter.state_dir)!r}), 30, 20)\n"
        f"h.run('goal', Path({str(ws)!r}))\n"
    )
    controller = subprocess.Popen([sys.executable, str(driver)])
    state = {}
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            path = adapter.state_dir / "state.json"
            if path.exists():
                state = load_state(path)
                if state.get("session_id"):
                    break
            time.sleep(0.01)
        assert state.get("session_id") == SESSION
        controller.kill()
        controller.wait(timeout=3)
        native_pid = int((adapter.profile.directory / "native.pid").read_text())
        deadline = time.monotonic() + 3
        while process_alive(native_pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not process_alive(native_pid)
        with pytest.raises(ResearchError, match=r"already in use|unfinished"):
            adapter.run("do not duplicate", ws, SESSION)
    finally:
        controller.kill() if controller.poll() is None else None
        controller.wait(timeout=3)
        if state.get("pid") and process_alive(state["pid"]):
            os.killpg(state["pid"], signal.SIGKILL)


@pytest.mark.parametrize("index", range(20))
def test_supervisor_death_cleans_native_writer(tmp_path, index):
    adapter, ws = harness(tmp_path, timeout=3, idle=2, child=True, hang=True)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(adapter.run, "goal", ws)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            path = adapter.state_dir / "state.json"
            if path.exists() and load_state(path).get("session_id"):
                break
            time.sleep(0.01)
        state = load_state(path)
        assert state["session_id"] == SESSION
        os.kill(state["pid"], signal.SIGKILL)
        result = future.result(timeout=3)
    assert result.is_error
    for name in ("native.pid", "child.pid"):
        assert not process_alive(int((adapter.profile.directory / name).read_text()))


def test_lost_followup_ack_is_pending_and_never_reexecuted(tmp_path):
    import hashlib

    from outerloop.research_cli import snapshot

    plan = plan_fixture(tmp_path)
    root = tmp_path / "research"
    run_plan(plan, root, 2)
    requests = root / "codex" / "requests"
    requests.mkdir()
    save_json(
        requests / "follow.json",
        {
            "schema": 1,
            "status": "accepted",
            "at": time.time(),
            "request_id": "follow",
            "prompt_sha256": hashlib.sha256(b"follow-up").hexdigest(),
        },
    )
    assert not snapshot(root)["complete"]
    assert not resume(root, "codex", "follow-up", "follow")["complete"]
    with pytest.raises(ResearchError, match="reconciliation"):
        resume(root, "codex", "another", "another")
    assert len((tmp_path / "codex/profile/calls.jsonl").read_text().splitlines()) == 1


def test_default_claude_profile_preserves_native_keychain_namespace(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USER", "native-user")
    (tmp_path / ".claude").mkdir()
    profile = SubscriptionProfile("claude", tmp_path / ".claude", "claude", "test-model")
    env = profile.environment(tmp_path / "private-home")
    assert "CLAUDE_CONFIG_DIR" not in env
    assert env["HOME"] == str(tmp_path) and env["USER"] == "native-user"


def test_followups_on_distinct_profiles_can_overlap(tmp_path):
    plan = plan_fixture(tmp_path, delay=0.3)
    root = tmp_path / "research"
    run_plan(plan, root, 2)
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda w: resume(root, w, "continue", "follow"), ["claude", "codex"]))
    a = load_state(root / "claude/state/state.json")
    b = load_state(root / "codex/state/state.json")
    assert max(a["started_at"], b["started_at"]) < min(a["finished_at"], b["finished_at"])


def test_permission_error_requires_no_live_group_members(monkeypatch):
    from outerloop.subscription import kill_group

    def denied(*args):
        raise PermissionError

    monkeypatch.setattr(os, "killpg", denied)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, "123 Z\n", "")
    )
    kill_group(123)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, "123 S\n", "")
    )
    with pytest.raises(ResearchError, match="cleanup"):
        kill_group(123)


def test_cli_interrupt_cleans_parallel_workers(tmp_path):
    plan = plan_fixture(tmp_path, child=True, hang=True)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(plan.model_dump_json())
    root = tmp_path / "research"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "outerloop.research_cli",
            "run",
            str(manifest),
            "--root",
            str(root),
            "--parallel",
            "2",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 5
        paths = [tmp_path / b / "profile/child.pid" for b in ("claude", "codex")]
        while not all(p.exists() for p in paths) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert all(p.exists() for p in paths)
        process.send_signal(signal.SIGINT)
        process.communicate(timeout=5)
        assert process.returncode == 130
        for path in paths:
            assert not process_alive(int(path.read_text()))
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def test_absent_legacy_state_starts_fresh_but_unknown_schema_is_refused(tmp_path):
    adapter, ws = harness(tmp_path)
    adapter.state_dir.mkdir()
    path = adapter.state_dir / "state.json"
    save_json(path, {"schema": 999, "session_id": SESSION})
    with pytest.raises(ResearchError, match="unsupported"):
        adapter.run("brief", ws)
    assert not (adapter.profile.directory / "calls.jsonl").exists()
    path.unlink()
    assert not adapter.run("brief", ws).is_error


@pytest.mark.parametrize("index", range(20))
def test_supervisor_keeps_watching_after_native_leader_exit(tmp_path, index):
    import select

    import outerloop._subscription_process as supervisor

    profile = fixture_profile(tmp_path, child=True)
    watch_read, watch_write = os.pipe()
    done_read, done_write = os.pipe()
    lock = os.open(tmp_path / "lock", os.O_CREAT | os.O_RDWR, 0o600)
    process = subprocess.Popen(
        [
            sys.executable,
            supervisor.__file__,
            str(watch_read),
            str(done_write),
            str(lock),
            *native_command(profile, tmp_path, "", "none", 1),
        ],
        env=profile.environment(tmp_path),
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        pass_fds=(watch_read, done_write, lock),
    )
    os.close(watch_read)
    os.close(done_write)
    try:
        assert process.stdin is not None
        process.stdin.write(b"brief")
        process.stdin.close()
        assert select.select([done_read], [], [], 5)[0]
        assert os.read(done_read, 32) == b"0\n"
        # The controller has not acknowledged completion. The old supervisor
        # exited here, leaving nobody to observe controller loss.
        assert process.poll() is None
        child = int((profile.directory / "child.pid").read_text())
        assert process_alive(child)
        os.close(watch_write)  # equivalent EOF to SIGKILL of the sole controller
        watch_write = -1
        process.wait(timeout=5)
        assert not process_alive(child)
    finally:
        if watch_write >= 0:
            os.close(watch_write)
        os.close(done_read)
        os.close(lock)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def test_snapshot_and_acceptance_cannot_publish_mixed_time_completion(tmp_path, monkeypatch):
    import contextlib
    import threading

    from outerloop import research_cli

    plan = plan_fixture(tmp_path)
    root = tmp_path / "research"
    run_plan(plan, root, 2)
    read_a, release_reader = threading.Event(), threading.Event()
    accepting, model_entered, finish_model = (threading.Event() for _ in range(3))
    original_load, original_lock = research_cli.load_state, research_cli.exclusive
    original_harness = research_cli._harness

    def paused_read(path):
        if (
            threading.current_thread().name.startswith("snapshot-reader")
            and path == root / "codex/state/state.json"
        ):
            read_a.set()
            assert release_reader.wait(5)
        return original_load(path)

    @contextlib.contextmanager
    def observed_lock(path, **kwargs):
        if threading.current_thread().name.startswith("follow") and path.name == "snapshot.lock":
            accepting.set()
        with original_lock(path, **kwargs) as fd:
            yield fd

    def paused_harness(*args):
        delegate = original_harness(*args)

        class Paused:
            def run(self, *a, **kw):
                model_entered.set()
                assert finish_model.wait(5)
                return delegate.run(*a, **kw)

        return Paused()

    monkeypatch.setattr(research_cli, "load_state", paused_read)
    monkeypatch.setattr(research_cli, "exclusive", observed_lock)
    monkeypatch.setattr(research_cli, "_harness", paused_harness)
    with (
        ThreadPoolExecutor(1, thread_name_prefix="snapshot-reader") as reader,
        ThreadPoolExecutor(1, thread_name_prefix="follow") as follower,
    ):
        try:
            observation = reader.submit(research_cli.snapshot, root)
            assert read_a.wait(5)
            follow = follower.submit(resume, root, "claude", "continue", "new")
            assert accepting.wait(5)
            assert not model_entered.is_set()
            assert not (root / "claude/requests/new.json").exists()
            release_reader.set()
            assert observation.result(timeout=5)["complete"]
            assert model_entered.wait(5)
            assert not load_state(root / "result.json")["complete"]
            assert not research_cli.snapshot(root)["complete"]
            finish_model.set()
            assert follow.result(timeout=5)["complete"]
        finally:
            release_reader.set()
            finish_model.set()
