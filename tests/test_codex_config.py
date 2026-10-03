"""OUTERLOOP_CODEX_CONFIG: the fleet's codex `-c` overrides, from the
environment to the climb, and from the tick to every climb and wake job."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import outerloop.attempt as climb_mod
from outerloop.compute import CommandResult, SlurmCompute
from outerloop.role_runner import codex_config_args, codex_config_entries, codex_config_from_text

NOW = 1_000_000.0


# ------------------------------------------------------------------ parsing


def test_environment_form_splits_on_semicolons_and_newlines() -> None:
    raw = 'use_legacy_landlock=true; model_providers.local.base_url="https://x.invalid/v1"\n'
    raw += "tools.allowed=[1, 2]; ;"
    assert codex_config_from_text(raw) == (
        "use_legacy_landlock=true",
        'model_providers.local.base_url="https://x.invalid/v1"',
        "tools.allowed=[1, 2]",  # a TOML value keeps its commas
    )
    assert codex_config_from_text("") == ()
    assert codex_config_args(("a=1", "b.c=2")) == ("-c", "a=1", "-c", "b.c=2")


@pytest.mark.parametrize("entry", ["novalue", "=x", "-c=1", "--yolo=1", "a b=1", ".a=1"])
def test_malformed_entries_are_refused_with_their_source(entry: str) -> None:
    with pytest.raises(ValueError, match="OUTERLOOP_CODEX_CONFIG"):
        codex_config_from_text(f"ok=1;{entry}")
    with pytest.raises(ValueError, match="--codex-config"):
        codex_config_entries([entry])


def test_value_may_hold_equals_signs_and_spaces() -> None:
    assert codex_config_entries([" a.b = x=y z "]) == ("a.b= x=y z",)


# --------------------------------------------------------------- the climb


@pytest.fixture
def climb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A fresh climb's command line with every side effect stubbed: returns the
    kwargs each build_harness call received."""
    pat = tmp_path / "pat"
    pat.write_text("ghp_x\n")
    pat.chmod(0o600)
    key = tmp_path / "codex_key"
    key.write_text("sk-codex-author\n")
    key.chmod(0o600)
    image = tmp_path / "img.sif"
    image.write_text("")
    (tmp_path / "state").mkdir()
    seen: dict[str, Any] = {"harness": []}

    def fake_build(api_key: str, spec: Any, **kwargs: Any) -> object:
        seen["harness"].append(kwargs)
        return object()

    def fake_live(**kwargs: Any) -> climb_mod.AttemptOutcome:
        seen["live"] = kwargs
        return climb_mod.AttemptOutcome(run_id="r", outcome="no-improvement")

    monkeypatch.setattr(climb_mod, "arm_sigterm_containment", lambda: None)
    monkeypatch.setattr(climb_mod, "build_harness", fake_build)
    monkeypatch.setattr(climb_mod, "live_attempt", fake_live)
    monkeypatch.setenv("OUTERLOOP_CODEX_KEY_FILE", str(key))
    monkeypatch.delenv("OUTERLOOP_CODEX_CONFIG", raising=False)
    seen["argv"] = [
        "climb",
        "--target",
        "o/r",
        "--benchmark",
        "b",
        "--run-root",
        str(tmp_path / "state"),
        "--image",
        str(image),
        "--pat-file",
        str(pat),
        "--panel",
        "",
        "--min-free-gb",
        "0",
        "--author-backend",
        "codex",
        "--model",
        "gpt-test",
    ]
    return seen


def test_the_environment_supplies_the_default(climb, monkeypatch) -> None:
    monkeypatch.setenv(
        "OUTERLOOP_CODEX_CONFIG", "use_legacy_landlock=true;model_reasoning_effort=high"
    )
    monkeypatch.setattr("sys.argv", climb["argv"])
    assert climb_mod.main() == 0
    (author,) = climb["harness"]
    assert author["backend"] == "codex"
    assert author["codex_extra_args"] == (
        "-c",
        "use_legacy_landlock=true",
        "-c",
        "model_reasoning_effort=high",
    )


def test_flags_replace_the_environment(climb, monkeypatch) -> None:
    monkeypatch.setenv("OUTERLOOP_CODEX_CONFIG", "from_env=1")
    monkeypatch.setattr("sys.argv", [*climb["argv"], "--codex-config", "from_flag=2"])
    assert climb_mod.main() == 0
    assert climb["harness"][0]["codex_extra_args"] == ("-c", "from_flag=2")


def test_a_malformed_value_refuses_a_codex_climb(climb, monkeypatch, capsys) -> None:
    monkeypatch.setenv("OUTERLOOP_CODEX_CONFIG", "use_legacy_landlock")
    monkeypatch.setattr("sys.argv", climb["argv"])
    with pytest.raises(SystemExit):
        climb_mod.main()
    assert "OUTERLOOP_CODEX_CONFIG" in capsys.readouterr().err
    assert climb["harness"] == []  # refused before any session was built


def test_a_malformed_value_does_not_stop_a_claude_climb(climb, monkeypatch, tmp_path) -> None:
    key = tmp_path / "claude_key"
    key.write_text("sk-ant-author\n")
    key.chmod(0o600)
    monkeypatch.setenv("OUTERLOOP_CLAUDE_KEY_FILE", str(key))
    monkeypatch.setenv("OUTERLOOP_CODEX_CONFIG", "use_legacy_landlock")
    argv = climb["argv"][: climb["argv"].index("--author-backend")]
    monkeypatch.setattr("sys.argv", [*argv, "--author-backend", "claude"])
    assert climb_mod.main() == 0
    assert climb["harness"][0]["backend"] == "claude"


def test_a_codex_wake_on_a_malformed_value_hands_its_lease_back(
    tmp_path, monkeypatch, capsys
) -> None:
    from outerloop.runstate import RunRecord, acquire_lease, read_lease, save_record

    run_id = "b-wake"
    save_record(
        tmp_path,
        RunRecord(
            run_id=run_id,
            target="o/r",
            task_title="t",
            benchmark="b",
            state="parked",
            deadline=NOW + 3600,
            author_backend="codex",
            author_model="gpt-test",
        ),
        NOW,
    )
    pat = tmp_path / "pat"
    pat.write_text("ghp_x\n")
    pat.chmod(0o600)
    image = tmp_path / "img.sif"
    image.write_text("")
    assert acquire_lease(tmp_path, run_id, "wake-job:1", "1", NOW)
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.delenv("OUTERLOOP_DISPATCH_WAKE", raising=False)
    monkeypatch.setenv("OUTERLOOP_CODEX_CONFIG", "not-an-override")
    monkeypatch.setattr(climb_mod, "arm_sigterm_containment", lambda: None)
    argv = ["climb", "--resume", run_id, "--run-root", str(tmp_path), "--image", str(image)]
    monkeypatch.setattr("sys.argv", [*argv, "--pat-file", str(pat), "--panel", ""])
    with pytest.raises(SystemExit):
        climb_mod.main()
    assert "OUTERLOOP_CODEX_CONFIG" in capsys.readouterr().err
    assert read_lease(tmp_path, run_id) is None  # released, not stranded until the TTL


# ------------------------------------------------------------------ the tick


def _contract():
    from outerloop.contract import load_contract

    return load_contract(
        """
benchmarks:
  - {name: tsp, command: c, metric: m, direction: min}
budgets: {gpu_hours_per_run: 1, runs_per_week: 3}
scope: {allowed: [src/]}
roadmap: docs/roadmap.md
""",
        "org/pilot",
    )


def _spec(tmp_path: Path, **kw: Any):
    from outerloop.tick import ServiceSpec

    panel_key = tmp_path / "verifier_key"
    panel_key.write_text("sk-verifier-key")
    panel_key.chmod(0o600)
    return ServiceSpec(
        target="org/pilot",
        account="acct",
        partition="part",
        run_root=tmp_path,
        image="img.sif",
        home=tmp_path,
        bot_login="bot",
        panel_key_file=str(panel_key),
        **kw,
    )


def _sbatch_recorder() -> tuple[SlurmCompute, list[str]]:
    submitted: list[str] = []

    def runner(argv, timeout_s):
        if argv[0] == "sbatch":
            submitted.append(" ".join(argv))
            return CommandResult(0, "123\n", "")
        return CommandResult(0, "RUNNING\n", "")

    return SlurmCompute(runner=runner), submitted


def test_the_tick_forwards_the_codex_config_to_climb_jobs(tmp_path, monkeypatch) -> None:
    from outerloop.tick import service_self_initiated

    monkeypatch.delenv("OUTERLOOP_AUTHOR_BACKEND", raising=False)
    compute, submitted = _sbatch_recorder()
    spec = _spec(tmp_path, codex_config="use_legacy_landlock=true; model_reasoning_effort=high")
    assert service_self_initiated(tmp_path, compute, spec, _contract(), NOW) == ("tsp", "123")
    (job,) = submitted
    assert "--codex-config use_legacy_landlock=true" in job
    assert "--codex-config model_reasoning_effort=high" in job


def test_the_tick_forwards_the_codex_config_to_wake_jobs(tmp_path, monkeypatch) -> None:
    from outerloop.runstate import RunRecord
    from outerloop.tick import JobWakeDispatcher

    monkeypatch.setattr(
        "outerloop.tick._flight_command", lambda home, name, now, argv: " ".join(argv)
    )
    compute, submitted = _sbatch_recorder()
    record = RunRecord(
        run_id="tsp-1",
        target="org/pilot",
        task_title="t",
        benchmark="tsp",
        state="parked",
        stage={"afterany": "afterany:501"},
    )
    spec = _spec(tmp_path, codex_config="use_legacy_landlock=true")
    JobWakeDispatcher(compute, spec, now=NOW).dispatch(record, "eval done")
    assert "--resume tsp-1" in submitted[0]
    assert "--codex-config use_legacy_landlock=true" in submitted[0]
    # nothing configured: no flag, so the job keeps its own environment's default
    submitted.clear()
    JobWakeDispatcher(compute, _spec(tmp_path), now=NOW).dispatch(record, "eval done")
    assert "--codex-config" not in submitted[0]


def test_a_malformed_value_is_not_forwarded_and_blocks_a_codex_fleet(tmp_path, monkeypatch) -> None:
    from outerloop.tick import _author_config_error, _codex_config_argv, service_self_initiated

    bad = _spec(tmp_path, codex_config="use_legacy_landlock")
    assert _codex_config_argv(bad) == []
    monkeypatch.setenv("OUTERLOOP_AUTHOR_BACKEND", "codex")
    monkeypatch.setenv("OUTERLOOP_AUTHOR_MODEL", "gpt-test")
    assert "OUTERLOOP_CODEX_CONFIG" in _author_config_error(bad)
    compute, submitted = _sbatch_recorder()
    assert service_self_initiated(tmp_path, compute, bad, _contract(), NOW) is None
    assert submitted == []  # refused on the tick host: nothing queued to fail
    # a claude fleet is not blocked by a codex setting it never uses
    monkeypatch.setenv("OUTERLOOP_AUTHOR_BACKEND", "claude")
    monkeypatch.delenv("OUTERLOOP_AUTHOR_MODEL", raising=False)
    assert _author_config_error(bad) == ""


def test_the_service_spec_reads_the_environment(tmp_path, monkeypatch) -> None:
    from outerloop.tick import _service_spec_from_env

    image = tmp_path / "img.sif"
    image.write_text("")
    pat = tmp_path / "pat"
    pat.write_text("ghp_x")
    pat.chmod(0o600)
    monkeypatch.setenv("OUTERLOOP_PAT_FILE", str(pat))
    monkeypatch.setenv("OUTERLOOP_HOME", str(tmp_path))
    monkeypatch.setenv("OUTERLOOP_TARGET", "org/pilot")
    monkeypatch.setenv("OUTERLOOP_IMAGE", str(image))
    monkeypatch.setenv("OUTERLOOP_CODEX_CONFIG", "a=1;b=2")
    _, spec = _service_spec_from_env(tmp_path)
    assert spec is not None and spec.codex_config == "a=1;b=2"


def test_wake_spec_round_trips_and_reads_legacy_files(tmp_path) -> None:
    """Upgrading: wake-spec.json gains `codex_config`. A spec an older tick
    wrote (no field) still loads with the default, and a newer spec's extra
    field is ignored by a reader that does not know it."""
    from outerloop.tick import WAKE_SPEC_NAME, load_wake_spec, write_wake_spec

    write_wake_spec(tmp_path, _spec(tmp_path, codex_config="a=1"))
    loaded = load_wake_spec(tmp_path)
    assert loaded is not None and loaded.codex_config == "a=1"
    legacy = json.loads((tmp_path / WAKE_SPEC_NAME).read_text())
    del legacy["codex_config"]
    legacy["field_from_a_newer_kernel"] = "x"
    (tmp_path / WAKE_SPEC_NAME).write_text(json.dumps(legacy))
    loaded = load_wake_spec(tmp_path)
    assert loaded is not None and loaded.codex_config == ""
