"""Live limits cover every kernel submission path and preserve contract caps."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from outerloop import cli, tick
from outerloop.attempt import _make_launcher, _park_run
from outerloop.compute import JobSpec
from outerloop.contract import load_contract
from outerloop.limits import effective_limits
from outerloop.measure import DispatchedMeasurer, DispatchSettings, Measure, MeasurementPending
from outerloop.operator_limits import (
    CapacityError,
    attempt_width,
    gpu_demand,
    read_limits,
    report,
    submit,
)
from outerloop.orchestrator import RunParked
from outerloop.runstate import RunRecord, load_record, save_record
from outerloop.syscall import Launch, SyscallRequest

TARGET = "owner/repo"
CONTRACT = """
benchmarks:
  - {name: bench, command: echo, metric: value, direction: min}
budgets: {gpu_hours_per_run: 1, runs_per_week: 20, max_active_attempts: 3}
scope: {allowed: [src/]}
steward: {allowed: [env/]}
roadmap: docs/roadmap.md
"""


def compute():
    backend = Mock()
    backend.gpu_jobs.return_value = []
    backend.has_lanes = False
    backend.status.return_value = "PENDING"
    backend.job_id_for_name.return_value = ""
    backend.submit.side_effect = (str(i) for i in range(100, 1000))
    return backend


def limits(root: Path, text: str = "[defaults]\nmax_gpus = 0\n") -> None:
    (root / "limits.toml").write_text(text)


def spec(gpus=1, array="") -> JobSpec:
    return JobSpec("job", "", "", 1, command="true", gpus=gpus, array=array)


def run(root: Path) -> Path:
    save_record(root, RunRecord("run", TARGET, "work", "running"), 1)
    return root / "runs" / "run"


@pytest.mark.parametrize("operator,expected", [(1, 1), (9, 3), (0, 0)])
def test_tighten_only(tmp_path, operator, expected):
    limits(tmp_path, f"[defaults]\nmax_active_attempts={operator}\nmax_gpus={operator}\n")
    assert attempt_width(tmp_path, TARGET, 3) == expected
    backend = compute()
    with pytest.raises(CapacityError):
        submit(tmp_path, TARGET, backend, spec(operator + 1))
    assert not backend.submit.called
    if expected:
        submit(tmp_path, TARGET, backend, spec(operator))


@pytest.mark.parametrize(
    "text",
    [
        "[",
        "max_gpus=2",
        "[defaults]\nmax_gpus=-1",
        "[defaults]\nmax_gpus=true",
        "[defaults]\nmax_gpus=1.5",
        '[targets."bad"]\nmax_gpus=1',
        "[defaults]\nmax_gpuss=2",
    ],
)
def test_malformed_fails_closed(tmp_path, text, caplog):
    limits(tmp_path, text)
    assert read_limits(tmp_path).error
    assert attempt_width(tmp_path, TARGET, 10) == 0
    backend = compute()
    with pytest.raises(CapacityError):
        submit(tmp_path, TARGET, backend, spec())
    assert not backend.submit.called
    assert "invalid limits.toml" in caplog.text


def test_lowering_is_live_and_never_cancels(tmp_path):
    backend = compute()
    submit(tmp_path, TARGET, backend, spec(4))
    limits(tmp_path)
    with pytest.raises(CapacityError):
        submit(tmp_path, TARGET, backend, spec())
    backend.cancel.assert_not_called()
    (tmp_path / "limits.toml").unlink()
    submit(tmp_path, TARGET, backend, spec())


@pytest.mark.parametrize("array,expected", [("", 2), ("0-9%3", 6), ("0-9", 20)])
def test_sweep_maximum_simultaneous_demand(array, expected):
    assert gpu_demand(spec(2, array)) == expected


@pytest.mark.parametrize("array", [1, 5])
def test_syscall_launch_path(tmp_path, array):
    directory = run(tmp_path)
    backend = compute()
    launcher = _make_launcher(
        DispatchSettings(backend, "", "", ""), directory, tmp_path / "workspace", "run", gpus=2
    )
    request = SyscallRequest(launches=(Launch("probe", "true", 1, array=array, concurrency=2),))
    demand = 2 * min(array, 2)
    limits(tmp_path, f"[defaults]\nmax_gpus={demand - 1}\n")
    with pytest.raises(CapacityError):
        launcher("sha", request)
    assert not backend.submit.called
    limits(tmp_path, f"[defaults]\nmax_gpus={demand}\n")
    assert launcher("sha", request) == "afterany:100"
    assert backend.submit.call_count == 1


def test_launch_batch_has_no_partial_admission(tmp_path):
    directory = run(tmp_path)
    backend = compute()
    launcher = _make_launcher(
        DispatchSettings(backend, "", "", ""), directory, tmp_path / "workspace", "run", gpus=1
    )
    limits(tmp_path, "[defaults]\nmax_gpus=1\n")
    with pytest.raises(CapacityError):
        launcher("sha", SyscallRequest(launches=(Launch("a", "x", 1), Launch("b", "x", 1))))
    backend.submit.assert_not_called()
    backend.cancel.assert_not_called()


def test_evaluation_waits_and_retry_cannot_bypass(tmp_path):
    directory = run(tmp_path)
    backend = compute()
    measurer = DispatchedMeasurer(backend, directory, tmp_path / "repo", "", "", "", 1)
    measure = Measure("candidate", "sha", "echo", "value", gpus=1)
    limits(tmp_path)
    for _ in range(2):
        with pytest.raises(MeasurementPending) as caught:
            measurer.results([measure])
        assert caught.value.capacity_wait
        assert not caught.value.job_ids
        assert not measurer._marker(measure)
        backend.submit.assert_not_called()
    limits(tmp_path, "[defaults]\nmax_gpus=1\n")
    with pytest.raises(MeasurementPending) as caught:
        measurer.results([measure])
    assert caught.value.job_ids == ("100",)
    assert not caught.value.capacity_wait
    assert backend.submit.call_count == 1


def test_capacity_park_does_not_exhaust_wake_retries(tmp_path):
    run(tmp_path)
    record = replace(load_record(tmp_path, "run"), wake_attempts=100)
    parked = RunParked(
        phase="candidate", afterany="", base_sha="base", seed=1, suite_seed=1, capacity_wait=True
    )
    _park_run(tmp_path, record, parked, "ref", 1, 1000, keep_wake_attempts=True)
    saved = load_record(tmp_path, "run")
    assert saved.wake_attempts == 0
    assert saved.stage["capacity_wait"] is True
    assert saved.deadline == 1060


@pytest.mark.parametrize("lane", ["self", "intake", "steward", "wake"])
def test_tick_submission_paths(tmp_path, monkeypatch, lane):
    backend = compute()
    contract = load_contract(CONTRACT, TARGET)
    service = tick.ServiceSpec(
        account="",
        partition="",
        run_root=tmp_path,
        image="",
        home=tmp_path,
        target=TARGET,
        panel="",
        steward_key_file="key",
    )
    monkeypatch.setattr(tick, "_flight_command", lambda *args: "true")
    monkeypatch.setattr(tick, "_author_config_error", lambda *args: "")
    monkeypatch.setattr(tick, "default_claude_model", lambda: "model")
    # Work specs currently request no GPUs; exercise GPU-bearing sessions too.
    monkeypatch.setattr(tick, "JobSpec", lambda **kw: JobSpec(**{**kw, "gpus": 1}))
    task = SimpleNamespace(number=1, benchmark="bench", body="work", title="work")
    monkeypatch.setattr("outerloop.intake.pick_issue", lambda *args: task)
    monkeypatch.setattr("outerloop.intake.issue_hypothesis", lambda *args: "work")
    monkeypatch.setattr("outerloop.steward.pick_steward_issue", lambda *args: task)
    monkeypatch.setattr("outerloop.steward.release_orphaned_claims", lambda *args, **kw: None)
    github = Mock()
    limits(tmp_path)

    def call():
        if lane == "self":
            return tick.service_self_initiated(tmp_path, backend, service, contract, 1000000)
        if lane == "intake":
            return tick.service_intake(tmp_path, github, backend, service, 1000, contract=contract)
        if lane == "steward":
            return tick.service_steward(
                tmp_path,
                github,
                backend,
                service,
                1000000,
                contract,
                effective_limits(contract.budgets),
            )
        return tick.JobWakeDispatcher(backend, service, 1000).dispatch(
            RunRecord("wake", TARGET, "work", "parked"), "retry"
        )

    if lane == "wake":
        with pytest.raises(CapacityError):
            call()
    else:
        assert call() is None
    backend.submit.assert_not_called()
    limits(tmp_path, "[defaults]\nmax_gpus=1\n")
    assert call() is not None
    assert backend.submit.call_count == 1


@pytest.mark.parametrize("lane", ["self", "intake", "steward"])
def test_zero_attempt_width_stops_fresh_lanes(tmp_path, monkeypatch, lane):
    contract = load_contract(CONTRACT, TARGET)
    service = tick.ServiceSpec(
        "", "", tmp_path, "", tmp_path, target=TARGET, panel="", steward_key_file="key"
    )
    backend = compute()
    limits(tmp_path, "[defaults]\nmax_active_attempts=0\n")
    if lane == "self":
        tick.service_self_initiated(tmp_path, backend, service, contract, 1000000)
    elif lane == "intake":
        tick.service_intake(tmp_path, Mock(), backend, service, 1000, contract=contract)
    else:
        tick.service_steward(
            tmp_path,
            Mock(),
            backend,
            service,
            1000000,
            contract,
            effective_limits(contract.budgets),
        )
    backend.submit.assert_not_called()


def test_dangling_control_file_fails_closed(tmp_path):
    (tmp_path / "limits.toml").symlink_to(tmp_path / "missing")
    assert read_limits(tmp_path).error


def test_missing_run_identity_cannot_bypass_target_limit(tmp_path):
    backend = compute()
    limits(tmp_path, '[targets."owner/repo"]\nmax_gpus=0\n')
    launcher = _make_launcher(
        DispatchSettings(backend, "", "", ""), tmp_path, tmp_path / "workspace", "missing", gpus=1
    )
    with pytest.raises(CapacityError, match="cannot attribute"):
        launcher("sha", SyscallRequest(launches=(Launch("probe", "true", 1),)))
    backend.submit.assert_not_called()


def test_scheduler_attribution_and_single_query(tmp_path):
    from outerloop.operator_limits import usage

    directory = run(tmp_path)
    # Legacy state requires only target, not new GPU fields or launch history.
    (directory / "state.json").write_text(json.dumps({"target": TARGET}))
    backend = compute()
    backend.gpu_jobs.return_value = [
        ("run-launch-probe", 2),
        ("eval-run-candidate", 1),
        ("operator-training", 100),
        ("wake-run", 1),
    ]
    assert usage(tmp_path, backend) == {TARGET: 4}
    backend.gpu_jobs.assert_called_once()
    limits(tmp_path, '[defaults]\nmax_gpus=8\n[targets."owner/repo"]\nmax_gpus=4\n')
    backend.gpu_jobs.reset_mock()
    with pytest.raises(CapacityError, match=TARGET):
        submit(tmp_path, TARGET, backend, spec())
    backend.gpu_jobs.assert_called_once()


def test_no_limits_or_no_gpu_ceiling_never_queries(tmp_path):
    backend = compute()
    backend.gpu_jobs.side_effect = RuntimeError("offline")
    submit(tmp_path, TARGET, backend, spec())
    assert list(tmp_path.iterdir()) == []
    limits(tmp_path, "[defaults]\nmax_active_attempts=1\n")
    submit(tmp_path, TARGET, backend, spec())
    backend.gpu_jobs.assert_not_called()
    limits(tmp_path)
    submit(tmp_path, TARGET, backend, spec(0))
    backend.gpu_jobs.assert_not_called()
    with pytest.raises(CapacityError, match=r"scheduler usage unavailable.*finite ceiling"):
        submit(tmp_path, TARGET, backend, spec())
    assert "scheduler usage unavailable" in report(tmp_path, backend)


def test_limits_cli_read_only(tmp_path, monkeypatch, capsys):
    backend = compute()
    limits(tmp_path)
    before = (tmp_path / "limits.toml").read_bytes()
    monkeypatch.setattr("outerloop.compute.compute_from_env", lambda: backend)
    monkeypatch.setattr(cli, "env_file_values", lambda **kw: {})
    assert cli.main(["limits", "--root", str(tmp_path)]) == 0
    assert "fleet running/pending GPUs=0" in capsys.readouterr().out
    assert (tmp_path / "limits.toml").read_bytes() == before
    assert len(list(tmp_path.iterdir())) == 1
    limits(tmp_path, "[")
    assert "invalid limits.toml" in report(tmp_path, backend)


def scheduler_job(job_id, name, state, gpus, *, parent=0, tasks=None, throttle=0):
    return {
        "job_id": job_id,
        "name": name,
        "job_state": [state],
        "tres_req_str": f"cpu=4,gres/gpu={gpus},gres/gpu:typed={gpus}",
        "tres_alloc_str": f"cpu=4,gres/gpu={gpus}" if state == "RUNNING" else None,
        "array_job_id": parent,
        "array_task_string": tasks,
        "array_max_tasks": {"set": bool(throttle), "number": throttle},
    }


def test_scheduler_arrays_and_requeued_jobs():
    from outerloop.compute import parse_gpu_jobs

    rows = [
        scheduler_job(10, "run-launch", "RUNNING", 2, parent=10, throttle=3),
        scheduler_job(11, "run-launch", "RUNNING", 2, parent=10, throttle=3),
        scheduler_job(12, "run-launch", "PENDING", 2, parent=10, tasks="2-9", throttle=3),
        scheduler_job(20, "run-requeued", "PENDING", 1),
        scheduler_job(30, "run-sweep", "PENDING", 4, parent=30, tasks="0-2,5", throttle=2),
        scheduler_job(40, "cpu", "RUNNING", 0),
        scheduler_job(41, "run-preempted", "PREEMPTED", 8),
        scheduler_job(42, "run-completed", "COMPLETED", 8),
        scheduler_job(50, "run-unthrottled", "PENDING", 1, parent=50, tasks="0-3"),
    ]
    assert parse_gpu_jobs(json.dumps({"jobs": rows})) == [
        ("run-launch", 6),
        ("run-requeued", 1),
        ("run-sweep", 8),
        ("run-unthrottled", 4),
    ]
    # All tasks instantiated individually after preemption: throttle still applies.
    rows = [scheduler_job(i, "run", "PENDING", 2, parent=10, throttle=1) for i in range(10, 15)]
    assert parse_gpu_jobs(json.dumps({"jobs": rows})) == [("run", 2)]


def test_slurm_gpu_query_is_one_user_scoped_call():
    from outerloop.compute import CommandResult, SlurmCompute

    output = json.dumps({"jobs": [scheduler_job(1, "run", "PENDING", 2)]})
    runner = Mock(return_value=CommandResult(0, output, ""))
    assert SlurmCompute(runner=runner).gpu_jobs() == [("run", 2)]
    runner.assert_called_once()
    assert runner.call_args.args[0] == ["squeue", "--me", "--json"]


def test_full_run_id_in_gpu_names(tmp_path):
    rid = "benchmark-with-a-long-name-20260101-agent-01"
    save_record(tmp_path, RunRecord(rid, TARGET, "work", "running"), 1)
    directory = tmp_path / "runs" / rid
    backend = compute()
    launcher = _make_launcher(
        DispatchSettings(backend, "", "", ""), directory, tmp_path / "ws", rid, gpus=1
    )
    launcher("sha", SyscallRequest(launches=(Launch("probe", "true", 1),)))
    assert rid in backend.submit.call_args.args[0].job_name
    measurer = DispatchedMeasurer(backend, directory, tmp_path / "repo", "", "", "", 1)
    measurer._dispatch(Measure("candidate", "sha", "echo", "value", gpus=1))
    assert rid in backend.submit.call_args.args[0].job_name


def test_control_plane_is_cpu_only(tmp_path):
    command = cli.StartPlan("slurm", tmp_path).command()
    assert not any("--gpu" in arg or "--gres" in arg for arg in command)
    for name in ("tick_chain.sbatch", "tick_resident.sh"):
        text = (Path(__file__).parents[1] / "scripts" / name).read_text()
        assert not re.search(r"--(?:gpus[\w-]*|gres)(?:=|\s)", text)


def test_local_reports_running_and_pending_array(tmp_path, monkeypatch):
    from outerloop.compute import LocalCompute

    monkeypatch.setenv("OUTERLOOP_ROOT", str(tmp_path))
    backend = LocalCompute()
    observed = []

    def execute(self, job, completed):
        observed.extend(self.gpu_jobs())
        for _ in range(8):
            completed()
        observed.extend(self.gpu_jobs())
        return "1"

    monkeypatch.setattr(LocalCompute, "_submit", execute)
    backend.submit(spec(2, "0-9%3"))
    assert observed == [("job", 6), ("job", 4)]
    assert backend.gpu_jobs() == []


def test_intake_reuses_tick_attempt_count(tmp_path, monkeypatch):
    contract = load_contract(CONTRACT, TARGET)
    service = tick.ServiceSpec("", "", tmp_path, "", tmp_path, target=TARGET)
    records = [RunRecord("active", TARGET, "work", "running")]
    pick = Mock(return_value=None)
    monkeypatch.setattr("outerloop.intake.pick_issue", pick)
    monkeypatch.setattr(tick, "list_runs", Mock(side_effect=AssertionError("extra scan")))
    limits(tmp_path, "[defaults]\nmax_active_attempts=1\n")
    tick.service_intake(
        tmp_path, Mock(), compute(), service, 1000, contract=contract, records=records
    )
    pick.assert_not_called()
    (tmp_path / "limits.toml").unlink()
    tick.service_intake(
        tmp_path, Mock(), compute(), service, 1000, contract=contract, records=records
    )
    pick.assert_called_once()  # absence preserves the lane's original behavior


def test_target_can_only_tighten_defaults_and_fleet_is_aggregate(tmp_path):
    backend = compute()
    run(tmp_path)
    backend.gpu_jobs.return_value = [("run-launch", 2)]
    limits(tmp_path, '[defaults]\nmax_gpus=2\n[targets."owner/repo"]\nmax_gpus=10\n')
    assert read_limits(tmp_path).value(TARGET, "max_gpus") == 2
    with pytest.raises(CapacityError, match="fleet"):
        submit(tmp_path, "owner/other", backend, spec())


def test_two_concurrent_admissions_accept_documented_race(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    barrier = Barrier(2)
    backend = compute()

    def snapshot():
        barrier.wait(timeout=5)
        return []

    backend.gpu_jobs.side_effect = snapshot
    limits(tmp_path, "[defaults]\nmax_gpus=1\n")
    with ThreadPoolExecutor(2) as workers:
        ids = list(workers.map(lambda _: submit(tmp_path, TARGET, backend, spec()), range(2)))
    assert len(ids) == 2
    assert set(p.name for p in tmp_path.iterdir()) == {"limits.toml"}
