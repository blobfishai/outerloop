"""A resumed author must be compared with the base its submission contains."""

from dataclasses import replace

import pytest

from outerloop import attempt
from outerloop.orchestrator import AttemptResult
from outerloop.roles import author_spec
from outerloop.runstate import load_record, save_record
from test_attempt import (
    CONTRACT_SYSCALLS,
    CommentingGitHub,
    NoAuth,
    ScriptedHarness,
    _fake_dispatch,
    _git,
    _write_parked_candidate,
)


@pytest.mark.parametrize(
    "sparse, change, fold",
    [
        (False, "visible", True),
        (True, "visible", True),
        (True, "visible", False),
        (True, "outside", False),
        (True, "root", True),
        (True, "contract", True),
    ],
)
def test_non_pr_author_wake_refreshes_measured_base(tmp_path, monkeypatch, sparse, change, fold):
    contract = CONTRACT_SYSCALLS
    if sparse:
        contract += "\nworkspace: {sparse: [src/pilot]}\n"
    state, run_id = _write_parked_candidate(tmp_path, monkeypatch, contract=contract)
    record = load_record(state, run_id)
    assert not record.pr_url
    save_record(
        state,
        replace(record, stage={**record.stage, "phase": "author-sleep", "syscall_launches": []}),
        1_000_001.0,
    )
    origin = tmp_path / f"origin-{run_id}.git"
    upstream = tmp_path / "upstream"
    _git(tmp_path, "clone", "-q", str(origin), str(upstream))
    path = (
        upstream
        / {
            "visible": "src/pilot/solvers/upstream.py",
            "outside": "other/asset.txt",
            "root": "shared.txt",
            "contract": ".outerloop.yaml",
        }[change]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contract + "\n# updated contract\n" if change == "contract" else "upstream\n")
    _git(upstream, "add", "-A")
    _git(upstream, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base improvement")
    _git(upstream, "push", "origin", "main")
    fresh_base = _git(upstream, "rev-parse", "HEAD").strip()
    expected = record.stage["base_sha"] if change == "outside" else fresh_base
    seen = []

    def author_leg(config, contract, workspace, harness, measurer, base_sha, snapshot, **kw):
        # A well-behaved author incorporates the fetched base before submitting.
        if fold:
            _git(workspace, "merge", "--ff-only", "origin/main")
        status = "ready" if fold or change == "outside" else "stale"
        assert kw["submit_preflight"]().status == status
        # Otherwise an independent base improvement is credited to this author.
        assert base_sha == expected
        assert load_record(state, run_id).stage["base_sha"] == expected
        if change == "contract":
            assert "# updated contract" in contract
        seen.append(base_sha)
        return AttemptResult(outcome="no-improvement")

    monkeypatch.setattr(attempt, "attempt_once", author_leg)
    result = attempt.resume_run(
        state,
        run_id,
        dispatch=_fake_dispatch(),
        github=CommentingGitHub(),  # type: ignore[arg-type]
        bot_auth=NoAuth(),
        now=1_000_100.0,
        harness=ScriptedHarness(edits={}),
        spec=author_spec(),
    )
    assert result.outcome == "no-improvement"
    assert seen == [expected]


def test_non_pr_wake_dispatches_the_current_measured_base(tmp_path, monkeypatch):
    from outerloop.measure import DispatchSettings, MeasurementPending

    state, run_id = _write_parked_candidate(tmp_path, monkeypatch, contract=CONTRACT_SYSCALLS)
    record = load_record(state, run_id)
    workspace = state / "runs" / run_id / "ws"
    (workspace / "eval-cache.tmp").unlink()
    save_record(
        state,
        replace(record, stage={**record.stage, "phase": "author-sleep", "syscall_launches": []}),
        1_000_001.0,
    )
    upstream = tmp_path / "upstream"
    _git(tmp_path, "clone", "-q", str(tmp_path / f"origin-{run_id}.git"), str(upstream))
    (upstream / "src/pilot/solvers/upstream.py").write_text("# base gained this improvement\n")
    _git(upstream, "add", "-A")
    _git(upstream, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base improvement")
    _git(upstream, "push", "origin", "main")
    fresh_base = _git(upstream, "rev-parse", "HEAD").strip()
    dispatched = []

    class Measurer:
        def results(self, measures):
            dispatched.extend(measures)
            raise MeasurementPending(("101", "102"))

    class FoldingHarness(ScriptedHarness):
        def run(self, brief_text, workspace, resume_session_id=None):
            assert resume_session_id == "s1"
            _git(workspace, "merge", "--ff-only", "origin/main")
            return super().run(brief_text, workspace, resume_session_id)

    monkeypatch.setattr(DispatchSettings, "measurer", lambda *a, **k: Measurer())
    result = attempt.resume_run(
        state,
        run_id,
        dispatch=_fake_dispatch(),
        github=CommentingGitHub(),  # type: ignore[arg-type]
        bot_auth=NoAuth(),
        now=1_000_100.0,
        harness=FoldingHarness(edits={}, submit=True),
        spec=author_spec(),
    )
    assert result.outcome == "parked"
    assert next(m.tree_sha for m in dispatched if m.name == "baseline") == fresh_base
    after = load_record(state, run_id)
    assert after.stage["phase"] == "candidate"
    assert after.stage["base_sha"] == fresh_base
    assert after.resume_session_id == "s1"

    # Main moving again cannot rewrite an already-dispatched pair on collection.
    (upstream / "src/pilot/solvers/upstream.py").write_text("# another independent improvement\n")
    _git(upstream, "add", "-A")
    _git(upstream, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "another move")
    _git(upstream, "push", "origin", "main")
    dispatched.clear()
    collected = attempt.resume_run(
        state,
        run_id,
        dispatch=_fake_dispatch(),
        github=CommentingGitHub(),  # type: ignore[arg-type]
        bot_auth=NoAuth(),
        now=1_000_200.0,
    )
    assert collected.outcome == "parked"
    assert next(m.tree_sha for m in dispatched if m.name == "baseline") == fresh_base
    assert load_record(state, run_id).stage["base_sha"] == fresh_base
