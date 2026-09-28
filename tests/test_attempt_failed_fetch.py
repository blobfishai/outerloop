"""A failed canonical fetch cannot authorize a changed measurement baseline."""

from dataclasses import replace

import pytest

from outerloop import attempt
from outerloop.dispatch import snapshot_tree
from outerloop.github import GitError, Workspace
from outerloop.orchestrator import AttemptResult
from outerloop.roles import author_spec
from outerloop.runstate import load_record, save_record
from test_attempt import (
    CONTRACT_DISPATCH,
    CONTRACT_LINES_DISPATCH,
    CommentingGitHub,
    NoAuth,
    ScriptedHarness,
    _fake_dispatch,
    _git,
    _write_parked_candidate,
)


@pytest.mark.parametrize("retarget_origin", [False, True])
def test_failed_fetch_keeps_the_recorded_line_base(tmp_path, monkeypatch, retarget_origin):
    state, run_id = _write_parked_candidate(
        tmp_path, monkeypatch, contract=CONTRACT_DISPATCH, agent_id="agent-02"
    )
    record = load_record(state, run_id)
    workspace = state / "runs" / run_id / "ws"
    old_main = str(record.stage["base_sha"])
    (workspace / "eval-cache.tmp").unlink()
    # Canonical main enables lines before this research run launches.
    (workspace / ".outerloop.yaml").write_text(CONTRACT_LINES_DISPATCH)
    (workspace / "src/pilot/solvers/tsp.py").write_text(
        'def solve(): return "shared improvement"\n'
    )
    _git(workspace, "add", ".outerloop.yaml", "src/pilot/solvers/tsp.py")
    _git(workspace, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "enable lines")
    main = _git(workspace, "rev-parse", "HEAD").strip()
    _git(workspace, "push", "origin", "HEAD:main")
    _git(workspace, "update-ref", attempt.BASE_REF, main)
    # The measured research line contains prior work beyond main.
    (workspace / "src/pilot/solvers/tsp.py").write_text('def solve(): return "earlier line work"\n')
    _git(workspace, "add", "src/pilot/solvers/tsp.py")
    _git(
        workspace, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "line improvement"
    )
    line_tip = _git(workspace, "rev-parse", "HEAD").strip()
    _git(workspace, "update-ref", "refs/heads/agents/agent-02", line_tip)
    _git(workspace, "update-ref", attempt.LINE_HEAD_REF, line_tip)
    (workspace / "src/pilot/solvers/tsp.py").write_text(
        'def solve(): return "current experiment"\n'
    )
    snap = snapshot_tree(Workspace(root=workspace), line_tip, exclude=attempt.LINE_MEMORY_PATHS)
    save_record(
        state,
        replace(
            record,
            stage={
                **record.stage,
                "phase": "author-sleep",
                "base_sha": line_tip,
                "candidate_sha": snap.commit,
                "candidate_ref": snap.ref,
                "syscall_launches": [],
            },
        ),
        1_000_001.0,
    )
    assert _git(workspace, "rev-parse", attempt.BASE_REF).strip() == main
    # A cached remote-tracking ref is writable even when the launch ref is intact.
    if retarget_origin:
        _git(workspace, "update-ref", "refs/remotes/origin/main", old_main)

    def unavailable(self):
        raise GitError("simulated canonical fetch outage")

    observed = []

    def author_leg(config, contract, workspace, harness, measurer, base_sha, snapshot, **kw):
        observed.append(base_sha)
        # Existing fetch-outage behavior is unknown; this check is about whether
        # the new classification silently rewrites the measurement base first.
        assert kw["submit_preflight"]().status == "unknown"
        assert base_sha == line_tip
        assert load_record(state, run_id).stage["base_sha"] == line_tip
        return AttemptResult(outcome="no-improvement")

    monkeypatch.setattr(Workspace, "fetch_origin", unavailable)
    monkeypatch.setattr(attempt, "attempt_once", author_leg)
    outcome = attempt.resume_run(
        state,
        run_id,
        dispatch=_fake_dispatch(),
        github=CommentingGitHub(),  # type: ignore[arg-type]
        bot_auth=NoAuth(),
        now=1_000_100.0,
        harness=ScriptedHarness(edits={}),
        spec=author_spec(),
    )
    assert outcome.outcome == "no-improvement"
    assert observed == [line_tip]
