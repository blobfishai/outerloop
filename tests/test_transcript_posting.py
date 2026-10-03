"""What reaches GitHub: credentials rotated during a run are redacted from
everything posted, and OUTERLOOP_POST_TRANSCRIPTS=off keeps session text in
the run's local records."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from outerloop.github import (
    FileTokenProvider,
    GitHubClient,
    remember_secret,
    seen_secrets,
)
from outerloop.harness import SessionResult, redact
from outerloop.orchestrator import AttemptResult, RunConfig, pr_body
from outerloop.posting import (
    TRANSCRIPT_WITHHELD,
    panel_summary,
    transcripts_posted,
    withhold_session_text,
)

OLD = "ghp_oldTokenValue0123456789"
NEW = "ghp_newTokenValue9876543210"


def _token_file(path: Path, value: str) -> Path:
    path.write_text(value + "\n")
    path.chmod(0o600)
    return path


# ------------------------------------------------------------ redaction


def test_a_token_rotated_during_the_run_is_redacted(tmp_path: Path) -> None:
    pat = _token_file(tmp_path / "pat", OLD)
    provider = FileTokenProvider(pat)
    snapshot = (provider.token(),)  # what a run captures at start
    _token_file(pat, NEW)  # rotated mid-run; the next git or API call reads it
    assert provider.token() == NEW
    text = f"git said: fatal: could not read from https://x-access-token:{NEW}@host"
    assert NEW not in redact(text, snapshot)
    assert OLD not in redact(f"stale {OLD}", ())  # the first value is remembered too


def test_short_values_are_never_registered() -> None:
    remember_secret("short")
    assert "short" not in seen_secrets()
    assert redact("a short note", ()) == "a short note"


def test_a_secret_inside_another_is_never_left_half_visible() -> None:
    inner, outer = "abcdefgh12", "abcdefgh1234567"
    assert redact(f"x {outer} y", (inner, outer)) == "x [redacted] y"


class _Capture:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, Any] | None]] = []

    def __call__(self, request: urllib.request.Request) -> Any:
        data = request.data if isinstance(request.data, bytes) else None
        self.requests.append((request.get_method(), json.loads(data) if data else None))
        return {"html_url": "https://github.com/o/r/pull/1"}


def test_every_posted_text_passes_write_time_redaction(tmp_path: Path) -> None:
    """The client is the last point before GitHub: a body built by any path,
    with any (stale) secret set, still cannot carry a rotated credential."""
    pat = _token_file(tmp_path / "pat", OLD)
    capture = _Capture()
    client = GitHubClient(auth=FileTokenProvider(pat), transport=capture)
    _token_file(pat, NEW)
    client.comment("o/r", 3, f"report text {NEW} more")  # the request itself re-reads NEW
    client.create_pull("o/r", title=f"[agent] {NEW}", head="feat/x", base="main", body=NEW)
    (_, comment), (_, pull) = capture.requests
    assert comment == {"body": "report text [redacted] more"}
    assert pull is not None
    assert pull["title"] == "[agent] [redacted]" and pull["body"] == "[redacted]"
    assert (pull["head"], pull["base"]) == ("feat/x", "main")  # other fields untouched


def test_nested_review_comments_are_scrubbed_too() -> None:
    from outerloop.github import _redact_posted_text

    remember_secret(NEW)
    payload = {"event": "COMMENT", "comments": [{"path": "a.py", "body": f"see {NEW}"}]}
    assert _redact_posted_text(payload) == {
        "event": "COMMENT",
        "comments": [{"path": "a.py", "body": "see [redacted]"}],
    }


# ----------------------------------------------------- the transcript switch


@pytest.mark.parametrize(
    ("value", "posted"),
    [(None, True), ("on", True), ("1", True), ("YES", True), ("off", False),
     ("0", False), (" False ", False), ("no", False),
     # set but empty is the deploy step's off-switch convention
     ("", False), ("  ", False),
     # a privacy switch fails closed: a trailing comment or a typo means off
     ("off  # private target", False), ("disabled", False), ("onn", False)],
)  # fmt: skip
def test_the_switch_reads_the_environment(value: str | None, posted: bool) -> None:
    env = {} if value is None else {"OUTERLOOP_POST_TRANSCRIPTS": value}
    assert transcripts_posted(env) is posted


def _improved(**over: Any) -> AttemptResult:
    fields: dict[str, Any] = dict(
        outcome="improved",
        baseline=13.0,
        candidate=12.0,
        session=SessionResult(
            stop_reason="end_turn",
            is_error=False,
            cost_usd=1.0,
            num_turns=4,
            session_id="s",
            final_text="SESSION_PROSE: I tried the customer's private notes",
            transcript_path="",
        ),
        submit_report="SUBMIT_PROSE: the idea, step by step",
        panel_transcript="**Verification round 1**\n- JUDGE_PROSE: line 3 is wrong",
        panel_rounds=1,
        panel_blocking_open=True,
    )
    fields.update(over)
    return AttemptResult(**fields)


def test_a_pr_body_without_session_text_keeps_the_measurements() -> None:
    config = RunConfig(target="o/r", benchmark="tsp")
    rows = [{"sleep": 1, "launch": "sweep", "why": "WHY_PROSE", "job": "j1", "result": "ok"}]
    body = pr_body(_improved(), config, (), experiments=rows, transcripts=False)
    for prose in ("SESSION_PROSE", "SUBMIT_PROSE", "JUDGE_PROSE", "WHY_PROSE"):
        assert prose not in body
    assert TRANSCRIPT_WITHHELD in body
    assert panel_summary(1, True, False) in body  # the panel's outcome still shows
    assert "| candidate | 12 |" in body or "| candidate | 12.0" in body
    assert "| sweep |" in body and "| j1 |" in body  # what ran, not why
    # posted (the default), every part is there as before
    full = pr_body(_improved(), config, (), experiments=rows, transcripts=True)
    for prose in ("SUBMIT_PROSE", "JUDGE_PROSE", "WHY_PROSE"):
        assert prose in full


def test_a_pr_body_follows_the_deployment_switch(monkeypatch) -> None:
    config = RunConfig(target="o/r", benchmark="tsp")
    monkeypatch.setenv("OUTERLOOP_POST_TRANSCRIPTS", "off")
    assert "SUBMIT_PROSE" not in pr_body(_improved(), config, ())
    monkeypatch.delenv("OUTERLOOP_POST_TRANSCRIPTS")
    assert "SUBMIT_PROSE" in pr_body(_improved(), config, ())


def test_the_posted_run_report_withholds_the_session_text() -> None:
    config = RunConfig(target="o/r", benchmark="tsp")
    posted = _improved().report(config, transcripts=False)
    assert "SESSION_PROSE" not in posted and TRANSCRIPT_WITHHELD in posted
    assert "Outcome: **improved**" in posted and "Candidate: 12.0" in posted
    assert "SESSION_PROSE" in _improved().report(config)  # the local copy keeps it


@pytest.mark.parametrize(
    "heading", ["## Agent's report", "## Stewardship report", "## Steward's report"]
)
def test_an_archived_report_loses_its_session_section(heading: str) -> None:
    report = f"# Run report\nOutcome: **improved**\nBaseline: 1\n\n{heading}\nPROSE here\n"
    cut = withhold_session_text(report)
    assert "PROSE" not in cut and heading not in cut
    assert cut.startswith("# Run report\nOutcome: **improved**\nBaseline: 1")
    assert TRANSCRIPT_WITHHELD in cut
    error_report = "# Steward report\nOutcome: **steward-error**\nNote: boom\n"
    assert withhold_session_text(error_report) == error_report


def test_the_climb_board_and_strip_drop_the_authors_words(tmp_path, monkeypatch) -> None:
    from outerloop.climbboard import collect_rows, collect_status
    from outerloop.runstate import ENDED, PARKED, RunRecord, run_dir, save_record

    save_record(
        tmp_path,
        RunRecord(
            run_id="r-ended",
            target="o/r",
            task_title="t",
            benchmark="tsp",
            state=ENDED,
            ending="negative-result",
            stage={"hypothesis": "HYP_PROSE"},
        ),
        1.0,
    )
    (run_dir(tmp_path, "r-ended") / "report.md").write_text(
        "# Run report\nOutcome: **negative-result**\n\n## Agent's report\n## Hypothesis\nHYP2\n"
    )
    save_record(
        tmp_path,
        RunRecord(
            run_id="r-live",
            target="o/r",
            task_title="t",
            benchmark="tsp",
            state=PARKED,
            deadline=10.0,
            stage={"report": "LIVE_PROSE: working on it", "hypothesis": "LIVE_HYP"},
        ),
        1.0,
    )
    monkeypatch.setenv("OUTERLOOP_POST_TRANSCRIPTS", "off")
    (row,) = collect_rows(tmp_path, "o/r")["tsp"]
    assert row.hypothesis == ""
    (live,) = collect_status(tmp_path, "o/r", 2.0)["runs"]
    assert live["hypothesis"] == "" and "LIVE_PROSE" not in live["direction"]
    monkeypatch.delenv("OUTERLOOP_POST_TRANSCRIPTS")
    (row,) = collect_rows(tmp_path, "o/r")["tsp"]
    assert row.hypothesis
    (live,) = collect_status(tmp_path, "o/r", 2.0)["runs"]
    assert live["hypothesis"] == "LIVE_HYP"


def test_an_unrecognised_switch_value_is_logged_once(caplog) -> None:
    import logging

    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            assert transcripts_posted({"OUTERLOOP_POST_TRANSCRIPTS": "maybe-later"}) is False
    assert caplog.text.count("maybe-later") == 1


def test_file_text_written_to_github_is_redacted(tmp_path: Path) -> None:
    import base64

    pat = _token_file(tmp_path / "pat", OLD)
    capture = _Capture()
    client = GitHubClient(auth=FileTokenProvider(pat), transport=capture)
    _token_file(pat, NEW)
    client.put_file("o/r", "reports/r.md", f"# report\nleaked {NEW}\n", "research-log", "r")
    (_, payload) = capture.requests[-1]
    assert payload is not None
    text = base64.b64decode(payload["content"]).decode()
    assert NEW not in text and "leaked [redacted]" in text


def test_the_board_drops_a_withdrawal_reason_recorded_before_the_switch(
    tmp_path, monkeypatch
) -> None:
    from outerloop.climbboard import collect_rows
    from outerloop.runstate import ENDED, REJECTED, RunRecord, run_dir, save_record

    save_record(
        tmp_path,
        RunRecord(
            run_id="r-w",
            target="o/r",
            task_title="t",
            benchmark="tsp",
            state=ENDED,
            ending=REJECTED,
            ending_note="Author withdrew: REASON_PROSE from its notes",
        ),
        1.0,
    )
    (run_dir(tmp_path, "r-w") / "report.md").write_text("# Run report\nOutcome: **rejected**\n")
    monkeypatch.setenv("OUTERLOOP_POST_TRANSCRIPTS", "off")
    (row,) = collect_rows(tmp_path, "o/r")["tsp"]
    assert row.note == "Author withdrew"
    monkeypatch.delenv("OUTERLOOP_POST_TRANSCRIPTS")
    (row,) = collect_rows(tmp_path, "o/r")["tsp"]
    assert "REASON_PROSE" in row.note
