"""Shared GitHub posting helpers for the reviewer and verifier.

Round-numbered comments, inline reviews, and skip stubs — the machinery for
getting a judge's findings onto a PR thread. Backend-agnostic on purpose: no
model dependency, so any judge backend posts through here.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping

from outerloop.github import GitHubClient, GitHubError
from outerloop.harness import redact
from outerloop.markers import marker

log = logging.getLogger(__name__)

# The operator's switch for session text on GitHub. Off (0, off, false, no),
# the author's report, the panel judges' transcripts and the author's notes
# on its launches stay in the run's records on the orchestrator host; pull
# requests, issue reports, the archived run reports and the climb board carry
# the measured results and a note saying where the text is. Meant for a
# private target, or sessions that read material that must not leave the
# deployment. The author's replies to review comments are messages it
# addresses to the reviewers, and are still posted. Read from the environment
# and, at every post, from the operator's .env (transcripts_posted).
TRANSCRIPTS_ENV = "OUTERLOOP_POST_TRANSCRIPTS"
_TRANSCRIPTS_ON = frozenset({"1", "on", "true", "yes"})
# empty is an explicit off-switch, as `OUTERLOOP_PANEL=""` is for the panel
_TRANSCRIPTS_OFF = frozenset({"", "0", "off", "false", "no"})
_WARNED_VALUES: set[str] = set()
_WARNED_FILES: set[str] = set()
TRANSCRIPT_WITHHELD = (
    f"*Session text is not posted for this deployment (`{TRANSCRIPTS_ENV}=off`); it "
    "stays in the run's records on the orchestrator host.*"
)
# The headings that open a run report's session text, as the climb
# (orchestrator.AttemptResult.report) and the steward (a re-based env, or a
# session that changed nothing) write them.
_SESSION_SECTIONS = ("## Agent's report", "## Stewardship report", "## Steward's report")


def transcripts_posted(environ: Mapping[str, str] | None = None) -> bool:
    """Whether session text may be posted to GitHub (see TRANSCRIPTS_ENV).
    Two places set it, and either one switching it off keeps the text off: the
    process environment, and the operator's `.env` (~/.config/outerloop/.env),
    read again at every call. A job keeps the environment it was submitted
    with, and a parked run arms its next wake from its own job, so reading the
    file at post time is what makes an off-switch reach every run already in
    flight. Absent from both, the text is posted. Present, the switch fails
    closed: only 1, on, true or yes post it; empty (the deploy step's
    off-switch convention), 0, off, false, no or any other value (such as one
    with a trailing comment, logged once) keep the text off GitHub, and so
    does a `.env` that exists but cannot be read under the deploy step's rule."""
    env = os.environ if environ is None else environ
    return _switch_on(env.get(TRANSCRIPTS_ENV)) and _switch_on(_operator_file_setting())


def _switch_on(setting: str | None) -> bool:
    """One source's reading of the switch: None (absent) and the on-values post."""
    if setting is None:
        return True
    value = setting.strip().casefold()
    if value in _TRANSCRIPTS_ON:
        return True
    if value not in _TRANSCRIPTS_OFF and value not in _WARNED_VALUES:
        _WARNED_VALUES.add(value)
        log.warning(
            "%s=%r is neither on nor off; session text stays off GitHub", TRANSCRIPTS_ENV, value
        )
    return False


def _operator_file_setting() -> str | None:
    """TRANSCRIPTS_ENV as the operator's `.env` sets it now; None when the
    file is absent or does not set it, "" (off) when it exists but cannot be
    trusted or read (it must be the operator's and not group/world-writable,
    as the deploy step requires)."""
    from outerloop import paths
    from outerloop.cli import StartError, env_file_values

    try:
        return env_file_values(paths.ENV_FILE, (TRANSCRIPTS_ENV,)).get(TRANSCRIPTS_ENV)
    except StartError as exc:
        if str(paths.ENV_FILE) not in _WARNED_FILES:
            _WARNED_FILES.add(str(paths.ENV_FILE))
            log.warning("%s; session text stays off GitHub", exc)
        return ""


def panel_summary(rounds: int, blocking_open: bool, degraded: bool) -> str:
    """The panel's outcome without its transcript, for a withheld section."""
    if blocking_open:
        state = "blocking findings open"
    elif degraded:
        state = "the final read was degraded (a lens produced no verdict)"
    else:
        state = "clean"
    return f"{rounds} panel read(s); {state}. Judge transcripts: {TRANSCRIPT_WITHHELD}"


def withhold_session_text(report: str) -> str:
    """A run report with its session text cut: from the first session heading
    on, the report is replaced by the withheld note. A report without one is
    returned unchanged."""
    cuts = [i for i in (report.find(heading) for heading in _SESSION_SECTIONS) if i >= 0]
    if not cuts:
        return report
    return f"{report[: min(cuts)].rstrip()}\n\n{TRANSCRIPT_WITHHELD}\n"


# Posting/transport failures an advisory role tolerates — logged, never fatal,
# because an advisory reviewer or verifier must not turn a target repo's CI red.
# Programming errors (AttributeError, KeyError, TypeError) deliberately
# propagate. Model-call errors are NOT here: they arise inside the agent
# session, which handles them itself — posting has nothing to do with the model.
EXPECTED_FAILURES = (
    GitHubError,
    ValueError,
    OSError,
    json.JSONDecodeError,
)


def post_round(
    client: GitHubClient,
    repo: str,
    number: int,
    marker: str,
    body: str,
    pr_data: dict,
    reviewed_by: str = "",
) -> str:
    """Post one NEW comment per round — numbered, stamped with the reviewed
    head — so every round notifies and stays visible (edits do neither).
    Shared by the advisory reviewer and the verifier; each counts rounds by
    ITS OWN marker. Runs are only PR-open or an explicit label request, so
    volume is human-bounded.
    """
    stamp, round_label = _round_stamp(client, repo, number, marker, pr_data, reviewed_by)
    client.comment(repo, number, body.replace(marker, f"{marker}\n{stamp}", 1))
    return round_label


def _round_stamp(
    client: GitHubClient,
    repo: str,
    number: int,
    marker: str,
    pr_data: dict,
    reviewed_by: str = "",
) -> tuple[str, str]:
    """(stamp line, round label): prior rounds are counted across BOTH
    issue comments and review bodies, so switching a role between posting
    styles never resets its numbering."""
    head = pr_data.get("head")
    head_sha = str(head.get("sha", ""))[:8] if isinstance(head, dict) else ""
    # attribution is render-side data (it can cross a job boundary in the
    # least-token split): strip backticks/newlines, cap, never trust. The cap
    # fits a panel line ("summarizer:<backend> over lens+lens+..."); a real
    # overflow ends in an ellipsis so it never reads as a mid-word bug.
    by = " ".join(str(reviewed_by).split()).replace("`", "")
    if len(by) > 120:
        by = by[:119].rstrip() + "…"
    # The round number is cosmetic: an EXPECTED failure counting prior
    # rounds must never cost the round itself. Programming errors still
    # propagate, per this module's policy.
    try:
        bodies = [str(c.get("body", "")) for c in client.list_comments(repo, number)]
        bodies += [str(r.get("body", "")) for r in client.list_pr_reviews(repo, number)]
        # STARTS WITH the marker: a quote-reply prefixes every line with
        # "> ", so it cannot match — and this stays true for any posting
        # identity (Actions token, GitHub App, or a self-hoster's
        # machine-user PAT, which posts as type User)
        prior = [b for b in bodies if b.lstrip().startswith(marker)]
        # Rounds count PER REVIEWER: with several standing opinions on one
        # PR, a shared counter reads as re-reviews that never happened
        # (terra "Round 1", claude "Round 2"). Unattributed rounds keep the
        # shared count.
        if by:
            prior = [b for b in prior if f"reviewer `{by}`" in b]
        round_label = f"**Round {len(prior) + 1}**"
        if head_sha and any(f"reviewed head `{head_sha}`" in b for b in prior):
            round_label += " (re-run on the same head)"
    except EXPECTED_FAILURES as exc:
        log.warning("could not count prior rounds: %s", exc)
        round_label = "**New round** (prior count unavailable)"
    by_clause = f" — reviewer `{by}`" if by else ""
    return f"{round_label} — reviewed head `{head_sha or 'unknown'}`{by_clause}.\n\n", round_label


def post_round_review(
    client: GitHubClient,
    repo: str,
    number: int,
    marker: str,
    body: str,
    inline: list[dict],
    pr_data: dict,
    fallback_body: str,
    reviewed_by: str = "",
) -> str:
    """The Reviews-API sibling of post_round: body summary plus anchored
    inline comments, event COMMENT always (the client hard-codes it). A
    posting failure falls back to a plain issue comment carrying
    fallback_body — the FULL single-comment rendering, because the review
    body alone may say no more than "findings are attached" while the
    findings live in the rejected inline payload."""
    stamp, round_label = _round_stamp(client, repo, number, marker, pr_data, reviewed_by)
    try:
        client.create_pr_review(repo, number, body.replace(marker, f"{marker}\n{stamp}", 1), inline)
    except EXPECTED_FAILURES as exc:
        log.warning("inline review failed (%s); falling back to a comment", exc)
        client.comment(repo, number, fallback_body.replace(marker, f"{marker}\n{stamp}", 1))
    return round_label


SKIP_MARKER = marker("round-skipped")


def post_skip_stub(
    client: GitHubClient,
    repo: str,
    number: int,
    role: str,
    exc: Exception,
    secrets: tuple[str, ...] = (),
) -> None:
    """Silence is invisible: when the model API refuses a round (dead
    credits, spend cap, auth), say so on the thread instead of leaving a
    gap only the Actions tab can see. A DIFFERENT marker than a real
    round, deliberately — a stub never counts toward round numbering and
    never rides as follow-up wake context (both match on their own
    markers).

    `secrets` are the model API key(s) the caller holds — an auth error is
    exactly the class that can echo request material, so we scrub them from the
    posted text. The caller supplies them (the harness owns its own key) so
    posting stays backend-agnostic — the key env var is provider-specific, this
    module is not.
    """
    note = redact(str(exc), tuple(s for s in secrets if s))[:200]
    try:
        client.comment(
            repo,
            number,
            f"{SKIP_MARKER}\n*The {role} round could not run — the model API "
            f"refused the request ({type(exc).__name__}: {note}). Treat this "
            f"as an outage, not a clean read; re-add the review label to "
            f"re-request once the API recovers.*",
        )
    except EXPECTED_FAILURES as post_exc:
        log.warning("could not post the skip stub: %s", post_exc)
