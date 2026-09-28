"""Pinned reads and compare-and-swap writes for the shared research ledger."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict

from outerloop.github import GitHubClient, GitHubError
from outerloop.progress import (
    LEADER_FILE,
    PROGRESS_FILE,
    LeaderEntry,
    LedgerReadError,
    PendingSubmission,
    parse_leader,
    parse_pending,
    render_markdown,
)

RESEARCH_LOG_BRANCH = "research-log"
TOMBSTONE_BLOB_SHA = hashlib.sha1(b"blob 5\0null\n").hexdigest()
Ledger = dict[str, LeaderEntry]
Pendings = dict[str, PendingSubmission]
LedgerEdit = Callable[[Ledger, Pendings], dict[str, str]]


class LedgerWriteError(RuntimeError):
    """A ledger write could not finish; retain the operation for the next tick."""


LEDGER_README = "README.md"
_LEDGER_README_TEXT = (
    "# Research log\n\n"
    "Outerloop's record branch: benchmark progress (`BENCHMARKS.md`, `results/`),\n"
    "run reports and the climb board. It shares no history with the code\n"
    "branches; never merge it.\n"
)


def create_ledger_branch(github: GitHubClient, target: str, pinned_commit: str) -> None:
    """Create the ledger branch as a PARENTLESS commit holding only a README.
    It never carries the default branch's tree: the ledger needs only its own
    files, and on a large repository GitHub truncates a recursive read of the
    default branch's tree. The pin — the default-branch head the ledger
    started beside — is recorded in the commit message."""
    commit = github.create_orphan_commit(
        target,
        {LEDGER_README: _LEDGER_README_TEXT},
        f"Start the research log (default branch at {pinned_commit})",
    )
    github.create_ref(target, f"refs/heads/{RESEARCH_LOG_BRANCH}", commit)


def ensure_ledger_branch(github: GitHubClient, target: str, pinned_commit: str) -> None:
    """Create an absent ledger branch (create_ledger_branch), preserving an
    existing branch."""
    if not pinned_commit:
        raise ValueError("branch creation requires a pinned commit")
    head = github.branch_head(target, RESEARCH_LOG_BRANCH)
    if head is None:
        raise LedgerReadError("ledger branch head unavailable")
    if head or github.dry_run:
        return
    try:
        create_ledger_branch(github, target, pinned_commit)
    except GitHubError as exc:
        if not github.branch_head(target, RESEARCH_LOG_BRANCH):
            raise LedgerWriteError("could not create ledger branch") from exc


def _complete_tree(tree: object) -> list[dict]:
    if not isinstance(tree, dict) or tree.get("truncated") is not False:
        raise LedgerReadError("incomplete ledger tree")
    entries = tree.get("tree")
    if not isinstance(entries, list):
        raise LedgerReadError("malformed ledger tree")
    return entries


def ledger_paths(github: GitHubClient, target: str, head: str) -> list[dict]:
    """The ledger's file entries at `head`, each with its full path: the
    branch root's own files plus everything under `results/`. Read without a
    recursive read of the whole branch — a ledger branch that carries a large
    default branch's tree (older kernels created it from one) truncates that
    read, and the ledger lives only in these few files."""
    root = _complete_tree(github.get_tree(target, head, recursive=False))
    paths = [item for item in root if item.get("type") == "blob"]
    results = next(
        (i for i in root if i.get("path") == "results" and i.get("type") == "tree"), None
    )
    if results is not None:
        for item in _complete_tree(github.get_tree(target, results["sha"])):
            if item.get("type") == "blob":
                paths.append({**item, "path": f"results/{item['path']}"})
    return paths


def _read_at(
    github: GitHubClient,
    target: str,
    head: str,
    *,
    unmeasured: set[int] | None = None,
) -> tuple[Ledger, Pendings]:
    if not head:
        raise LedgerReadError("ledger branch does not exist; seed it from a pinned commit")
    try:
        # Tree reads avoid the contents API's silent 1,000-entry cap.
        paths = ledger_paths(github, target, head)
        leader: Ledger = {}
        pendings: Pendings = {}
        for item in paths:
            path = item["path"]
            if path == LEADER_FILE:
                leader = parse_leader(github.get_file(target, path, head))
            elif (
                path.startswith("results/submissions/")
                and path.endswith(".json")
                and item.get("sha") != TOMBSTONE_BLOB_SHA
            ):
                pending = parse_pending(github.get_file(target, path, head))
                if pending is not None:
                    if pending.path != path:
                        raise LedgerReadError("submission identity does not match its path")
                    if pending.status == "PENDING":
                        pendings[path] = pending
                    elif unmeasured is not None:
                        unmeasured.add(pending.pr_number)
        return leader, pendings
    except (GitHubError, KeyError, TypeError, ValueError) as exc:
        raise LedgerReadError(f"cannot read ledger at {head}") from exc


def read_ledger(
    github: GitHubClient, target: str, *, unmeasured: set[int] | None = None
) -> tuple[str, Ledger, Pendings]:
    """Read leader and live submissions; optionally collect terminal unmeasured PRs."""
    head = github.branch_head(target, RESEARCH_LOG_BRANCH)
    if head is None:
        raise LedgerReadError("ledger branch head unavailable")
    leader, pendings = _read_at(github, target, head, unmeasured=unmeasured)
    return head, leader, pendings


def progress_link(target: str) -> str:
    return f"[Benchmark progress](https://github.com/{target}/blob/{RESEARCH_LOG_BRANCH}/BENCHMARKS.md)"


def write_ledger(
    github: GitHubClient,
    target: str,
    expected_head: str,
    files: LedgerEdit,
    digits: dict[str, int] | None = None,
) -> None:
    """Apply a pure file-patch callback, recomputing on conflicts up to three times.

    The callback receives the current leader and pendings. Include leader.json
    to change the leader; its Markdown is always rendered in the same commit.
    Use record_pending/reject patches to add or remove submissions.
    """
    head = expected_head
    for attempt in range(3):
        leader, pendings = _read_at(github, target, head)
        patch = dict(files(leader, pendings))
        for path, content in patch.items():
            if path in {LEADER_FILE, PROGRESS_FILE}:
                continue
            if not path.startswith("results/submissions/") or not path.endswith(".json"):
                raise ValueError("ledger patch contains an unrelated path")
            pending = parse_pending(content)
            if pending is not None and pending.path != path:
                raise ValueError("submission identity does not match patch path")
            parts = path.split("/")
            if len(parts) != 4 or any(part in {"", ".", ".."} for part in parts):
                raise ValueError("invalid submission path")
        if LEADER_FILE in patch:
            leader = parse_leader(patch[LEADER_FILE])
        patch[LEADER_FILE] = (
            json.dumps({name: asdict(entry) for name, entry in sorted(leader.items())}, indent=2)
            + "\n"
        )
        patch[PROGRESS_FILE] = render_markdown(leader, target, digits)
        if github.put_files(
            target, patch, RESEARCH_LOG_BRANCH, "Update benchmark ledger", expected_head=head
        ):
            return
        new_head = github.branch_head(target, RESEARCH_LOG_BRANCH)
        if not new_head or new_head == head:
            raise LedgerWriteError("ledger write failed without a confirmed head change")
        head = new_head
        if attempt == 2:
            raise LedgerWriteError("ledger head moved during all three write attempts")
