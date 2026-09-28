"""The PR-level base-moved notice against real Git history (#7 review cases).

A fake GitHub answers the compare API from a real repository, so the notice's
decisions are checked on the histories review found: an automatic-merge PR that
still needs its fold, and a base rewritten under a PR's recorded pin.
"""

import subprocess
import urllib.parse
from dataclasses import replace
from typing import cast

import pytest

from fakes import RecordingDispatcher
from ledger_fake import LedgerGitHub
from outerloop.github import GitHubClient
from outerloop.inbox import gather_github_messages, pending, wake_pending
from outerloop.runstate import PARKED, RunRecord, run_dir, save_record
from outerloop.tick import sweep
from test_tick import NOW, FakeSlurm

CONTRACT = """benchmarks: [{name: x, command: echo, metric: score, direction: max}]
budgets: {gpu_hours_per_run: 1, runs_per_week: 3}
scope: {allowed: [src/pilot/]}
roadmap: docs/roadmap.md
merge: auto
workspace: {sparse: [src/pilot]}
"""


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def _commit(repo, changes, message):
    for name, content in changes.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", message)
    return _git(repo, "rev-parse", "HEAD").strip()


def _history(tmp_path, *, rewritten=False):
    """A PR on the recorded pin, and a base that moved outside the cone. With
    `rewritten`, the pin itself changed a visible path and the base was
    rewritten from before it."""
    repo = tmp_path / "git"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "user.email", "t@t")
    first = _commit(
        repo,
        {
            "BENCHMARKS.md": CONTRACT,
            "docs/roadmap.md": "roadmap\n",
            "src/pilot/shared.txt": "original shared\n",
            "src/pilot/solver.txt": "original solver\n",
            "unrelated/asset.txt": "original asset\n",
        },
        "initial",
    )
    pin = first
    if rewritten:
        pin = _commit(repo, {"src/pilot/shared.txt": "published base content\n"}, "old base")
    _git(repo, "checkout", "-qb", "pr")
    head = _commit(repo, {"src/pilot/solver.txt": "measured improvement\n"}, "proposal")
    _git(repo, "checkout", "-qB", "main", first)
    tip = _commit(repo, {"unrelated/asset.txt": "current base content\n"}, "new base")
    return repo, pin, head, tip


class RealGitHub(LedgerGitHub):
    """Answers GitHub's three-dot compare from the real repository."""

    compare = GitHubClient.compare
    compare_files = GitHubClient.compare_files
    head_contains = GitHubClient.head_contains
    _expect_dict = staticmethod(GitHubClient._expect_dict)

    def __init__(self, repo, head, tip):
        super().__init__()
        self.repo, self.head, self.tip = repo, head, tip
        self.merged: list[tuple] = []

    def _request(self, method, path, body=None):
        assert method == "GET" and "/compare/" in path
        base, head = urllib.parse.unquote(path.split("/compare/")[1]).split("...")
        counts = _git(self.repo, "rev-list", "--left-right", "--count", f"{base}...{head}")
        behind, ahead = map(int, counts.split())
        status = (
            "identical"
            if ahead == behind == 0
            else "ahead"
            if behind == 0
            else "behind"
            if ahead == 0
            else "diverged"
        )
        names = _git(self.repo, "diff", "--name-only", "--no-renames", f"{base}...{head}")
        return {
            "status": status,
            "ahead_by": ahead,
            "behind_by": behind,
            "files": [{"filename": n} for n in names.splitlines() if n],
            "merge_base_commit": {"sha": _git(self.repo, "merge-base", base, head).strip()},
        }

    def branch_sha(self, repo, branch):
        assert branch == "main"
        return self.tip

    def get_pull_request(self, *args):
        return {
            "state": "closed" if self.merged else "open",
            "merged": bool(self.merged),
            "base": {"sha": self.tip, "ref": "main"},
            "head": {"sha": self.head},
            "draft": False,
            "mergeable_state": "clean",
        }

    def list_pr_reviews(self, *args):
        return []

    list_pr_review_comments = list_pr_reviews
    list_check_runs = list_pr_reviews

    def get_file_content(self, *args, **kwargs):
        return CONTRACT

    def allowed_merge_methods(self, repo):
        return ["MERGE"]

    def merge_pull(self, repo, number, method, expected_head):
        self.merged.append((repo, number, method, expected_head))
        return True

    def comment(self, repo, number, body):
        self.comments.append(
            {
                "id": len(self.comments) + 1,
                "body": body,
                "user": {"login": "bot"},
                "author_association": "MEMBER",
            }
        )


def _record(root, pin, head, *, sparse, blessed):
    record = RunRecord(
        "review",
        "org/repo",
        "proposal",
        PARKED,
        pr_url="https://github.com/org/repo/pull/9",
        agent_id="agent-01",
        auto_blessed_head=head if blessed else "",
        auto_bless_base=pin,
        auto_publish_head=head,
        deadline=0,
        stage={"base_sha": pin, "base_branch": "main", "cone": ["src/pilot"] if sparse else []},
    )
    save_record(root, record, NOW)
    return record


@pytest.mark.parametrize("bless", ["blessed", "legacy-mismatch", "legacy-moved-past"])
@pytest.mark.parametrize("sparse", [False, True])
def test_an_automatic_merge_pr_is_still_woken_to_fold(tmp_path, sparse, bless):
    """The automatic merge requires the head to contain the latest tip, so a
    PR on that path must still be woken to fold a move outside its cone,
    whether its head is blessed or its bless waits on the base in either
    legacy reason text (records from before the typed reason kind)."""
    repo, pin, head, tip = _history(tmp_path)
    root = tmp_path / "runs-root"
    record = _record(root, pin, head, sparse=sparse, blessed=bless == "blessed")
    if bless != "blessed":
        reason = {
            "legacy-mismatch": f"base moved: main {tip[:12]} != measured {pin}",
            "legacy-moved-past": (
                f"base moved: origin/main tip {tip[:12]} moved past measured base {pin}"
            ),
        }[bless]
        record = replace(record, auto_bless_reason=reason, auto_bless_reason_kind="")
        save_record(root, record, NOW)
    github = RealGitHub(repo, head, tip)
    client = cast(GitHubClient, github)
    assert client.compare_files(record.target, head, tip) == (["unrelated/asset.txt"], pin)
    dispatcher = RecordingDispatcher()
    for now in (NOW, NOW + 60, NOW + 120):
        sweep(root, FakeSlurm().compute(), dispatcher, now, github=github, bot_login="bot")
    assert not github.merged  # never merged without the latest tip
    assert dispatcher.dispatched


@pytest.mark.parametrize("sparse", [False, True])
def test_a_base_rewritten_under_the_pin_notifies(tmp_path, sparse):
    """The pin changed a visible path and the base was rewritten from before
    it. The compare lists only an outside path, but merging the PR would bring
    the dropped change back, so the author is told."""
    repo, pin, head, tip = _history(tmp_path, rewritten=True)
    root = tmp_path / "runs-root"
    record = _record(root, pin, head, sparse=sparse, blessed=False)
    github = RealGitHub(repo, head, tip)
    client = cast(GitHubClient, github)
    moved = client.compare_files(record.target, head, tip)
    assert moved is not None and moved[0] == ["unrelated/asset.txt"] and moved[1] != pin
    gather_github_messages(
        run_dir(root, record.run_id), record, client, "bot", NOW, github.get_pull_request()
    )
    assert wake_pending(run_dir(root, record.run_id), record)


@pytest.mark.parametrize("sparse", [False, True])
def test_an_unseen_move_on_a_held_pin_wakes_no_one(tmp_path, sparse):
    """The saving #7 is for: outside the cone, the pin still held, and no
    automatic merge waiting on a fold. A whole tree still notifies."""
    repo, pin, head, tip = _history(tmp_path)
    root = tmp_path / "runs-root"
    record = _record(root, pin, head, sparse=sparse, blessed=False)
    github = RealGitHub(repo, head, tip)
    gather_github_messages(
        run_dir(root, record.run_id),
        record,
        cast(GitHubClient, github),
        "bot",
        NOW,
        github.get_pull_request(),
    )
    moved = [m for m in pending(run_dir(root, record.run_id), 0) if m.kind == "base-moved"]
    assert bool(moved) == (not sparse)
    assert wake_pending(run_dir(root, record.run_id), record) == (not sparse)


@pytest.mark.parametrize(
    "kind, reason, expected",
    [
        ("base_moved", "base moved: anything", (True, "base0000")),
        ("other", "contract merge is manual", (False, "")),
        ("", "base moved: main 1234567 != measured abcdef0", (True, "abcdef0")),
        (
            "",
            "base moved: origin/main tip 1234567 moved past measured base abcdef0",
            (True, "abcdef0"),
        ),
        ("", "panel did not run", (False, "")),
        ("", "", (False, "")),
    ],
)
def test_base_moved_refusal_reads_the_typed_kind_and_both_legacy_texts(kind, reason, expected):
    """One recognition, shared by the tick's re-bless and the base-moved notice."""
    from outerloop.runstate import base_moved_refusal

    record = RunRecord(
        "run",
        "org/repo",
        "task",
        PARKED,
        auto_bless_reason=reason,
        auto_bless_reason_kind=kind,
        auto_bless_base="base0000",
    )
    assert base_moved_refusal(record) == expected
