"""The ledger branch on a large repository: created parentless, and read
without a recursive read of the whole branch (which GitHub truncates once a
tree passes its entry limit — a ledger branch created from a monorepo's
default branch carried that whole tree, and every ledger read failed)."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict
from typing import cast

from ledger_fake import LedgerGitHub
from outerloop.github import FileTokenProvider, GitHubClient
from outerloop.ledger_branch import LEDGER_README, ensure_ledger_branch, read_ledger
from outerloop.progress import LEADER_FILE, LeaderEntry

ENTRY = LeaderEntry("bench", "loss", "min", 3.0, 2.0, "run-1", "2026-09-20", 7)
LEADER = json.dumps({"bench": asdict(ENTRY)})


def _gh(**kwargs) -> GitHubClient:
    return cast(GitHubClient, LedgerGitHub(**kwargs))


def test_reads_a_ledger_whose_full_tree_is_truncated():
    """An existing branch that shares a huge default branch's tree (created
    by an older kernel) still reads: only the root listing and results/
    subtree are fetched."""
    files = {f"src/pkg{i}/mod.py": "x" for i in range(50)} | {LEADER_FILE: LEADER}
    gh = _gh(ledger_files=files, truncated_full_tree=True)
    head, leader, pendings = read_ledger(gh, "o/r")
    assert head == "ledger-0"
    assert leader["bench"].best == 2.0 and pendings == {}


def test_absent_branch_is_created_parentless_with_only_a_readme():
    fake = LedgerGitHub(ledger_head="")
    ensure_ledger_branch(cast(GitHubClient, fake), "o/r", "main-pin")
    assert fake.ledger_head == "orphan-0"
    files, message = fake.orphans[0]
    assert set(files) == {LEDGER_README}  # never the default branch's tree
    assert "main-pin" in message  # the pin survives as provenance
    head, leader, pendings = read_ledger(cast(GitHubClient, fake), "o/r")
    assert (head, leader, pendings) == ("orphan-0", {}, {})


def test_existing_branch_is_never_recreated():
    fake = LedgerGitHub(ledger_head="ledger-7")
    ensure_ledger_branch(cast(GitHubClient, fake), "o/r", "main-pin")
    assert fake.ledger_head == "ledger-7" and not fake.orphans


class _Transport:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.requests: list[urllib.request.Request] = []

    def __call__(self, request: urllib.request.Request) -> object:
        self.requests.append(request)
        return self.responses.pop(0)


def _client(tmp_path, responses: list[object]) -> tuple[GitHubClient, _Transport]:
    pat = tmp_path / "pat"
    pat.write_text("github_pat_test123\n")
    pat.chmod(0o600)
    transport = _Transport(responses)
    return GitHubClient(auth=FileTokenProvider(pat), transport=transport), transport


def test_client_reads_a_tree_non_recursively(tmp_path):
    client, transport = _client(tmp_path, [{"sha": "t", "truncated": False, "tree": []}])
    client.get_tree("org/repo", "head", recursive=False)
    assert transport.requests[-1].full_url.endswith("/git/trees/head")


def test_client_orphan_commit_has_no_parent_and_no_base_tree(tmp_path):
    client, transport = _client(tmp_path, [{"sha": "blob1"}, {"sha": "tree1"}, {"sha": "c1"}])
    assert client.create_orphan_commit("org/repo", {"README.md": "hi\n"}, "start") == "c1"
    blob, tree, commit = (json.loads(cast(bytes, r.data)) for r in transport.requests)
    assert blob["encoding"] == "base64"
    assert "base_tree" not in tree
    assert tree["tree"] == [{"path": "README.md", "mode": "100644", "type": "blob", "sha": "blob1"}]
    assert commit == {"message": "start", "tree": "tree1", "parents": []}
