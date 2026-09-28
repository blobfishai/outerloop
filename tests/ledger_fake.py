"""In-memory GitHub branch/CAS surface shared by writer integration tests."""

import hashlib
from dataclasses import dataclass, field

from outerloop.github import GitHubError


def _blob(path: str, content: str) -> dict:
    data = content.encode()
    sha = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    return {"path": path, "type": "blob", "sha": sha}


@dataclass
class LedgerGitHub:
    comments: list[dict] = field(default_factory=list)
    dry_run: bool = False
    ledger_head: str = "ledger-0"
    ledger_files: dict[str, str] = field(default_factory=dict)
    ledger_snapshots: dict[str, dict[str, str]] = field(default_factory=dict)
    ledger_writes: list[dict[str, str]] = field(default_factory=list)
    ledger_fail: bool = False
    commit_trees: dict[str, str] = field(default_factory=dict)
    ancestry: list[str] = field(default_factory=list)
    pull_requests: dict[int, dict] = field(default_factory=dict)
    # a branch carrying a huge default-branch tree: GitHub truncates a
    # RECURSIVE read of its root (subtree and non-recursive reads still work)
    truncated_full_tree: bool = False
    orphans: list[tuple[dict[str, str], str]] = field(default_factory=list)

    def default_branch(self, repo):
        return "main"

    def branch_sha(self, repo, branch):
        return "main-pin"

    def branch_head(self, repo, branch):
        assert branch == "research-log"
        return self.ledger_head

    def create_orphan_commit(self, repo, files, message):
        sha = f"orphan-{len(self.orphans)}"
        self.orphans.append((dict(files), message))
        self.ledger_snapshots[sha] = dict(files)
        return sha

    def create_ref(self, repo, ref, sha):
        # the ledger branch starts at a parentless commit, never a main pin
        assert ref == "refs/heads/research-log" and sha.startswith("orphan-")
        self.ledger_head = sha
        self.ledger_files = dict(self.ledger_snapshots[sha])

    def commit_tree(self, repo, sha):
        if sha not in self.commit_trees:
            raise GitHubError(404, "commit", "missing")
        return self.commit_trees[sha]

    def get_tree(self, repo, sha, recursive=True):
        if self.ledger_fail:
            raise GitHubError(503, "tree", "unavailable")
        commit, _, sub = sha.partition(":")
        files = self.ledger_snapshots.get(commit, self.ledger_files)
        if sub:
            files = {p[len(sub) + 1 :]: c for p, c in files.items() if p.startswith(sub + "/")}
        if recursive:
            return {
                "truncated": self.truncated_full_tree and not sub,
                "tree": [_blob(p, c) for p, c in files.items()],
            }
        dirs = sorted({p.split("/", 1)[0] for p in files if "/" in p})
        return {
            "truncated": False,
            "tree": [_blob(p, c) for p, c in files.items() if "/" not in p]
            + [{"path": d, "type": "tree", "sha": f"{commit}:{d}"} for d in dirs],
        }

    def get_file(self, repo, path, ref):
        if self.ledger_fail:
            raise GitHubError(503, "file", "unavailable")
        files = self.ledger_snapshots.get(ref, self.ledger_files)
        if path not in files:
            raise GitHubError(404, "file", "missing")
        return files[path]

    def put_files(self, repo, files, branch, message, *, expected_head):
        assert branch == "research-log"
        if self.ledger_fail or expected_head != self.ledger_head:
            return False
        self.ledger_snapshots[self.ledger_head] = dict(self.ledger_files)
        self.ledger_files.update(files)
        self.ledger_writes.append(files)
        self.ledger_head = f"ledger-{len(self.ledger_writes)}"
        return True

    def head_contains(self, repo, base, head):
        return self.compare(repo, base, head)["status"] in ("ahead", "identical")

    def compare(self, repo, base, head):
        a, b = self.ancestry.index(base), self.ancestry.index(head)
        return {"status": "identical" if a == b else "ahead" if a < b else "behind"}

    def get_pull_request(self, repo, number):
        return self.pull_requests[number]

    def list_comments(self, repo, number):
        return self.comments

    def comment(self, repo, number, body):
        self.comments.append({"body": body})
