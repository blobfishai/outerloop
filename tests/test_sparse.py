"""Sparse workspaces: the contract's cone (docs/contract.md `workspace.sparse`)
and the invariant that the KERNEL's cone — never the workspace's own sparse
state — decides what every tree the kernel builds contains."""

from __future__ import annotations

import itertools
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from outerloop.contract import ContractError, ScopeError, load_contract
from outerloop.dispatch import snapshot_tree, write_eval_job
from outerloop.github import Workspace
from outerloop.sparse import (
    FILES_ONLY,
    SparseError,
    cone_patterns,
    git_pins,
    in_cone,
    normalize_cone,
    outside,
    workspace_cone,
)

FILES = (
    "top.txt",
    "pkg/own.txt",
    "pkg/a/x.py",
    "pkg/a/deep/y.py",
    "pkg/b/z.py",
    "data/github/t.json",
    "data/stripe/t.json",
    "docs/roadmap.md",
    "src/m.py",
)


def _git(root: Path, *args: str, input: str | None = None) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        input=input,
    ).stdout.strip()


def _repo(tmp_path: Path, files: tuple[str, ...] = FILES) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    for rel in files:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{rel}\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    return root


def _files(tree: Path) -> set[str]:
    return {
        str(p.relative_to(tree))
        for p in tree.rglob("*")
        if p.is_file() and ".git" not in p.relative_to(tree).parts
    }


def _contract(**extra: str) -> str:
    body = {
        "scope": "scope:\n  allowed: [pkg/a/x.py]\n",
        "roadmap": "roadmap: docs/roadmap.md\n",
        "workspace": "",
    } | extra
    return (
        "benchmarks:\n  - {name: b, command: 'true', metric: m, direction: max}\n"
        "budgets: {gpu_hours_per_run: 0, runs_per_week: 1}\n"
        + body["scope"]
        + body["roadmap"]
        + body["workspace"]
    )


# ------------------------------------------------------------------ the cone


def test_normalize_cone_folds_nested_and_dedupes():
    assert normalize_cone(["pkg/a", "pkg", "src", "src", "pkg/a/deep"]) == ("pkg", "src")
    assert normalize_cone(["b", "a/c", "a/b"]) == ("a/b", "a/c", "b")
    assert normalize_cone([]) == ()


@pytest.mark.parametrize("bad", ["has space", "-leading", "a/-b", "a*", "a?b", "x\\y", "a/[b]"])
def test_normalize_cone_refuses_what_a_pattern_cannot_carry(bad):
    with pytest.raises(SparseError):
        normalize_cone([bad])


def test_cone_patterns_match_what_git_itself_writes(tmp_path):
    """The pattern file the job and panel write must be byte-for-byte the one
    `git sparse-checkout set --cone` writes, or checkouts would differ."""
    root = _repo(tmp_path)
    for dirs in [
        ("pkg/a",),
        ("pkg/a/deep", "src"),
        ("data/github", "pkg", "src"),
        ("docs",),
        (f"pkg/{FILES_ONLY}", "src"),
        (f"pkg/a/{FILES_ONLY}", f"data/{FILES_ONLY}"),
    ]:
        cone = normalize_cone(dirs)
        _git(root, "sparse-checkout", "set", "--cone", *cone)
        assert (root / ".git" / "info" / "sparse-checkout").read_text() == cone_patterns(cone)
    _git(root, "sparse-checkout", "disable")


def test_in_cone_agrees_with_a_real_cone_checkout(tmp_path):
    """in_cone decides the seal's skip-worktree bits; git decides what a job
    checks out. They must agree on every path, for every cone."""
    root = _repo(tmp_path)
    candidates = (
        "pkg",
        "pkg/a",
        "pkg/a/deep",
        "pkg/b",
        "data/github",
        "src",
        "docs",
        f"pkg/{FILES_ONLY}",
        f"pkg/a/{FILES_ONLY}",
    )
    for n in (1, 2, 3):
        for dirs in itertools.combinations(candidates, n):
            cone = normalize_cone(dirs)
            _git(root, "sparse-checkout", "set", "--cone", *cone)
            present = _files(root)
            assert present == {p for p in FILES if in_cone(p, cone)}, cone
            assert set(outside(FILES, cone)) == set(FILES) - present, cone
    _git(root, "sparse-checkout", "disable")


def test_git_pins():
    assert git_pins(()) == [("core.sparseCheckout", "false")]
    assert ("core.sparseCheckout", "true") in git_pins(("src",))
    assert ("core.sparseCheckoutCone", "true") in git_pins(("src",))
    assert ("index.sparse", "false") in git_pins(("src",))


# ---------------------------------------------------------- contract + cone


def test_contract_workspace_defaults_to_the_whole_tree():
    contract = load_contract(_contract(), "o/r")
    assert contract.workspace.sparse == []


def test_contract_workspace_sparse_parses_and_is_strict():
    contract = load_contract(
        _contract(workspace="workspace:\n  sparse: [src, data/github]\n"), "o/r"
    )
    assert contract.workspace.sparse == ["src", "data/github"]
    with pytest.raises(ValidationError):
        load_contract(_contract(workspace="workspace:\n  sprase: [src]\n"), "o/r")


@pytest.mark.parametrize("bad", ["/abs", "../up", "src/*", ".git/x", "."])
def test_contract_workspace_sparse_rejects_unsafe_entries(bad):
    with pytest.raises((ScopeError, ContractError)):
        load_contract(_contract(workspace=f"workspace:\n  sparse: ['{bad}']\n"), "o/r")


def _kind(root: Path):
    def kind(path: str) -> str | None:
        try:
            return _git(root, "cat-file", "-t", f"HEAD:{path}")
        except subprocess.CalledProcessError:
            return None

    return kind


def test_workspace_cone_adds_what_the_agent_must_see(tmp_path):
    root = _repo(tmp_path)
    contract = load_contract(
        _contract(
            scope="scope:\n  allowed: [pkg/a/x.py, pkg/b, pkg/new/mod.py]\n",
            workspace="workspace:\n  sparse: [data/github]\n",
        ),
        "o/r",
    )
    cone = workspace_cone(contract, _kind(root), line_dirs=("agent_memory",))
    # the declared dir and a scope DIR whole; for a scope FILE, an absent
    # scope path and the roadmap, only the files beside them; line memory
    assert cone == (
        "agent_memory",
        "data/github",
        f"docs/{FILES_ONLY}",
        f"pkg/a/{FILES_ONLY}",
        "pkg/b",
        "pkg/new/mod.py",
    )
    # a scope file brings its siblings, never its directory's subdirectories
    assert in_cone("pkg/a/x.py", cone) and not in_cone("pkg/a/deep/y.py", cone)
    assert in_cone("docs/roadmap.md", cone)


def test_workspace_cone_is_empty_without_a_declaration(tmp_path):
    root = _repo(tmp_path)
    assert workspace_cone(load_contract(_contract(), "o/r"), _kind(root)) == ()


def test_workspace_cone_refuses_a_declared_path_that_is_not_a_directory(tmp_path):
    root = _repo(tmp_path)
    for entry in ("data/nope", "top.txt"):
        contract = load_contract(_contract(workspace=f"workspace:\n  sparse: [{entry}]\n"), "o/r")
        with pytest.raises(SparseError, match="not a directory"):
            workspace_cone(contract, _kind(root))


# --------------------------------------------- the kernel's cone, not the session's


def _session_hides_stripe(root: Path) -> None:
    """What a session can do inside its own workspace: narrow its checkout so
    `data/stripe` is not on disk."""
    _git(root, "sparse-checkout", "set", "--no-cone", "/*", "!/data/stripe/")
    assert not (root / "data" / "stripe").exists()


def _measured_tree(tmp_path: Path, root: Path, sha: str, sparse: tuple[str, ...]) -> set[str]:
    """Run the REAL job script (no image: the command runs uncontained in
    "$TREE") and return the files the eval saw."""
    run_dir = tmp_path / f"run-{len(sparse)}"
    script = write_eval_job(
        run_dir,
        "e",
        repo_root=root,
        snapshot_sha=sha,
        command="find . -type f | sed 's|^\\./||' | sort",
        image="",
        sparse=sparse,
    )
    subprocess.run(["sh", str(script)], check=True, capture_output=True)
    ev = run_dir / "eval-e"
    assert (ev / "exit-code").read_text().strip() == "0", (ev / "setup.log").read_text()
    return set((ev / "stdout").read_text().split())


def test_job_measures_the_whole_committed_tree_whatever_the_session_hid(tmp_path):
    """Regression: `git worktree add` copies the workspace's sparse patterns,
    so a session that narrowed its own checkout also narrowed the tree every
    eval measured — while the sealed commit (and the PR) kept the whole tree.
    Sparse checkout is pinned off for a whole-tree workspace."""
    root = _repo(tmp_path)
    _session_hides_stripe(root)
    sha = _git(root, "rev-parse", "HEAD")
    assert _measured_tree(tmp_path, root, sha, ()) == set(FILES)


def test_job_measures_the_kernel_cone_not_the_session_patterns(tmp_path):
    root = _repo(tmp_path)
    _session_hides_stripe(root)
    sha = _git(root, "rev-parse", "HEAD")
    cone = normalize_cone(["data", "pkg/a"])
    assert _measured_tree(tmp_path, root, sha, cone) == {p for p in FILES if in_cone(p, cone)}
    assert "data/stripe/t.json" in _measured_tree(tmp_path, root, sha, cone)


def test_panel_worktrees_use_the_kernel_tree(tmp_path):
    root = _repo(tmp_path)
    _session_hides_stripe(root)
    sha = _git(root, "rev-parse", "HEAD")
    Workspace(root=root).add_worktree(tmp_path / "whole", sha)
    assert _files(tmp_path / "whole") == set(FILES)
    cone = normalize_cone(["src"])
    Workspace(root=root, sparse=cone).add_worktree(tmp_path / "cone", sha)
    assert _files(tmp_path / "cone") == {p for p in FILES if in_cone(p, cone)}


# ------------------------------------------------------------------- sealing


def _sparse_workspace(tmp_path: Path, dirs: tuple[str, ...]) -> Workspace:
    origin = _repo(tmp_path)
    dest = tmp_path / "ws"
    ws = Workspace.clone(str(origin), dest, checkout=False)
    ws.sparse = normalize_cone(dirs)
    ws.apply_sparse()
    ws.git("checkout", "-q", "-B", "main", "origin/main")
    _git(dest, "config", "user.email", "t@t")
    _git(dest, "config", "user.name", "t")
    return ws


def _whole_tree_workspace(tmp_path: Path) -> Workspace:
    origin = _repo(tmp_path)
    ws = Workspace.clone(str(origin), tmp_path / "ws")
    _git(ws.root, "config", "user.email", "t@t")
    _git(ws.root, "config", "user.name", "t")
    return ws


def test_seal_keeps_what_a_session_hid_in_a_whole_tree_workspace(tmp_path):
    """Regression: with sparse pinned off, a session that narrowed its own
    checkout had the hidden files sealed as DELETIONS (a false out-of-scope
    refusal). An absent file the session's index marks skip-worktree is
    unchanged; the eval still measures it."""
    ws = _whole_tree_workspace(tmp_path)
    base = ws.git("rev-parse", "HEAD")
    _session_hides_stripe(ws.root)
    (ws.root / "pkg" / "a" / "x.py").write_text("edited\n")
    snap = snapshot_tree(ws, base)
    assert set(ws.git("ls-tree", "-r", "--name-only", snap.commit).split("\n")) == set(FILES)
    assert ws.git("show", f"{snap.commit}:pkg/a/x.py") == "edited"
    assert ws.git("diff", "--name-only", base, snap.commit) == "pkg/a/x.py"
    assert _measured_tree(tmp_path, ws.root, snap.commit, ()) == set(FILES)


def test_seal_still_records_a_real_deletion(tmp_path):
    ws = _whole_tree_workspace(tmp_path)
    base = ws.git("rev-parse", "HEAD")
    (ws.root / "src" / "m.py").unlink()
    snap = snapshot_tree(ws, base)
    assert "src/m.py" not in ws.git("ls-tree", "-r", "--name-only", snap.commit).split("\n")


def test_snapshot_stages_an_out_of_cone_file_that_is_on_disk(tmp_path):
    """Only an ABSENT out-of-cone file reads as unchanged. One on disk — a
    restored line file, or a session edit the scope check must see — is
    sealed exactly as in a whole-tree workspace."""
    ws = _sparse_workspace(tmp_path, ("pkg/a",))
    base = ws.git("rev-parse", "HEAD")
    (ws.root / "src").mkdir(exist_ok=True)
    (ws.root / "src" / "m.py").write_text("materialized and edited\n")
    snap = snapshot_tree(ws, base)
    assert ws.git("show", f"{snap.commit}:src/m.py") == "materialized and edited"
    assert ws.git("show", f"{snap.commit}:data/stripe/t.json") == "data/stripe/t.json"


def test_clone_without_checkout_then_cone_writes_only_the_cone(tmp_path):
    ws = _sparse_workspace(tmp_path, ("pkg/a",))
    assert _files(ws.root) == {p for p in FILES if in_cone(p, ws.sparse)}


def test_snapshot_in_a_cone_keeps_what_the_cone_leaves_out(tmp_path):
    """An absent out-of-cone file is unchanged, never a deletion; an edit in
    the cone is sealed; a session file outside the cone is sealed too, so the
    scope check sees it."""
    ws = _sparse_workspace(tmp_path, ("pkg/a",))
    base = ws.git("rev-parse", "HEAD")
    (ws.root / "pkg" / "a" / "x.py").write_text("edited\n")
    (ws.root / "data" / "new").mkdir(parents=True)
    (ws.root / "data" / "new" / "f.txt").write_text("outside\n")
    snap = snapshot_tree(ws, base)
    tree = set(ws.git("ls-tree", "-r", "--name-only", snap.commit).split("\n"))
    assert tree == set(FILES) | {"data/new/f.txt"}
    assert ws.git("show", f"{snap.commit}:pkg/a/x.py") == "edited"
    # the working index is untouched by the seal
    assert ws.git("diff", "--cached", "--name-only") == ""


def test_add_all_in_a_cone_stages_a_session_file_outside_it(tmp_path):
    ws = _sparse_workspace(tmp_path, ("pkg/a",))
    (ws.root / "src").mkdir(exist_ok=True)
    (ws.root / "src" / "new.py").write_text("x\n")
    (ws.root / "pkg" / "a" / "x.py").write_text("edited\n")
    ws.add_all()
    staged = set(ws.git("diff", "--cached", "--name-only").split("\n"))
    assert staged == {"src/new.py", "pkg/a/x.py"}  # nothing outside read as deleted


def test_apply_sparse_undoes_a_session_that_dropped_the_cone(tmp_path):
    ws = _sparse_workspace(tmp_path, ("pkg/a",))
    _git(ws.root, "sparse-checkout", "disable")  # the session widened to everything
    assert _files(ws.root) == set(FILES)
    ws.apply_sparse()
    assert _files(ws.root) == {p for p in FILES if in_cone(p, ws.sparse)}


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("recorded_base", [False, True])
def test_terminal_notebook_preserves_omitted_files(tmp_path, monkeypatch, sparse, recorded_base):
    """The terminal fallback reconstructs a workspace from a saved run record.
    Its published notebook must retain omitted files and capture visible edits,
    including for records without a base SHA and legacy whole-tree contracts.
    """
    from outerloop import attempt
    from outerloop.runstate import (
        ABORTED,
        ENDED,
        RUNNING,
        RunRecord,
        load_record,
        run_dir,
        save_record,
    )

    origin = _repo(tmp_path)
    contract = _contract(
        scope="scope:\n  allowed: [pkg/a]\n",
        workspace="workspace:\n  sparse: [pkg/a]\n" if sparse else "",
    ).replace("direction: max", "direction: max, lines: true")
    (origin / ".outerloop.yaml").write_text(contract)
    _git(origin, "add", ".outerloop.yaml")
    _git(origin, "commit", "-qm", "contract")
    bare = tmp_path / "origin.git"
    _git(origin, "clone", "--bare", str(origin), str(bare))
    base = _git(bare, "rev-parse", "main")
    root = tmp_path / "runs-root"
    record = RunRecord(
        run_id="r1",
        target="owner/repo",
        task_title="terminal notebook",
        state=RUNNING,
        benchmark="b",
        stage={"base_sha": base} if recorded_base else {},
    )
    save_record(root, record, 1)

    class LocalAuth:
        def token(self) -> str:
            return "local-test-token"

    auth = LocalAuth()
    ws = Workspace.clone(str(bare), run_dir(root, record.run_id) / "ws", auth=auth, checkout=False)
    ws.sparse = attempt._kernel_cone(
        ws, load_contract(contract, record.target), base, record.benchmark, record.agent_id
    )
    ws.apply_sparse()
    line = "agents/agent-01"
    ws.git("checkout", "-q", "-B", line, base)
    ws.push(line)
    (ws.root / "pkg/a/x.py").write_text("edited\n")
    (ws.root / "pkg/a/deep/y.py").unlink()
    # The terminal must use the saved base contract, not the session's file.
    (ws.root / ".outerloop.yaml").write_text(_contract())
    monkeypatch.setattr(attempt, "target_clone_url", lambda target: str(bare))

    attempt.finish_run(root, record, ABORTED, "stopped", 2, auth=auth, bot_login="test-bot")

    assert load_record(root, record.run_id).state == ENDED
    assert _git(bare, "show", f"{line}:data/stripe/t.json") == "data/stripe/t.json"
    assert _git(bare, "show", f"{line}:pkg/a/x.py") == "edited"
    assert set(_git(bare, "diff", "--name-status", base, line).splitlines()) == {
        "M\t.outerloop.yaml",
        "M\tpkg/a/x.py",
        "D\tpkg/a/deep/y.py",
    }


@pytest.mark.parametrize("new_path", ["newpkg", "pkg/newpkg", "pkg/newfile.py"])
def test_new_scope_subtree_is_present_in_eval_and_panel(tmp_path, new_path):
    """A scope path absent at the base can become a directory or a file;
    either must be visible in the same committed tree the kernel measures.
    """
    root = _repo(tmp_path)
    contract = load_contract(
        _contract(
            scope=f"scope:\n  allowed: [{new_path}]\n",
            workspace="workspace:\n  sparse: [src]\n",
        ),
        "owner/repo",
    )
    cone = workspace_cone(contract, _kind(root))
    created = new_path if new_path.endswith(".py") else f"{new_path}/module.py"
    destination = root / created
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("new implementation\n")
    ws = Workspace(root=root, sparse=cone)
    snap = snapshot_tree(ws, ws.git("rev-parse", "HEAD"))
    assert ws.git("show", f"{snap.commit}:{created}") == "new implementation"
    assert created in _measured_tree(tmp_path, root, snap.commit, cone)
    ws.add_worktree(tmp_path / "panel", snap.commit)
    assert (tmp_path / "panel" / created).read_text() == "new implementation\n"
    ws.apply_sparse()
    assert destination.read_text() == "new implementation\n"
