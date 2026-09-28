"""Sparse workspaces: the contract's `workspace.sparse` cone, and the rule
that the KERNEL's cone — never the workspace's own sparse state — decides
what every tree the kernel builds contains (docs/contract.md).

Each tree a run uses starts from the session's clone: the session edits it,
each dispatched job materializes a snapshot from it with `git worktree add`,
and the panel reads its base and head the same way. On a monorepo target
every one of those is a full checkout; a declared cone narrows them to the
directories the benchmarks and the scope need.

The session owns that clone's `.git/info/sparse-checkout` and its index's
skip-worktree bits, and `git worktree add` copies a worktree's sparse
patterns into the new one. Unpinned, a session could narrow the tree a job
measures or a judge reads — hiding exactly what its change regresses — while
the sealed commit, and so the PR, still carries the whole tree. So every
kernel git call pins sparse checkout: off when the contract declares no cone,
on with the kernel's own patterns when it does (git_pins, cone_patterns).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from outerloop.contract import Contract, normalize_path

# A cone entry reaches git as a pattern line and as an argument. One
# conservative charset keeps escaping out of the question — no glob,
# whitespace, backslash or comment character can change what a pattern means
# — and no part may start with "-", so no entry can read as an option.
_PART = r"[A-Za-z0-9._@+][A-Za-z0-9._@+-]*"
_SAFE_DIR = re.compile(rf"^{_PART}(?:/{_PART})*$")

# What a path is in the base tree: "tree", "blob", or None when absent.
Kind = Callable[[str], str | None]

# A cone entry that names no real directory: listing `<dir>/FILES_ONLY` makes
# <dir> a cone PARENT, whose own files git checks out, without any of its
# subdirectories. A scope file then brings the files beside it — never a
# whole package of unrelated assets.
FILES_ONLY = ".outerloop-cone"


class SparseError(ValueError):
    """The contract's cone cannot be built at this base."""


def workspace_cone(
    contract: Contract, kind: Kind, *, line_dirs: Iterable[str] = ()
) -> tuple[str, ...]:
    """The kernel's cone for `contract` at the base `kind` describes, or ()
    for a whole-tree checkout.

    The cone is the declared directories plus what an agent must see: each
    scope directory (solver and steward) whole, and for a scope file or the
    roadmap, the files directly beside it (FILES_ONLY). An absent scope path
    may become a file or directory, so include its whole subtree. Also include
    the research line's memory folder when lines are on (`line_dirs`). Root-level
    files are always in a cone. A declared entry that is not a directory at the
    base is a contract error, never a silently empty tree."""
    declared = contract.workspace.sparse
    if not declared:
        return ()
    dirs: list[str] = []
    for entry in declared:
        path = str(normalize_path(entry))
        if kind(path) != "tree":
            raise SparseError(f"workspace.sparse entry {entry!r} is not a directory at the base")
        dirs.append(path)
    steward = contract.steward.allowed if contract.steward is not None else []
    for entry in (*contract.scope.allowed, *steward, contract.roadmap):
        path = str(normalize_path(entry))
        if kind(path) in ("tree", None):
            # A path absent at the base may become a directory. Including the
            # path itself also includes a future file there through its parent.
            dirs.append(path)
        else:
            parent = path.rpartition("/")[0]
            if parent:
                dirs.append(f"{parent}/{FILES_ONLY}")
    dirs.extend(str(normalize_path(d)) for d in line_dirs)
    return normalize_cone(dirs)


def normalize_cone(dirs: Iterable[str]) -> tuple[str, ...]:
    """Deduplicated, sorted cone directories with nested entries folded into
    their ancestor (a cone directory already includes everything below it)."""
    unique = sorted(set(dirs))
    for d in unique:
        if not _SAFE_DIR.fullmatch(d):
            raise SparseError(f"cone directory {d!r} has characters a sparse pattern cannot carry")
    kept: list[str] = []
    for d in unique:  # sorted: an ancestor precedes its descendants
        if not any(d.startswith(k + "/") for k in kept):
            kept.append(d)
    return tuple(kept)


def _parents(dirs: tuple[str, ...]) -> set[str]:
    parents: set[str] = set()
    for d in dirs:
        parts = d.split("/")
        parents.update("/".join(parts[:i]) for i in range(1, len(parts)))
    return parents - set(dirs)


def cone_patterns(dirs: tuple[str, ...]) -> str:
    """The `info/sparse-checkout` file git itself writes for these cone
    directories (`git sparse-checkout set --cone`): root files, each parent
    directory's own files, then each cone directory whole."""
    lines = ["/*", "!/*/"]
    for parent in sorted(_parents(dirs)):
        lines += [f"/{parent}/", f"!/{parent}/*/"]
    lines += [f"/{d}/" for d in dirs]
    return "\n".join(lines) + "\n"


def in_cone(path: str, dirs: tuple[str, ...], parents: set[str] | None = None) -> bool:
    """Whether a checkout with these cone directories contains `path` (a file
    path): root files, everything under a cone directory, and the files
    directly inside a cone directory's parents — git's cone-mode rule."""
    if not dirs or "/" not in path:
        return True
    if any(path.startswith(d + "/") for d in dirs):
        return True
    return path.rpartition("/")[0] in (parents if parents is not None else _parents(dirs))


def outside(paths: Iterable[str], dirs: tuple[str, ...]) -> list[str]:
    """The paths a checkout with these cone directories leaves out."""
    if not dirs:
        return []
    parents = _parents(dirs)
    return [p for p in paths if not in_cone(p, dirs, parents)]


def git_pins(dirs: tuple[str, ...]) -> list[tuple[str, str]]:
    """GIT_CONFIG pairs every kernel git call carries: sparse checkout off for
    a whole-tree workspace (a session-enabled cone never narrows a kernel
    tree), or on in cone mode for a declared cone — with a full index, so
    each skip-worktree bit stays a plain per-entry fact."""
    if not dirs:
        return [("core.sparseCheckout", "false")]
    return [
        ("core.sparseCheckout", "true"),
        ("core.sparseCheckoutCone", "true"),
        ("index.sparse", "false"),
    ]
