#!/bin/bash
# Install verified source at <repo>, and its runtime at <repo>.runtime/<sha>/.
# The sibling runtime contains python/ (uv-managed Python) and venv/; neither
# can dirty the checkout or be removed by its git clean. Both source and the
# whole runtime are bound read-only at their host paths during contained runs.
# .complete is written last; reruns reuse a completed runtime for this pin.
#
# Usage: install_hermes.sh [target_dir]
#   target_dir  where the clone lives (default: $REVIEW_HERMES_REPO,
#               else ~/hermes-agent)
set -euo pipefail

# Pinned tag AND its COMMIT sha: the tag names the version for humans, the
# sha is the integrity pin (tags are mutable; a moved tag must fail loudly,
# never run with the panel key). Bump both together, in lockstep with the
# GH review workflows' HERMES_REF. NOTE: v-tags here are ANNOTATED — plain
# `git ls-remote` shows the TAG OBJECT's sha; pin the dereferenced `^{}`
# line (the commit), which is what `rev-parse HEAD` yields after checkout.
WANT="v2026.8.13"
WANT_SHA="f80f453ae0679347e38abc917c7f94f717bf96c5"
TARGET="${1:-${REVIEW_HERMES_REPO:-$HOME/hermes-agent}}"

if [ ! -d "$TARGET/.git" ]; then
    git clone --depth 1 --branch "$WANT" \
        https://github.com/NousResearch/hermes-agent "$TARGET"
fi
head=$(git -C "$TARGET" rev-parse HEAD)
dirty=$(git -C "$TARGET" status --porcelain)
if [ "$head" != "$WANT_SHA" ] || [ -n "$dirty" ]; then
    # a wrong or DIRTY checkout must never run with the panel key: re-pin
    # hard (this clone is a provisioned artifact, not a dev tree)
    echo "hermes-agent at $TARGET is $head (dirty=$([ -n "$dirty" ] && echo yes || echo no)); re-pinning to $WANT"
    git -C "$TARGET" fetch --depth 1 origin "refs/tags/$WANT:refs/tags/$WANT"
    git -C "$TARGET" checkout -q --detach "tags/$WANT"
    git -C "$TARGET" reset --hard -q "tags/$WANT"
    git -C "$TARGET" clean -fdxq
fi
head=$(git -C "$TARGET" rev-parse HEAD)
if [ "$head" != "$WANT_SHA" ]; then
    echo "hermes-agent: tag $WANT resolves to $head, expected $WANT_SHA — refusing" >&2
    exit 1
fi
TARGET=$(cd "$TARGET" && pwd -P)
RUNTIME="${TARGET}.runtime/$WANT_SHA"
if [ -f "$RUNTIME/.complete" ] && [ "$(cat "$RUNTIME/.complete")" = "$WANT_SHA" ] && \
   [ -x "$RUNTIME/venv/bin/python" ]; then
    echo "hermes-agent $WANT ($WANT_SHA) ready at $TARGET"
    exit 0
fi
mkdir -p "$RUNTIME"
# Refuse concurrent provisioning; an interrupted build has no completion marker.
if ! mkdir "$RUNTIME/.installing" 2>/dev/null; then
    echo "hermes-agent: installation in progress; if interrupted, remove $RUNTIME/.installing and retry" >&2
    exit 1
fi
trap 'rmdir "$RUNTIME/.installing"' EXIT
rm -f "$RUNTIME/.complete"
export UV_PYTHON_INSTALL_DIR="$RUNTIME/python"
export UV_PROJECT_ENVIRONMENT="$RUNTIME/venv"
export UV_CACHE_DIR="$RUNTIME/cache"
export UV_PYTHON_PREFERENCE=only-managed
export UV_LINK_MODE=copy
uv python install --no-bin 3.12
python=$(uv python find 3.12)  # UV_PYTHON_PREFERENCE=only-managed restricts the search
case "$python" in
    "$RUNTIME/python/"*) ;;
    *) echo "hermes-agent: Python must live under $RUNTIME/python" >&2; exit 1 ;;
esac
uv sync --project "$TARGET" --frozen --no-install-project --python "$python"
"$RUNTIME/venv/bin/python" -B -c 'import sys; assert sys.version_info >= (3, 12)'
rm -rf "$UV_CACHE_DIR"  # the venv is complete; sessions never need the download cache
printf '%s\n' "$WANT_SHA" > "$RUNTIME/.complete"
echo "hermes-agent $WANT ($WANT_SHA) ready at $TARGET"
