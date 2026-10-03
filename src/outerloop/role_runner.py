"""The role-runner: build a harness for a role, run one session, read its result.

One loop runs every role (docs/design/consolidation.md), and ONE
construction builds every harness: `build_harness`
maps a RoleSpec to any backend uniformly — backends are interchangeable, and
containment is the deployment's business (`container_image` where a jail
exists, the ephemeral runner where one doesn't), never a per-role tool posture.
Roles differ by prompt, verbs, and output handling.

`run_role` runs a RoleSpec on a Harness. A role WITH an `output_schema` (a
judge) records its verdict through the installed syscall tool (`finding` /
`conclude`) and the kernel reads it back authoritatively (`read_verdict`);
each call is validated in-session. It does NOT judge, gate, measure, or
post — the result-policy (kernel) acts on the RoleResult.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from outerloop.harness import (
    HERMES_ENDPOINT_KEY_ENV,
    HERMES_ENDPOINT_PROVIDER,
    ClaudeCodeHarness,
    CodexHarness,
    Harness,
    HermesHarness,
    SessionResult,
    default_claude_model,
    vertex_from_env,
)
from outerloop.rolespec import RoleSpec

log = logging.getLogger(__name__)

# Native Claude Code tool names a spec may grant. A spec's other tools
# (pr-context-read, retriever) are harness-provided MCP tools, wired
# separately — never passed as native CLI tools.
_NATIVE_TOOLS = frozenset(
    {"Read", "Grep", "Glob", "Write", "Edit", "Bash", "WebSearch", "WebFetch"}
)

# hermes names capabilities as toolsets: `file` is the read/edit surface,
# `terminal` the shell. Everything else stays disabled for parity with the
# other backends — no spec grants web/browser/... tools there either.
_HERMES_TOOLSETS = (
    "file",
    "terminal",
    "web",
    "search",
    "browser",
    "computer_use",
    "code_execution",
    "delegation",
    "cronjob",
    "skills",
    "memory",
)

# hermes resolves credentials per provider (a registry); "openai" maps to its
# canonical `openai-api` provider id (api-key auth against api.openai.com —
# plain "openai" is a provider GROUP there, not an id). "custom" is any
# OpenAI-compatible endpoint (a self-hosted inference server, a managed model
# API): the harness seeds a named provider entry carrying the endpoint's base
# URL, and hermes reads the key from the variable that entry names.
_HERMES_PROVIDERS = {
    "openrouter": ("openrouter", "OPENROUTER_API_KEY"),
    "openai": ("openai-api", "OPENAI_API_KEY"),
    "custom": (f"custom:{HERMES_ENDPOINT_PROVIDER}", HERMES_ENDPOINT_KEY_ENV),
}


def hermes_endpoint_error(provider: str, base_url: str) -> str:
    """Why a hermes (provider, base URL) pair cannot run ("" when it can): the
    one owner of the rules the climb, the tick preflight and init share. The
    `custom` provider needs a plain http(s) base URL; the others take none, so
    a key issued for OpenRouter or OpenAI is never sent to some other host.

    The answer reaches logs, job command lines, pull requests, issues and the
    author's inbox, and a refused value may hold a credential, so it never
    repeats the URL: it names only the scheme and host of a URL that has them,
    and a provider only when it reads as a provider name."""
    name = (provider or "openrouter").strip().lower()
    if name not in _HERMES_PROVIDERS:
        shown = f" {name!r}" if re.fullmatch(r"[a-z0-9_-]{1,24}", name) else ""
        return f"unknown hermes provider{shown} (have: {sorted(_HERMES_PROVIDERS)})"
    url = base_url.strip()
    if name != "custom":
        if url:
            return (
                f"a hermes base URL needs the custom provider, not {name!r}: "
                f"a {name} key must never be sent to another host"
            )
        return ""
    if not url:
        return "the custom hermes provider needs a base URL (an OpenAI-compatible endpoint)"
    return _plain_url_error(url, "the hermes base URL")


def _plain_url_error(url: str, what: str) -> str:
    """Why `url` is not a plain http(s) endpoint URL ("" when it is). A plain
    URL has a host and no user, password, query string or fragment, which is
    where credentials hide, and no quotes, backslashes or whitespace. `url` is
    never quoted back (see hermes_endpoint_error): a refusal names `what` and,
    once the URL parses as http(s) with a host, that scheme and host."""
    plain = "is not a plain http(s) URL"
    try:
        parts = urllib.parse.urlsplit(url)
        host = parts.hostname or ""
    except ValueError:
        return f"{what} {plain} (it does not parse)"
    if parts.scheme not in ("http", "https") or not host:
        return f"{what} {plain} with a host"
    where = f"{what} for {parts.scheme}://{host}"
    if "@" in parts.netloc:
        return f"{where} {plain}: it carries a user or password (a key belongs in the key file)"
    if "?" in url or "#" in url:
        return f"{where} {plain}: it has a query string or fragment (a key belongs in the key file)"
    if any(c.isspace() or c in "\"'\\" for c in url):
        return f"{where} {plain}: it holds a quote, backslash or whitespace"
    return ""


# Codex `-c KEY=VALUE` overrides for the codex sessions a climb starts. The
# environment form (`OUTERLOOP_CODEX_CONFIG`) separates entries with ";" or a
# newline, never a comma: a codex value is TOML and may itself hold commas
# (an array, an inline table). A key is a dotted path of plain names, so an
# entry can never be read as another codex flag.
CODEX_CONFIG_ENV = "OUTERLOOP_CODEX_CONFIG"
_CODEX_CONFIG_KEY = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")

# A codex config never carries a credential. It reaches job command lines
# (argv is world-readable on a shared node) and wake-spec.json, and the panel's
# codex judges receive the author's config, so a credential in it would be
# readable by other users and would let every judge authenticate as the
# author. A provider's key reaches a session only from its role's key file,
# through the provider's `env_key`. Refused (codex 0.130.0's provider fields):
# a static bearer token, a command-backed `auth` table, a static header or
# query parameter whose name reads as a credential, and a base URL with a user,
# password, query string or fragment.
_CODEX_CREDENTIAL_FIELDS = frozenset({"experimental_bearer_token", "bearer_token", "auth"})
_CODEX_LITERAL_MAPS = frozenset({"http_headers", "query_params"})
_CREDENTIAL_NAME = re.compile(
    r"auth|token|secret|passw|api[-_]?key|cookie|credential|signature|[-_]key$|^key$|^sig$|^code$"
)


def _codex_credential(path: tuple[str, ...], value: object) -> str:
    """The dotted path of a credential the override `path` = `value` would
    set, or "". `value` is the override's parsed TOML: an inline table is
    searched like the dotted keys it stands for."""
    names = [part.lower() for part in path]
    for i, name in enumerate(names):
        if name in _CODEX_CREDENTIAL_FIELDS:
            return ".".join(path[: i + 1])
        named = names[i + 1] if i + 1 < len(names) else ""
        if name in _CODEX_LITERAL_MAPS and named and _CREDENTIAL_NAME.search(named):
            return ".".join(path[: i + 2])
    if isinstance(value, dict):
        for key, sub in value.items():
            found = _codex_credential((*path, str(key)), sub)
            if found:
                return found
        return ""
    if names and names[-1].endswith("base_url") and isinstance(value, str):
        try:
            netloc = urllib.parse.urlsplit(value).netloc
        except ValueError:
            netloc = ""
        if "@" in netloc or "?" in value or "#" in value:
            return ".".join(path)
    return ""


def _toml_value(raw: str) -> object:
    """An override's value as codex reads it: TOML, else the plain string."""
    import tomllib

    try:
        return tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw


def codex_config_entries(entries: Iterable[str], source: str = "--codex-config") -> tuple[str, ...]:
    """Validated codex config overrides, in order (codex applies them in order,
    so a later entry for the same key wins). Blank entries are dropped; an
    entry that is not `KEY=VALUE`, or that would carry a credential (see
    _CODEX_CREDENTIAL_FIELDS), raises ValueError naming `source`. The error
    names a malformed entry by its position and a refused one by its key,
    never by its text: a mistyped entry may hold the credential itself."""
    out: list[str] = []
    for raw in entries:
        entry = raw.strip()
        if not entry:
            continue
        key, sep, value = entry.partition("=")
        key = key.strip()
        if not sep or not _CODEX_CONFIG_KEY.fullmatch(key):
            raise ValueError(
                f"{source}: entry {len(out) + 1} is not a codex KEY=VALUE override "
                "(KEY is a dotted path such as model_providers.local.base_url)"
            )
        found = _codex_credential(tuple(key.split(".")), _toml_value(value.strip()))
        if found:
            raise ValueError(
                f"{source}: {found!r} would put a credential in the codex config, which "
                "reaches job command lines and every codex session it configures; a "
                "provider's key comes from its role's key file through env_key"
            )
        out.append(f"{key}={value.lstrip()}")
    return tuple(out)


def codex_config_from_text(raw: str, source: str = CODEX_CONFIG_ENV) -> tuple[str, ...]:
    """Entries from the environment form: separated by ";" or newlines."""
    return codex_config_entries(re.split(r"[;\n]", raw), source)


def codex_config_args(entries: Iterable[str]) -> tuple[str, ...]:
    """The codex argv for validated entries: one `-c KEY=VALUE` pair each."""
    return tuple(arg for entry in entries for arg in ("-c", entry))


def role_key(key_file: str | Path, backend: str = "claude") -> str:
    """Read a role's API key file — tolerating its ABSENCE exactly when the
    deployment's Vertex config covers the claude backend (an ADC-only
    deployment holds no Anthropic key at all; the harness then authenticates
    via ADC and ignores api_key). Every other backend, and claude without
    Vertex, still fails loudly on a missing/lax key file."""
    from outerloop.harness import vertex_from_env

    path = Path(key_file).expanduser()
    if backend == "claude" and vertex_from_env() is not None and not path.is_file():
        return ""
    from outerloop.github import FileTokenProvider

    return FileTokenProvider(path).token()


def build_harness(
    api_key: str,
    spec: RoleSpec,
    *,
    backend: str = "claude",
    binary: str | None = None,
    model: str | None = None,
    container_image: str = "",
    codex_extra_args: tuple[str, ...] = (),
    hermes_repo: Path | None = None,
    hermes_provider: str = "",
    hermes_base_url: str = "",
) -> Harness:
    """Construct the harness for any role on any backend — the ONE deployment
    wiring (`spec.tools` → native flags, `spec.budget` → turns/walltime,
    `spec.execution` → the backend's execution surface). Each branch below is
    the backend's irreducible calling convention, nothing more:

    - claude: `spec.tools` filtered to the native CLI tools; a judge's cwd is an
      untrusted checkout, so judges (specs with an `output_schema`) run `--bare`
      — never loading the tree's CLAUDE.md / hooks as instructions. Editors keep
      instruction discovery (the target repo's guidance is legitimate for them).
    - codex: `danger-full-access` uniformly — codex's own sandbox needs
      bubblewrap (absent in the image, unreliable nested in apptainer), so the
      boundary is the deployment's container or ephemeral runner, exactly as it
      is for every other backend.
    - hermes: toolsets from the spec's execution (`terminal` for a role that
      executes); provider/key seeded per the registry above, and the base URL
      of an OpenAI-compatible endpoint for the `custom` provider.

    Containment is NOT decided here: pass `container_image` where the
    deployment has a jail (the cluster), pass none where the runner itself is
    the ephemeral boundary (CI). The tokenless split keeps credentials out of
    the session either way."""
    if backend == "codex":
        # the web, when the spec grants it: the config override, because
        # `codex exec` does not accept `--search`
        web = ("-c", "tools.web_search=true") if "WebSearch" in spec.tools else ()
        return CodexHarness(
            api_key=api_key,
            binary=binary or "codex",
            model=model or "",  # "" -> codex's configured default; pin a verified id
            sandbox="danger-full-access",
            timeout_s=spec.budget.walltime_s,
            container_image=container_image,
            extra_args=(*codex_extra_args, *web),
        )
    if backend == "hermes":
        if hermes_repo is None:
            raise ValueError("hermes backend needs hermes_repo (the pinned clone)")
        endpoint_error = hermes_endpoint_error(hermes_provider, hermes_base_url)
        if endpoint_error:
            raise ValueError(endpoint_error)
        seed, key_env = _HERMES_PROVIDERS[(hermes_provider or "openrouter").strip().lower()]
        # `terminal` (the shell) is keyed on the SAME signal claude uses — the
        # spec granting the Bash tool — not on can_execute, so every backend
        # gives a role the same shell/no-shell whether or not those two ever
        # diverge for a future spec.
        enabled = ("file", "terminal") if "Bash" in spec.tools else ("file",)
        if "WebSearch" in spec.tools:
            enabled = (*enabled, "web", "search")
        return HermesHarness(
            api_key=api_key,
            key_env=key_env,
            repo_dir=hermes_repo,
            provider=seed,
            model=model or "",
            base_url=hermes_base_url.strip(),
            max_turns=spec.budget.max_turns,
            timeout_s=spec.budget.walltime_s,
            enabled_toolsets=enabled,
            disabled_toolsets=tuple(t for t in _HERMES_TOOLSETS if t not in enabled),
            container_image=container_image,
        )
    if backend != "claude":
        raise ValueError(f"unknown backend: {backend!r}")
    return ClaudeCodeHarness(
        api_key=api_key,
        binary=binary or "claude",
        model=model or default_claude_model(),
        max_turns=spec.budget.max_turns,
        timeout_s=spec.budget.walltime_s,
        allowed_tools=tuple(tool for tool in spec.tools if tool in _NATIVE_TOOLS),
        container_image=container_image,
        # a judge's cwd contains an untrusted checkout: never load its CLAUDE.md
        # / hooks / project settings as instructions (defence in depth beside
        # the caller's sanitize_checkout)
        bare=spec.output_schema is not None,
        # Vertex (ADC) billing when the deployment configures it; the env
        # contract has ONE owner (harness.vertex_from_env), so every claude
        # role on every CLI flips together and the API key stays the fallback
        vertex=vertex_from_env(),
    )


@dataclass(frozen=True)
class RoleResult:
    """The outcome of one role run: the session plus, for judge roles, the
    validated verdict. `ok` is False when the session errored or no valid
    verdict was committed; `data` is the validated object (None for an editing
    role, whose artifact is the workspace diff, or on failure)."""

    ok: bool
    session: SessionResult
    data: dict[str, Any] | None = None
    error: str = ""


def run_role(
    spec: RoleSpec,
    harness: Harness,
    brief_text: str,
    workspace: Path,
    resume_session_id: str | None = None,
) -> RoleResult:
    """Run one role session; for a judge (a spec with an `output_schema`), read
    the verdict it committed through the syscall tool.

    The `harness` is assumed already constructed for this role
    (`build_harness`). A judge records each finding as one validated tool call
    and commits with `conclude`; `read_verdict` is the authoritative kernel-side
    check, so there is no repair loop — a missing verdict (the judge never
    concluded) or a malformed one is a failure the caller surfaces (a skip
    stub), never a clean read (silence is never endorsement). Installing the
    tool BEFORE the session force-owns the `.outerloop/` channel, so a
    pre-planted or stale ABI never survives into the read — a resumed (revise)
    session likewise starts from a clean channel and commits a fresh verdict.
    """
    is_judge = spec.output_schema is not None
    if is_judge:
        from outerloop.syscall import install_tool

        install_tool(workspace)
    session = harness.run(brief_text, workspace, resume_session_id)
    if session.is_error:
        return RoleResult(
            ok=False, session=session, error=session.error_detail or session.stop_reason
        )
    if not is_judge:
        return RoleResult(ok=True, session=session)  # editing role: artifact is the diff
    from outerloop.syscall import VerdictError, read_verdict

    try:
        data = read_verdict(workspace)
    except VerdictError as exc:
        return RoleResult(ok=False, session=session, error=f"invalid verdict: {exc}")
    if data is None:
        return RoleResult(ok=False, session=session, error="judge produced no verdict")
    return RoleResult(ok=True, session=session, data=data)
