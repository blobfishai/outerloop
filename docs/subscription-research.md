# Parallel research with native CLIs

This fork adds `outerloop research`: give one goal to a bounded set of Claude
Code and Codex workers, collect their answers, and resume their native sessions.
Each worker uses an explicitly selected authentication mode and model. Subscription
login remains the default. API and Vertex require explicit plan fields; authentication
failure stops that worker, without fallback or account rotation.

This command produces research answers and execution receipts. The existing
experiment scheduler, author/panel authentication, benchmark gates and publication
path are unchanged. A completed worker is not a verified scientific result.
Choose an independent acceptance test before claiming a research improvement.

## Set up

Use a separate checkout/install of this fork. Keep already-running research loops
on their original installation. Python 3.12+, macOS or Linux, and recent native
CLIs are required. Tested CLI versions are Claude Code 2.1.283 and Codex 0.157.1;
older CLIs may not support the isolation flags.

```bash
uv sync --locked
claude auth login
codex login
```

The default native profiles are `~/.claude` and `~/.codex`. For another account,
sign in through the native CLI using a separate profile directory:

```bash
CLAUDE_CONFIG_DIR=/absolute/path/to/claude-profile claude auth login
CODEX_HOME=/absolute/path/to/codex-profile codex login
```

Do not copy authentication files between profiles. One active worker is admitted
per canonical profile directory, including across different research roots.
Multiple paths to the same directory do not create independent accounts.
Locks cover this runner; other native CLI processes do not participate in them.

On macOS, explicitly setting `CLAUDE_CONFIG_DIR` changes Claude's keychain namespace,
even when it names `~/.claude`. The default profile therefore keeps its native home
and USER context. Custom profiles retain their explicit directory. Codex uses its
selected `CODEX_HOME`. Native history stays with the selected profile; profiles
must remain available for resume. Environment variables carrying API keys, gateway
credentials, GitHub tokens and alternate model endpoints are not inherited.

## Run a goal

Save `research.json`, replacing the model values with explicit models available
to your subscriptions:

```json
{
  "version": 1,
  "goal": "Find a faster algorithm for our stated graph problem, preserving exact answers.",
  "timeout_s": 900,
  "idle_timeout_s": 300,
  "tools": "read",
  "max_turns": 24,
  "workers": [
    {
      "id": "algorithms",
      "backend": "claude",
      "profile": "~/.claude",
      "binary": "claude",
      "model": "YOUR_CLAUDE_MODEL",
      "prompt": "Compare candidate algorithms and identify their assumptions. Cite primary sources."
    },
    {
      "id": "counterexamples",
      "backend": "codex",
      "profile": "~/.codex",
      "binary": "codex",
      "model": "YOUR_CODEX_MODEL",
      "prompt": "Construct counterexamples and design correctness and performance tests."
    }
  ]
}
```

```bash
uv run outerloop research run research.json --root /private/path/to/research-01 --parallel 2
uv run outerloop research status --root /private/path/to/research-01
```

There are at most eight assignments and four concurrent workers. Each gets its own
directory. Both providers use maximum reasoning effort. `tools: "none"` supports
closed-book fixtures; `read` permits research tools without shell access or
workspace edits. Claude has Read/Glob/Grep/WebSearch/WebFetch; Codex has native
web search with its shell and extension tools disabled. Codex's native tool host
remains enabled in `read` mode because current CLIs use it for live web search;
shell, unified execution, image generation and extension tools stay disabled.
Their tool surfaces are
different; control the supplied context when comparing their results. Put shared
source material in the goal or assignment text. There is no automatic task splitting,
code experiment launch, synthesis, evaluator, or cost-based account switching.

Walltime is bounded to 90 minutes per turn, default 15 minutes. The idle deadline
is separate and resets on stdout activity. `max_turns` applies to Claude's native
limit; Codex executes one native turn under the walltime limit. Provider quota
errors are failures and are never retried using another credential.

The root holds private plan, state, result and normalized event files outside each
worker's directory. Each turn also retains the provider-emitted model/tool JSON
events in `native-events.jsonl`, at mode0600 with a SHA256 and byte count in its
state receipt. Known API credentials are redacted, and the transcript is bounded
to64MiB. Auth-status responses and stderr are not journaled. Only events the
native provider emits are available; this does not recover hidden reasoning.
`status` reads these files without launching anything. Native IDs are persisted
as soon as the CLI emits them, before the terminal response. Completion requires
the terminal success event, a native identity, final text, and exit status zero.
Process exit or a partial answer alone cannot mark the worker complete.

Running the same plan against the same root again reads its existing state. It
does not make another model call, including after partial execution. Use a new
root for a deliberate replication or sequential baseline (`--parallel 1`).

## Resume

Write the next instruction to `followup.txt`, then use a new stable request ID:

```bash
uv run outerloop research resume --root /private/path/to/research-01 \
  --worker algorithms --request-id check-01 --prompt-file followup.txt
```

The runner checks the original backend, model, native profile and worker directory.
It resumes the exact recorded session, never the most recent session. A repeated
request ID observes its receipt; using that ID for another prompt is refused.
Different workers can receive follow-ups concurrently. Accepted but uncertain
requests block later control until reconciled. They never become implicit success.
Acceptance and whole-goal snapshots share a short metadata lock. Accepting a
follow-up publishes incomplete status before the model call, so a concurrent
reader cannot combine old and new worker states into a false completion.

The runner uses process groups and a separate liveness supervisor. Timeout kills
the owned group, including children that ignore TERM. Controller death closes the
liveness pipe and stops the group; supervisor death is detected by the controller
and stops the same group. This is local POSIX process control, not a filesystem
sandbox or cgroup boundary. A descendant deliberately escaping with `setsid`, or
simultaneous loss of controller and supervisor, requires stronger OS containment.
After the native leader exits, its supervisor remains alive until cleanup. It
reports the exit code through a separate pipe that the native CLI cannot write;
controller loss during the completion handoff still stops background children.

After an abrupt controller exit, a record can still say `running` although the
supervisor cleaned up. The runner refuses automatic replay because it cannot know
whether the last turn took effect. This first version has no automatic recovery or
reconciliation command. Keep its records and native history for inspection.

## Compatibility and provenance

Fork base: `outerloop-science/outerloop` at
`5563c46cee5371f5056690bc71524af04643580e` (package version 0.2.1, Apache-2.0).
The initial audit used `224c4c7`; upstream harness pinning and operator-limit
changes were merged before final qualification. The maintained fork's ledger and
native-author-control fixes through `1456b1d` are also included.
The implementation is original code against the existing `Harness.run` protocol.
No Hopper, Agent Orchestrator or OpenSwarm runtime code is incorporated.

Research records use schema 1 in a separate root. Existing Outerloop run records
and parked sessions need no migration. An unknown research schema is refused;
missing legacy research roots are simply absent until this command is used.
Existing API-backed harness tests remain part of validation.

Authentication references: [Claude Code](https://code.claude.com/docs/en/authentication)
and [Codex](https://developers.openai.com/codex/auth/).

## Explicit API and Vertex authentication

For Claude API calls, create a dedicated writable native profile for session
history, and mount or provision an owner-only0600 file containing the API key.
Select these fields on the worker; keep the existing explicit model and prompt:

```json
{
  "id": "researcher",
  "backend": "claude",
  "profile": "/private/native-profile",
  "binary": "/absolute/path/to/claude",
  "model": "YOUR_EXPLICIT_MODEL",
  "auth_mode": "api-key",
  "api_key_file": "/private/credential-file",
  "workspace_id": "YOUR_WORKSPACE_ID",
  "prompt": "Research the stated question using primary sources."
}
```

Omit `workspace_id` for a workspace-scoped key. An organization-scoped key
requires the actual workspace ID; the driver passes it in
`ANTHROPIC_CUSTOM_HEADERS` as `anthropic-workspace-id`. The native CLI reads
only the selected key file. Other ambient API credentials and alternate model
endpoints remain excluded. Native auth status establishes the selected mode;
model/workspace permission and quota are established by an actual native turn.

Codex accepts explicit `auth_mode: "api-key"` with a private `api_key_file`,
or a native profile previously authenticated through `codex login --with-api-key`.
It runs with `forced_login_method="api"`. Neither API mode converts or copies a
subscription login.

For Claude on Vertex, select `auth_mode: "vertex"` and public coordinates:

```json
{"vertex":{"project":"YOUR_PROJECT","region":"global","adc_file":"","small_model":"YOUR_EXPLICIT_MODEL"}}
```

An empty `adc_file` uses service metadata/Workload Identity. An explicit file
uses that ADC path, with no credential content in the plan. This mode retains
the same persistent native session, containment and research tool policy; it
does not force a Claude subscription login. Prepare the native executable and
profile inside the actual Linux host/container. Changing auth mode, workspace,
credential path or Vertex coordinates refuses resume rather than silently
switching the conversation's provider. Do not commit any credential, native
transcript or execution artifact.
