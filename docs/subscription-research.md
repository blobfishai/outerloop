# Parallel research with native subscriptions

This fork adds `outerloop research`: give one goal to a bounded set of Claude
Code and Codex workers, collect their answers, and resume their native sessions.
Each worker uses an explicitly selected subscription login and model. Authentication
failure stops that worker; there is no API-key fallback or account rotation.

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
closed-book fixtures; `read` permits research tools without code execution or
workspace edits. Claude has Read/Glob/Grep/WebSearch/WebFetch; Codex has native
web search with its shell and extension tools disabled. Their tool surfaces are
different; control the supplied context when comparing their results. Put shared
source material in the goal or assignment text. There is no automatic task splitting,
code experiment launch, synthesis, evaluator, or cost-based account switching.

Walltime is bounded to 90 minutes per turn, default 15 minutes. The idle deadline
is separate and resets on stdout activity. `max_turns` applies to Claude's native
limit; Codex executes one native turn under the walltime limit. Provider quota
errors are failures and are never retried using another credential.

The root holds private plan, state, result and normalized event files outside each
worker's directory. Raw auth responses and partial reasoning are not journaled.
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
`224c4c7a48bb62b51537cc5ad46fdf710da21cc3` (package version 0.2.1, Apache-2.0).
The implementation is original code against the existing `Harness.run` protocol.
No Hopper, Agent Orchestrator or OpenSwarm runtime code is incorporated.

Research records use schema 1 in a separate root. Existing Outerloop run records
and parked sessions need no migration. An unknown research schema is refused;
missing legacy research roots are simply absent until this command is used.
Existing API-backed harness tests remain part of validation.

Authentication references: [Claude Code](https://code.claude.com/docs/en/authentication)
and [Codex](https://developers.openai.com/codex/auth/).
