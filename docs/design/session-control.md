# Native author session control

Fresh research attempts and wake legs wrap the existing Codex or Claude Code
harness. The kernel still decides research outcomes, budgets, inbox delivery,
evaluation and PR policy. Judges and other harnesses retain their existing path.

The author streams native JSONL through one bounded pipe reader. It drains
stderr and writes stdin concurrently, handles fragmented UTF-8, and kills the
whole local process group on timeout or observer failure. Frames are limited to
4 MiB, total stdout to 64 MiB, and retained stderr to its last 1 MiB. Native IDs
and completion events must be valid; an init event is never a final result.
Claude uses `--output-format stream-json --verbose`; Codex retains `exec --json`.

`session-control/binding.json` is the canonical author identity. Atomic writes
flush the file and directory. Each turn gets a generation; conflicting identities
and stale writers fail. A private execution lock covers the turn. The process
group is recorded before sending the prompt. Events are redacted before their
private journal is flushed, and binding persistence precedes event publication.
Control files are outside the workspace and its native home, which are the only
writable mounts exposed to a contained author.

Consumers use `outerloop.session_control.read_binding(run_dir)` and
`read_events(run_dir, generation)`. Journal envelopes contain `generation`,
`sequence` and the native `event`. An incomplete final journal line is invisible.
The general run record projects the canonical session ID when read or written;
an old in-memory record cannot erase a newer binding or resurrect an ended run.

## Recovery and compatibility

Legacy records without the sidecar retain their existing behavior. Their next
controlled turn imports the existing resume ID and home without a backfill.
Missing or corrupt sidecars cannot silently authorize a replacement author.
An invalid sidecar leaves legacy record reads available, but author admission
refuses it. A bound home must still exist.

After a controller dies, a replacement checks the prior recorded process group.
A live or inaccessible group, an unresolved foreign host, or an interrupted
launch without a recorded PID blocks execution. A normally returned turn records
completed process-group termination and can resume on another node. Once an
unresolved local group is gone, another controller can
resume the same native session. A reused PID can conservatively block recovery;
it never authorizes killing an unrelated process. Cross-host recovery requires
operator reconciliation. An unrecorded spawn is intentionally unresolved.

There is no automatic orphan termination after controller SIGKILL and no claim
to contain a descendant that leaves the process group with setsid. Use the
existing container/job boundary for that isolation. Local filesystem flushes do
not establish power-loss durability on every shared filesystem. Run state and
the sidecar are not one transaction; the sidecar is authoritative for identity.
Neither binding nor progress acknowledges an inbox message or changes an ending.

Drain authors before switching kernel versions. Retain the sidecar and native
home together. Older kernels ignore this additive directory and cannot enforce
its writer exclusion, so do not run old and new controllers concurrently.
