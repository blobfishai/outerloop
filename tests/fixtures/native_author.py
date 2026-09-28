"""Deterministic local provider protocol, including persistent native state.

Configuration lives in the test workspace. No provider or network is contacted.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

if "login" in sys.argv:
    sys.stdin.read()
    raise SystemExit(0)

workspace = Path.cwd()
config = json.loads((workspace / "fixture.json").read_text())
backend = config["backend"]
mode = config.get("mode", "normal")
home = Path(os.environ["HOME"])
state_path = home / "fixture-state.json"
resume = None
if "--resume" in sys.argv:
    resume = sys.argv[sys.argv.index("--resume") + 1]
elif "resume" in sys.argv:
    resume = sys.argv[sys.argv.index("resume") + 1]
if resume:
    state = json.loads(state_path.read_text())
    assert resume == state["id"]
    state["counter"] += 1
else:
    assert not state_path.exists(), "unexpected fresh writer"
    state = {"id": "fixture-native-session", "counter": 1}
state_path.write_text(json.dumps(state))
child = None
if mode in ("timeout", "observer", "restart", "completed-child"):
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            ("import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"),
        ]
    )
(workspace / "started.json").write_text(
    json.dumps(
        {
            "pid": os.getpid(),
            "child": child.pid if child else None,
            "resume": resume,
            "home": str(home),
            "counter": state["counter"],
            "argv": sys.argv,
        }
    )
)
if mode == "closed-stdin":
    sys.stdin.close()
else:
    prompt = sys.stdin.read()
    (workspace / "received-prompt.json").write_text(json.dumps(prompt))


def emit(event):
    if backend == "claude-code" and "stream-json" not in sys.argv and event["type"] != "result":
        return
    data = (json.dumps(event, ensure_ascii=False) + "\n").encode()
    if mode == "fragmented":
        for byte in data:
            os.write(1, bytes([byte]))
    else:
        os.write(1, data)


if mode == "malformed":
    os.write(1, b'not json\n[]\n{"type":invalid}\n\xff\n')
if mode == "escaped-secret":
    os.write(1, b'{"type":"fixture.secret","text":"\\u0066ixture-secret"}\n')
if mode == "no-newline":
    os.write(1, b'{"type":')
    time.sleep(30)
if mode == "stderr":
    os.write(2, b"diagnostic" * 200000)
if mode != "missing-id":
    emit(
        {"type": "thread.started", "thread_id": state["id"]}
        if backend == "codex"
        else {"type": "system", "subtype": "init", "session_id": state["id"]}
    )
stamp = time.monotonic_ns()
emit({"type": "fixture.progress", "seq": 7, "emitted_ns": stamp, "text": "résumé"})
if mode == "conflict":
    emit(
        {"type": "thread.started", "thread_id": "another-id"}
        if backend == "codex"
        else {"type": "system", "subtype": "init", "session_id": "another-id"}
    )
if mode in ("timeout", "observer"):
    time.sleep(30)
if mode in ("barrier", "restart"):
    deadline = time.monotonic() + 10
    while not (workspace / "release").exists():
        assert time.monotonic() < deadline, "fixture release watchdog expired"
        time.sleep(0.002)
time.sleep(config.get("delay", 0))
if child and mode != "completed-child":
    child.kill()
    child.wait()
text = f"native counter {state['counter']}"
if backend == "codex":
    if "--output-last-message" in sys.argv:
        Path(sys.argv[sys.argv.index("--output-last-message") + 1]).write_text(text)
    emit({"type": "turn.completed"})
else:
    if mode == "duplicate-result":
        emit(
            {
                "type": "result",
                "subtype": "error",
                "is_error": True,
                "session_id": state["id"],
                "result": "first error",
                "total_cost_usd": 9,
            }
        )
    emit(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "session_id": "" if mode == "missing-id" else state["id"],
            "result": text,
            "num_turns": 1,
            "total_cost_usd": 0,
        }
    )
raise SystemExit(3 if mode == "nonzero" else 0)
