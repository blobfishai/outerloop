"""Bounded parallel research using explicitly selected native subscriptions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from outerloop.subscription import (
    ResearchError,
    SubscriptionHarness,
    SubscriptionProfile,
    exclusive,
    load_state,
    private_dir,
    save_json,
)

Name = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")]


class Worker(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: Name
    backend: str
    profile: str
    binary: str
    model: str
    prompt: Annotated[str, Field(min_length=1, max_length=200_000)]


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = Field(default=1, ge=1, le=1)
    goal: Annotated[str, Field(min_length=1, max_length=100_000)]
    workers: list[Worker] = Field(min_length=1, max_length=8)
    timeout_s: float = Field(default=900, gt=0, le=5400, allow_inf_nan=False)
    idle_timeout_s: float = Field(default=300, gt=0, le=5400, allow_inf_nan=False)
    tools: str = "read"
    max_turns: int = Field(default=24, ge=1, le=120)

    def profiles(self) -> list[SubscriptionProfile]:
        if self.tools not in {"none", "read"} or self.idle_timeout_s > self.timeout_s:
            raise ResearchError("invalid tool policy or idle timeout")
        profiles = []
        for worker in self.workers:
            executable = shutil.which(str(Path(worker.binary).expanduser()))
            if not executable:
                raise ResearchError(f"{worker.id}: native CLI executable not found")
            profiles.append(
                SubscriptionProfile(
                    worker.backend,
                    Path(worker.profile),
                    str(Path(executable).resolve()),
                    worker.model,
                )
            )
        if len({w.id for w in self.workers}) != len(self.workers):
            raise ResearchError("worker IDs must be unique")
        if len({p.directory for p in profiles}) != len(profiles):
            raise ResearchError("each worker needs a distinct native profile")
        return profiles


def _harness(plan: Plan, profile: SubscriptionProfile, worker_dir: Path) -> SubscriptionHarness:
    return SubscriptionHarness(
        profile,
        worker_dir / "state",
        plan.timeout_s,
        plan.idle_timeout_s,
        plan.tools,
        plan.max_turns,
    )


def snapshot(root: Path) -> dict:
    # Acceptance can change a previously completed worker back to active. Hold
    # one short metadata lock over the whole observation so those transitions
    # cannot produce a mixed-time false positive. Native turns do not hold it.
    with exclusive(root / "snapshot.lock", wait=True):
        return _snapshot_unlocked(root)


def _publish_snapshot_unlocked(root: Path) -> dict:
    result = _snapshot_unlocked(root)
    save_json(root / "result.json", result)
    return result


def _snapshot_unlocked(root: Path) -> dict:
    manifest = load_state(root / "plan.json")
    workers = []
    for spec in manifest["plan"]["workers"]:
        state_path = root / spec["id"] / "state" / "state.json"
        receipt = root / spec["id"] / "receipt.json"
        if state_path.exists():
            state = load_state(state_path)
            workers.append({"id": spec["id"], **state})
        elif receipt.exists():
            workers.append({"id": spec["id"], **load_state(receipt)})
        else:
            workers.append({"id": spec["id"], "status": "not-started"})
        requests = sorted(
            (load_state(p) for p in (root / spec["id"] / "requests").glob("*.json")),
            key=lambda receipt: receipt["at"],
        )
        workers[-1]["requests"] = requests
    return {
        "schema": 1,
        "goal": manifest["plan"]["goal"],
        "parallel": manifest["parallel"],
        "complete": all(
            w["status"] == "completed"
            and not any(r["status"] == "accepted" for r in w["requests"])
            and (not w["requests"] or w["requests"][-1]["status"] == "completed")
            for w in workers
        ),
        "workers": workers,
    }


def run_plan(plan: Plan, root: Path, parallel: int) -> dict:
    if not 1 <= parallel <= 4:
        raise ResearchError("parallel must be between 1 and 4")
    profiles = plan.profiles()  # validate every worker before accepting any
    # Freeze resolved paths before persistence. PATH changes or symlink aliases
    # must not silently choose another binary/account when a worker resumes.
    for worker, profile in zip(plan.workers, profiles, strict=True):
        worker.profile, worker.binary = str(profile.directory), profile.binary
    private_dir(root)
    root = root.resolve()
    with exclusive(root / "goal.lock"):
        saved_path = root / "plan.json"
        manifest = {"schema": 1, "plan": plan.model_dump(), "parallel": parallel}
        if saved_path.exists():
            if load_state(saved_path) != manifest:
                raise ResearchError("this root already belongs to another research plan")
            # Retry is observation, never another model call (including after
            # controller death or partial acceptance).
            return snapshot(root)
        for worker, profile in zip(plan.workers, profiles, strict=True):
            worker_dir = root / worker.id
            private_dir(worker_dir)
            private_dir(worker_dir / "workspace")
            home = worker_dir / "state" / "home"
            private_dir(home)
            with exclusive(profile.directory / ".outerloop-research.lock"):
                profile.check_auth(home)
        save_json(saved_path, manifest)
        cancellation = threading.Event()

        def work(pair: tuple[Worker, SubscriptionProfile]) -> None:
            worker, profile = pair
            worker_dir = root / worker.id
            brief = f"Research goal:\n{plan.goal}\n\nYour assignment:\n{worker.prompt}"
            try:
                harness = _harness(plan, profile, worker_dir)
                harness.cancel_event = cancellation
                if not cancellation.is_set():
                    harness.run(brief, worker_dir / "workspace")
            except (ResearchError, OSError, subprocess.SubprocessError) as exc:
                # Preserve a prior active state on exceptions; it deliberately
                # blocks blind replay. This receipt is useful before launch too.
                save_json(
                    worker_dir / "receipt.json",
                    {
                        "schema": 1,
                        "status": "error",
                        "error": str(exc),
                        "at": time.time(),
                    },
                )

        pool = ThreadPoolExecutor(max_workers=min(parallel, len(plan.workers)))
        try:
            list(pool.map(work, zip(plan.workers, profiles, strict=True)))
        except BaseException:
            cancellation.set()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        with exclusive(root / "snapshot.lock", wait=True):
            return _publish_snapshot_unlocked(root)


def resume(root: Path, worker_id: str, prompt: str, request_id: str) -> dict:
    # Validate before using either as a path component.
    import re

    if not all(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", s) for s in (worker_id, request_id)):
        raise ResearchError("invalid worker or request ID")
    root = root.resolve(strict=True)
    if not (root / worker_id).is_dir():
        raise ResearchError("unknown worker")
    with exclusive(root / worker_id / "requests.lock"):
        plan = Plan.model_validate(load_state(root / "plan.json")["plan"])
        profiles = plan.profiles()
        found = [(w, p) for w, p in zip(plan.workers, profiles, strict=True) if w.id == worker_id]
        if not found:
            raise ResearchError("unknown worker")
        _, profile = found[0]
        worker_dir = root / worker_id
        requests = worker_dir / "requests"
        private_dir(requests)
        path = requests / f"{request_id}.json"
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        with exclusive(root / "snapshot.lock", wait=True):
            if path.exists():
                if load_state(path).get("prompt_sha256") != digest:
                    raise ResearchError("request ID already belongs to a different prompt")
                return _snapshot_unlocked(root)
            if any(load_state(p)["status"] == "accepted" for p in requests.glob("*.json")):
                raise ResearchError("an earlier accepted request needs reconciliation")
            state = load_state(worker_dir / "state" / "state.json")
            if state["status"] in {"starting", "running"} or not state.get("session_id"):
                raise ResearchError("worker has no finished turn with a resumable identity")
            # Acceptance and its published incomplete snapshot commit while new
            # snapshot readers are excluded. No completed cache survives a start.
            receipt = {
                "schema": 1,
                "request_id": request_id,
                "prompt_sha256": digest,
                "status": "accepted",
                "at": time.time(),
            }
            save_json(path, receipt)
            _publish_snapshot_unlocked(root)
        try:
            result = _harness(plan, profile, worker_dir).run(
                prompt,
                worker_dir / "workspace",
                resume_session_id=state["session_id"],
            )
        except (ResearchError, OSError, subprocess.SubprocessError):
            with exclusive(root / "snapshot.lock", wait=True):
                receipt.update(status="error", finished_at=time.time())
                save_json(path, receipt)
                _publish_snapshot_unlocked(root)
            raise
        with exclusive(root / "snapshot.lock", wait=True):
            receipt.update(status=result.stop_reason, finished_at=time.time())
            save_json(path, receipt)
            return _publish_snapshot_unlocked(root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="outerloop research")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="execute a bounded subscription research plan")
    run.add_argument("manifest", type=Path)
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--parallel", type=int, default=2)
    status = commands.add_parser("status", help="read durable worker results without launching")
    status.add_argument("--root", type=Path, required=True)
    follow = commands.add_parser("resume", help="send one idempotent native follow-up")
    follow.add_argument("--root", type=Path, required=True)
    follow.add_argument("--worker", required=True)
    follow.add_argument("--request-id", required=True)
    follow.add_argument("--prompt-file", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = run_plan(
                Plan.model_validate_json(args.manifest.read_text()), args.root, args.parallel
            )
        elif args.command == "resume":
            result = resume(args.root, args.worker, args.prompt_file.read_text(), args.request_id)
        else:
            result = snapshot(args.root)
        print(json.dumps(result, indent=2))
        return 0 if result["complete"] else 1
    except KeyboardInterrupt:
        print("outerloop research: interrupted; inspect the retained worker state", file=sys.stderr)
        return 130
    except (ResearchError, ValidationError, OSError, subprocess.SubprocessError) as exc:
        print(f"outerloop research: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
