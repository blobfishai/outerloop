"""hermes as the climbing author, beside claude and codex: its key file, its
startup checks, one harness construction for the climb and the wake, the
tick's preflight, `start`, init, and the panel it defaults to."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

import outerloop.attempt as climb_mod
from outerloop.attempt import (
    HERMES_KEY_DEFAULT,
    author_config_error,
    resolve_author_key_file,
    resume_author,
)
from outerloop.harness import HERMES_ENDPOINT_KEY_ENV, HermesHarness
from outerloop.hermes_install import HERMES_SHA, hermes_runtime

ENDPOINT = "https://models.example.com/v1"
NOW = 1_000_000.0
ROOT = Path(__file__).resolve().parents[1]


def _ready(repo: Path) -> Path:
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "run_agent.py").touch()
    runtime = hermes_runtime(repo)
    (runtime / "venv/bin").mkdir(parents=True, exist_ok=True)
    python = runtime / "venv/bin/python"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    (runtime / ".complete").write_text(HERMES_SHA)
    return repo


def _key(path: Path, value: str) -> Path:
    path.write_text(value + "\n")
    path.chmod(0o600)
    return path


@pytest.fixture
def hermes_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """A deployment whose author is hermes on an OpenAI-compatible endpoint."""
    paths = {
        "repo": _ready(tmp_path / "hermes-agent"),
        "key": _key(tmp_path / "hermes_key", "sk-hermes-author"),
        "image": tmp_path / "img.sif",
        "pat": _key(tmp_path / "pat", "ghp_x"),
    }
    paths["image"].write_text("")
    monkeypatch.setenv("REVIEW_HERMES_REPO", str(paths["repo"]))
    monkeypatch.setenv("OUTERLOOP_HERMES_KEY_FILE", str(paths["key"]))
    monkeypatch.setenv("OUTERLOOP_HERMES_PROVIDER", "custom")
    monkeypatch.setenv("OUTERLOOP_HERMES_BASE_URL", ENDPOINT)
    monkeypatch.setenv("OUTERLOOP_AUTHOR_BACKEND", "hermes")
    monkeypatch.setenv("OUTERLOOP_AUTHOR_MODEL", "open-model")
    monkeypatch.setattr(climb_mod, "arm_sigterm_containment", lambda: None)
    return paths


# ------------------------------------------------------------- the rules


def test_the_author_key_file_resolves_per_backend(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("OUTERLOOP_HERMES_KEY_FILE", raising=False)
    assert resolve_author_key_file("hermes") == str(Path(HERMES_KEY_DEFAULT).expanduser())
    monkeypatch.setenv("OUTERLOOP_HERMES_KEY_FILE", "~/keys/hermes")
    assert resolve_author_key_file("hermes") == str(Path("~/keys/hermes").expanduser())
    assert resolve_author_key_file("hermes", "/explicit") == "/explicit"


def test_startup_checks(hermes_env, monkeypatch) -> None:
    image = str(hermes_env["image"])
    assert author_config_error("hermes", "open-model", image) == ""
    assert "requires --image" in author_config_error("hermes", "open-model", "")
    assert "OUTERLOOP_AUTHOR_MODEL" in author_config_error("hermes", "", image)
    monkeypatch.setenv("OUTERLOOP_HERMES_BASE_URL", "")
    assert "needs a base URL" in author_config_error("hermes", "open-model", image)
    monkeypatch.setenv("OUTERLOOP_HERMES_PROVIDER", "openrouter")
    assert author_config_error("hermes", "vendor/open-model", image) == ""
    monkeypatch.setenv("OUTERLOOP_HERMES_BASE_URL", ENDPOINT)
    assert "custom provider" in author_config_error("hermes", "open-model", image)
    monkeypatch.setenv("REVIEW_HERMES_REPO", str(hermes_env["repo"].parent / "absent"))
    assert "install_hermes.sh" in author_config_error("hermes", "open-model", image)
    assert "unknown author backend" in author_config_error("hermez", "m", image)


def test_a_parked_hermes_run_wakes_as_hermes(tmp_path) -> None:
    class Record:
        author_backend = "hermes"
        author_model = "open-model"
        author_key_file = "/keys/hermes"
        agent_id = "agent-01"

    # a claude or codex fleet never turns a hermes run into another backend
    assert resume_author(Record(), "gpt-x", "codex") == ("hermes", "open-model", "/keys/hermes")


# ------------------------------------------------------------- the climb


def _climb_argv(paths: dict[str, Path], state: Path) -> list[str]:
    state.mkdir(exist_ok=True)
    return [
        "climb",
        "--target",
        "o/r",
        "--benchmark",
        "b",
        "--run-root",
        str(state),
        "--image",
        str(paths["image"]),
        "--pat-file",
        str(paths["pat"]),
        "--panel",
        "",
        "--min-free-gb",
        "0",
    ]


def test_a_fresh_climb_authors_on_hermes(hermes_env, monkeypatch, tmp_path) -> None:
    seen: dict[str, Any] = {}

    def fake_live(**kwargs: Any) -> climb_mod.AttemptOutcome:
        seen.update(kwargs)
        return climb_mod.AttemptOutcome(run_id="r", outcome="no-improvement")

    monkeypatch.setattr(climb_mod, "live_attempt", fake_live)
    monkeypatch.setattr("sys.argv", _climb_argv(hermes_env, tmp_path / "state"))
    assert climb_mod.main() == 0
    harness = seen["harness"]
    assert isinstance(harness, HermesHarness)
    assert harness.provider == "custom:outerloop" and harness.base_url == ENDPOINT
    assert harness.key_env == HERMES_ENDPOINT_KEY_ENV
    assert harness.api_key == "sk-hermes-author"
    assert harness.model == "open-model"
    assert harness.container_image == str(hermes_env["image"])
    assert Path(harness.repo_dir) == hermes_env["repo"]
    assert "terminal" in harness.enabled_toolsets  # the author executes
    # the run records its author, so a wake reproduces it
    assert (seen["author_backend"], seen["author_model"]) == ("hermes", "open-model")
    assert seen["author_key_file"] == str(hermes_env["key"])
    assert "sk-hermes-author" in seen["secrets"]


def test_a_misconfigured_hermes_climb_never_starts(hermes_env, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(climb_mod, "live_attempt", lambda **k: pytest.fail("must not start"))
    monkeypatch.delenv("OUTERLOOP_HERMES_BASE_URL")
    monkeypatch.setattr("sys.argv", _climb_argv(hermes_env, tmp_path / "state"))
    with pytest.raises(SystemExit):
        climb_mod.main()
    assert "needs a base URL" in capsys.readouterr().err


def test_a_wake_rebuilds_the_hermes_author(hermes_env, monkeypatch, tmp_path) -> None:
    from outerloop.runstate import RunRecord, acquire_lease, save_record

    run_id = "b-hermes"
    save_record(
        tmp_path,
        RunRecord(
            run_id=run_id,
            target="o/r",
            task_title="t",
            benchmark="b",
            state="parked",
            deadline=NOW + 3600,
            stage={"phase": "author-sleep", "candidate_sha": "abc"},
            author_backend="hermes",
            author_model="open-model",
            author_key_file=str(hermes_env["key"]),
        ),
        NOW,
    )
    assert acquire_lease(tmp_path, run_id, "wake-job:1", "1", NOW)
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.delenv("OUTERLOOP_DISPATCH_WAKE", raising=False)
    # the fleet moved to codex since this run parked: the wake stays hermes
    monkeypatch.setenv("OUTERLOOP_AUTHOR_BACKEND", "codex")
    monkeypatch.setenv("OUTERLOOP_AUTHOR_MODEL", "gpt-x")
    seen: dict[str, Any] = {}

    def fake_resume(*args: Any, **kwargs: Any) -> climb_mod.AttemptOutcome:
        seen.update(kwargs)
        return climb_mod.AttemptOutcome(run_id=run_id, outcome="parked")

    monkeypatch.setattr(climb_mod, "resume_run", fake_resume)
    argv = ["climb", "--resume", run_id, "--run-root", str(tmp_path)]
    argv += ["--image", str(hermes_env["image"]), "--pat-file", str(hermes_env["pat"])]
    monkeypatch.setattr("sys.argv", [*argv, "--panel", ""])
    assert climb_mod.main() == 0
    harness = seen["harness"]
    assert isinstance(harness, HermesHarness)
    assert harness.model == "open-model" and harness.base_url == ENDPOINT
    assert harness.api_key == "sk-hermes-author"


# ------------------------------------------------------------- the panel


def test_the_default_panel_of_a_hermes_author_is_hermes(hermes_env, monkeypatch, tmp_path):
    import argparse

    judge = _key(tmp_path / "hermes_judge", "sk-hermes-judge")
    monkeypatch.setenv("OUTERLOOP_PANEL_HERMES_KEY_FILE", str(judge))
    monkeypatch.setenv("REVIEW_HERMES_PROVIDER", "custom")
    monkeypatch.setenv("REVIEW_HERMES_BASE_URL", "https://judges.example.com/v1")
    args = argparse.Namespace(
        panel="verify,review:hermes:judge-model",
        panel_key_file=str(tmp_path / "no-claude-key"),
        claude_bin="claude",
        codex_bin="codex",
        image=str(hermes_env["image"]),
    )
    lenses, secrets = climb_mod._panel_lenses_from_args(
        args,
        author_backend="hermes",
        author_model="open-model",
        author_key_file=str(hermes_env["key"]),
    )
    judges = [lens.harness for lens in lenses if isinstance(lens.harness, HermesHarness)]
    assert [lens.kind for lens in lenses] == ["verify", "review"] and len(judges) == 2
    assert [judge.model for judge in judges] == ["open-model", "judge-model"]
    assert {judge.base_url for judge in judges} == {"https://judges.example.com/v1"}
    assert secrets == ("sk-hermes-judge",)
    # the judge never runs on the hermes author's key file
    monkeypatch.setenv("OUTERLOOP_PANEL_HERMES_KEY_FILE", str(hermes_env["key"]))
    with pytest.raises(ValueError, match="role separation"):
        climb_mod._panel_lenses_from_args(
            args,
            author_backend="hermes",
            author_model="open-model",
            author_key_file=str(hermes_env["key"]),
        )


# -------------------------------------------------------------- the tick


def test_the_tick_preflights_a_hermes_fleet(hermes_env, monkeypatch, tmp_path) -> None:
    from outerloop.tick import ServiceSpec, _author_config_error, _panel_preflight_error

    spec = ServiceSpec(
        target="org/pilot",
        account="a",
        partition="p",
        run_root=tmp_path,
        image=str(hermes_env["image"]),
        home=tmp_path,
        panel="verify:hermes:judge-model",
    )
    assert _author_config_error(spec) == ""
    monkeypatch.delenv("OUTERLOOP_AUTHOR_MODEL")
    assert "OUTERLOOP_AUTHOR_MODEL" in _author_config_error(spec)
    # a hermes judge holding a copy of the hermes author's key is refused
    monkeypatch.setenv(
        "OUTERLOOP_PANEL_HERMES_KEY_FILE",
        str(_key(tmp_path / "judge_copy", "sk-hermes-author")),
    )
    monkeypatch.setenv("REVIEW_HERMES_PROVIDER", "openai")
    assert "holds the author key" in _panel_preflight_error(spec)


def test_harness_upgrade_counts_a_hermes_author_as_used() -> None:
    from outerloop.harness_cli import used_harnesses

    assert "hermes" in used_harnesses({"OUTERLOOP_AUTHOR_BACKEND": "hermes", "OUTERLOOP_PANEL": ""})


# ------------------------------------------------------------------ init


def test_init_records_a_hermes_author(monkeypatch, tmp_path) -> None:
    from outerloop import init

    monkeypatch.setattr(init, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(init, "ensure_image", lambda **kw: "")
    target = tmp_path / "src" / "hermes-agent"
    monkeypatch.setenv("REVIEW_HERMES_REPO", str(target))
    calls: list[list[str]] = []

    def run(argv: list[str], *, check: bool) -> None:
        calls.append(argv)
        assert argv == ["bash", str(ROOT / "scripts/install_hermes.sh"), str(target)]
        _ready(target)

    monkeypatch.setattr(init.subprocess, "run", run)
    args = ["--yes", "--compute", "local", "--target", "o/r", "--author-backend", "hermes"]
    assert init.main(args) == 0
    env = (tmp_path / "config" / ".env").read_text()
    assert "OUTERLOOP_AUTHOR_BACKEND=hermes" in env
    assert f"REVIEW_HERMES_REPO={target}" in env
    assert len(calls) == 1  # the missing runtime was installed once
    # a ready checkout is recorded, not reinstalled
    assert init.main([*args, "--force"]) == 0
    assert len(calls) == 1
    assert init.locate_harness("hermes") == str(target)
    assert init.author_bin_env("hermes") == "REVIEW_HERMES_REPO"
    assert init.author_key_env("hermes") == "OUTERLOOP_HERMES_KEY_FILE"


def test_init_keeps_the_hermes_endpoint_settings(monkeypatch, tmp_path) -> None:
    from outerloop import init

    monkeypatch.setattr(init, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(init, "ensure_image", lambda **kw: "")
    monkeypatch.setenv("REVIEW_HERMES_REPO", str(_ready(tmp_path / "hermes-agent")))
    monkeypatch.setenv("OUTERLOOP_HERMES_PROVIDER", "custom")
    monkeypatch.setenv("OUTERLOOP_HERMES_BASE_URL", ENDPOINT)
    monkeypatch.setattr(init.subprocess, "run", lambda *a, **k: pytest.fail("no install"))
    args = ["--yes", "--compute", "local", "--target", "o/r", "--author-backend", "hermes"]
    assert init.main(args) == 0
    env = (tmp_path / "config" / ".env").read_text()
    assert "OUTERLOOP_HERMES_PROVIDER=custom" in env
    assert f"OUTERLOOP_HERMES_BASE_URL={ENDPOINT}" in env


def test_the_installer_script_is_the_hermes_installer() -> None:
    # the init path above runs the real script path; keep it pointing at a file
    assert (ROOT / "scripts/install_hermes.sh").is_file()
    assert subprocess.run(["bash", "-n", str(ROOT / "scripts/install_hermes.sh")]).returncode == 0


def test_init_finds_a_checkout_recorded_only_in_the_env_file(monkeypatch, tmp_path) -> None:
    """A rerun of init on a deployment whose .env records the checkout (and
    whose shell does not export it) keeps that checkout: no second install
    into the default directory, and the record stays."""
    from outerloop import init

    config = tmp_path / "config"
    config.mkdir()
    monkeypatch.setattr(init, "CONFIG_DIR", config)
    monkeypatch.setattr(init, "ensure_image", lambda **kw: "")
    monkeypatch.delenv("REVIEW_HERMES_REPO", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo = _ready(tmp_path / "elsewhere" / "hermes-agent")
    env = config / ".env"
    env.write_text(f"OUTERLOOP_AUTHOR_BACKEND=hermes\nREVIEW_HERMES_REPO={repo}\n")
    env.chmod(0o600)
    monkeypatch.setattr(init.subprocess, "run", lambda *a, **k: pytest.fail("no reinstall"))
    assert init.locate_harness("hermes") == str(repo)
    args = ["--yes", "--force", "--compute", "local", "--target", "o/r"]
    assert init.main([*args, "--author-backend", "hermes"]) == 0
    assert f"REVIEW_HERMES_REPO={repo}" in env.read_text()
