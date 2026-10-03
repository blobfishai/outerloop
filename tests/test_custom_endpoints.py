"""Panel judges on a custom OpenAI-compatible endpoint: a hermes judge's
provider and base URL, a codex judge's `-c` config, the tick preflight that
mirrors both, and the rule that a judge never holds the author's key."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

import outerloop.attempt as climb_mod
import outerloop.harness as harness_mod
from outerloop.harness import HERMES_ENDPOINT_KEY_ENV, HermesHarness
from outerloop.hermes_install import HERMES_SHA, hermes_runtime
from outerloop.role_runner import build_harness, hermes_endpoint_error
from outerloop.roles import reviewer_spec

ENDPOINT = "https://models.example.com/v1"


def _ready_hermes(root: Path) -> Path:
    """A hermes checkout whose pinned runtime looks installed."""
    repo = root / "hermes-agent"
    repo.mkdir(exist_ok=True)
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


# ------------------------------------------------------------ the rules


@pytest.mark.parametrize(
    ("provider", "url", "problem"),
    [
        ("", "", ""),
        ("openai", "", ""),
        ("OpenRouter", "", ""),
        ("custom", ENDPOINT, ""),
        ("custom", "http://10.0.0.5:8000/v1", ""),
        ("custom", "", "needs a base URL"),
        ("openrouter", ENDPOINT, "needs the custom provider"),
        ("openai", ENDPOINT, "never be sent to another host"),
        ("custom", "ftp://models.example.com", "plain http(s)"),
        ("custom", "https://user:pw@models.example.com/v1", "plain http(s)"),
        ("custom", 'https://models.example.com/v1"', "plain http(s)"),
        ("custom", "https://models.example.com/v 1", "plain http(s)"),
        ("custom", "https:///v1", "plain http(s)"),
        ("wat", "", "unknown hermes provider"),
    ],
)
def test_hermes_endpoint_rules(provider: str, url: str, problem: str) -> None:
    error = hermes_endpoint_error(provider, url)
    assert (problem in error) if problem else error == ""


def test_build_harness_points_a_custom_hermes_at_its_endpoint(tmp_path: Path) -> None:
    h = build_harness(
        "sk-judge",
        reviewer_spec(),
        backend="hermes",
        model="open-model",
        hermes_repo=tmp_path,
        hermes_provider="custom",
        hermes_base_url=ENDPOINT,
    )
    assert isinstance(h, HermesHarness)
    assert h.provider == "custom:outerloop"
    assert h.key_env == HERMES_ENDPOINT_KEY_ENV
    assert h.base_url == ENDPOINT
    with pytest.raises(ValueError, match="needs a base URL"):
        build_harness("k", reviewer_spec(), backend="hermes", hermes_repo=tmp_path,
                      hermes_provider="custom")  # fmt: skip
    with pytest.raises(ValueError, match="custom provider"):
        build_harness("k", reviewer_spec(), backend="hermes", hermes_repo=tmp_path,
                      hermes_base_url=ENDPOINT)  # fmt: skip


# ------------------------------------------------------- the hermes session


def _capture_session(monkeypatch: Any, home: Path) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    class FakePopen:
        returncode = 0

        def __init__(self, command: list[str], **kwargs: Any) -> None:
            seen["argv"] = command
            seen["env"] = kwargs.get("env", {})
            seen["config"] = (home / ".hermes" / "config.yaml").read_text()

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            return "", ""

    monkeypatch.setattr(harness_mod.subprocess, "Popen", FakePopen)
    return seen


def test_a_custom_endpoint_is_a_named_provider_entry(tmp_path, monkeypatch) -> None:
    repo = _ready_hermes(tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()
    seen = _capture_session(monkeypatch, tmp_path / "ws-home")
    HermesHarness(
        api_key="sk-endpoint-SECRET",
        repo_dir=repo,
        provider="custom:outerloop",
        key_env=HERMES_ENDPOINT_KEY_ENV,
        model="open-model",
        base_url=ENDPOINT,
    ).run("brief", ws)
    assert seen["config"] == (
        "model:\n"
        '  default: "open-model"\n'
        '  provider: "custom:outerloop"\n'
        "providers:\n"
        "  outerloop:\n"
        f'    base_url: "{ENDPOINT}"\n'
        f'    key_env: "{HERMES_ENDPOINT_KEY_ENV}"\n'
        '    default_model: "open-model"\n'
    )
    # the key rides the session environment under the name the entry gives,
    # never argv and never the config file
    assert seen["env"][HERMES_ENDPOINT_KEY_ENV] == "sk-endpoint-SECRET"
    assert "sk-endpoint-SECRET" not in " ".join(seen["argv"])
    assert "sk-endpoint-SECRET" not in seen["config"]
    assert f"--base_url={ENDPOINT}" in seen["argv"]


def test_registry_providers_keep_their_config(tmp_path, monkeypatch) -> None:
    repo = _ready_hermes(tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()
    seen = _capture_session(monkeypatch, tmp_path / "ws-home")
    HermesHarness(api_key="k", repo_dir=repo, provider="openrouter", model="m/x").run("b", ws)
    assert seen["config"] == 'model:\n  default: "m/x"\n  provider: "openrouter"\n'


def test_a_custom_provider_without_its_url_never_starts(tmp_path, monkeypatch) -> None:
    repo = _ready_hermes(tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()

    def unexpected(*a: Any, **k: Any) -> None:
        pytest.fail("a custom provider without a base URL must not spawn hermes")

    monkeypatch.setattr(harness_mod.subprocess, "Popen", unexpected)
    result = HermesHarness(api_key="k", repo_dir=repo, provider="custom:outerloop").run("b", ws)
    assert result.is_error and result.stop_reason == "config-error"


# --------------------------------------------------------------- the panel


def _panel_args(tmp_path: Path, panel: str, **extra: Any) -> argparse.Namespace:
    return argparse.Namespace(
        panel=panel,
        panel_key_file=str(tmp_path / "no-claude-panel-key"),
        claude_bin="claude",
        codex_bin="/opt/codex",
        image="/img.sif",
        **extra,
    )


@pytest.fixture
def judge_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Judge keys of their own, author keys elsewhere, hermes installed."""
    monkeypatch.setenv(
        "OUTERLOOP_PANEL_HERMES_KEY_FILE", str(_key(tmp_path / "hermes_judge", "sk-hermes-judge"))
    )
    monkeypatch.setenv(
        "OUTERLOOP_PANEL_CODEX_KEY_FILE", str(_key(tmp_path / "codex_judge", "sk-codex-judge"))
    )
    monkeypatch.setenv("OUTERLOOP_CODEX_KEY_FILE", str(_key(tmp_path / "codex_key", "sk-codex-a")))
    monkeypatch.setenv("REVIEW_HERMES_REPO", str(_ready_hermes(tmp_path)))
    monkeypatch.delenv("REVIEW_HERMES_PROVIDER", raising=False)
    monkeypatch.delenv("REVIEW_HERMES_BASE_URL", raising=False)
    return tmp_path


def test_a_hermes_judge_reaches_a_custom_endpoint(judge_env, monkeypatch) -> None:
    monkeypatch.setenv("REVIEW_HERMES_PROVIDER", "Custom")
    monkeypatch.setenv("REVIEW_HERMES_BASE_URL", ENDPOINT)
    lenses, secrets = climb_mod._panel_lenses_from_args(
        _panel_args(judge_env, "verify:hermes:judge-model")
    )
    (lens,) = lenses
    assert isinstance(lens.harness, HermesHarness)
    assert lens.harness.provider == "custom:outerloop" and lens.harness.base_url == ENDPOINT
    assert lens.harness.model == "judge-model"
    assert secrets == ("sk-hermes-judge",)


@pytest.mark.parametrize(
    ("provider", "url", "problem"),
    [("custom", "", "needs a base URL"), ("openrouter", ENDPOINT, "custom provider")],
)
def test_a_hermes_judge_endpoint_misconfiguration_is_loud(
    judge_env, monkeypatch, provider, url, problem
) -> None:
    monkeypatch.setenv("REVIEW_HERMES_PROVIDER", provider)
    monkeypatch.setenv("REVIEW_HERMES_BASE_URL", url)
    with pytest.raises(ValueError, match=problem):
        climb_mod._panel_lenses_from_args(_panel_args(judge_env, "review:hermes:judge-model"))


def test_a_codex_judge_gets_the_author_config_then_its_own(judge_env) -> None:
    from outerloop.harness import CodexHarness

    args = _panel_args(
        judge_env,
        "review:codex:judge-model",
        codex_extra=("-c", "model_provider=local", "-c", "model_providers.local.base_url=a"),
        panel_codex_extra=("-c", "model_providers.local.base_url=b"),
    )
    (lens,), _ = climb_mod._panel_lenses_from_args(args)
    assert isinstance(lens.harness, CodexHarness)
    # codex applies -c in order: the panel's own value for a shared key wins
    assert lens.harness.extra_args[:6] == (
        "-c",
        "model_provider=local",
        "-c",
        "model_providers.local.base_url=a",
        "-c",
        "model_providers.local.base_url=b",
    )


@pytest.mark.parametrize("which", ["codex_config_error", "panel_codex_config_error"])
def test_a_codex_judge_refuses_a_malformed_config(judge_env, which) -> None:
    args = _panel_args(judge_env, "review:codex:judge-model", **{which: "bad entry"})
    with pytest.raises(ValueError, match="bad entry"):
        climb_mod._panel_lenses_from_args(args)


def test_a_copy_of_the_anthropic_key_never_reaches_another_provider(judge_env) -> None:
    claude_panel = _key(judge_env / "verifier_key", "sk-ant-panel")
    _key(judge_env / "hermes_judge", "sk-ant-panel")  # the judge file holds a COPY
    args = _panel_args(judge_env, "review:hermes:judge-model")
    args.panel_key_file = str(claude_panel)
    with pytest.raises(ValueError, match="another provider"):
        climb_mod._panel_lenses_from_args(args)


def test_a_judge_key_file_is_never_the_runs_author_key_file(judge_env) -> None:
    author = _key(judge_env / "some_author_key", "sk-author")
    args = _panel_args(judge_env, "review:hermes:judge-model")
    with pytest.raises(ValueError, match="role separation"):
        climb_mod._panel_lenses_from_args(
            args,
            author_backend="claude",
            author_key_file=str(judge_env / "hermes_judge"),
        )
    # a separate author file is fine
    lenses, _ = climb_mod._panel_lenses_from_args(
        args, author_backend="claude", author_key_file=str(author)
    )
    assert len(lenses) == 1


def test_a_climb_refuses_a_judge_holding_the_authors_key(tmp_path, monkeypatch, capsys) -> None:
    """The wake skips the panel on equal key values; a climb started by hand
    refuses outright, the same rule the tick preflight applies before queueing."""
    pat = _key(tmp_path / "pat", "ghp_x")
    author = _key(tmp_path / "claude_key", "sk-ant-same")
    verifier = _key(tmp_path / "verifier_key", "sk-ant-same")
    image = tmp_path / "img.sif"
    image.write_text("")
    (tmp_path / "state").mkdir()
    monkeypatch.setenv("OUTERLOOP_CLAUDE_KEY_FILE", str(author))
    monkeypatch.setattr(climb_mod, "arm_sigterm_containment", lambda: None)
    monkeypatch.setattr(climb_mod, "build_harness", lambda *a, **k: object())
    ran: list[Any] = []
    monkeypatch.setattr(climb_mod, "live_attempt", lambda **k: ran.append(k))
    argv = ["climb", "--target", "o/r", "--benchmark", "b", "--run-root", str(tmp_path / "state")]
    argv += ["--image", str(image), "--pat-file", str(pat), "--min-free-gb", "0"]
    monkeypatch.setattr("sys.argv", [*argv, "--panel", "verify", "--panel-key-file", str(verifier)])
    with pytest.raises(SystemExit):
        climb_mod.main()
    assert "holds the author's key" in capsys.readouterr().err
    assert ran == []


# ---------------------------------------------------------------- the tick


def _spec(tmp_path: Path, **kw: Any):
    from outerloop.tick import ServiceSpec

    image = tmp_path / "image.sif"
    image.touch()
    return ServiceSpec(
        target="org/pilot",
        account="a",
        partition="p",
        run_root=tmp_path,
        image=str(image),
        home=tmp_path,
        **kw,
    )


def test_the_tick_forwards_the_panel_codex_config(tmp_path) -> None:
    from outerloop.tick import _climb_panel_argv

    spec = _spec(tmp_path, panel="review:codex:m", panel_codex_config="a=1; b=2")
    assert _climb_panel_argv(spec) == [
        "--panel",
        "review:codex:m",
        "--panel-codex-config",
        "a=1",
        "--panel-codex-config",
        "b=2",
    ]
    assert _climb_panel_argv(_spec(tmp_path, panel="", panel_codex_config="a=1")) == []
    malformed = _spec(tmp_path, panel="review:codex:m", panel_codex_config="nope")
    assert _climb_panel_argv(malformed) == ["--panel", "review:codex:m"]


def test_preflight_mirrors_the_hermes_endpoint_rules(judge_env, monkeypatch) -> None:
    from outerloop.tick import _panel_preflight_error

    spec = _spec(judge_env, panel="review:hermes:judge-model")
    monkeypatch.setenv("REVIEW_HERMES_PROVIDER", "custom")
    assert "needs a base URL" in _panel_preflight_error(spec)
    monkeypatch.setenv("REVIEW_HERMES_BASE_URL", ENDPOINT)
    assert _panel_preflight_error(spec) == ""
    monkeypatch.setenv("REVIEW_HERMES_PROVIDER", "openrouter")
    assert "custom provider" in _panel_preflight_error(spec)


def test_preflight_mirrors_the_codex_judge_config(judge_env) -> None:
    from outerloop.tick import _panel_preflight_error

    ok = _spec(judge_env, panel="review:codex:judge-model", panel_codex_config="a=1")
    assert _panel_preflight_error(ok) == ""
    bad = _spec(judge_env, panel="review:codex:judge-model", panel_codex_config="nope")
    assert "OUTERLOOP_PANEL_CODEX_CONFIG" in _panel_preflight_error(bad)
    bad_author = _spec(judge_env, panel="review:codex:judge-model", codex_config="nope")
    assert "OUTERLOOP_CODEX_CONFIG" in _panel_preflight_error(bad_author)


def test_preflight_compares_judge_keys_with_the_fleet_author_by_value(
    judge_env, monkeypatch
) -> None:
    from outerloop.tick import _panel_preflight_error

    monkeypatch.setenv("REVIEW_HERMES_PROVIDER", "openai")
    spec = _spec(judge_env, panel="review:hermes:judge-model")
    monkeypatch.delenv("OUTERLOOP_AUTHOR_BACKEND", raising=False)
    copy = _key(judge_env / "claude_author_key", "sk-hermes-judge")  # same value
    monkeypatch.setenv("OUTERLOOP_CLAUDE_KEY_FILE", str(copy))
    error = _panel_preflight_error(spec)
    assert "holds the author key" in error and "sk-hermes-judge" not in error
    _key(copy, "sk-claude-author")
    assert _panel_preflight_error(spec) == ""


def test_the_service_spec_reads_the_panel_codex_config(tmp_path, monkeypatch) -> None:
    from outerloop.tick import _service_spec_from_env

    image = tmp_path / "img.sif"
    image.write_text("")
    monkeypatch.setenv("OUTERLOOP_PAT_FILE", str(_key(tmp_path / "pat", "ghp_x")))
    monkeypatch.setenv("OUTERLOOP_HOME", str(tmp_path))
    monkeypatch.setenv("OUTERLOOP_TARGET", "org/pilot")
    monkeypatch.setenv("OUTERLOOP_IMAGE", str(image))
    monkeypatch.setenv("OUTERLOOP_PANEL_CODEX_CONFIG", "a=1")
    _, spec = _service_spec_from_env(tmp_path)
    assert spec is not None and spec.panel_codex_config == "a=1"


def test_wake_spec_carries_the_panel_codex_config(tmp_path) -> None:
    """Upgrading: a wake spec written before the field loads with it empty."""
    import json

    from outerloop.tick import WAKE_SPEC_NAME, load_wake_spec, write_wake_spec

    write_wake_spec(tmp_path, _spec(tmp_path, panel_codex_config="a=1"))
    loaded = load_wake_spec(tmp_path)
    assert loaded is not None and loaded.panel_codex_config == "a=1"
    legacy = json.loads((tmp_path / WAKE_SPEC_NAME).read_text())
    del legacy["panel_codex_config"]
    (tmp_path / WAKE_SPEC_NAME).write_text(json.dumps(legacy))
    loaded = load_wake_spec(tmp_path)
    assert loaded is not None and loaded.panel_codex_config == ""
