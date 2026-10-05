from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import astar.training.continuation_wandb as tracking

SOURCE = "yohanab/astar/4wzq1hru"


def _fake_wandb(monkeypatch):
    monkeypatch.setattr(tracking.wandb.util, "generate_id", lambda: "newrun42")
    calls, runs = [], []

    def init(**kwargs):
        calls.append({**kwargs, "config": dict(kwargs.get("config", {}))})
        run = SimpleNamespace(
            id=kwargs.get("id", "disabled"),
            config=Mock(),
            define_metric=Mock(),
            log=Mock(),
        )
        runs.append(run)
        return run

    monkeypatch.setattr(tracking.wandb, "init", init)
    return calls, runs


def _start(tmp_path, **kwargs):
    return tracking.init_continuation_wandb(
        tmp_path,
        {"source_checkpoint": "/archive/35000", "project_name": "must-not-override-source"},
        enabled=kwargs.pop("enabled", True),
        source_run_path=SOURCE,
        source_optimizer_step=35000,
        name="full-dataset",
        resume=kwargs.pop("resume", False),
        **kwargs,
    )


def test_linked_run_records_source_and_never_resumes_or_forks_it(tmp_path, monkeypatch):
    calls, runs = _fake_wandb(monkeypatch)
    _start(tmp_path)
    call = calls[0]
    assert call["id"] == "newrun42"
    assert call["entity"] == "yohanab" and call["project"] == "astar"
    assert "resume" not in call and "resume_from" not in call and "fork_from" not in call
    assert call["config"]["source_checkpoint"] == "/archive/35000"
    assert call["config"]["continuation_source_run"] == SOURCE
    assert call["config"]["continuation_source_optimizer_step"] == 35000
    assert call["config"]["continuation_source_url"] == "https://wandb.ai/yohanab/astar/runs/4wzq1hru"
    assert call["config"]["continuation_history_mode"] == "linked_run"
    assert (tmp_path / "wandb_id.txt").read_text().strip() == "newrun42"
    runs[0].log.assert_called_once_with({"optimizer_step": 35000})
    runs[0].define_metric.assert_any_call("*", step_metric="optimizer_step")


def test_resume_uses_persisted_new_run_without_querying_or_resetting_source_step(tmp_path, monkeypatch):
    calls, runs = _fake_wandb(monkeypatch)
    _start(tmp_path)
    for _ in range(2):
        _start(tmp_path, resume=True)
    for call, run in zip(calls[1:], runs[1:], strict=True):
        assert call["id"] == "newrun42" and call["resume"] == "must"
        assert "fork_from" not in call and "resume_from" not in call
        run.log.assert_not_called()


def test_source_run_id_cannot_be_resumed_as_continuation(tmp_path, monkeypatch):
    calls, _ = _fake_wandb(monkeypatch)
    (tmp_path / "wandb_id.txt").write_text("4wzq1hru\n")
    with pytest.raises(ValueError, match="different from the source"):
        _start(tmp_path, resume=True)
    assert not calls


def test_existing_continuation_id_requires_explicit_resume(tmp_path, monkeypatch):
    calls, _ = _fake_wandb(monkeypatch)
    (tmp_path / "wandb_id.txt").write_text("oldchild\n")
    with pytest.raises(FileExistsError, match="resume=True"):
        _start(tmp_path)
    assert not calls
    assert (tmp_path / "wandb_id.txt").read_text() == "oldchild\n"


def test_disabled_logging_has_no_remote_read_or_id_side_effect(tmp_path, monkeypatch):
    calls, _ = _fake_wandb(monkeypatch)
    _start(tmp_path, enabled=False)
    assert calls == [{"mode": "disabled", "config": {}}]
    assert not (tmp_path / "wandb_id.txt").exists()
