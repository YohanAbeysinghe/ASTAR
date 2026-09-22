"""Safety checks for reusing paths with a new multimodal condition pool."""
import argparse

import jax
import numpy as np
import pytest

import astar.training.aggregation as trainer
from astar.training.replay import AggregationReplay


def parsed(monkeypatch):
    monkeypatch.setattr("sys.argv", ["trainer", "--exp-name", "source"])
    return trainer.parse_args()


def test_wandb_defaults_to_personal_astar_project(monkeypatch):
    args = parsed(monkeypatch)
    assert args.wandb_entity == "yohanab"
    assert args.project_name == "astar"


def source_for(args):
    return {
        **vars(args),
        "aggregation_algorithm_version": trainer.AGGREGATION_ALGORITHM_VERSION,
        "process_count": jax.process_count(),
    }


def test_explicit_migration_allows_goal_mix_and_cutoff_only(monkeypatch):
    old = parsed(monkeypatch)
    source = source_for(old)
    new = argparse.Namespace(**vars(old))
    new.exp_name = "multimodal"
    new.manifest_path = "/pinned/manifest.json"
    new.sampled_goal_fraction = 1 / 6
    new.object_text_goal_prob = 1 / 3
    new.object_image_goal_prob = 1 / 3
    new.object_waypoint_goal_prob = 1 / 6
    new.esdf_learning_cutoff_m = 0.2
    assert trainer.goal_conditioning_source_signature(new, source) == trainer._resume_signature(old)
    new.peak_lr *= 2
    with pytest.raises(ValueError, match="only"):
        trainer.goal_conditioning_source_signature(new, source)


def test_condition_migration_preserves_every_path_and_replaces_fingerprints():
    paths = np.arange(2 * 3 * 4 * 2, dtype=np.float32).reshape(2, 3, 4, 2)
    replay = AggregationReplay(
        config_signature="old", condition_fingerprints=["old-a", "old-b"],
        current_paths=paths * 3,
    )
    replay.append_current_round(np.ones_like(paths), np.ones((2, 3)), {"energy": 1})
    replay.finish_round(paths * 2, {"round": 1}, convergence_streak=1)
    before_visited = [value.copy() for value in replay.visited_paths]
    before_current = replay.current_paths.copy()
    migrated = trainer.relabel_max_step_replay(
        replay,
        lambda batch, value: (np.full_like(value, batch), np.zeros(3), {"energy": batch}),
        "new",
        condition_fingerprints=["new-a", "new-b"],
    )
    assert migrated.condition_fingerprints == ["new-a", "new-b"]
    np.testing.assert_array_equal(migrated.current_paths, before_current)
    for actual, expected in zip(migrated.visited_paths, before_visited, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert migrated.config_signature == "new"
    assert migrated.convergence_streak == 0
    assert migrated.validation_records == [] and migrated.benefit_state == {}


def test_migration_requires_resume_and_cannot_stack_with_cap_migration(monkeypatch):
    args = parsed(monkeypatch)
    args.migrate_goal_conditioning_from_config = "/source.json"
    config = trainer.ObstacleEnergyConfig()
    with pytest.raises(ValueError, match="requires --resume"):
        trainer._validate_args(args, config)
    args.resume = True
    args.relabel_esdf_cutoff_from_config = "/other.json"
    with pytest.raises(ValueError, match="already includes"):
        trainer._validate_args(args, config)


def test_forced_round_target_can_start_fresh(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["trainer", "--exp-name", "multimodal", "--continue-aggregation-until-round", "64"],
    )
    args = trainer.parse_args()
    assert args.aggregation_rounds == args.continue_aggregation_until_round == 64
    trainer._validate_args(args, trainer.ObstacleEnergyConfig())

    args.post_aggregation_updates = 1
    with pytest.raises(ValueError, match="cannot be combined"):
        trainer._validate_args(args, trainer.ObstacleEnergyConfig())
