from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import astar.evaluation.aggregation as evaluator


def _identity(
    index: int,
    clip: str,
    sample: str,
    goal: int = 0,
) -> evaluator.ExampleIdentity:
    example_id = f'["{clip}","{sample}","sampled",{goal}]'
    import hashlib

    return evaluator.ExampleIdentity(
        dataset_index=index,
        example_id=example_id,
        digest_hex=hashlib.sha256(example_id.encode()).hexdigest(),
        clip_id=clip,
        sample_id=sample,
        goal_kind="sampled",
        goal_index=goal,
        goal_x_m=1.0,
        goal_y_m=0.0,
    )


def _energy_fixture(batch_size: int = 1):
    metrics = {
        "esdf": jnp.full((batch_size, 64, 64), 5.0, dtype=jnp.float32),
        "esdf_x_min": jnp.zeros((batch_size,), dtype=jnp.float32),
        "esdf_y_min": jnp.full((batch_size,), -3.2, dtype=jnp.float32),
        "esdf_resolution": jnp.full((batch_size,), 0.1, dtype=jnp.float32),
        "goal_xy": jnp.tile(jnp.asarray([[1.2, 0.0]], dtype=jnp.float32), (batch_size, 1)),
    }
    config = evaluator.ObstacleEnergyConfig(
        collision_weight=0.0,
        clearance_weight=0.0,
        goal_weight=1.0,
        progress_weight=0.0,
        early_heading_weight=0.0,
        smoothness_weight=0.0,
        segment_samples=3,
        max_step_length_m=0.4,
        required_progress_fraction=0.65,
        train_particles=1,
        diversity_direction_weight=0.0,
    )
    return metrics, config


def test_clip_balanced_selection_is_stable_unique_and_one_goal_per_frame() -> None:
    candidates = {
        "clip-b": [_identity(5, "clip-b", "f1"), _identity(6, "clip-b", "f2")],
        "clip-a": [
            _identity(1, "clip-a", "f1", 0),
            _identity(2, "clip-a", "f1", 1),
            _identity(3, "clip-a", "f2", 0),
            _identity(4, "clip-a", "f3", 0),
        ],
    }
    reversed_candidates = {
        clip: list(reversed(items)) for clip, items in reversed(list(candidates.items()))
    }
    first = evaluator.select_clip_balanced_examples(
        candidates,
        examples_per_clip=2,
        seed=17,
    )
    second = evaluator.select_clip_balanced_examples(
        reversed_candidates,
        examples_per_clip=2,
        seed=17,
    )
    assert [item.example_id for item in first] == [item.example_id for item in second]
    assert len(first) == 4
    assert len({(item.clip_id, item.sample_id) for item in first}) == len(first)


def test_global_example_cap_preserves_maximum_clip_coverage() -> None:
    candidates = {
        clip: [_identity(clip_index * 10 + index, clip, f"f{index}") for index in range(3)]
        for clip_index, clip in enumerate(("a", "b", "c"))
    }
    selected = evaluator.select_clip_balanced_examples(
        candidates,
        examples_per_clip=3,
        max_examples=3,
        seed=4,
    )
    assert {item.clip_id for item in selected} == {"a", "b", "c"}


def test_strict_bool_does_not_treat_false_string_as_true() -> None:
    assert evaluator._strict_bool("false", name="flag") is False
    assert evaluator._strict_bool("TRUE", name="flag") is True
    with pytest.raises(ValueError, match="must be a boolean"):
        evaluator._strict_bool("sometimes", name="flag")


def test_training_namespace_coerces_wandb_scientific_notation_strings() -> None:
    values = {
        "config_name": "pi05_vega",
        "project_name": "test",
        "exp_name": "test",
        "action_dim": 2,
        "action_horizon": 16,
        "discrete_state_input": True,
        "use_goal_waypoint_adapter": True,
        "goal_waypoint_dim": 2,
        "max_goal_waypoints": 1,
        "train_scope": "adapter",
        "warmup_steps": 1000,
        "peak_lr": "1e-4",
        "decay_steps": 30000,
        "decay_lr": "1e-5",
        "clip_gradient_norm": 1,
        "ema_decay": "0.999",
        "num_train_steps": 30000,
        "log_interval": 100,
        "save_interval": 5000,
    }
    namespace = evaluator.build_training_namespace(
        values,
        batch_size=8,
        fsdp_devices=4,
    )
    assert namespace.peak_lr == pytest.approx(1.0e-4)
    assert namespace.decay_lr == pytest.approx(1.0e-5)
    assert namespace.ema_decay == pytest.approx(0.999)


def test_baseline_only_validation_does_not_require_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    args = evaluator.parse_args(
        [
            "--checkpoint-root",
            str(tmp_path / "missing"),
            "--modes",
            "prior",
            "--batch-size",
            "1",
            "--fsdp-devices",
            "1",
        ]
    )
    monkeypatch.setattr(evaluator.jax, "process_count", lambda: 1)
    monkeypatch.setattr(evaluator.jax, "device_count", lambda: 1)
    evaluator._validate_args(args, {"train_particles": 1})


def test_train_and_eval_splits_must_be_disjoint(tmp_path: Path) -> None:
    train_path = tmp_path / "train.txt"
    eval_path = tmp_path / "eval.txt"
    train_path.write_text("# comment\na\nb\n", encoding="utf-8")
    eval_path.write_text("c\nd\n", encoding="utf-8")
    resolved, train_ids, eval_ids = evaluator.validate_held_out_split(
        {"train_split_ids_path": str(train_path)}, eval_path
    )
    assert resolved == train_path
    assert train_ids == ("a", "b")
    assert eval_ids == ("c", "d")

    eval_path.write_text("b\nc\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps"):
        evaluator.validate_held_out_split(
            {"train_split_ids_path": str(train_path)}, eval_path
        )


def test_model_only_transition_never_calls_energy_backtracking(monkeypatch) -> None:
    paths = jnp.zeros((1, 4, 2), dtype=jnp.float32)
    direction = jnp.zeros_like(paths).at[:, 1:, 0].set(0.25)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("model-only transition consulted the energy guard")

    monkeypatch.setattr(evaluator, "backtracking_projected_step", fail_if_called)
    next_paths, capped, info = evaluator.apply_field_direction(
        paths,
        direction,
        {},
        evaluator.ObstacleEnergyConfig(max_step_length_m=0.4),
        max_direction_rms=1.0,
        step_size=0.1,
        guarded=False,
        backtracks=4,
        backtrack_factor=0.5,
        energy_tolerance=0.0,
    )
    assert bool(jnp.all(jnp.isfinite(next_paths)))
    assert bool(jnp.all(jnp.isfinite(capped)))
    assert float(info["raw_direction_finite"][0]) == 1.0
    assert float(info["moved"][0]) == 1.0


def test_guard_rejects_an_energy_increase_that_model_only_applies() -> None:
    metrics, config = _energy_fixture()
    paths = jnp.zeros((1, 4, 2), dtype=jnp.float32)
    away = jnp.zeros_like(paths).at[:, 1:, 0].set(-10.0)
    model_paths, _, _ = evaluator.apply_field_direction(
        paths,
        away,
        metrics,
        config,
        max_direction_rms=1.0,
        step_size=0.1,
        guarded=False,
        backtracks=5,
        backtrack_factor=0.5,
        energy_tolerance=0.0,
    )
    guarded_paths, _, guarded_info = evaluator.apply_field_direction(
        paths,
        away,
        metrics,
        config,
        max_direction_rms=1.0,
        step_size=0.1,
        guarded=True,
        backtracks=5,
        backtrack_factor=0.5,
        energy_tolerance=0.0,
    )
    model_energy, _ = evaluator.compute_obstacle_energy(model_paths, metrics, config)
    initial_energy, _ = evaluator.compute_obstacle_energy(paths, metrics, config)
    assert float(model_energy) > float(initial_energy)
    np.testing.assert_allclose(np.asarray(guarded_paths), np.asarray(paths))
    assert float(guarded_info["update_applied"][0]) == 0.0


def test_per_example_scoring_does_not_broadcast_a_batch_mean() -> None:
    metrics, config = _energy_fixture(batch_size=2)
    paths = jnp.zeros((2, 4, 2), dtype=jnp.float32)
    paths = paths.at[1, 1:, 0].set(jnp.asarray([0.4, 0.8, 1.2]))
    scores = evaluator.score_paths_per_example(paths, metrics, config)
    explicit = []
    for index in range(2):
        one_metrics = {key: value[index : index + 1] for key, value in metrics.items()}
        _, info = evaluator.compute_obstacle_energy(
            paths[index : index + 1],
            one_metrics,
            config,
        )
        explicit.append(float(info["goal_energy"]))
    np.testing.assert_allclose(np.asarray(scores["goal_energy"]), explicit)
    assert explicit[0] != explicit[1]


def test_per_example_scoring_accepts_different_configured_robot_radii() -> None:
    metrics, config = _energy_fixture(batch_size=2)
    metrics["esdf"] = jnp.full_like(metrics["esdf"], 0.4)
    metrics["robot_radius_m"] = jnp.asarray([0.1, 0.5])
    paths = jnp.tile(jnp.asarray([[[0.0, 0.0], [0.2, 0.0], [0.4, 0.0], [0.6, 0.0]]]), (2, 1, 1))
    config = dataclasses.replace(config, strict_esdf_coverage=True, safety_margin_m=0.1)
    scores = evaluator.score_paths_per_example(paths, metrics, config)
    np.testing.assert_allclose(scores["safe_radius_m"], [0.2, 0.6], atol=1e-6)
    np.testing.assert_allclose(scores["min_clearance_m"], [0.3, -0.1], atol=1e-6)
    np.testing.assert_allclose(scores["collision_rate"], [0.0, 1.0])
    np.testing.assert_allclose(scores["unsafe_rate"], [0.0, 1.0])


def test_nonfinite_field_is_visible_even_after_safe_sanitization() -> None:
    paths = jnp.zeros((1, 4, 2), dtype=jnp.float32)
    direction = jnp.full_like(paths, jnp.nan)
    next_paths, _, info = evaluator.apply_field_direction(
        paths,
        direction,
        {},
        evaluator.ObstacleEnergyConfig(max_step_length_m=0.4),
        max_direction_rms=1.0,
        step_size=0.1,
        guarded=False,
        backtracks=4,
        backtrack_factor=0.5,
        energy_tolerance=0.0,
    )
    assert float(info["raw_direction_finite"][0]) == 0.0
    assert bool(jnp.all(jnp.isfinite(next_paths)))


def test_undefined_oracle_cosine_is_excluded_not_averaged_as_zero() -> None:
    predicted = jnp.ones((2, 4, 2), dtype=jnp.float32)
    target = jnp.ones_like(predicted).at[1].set(0.0)
    diagnostics = evaluator.field_oracle_diagnostics(predicted, target)
    cosine = np.asarray(diagnostics["field_oracle_cosine"])
    np.testing.assert_allclose(cosine[0], 1.0, rtol=1.0e-6)
    assert np.isnan(cosine[1])
    np.testing.assert_array_equal(
        np.asarray(diagnostics["oracle_target_valid"]), np.asarray([1.0, 0.0])
    )

    rows = [
        {
            "checkpoint": "30000",
            "parameter_source": "raw",
            "mode": "model_only",
            "depth": 1,
            "dataset_index": index,
            "clip_id": f"clip-{index}",
            "field_oracle_cosine": float(value),
        }
        for index, value in enumerate(cosine)
    ]
    summary = evaluator.summarize_records(rows, bootstrap_samples=20, seed=0)
    metric = next(row for row in summary if row["metric"] == "field_oracle_cosine")
    assert metric["count"] == 1
    assert metric["mean"] == pytest.approx(1.0)


def test_jsonl_serializes_nonfinite_diagnostics_as_null(tmp_path: Path) -> None:
    output = tmp_path / "records.jsonl"
    with evaluator.JsonlWriter(output) as writer:
        writer.write({"finite": 1.0, "undefined": math.nan})
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "finite": 1.0,
        "undefined": None,
    }


def test_summary_reports_micro_and_clip_macro_means() -> None:
    rows = []
    for index, (clip, value) in enumerate((("a", 0.0), ("a", 0.0), ("b", 1.0))):
        rows.append(
            {
                "checkpoint": "30000",
                "parameter_source": "raw",
                "mode": "model_only",
                "depth": 12,
                "dataset_index": index,
                "clip_id": clip,
                "metric_value": value,
            }
        )
    summary = evaluator.summarize_records(rows, bootstrap_samples=100, seed=0)
    metric = next(row for row in summary if row["metric"] == "metric_value")
    assert metric["mean"] == 1.0 / 3.0
    assert metric["clip_macro_mean"] == 0.5


def test_paired_checkpoint_summary_uses_matched_clip_differences() -> None:
    records = []
    examples = (("a-1", "a", 0.0, 1.0), ("a-2", "a", 0.0, 1.0), ("b-1", "b", 1.0, 0.0))
    for example_id, clip_id, left_value, right_value in examples:
        for checkpoint, value in (("27500", left_value), ("30000", right_value)):
            records.append(
                {
                    "checkpoint": checkpoint,
                    "parameter_source": "raw",
                    "mode": "model_only",
                    "depth": 12,
                    "example_id": example_id,
                    "clip_id": clip_id,
                    "success": value,
                }
            )
    summary = evaluator.summarize_paired_differences(
        records,
        bootstrap_samples=100,
        seed=0,
    )
    metric = next(
        row
        for row in summary
        if row["comparison_type"] == "checkpoint" and row["metric"] == "success"
    )
    assert metric["difference"] == "right_minus_left"
    assert metric["left_checkpoint"] == "27500"
    assert metric["right_checkpoint"] == "30000"
    assert metric["mean_delta"] == pytest.approx(1.0 / 3.0)
    assert metric["clip_macro_mean_delta"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("invalid_key", "invalid_value"),
    [("dense_all_invalid_traj_rate", 1.0), ("dense_invalid_esdf_rate", 0.01)],
)
def test_record_success_fails_closed_on_invalid_geometry(invalid_key, invalid_value) -> None:
    identity = _identity(0, "clip", "frame")
    base_scores = {
        "dense_collision_rate": np.asarray([0.0]),
        "dense_unsafe_rate": np.asarray([0.0]),
        "dense_all_invalid_traj_rate": np.asarray([0.0]),
        "dense_invalid_esdf_rate": np.asarray([0.0]),
        "action_finite_rate": np.asarray([1.0]),
        "achieved_required_progress_ratio": np.asarray([0.95]),
        "obstacle_energy": np.asarray([1.0]),
    }
    valid = evaluator.records_for_depth(
        checkpoint="30000",
        parameter_source="raw",
        mode="model_only",
        depth=1,
        identities=[identity],
        scores=base_scores,
        transition=None,
        previous_energy=None,
        rollout_valid=np.asarray([True]),
        progress_threshold=0.9,
    )[0]
    assert valid["safe_success"] == 1.0

    invalid_scores = dict(base_scores)
    invalid_scores[invalid_key] = np.asarray([invalid_value])
    invalid = evaluator.records_for_depth(
        checkpoint="30000",
        parameter_source="raw",
        mode="model_only",
        depth=1,
        identities=[dataclasses.replace(identity, dataset_index=1)],
        scores=invalid_scores,
        transition=None,
        previous_energy=None,
        rollout_valid=np.asarray([True]),
        progress_threshold=0.9,
    )[0]
    assert invalid["success"] == 0.0
    assert invalid["safe_success"] == 0.0


def test_dense_safety_does_not_inherit_legacy_near_field_exemption() -> None:
    metrics, legacy_config = _energy_fixture()
    metrics = metrics | {
        "esdf": jnp.full((1, 64, 64), -1.0),
        "esdf_x_min": jnp.asarray([-0.1]),
    }
    values = {
        "train_particles": 1, "oracle_gradient_floor": 1.0e-4,
        "max_field_direction_rms": 1.0, "diversity_direction_weight": 0.0,
        "aggregation_step_size": 0.1, "rollout_backtracks": 2,
        "backtrack_factor": 0.5, "energy_increase_tolerance": 0.0,
        "action_horizon": 4, "action_dim": 2,
    }
    kernels = evaluator.build_kernels(
        values, SimpleNamespace(batch_size=1), legacy_config,
        prior_seed=0, safety_sample_spacing_m=0.05,
    )
    paths = jnp.asarray([[[0., 0.], [.2, 0.], [.3, 0.], [.4, 0.]]])
    legacy = kernels.score_paths(paths, metrics)
    safety = kernels.score_dense_safety(paths, metrics)
    assert float(legacy["collision_rate"][0]) == 0.0
    assert float(safety["invalid_esdf_rate"][0]) == 0.0
    assert float(safety["collision_rate"][0]) == 1.0
    assert float(safety["unsafe_rate"][0]) == 1.0
