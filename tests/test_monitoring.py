from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import astar.training.aggregation as trainer
import astar.training.monitoring as monitoring
from astar.training.replay import AggregationReplay
from astar.training.replay import BenefitTracker
from astar.training.replay import ConvergenceConfig
from astar.training.replay import ConvergenceTracker
from astar.waypoint_energy import ObstacleEnergyConfig


def metrics_fixture(batch=2):
    return {
        "esdf": jnp.full((batch, 20, 30), 5.0),
        "esdf_x_min": jnp.full((batch,), -1.0),
        "esdf_y_min": jnp.full((batch,), -1.0),
        "esdf_resolution": jnp.full((batch,), 0.1),
        "goal_xy": jnp.tile(jnp.asarray([[0.6, 0.0]]), (batch, 1)),
        "robot_radius_m": jnp.full((batch,), 0.5),
    }


def validation_metrics(**changes):
    return {
        "success_rate": 0.0,
        "safe_success_rate": 0.0,
        "bounded_progress_ratio": 0.1,
        "collision_rate": 1.0,
        "invalid_coverage_rate": 0.5,
        "goal_retreat_segment_rate": 0.2,
        **changes,
    }


def test_wandb_esdf_overlay_matches_regenerated_sheet_color_style():
    grid = np.asarray([[-2.0, 0.0], [1.0, 4.0]], dtype=np.float32)
    context = {
        "esdf": grid,
        "esdf_x_min": np.asarray(0.0),
        "esdf_y_min": np.asarray(-1.0),
        "esdf_resolution": np.asarray(1.0),
        "goal_xy": np.asarray([1.0, 0.0]),
        "robot_radius_m": np.asarray(0.5),
    }
    path = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    record = {
        "clip_id": "clip",
        "sample_id": "frame",
        "progress_ratio": 1.0,
        "collision_rate": 0.0,
        "invalid_coverage_rate": 0.0,
    }

    figure = monitoring.path_overlay_figure(path, path, context, record)
    image = figure.axes[0].images[0]
    lower, upper = image.get_clim()
    assert image.get_cmap().name == "coolwarm"
    assert lower == pytest.approx(-upper)
    assert upper == pytest.approx(monitoring.esdf_display_scale(grid))

    import matplotlib.pyplot as plt

    plt.close(figure)


def test_wandb_esdf_overlay_can_render_prior_only():
    grid = np.zeros((2, 2), dtype=np.float32)
    context = {
        "esdf": grid,
        "esdf_x_min": np.asarray(0.0),
        "esdf_y_min": np.asarray(-1.0),
        "esdf_resolution": np.asarray(1.0),
        "goal_xy": np.asarray([1.0, 0.0]),
        "robot_radius_m": np.asarray(0.5),
    }
    path = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    record = {
        "clip_id": "clip",
        "sample_id": "frame",
        "progress_ratio": 1.0,
        "collision_rate": 0.0,
        "invalid_coverage_rate": 0.0,
    }

    figure = monitoring.path_overlay_figure(path, None, context, record)
    labels = [line.get_label() for line in figure.axes[0].lines]
    assert labels == ["prior", "robot", "goal"]

    import matplotlib.pyplot as plt

    plt.close(figure)


def test_benefit_can_improve_beyond_twelve_rounds_then_stop():
    tracker = BenefitTracker(min_rounds=6, patience=4)
    for round_number in range(21):
        assert not tracker.observe(
            round_number,
            validation_metrics(bounded_progress_ratio=-0.9 + 0.04 * round_number),
        )
    for round_number in (21, 22, 23):
        assert not tracker.observe(
            round_number, validation_metrics(bounded_progress_ratio=-0.1)
        )
    assert tracker.observe(24, validation_metrics(bounded_progress_ratio=-0.1))


def test_small_improvements_accumulate_and_safety_improvement_resets_patience():
    tracker = BenefitTracker(min_rounds=1, patience=3)
    assert not tracker.observe(0, validation_metrics())
    assert not tracker.observe(1, validation_metrics(bounded_progress_ratio=0.109))
    assert not tracker.observe(2, validation_metrics(bounded_progress_ratio=0.118))
    assert not tracker.observe(3, validation_metrics(bounded_progress_ratio=0.127))
    assert tracker.bad_rounds == 0
    assert not tracker.observe(4, validation_metrics())
    assert not tracker.observe(5, validation_metrics(collision_rate=0.95))
    assert tracker.bad_rounds == 0


def test_plateau_is_distinct_from_unsafe_convergence():
    tracker = ConvergenceTracker(ConvergenceConfig(min_rounds=1, patience=1))
    stationary = {
        "path_change_median": 0.0,
        "path_change_p95": 0.0,
        "relative_energy_improvement": 0.0,
        "oracle_grad_rms": 0.0,
        "collision_rate": 0.0,
        "progress_ratio": 1.0,
        "invalid_esdf_rate": 0.5,
    }
    assert tracker.observe(1, stationary) == (False, "stalled")
    benefit = BenefitTracker(min_rounds=1, patience=1)
    benefit.observe(0, validation_metrics())
    assert benefit.observe(1, validation_metrics())
    with pytest.raises(ValueError, match="Nonfinite"):
        benefit.observe(2, validation_metrics(collision_rate=float("nan")))


def test_loss_baseline_excludes_origin_and_ratio_uses_same_window():
    target = jnp.asarray([[[999.0, 999.0], [0.4, 0.2]]])
    baseline = trainer._masked_field_loss(jnp.zeros_like(target), target)
    assert float(baseline) == pytest.approx(0.05)
    first = {"loss": jnp.asarray(0.02), "zero_predictor_loss": jnp.asarray(0.04)}
    second = {"loss": jnp.asarray(0.03), "zero_predictor_loss": jnp.asarray(0.01)}
    window = trainer._mean_metric_dict([first, second])
    assert monitoring.learning_metrics(window)["train/loss_vs_zero"] == pytest.approx(
        1.0
    )
    assert (
        monitoring.learning_metrics({"loss": 0.1, "zero_predictor_loss": 0.0})[
            "train/loss_vs_zero"
        ]
        is None
    )


def test_model_only_rollout_uses_fixed_prior_and_no_energy_guard(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Model-only integration consulted energy")

    monkeypatch.setattr(monitoring, "compute_obstacle_energy", fail)
    initial = jnp.zeros((1, 4, 2))

    def run():
        return monitoring.rollout_model_only(
            initial,
            lambda p: jnp.ones_like(p),
            lambda p: p,
            steps=3,
            step_size=0.2,
            max_direction_rms=0.5,
        )

    paths, valid = jax.jit(run)()
    np.testing.assert_allclose(paths[:, 1:], 0.3, atol=1e-6)
    np.testing.assert_allclose(paths[:, 0], 0.0)
    np.testing.assert_array_equal(run()[0], paths)
    assert bool(valid[0])
    bad, valid = monitoring.rollout_model_only(
        initial,
        lambda p: jnp.full_like(p, jnp.nan),
        lambda p: p,
        steps=2,
        step_size=1.0,
        max_direction_rms=1.0,
    )
    assert not bool(valid[0])
    assert np.isfinite(bad).all()


def test_validation_success_requires_progress_coverage_finiteness_and_margin():
    context = metrics_fixture(batch=4)
    context["esdf"] = context["esdf"].at[1].set(0.6).at[2].set(0.4)
    paths = jnp.tile(
        jnp.asarray([[[0.0, 0.0], [0.2, 0.0], [0.4, 0.0], [0.6, 0.0]]]), (4, 1, 1)
    )
    config = ObstacleEnergyConfig(
        strict_esdf_coverage=True, safety_margin_m=0.25, segment_samples=8
    )
    score = jax.jit(
        lambda p, m, v: monitoring.score_validation_paths(
            p,
            m,
            v,
            energy_config=config,
            progress_threshold=0.9,
        )
    )
    result = score(paths, context, jnp.asarray([True, True, True, False]))
    np.testing.assert_array_equal(result["success_rate"], [1, 1, 0, 0])
    np.testing.assert_array_equal(result["safe_success_rate"], [1, 0, 0, 0])
    invalid_context = context | {"esdf_x_min": jnp.full((4,), 0.2)}
    invalid = score(paths, invalid_context, jnp.ones(4, dtype=bool))
    assert np.all(invalid["invalid_coverage_rate"] > 0)
    assert np.all(invalid["success_rate"] == 0)


def test_model_only_rollout_accepts_dynamic_round_depth_including_zero():
    initial = jnp.asarray([[[0.0, 0.0], [0.3, 0.1], [0.7, 0.2]]])

    @jax.jit
    def run(completed_rounds):
        return monitoring.rollout_model_only(
            initial,
            lambda paths: jnp.ones_like(paths),
            lambda paths: paths,
            steps=completed_rounds,
            step_size=1.0,
            step_size_start=0.1,
            max_direction_rms=0.5,
        )

    for rounds in (0, 1, 3, 13):
        paths, valid = run(jnp.asarray(rounds, dtype=jnp.int32))
        expected = np.asarray(initial) + rounds * np.asarray(
            [[[0.0, 0.0], [0.05, 0.05], [0.5, 0.5]]]
        )
        np.testing.assert_allclose(paths, expected, atol=1e-6)
        assert bool(valid[0])
    np.testing.assert_array_equal(run(jnp.asarray(0, dtype=jnp.int32))[0], initial)


def test_goal_retreat_metric_counts_distance_to_goal_not_local_x():
    context = metrics_fixture(batch=1) | {"goal_xy": jnp.asarray([[-0.5, 0.0]])}
    paths = jnp.asarray([[[0.0, 0.0], [-0.3, 0.0], [-0.1, 0.0], [-0.4, 0.0]]])
    score = monitoring.score_validation_paths(
        paths,
        context,
        jnp.asarray([True]),
        energy_config=ObstacleEnergyConfig(strict_esdf_coverage=True),
        progress_threshold=0.9,
    )
    assert float(score["goal_retreat_segment_rate"][0]) == pytest.approx(1 / 3)
    assert float(score["goal_retreat_m"][0]) == pytest.approx(0.2)


def test_wandb_round_allowlist_has_active_components_without_duplicate_counters():
    config = ObstacleEnergyConfig(
        collision_weight=100.0, clearance_weight=0.0, goal_weight=30.0,
        progress_weight=10.0, early_heading_weight=0.25, smoothness_weight=0.5,
    )
    raw_terms = dict(zip(monitoring.ENERGY_TERMS, [0.01, -2.0, 0.2, 0.3, 0.4, 0.5]))
    total = sum(value * getattr(config, f"{term}_weight") for term, value in raw_terms.items())
    info = {
        "field_oracle_cosine": 0.2,
        "rollout_acceptance_rate": 0.3,
        "energy_after": total,
        "particle_path_diversity_m": 0.0,
        "source_depth": 7,
        "target_depth": 8,
        **{
            f"weighted_{term}_energy": value * getattr(config, f"{term}_weight")
            for term, value in raw_terms.items()
        },
        **{f"{term}_energy": value for term, value in raw_terms.items()},
    }
    # Check the training-round handoff, including disabled clearance, whose
    # raw value cannot be recovered by dividing its weighted contribution.
    rollout_info = {
        **info,
        "energy_before": total + 1,
        "relative_energy_improvement": 0.1,
        "path_change_per_path_m": np.asarray([0.1, 0.2]),
        "field_oracle_cosine_valid_count": 2,
        "collision_rate": 0.5,
        "invalid_esdf_rate": 0.0,
        "achieved_required_progress_ratio": 0.6,
        "achieved_progress_m": 3.0,
        "required_progress_m": 5.0,
    }
    info = trainer._round_metrics(
        {"oracle_acceptance_rate": 1.0}, np.asarray([0.1, 0.2]), [rollout_info]
    )
    payload = monitoring.round_wandb_metrics(info, config)
    assert len(payload) == 15
    assert "energy/clearance" not in payload
    for term, name in monitoring.ENERGY_EQUATION_NAMES.items():
        assert payload[f"energy/{name}"] == pytest.approx(raw_terms[term])
    assert payload["energy/Eclearance"] == pytest.approx(-2.0)
    assert payload["energy/E"] == payload["energy/total"]
    assert sum(
        payload.get(f"energy/{term}", 0) for term in monitoring.ENERGY_TERMS
    ) == pytest.approx(total)
    assert payload["energy/E"] == pytest.approx(total)
    assert not any("diversity" in key or "depth" in key for key in payload)


def test_resume_preserves_plateau_and_post_aggregation_updates(tmp_path):
    replay = AggregationReplay("config", ["train"], np.zeros((1, 1, 4, 2), np.float32))
    replay.append_current_round(np.ones_like(replay.current_paths), np.ones((1, 1)), {})
    replay.finish_round(
        replay.current_paths, {"status": "validation_plateau"}, convergence_streak=0
    )
    replay.aggregation_stop_reason = "validation_plateau"
    replay.replay_updates = 7
    replay.validation_fingerprints = ["validation"]
    tracker = BenefitTracker(min_rounds=1, patience=3)
    tracker.observe(0, validation_metrics())
    tracker.observe(1, validation_metrics())
    replay.benefit_state = {"best": tracker.best, "bad_rounds": tracker.bad_rounds}
    restored = AggregationReplay.load(
        replay.save(tmp_path / "replay.npz"),
        expected_config_signature="config",
        expected_condition_fingerprints=["train"],
    )
    assert restored.replay_updates == 7
    assert restored.aggregation_stop_reason == "validation_plateau"
    assert restored.validation_fingerprints == ["validation"]
    resumed = BenefitTracker(min_rounds=1, patience=3, **restored.benefit_state)
    assert not resumed.observe(2, validation_metrics())
    assert resumed.observe(3, validation_metrics())
    with pytest.raises(RuntimeError, match="stopped"):
        restored.append_current_round(
            np.ones_like(replay.current_paths), np.ones((1, 1)), {}
        )


def test_resume_can_extend_budgets_but_cannot_change_labels_or_validation(monkeypatch):
    monkeypatch.setattr("sys.argv", ["trainer", "--exp-name", "test"])
    args = trainer.parse_args()
    signature = trainer._resume_signature(args)
    for key, value in {
        "post_aggregation_updates": 5000,
        "aggregation_rounds": 100,
        "replay_only": True,
    }.items():
        changed = SimpleNamespace(**(vars(args) | {key: value}))
        assert trainer._resume_signature(changed) == signature
    for key, value in {
        "progress_weight": 20,
        "prior_length_scale": 0.5,
        "eval_seed": 987,
        "updates_per_round": 11,
        "benefit_patience": 7,
    }.items():
        changed = SimpleNamespace(**(vars(args) | {key: value}))
        assert trainer._resume_signature(changed) != signature


def test_validation_selection_has_unique_frames_and_disjoint_clip_check():
    class Dataset:
        _split_ids = {"a", "b"}
        goal_references = [
            SimpleNamespace(byte_offset=i // 2, goal_kind="sampled", goal_index=i % 2)
            for i in range(12)
        ]

        def _read_record_at(self, offset):
            return {"clip_id": "a" if offset < 3 else "b", "sample_id": str(offset)}

    dataset = Dataset()
    selected = monitoring.select_validation_examples(dataset, 4, seed=1)
    assert selected == monitoring.select_validation_examples(dataset, 4, seed=1)
    assert len({(item["clip_id"], item["sample_id"]) for item in selected}) == 4
    assert [item["clip_id"] for item in selected].count("a") == 2
    with pytest.raises(ValueError, match="overlaps"):
        monitoring.cache_validation_pool(
            SimpleNamespace(dataset=dataset),
            {"a"},
            num_batches=1,
            seed=1,
            sanitize=None,
            fingerprint=None,
            initialize=None,
        )


def test_validation_summary_weights_clips_equally():
    def record(clip, value):
        return {
            "clip_id": clip,
            **{
                k: value
                for k in (
                    *monitoring.VALIDATION_METRICS,
                    "bounded_progress_ratio",
                    "goal_retreat_m",
                    "rollout_valid",
                )
            },
        }

    result = monitoring.summarize_validation(
        [record("a", 1.0), record("b", 0.0), record("b", 0.0)]
    )
    assert result["success_rate"] == 0.5
