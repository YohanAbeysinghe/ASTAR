"""Check ramp integration, oracle label semantics, and evaluator agreement."""

import functools
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import astar.evaluation.aggregation as evaluator
import astar.training.aggregation as trainer
from astar.training.monitoring import rollout_model_only
from astar.training.steps import waypoint_step_sizes


@pytest.mark.parametrize("horizon", [2, 4, 16])
def test_linear_step_sizes_and_fixed_origin(horizon):
    paths = jnp.zeros((2, horizon, 2))
    actual = jax.jit(lambda p: waypoint_step_sizes(p, 1.0, 0.1))(paths)
    expected = np.r_[0.0, np.linspace(0.1, 1.0, horizon - 1)]
    np.testing.assert_allclose(actual[0, :, 0], expected, atol=1e-7)
    assert actual.shape == (1, horizon, 1)


@pytest.mark.parametrize(
    "start,end",
    [(0, 1), (-0.1, 1), (2, 1), (0.1, 0), (float("nan"), 1), (0.1, float("inf"))],
)
def test_reject_invalid_schedule(start, end):
    with pytest.raises(ValueError):
        waypoint_step_sizes(jnp.zeros((1, 16, 2)), end, start)


def test_uniform_legacy_steps():
    actual = waypoint_step_sizes(jnp.zeros((1, 16, 2)), 2.0)
    np.testing.assert_array_equal(actual[0, :, 0], [0] + [2] * 15)


def test_updates_match_in_guarded_and_model_only_evaluation():
    paths = jnp.zeros((2, 16, 2))
    metrics = {
        "esdf": jnp.full((2, 64, 64), 5.0),
        "esdf_x_min": jnp.full((2,), -3.2),
        "esdf_y_min": jnp.full((2,), -3.2),
        "esdf_resolution": jnp.full((2,), 0.1),
        "goal_xy": jnp.asarray([[2.0, 2.0], [2.0, 2.0]]),
    }
    config = trainer.ObstacleEnergyConfig(
        collision_weight=0,
        clearance_weight=0,
        goal_weight=0,
        progress_weight=0,
        early_heading_weight=0,
        smoothness_weight=0,
        max_step_length_m=1.0,
    )
    # The constant field exceeds the cap: capping must happen BEFORE the ramp.
    raw = jnp.full_like(paths, 2.0)
    expected = np.broadcast_to(
        np.r_[0.0, np.linspace(0.1, 1.0, 15)][None, :, None], paths.shape
    )
    for guarded in (False, True):
        step = jax.jit(
            functools.partial(
                evaluator.apply_field_direction,
                energy_config=config,
                max_direction_rms=1.0,
                step_size=1.0,
                step_size_start=0.1,
                guarded=guarded,
                backtracks=3,
                backtrack_factor=0.5,
                energy_tolerance=0.0,
            )
        )
        actual, _, _ = step(paths, raw, metrics)
        np.testing.assert_allclose(actual, expected, atol=1e-6)
    actual, valid = rollout_model_only(
        paths,
        lambda _: raw,
        lambda p: trainer.project_bounded_actions(p, metrics, config),
        steps=1,
        step_size=1.0,
        step_size_start=0.1,
        max_direction_rms=1.0,
    )
    assert bool(jnp.all(valid))
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    oracle_next, _ = evaluator.oracle_projected_step(
        paths,
        raw / 2,
        metrics,
        config,
        step_size=1.0,
        step_size_start=0.1,
    )
    np.testing.assert_allclose(oracle_next, expected, atol=1e-6)


def test_backtracked_oracle_label_reproduces_proposal_without_squaring_ramp(
    monkeypatch,
):
    paths = jnp.zeros((1, 16, 2))
    target = jnp.full_like(paths, 0.2).at[:, 0].set(0)
    config = trainer.ObstacleEnergyConfig()

    def energy(p, *_):
        return jnp.mean(jnp.square(p[:, 1:] - target[:, 1:])), {}

    # Isolate integration from ESDF geometry so every waypoint has a known
    # oracle direction of one and backtracking must shrink the full proposal.
    monkeypatch.setattr(trainer, "compute_obstacle_energy", energy)
    monkeypatch.setattr(trainer, "project_bounded_actions", lambda p, *_: p)
    direction, info = trainer.query_energy_oracle(
        paths,
        {},
        config,
        base_batch_size=1,
        particles=1,
        gradient_floor=1e-8,
        max_direction_rms=1,
        diversity_direction_weight=0,
        step_size=1.0,
        step_size_start=0.1,
        backtracks=8,
        backtrack_factor=0.5,
        energy_tolerance=0,
    )
    fraction = float(info["oracle_step_size_mean"])
    assert 0 < fraction < 1
    np.testing.assert_allclose(direction[:, 1:], fraction, atol=1e-6)
    unscaled = jnp.ones_like(paths).at[:, 0].set(0)
    accepted, _ = trainer.backtracking_projected_step(
        paths,
        unscaled,
        {},
        config,
        step_size=1.0,
        step_size_start=0.1,
        backtracks=8,
        backtrack_factor=0.5,
        energy_tolerance=0,
    )
    reproduced = paths + waypoint_step_sizes(paths, 1.0, 0.1) * direction
    np.testing.assert_allclose(reproduced, accepted, atol=1e-6)
    assert float(energy(reproduced)[0]) < float(energy(paths)[0])


def test_new_defaults_and_resume_signature_include_schedule(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["trainer", "--exp-name", "linearstep_test"])
    args = trainer.parse_args()
    assert args.aggregation_step_size_start == 0.1
    assert args.aggregation_step_size == 1.0
    assert args.collision_weight == 0.0
    assert args.clearance_weight == 1.0
    assert args.clearance_cap_m == 0.25
    assert args.goal_weight == 5.0
    assert args.path_detour_factor == 1.25
    before = trainer._resume_signature(args)
    args.aggregation_step_size_start = 0.2
    assert trainer._resume_signature(args) != before
    args.aggregation_step_size_start = 0.1
    args.aggregation_step_size = 2.0
    assert trainer._resume_signature(args) != before
