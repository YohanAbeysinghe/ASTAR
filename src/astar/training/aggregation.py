"""Train an autonomous pi0.5 waypoint field with energy-oracle aggregation.

This is an ASTAR adaptation of ``ScalarFlows/train_aggregation.py``. It never
constructs a flow-matching interpolation from a dataset trajectory:

1. create a fixed pool of observations and goal-biased unicycle paths ``x_0``;
2. query the analytic trajectory-energy oracle once at the current ``x_k``;
3. append the cached ``(observation, x_k, oracle_direction)`` states to replay;
4. regress the pi0.5 field on all replay rounds;
5. outside SGD, advance once with the learned field to obtain ``x_{k+1}``;
6. repeat until convergence or validation benefit plateaus;
7. optionally continue SGD on the frozen replay, including after resume.

Aggregation depth ``k`` is not flow time.  The pi0.5 time token remains fixed
so the learned field is autonomous, like the ScalarFlows experiment.  OpenPI's
raw velocity has the reverse-flow sign, hence the physical descent direction
used here is ``-decode_actions(...)``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import functools
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import signal
from typing import Any

import einops
from etils import epath
import flax.nnx as nnx
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask
from openpi.shared import array_typing as at
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import optax
import tqdm_loggable.auto as tqdm
import wandb

from astar.dataloader import DEFAULT_OBJECT_IMAGE_GOAL_PROB
from astar.dataloader import DEFAULT_OBJECT_TEXT_GOAL_PROB
from astar.dataloader import DEFAULT_OBJECT_WAYPOINT_GOAL_PROB
from astar.dataloader import DEFAULT_SAMPLED_GOAL_FRACTION
from astar.dataloader import OPENPI_IMAGE_SIZE
from astar.dataloader import AstarNavigationDataLoader
from astar.dataloader import AstarNavigationDataset
from astar.dataloader import GoalModalityConfig
from astar.energy_checkpoints import initialize_bounded_checkpoint_dir
from astar.energy_checkpoints import prune_auxiliary_checkpoints
from astar.energy_checkpoints import save_trainable_params_checkpoint
from astar.path_sampler import DynamicWaypointPriorConfig
from astar.path_sampler import maybe_normalize_actions
from astar.path_sampler import maybe_unnormalize_actions
from astar.path_sampler import sample_dynamic_waypoint_prior
from astar.training.monitoring import ENERGY_TERMS
from astar.training.monitoring import VALIDATION_METRICS
from astar.training.monitoring import cache_validation_pool
from astar.training.monitoring import esdf_display_scale
from astar.training.monitoring import evaluate_validation_pool
from astar.training.monitoring import host_numpy
from astar.training.monitoring import learning_metrics
from astar.training.monitoring import rollout_model_only
from astar.training.monitoring import round_wandb_metrics
from astar.training.monitoring import score_validation_paths
from astar.training.replay import AggregationReplay
from astar.training.replay import BenefitTracker
from astar.training.replay import ConvergenceConfig
from astar.training.replay import ConvergenceTracker
from astar.training.replay import deterministic_replay_slot
from astar.training.replay import stable_config_signature
from astar.training.steps import waypoint_step_sizes
from astar.training_common import ACTION_STACK_PRETRAIN_SKIP_PATTERNS
from astar.training_common import PI05_BASE_PARAMS
from astar.training_common import create_train_config
from astar.training_common import init_logging
from astar.training_common import init_train_state
from astar.waypoint_energy import ObstacleEnergyConfig
from astar.waypoint_energy import _repeat_batch_tree
from astar.waypoint_energy import bias_prior_toward_goal
from astar.waypoint_energy import compute_obstacle_energy
from astar.waypoint_energy import compute_particle_path_diversity
from astar.waypoint_energy import decode_bounded_waypoint_path

GROUND_TRUTH_METRIC_KEYS = frozenset(
    {"actions_vw", "path_xy", "path_xytheta", "path_step_mask"}
)
ALLOWED_ORACLE_METRIC_KEYS = frozenset(
    {
        "aggregation_id_hi",
        "aggregation_id_lo",
        "bbox",
        "esdf",
        "esdf_resolution",
        "esdf_x_min",
        "esdf_y_min",
        "goal_image_available_mask",
        "goal_image_condition_mask",
        "goal_index",
        "goal_text_condition_mask",
        "goal_waypoint_condition_mask",
        "goal_xy",
        "is_object_goal",
        "robot_radius_m",
    }
)

TRAIN_CONSOLE_KEYS = (
    "loss",
    "field_oracle_cosine",
    "field_direction_rms",
    "oracle_direction_rms",
    "grad_norm",
)
AGGREGATION_ALGORITHM_VERSION = 7


class OracleOnlyNavigationDataset(AstarNavigationDataset):
    """Navigation view that never requires or loads a planned trajectory.

    The shared ASTAR loader is path-supervised by construction.  This isolated
    view retains only the image, goal and ESDF inputs needed by the energy
    oracle, while providing zero placeholders to the legacy OpenPI adapter.
    Those placeholders are discarded before the condition pool is cached.
    """

    def _usable_goal(self, goal: Any) -> bool:
        if not isinstance(goal, Mapping):
            return False
        goal_xy = goal.get("goal_xy_m")
        if isinstance(goal_xy, str):
            return False
        try:
            xy = np.asarray(goal_xy[:2], dtype=np.float32)
        except (IndexError, TypeError, ValueError):
            return False
        return xy.shape == (2,) and bool(np.all(np.isfinite(xy)))

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self._index[int(index)]
        record = self._read_record_at(ref.byte_offset)
        goals_key = "object_goals" if ref.goal_kind == "object" else "sampled_goals"
        goal = record[goals_key][ref.goal_index]
        is_object_goal = ref.goal_kind == "object"

        front_image_path = self._resolve_path(record["image_path"], record)
        front_image = self._load_image(front_image_path, self.config.image_size)
        bbox = np.zeros((4,), dtype=np.float32)
        goal_image = None
        if is_object_goal:
            bbox = np.asarray(
                goal.get("bbox") or [0.0, 0.0, 0.0, 0.0], dtype=np.float32
            )
            goal_image = self._crop_goal_image(
                front_image_path,
                bbox,
                self.config.goal_image_size,
            )

        esdf_config = self._esdf_config(record)
        identity = json.dumps(
            [
                str(record.get("clip_id", "")),
                str(record.get("sample_id", "")),
                ref.goal_kind,
                str(ref.goal_index),
            ],
            separators=(",", ":"),
        )
        identity_digest = hashlib.sha256(identity.encode("utf-8")).digest()[:8]
        identity_words = np.frombuffer(identity_digest, dtype="<u4").astype(
            np.uint32, copy=False
        )

        # These are interface placeholders, not labels.  The aggregation
        # trainer discards them immediately and asserts that no path key enters
        # its condition pool or supervised train step.
        horizon = self.config.action_horizon
        sample: dict[str, Any] = {
            "front_image": front_image,
            "front_image_path": front_image_path.as_posix(),
            "goal_image": goal_image,
            "goal_image_mask": np.bool_(goal_image is not None),
            "state": np.asarray(self.config.current_pose_state, dtype=np.float32),
            "prompt": self._prompt_for_goal(goal, ref.goal_kind),
            "actions": np.zeros((horizon, 2), dtype=np.float32),
            "actions_vw": np.zeros((horizon, 2), dtype=np.float32),
            "path_xytheta": np.zeros((horizon, 3), dtype=np.float32),
            "path_xy": np.zeros((horizon, 2), dtype=np.float32),
            "path_step_mask": np.zeros((horizon,), dtype=bool),
            "goal_xy": np.asarray(goal["goal_xy_m"][:2], dtype=np.float32),
            "is_object_goal": np.bool_(is_object_goal),
            "goal_kind": ref.goal_kind,
            "goal_index": np.asarray(ref.goal_index, dtype=np.int32),
            "aggregation_id_hi": np.asarray(identity_words[0], dtype=np.uint32),
            "aggregation_id_lo": np.asarray(identity_words[1], dtype=np.uint32),
            "bbox": bbox,
            "sample_id": str(record.get("sample_id", "")),
            "clip_id": str(record.get("clip_id", "")),
            "label": str(goal.get("label") or ""),
            "raw_label": str(goal.get("raw_label") or ""),
            "caption": str(goal.get("caption") or ""),
            "path_plan_status": "not_loaded_for_energy_aggregation",
            "esdf_path": self._resolve_esdf_path(record).as_posix(),
            "esdf_x_min": np.asarray(esdf_config["x_min"], dtype=np.float32),
            "esdf_y_min": np.asarray(esdf_config["y_min"], dtype=np.float32),
            "esdf_resolution": np.asarray(esdf_config["resolution"], dtype=np.float32),
            "robot_radius_m": np.asarray(esdf_config["robot_radius"], dtype=np.float32),
        }
        if self.config.load_esdf:
            sample["esdf"] = np.load(sample["esdf_path"]).astype(np.float32, copy=False)
        return sample


class OracleOnlyNavigationDataLoader(AstarNavigationDataLoader):
    """Expose stable numeric identities alongside the shared OpenPI format."""

    def _openpi_item(self, sample, goal_mask):
        data, metrics = super()._openpi_item(sample, goal_mask)
        metrics["aggregation_id_hi"] = np.asarray(
            sample["aggregation_id_hi"], dtype=np.uint32
        )
        metrics["aggregation_id_lo"] = np.asarray(
            sample["aggregation_id_lo"], dtype=np.uint32
        )
        return data, metrics


def create_oracle_only_data_loader(
    config: _config.TrainConfig,
    args: argparse.Namespace,
    data_sharding: jax.sharding.NamedSharding,
    *,
    split: str,
    num_batches: int,
) -> OracleOnlyNavigationDataLoader:
    """Build a deterministic finite condition pool without path labels."""
    dataset = OracleOnlyNavigationDataset(
        data_root=Path(args.data_root or "../data"),
        manifest_path=Path(args.manifest_path).expanduser().resolve()
        if args.manifest_path
        else None,
        split=split,
        split_ids_path=Path(
            args.train_split_ids_path if split == "train" else args.eval_split_ids_path
        ).expanduser()
        if (args.train_split_ids_path if split == "train" else args.eval_split_ids_path)
        else None,
        action_horizon=int(config.model.action_horizon),
        path_stride=args.path_stride,
        image_size=OPENPI_IMAGE_SIZE,
        goal_image_size=OPENPI_IMAGE_SIZE,
        load_esdf=True,
        include_object_goals=split == "train" or args.sampled_goal_fraction < 1.0,
        include_sampled_goals=split == "train" or args.sampled_goal_fraction > 0.0,
    )
    if int(config.batch_size) % jax.process_count() != 0:
        raise ValueError("Global batch size must be divisible by the JAX process count.")
    return OracleOnlyNavigationDataLoader(
        dataset,
        batch_size=int(config.batch_size) // jax.process_count(),
        sampled_goal_fraction=args.sampled_goal_fraction,
        goal_modality_config=GoalModalityConfig(
            object_text_prob=args.object_text_goal_prob,
            object_image_prob=args.object_image_goal_prob,
            object_waypoint_prob=args.object_waypoint_goal_prob,
        ),
        output_format="openpi",
        max_token_len=int(config.model.max_token_len),
        discrete_state_input=bool(getattr(config.model, "discrete_state_input", False)),
        # The dataset index is grouped by clip/frame/goal.  A seeded shuffle
        # keeps the condition pool exactly reproducible without filling it
        # from one early frame in the manifest.
        shuffle=True,
        seed=int(config.seed) + 1_000_003 * jax.process_index(),
        sharding=data_sharding,
        num_batches=num_batches,
        return_metric_tensors=True,
        make_jax_arrays=True,
    )


def sanitize_metric_tensors(metric_tensors: dict[str, at.Array]) -> dict[str, at.Array]:
    """Fail closed on real trajectory labels, then retain oracle context only."""
    leaked = GROUND_TRUTH_METRIC_KEYS.intersection(metric_tensors)
    nonzero_labels = []
    for key in sorted(leaked):
        value = host_numpy(metric_tensors[key])
        if not np.all(np.isfinite(value)) or np.any(value != 0):
            nonzero_labels.append(key)
    if nonzero_labels:
        raise AssertionError(
            "Nonzero ground-truth path tensors reached the aggregation boundary: "
            f"{nonzero_labels}. Use OracleOnlyNavigationDataset."
        )
    sanitized = {
        key: value
        for key, value in metric_tensors.items()
        if key in ALLOWED_ORACLE_METRIC_KEYS
    }
    assert not GROUND_TRUTH_METRIC_KEYS.intersection(sanitized)
    required = {"esdf", "esdf_x_min", "esdf_y_min", "esdf_resolution", "goal_xy"}
    missing = required - set(sanitized)
    if missing:
        raise ValueError(f"Energy aggregation condition is missing: {sorted(missing)}")
    if "robot_radius_m" in sanitized:
        radii = host_numpy(sanitized["robot_radius_m"])
        if not np.all(np.isfinite(radii)) or np.any(radii < 0.0):
            raise ValueError(
                "robot_radius_m must contain finite, nonnegative radii in meters."
            )
    return sanitized


def condition_fingerprint(
    observation: _model.Observation,
    metric_tensors: dict[str, at.Array],
) -> str:
    """Fingerprint the complete label-free conditioning content for safe resume."""
    leaves = list(jax.tree.leaves(observation))
    leaves.extend(metric_tensors[key] for key in sorted(metric_tensors))
    digest = hashlib.sha256()
    for leaf in leaves:
        array = host_numpy(leaf)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def cache_condition_pool(data_loader, num_batches: int):
    """Keep immutable conditions once; discard all action placeholders."""
    conditions = []
    fingerprints = []
    iterator = iter(data_loader)
    for batch_index in range(num_batches):
        observation, target_actions, metric_tensors = next(iterator)
        target_array = host_numpy(target_actions)
        if not np.all(np.isfinite(target_array)) or np.any(target_array != 0):
            raise AssertionError(
                "Nonzero target actions reached the aggregation boundary. "
                "Use OracleOnlyNavigationDataset."
            )
        del target_actions
        metric_tensors = sanitize_metric_tensors(metric_tensors)
        conditions.append((observation, metric_tensors))
        fingerprints.append(condition_fingerprint(observation, metric_tensors))
        logging.info("Cached condition batch %d/%d", batch_index + 1, num_batches)
    return conditions, fingerprints


def _host_batch_to_global(array: np.ndarray, data_sharding) -> jax.Array:
    """Create one global batch from an identical full host copy on every process."""
    array = np.asarray(array)
    if array.shape[0] % jax.process_count() != 0:
        raise ValueError("Replay batch is not divisible by the JAX process count.")
    local_size = array.shape[0] // jax.process_count()
    start = jax.process_index() * local_size
    local = array[start : start + local_size]
    return jax.make_array_from_process_local_data(data_sharding, local)


def _decode_model_velocity(
    model: _model.BaseModel,
    observation: _model.Observation,
    candidate_paths: _model.Actions,
    *,
    field_time: float,
) -> _model.Actions:
    """Evaluate pi0.5 at one fixed time without constructing GT interpolation."""
    observation = _model.preprocess_observation(None, observation, train=False)
    batch_size = candidate_paths.shape[0]
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm(
        [prefix_tokens, None],
        mask=prefix_attn_mask,
        positions=positions,
    )

    times = jnp.full((batch_size,), field_time, dtype=candidate_paths.dtype)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
        observation,
        candidate_paths,
        times,
    )
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_for_suffix = einops.repeat(
        prefix_mask,
        "b p -> b s p",
        s=suffix_tokens.shape[1],
    )
    full_attn_mask = jnp.concatenate([prefix_for_suffix, suffix_attn_mask], axis=-1)
    suffix_positions = (
        jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    )
    (prefix_out, suffix_out), _ = model.PaliGemma.llm(
        [None, suffix_tokens],
        mask=full_attn_mask,
        positions=suffix_positions,
        kv_cache=kv_cache,
        adarms_cond=[None, adarms_cond],
    )
    assert prefix_out is None
    return model.decode_actions(suffix_out[:, -model.action_horizon :])


def predict_descent_field(
    model: _model.BaseModel,
    observation: _model.Observation,
    candidate_paths: _model.Actions,
    *,
    field_time: float,
) -> _model.Actions:
    """Return the path-improvement field; OpenPI raw velocity has opposite sign."""
    velocity = _decode_model_velocity(
        model,
        observation,
        candidate_paths,
        field_time=field_time,
    )
    return -velocity


def project_bounded_actions(
    actions: _model.Actions,
    metric_tensors: dict[str, at.Array],
    energy_config: ObstacleEnergyConfig,
) -> _model.Actions:
    """Project absolute waypoint states to origin-fixed, bounded increments."""
    physical_path, _ = decode_bounded_waypoint_path(
        actions, metric_tensors, energy_config
    )
    return maybe_normalize_actions(physical_path, metric_tensors)


def _rms_per_path(values: at.Array) -> at.Array:
    return jnp.sqrt(jnp.mean(jnp.square(values[:, 1:, :]), axis=(-2, -1)) + 1.0e-12)


def _cap_direction_rms(values: at.Array, max_rms: float) -> tuple[at.Array, at.Array]:
    rms = _rms_per_path(values)
    scale = jnp.minimum(1.0, max_rms / jnp.maximum(rms, 1.0e-6))
    capped = values * scale[:, None, None]
    capped = capped.at[:, 0, :].set(0.0)
    finite = jnp.all(jnp.isfinite(capped), axis=(-2, -1))
    capped = jnp.where(finite[:, None, None], capped, 0.0)
    return capped, rms


def _energy_per_path(
    actions: _model.Actions,
    metric_tensors: dict[str, at.Array],
    energy_config: ObstacleEnergyConfig,
) -> at.Array:
    """Vectorize the existing batch energy into one scalar per path."""

    def one_path(action, one_metrics):
        batched_metrics = jax.tree.map(lambda value: value[None], one_metrics)
        energy, _ = compute_obstacle_energy(
            action[None], batched_metrics, energy_config
        )
        return energy

    return jax.vmap(one_path)(actions, metric_tensors)


def backtracking_projected_step(
    current_paths: _model.Actions,
    directions: _model.Actions,
    metric_tensors: dict[str, at.Array],
    energy_config: ObstacleEnergyConfig,
    *,
    step_size: float,
    step_size_start: float | None = None,
    backtracks: int,
    backtrack_factor: float,
    energy_tolerance: float,
) -> tuple[_model.Actions, dict[str, at.Array]]:
    """Accept the first finite, non-energy-increasing projected step per path."""
    current_energy = _energy_per_path(current_paths, metric_tensors, energy_config)
    batch_size = current_paths.shape[0]
    accepted = jnp.zeros((batch_size,), dtype=bool)
    accepted_paths = current_paths
    accepted_energy = current_energy
    accepted_scale = jnp.zeros((batch_size,), dtype=current_paths.dtype)
    trial_scale = jnp.full((batch_size,), step_size, dtype=current_paths.dtype)
    # Backtrack a common fraction of the whole ramp; cap the field before this.
    directions = directions * (
        waypoint_step_sizes(current_paths, step_size, step_size_start) / step_size
    )

    def body(_, carry):
        best_paths, best_energy, best_scale, already_accepted, scales = carry
        proposal = project_bounded_actions(
            current_paths + scales[:, None, None] * directions,
            metric_tensors,
            energy_config,
        )
        proposal_energy = _energy_per_path(proposal, metric_tensors, energy_config)
        finite = jnp.all(jnp.isfinite(proposal), axis=(-2, -1)) & jnp.isfinite(
            proposal_energy
        )
        accept_now = (
            ~already_accepted
            & finite
            & (proposal_energy <= current_energy + energy_tolerance)
        )
        best_paths = jnp.where(accept_now[:, None, None], proposal, best_paths)
        best_energy = jnp.where(accept_now, proposal_energy, best_energy)
        best_scale = jnp.where(accept_now, scales, best_scale)
        already_accepted = already_accepted | accept_now
        scales = jnp.where(already_accepted, scales, scales * backtrack_factor)
        return best_paths, best_energy, best_scale, already_accepted, scales

    accepted_paths, accepted_energy, accepted_scale, accepted, _ = jax.lax.fori_loop(
        0,
        backtracks + 1,
        body,
        (accepted_paths, accepted_energy, accepted_scale, accepted, trial_scale),
    )
    return accepted_paths, {
        "energy_before_per_path": current_energy,
        "energy_after_per_path": accepted_energy,
        "accepted": accepted,
        "accepted_step_size": accepted_scale,
    }


def initialize_prior_paths(
    rng: at.KeyArrayLike,
    metric_tensors: dict[str, at.Array],
    energy_config: ObstacleEnergyConfig,
    *,
    base_batch_size: int,
    particles: int,
    action_horizon: int,
    action_dim: int,
) -> _model.Actions:
    particle_metrics = _repeat_batch_tree(metric_tensors, base_batch_size, particles)
    paths = sample_dynamic_waypoint_prior(
        rng,
        batch_size=base_batch_size * particles,
        horizon=action_horizon,
        action_dim=action_dim,
        config=energy_config.prior,
        metric_tensors=particle_metrics,
    )
    paths = bias_prior_toward_goal(paths, particle_metrics, energy_config)
    return project_bounded_actions(paths, particle_metrics, energy_config)


def query_energy_oracle(
    current_paths: _model.Actions,
    metric_tensors: dict[str, at.Array],
    energy_config: ObstacleEnergyConfig,
    *,
    base_batch_size: int,
    particles: int,
    gradient_floor: float,
    max_direction_rms: float,
    diversity_direction_weight: float,
    step_size: float,
    step_size_start: float | None = None,
    backtracks: int,
    backtrack_factor: float,
    energy_tolerance: float,
) -> tuple[_model.Actions, dict[str, at.Array]]:
    """Query and cache a projection-aware local energy-oracle direction.

    The raw negative gradient is normalized only above ``gradient_floor``.
    A guarded oracle proposal is projected and backtracked.  The returned label
    is the pre-projection direction scaled by the accepted backtracking
    fraction.  It therefore respects the field cap, and a perfect learned field
    reproduces the same feasible projected update when rollout reapplies the
    projection.
    """
    particle_metrics = _repeat_batch_tree(metric_tensors, base_batch_size, particles)

    def attraction_objective(actions):
        energy, energy_info = compute_obstacle_energy(
            actions, particle_metrics, energy_config
        )
        return energy / energy_config.energy_temperature, (energy, energy_info)

    (scaled_energy, (energy, energy_info)), attraction_grad = jax.value_and_grad(
        attraction_objective,
        has_aux=True,
    )(current_paths)
    # ``compute_obstacle_energy`` returns a batch mean.  Undo that reduction
    # for a batch-size-independent per-path gradient magnitude and threshold.
    attraction_grad = attraction_grad * current_paths.shape[0]
    raw_grad_rms = _rms_per_path(attraction_grad)
    attraction_descent = -attraction_grad / jnp.maximum(
        raw_grad_rms[:, None, None],
        gradient_floor,
    )
    attraction_descent = jnp.where(
        (raw_grad_rms > gradient_floor)[:, None, None],
        attraction_descent,
        0.0,
    )

    if particles > 1 and diversity_direction_weight > 0.0:

        def negative_diversity(actions):
            return -compute_particle_path_diversity(
                actions,
                particle_metrics,
                energy_config,
                batch_size=base_batch_size,
                particles=particles,
            )

        negative_diversity_value, diversity_grad = jax.value_and_grad(
            negative_diversity
        )(current_paths)
        diversity_grad = diversity_grad * current_paths.shape[0]
        diversity_grad_rms = _rms_per_path(diversity_grad)
        diversity_descent = -diversity_grad / jnp.maximum(
            diversity_grad_rms[:, None, None],
            gradient_floor,
        )
        diversity_descent = jnp.where(
            (diversity_grad_rms > gradient_floor)[:, None, None],
            diversity_descent,
            0.0,
        )
        proposed_direction = (
            attraction_descent + diversity_direction_weight * diversity_descent
        )
        path_diversity = -negative_diversity_value
    else:
        diversity_grad_rms = jnp.zeros_like(raw_grad_rms)
        proposed_direction = attraction_descent
        path_diversity = jnp.asarray(0.0, dtype=current_paths.dtype)

    proposed_direction, proposed_rms = _cap_direction_rms(
        proposed_direction,
        max_direction_rms,
    )
    _, acceptance = backtracking_projected_step(
        current_paths,
        proposed_direction,
        particle_metrics,
        energy_config,
        step_size=step_size,
        step_size_start=step_size_start,
        backtracks=backtracks,
        backtrack_factor=backtrack_factor,
        energy_tolerance=energy_tolerance,
    )
    accepted_fraction = acceptance["accepted_step_size"] / step_size
    # Cache only the backtracking fraction. Rollouts apply the waypoint ramp
    # once to this label; embedding it here too would square the ramp.
    oracle_direction = proposed_direction * accepted_fraction[:, None, None]
    oracle_direction = oracle_direction.at[:, 0, :].set(0.0)
    oracle_direction = jnp.where(
        jnp.isfinite(oracle_direction),
        oracle_direction,
        0.0,
    )
    oracle_direction_rms = _rms_per_path(oracle_direction)
    return oracle_direction, {
        **energy_info,
        "scaled_energy": scaled_energy,
        "diagnostic_energy": energy,
        "raw_oracle_grad_rms_per_path": raw_grad_rms,
        "raw_oracle_grad_rms": jnp.mean(raw_grad_rms),
        "oracle_direction_rms": jnp.mean(oracle_direction_rms),
        "proposed_direction_rms": jnp.mean(proposed_rms),
        "diversity_grad_rms": jnp.mean(diversity_grad_rms),
        "particle_path_diversity_m": path_diversity,
        "oracle_acceptance_rate": jnp.mean(acceptance["accepted"].astype(jnp.float32)),
        "oracle_step_size_mean": jnp.mean(acceptance["accepted_step_size"]),
    }


def _masked_field_loss(
    predicted_direction: _model.Actions,
    oracle_direction: _model.Actions,
) -> at.Array:
    """Waypoint zero is a fixed origin and is excluded from field regression."""
    error = predicted_direction[:, 1:, :] - oracle_direction[:, 1:, :]
    return 0.5 * jnp.mean(jnp.square(error))


def _field_oracle_cosine(
    predicted_direction: _model.Actions,
    oracle_direction: _model.Actions,
) -> at.Array:
    predicted = predicted_direction[:, 1:, :].reshape(
        (predicted_direction.shape[0], -1)
    )
    target = oracle_direction[:, 1:, :].reshape((oracle_direction.shape[0], -1))
    target_norm = jnp.linalg.norm(target, axis=-1)
    predicted_norm = jnp.linalg.norm(predicted, axis=-1)
    cosine = jnp.sum(predicted * target, axis=-1) / jnp.maximum(
        predicted_norm * target_norm,
        1.0e-8,
    )
    valid = target_norm > 1.0e-6
    return jnp.sum(jnp.where(valid, cosine, 0.0)) / jnp.maximum(jnp.sum(valid), 1)


def train_field_step(
    config: _config.TrainConfig,
    field_time: float,
    state: training_utils.TrainState,
    observation: _model.Observation,
    candidate_paths: _model.Actions,
    oracle_direction: _model.Actions,
    *,
    base_batch_size: int,
    particles: int,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """One replay SGD update; no ESDF or oracle call exists in this graph."""
    model = nnx.merge(state.model_def, state.params)
    model.train()
    particle_observation = _repeat_batch_tree(observation, base_batch_size, particles)

    def loss_fn(model):
        predicted_direction = predict_descent_field(
            model,
            particle_observation,
            candidate_paths,
            field_time=field_time,
        )
        predicted_direction = predicted_direction.at[:, 0, :].set(0.0)
        loss = _masked_field_loss(predicted_direction, oracle_direction)
        return loss, {
            "zero_predictor_loss": 0.5 * jnp.mean(jnp.square(oracle_direction[:, 1:])),
            "field_oracle_cosine": _field_oracle_cosine(
                predicted_direction,
                oracle_direction,
            ),
            "field_direction_rms": jnp.mean(_rms_per_path(predicted_direction)),
            "oracle_direction_rms": jnp.mean(_rms_per_path(oracle_direction)),
        }

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model
    )
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_trainable_params = optax.apply_updates(params, updates)
    nnx.update(model, new_trainable_params)
    new_params = nnx.state(model)
    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
    )
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1.0 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )
    return new_state, {"loss": loss, **info, "grad_norm": optax.global_norm(grads)}


def rollout_learned_field(
    state: training_utils.TrainState,
    observation: _model.Observation,
    current_paths: _model.Actions,
    oracle_direction: _model.Actions,
    metric_tensors: dict[str, at.Array],
    energy_config: ObstacleEnergyConfig,
    *,
    base_batch_size: int,
    particles: int,
    field_time: float,
    max_direction_rms: float,
    step_size: float,
    step_size_start: float | None = None,
    backtracks: int,
    backtrack_factor: float,
    energy_tolerance: float,
) -> tuple[_model.Actions, dict[str, at.Array]]:
    """Advance one model-induced state outside the optimizer/autodiff graph."""
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    particle_observation = _repeat_batch_tree(observation, base_batch_size, particles)
    particle_metrics = _repeat_batch_tree(metric_tensors, base_batch_size, particles)
    predicted_direction = predict_descent_field(
        model,
        particle_observation,
        current_paths,
        field_time=field_time,
    )
    predicted_direction, uncapped_rms = _cap_direction_rms(
        predicted_direction,
        max_direction_rms,
    )
    next_paths, acceptance = backtracking_projected_step(
        current_paths,
        predicted_direction,
        particle_metrics,
        energy_config,
        step_size=step_size,
        step_size_start=step_size_start,
        backtracks=backtracks,
        backtrack_factor=backtrack_factor,
        energy_tolerance=energy_tolerance,
    )
    current_physical = maybe_unnormalize_actions(current_paths, particle_metrics)
    next_physical = maybe_unnormalize_actions(next_paths, particle_metrics)
    path_change = _rms_per_path(next_physical - current_physical)
    energy_after, energy_info = compute_obstacle_energy(
        next_paths,
        particle_metrics,
        energy_config,
    )
    energy_before_per_path = acceptance["energy_before_per_path"]
    energy_after_per_path = acceptance["energy_after_per_path"]
    relative_improvement = (
        jnp.mean(energy_before_per_path) - jnp.mean(energy_after_per_path)
    ) / jnp.maximum(jnp.abs(jnp.mean(energy_before_per_path)), 1.0e-6)
    return next_paths, {
        **energy_info,
        "energy_after": energy_after,
        "energy_before": jnp.mean(energy_before_per_path),
        "relative_energy_improvement": relative_improvement,
        "path_change_per_path_m": path_change,
        "field_oracle_cosine": _field_oracle_cosine(
            predicted_direction,
            oracle_direction,
        ),
        "field_oracle_cosine_valid_count": jnp.sum(
            jnp.linalg.norm(
                oracle_direction[:, 1:].reshape((current_paths.shape[0], -1)), axis=-1
            )
            > 1e-6
        ),
        "field_direction_rms": jnp.mean(_rms_per_path(predicted_direction)),
        "field_uncapped_rms": jnp.mean(uncapped_rms),
        "rollout_acceptance_rate": jnp.mean(acceptance["accepted"].astype(jnp.float32)),
        "rollout_step_size_mean": jnp.mean(acceptance["accepted_step_size"]),
    }


def validation_model_rollout(
    state,
    observation,
    initial_paths,
    metric_tensors,
    steps,
    *,
    energy_config,
    field_time,
    step_size,
    step_size_start=None,
    max_direction_rms,
):
    """Restart from the prior and apply the current model for completed rounds."""
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    return rollout_model_only(
        initial_paths,
        lambda paths: predict_descent_field(
            model, observation, paths, field_time=field_time
        ),
        lambda paths: project_bounded_actions(paths, metric_tensors, energy_config),
        steps=steps,
        step_size=step_size,
        step_size_start=step_size_start,
        max_direction_rms=max_direction_rms,
    )


def _format_metrics(metrics: dict[str, float], keys: Sequence[str]) -> str:
    return ", ".join(f"{key}={metrics[key]:.4f}" for key in keys if key in metrics)


def _mean_metric_dict(infos: list[dict[str, Any]]) -> dict[str, float]:
    if not infos:
        return {}
    scalar_infos = []
    for info in infos:
        scalar_infos.append(
            {
                key: np.mean(host_numpy(value))
                for key, value in info.items()
                if np.size(value) > 0
            }
        )
    return {
        key: float(np.mean([info[key] for info in scalar_infos]))
        for key in scalar_infos[0]
    }


def _resume_signature(args: argparse.Namespace) -> str:
    excluded = {
        "init_only",
        "log_interval",
        "overwrite",
        "resume",
        "save_interval",
        "aggregation_plot_interval",
        "train_plot_num_images",
        "trainable_snapshot_interval",
        "wandb_enabled",
        "wandb_entity",
        "relabel_max_step_from_config",
        "relabel_esdf_cutoff_from_config",
        "migrate_goal_conditioning_from_config",
        "require_all_goal_modalities",
        # Execution budgets can be extended without changing the oracle,
        # replay draws, optimizer schedule, or validation stopping rule.
        "aggregation_rounds",
        "continue_aggregation_until_round",
        "post_aggregation_updates",
        "replay_only",
        "num_train_steps",
        "eval_plot_count",
        "eval_plot_interval",
        "replay_eval_interval",
    }
    resume_config = {
        key: value for key, value in vars(args).items() if key not in excluded
    }
    # Preserve exact signatures for historical runs with the feature disabled.
    if resume_config.get("esdf_learning_cutoff_m") is None:
        resume_config.pop("esdf_learning_cutoff_m", None)
    resume_config["aggregation_algorithm_version"] = AGGREGATION_ALGORITHM_VERSION
    # Fixed-depth validation has a different plateau history and must not be
    # silently reused with the new stopping measurements on resume.
    resume_config["validation_rollout_policy"] = "completed_aggregation_rounds"
    resume_config["process_count"] = jax.process_count()
    return stable_config_signature(resume_config)


def _replay_checkpoint_path(checkpoint_dir: epath.Path, step: int) -> epath.Path:
    return checkpoint_dir / f"aggregation_state_{step:08d}.npz"


def max_step_source_signature(args, source):
    """Validate an explicit cap change without relaxing other resume guards."""
    return _objective_source_signature(args, source, "max_step_length_m")


def esdf_cutoff_source_signature(args, source):
    """Allow only an explicit ESDF cutoff change; preserve all other guards."""
    return _objective_source_signature(args, source, "esdf_learning_cutoff_m")


def goal_conditioning_source_signature(args, source):
    """Authorize an explicit condition-pool migration and optional ESDF cutoff."""
    if source["aggregation_algorithm_version"] != AGGREGATION_ALGORITHM_VERSION:
        raise ValueError("Goal migration requires the same aggregation algorithm.")
    if source["process_count"] != jax.process_count():
        raise ValueError("Goal migration requires the same process count.")
    old = dict(source)
    for key in ("aggregation_algorithm_version", "validation_rollout_policy",
                "process_count", "global_device_count", "validation_role"):
        old.pop(key, None)
    candidate = argparse.Namespace(**vars(args))
    candidate.exp_name = old["exp_name"]
    candidate.manifest_path = old["manifest_path"]
    for key in (
        "sampled_goal_fraction",
        "object_text_goal_prob",
        "object_image_goal_prob",
        "object_waypoint_goal_prob",
        "esdf_learning_cutoff_m",
    ):
        setattr(candidate, key, old.get(key))
    signature = _resume_signature(argparse.Namespace(**old))
    if _resume_signature(candidate) != signature:
        raise ValueError(
            "Goal migration may change only goal-mixture settings, the ESDF learning "
            "cutoff, exp_name, and the verified manifest location."
        )
    return signature


def _objective_source_signature(args, source, changed_field):
    if source["aggregation_algorithm_version"] != AGGREGATION_ALGORITHM_VERSION:
        raise ValueError("Cap relabel requires the same aggregation algorithm.")
    if source["process_count"] != jax.process_count():
        raise ValueError("Cap relabel requires the same process count.")
    old = dict(source)
    for key in ("aggregation_algorithm_version", "validation_rollout_policy",
                "process_count", "global_device_count", "validation_role"):
        old.pop(key, None)
    candidate = argparse.Namespace(**vars(args))
    candidate.exp_name = old["exp_name"]
    setattr(candidate, changed_field, old.get(changed_field))
    # A pinned copy may live at a different manifest path. The full training
    # and validation condition fingerprints below must still match exactly;
    # changing the manifest location never authorizes changing its contents.
    candidate.manifest_path = old["manifest_path"]
    signature = _resume_signature(argparse.Namespace(**old))
    if _resume_signature(candidate) != signature:
        raise ValueError(f"Cap relabel may change only {changed_field}, exp_name, and the verified manifest location.")
    return signature


def relabel_max_step_replay(
    replay, query, signature, *, condition_fingerprints=None
):
    """Refresh all historical labels atomically at a completed round boundary."""
    if replay.update_in_round or replay.replay_updates or replay.labelled_rounds != replay.round_index:
        raise ValueError("Cap relabel requires a completed round without replay-only updates.")
    directions, gradients, records = [], [], []
    for round_paths in replay.visited_paths:
        results = [query(batch, paths) for batch, paths in enumerate(round_paths)]
        directions.append(np.stack([r[0] for r in results]))
        gradients.append(np.stack([r[1] for r in results]))
        records.append({key: float(np.mean([r[2][key] for r in results])) for key in results[0][2]})
    result = dataclasses.replace(
        replay, config_signature=signature, oracle_directions=directions,
        oracle_gradient_rms=gradients, oracle_records=records,
        condition_fingerprints=(
            list(condition_fingerprints)
            if condition_fingerprints is not None
            else replay.condition_fingerprints
        ),
        convergence_streak=0, benefit_state={}, validation_records=[],
        aggregation_stop_reason="",
    )
    result.validate()
    return result


def save_training_progress(
    replay: AggregationReplay,
    checkpoint_manager,
    train_state: training_utils.TrainState,
    data_loader,
    config: _config.TrainConfig,
    *,
    save_trainable: bool,
    wait: bool,
) -> int:
    """Persist replay first, then the matching model/optimizer step."""
    if wait:
        # Resolve any earlier async save before deciding whether this exact
        # optimizer step already has a committed model checkpoint.
        checkpoint_manager.wait_until_finished()
    step = int(jax.device_get(train_state.step))
    if jax.process_index() == 0:
        replay.save(_replay_checkpoint_path(config.checkpoint_dir, step))
    if jax.process_count() > 1:
        multihost_utils.sync_global_devices(f"replay_saved_{step}")
    if step not in set(checkpoint_manager.all_steps()):
        _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
    if save_trainable:
        save_trainable_params_checkpoint(train_state, config, step)
    if wait:
        checkpoint_manager.wait_until_finished()
        if jax.process_index() == 0:
            prune_auxiliary_checkpoints(
                config.checkpoint_dir,
                set(checkpoint_manager.all_steps()),
            )
        if jax.process_count() > 1:
            multihost_utils.sync_global_devices(f"auxiliary_checkpoints_pruned_{step}")
    logging.info("Saved model and aggregation replay at optimizer step %d", step)
    return step


def init_aggregation_wandb(
    config: _config.TrainConfig,
    energy_config: ObstacleEnergyConfig,
    args: argparse.Namespace,
    modality_counts: dict[str, int],
    *,
    resuming: bool,
) -> None:
    if not config.wandb_enabled or jax.process_index() != 0:
        wandb.init(mode="disabled")
        return
    if not wandb.login():
        raise RuntimeError(
            "W&B authentication failed. Run `wandb login` in the ASTAR environment "
            "or export WANDB_API_KEY before submitting."
        )
    if resuming and not (
        (
            args.relabel_max_step_from_config
            or args.relabel_esdf_cutoff_from_config
            or args.migrate_goal_conditioning_from_config
        )
        and not (config.checkpoint_dir / "wandb_id.txt").exists()
    ):
        run_id_path = config.checkpoint_dir / "wandb_id.txt"
        if not run_id_path.exists():
            raise FileNotFoundError(f"Cannot resume W&B; missing {run_id_path}")
        wandb.init(
            id=run_id_path.read_text().strip(),
            resume="must",
            project=config.project_name,
            entity=args.wandb_entity,
        )
        wandb.config.update(
            {
                **vars(args),
                "aggregation_algorithm_version": AGGREGATION_ALGORITHM_VERSION,
                "condition_pool_examples": config.batch_size * args.aggregation_batches,
                "goal_conditioning_counts": modality_counts,
            },
            allow_val_change=True,
        )
        wandb.define_metric("*", step_metric="optimizer_step")
        return

    wandb.init(
        name=config.exp_name,
        project=config.project_name,
        entity=args.wandb_entity,
        config={
            **vars(args),
            "train_objective": "energy_oracle_dataset_aggregation",
            "ground_truth_trajectory_used": False,
            "field_type": "autonomous_fixed_time_pi05",
            "field_time": args.field_time,
            "replay_factorization": "fixed_conditions_plus_path_direction_history",
            "sampling_prior": "goal_biased_forward_unicycle",
            "aggregation_algorithm_version": AGGREGATION_ALGORITHM_VERSION,
            "condition_pool_examples": config.batch_size * args.aggregation_batches,
            "goal_conditioning_counts": modality_counts,
            "collision_weight": energy_config.collision_weight,
            "goal_weight": energy_config.goal_weight,
            "progress_weight": energy_config.progress_weight,
        },
    )
    (config.checkpoint_dir / "wandb_id.txt").write_text(wandb.run.id)
    wandb.define_metric("*", step_metric="optimizer_step")


def aggregation_path_figure(
    replay: AggregationReplay,
    metric_tensors: dict[str, at.Array],
    energy_config: ObstacleEnergyConfig,
    *,
    particles: int,
    max_rounds_to_show: int = 12,
):
    """Visualize visited paths only; no dataset/MPPI path is displayed."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib is unavailable; skipping aggregation path plot.")
        return None

    rounds = replay.visited_paths[-max_rounds_to_show:]
    display_paths = list(rounds) + [replay.current_paths]
    repeated_metrics = _repeat_batch_tree(
        metric_tensors, replay.current_paths.shape[1] // particles, particles
    )
    physical = []
    for paths in display_paths:
        decoded, _ = decode_bounded_waypoint_path(
            jnp.asarray(paths[0]),
            repeated_metrics,
            energy_config,
        )
        physical.append(
            np.asarray(jax.device_get(decoded[:particles]), dtype=np.float32)
        )

    esdf = np.asarray(jax.device_get(metric_tensors["esdf"][0]), dtype=np.float32)
    x_min = float(np.asarray(jax.device_get(metric_tensors["esdf_x_min"][0])))
    y_min = float(np.asarray(jax.device_get(metric_tensors["esdf_y_min"][0])))
    resolution = float(np.asarray(jax.device_get(metric_tensors["esdf_resolution"][0])))
    goal = np.asarray(jax.device_get(metric_tensors["goal_xy"][0]), dtype=np.float32)
    rows_count, cols_count = esdf.shape
    extent = [
        x_min,
        x_min + cols_count * resolution,
        y_min,
        y_min + rows_count * resolution,
    ]
    fig, ax = plt.subplots(figsize=(8.0, 6.5), dpi=130)
    finite_esdf = np.where(np.isfinite(esdf), esdf, np.nan)
    color_scale = esdf_display_scale(esdf)
    image = ax.imshow(
        finite_esdf,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        vmin=-color_scale,
        vmax=color_scale,
        aspect="equal",
    )
    colors = plt.cm.plasma(np.linspace(0.05, 0.9, len(physical)))
    first_round_number = max(0, replay.labelled_rounds - len(rounds))
    for display_index, paths in enumerate(physical):
        is_current = display_index == len(physical) - 1
        round_label = (
            f"frontier k={replay.round_index}"
            if is_current
            else f"visited k={first_round_number + display_index}"
        )
        for particle_index, path in enumerate(paths):
            ax.plot(
                path[:, 0],
                path[:, 1],
                color=colors[display_index],
                linewidth=2.6 if is_current else 1.1,
                alpha=1.0 if is_current else 0.55,
                label=round_label if particle_index == 0 else None,
            )
    ax.scatter(
        [0.0], [0.0], marker="o", color="white", edgecolor="black", s=55, label="robot"
    )
    ax.scatter([goal[0]], [goal[1]], marker="*", color="red", s=120, label="goal")
    ax.set_title("Energy-oracle dataset aggregation (no dataset trajectory)")
    ax.set_xlabel("local x [m]")
    ax.set_ylabel("local y [m]")
    ax.legend(fontsize=7, loc="best")
    fig.colorbar(image, ax=ax, label="ESDF [m]")
    fig.tight_layout()
    wandb_image = wandb.Image(fig)
    plt.close(fig)
    return wandb_image


def aggregation_transition_figure(
    replay: AggregationReplay,
    metric_tensors: dict[str, at.Array],
    energy_config: ObstacleEnergyConfig,
    round_record: Mapping[str, float | int | str],
    *,
    particles: int,
):
    """Show the latest learned-field transition as x_k, overlay, and x_{k+1}."""
    if not replay.visited_paths:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning(
            "matplotlib is unavailable; skipping aggregation transition plot."
        )
        return None

    before_bank = replay.visited_paths[-1]
    after_bank = replay.current_paths
    base_batch_size = after_bank.shape[1] // particles
    repeated_metrics = _repeat_batch_tree(
        metric_tensors,
        base_batch_size,
        particles,
    )

    def decode_first_condition(path_bank: np.ndarray) -> np.ndarray:
        decoded, _ = decode_bounded_waypoint_path(
            jnp.asarray(path_bank[0]),
            repeated_metrics,
            energy_config,
        )
        return np.asarray(jax.device_get(decoded[:particles]), dtype=np.float32)

    before_paths = decode_first_condition(before_bank)
    after_paths = decode_first_condition(after_bank)
    esdf = np.asarray(jax.device_get(metric_tensors["esdf"][0]), dtype=np.float32)
    x_min = float(np.asarray(jax.device_get(metric_tensors["esdf_x_min"][0])))
    y_min = float(np.asarray(jax.device_get(metric_tensors["esdf_y_min"][0])))
    resolution = float(np.asarray(jax.device_get(metric_tensors["esdf_resolution"][0])))
    goal = np.asarray(jax.device_get(metric_tensors["goal_xy"][0]), dtype=np.float32)
    rows_count, cols_count = esdf.shape
    extent = [
        x_min,
        x_min + cols_count * resolution,
        y_min,
        y_min + rows_count * resolution,
    ]
    finite_esdf = np.where(np.isfinite(esdf), esdf, np.nan)
    color_scale = esdf_display_scale(esdf)
    colors = plt.cm.tab10(np.linspace(0.0, 0.8, particles))
    source_depth = int(round_record["source_depth"])
    target_depth = int(round_record["target_depth"])

    fig, axes = plt.subplots(
        1, 3, figsize=(16.5, 5.4), dpi=130, sharex=True, sharey=True
    )
    panels = (
        (f"Before: x{source_depth}", True, False),
        (f"Transition: x{source_depth} → x{target_depth}", True, True),
        (f"After: x{target_depth}", False, True),
    )
    background = None
    for ax, (title, show_before, show_after) in zip(axes, panels, strict=True):
        background = ax.imshow(
            finite_esdf,
            origin="lower",
            extent=extent,
            cmap="coolwarm",
            vmin=-color_scale,
            vmax=color_scale,
            aspect="equal",
        )
        for particle_index, color in enumerate(colors):
            if show_before:
                path = before_paths[particle_index]
                ax.plot(
                    path[:, 0],
                    path[:, 1],
                    color=color,
                    linestyle="--",
                    linewidth=1.8,
                    alpha=0.8,
                    label=f"particle {particle_index + 1}: before",
                )
            if show_after:
                path = after_paths[particle_index]
                ax.plot(
                    path[:, 0],
                    path[:, 1],
                    color=color,
                    linestyle="-",
                    linewidth=2.6,
                    alpha=1.0,
                    label=f"particle {particle_index + 1}: after",
                )
        ax.scatter(
            [0.0],
            [0.0],
            marker="o",
            color="white",
            edgecolor="black",
            s=42,
            label="robot",
        )
        ax.scatter([goal[0]], [goal[1]], marker="*", color="red", s=105, label="goal")
        ax.set_title(title)
        ax.set_xlabel("local x [m]")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("local y [m]")
    axes[1].legend(fontsize=7, loc="best")
    assert background is not None
    fig.colorbar(background, ax=axes, label="ESDF [m]", fraction=0.025, pad=0.02)
    fig.suptitle(
        "Aggregation "
        f"{int(round_record['round'])}: energy {float(round_record['energy_before']):.3f} "
        f"→ {float(round_record['energy_after']):.3f}, median Δpath "
        f"{float(round_record['path_change_median']):.3f} m",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.06, right=0.92, bottom=0.10, top=0.84, wspace=0.12)
    caption = (
        f"Aggregation {int(round_record['round'])}: x{source_depth} to x{target_depth}; "
        "dashed paths are before the learned-field step and solid paths are after it."
    )
    wandb_image = wandb.Image(fig, caption=caption)
    plt.close(fig)
    return wandb_image


def _round_metrics(
    oracle_summary: dict[str, float],
    raw_gradients: np.ndarray,
    rollout_infos: list[dict[str, Any]],
) -> dict[str, float]:
    rollout_summary = _mean_metric_dict(rollout_infos)
    path_changes = np.concatenate(
        [
            np.asarray(host_numpy(info["path_change_per_path_m"]), dtype=np.float32)
            for info in rollout_infos
        ]
    )
    raw_gradients = np.asarray(raw_gradients, dtype=np.float32).reshape(-1)
    cosine_count = sum(
        float(info["field_oracle_cosine_valid_count"]) for info in rollout_infos
    )
    cosine_sum = sum(
        float(info["field_oracle_cosine"])
        * float(info["field_oracle_cosine_valid_count"])
        for info in rollout_infos
    )
    return {
        "energy_before": rollout_summary["energy_before"],
        "energy_after": rollout_summary["energy_after"],
        "relative_energy_improvement": rollout_summary["relative_energy_improvement"],
        "path_change_mean": float(np.mean(path_changes)),
        "path_change_median": float(np.median(path_changes)),
        "path_change_p95": float(np.quantile(path_changes, 0.95)),
        "oracle_grad_rms": float(np.mean(raw_gradients)),
        "oracle_grad_rms_p95": float(np.quantile(raw_gradients, 0.95)),
        "oracle_acceptance_rate": oracle_summary["oracle_acceptance_rate"],
        "rollout_acceptance_rate": rollout_summary["rollout_acceptance_rate"],
        "field_oracle_cosine": cosine_sum / cosine_count if cosine_count else None,
        "collision_rate": rollout_summary["collision_rate"],
        "clearance_violation_rate": rollout_summary.get(
            "unsafe_rate", rollout_summary["collision_rate"]
        ),
        "invalid_esdf_rate": rollout_summary["invalid_esdf_rate"],
        "progress_ratio": rollout_summary["achieved_required_progress_ratio"],
        "achieved_progress_m": rollout_summary["achieved_progress_m"],
        "required_progress_m": rollout_summary["required_progress_m"],
        **{f"{term}_energy": rollout_summary[f"{term}_energy"] for term in ENERGY_TERMS},
        **{
            key: value
            for key, value in rollout_summary.items()
            if key.startswith("weighted_") and key.endswith("_energy")
        },
    }


def _convergence_config_from_args(args: argparse.Namespace) -> ConvergenceConfig:
    return ConvergenceConfig(
        min_rounds=args.convergence_min_rounds,
        patience=args.convergence_patience,
        median_path_change_m=args.convergence_median_path_change_m,
        p95_path_change_m=args.convergence_p95_path_change_m,
        relative_energy_change=args.convergence_relative_energy_change,
        oracle_grad_rms=args.convergence_oracle_grad_rms,
        max_clearance_violation_rate=args.convergence_max_clearance_violation_rate,
        min_progress_ratio=args.convergence_min_progress_ratio,
    )


def _validate_args(
    args: argparse.Namespace, energy_config: ObstacleEnergyConfig
) -> None:
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive.")
    if args.relabel_max_step_from_config and not args.resume:
        raise ValueError("--relabel-max-step-from-config requires --resume.")
    if args.relabel_esdf_cutoff_from_config and not args.resume:
        raise ValueError("--relabel-esdf-cutoff-from-config requires --resume.")
    if args.migrate_goal_conditioning_from_config and not args.resume:
        raise ValueError("--migrate-goal-conditioning-from-config requires --resume.")
    if args.relabel_esdf_cutoff_from_config and args.relabel_max_step_from_config:
        raise ValueError("Choose only one objective migration per run.")
    if args.migrate_goal_conditioning_from_config and (
        args.relabel_esdf_cutoff_from_config or args.relabel_max_step_from_config
    ):
        raise ValueError("Goal conditioning migration already includes objective relabeling.")
    if args.action_dim != 2:
        raise ValueError("Energy aggregation currently requires --action-dim 2.")
    if args.action_horizon < 2:
        raise ValueError("--action-horizon must be at least 2.")
    if args.goal_waypoint_dim != 2:
        raise ValueError("--goal-waypoint-dim must be 2 for the ASTAR goal projector.")
    if args.max_goal_waypoints != 1:
        raise ValueError("--max-goal-waypoints must be 1 for this dataset view.")
    if args.random_init_action_stack and args.train_scope == "adapter":
        raise ValueError(
            "--random-init-action-stack requires --train-scope action_stack or all; "
            "adapter scope would freeze the randomly initialized action expert."
        )
    if args.checkpoint_mode != "full":
        raise ValueError("Replay-safe aggregation requires --checkpoint-mode full.")
    if args.aggregation_rounds < 0:
        raise ValueError(
            "--aggregation-rounds must be nonnegative (0 means no round cap)."
        )
    if args.continue_aggregation_until_round < 0:
        raise ValueError("--continue-aggregation-until-round must be nonnegative.")
    if args.continue_aggregation_until_round and (
        args.replay_only or args.post_aggregation_updates
    ):
        raise ValueError(
            "--continue-aggregation-until-round cannot be combined with replay-only "
            "or post-aggregation updates."
        )
    if args.post_aggregation_updates < 0:
        raise ValueError("--post-aggregation-updates must be nonnegative.")
    if args.replay_only and not args.resume:
        raise ValueError(
            "--replay-only requires --resume and an existing replay archive."
        )
    if args.eval_batches <= 0:
        raise ValueError("Validation batches must be positive.")
    if (
        args.eval_plot_count < 0
        or args.eval_plot_interval <= 0
        or args.replay_eval_interval <= 0
    ):
        raise ValueError(
            "Plot count must be nonnegative and evaluation intervals positive."
        )
    if (
        not 0.0 < args.eval_progress_threshold <= 1.0
        or args.eval_safety_spacing_m <= 0.0
    ):
        raise ValueError(
            "Validation progress threshold must be in (0, 1] and spacing positive."
        )
    BenefitTracker(
        min_rounds=args.benefit_min_rounds,
        patience=args.benefit_patience,
        rate_delta=args.benefit_rate_delta,
        progress_delta=args.benefit_progress_delta,
    )
    if args.updates_per_round <= 0:
        raise ValueError("--updates-per-round must be positive.")
    if args.aggregation_batches <= 0:
        raise ValueError("--aggregation-batches must be positive.")
    if args.train_particles <= 0:
        raise ValueError("--train-particles must be positive.")
    if not 0.0 <= args.field_time <= 1.0:
        raise ValueError("--field-time must be in [0, 1].")
    if not math.isfinite(args.aggregation_step_size) or args.aggregation_step_size <= 0.0:
        raise ValueError("--aggregation-step-size must be finite and positive.")
    if (
        not math.isfinite(args.aggregation_step_size_start)
        or not 0.0 < args.aggregation_step_size_start <= args.aggregation_step_size
    ):
        raise ValueError(
            "--aggregation-step-size-start must be finite and in (0, --aggregation-step-size]."
        )
    if args.oracle_gradient_floor <= 0.0:
        raise ValueError("--oracle-gradient-floor must be positive.")
    if args.max_field_direction_rms <= 0.0:
        raise ValueError("--max-field-direction-rms must be positive.")
    if args.rollout_backtracks < 0:
        raise ValueError("--rollout-backtracks must be non-negative.")
    if not 0.0 < args.backtrack_factor < 1.0:
        raise ValueError("--backtrack-factor must be in (0, 1).")
    if args.energy_increase_tolerance < 0.0:
        raise ValueError("--energy-increase-tolerance must be non-negative.")
    if args.diversity_direction_weight < 0.0:
        raise ValueError("--diversity-direction-weight must be non-negative.")
    if args.diversity_direction_weight > 0.0:
        logging.warning(
            "A nonzero diversity oracle is cohort-dependent rather than strictly state-local. "
            "Replay preserves adjacent particle cohorts, but 0 is the clean ScalarFlows setting."
        )
    if args.save_interval <= 0:
        raise ValueError("--save-interval must be positive.")
    if args.trainable_snapshot_interval < 0:
        raise ValueError("--trainable-snapshot-interval must be non-negative.")
    if args.aggregation_plot_interval < 0:
        raise ValueError("--aggregation-plot-interval must be non-negative.")
    if args.train_plot_num_images <= 0:
        raise ValueError("--train-plot-num-images must be positive.")
    if args.path_stride <= 0:
        raise ValueError("--path-stride must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.fsdp_devices <= 0:
        raise ValueError("--fsdp-devices must be positive.")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative.")
    if args.peak_lr <= 0.0:
        raise ValueError("--peak-lr must be positive.")
    if args.decay_steps <= 0:
        raise ValueError("--decay-steps must be positive.")
    if args.warmup_steps >= args.decay_steps:
        raise ValueError("--warmup-steps must be smaller than --decay-steps.")
    if args.decay_lr < 0.0:
        raise ValueError("--decay-lr must be non-negative.")
    if args.clip_gradient_norm <= 0.0:
        raise ValueError("--clip-gradient-norm must be positive.")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be in [0, 1).")
    if args.log_interval <= 0:
        raise ValueError("--log-interval must be positive.")
    if not 0.0 <= args.sampled_goal_fraction <= 1.0:
        raise ValueError("--sampled-goal-fraction must be in [0, 1].")
    for name in (
        "object_text_goal_prob",
        "object_image_goal_prob",
        "object_waypoint_goal_prob",
    ):
        if not 0.0 <= getattr(args, name) <= 1.0:
            option = "--" + name.replace("_", "-")
            raise ValueError(f"{option} must be in [0, 1].")
    if energy_config.prior.dt_s <= 0.0:
        raise ValueError("--prior-dt-s must be positive.")
    if energy_config.prior.min_v_mps < 0.0:
        raise ValueError("--prior-min-v-mps must be non-negative.")
    if energy_config.prior.max_v_mps < energy_config.prior.min_v_mps:
        raise ValueError("--prior-max-v-mps must be at least --prior-min-v-mps.")
    if energy_config.prior.max_omega_radps < 0.0:
        raise ValueError("--prior-max-omega-radps must be non-negative.")
    if (
        not math.isfinite(energy_config.prior.length_scale)
        or energy_config.prior.length_scale <= 0.0
    ):
        raise ValueError("--prior-length-scale must be finite and positive.")
    if energy_config.safety_margin_m < 0.0:
        raise ValueError("--obstacle-safety-margin-m must be non-negative.")
    if energy_config.clearance_cap_m <= 0.0:
        raise ValueError("--clearance-cap-m must be positive.")
    if energy_config.esdf_learning_cutoff_m is not None and not (
        math.isfinite(energy_config.esdf_learning_cutoff_m)
        and energy_config.esdf_learning_cutoff_m > 0.0
        and energy_config.esdf_learning_cutoff_m >= energy_config.safety_margin_m
    ):
        raise ValueError(
            "--esdf-learning-cutoff-m is measured after subtracting robot radius "
            "and must be at least --obstacle-safety-margin-m."
        )
    if not 0.0 <= energy_config.esdf_ramp_start_m < energy_config.esdf_min_x_m:
        raise ValueError("Require 0 <= --esdf-ramp-start-m < --esdf-min-x-m.")
    if energy_config.segment_samples <= 0:
        raise ValueError("--segment-samples must be positive.")
    if energy_config.min_step_scale_m <= 0.0:
        raise ValueError("--min-step-scale-m must be positive.")
    if energy_config.max_step_length_m <= 0.0:
        raise ValueError("--max-step-length-m must be positive.")
    if not math.isfinite(energy_config.path_detour_factor) or (
        energy_config.path_detour_factor < 1.0
    ):
        raise ValueError("--path-detour-factor must be finite and at least 1.")
    if energy_config.max_increment_correction_m <= 0.0:
        raise ValueError("--max-increment-correction-m must be positive.")
    if not 0.0 <= energy_config.prior_goal_heading_fraction <= 1.0:
        raise ValueError("--prior-goal-heading-fraction must be in [0, 1].")
    if not 0.0 < energy_config.prior_goal_heading_limit_rad <= math.pi:
        raise ValueError("--prior-goal-heading-limit-rad must be in (0, pi].")
    if not 0.0 < energy_config.required_progress_fraction <= 1.0:
        raise ValueError("--required-progress-fraction must be in (0, 1].")
    if energy_config.energy_temperature <= 0.0:
        raise ValueError("--energy-temperature must be positive.")
    for name in (
        "collision_weight",
        "clearance_weight",
        "goal_weight",
        "progress_weight",
        "early_heading_weight",
        "smoothness_weight",
    ):
        if getattr(energy_config, name) < 0.0:
            raise ValueError(f"Energy weight {name} must be non-negative.")
    _convergence_config_from_args(args)


def main(args: argparse.Namespace) -> None:
    init_logging()
    logging.info("Running ASTAR energy-oracle aggregation on %s", platform.node())
    primary = jax.process_index() == 0
    logging.info(
        "JAX process %d/%d: %d local devices, %d global devices",
        jax.process_index(), jax.process_count(), jax.local_device_count(), jax.device_count(),
    )

    shutdown_requested = False
    local_shutdown_requested = False

    def should_stop():
        nonlocal shutdown_requested
        requested = np.asarray(local_shutdown_requested or shutdown_requested)
        if jax.process_count() > 1:
            requested = multihost_utils.process_allgather(requested)
        shutdown_requested = bool(np.any(requested))
        return shutdown_requested

    def request_checkpoint_and_stop(signum, _frame):
        nonlocal local_shutdown_requested
        local_shutdown_requested = True
        logging.warning(
            "Received signal %s; will save model plus replay after the current update.",
            signum,
        )

    signal.signal(signal.SIGUSR1, request_checkpoint_and_stop)

    # This compatibility value is not the loop termination criterion. LR decay
    # depends on decay_steps and remains at decay_lr during longer runs.
    args.num_train_steps = max(
        args.decay_steps, args.aggregation_rounds * args.updates_per_round
    )
    prior_config = DynamicWaypointPriorConfig(
        dt_s=args.prior_dt_s,
        min_v_mps=args.prior_min_v_mps,
        max_v_mps=args.prior_max_v_mps,
        max_omega_radps=args.prior_max_omega_radps,
        length_scale=args.prior_length_scale,
        forward_only=args.prior_forward_only,
    )
    energy_config = ObstacleEnergyConfig(
        sample_steps=1,
        prior=prior_config,
        collision_weight=args.collision_weight,
        clearance_weight=args.clearance_weight,
        goal_weight=args.goal_weight,
        progress_weight=args.progress_weight,
        early_heading_weight=args.early_heading_weight,
        smoothness_weight=args.smoothness_weight,
        safety_margin_m=args.obstacle_safety_margin_m,
        clearance_cap_m=args.clearance_cap_m,
        esdf_learning_cutoff_m=args.esdf_learning_cutoff_m,
        esdf_ramp_start_m=args.esdf_ramp_start_m,
        esdf_min_x_m=args.esdf_min_x_m,
        strict_esdf_coverage=args.strict_esdf_coverage,
        min_step_scale_m=args.min_step_scale_m,
        segment_samples=args.segment_samples,
        max_step_length_m=args.max_step_length_m,
        path_detour_factor=args.path_detour_factor,
        max_increment_correction_m=args.max_increment_correction_m,
        prior_goal_heading_fraction=args.prior_goal_heading_fraction,
        prior_goal_heading_limit_rad=args.prior_goal_heading_limit_rad,
        required_progress_fraction=args.required_progress_fraction,
        oracle_step_size=args.aggregation_step_size,
        energy_temperature=args.energy_temperature,
        train_particles=args.train_particles,
        diversity_direction_weight=args.diversity_direction_weight,
    )
    _validate_args(args, energy_config)
    logging.info(
        "Waypoint adjustment: fixed origin; %d future coefficients linearly spaced: %s",
        args.action_horizon - 1,
        np.linspace(
            args.aggregation_step_size_start, args.aggregation_step_size,
            args.action_horizon - 1,
        ).round(6).tolist(),
    )
    config = create_train_config(args)
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by {jax.device_count()} devices."
        )
    if config.batch_size * args.train_particles % jax.device_count() != 0:
        raise ValueError(
            "Particle-expanded batch must be divisible by the device count."
        )

    jax_cache_dir = epath.Path(
        os.environ.get(
            "JAX_COMPILATION_CACHE_DIR",
            f"/tmp/astar_jax_cache_{os.environ.get('SLURM_JOB_ID', os.getpid())}",
        )
    ).expanduser()
    jax_cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(jax_cache_dir))
    logging.info("Using JAX compilation cache at %s", jax_cache_dir)
    root_rng = jax.random.key(config.seed)
    train_rng, init_rng, prior_rng = jax.random.split(root_rng, 3)
    del train_rng  # Replay selection is deterministically derived from optimizer step.

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(sharding.DATA_AXIS),
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = initialize_bounded_checkpoint_dir(
        config.checkpoint_dir,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    train_state, train_state_sharding = init_train_state(
        config,
        init_rng,
        mesh,
        resume=resuming,
        skip_pretrained_load_patterns=(
            ACTION_STACK_PRETRAIN_SKIP_PATTERNS if args.random_init_action_stack else ()
        ),
    )
    condition_loader = create_oracle_only_data_loader(
        config,
        args,
        data_sharding,
        split="train",
        num_batches=args.aggregation_batches,
    )
    if resuming:
        train_state = _checkpoints.restore_state(
            checkpoint_manager,
            train_state,
            condition_loader,
        )
    jax.block_until_ready(train_state)
    logging.info("Initialized/restored train state at step %d", int(train_state.step))

    conditions, fingerprints = cache_condition_pool(
        condition_loader,
        args.aggregation_batches,
    )
    modality_counts = {
        key: int(
            sum(
                np.sum(host_numpy(metrics[key]))
                for _, metrics in conditions
                if key in metrics
            )
        )
        for key in (
            "goal_text_condition_mask",
            "goal_image_condition_mask",
            "goal_waypoint_condition_mask",
            "is_object_goal",
        )
    }
    modality_counts["sampled_goal"] = (
        config.batch_size * args.aggregation_batches - modality_counts["is_object_goal"]
    )
    logging.info("Cached goal conditioning counts: %s", modality_counts)
    if args.require_all_goal_modalities and any(
        modality_counts[key] <= 0
        for key in (
            "goal_text_condition_mask",
            "goal_image_condition_mask",
            "goal_waypoint_condition_mask",
            "is_object_goal",
            "sampled_goal",
        )
    ):
        raise ValueError(f"Requested multimodal pool is missing a goal modality: {modality_counts}")
    signature = _resume_signature(args)

    pinitialize_prior = jax.jit(
        functools.partial(
            initialize_prior_paths,
            energy_config=energy_config,
            base_batch_size=config.batch_size,
            particles=args.train_particles,
            action_horizon=args.action_horizon,
            action_dim=args.action_dim,
        ),
        in_shardings=(replicated_sharding, data_sharding),
        out_shardings=data_sharding,
    )
    pquery_oracle = jax.jit(
        functools.partial(
            query_energy_oracle,
            energy_config=energy_config,
            base_batch_size=config.batch_size,
            particles=args.train_particles,
            gradient_floor=args.oracle_gradient_floor,
            max_direction_rms=args.max_field_direction_rms,
            diversity_direction_weight=args.diversity_direction_weight,
            step_size=args.aggregation_step_size,
            step_size_start=args.aggregation_step_size_start,
            backtracks=args.rollout_backtracks,
            backtrack_factor=args.backtrack_factor,
            energy_tolerance=args.energy_increase_tolerance,
        ),
        in_shardings=(data_sharding, data_sharding),
        out_shardings=(data_sharding, replicated_sharding),
    )
    ptrain_field = jax.jit(
        functools.partial(
            train_field_step,
            config,
            args.field_time,
            base_batch_size=config.batch_size,
            particles=args.train_particles,
        ),
        in_shardings=(
            train_state_sharding,
            data_sharding,
            data_sharding,
            data_sharding,
        ),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(0,),
    )
    prollout_field = jax.jit(
        functools.partial(
            rollout_learned_field,
            energy_config=energy_config,
            base_batch_size=config.batch_size,
            particles=args.train_particles,
            field_time=args.field_time,
            max_direction_rms=args.max_field_direction_rms,
            step_size=args.aggregation_step_size,
            step_size_start=args.aggregation_step_size_start,
            backtracks=args.rollout_backtracks,
            backtrack_factor=args.backtrack_factor,
            energy_tolerance=args.energy_increase_tolerance,
        ),
        in_shardings=(
            train_state_sharding,
            data_sharding,
            data_sharding,
            data_sharding,
            data_sharding,
        ),
        out_shardings=(data_sharding, replicated_sharding),
    )

    restored_step = int(jax.device_get(train_state.step))
    relabel_cap = False
    migrate_conditions = False
    relabel_source = (
        args.migrate_goal_conditioning_from_config
        or args.relabel_esdf_cutoff_from_config
        or args.relabel_max_step_from_config
    )
    if resuming:
        load_signature = signature
        expected_fingerprints = fingerprints
        if relabel_source:
            with np.load(_replay_checkpoint_path(config.checkpoint_dir, restored_step), allow_pickle=False) as archive:
                saved_metadata = json.loads(str(archive["metadata_json"].item()))
                saved_signature = saved_metadata["config_signature"]
            if saved_signature != signature:
                source = json.loads(Path(relabel_source).read_text())
                validate_source = (
                    goal_conditioning_source_signature
                    if args.migrate_goal_conditioning_from_config
                    else (
                        esdf_cutoff_source_signature
                        if args.relabel_esdf_cutoff_from_config
                        else max_step_source_signature
                    )
                )
                load_signature = validate_source(args, source)
                relabel_cap = True
                migrate_conditions = bool(args.migrate_goal_conditioning_from_config)
                if migrate_conditions:
                    expected_fingerprints = saved_metadata["condition_fingerprints"]
        replay = AggregationReplay.load(
            _replay_checkpoint_path(config.checkpoint_dir, restored_step),
            expected_config_signature=load_signature,
            expected_condition_fingerprints=expected_fingerprints,
        )
    else:
        initial_paths = []
        for batch_index, (_, metric_tensors) in enumerate(conditions):
            with sharding.set_mesh(mesh):
                paths = pinitialize_prior(
                    jax.random.fold_in(prior_rng, batch_index),
                    metric_tensors,
                )
            initial_paths.append(np.asarray(host_numpy(paths), dtype=np.float32))
        replay = AggregationReplay(
            config_signature=signature,
            condition_fingerprints=fingerprints,
            current_paths=np.stack(initial_paths, axis=0),
        )

    expected_step = (
        replay.round_index * args.updates_per_round
        + replay.update_in_round
        + replay.replay_updates
    )
    if restored_step != expected_step:
        raise RuntimeError(
            f"Model step {restored_step} does not match replay phase step {expected_step}."
        )
    if replay.update_in_round > args.updates_per_round:
        raise RuntimeError(
            f"Replay update {replay.update_in_round} exceeds the configured "
            f"{args.updates_per_round} updates per round."
        )
    # Validation uses one independent candidate per example even when training
    # has multiple particles. The examples and initial paths never enter replay.
    validation_config = dataclasses.replace(
        energy_config, train_particles=1, diversity_direction_weight=0.0
    )
    validation_loader = create_oracle_only_data_loader(
        config,
        args,
        data_sharding,
        split="eval",
        num_batches=args.eval_batches,
    )
    peval_initialize = jax.jit(
        functools.partial(
            initialize_prior_paths,
            energy_config=validation_config,
            base_batch_size=config.batch_size,
            particles=1,
            action_horizon=args.action_horizon,
            action_dim=args.action_dim,
        ),
        in_shardings=(replicated_sharding, data_sharding),
        out_shardings=data_sharding,
    )
    with sharding.set_mesh(mesh):
        validation_batches, validation_fingerprints, selected_examples = (
            cache_validation_pool(
                validation_loader,
                condition_loader.dataset._split_ids,
                num_batches=args.eval_batches,
                seed=args.eval_seed,
                sanitize=sanitize_metric_tensors,
                fingerprint=condition_fingerprint,
                initialize=peval_initialize,
            )
        )
    if (
        resuming
        and not migrate_conditions
        and replay.validation_fingerprints != validation_fingerprints
    ):
        raise ValueError(
            "Validation conditions differ from the saved stopping history."
        )
    replay.validation_fingerprints = validation_fingerprints
    if relabel_cap and not args.init_only:
        source_archive = _replay_checkpoint_path(config.checkpoint_dir, restored_step)
        if primary:
            backup_label = "condition_migration" if migrate_conditions else "cap_relabel"
            backup = Path(config.checkpoint_dir) / f"before_{backup_label}_{restored_step:08d}.npz"
            if not backup.exists():
                backup.write_bytes(Path(source_archive).read_bytes())
        queried = 0

        def refresh(batch, paths):
            nonlocal queried
            if should_stop():
                raise RuntimeError("Stopped during replay relabel; original replay is preserved.")
            current = _host_batch_to_global(paths, data_sharding)
            with sharding.set_mesh(mesh):
                direction, info = pquery_oracle(current, conditions[batch][1])
            labels = np.asarray(host_numpy(direction), dtype=np.float32)
            gradients = np.asarray(host_numpy(info["raw_oracle_grad_rms_per_path"]), dtype=np.float32)
            summary = _mean_metric_dict([info])
            queried += 1
            if queried % len(conditions) == 0:
                logging.info("Replay relabel: refreshed historical round %d/%d", queried // len(conditions), replay.labelled_rounds)
            return labels, gradients, summary

        replay = relabel_max_step_replay(
            replay,
            refresh,
            signature,
            condition_fingerprints=fingerprints if migrate_conditions else None,
        )
        if primary:
            replay.save(source_archive)
            migration_record = (
                "goal_conditioning_migration.json" if migrate_conditions else "cap_relabel.json"
            )
            (Path(config.checkpoint_dir) / migration_record).write_text(json.dumps({
                "optimizer_step": restored_step, "max_step_length_m": args.max_step_length_m,
                "source_configuration": relabel_source,
                "esdf_learning_cutoff_m": args.esdf_learning_cutoff_m,
                "goal_conditioning_migrated": migrate_conditions,
                "goal_conditioning_counts": modality_counts,
                "retained_rounds": replay.labelled_rounds,
                "retained_path_count": sum(int(np.prod(paths.shape[:2])) for paths in replay.visited_paths),
            }, indent=2) + "\n")
        multihost_utils.sync_global_devices("cap_relabel_saved")
    monitoring_dir = Path(config.checkpoint_dir) / "monitoring"
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    if primary:
        (monitoring_dir / "selected_examples.json").write_text(
            json.dumps(selected_examples, indent=2) + "\n"
        )
        (monitoring_dir / "configuration.json").write_text(
            json.dumps(
                {
                    **vars(args),
                    "aggregation_algorithm_version": AGGREGATION_ALGORITHM_VERSION,
                    "validation_rollout_policy": "completed_aggregation_rounds",
                    "process_count": jax.process_count(),
                    "global_device_count": jax.device_count(),
                    "validation_role": "clip-disjoint validation used for stopping; not an untouched test set",
                },
                indent=2,
                default=str,
            )
            + "\n"
        )
    if args.init_only:
        logging.info(
            "Model, training pool, and disjoint validation pool preflight completed; exiting."
        )
        return
    init_aggregation_wandb(
        config,
        energy_config,
        args,
        modality_counts,
        resuming=resuming,
    )
    peval_rollout = jax.jit(
        functools.partial(
            validation_model_rollout,
            energy_config=validation_config,
            field_time=args.field_time,
            step_size=args.aggregation_step_size,
            step_size_start=args.aggregation_step_size_start,
            max_direction_rms=args.max_field_direction_rms,
        ),
        in_shardings=(
            train_state_sharding,
            data_sharding,
            data_sharding,
            data_sharding,
            replicated_sharding,
        ),
        out_shardings=(data_sharding, data_sharding),
    )
    dense_config = dataclasses.replace(
        validation_config,
        strict_esdf_coverage=True,
        segment_samples=max(
            energy_config.segment_samples,
            math.ceil(energy_config.max_step_length_m / args.eval_safety_spacing_m),
        ),
    )
    peval_score = jax.jit(
        functools.partial(
            score_validation_paths,
            energy_config=dense_config,
            progress_threshold=args.eval_progress_threshold,
        ),
        in_shardings=(data_sharding, data_sharding, data_sharding),
        out_shardings=data_sharding,
    )
    benefit_tracker = BenefitTracker(
        min_rounds=args.benefit_min_rounds,
        patience=args.benefit_patience,
        rate_delta=args.benefit_rate_delta,
        progress_delta=args.benefit_progress_delta,
        **replay.benefit_state,
    )

    def evaluate_current_model(*, track_benefit):
        step = int(jax.device_get(train_state.step))
        # round_index counts completed path advances, including after resume.
        # Replay-only SGD leaves this depth unchanged. A scalar loop bound
        # lets the compiled rollout handle each depth without recompilation.
        rollout_steps = replay.round_index

        def rollout_at_current_depth(state, observation, paths, metrics):
            return peval_rollout(
                state, observation, paths, metrics,
                jnp.asarray(rollout_steps, dtype=jnp.int32),
            )

        show_plots = (
            replay.round_index % args.eval_plot_interval == 0 or not track_benefit
        )
        with sharding.set_mesh(mesh):
            summary, plots = evaluate_validation_pool(
                train_state,
                validation_batches,
                rollout_at_current_depth,
                peval_score,
                output_dir=monitoring_dir,
                step=step,
                plot_count=args.eval_plot_count if show_plots else 0,
                rollout_steps=rollout_steps,
            )
        plateau = (
            benefit_tracker.observe(replay.round_index, summary)
            if track_benefit
            else False
        )
        replay.benefit_state = {
            "best": dict(benefit_tracker.best),
            "bad_rounds": benefit_tracker.bad_rounds,
        }
        replay.validation_records.append(
            {
                "optimizer_step": step,
                "aggregation_round": replay.round_index,
                "rollout_steps": rollout_steps,
                "phase": "replay" if replay.aggregation_stop_reason else "aggregation",
                "plateau_checks": benefit_tracker.bad_rounds,
                **summary,
            }
        )
        logging.info(
            "Validation step %d (%d path adjustments): success=%.1f%% safe_success=%.1f%% progress=%.3f "
            "collision=%.1f%% clearance_violation=%.1f%% invalid=%.1f%% plateau_checks=%d/%d",
            step,
            rollout_steps,
            100 * summary["success_rate"],
            100 * summary["safe_success_rate"],
            summary["progress_ratio"],
            100 * summary["collision_rate"],
            100 * summary["clearance_violation_rate"],
            100 * summary["invalid_coverage_rate"],
            benefit_tracker.bad_rounds,
            benefit_tracker.patience,
        )
        payload = {f"eval/model_only/{key}": summary[key] for key in VALIDATION_METRICS}
        payload["eval/model_only/rollout_steps"] = rollout_steps
        if plots:
            payload["eval/model_only/paths"] = [
                wandb.Image(str(path)) for path in plots
            ]
        return plateau, payload

    def log_payload(payload):
        # A shared explicit axis allows round/evaluation logs at the same
        # optimizer step without W&B dropping an already committed step.
        if not primary:
            return
        wandb.log(
            {
                "optimizer_step": int(jax.device_get(train_state.step)),
                "aggregation/round": replay.round_index,
                **payload,
            }
        )

    def train_cached_updates(target, *, post_aggregation):
        nonlocal train_state
        initial = replay.replay_updates if post_aggregation else replay.update_in_round
        update_infos = []
        for update_index in tqdm.tqdm(
            range(initial, target),
            initial=initial,
            total=target,
            desc="cached replay SGD"
            if post_aggregation
            else f"round {replay.round_index + 1}",
        ):
            step_before = int(jax.device_get(train_state.step))
            replay_round, condition_index = deterministic_replay_slot(
                seed=config.seed,
                global_step=step_before,
                num_rounds=replay.labelled_rounds,
                num_condition_batches=replay.num_condition_batches,
            )
            candidate_np, oracle_np = replay.replay_batch(replay_round, condition_index)
            observation, _ = conditions[condition_index]
            with sharding.set_mesh(mesh):
                train_state, update_info = ptrain_field(
                    train_state,
                    observation,
                    _host_batch_to_global(candidate_np, data_sharding),
                    _host_batch_to_global(oracle_np, data_sharding),
                )
            if post_aggregation:
                replay.replay_updates = update_index + 1
            else:
                replay.update_in_round = update_index + 1
            update_infos.append(update_info)
            step = int(jax.device_get(train_state.step))
            if (
                step % config.log_interval == 0
                or update_index == initial
                or update_index + 1 == target
            ):
                train_metrics = _mean_metric_dict(update_infos)
                logging.info(
                    "Train step %d: %s",
                    step,
                    _format_metrics(train_metrics, TRAIN_CONSOLE_KEYS),
                )
                log_payload(learning_metrics(train_metrics))
                if primary:
                    with (monitoring_dir / "training_diagnostics.jsonl").open("a") as handle:
                        handle.write(
                            json.dumps({"optimizer_step": step, **train_metrics}) + "\n"
                        )
                update_infos = []
            if (
                post_aggregation
                and not should_stop()
                and (
                    replay.replay_updates % args.replay_eval_interval == 0
                    or update_index + 1 == target
                )
            ):
                _, payload = evaluate_current_model(track_benefit=False)
                log_payload(payload)
            periodic_snapshot = (
                args.trainable_snapshot_interval > 0
                and step % args.trainable_snapshot_interval == 0
            )
            if (
                step % config.save_interval == 0
                or periodic_snapshot
                or should_stop()
            ):
                save_training_progress(
                    replay,
                    checkpoint_manager,
                    train_state,
                    condition_loader,
                    config,
                    save_trainable=periodic_snapshot or shutdown_requested,
                    wait=shutdown_requested,
                )
            if should_stop():
                # A signal can arrive during an asynchronous periodic save.
                # Commit that save before returning control to the job launcher.
                checkpoint_manager.wait_until_finished()
                logging.warning("Saved interrupted SGD with matching replay phase.")
                return False
        return True

    if not replay.validation_records:
        _, initial_payload = evaluate_current_model(track_benefit=True)
        log_payload(initial_payload)
    if args.replay_only:
        if not replay.labelled_rounds:
            raise ValueError(
                "Cannot train on frozen replay before any oracle labels exist."
            )
        replay.aggregation_stop_reason = (
            replay.aggregation_stop_reason or "manual_replay"
        )
    if replay.converged:
        replay.aggregation_stop_reason = replay.aggregation_stop_reason or "converged"
    if args.continue_aggregation_until_round:
        if replay.replay_updates:
            raise ValueError("Cannot reopen aggregation after frozen-replay SGD.")
        if replay.aggregation_stop_reason not in {"", "validation_plateau", "resource_limit"}:
            raise ValueError(
                f"Cannot reopen aggregation stopped for {replay.aggregation_stop_reason!r}."
            )
        if replay.round_index < args.continue_aggregation_until_round:
            logging.info(
                "Explicit aggregation continuation: round %d -> %d; previous stop=%s. "
                "Validation remains logged, but plateau stopping is suspended until "
                "the round cap; convergence stopping remains active.",
                replay.round_index,
                args.continue_aggregation_until_round,
                replay.aggregation_stop_reason or "none",
            )
            replay.aggregation_stop_reason = ""
    convergence_tracker = ConvergenceTracker(
        _convergence_config_from_args(args),
        stable_rounds=replay.convergence_streak,
    )

    while not replay.aggregation_stop_reason:
        if args.aggregation_rounds and replay.round_index >= args.aggregation_rounds:
            replay.aggregation_stop_reason = "resource_limit"
            break
        round_index = replay.round_index
        logging.info(
            "Aggregation round %d (cap=%s): %d replay rounds, update %d/%d",
            round_index + 1,
            args.aggregation_rounds or "none",
            replay.labelled_rounds,
            replay.update_in_round,
            args.updates_per_round,
        )

        if replay.labelled_rounds == round_index:
            round_directions = []
            oracle_infos = []
            raw_gradient_batches = []
            for condition_index, (_, metric_tensors) in enumerate(conditions):
                current = _host_batch_to_global(
                    replay.current_paths[condition_index], data_sharding
                )
                with sharding.set_mesh(mesh):
                    direction, oracle_info = pquery_oracle(current, metric_tensors)
                round_directions.append(
                    np.asarray(host_numpy(direction), dtype=np.float32)
                )
                raw_gradient_batches.append(
                    np.asarray(
                        host_numpy(oracle_info["raw_oracle_grad_rms_per_path"]),
                        dtype=np.float32,
                    )
                )
                oracle_infos.append(oracle_info)
            oracle_summary = _mean_metric_dict(oracle_infos)
            replay.append_current_round(
                np.stack(round_directions, axis=0),
                np.stack(raw_gradient_batches, axis=0),
                oracle_summary,
            )
            logging.info(
                "Cached round %d oracle labels once: energy=%.4f, grad_rms=%.6f, "
                "acceptance=%.1f%%",
                round_index,
                oracle_summary["obstacle_energy"],
                oracle_summary["raw_oracle_grad_rms"],
                100.0 * oracle_summary["oracle_acceptance_rate"],
            )
            # A step-zero checkpoint is treated as uninitialized by the shared
            # manager, so let the very first round complete one SGD update.
            # Every later labelled frontier can be resumed directly.
            if should_stop() and int(jax.device_get(train_state.step)) > 0:
                save_training_progress(
                    replay,
                    checkpoint_manager,
                    train_state,
                    condition_loader,
                    config,
                    save_trainable=True,
                    wait=True,
                )
                wandb.finish()
                logging.warning(
                    "Saved the labelled frontier before replay training; exiting cleanly."
                )
                return

        if not train_cached_updates(args.updates_per_round, post_aggregation=False):
            wandb.finish()
            return

        # A signal can land after the final loop-body check.  Preserve the
        # fully-trained, pre-rollout phase; resume will skip SGD and redo only
        # the deterministic outer rollout.
        if should_stop():
            save_training_progress(
                replay,
                checkpoint_manager,
                train_state,
                condition_loader,
                config,
                save_trainable=True,
                wait=True,
            )
            wandb.finish()
            logging.warning("Saved before the outer rollout; exiting cleanly.")
            return

        next_path_batches = []
        rollout_infos = []
        for condition_index, (observation, metric_tensors) in enumerate(conditions):
            current = _host_batch_to_global(
                replay.current_paths[condition_index], data_sharding
            )
            oracle_direction = _host_batch_to_global(
                replay.oracle_directions[round_index][condition_index],
                data_sharding,
            )
            with sharding.set_mesh(mesh):
                next_paths, rollout_info = prollout_field(
                    train_state,
                    observation,
                    current,
                    oracle_direction,
                    metric_tensors,
                )
            next_path_batches.append(
                np.asarray(host_numpy(next_paths), dtype=np.float32)
            )
            rollout_infos.append(rollout_info)
            if should_stop():
                # Partial rollout results are deliberately discarded.  The
                # replay still describes the pre-rollout phase exactly, so a
                # resume deterministically recomputes the whole frontier.
                save_training_progress(
                    replay,
                    checkpoint_manager,
                    train_state,
                    condition_loader,
                    config,
                    save_trainable=True,
                    wait=True,
                )
                wandb.finish()
                logging.warning(
                    "Saved during the outer rollout; partial rollout discarded for resume."
                )
                return

        round_metrics = _round_metrics(
            replay.oracle_records[round_index],
            replay.oracle_gradient_rms[round_index],
            rollout_infos,
        )
        converged, convergence_status = convergence_tracker.observe(
            round_index + 1,
            round_metrics,
        )
        round_record: dict[str, float | int | str] = {
            "round": round_index + 1,
            "source_depth": round_index,
            "target_depth": round_index + 1,
            "optimizer_step": int(jax.device_get(train_state.step)),
            "status": convergence_status,
            **round_metrics,
        }
        replay.finish_round(
            np.stack(next_path_batches, axis=0),
            round_record,
            convergence_streak=convergence_tracker.stable_rounds,
        )
        global_step = int(jax.device_get(train_state.step))
        logging.info(
            "Finished aggregation round %d: energy %.4f -> %.4f, path change "
            "median=%.4fm p95=%.4fm, grad_rms=%.6f, collision=%.2f%%, "
            "clearance_violation=%.2f%%, status=%s",
            round_index,
            round_metrics["energy_before"],
            round_metrics["energy_after"],
            round_metrics["path_change_median"],
            round_metrics["path_change_p95"],
            round_metrics["oracle_grad_rms"],
            100.0 * round_metrics["collision_rate"],
            100.0 * round_metrics["clearance_violation_rate"],
            convergence_status,
        )
        plateau, validation_payload = evaluate_current_model(track_benefit=True)
        if converged:
            replay.aggregation_stop_reason = "converged"
        elif plateau and not args.continue_aggregation_until_round:
            replay.aggregation_stop_reason = "validation_plateau"
        elif args.aggregation_rounds and replay.round_index >= args.aggregation_rounds:
            replay.aggregation_stop_reason = "resource_limit"
        if replay.aggregation_stop_reason:
            replay.round_records[-1]["status"] = replay.aggregation_stop_reason
        wandb_payload = (
            round_wandb_metrics(round_metrics, energy_config) | validation_payload
        )
        wandb_payload["aggregation/status"] = (
            replay.aggregation_stop_reason or convergence_status
        )
        if (
            args.aggregation_plot_interval > 0
            and replay.round_index % args.aggregation_plot_interval == 0
        ):
            plot_metrics = jax.tree.map(host_numpy, conditions[0][1])
            if primary:
                plot = aggregation_path_figure(
                    replay,
                    plot_metrics,
                    energy_config,
                    particles=args.train_particles,
                    max_rounds_to_show=args.train_plot_num_images,
                )
                if plot is not None:
                    wandb_payload["aggregation/path_history"] = plot
        log_payload(wandb_payload)

        save_training_progress(
            replay,
            checkpoint_manager,
            train_state,
            condition_loader,
            config,
            save_trainable=(
                args.trainable_snapshot_interval > 0
                and global_step % args.trainable_snapshot_interval == 0
            )
            or bool(replay.aggregation_stop_reason),
            wait=True,
        )
        if should_stop():
            wandb.finish()
            logging.warning("Saved the completed round; exiting cleanly for resume.")
            return
    logging.info(
        "Aggregation stopped after %d rounds: %s. Cached-replay SGD: %d/%d updates.",
        replay.round_index,
        replay.aggregation_stop_reason,
        replay.replay_updates,
        args.post_aggregation_updates,
    )
    # A stop in collection does not freeze the optimizer. The target is an
    # absolute post-aggregation update count, so interrupted jobs resume it and
    # increasing it extends training without relabelling paths.
    save_training_progress(
        replay,
        checkpoint_manager,
        train_state,
        condition_loader,
        config,
        save_trainable=False,
        wait=True,
    )
    if not should_stop():
        train_cached_updates(args.post_aggregation_updates, post_aggregation=True)
    save_training_progress(
        replay,
        checkpoint_manager,
        train_state,
        condition_loader,
        config,
        save_trainable=True,
        wait=True,
    )
    wandb.finish()
    logging.info(
        "Saved training at step %d (%d aggregation rounds, %d frozen-replay updates).",
        int(jax.device_get(train_state.step)),
        replay.round_index,
        replay.replay_updates,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ground-truth-free pi0.5 energy-oracle dataset aggregation."
    )
    parser.add_argument(
        "--config-name", default="astar_pi05_energy_aggregation"
    )
    parser.add_argument("--project-name", default="astar")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--pretrained-params", default=PI05_BASE_PARAMS)
    parser.add_argument("--action-dim", type=int, default=2, choices=(2,))
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument(
        "--discrete-state-input",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-goal-waypoint-adapter",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--goal-waypoint-dim", type=int, default=2, choices=(2,))
    parser.add_argument("--max-goal-waypoints", type=int, default=1, choices=(1,))
    parser.add_argument(
        "--train-scope",
        choices=("adapter", "action_stack", "all"),
        default="adapter",
    )
    parser.add_argument("--random-init-action-stack", action="store_true")

    aggregation = parser.add_argument_group("dataset aggregation")
    aggregation.add_argument("--relabel-max-step-from-config", default=None,
                             help="Explicitly refresh retained replay after changing the segment cap on resume.")
    aggregation.add_argument("--relabel-esdf-cutoff-from-config", default=None,
                             help="Refresh retained replay after enabling, changing, or disabling the ESDF learning cutoff.")
    aggregation.add_argument("--migrate-goal-conditioning-from-config", default=None,
                             help="Reuse every retained path while rebuilding conditions and oracle labels for a new goal-input mix.")
    aggregation.add_argument("--require-all-goal-modalities", action="store_true",
                             help="Fail startup unless the cached pool contains text, image, waypoint, object, and sampled goals.")
    aggregation.add_argument(
        "--aggregation-rounds",
        type=int,
        default=0,
        help="Optional resource cap; 0 aggregates until benefit stops.",
    )
    aggregation.add_argument("--updates-per-round", type=int, default=3000)
    aggregation.add_argument(
        "--continue-aggregation-until-round",
        type=int,
        default=0,
        help="Force a fresh or resumed run toward an absolute round cap, bypassing "
        "validation plateau stopping while preserving convergence checks.",
    )
    aggregation.add_argument(
        "--post-aggregation-updates",
        type=int,
        default=0,
        help="Total cached-replay SGD updates after collection stops; increase on resume.",
    )
    aggregation.add_argument(
        "--replay-only",
        action="store_true",
        help="On resume, freeze the existing replay and perform only post-aggregation SGD.",
    )
    aggregation.add_argument("--benefit-min-rounds", type=int, default=6)
    aggregation.add_argument("--benefit-patience", type=int, default=4)
    aggregation.add_argument("--benefit-rate-delta", type=float, default=0.01)
    aggregation.add_argument("--benefit-progress-delta", type=float, default=0.02)
    aggregation.add_argument(
        "--aggregation-batches",
        type=int,
        default=16,
        help="Number of immutable condition batches in the replay pool.",
    )
    aggregation.add_argument(
        "--field-time",
        type=float,
        default=1.0,
        help="Fixed pi0.5 time token; aggregation depth is not flow time.",
    )
    aggregation.add_argument(
        "--aggregation-step-size", type=float, default=1.0,
        help="Adjustment coefficient at the last future waypoint (end of the linear ramp).",
    )
    aggregation.add_argument(
        "--aggregation-step-size-start", type=float, default=0.1,
        help="Adjustment coefficient at the first future waypoint; origin stays fixed.",
    )
    aggregation.add_argument("--oracle-gradient-floor", type=float, default=1.0e-4)
    aggregation.add_argument("--max-field-direction-rms", type=float, default=1.0)
    aggregation.add_argument("--rollout-backtracks", type=int, default=6)
    aggregation.add_argument("--backtrack-factor", type=float, default=0.5)
    aggregation.add_argument("--energy-increase-tolerance", type=float, default=1.0e-6)
    aggregation.add_argument(
        "--diversity-direction-weight",
        type=float,
        default=0.0,
        help="Keep 0 for a strictly state-local oracle; nonzero labels depend on particle peers.",
    )

    convergence = parser.add_argument_group("convergence")
    convergence.add_argument("--convergence-min-rounds", type=int, default=3)
    convergence.add_argument("--convergence-patience", type=int, default=2)
    convergence.add_argument(
        "--convergence-median-path-change-m", type=float, default=0.01
    )
    convergence.add_argument(
        "--convergence-p95-path-change-m", type=float, default=0.03
    )
    convergence.add_argument(
        "--convergence-relative-energy-change", type=float, default=1.0e-3
    )
    convergence.add_argument(
        "--convergence-oracle-grad-rms", type=float, default=1.0e-4
    )
    convergence.add_argument(
        "--convergence-max-clearance-violation-rate",
        "--convergence-max-collision-rate",
        dest="convergence_max_clearance_violation_rate",
        type=float,
        default=0.05,
        help=(
            "Maximum fraction of paths below the requested footprint-clearance "
            "margin. The old collision-rate spelling is a compatibility alias."
        ),
    )
    convergence.add_argument(
        "--convergence-min-progress-ratio", type=float, default=0.9
    )

    prior = parser.add_argument_group("structured initial prior")
    prior.add_argument("--prior-dt-s", type=float, default=0.2)
    prior.add_argument("--prior-min-v-mps", type=float, default=0.8)
    prior.add_argument("--prior-max-v-mps", type=float, default=2.0)
    prior.add_argument("--prior-max-omega-radps", type=float, default=0.5)
    prior.add_argument(
        "--prior-length-scale",
        type=float,
        default=0.25,
        help=(
            "Scale prior translation without changing its sampled turn sequence; "
            "0.25 produces quarter-length priors."
        ),
    )
    prior.add_argument(
        "--prior-forward-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    prior.add_argument("--prior-goal-heading-fraction", type=float, default=1.0)
    prior.add_argument(
        "--prior-goal-heading-limit-rad", type=float, default=1.0471975512
    )

    energy = parser.add_argument_group("trajectory energy")
    energy.add_argument(
        "--collision-weight",
        type=float,
        default=0.0,
        help="Deprecated compatibility option; collision is diagnostic only.",
    )
    energy.add_argument("--clearance-weight", type=float, default=1.0)
    energy.add_argument("--goal-weight", type=float, default=5.0)
    energy.add_argument("--progress-weight", type=float, default=10.0)
    energy.add_argument("--early-heading-weight", type=float, default=0.25)
    energy.add_argument("--smoothness-weight", type=float, default=0.5)
    energy.add_argument("--obstacle-safety-margin-m", type=float, default=0.25)
    energy.add_argument(
        "--clearance-cap-m",
        type=float,
        default=0.25,
        help=(
            "Deprecated saved-config field ignored by aggregation algorithm v7; "
            "use --obstacle-safety-margin-m."
        ),
    )
    energy.add_argument("--esdf-learning-cutoff-m", type=float, default=None,
                        help="Optional cutoff in footprint-clearance meters after subtracting robot radius; must be at least the safety margin.")
    energy.add_argument("--esdf-ramp-start-m", type=float, default=0.5)
    energy.add_argument("--esdf-min-x-m", type=float, default=2.0)
    energy.add_argument("--min-step-scale-m", type=float, default=0.1)
    energy.add_argument("--segment-samples", type=int, default=6)
    energy.add_argument("--max-step-length-m", type=float, default=0.4)
    energy.add_argument(
        "--path-detour-factor",
        type=float,
        default=1.25,
        help=(
            "Slack over straight-line goal distance used by the per-path segment "
            "cap; max-step-length-m remains the physical hard cap."
        ),
    )
    energy.add_argument("--required-progress-fraction", type=float, default=0.65)
    energy.add_argument("--energy-temperature", type=float, default=1.0)
    energy.add_argument("--train-particles", type=int, default=2)

    optimizer = parser.add_argument_group("optimizer")
    optimizer.add_argument("--batch-size", type=int, default=8)
    optimizer.add_argument("--warmup-steps", type=int, default=1000)
    optimizer.add_argument("--peak-lr", type=float, default=1.0e-4)
    optimizer.add_argument("--decay-steps", type=int, default=30_000)
    optimizer.add_argument("--decay-lr", type=float, default=1.0e-5)
    optimizer.add_argument("--clip-gradient-norm", type=float, default=1.0)
    optimizer.add_argument("--ema-decay", type=float, default=0.999)
    optimizer.add_argument("--log-interval", type=int, default=100)

    data = parser.add_argument_group("data")
    data.add_argument("--data-root", default=None)
    data.add_argument("--manifest-path", default=None)
    data.add_argument("--train-split-ids-path", default=None)
    data.add_argument(
        "--eval-split-ids-path",
        default=None,
        help="Disjoint validation clips; defaults to data-root/splits/eval_clip_ids.txt.",
    )
    data.add_argument(
        "--sampled-goal-fraction", type=float, default=DEFAULT_SAMPLED_GOAL_FRACTION
    )
    data.add_argument(
        "--object-text-goal-prob", type=float, default=DEFAULT_OBJECT_TEXT_GOAL_PROB
    )
    data.add_argument(
        "--object-image-goal-prob", type=float, default=DEFAULT_OBJECT_IMAGE_GOAL_PROB
    )
    data.add_argument(
        "--object-waypoint-goal-prob",
        type=float,
        default=DEFAULT_OBJECT_WAYPOINT_GOAL_PROB,
    )

    validation = parser.add_argument_group("fixed model-only validation")
    validation.add_argument("--eval-batches", type=int, default=4)
    validation.add_argument("--eval-seed", type=int, default=123)
    validation.add_argument("--eval-progress-threshold", type=float, default=0.9)
    validation.add_argument("--eval-safety-spacing-m", type=float, default=0.02)
    validation.add_argument("--eval-plot-count", type=int, default=4)
    validation.add_argument("--eval-plot-interval", type=int, default=1)
    validation.add_argument("--replay-eval-interval", type=int, default=2500)

    checkpoint = parser.add_argument_group("checkpointing and logging")
    checkpoint.add_argument("--save-interval", type=int, default=5000)
    checkpoint.add_argument(
        "--checkpoint-mode",
        choices=("full",),
        default="full",
    )
    checkpoint.add_argument("--trainable-snapshot-interval", type=int, default=5000)
    checkpoint.add_argument("--assets-base-dir", default="./assets")
    checkpoint.add_argument("--checkpoint-base-dir", default="./checkpoints")
    checkpoint.add_argument("--fsdp-devices", type=int, default=1)
    checkpoint.add_argument("--wandb-entity", default="yohanab")
    checkpoint.add_argument(
        "--wandb-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    checkpoint.add_argument("--aggregation-plot-interval", type=int, default=1)
    checkpoint.add_argument(
        "--train-plot-num-images",
        type=int,
        default=10,
        help="Maximum aggregation rounds shown in the path-history plot.",
    )
    checkpoint.add_argument("--overwrite", action="store_true")
    checkpoint.add_argument("--resume", action="store_true")
    checkpoint.add_argument("--init-only", action="store_true")
    args = parser.parse_args()
    if args.continue_aggregation_until_round:
        args.aggregation_rounds = args.continue_aggregation_until_round
    # Compatibility fields required by shared ASTAR configuration/data classes;
    # none changes this path-free aggregation algorithm.
    args.path_stride = 10
    args.max_increment_correction_m = 0.08
    args.keep_period = 0
    args.strict_esdf_coverage = True
    # create_train_config expects this field; the outer-loop sizes are the sole source of truth.
    args.num_train_steps = max(
        args.decay_steps, args.aggregation_rounds * args.updates_per_round
    )
    return args


def cli() -> None:
    main(parse_args())


if __name__ == "__main__":
    cli()
