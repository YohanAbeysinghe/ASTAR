#!/usr/bin/env python3
"""Evaluate a pi0.5 energy-aggregation checkpoint on held-out ASTAR clips.

The evaluator deliberately separates learned-policy performance from the
oracle-assisted rollout used by the aggregation trainer:

* ``prior`` scores the deterministic goal-biased unicycle initialization;
* ``model_only`` applies the learned field and kinematic projection only;
* ``guarded`` additionally uses the analytic energy for backtracking;
* ``oracle`` follows the analytic negative-energy gradient as a privileged
  local reference.

No optimizer update is constructed or applied.  Full checkpoints are opened
through parameter-only Orbax restores; optimizer state is never materialized.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
import csv
import dataclasses
import functools
import gc
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import subprocess
import time
from typing import Any
from typing import Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi.shared import array_typing as at
import openpi.training.sharding as sharding
import orbax.checkpoint as ocp

from astar.dataloader import OPENPI_IMAGE_SIZE
from astar.dataloader import GoalModalityConfig
from astar.path_sampler import DynamicWaypointPriorConfig
from astar.path_sampler import maybe_unnormalize_actions
from astar.training.aggregation import OracleOnlyNavigationDataLoader
from astar.training.aggregation import OracleOnlyNavigationDataset
from astar.training.aggregation import _cap_direction_rms
from astar.training.aggregation import _rms_per_path
from astar.training.aggregation import backtracking_projected_step
from astar.training.aggregation import initialize_prior_paths
from astar.training.aggregation import predict_descent_field
from astar.training.aggregation import project_bounded_actions
from astar.training.aggregation import query_energy_oracle
from astar.training.aggregation import sanitize_metric_tensors
from astar.training.monitoring import esdf_display_scale
from astar.training.steps import waypoint_step_sizes
from astar.training_common import create_train_config
from astar.training_common import init_logging
from astar.training_common import init_train_state
from astar.waypoint_energy import ObstacleEnergyConfig
from astar.waypoint_energy import compute_obstacle_energy
from astar.waypoint_energy import decode_bounded_waypoint_path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVAL_SPLIT = REPOSITORY_ROOT / "configs" / "splits" / "eval_clip_ids.txt"
DEFAULT_RESULTS_ROOT = REPOSITORY_ROOT / "outputs" / "evaluation"

Mode = Literal["prior", "model_only", "guarded", "oracle"]
ParameterSource = Literal["ema", "raw"]

MODEL_MODES = frozenset({"model_only", "guarded"})
BASELINE_MODES = frozenset({"prior", "oracle"})
VALID_MODES = tuple(sorted(MODEL_MODES | BASELINE_MODES))

SCORE_KEYS = (
    "obstacle_energy",
    "collision_energy",
    "clearance_energy",
    "goal_energy",
    "progress_energy",
    "early_heading_energy",
    "smoothness_energy",
    "weighted_collision_energy",
    "weighted_clearance_energy",
    "weighted_goal_energy",
    "weighted_progress_energy",
    "weighted_early_heading_energy",
    "weighted_smoothness_energy",
    "collision_rate",
    "unsafe_rate",
    "invalid_esdf_rate",
    "esdf_valid_rate",
    "all_invalid_traj_rate",
    "esdf_active_rate",
    "esdf_weight_mean",
    "action_finite_rate",
    "outside_esdf_distance_m",
    "pred_xy_abs_max_m",
    "min_esdf_m",
    "min_clearance_m",
    "safe_radius_m",
    "goal_distance_m",
    "clamped_goal_distance_m",
    "required_progress_m",
    "achieved_progress_m",
    "progress_shortfall_m",
    "achieved_required_progress_ratio",
    "path_length_m",
    "endpoint_radius_m",
    "mean_segment_length_m",
    "max_segment_length_m",
    "step_violation_rate",
    "goal_clamp_distance_m",
    "goal_was_clamped_rate",
    "final_goal_error_m",
    "goal_inside_esdf_rate",
)

PRIMARY_METRICS = frozenset(
    {
        "success",
        "safe_success",
        "progress_success",
        "rollout_valid",
        "collision_rate",
        "unsafe_rate",
        "dense_collision_rate",
        "dense_unsafe_rate",
        "obstacle_energy",
        "achieved_required_progress_ratio",
        "achieved_progress_m",
        "final_goal_error_m",
        "min_clearance_m",
        "field_oracle_cosine",
        "energy_nonincrease",
    }
)

IDENTITY_KEYS = frozenset(
    {
        "checkpoint",
        "parameter_source",
        "mode",
        "depth",
        "dataset_index",
        "example_id",
        "digest_hex",
        "clip_id",
        "sample_id",
        "goal_kind",
        "goal_index",
    }
)


@dataclasses.dataclass(frozen=True)
class ExampleIdentity:
    dataset_index: int
    example_id: str
    digest_hex: str
    clip_id: str
    sample_id: str
    goal_kind: str
    goal_index: int
    goal_x_m: float
    goal_y_m: float


@dataclasses.dataclass(frozen=True)
class EvaluationKernels:
    initialize_prior: Any
    query_oracle: Any
    score_paths: Any
    score_dense_safety: Any
    oracle_step: Any
    decode_paths: Any


@dataclasses.dataclass(frozen=True)
class ModelKernels:
    model_only_step: Any
    guarded_step: Any
    alignment: Any


class JsonlWriter:
    """Flush each result row so a preempted evaluation remains inspectable."""

    def __init__(self, path: Path):
        self.path = path
        self._handle = path.open("w", encoding="utf-8")

    def write(self, record: Mapping[str, Any]) -> None:
        self._handle.write(
            json.dumps(_json_safe(record), sort_keys=True, allow_nan=False) + "\n"
        )
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _json_safe(value: Any) -> Any:
    """Convert non-finite scalars to JSON null while preserving diagnostics."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _strict_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repository: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _read_clip_ids(path: Path) -> tuple[str, ...]:
    clip_ids = tuple(
        line
        for raw_line in path.read_text(encoding="utf-8").splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    )
    if not clip_ids:
        raise ValueError(f"Clip split is empty: {path}")
    if len(clip_ids) != len(set(clip_ids)):
        raise ValueError(f"Clip split contains duplicate IDs: {path}")
    return clip_ids


def validate_held_out_split(
    values: Mapping[str, Any],
    eval_split_path: Path,
) -> tuple[Path, tuple[str, ...], tuple[str, ...]]:
    """Resolve the recorded training split and prove clip-level disjointness."""
    configured_path = values.get("train_split_ids_path")
    if not configured_path:
        raise ValueError(
            "Training config has no train_split_ids_path; held-out status cannot be verified"
        )
    train_split_path = Path(str(configured_path)).expanduser()
    if not train_split_path.is_absolute():
        train_split_path = REPOSITORY_ROOT / train_split_path
    train_split_path = train_split_path.resolve()
    if not train_split_path.is_file():
        raise FileNotFoundError(f"Recorded training split not found: {train_split_path}")
    if not eval_split_path.is_file():
        raise FileNotFoundError(f"Evaluation split not found: {eval_split_path}")

    train_ids = _read_clip_ids(train_split_path)
    eval_ids = _read_clip_ids(eval_split_path)
    overlap = sorted(set(train_ids).intersection(eval_ids))
    if overlap:
        raise ValueError(
            "Evaluation split overlaps the recorded training split: "
            + ", ".join(overlap)
        )
    return train_split_path, train_ids, eval_ids


def load_training_config(path: Path) -> dict[str, Any]:
    """Read ASTAR's flat JSON or W&B's wrapped YAML configuration."""
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            raw = json.load(handle)
        else:
            try:
                import yaml
            except ImportError as error:  # pragma: no cover - environment setup failure
                raise ImportError("PyYAML is required to read a YAML training config.") from error
            raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Training config is not a mapping: {path}")

    values: dict[str, Any] = {}
    for key, entry in raw.items():
        if key == "_wandb":
            continue
        if isinstance(entry, Mapping) and "value" in entry:
            values[str(key)] = entry["value"]
        else:
            values[str(key)] = entry
    return values


# Backwards-compatible name for callers evaluating historical W&B runs.
load_wandb_config = load_training_config


def _required(values: Mapping[str, Any], *names: str) -> None:
    missing = sorted(name for name in names if name not in values)
    if missing:
        raise ValueError(f"Training config is missing required values: {missing}")


def build_training_namespace(
    values: Mapping[str, Any],
    *,
    batch_size: int,
    fsdp_devices: int,
) -> argparse.Namespace:
    """Construct only the shared fields needed to recreate model/state shapes."""
    required = (
        "config_name",
        "project_name",
        "exp_name",
        "action_dim",
        "action_horizon",
        "discrete_state_input",
        "use_goal_waypoint_adapter",
        "goal_waypoint_dim",
        "max_goal_waypoints",
        "train_scope",
        "warmup_steps",
        "peak_lr",
        "decay_steps",
        "decay_lr",
        "clip_gradient_norm",
        "ema_decay",
        "num_train_steps",
        "log_interval",
        "save_interval",
    )
    _required(values, *required)
    merged = dict(values)
    merged.update(
        {
            "config_name": str(values["config_name"]),
            "project_name": str(values["project_name"]),
            "exp_name": str(values["exp_name"]),
            "action_dim": int(values["action_dim"]),
            "action_horizon": int(values["action_horizon"]),
            "discrete_state_input": _strict_bool(
                values["discrete_state_input"], name="discrete_state_input"
            ),
            "use_goal_waypoint_adapter": _strict_bool(
                values["use_goal_waypoint_adapter"],
                name="use_goal_waypoint_adapter",
            ),
            "goal_waypoint_dim": int(values["goal_waypoint_dim"]),
            "max_goal_waypoints": int(values["max_goal_waypoints"]),
            "train_scope": str(values["train_scope"]),
            "warmup_steps": int(values["warmup_steps"]),
            "peak_lr": float(values["peak_lr"]),
            "decay_steps": int(values["decay_steps"]),
            "decay_lr": float(values["decay_lr"]),
            "clip_gradient_norm": float(values["clip_gradient_norm"]),
            "ema_decay": (
                None if values["ema_decay"] is None else float(values["ema_decay"])
            ),
            "num_train_steps": int(values["num_train_steps"]),
            "log_interval": int(values["log_interval"]),
            "save_interval": int(values["save_interval"]),
            "batch_size": batch_size,
            "fsdp_devices": fsdp_devices,
            "pretrained_params": str(REPOSITORY_ROOT / "checkpoints" / "pi05_base" / "params"),
            "checkpoint_base_dir": str(REPOSITORY_ROOT / "checkpoints"),
            "assets_base_dir": str(REPOSITORY_ROOT / "assets"),
            "overwrite": False,
            "resume": True,
            "wandb_enabled": False,
            "keep_period": 0,
        }
    )
    return argparse.Namespace(**merged)


def build_energy_config(values: Mapping[str, Any]) -> ObstacleEnergyConfig:
    required = (
        "prior_dt_s",
        "prior_min_v_mps",
        "prior_max_v_mps",
        "prior_max_omega_radps",
        "prior_forward_only",
        "collision_weight",
        "clearance_weight",
        "goal_weight",
        "progress_weight",
        "early_heading_weight",
        "smoothness_weight",
        "obstacle_safety_margin_m",
        "clearance_cap_m",
        "esdf_ramp_start_m",
        "esdf_min_x_m",
        "min_step_scale_m",
        "segment_samples",
        "max_step_length_m",
        "max_increment_correction_m",
        "prior_goal_heading_fraction",
        "prior_goal_heading_limit_rad",
        "required_progress_fraction",
        "aggregation_step_size",
        "energy_temperature",
        "train_particles",
        "diversity_direction_weight",
    )
    _required(values, *required)
    prior = DynamicWaypointPriorConfig(
        dt_s=float(values["prior_dt_s"]),
        min_v_mps=float(values["prior_min_v_mps"]),
        max_v_mps=float(values["prior_max_v_mps"]),
        max_omega_radps=float(values["prior_max_omega_radps"]),
        # Historical checkpoints predate this field and retain their original
        # unscaled prior behavior.
        length_scale=float(values.get("prior_length_scale", 1.0)),
        forward_only=_strict_bool(
            values["prior_forward_only"], name="prior_forward_only"
        ),
    )
    return ObstacleEnergyConfig(
        sample_steps=1,
        prior=prior,
        collision_weight=float(values["collision_weight"]),
        clearance_weight=float(values["clearance_weight"]),
        goal_weight=float(values["goal_weight"]),
        progress_weight=float(values["progress_weight"]),
        early_heading_weight=float(values["early_heading_weight"]),
        smoothness_weight=float(values["smoothness_weight"]),
        safety_margin_m=float(values["obstacle_safety_margin_m"]),
        clearance_cap_m=float(values["clearance_cap_m"]),
        esdf_learning_cutoff_m=(
            None if values.get("esdf_learning_cutoff_m") is None
            else float(values["esdf_learning_cutoff_m"])
        ),
        esdf_ramp_start_m=float(values["esdf_ramp_start_m"]),
        esdf_min_x_m=float(values["esdf_min_x_m"]),
        strict_esdf_coverage=_strict_bool(
            values.get("strict_esdf_coverage", False), name="strict_esdf_coverage"
        ),
        min_step_scale_m=float(values["min_step_scale_m"]),
        segment_samples=int(values["segment_samples"]),
        max_step_length_m=float(values["max_step_length_m"]),
        max_increment_correction_m=float(values["max_increment_correction_m"]),
        prior_goal_heading_fraction=float(values["prior_goal_heading_fraction"]),
        prior_goal_heading_limit_rad=float(values["prior_goal_heading_limit_rad"]),
        required_progress_fraction=float(values["required_progress_fraction"]),
        oracle_step_size=float(values["aggregation_step_size"]),
        energy_temperature=float(values["energy_temperature"]),
        train_particles=int(values["train_particles"]),
        diversity_direction_weight=float(values["diversity_direction_weight"]),
    )


def _example_digest(seed: int, namespace: str, example_id: str) -> bytes:
    payload = f"{seed}:{namespace}:{example_id}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def group_dataset_examples_by_clip(
    dataset: OracleOnlyNavigationDataset,
    *,
    goal_kind: str,
) -> dict[str, list[ExampleIdentity]]:
    """Index logical examples by clip without loading images or ESDF grids."""
    groups: dict[str, list[ExampleIdentity]] = defaultdict(list)
    last_offset = -1
    record: Mapping[str, Any] = {}
    with dataset.samples_path.open("rb") as handle:
        for dataset_index, reference in enumerate(dataset.goal_references):
            if goal_kind != "all" and reference.goal_kind != goal_kind:
                continue
            if reference.byte_offset != last_offset:
                handle.seek(reference.byte_offset)
                record = json.loads(handle.readline())
                last_offset = reference.byte_offset
            clip_id = str(record.get("clip_id", ""))
            if not clip_id:
                raise ValueError(f"Dataset record at offset {reference.byte_offset} has no clip_id")
            sample_id = str(record.get("sample_id", ""))
            goals_key = (
                "object_goals" if reference.goal_kind == "object" else "sampled_goals"
            )
            goal = (record.get(goals_key) or [])[reference.goal_index]
            goal_xy = np.asarray(goal["goal_xy_m"][:2], dtype=np.float32)
            example_id = json.dumps(
                [clip_id, sample_id, reference.goal_kind, int(reference.goal_index)],
                separators=(",", ":"),
            )
            groups[clip_id].append(
                ExampleIdentity(
                    dataset_index=dataset_index,
                    example_id=example_id,
                    digest_hex=hashlib.sha256(example_id.encode("utf-8")).hexdigest(),
                    clip_id=clip_id,
                    sample_id=sample_id,
                    goal_kind=reference.goal_kind,
                    goal_index=int(reference.goal_index),
                    goal_x_m=float(goal_xy[0]),
                    goal_y_m=float(goal_xy[1]),
                )
            )
    if not groups:
        raise ValueError(f"No {goal_kind!r} evaluation examples were found")
    return dict(groups)


def select_clip_balanced_examples(
    groups: Mapping[str, Sequence[ExampleIdentity]],
    *,
    examples_per_clip: int,
    seed: int,
    max_examples: int | None = None,
) -> list[ExampleIdentity]:
    """Select one goal per distinct frame using stable logical-ID hashes."""
    if examples_per_clip <= 0:
        raise ValueError("examples_per_clip must be positive")
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive when provided")

    selected_by_clip: dict[str, list[ExampleIdentity]] = {}
    for clip_id in sorted(groups):
        # Multiple goals from one video frame are highly correlated.  Pick one
        # goal deterministically per frame, then rank frames by a separate hash.
        per_sample: dict[str, list[ExampleIdentity]] = defaultdict(list)
        for candidate in groups[clip_id]:
            per_sample[candidate.sample_id].append(candidate)
        representatives = [
            min(
                candidates,
                key=lambda item: _example_digest(seed, "goal", item.example_id),
            )
            for candidates in per_sample.values()
        ]
        representatives.sort(
            key=lambda item: _example_digest(seed, "frame", item.example_id)
        )
        selected_by_clip[clip_id] = representatives[:examples_per_clip]

    # Apply a global cap by strata: every clip contributes its first-ranked
    # frame before any clip contributes its second. This preserves the maximum
    # possible clip coverage for small smoke-test caps.
    selected: list[ExampleIdentity] = []
    for rank in range(examples_per_clip):
        rank_items = [
            items[rank] for items in selected_by_clip.values() if rank < len(items)
        ]
        rank_items.sort(key=lambda item: _example_digest(seed, "cap", item.example_id))
        selected.extend(rank_items)
        if max_examples is not None and len(selected) >= max_examples:
            selected = selected[:max_examples]
            break
    selected.sort(key=lambda item: _example_digest(seed, "batch", item.example_id))
    example_ids = [item.example_id for item in selected]
    frame_ids = [(item.clip_id, item.sample_id) for item in selected]
    if len(example_ids) != len(set(example_ids)):
        raise AssertionError("Evaluation sampling produced duplicate logical examples")
    if len(frame_ids) != len(set(frame_ids)):
        raise AssertionError("Evaluation sampling produced multiple goals from one frame")
    return selected


def _goal_modality_config(mode: str, values: Mapping[str, Any]) -> GoalModalityConfig:
    if mode == "training":
        return GoalModalityConfig(
            object_text_prob=float(values["object_text_goal_prob"]),
            object_image_prob=float(values["object_image_goal_prob"]),
            object_waypoint_prob=float(values["object_waypoint_goal_prob"]),
        )
    enabled = {
        "text": (1.0, 0.0, 0.0),
        "image": (0.0, 1.0, 0.0),
        "waypoint": (0.0, 0.0, 1.0),
        "all": (1.0, 1.0, 1.0),
    }[mode]
    return GoalModalityConfig(
        object_text_prob=enabled[0],
        object_image_prob=enabled[1],
        object_waypoint_prob=enabled[2],
    )


def build_dataset(
    args: argparse.Namespace,
    values: Mapping[str, Any],
    model_config: Any,
) -> OracleOnlyNavigationDataset:
    return OracleOnlyNavigationDataset(
        data_root=args.data_root,
        manifest_path=args.manifest_path,
        split="eval",
        split_ids_path=args.eval_split_ids,
        include_object_goals=args.goal_kind in {"all", "object"},
        include_sampled_goals=args.goal_kind in {"all", "sampled"},
        action_horizon=int(model_config.model.action_horizon),
        path_stride=int(values["path_stride"]),
        image_size=OPENPI_IMAGE_SIZE,
        goal_image_size=OPENPI_IMAGE_SIZE,
        load_esdf=True,
    )


def build_batcher(
    args: argparse.Namespace,
    values: Mapping[str, Any],
    model_config: Any,
    data_sharding: jax.sharding.NamedSharding,
    dataset: OracleOnlyNavigationDataset,
) -> OracleOnlyNavigationDataLoader:
    return OracleOnlyNavigationDataLoader(
        dataset,
        batch_size=args.batch_size,
        sampled_goal_fraction=None,
        goal_modality_config=_goal_modality_config(args.object_goal_modality, values),
        output_format="openpi",
        max_token_len=int(model_config.model.max_token_len),
        discrete_state_input=bool(model_config.model.discrete_state_input),
        shuffle=False,
        seed=args.seed,
        sharding=data_sharding,
        num_batches=None,
        return_metric_tensors=True,
        make_jax_arrays=True,
    )


def iter_evaluation_batches(
    dataset: OracleOnlyNavigationDataset,
    batcher: OracleOnlyNavigationDataLoader,
    selected_examples: Sequence[ExampleIdentity],
    *,
    batch_size: int,
    seed: int,
) -> Iterable[tuple[int, Any, dict[str, at.Array], list[ExampleIdentity], int]]:
    """Yield explicitly indexed batches and pad only the final device batch."""
    rng = np.random.default_rng(seed)
    for batch_index, start in enumerate(range(0, len(selected_examples), batch_size)):
        identities = list(selected_examples[start : start + batch_size])
        samples = [dataset[item.dataset_index] for item in identities]
        valid_count = len(samples)
        if valid_count < batch_size:
            samples.extend([samples[-1]] * (batch_size - valid_count))

        for identity, sample in zip(identities, samples[:valid_count], strict=True):
            actual = (
                str(sample["clip_id"]),
                str(sample["sample_id"]),
                str(sample["goal_kind"]),
                int(np.asarray(sample["goal_index"])),
            )
            expected = (
                identity.clip_id,
                identity.sample_id,
                identity.goal_kind,
                identity.goal_index,
            )
            if actual != expected:
                raise RuntimeError(
                    f"Dataset changed after selection: expected {expected}, loaded {actual}"
                )
        observation, placeholder_actions, metric_tensors = batcher._openpi_batch(samples, rng)
        placeholders = np.asarray(jax.device_get(placeholder_actions))
        if np.any(placeholders != 0.0) or not np.all(np.isfinite(placeholders)):
            raise AssertionError("Evaluation loader exposed nonzero trajectory placeholders")
        metric_tensors = sanitize_metric_tensors(metric_tensors)
        yield batch_index, observation, metric_tensors, identities, valid_count


def _cosine_per_path(predicted: at.Array, target: at.Array) -> at.Array:
    predicted_flat = predicted[:, 1:, :].reshape((predicted.shape[0], -1))
    target_flat = target[:, 1:, :].reshape((target.shape[0], -1))
    numerator = jnp.sum(predicted_flat * target_flat, axis=-1)
    denominator = jnp.linalg.norm(predicted_flat, axis=-1) * jnp.linalg.norm(
        target_flat, axis=-1
    )
    valid = jnp.linalg.norm(target_flat, axis=-1) > 1.0e-6
    return jnp.where(
        valid,
        numerator / jnp.maximum(denominator, 1.0e-8),
        jnp.asarray(jnp.nan, dtype=predicted.dtype),
    )


def score_paths_per_example(
    paths: at.Array,
    metric_tensors: dict[str, at.Array],
    energy_config: ObstacleEnergyConfig,
) -> dict[str, at.Array]:
    """Return the energy routine's batch metrics independently per example."""

    def score_one(path, one_metrics):
        batched_metrics = jax.tree.map(lambda value: value[None], one_metrics)
        _, info = compute_obstacle_energy(path[None], batched_metrics, energy_config)
        physical, _ = decode_bounded_waypoint_path(
            path[None],
            batched_metrics,
            energy_config,
        )
        increments = physical[:, 1:, :] - physical[:, :-1, :]
        score = {key: info[key] for key in SCORE_KEYS}
        score.update(
            {
                "min_path_x_m": jnp.min(physical[..., 0]),
                "backward_segment_rate": jnp.mean(
                    (increments[..., 0] < -1.0e-6).astype(jnp.float32)
                ),
            }
        )
        return score

    return jax.vmap(score_one)(paths, metric_tensors)


def learned_field_step(
    model_def: Any,
    params: Any,
    observation: Any,
    current_paths: at.Array,
    metric_tensors: dict[str, at.Array],
    energy_config: ObstacleEnergyConfig,
    *,
    field_time: float,
    max_direction_rms: float,
    step_size: float,
    step_size_start: float | None = None,
    guarded: bool,
    backtracks: int,
    backtrack_factor: float,
    energy_tolerance: float,
) -> tuple[at.Array, at.Array, dict[str, at.Array]]:
    """Predict and apply one learned update."""
    model = nnx.merge(model_def, params)
    model.eval()
    raw_direction = predict_descent_field(
        model,
        observation,
        current_paths,
        field_time=field_time,
    )
    return apply_field_direction(
        current_paths,
        raw_direction,
        metric_tensors,
        energy_config,
        max_direction_rms=max_direction_rms,
        step_size=step_size,
        step_size_start=step_size_start,
        guarded=guarded,
        backtracks=backtracks,
        backtrack_factor=backtrack_factor,
        energy_tolerance=energy_tolerance,
    )


def apply_field_direction(
    current_paths: at.Array,
    raw_direction: at.Array,
    metric_tensors: dict[str, at.Array],
    energy_config: ObstacleEnergyConfig,
    *,
    max_direction_rms: float,
    step_size: float,
    step_size_start: float | None = None,
    guarded: bool,
    backtracks: int,
    backtrack_factor: float,
    energy_tolerance: float,
) -> tuple[at.Array, at.Array, dict[str, at.Array]]:
    """Apply a raw field; only guarded mode may consult trajectory energy."""
    raw_finite = jnp.all(jnp.isfinite(raw_direction), axis=(-2, -1))
    raw_rms = _rms_per_path(raw_direction)
    predicted_direction = raw_direction
    predicted_direction, uncapped_rms = _cap_direction_rms(
        predicted_direction,
        max_direction_rms,
    )

    if guarded:
        next_paths, acceptance = backtracking_projected_step(
            current_paths,
            predicted_direction,
            metric_tensors,
            energy_config,
            step_size=step_size,
            step_size_start=step_size_start,
            backtracks=backtracks,
            backtrack_factor=backtrack_factor,
            energy_tolerance=energy_tolerance,
        )
        accepted = acceptance["accepted"].astype(jnp.float32)
        applied_step_size = acceptance["accepted_step_size"]
    else:
        next_paths = project_bounded_actions(
            current_paths
            + waypoint_step_sizes(current_paths, step_size, step_size_start) * predicted_direction,
            metric_tensors,
            energy_config,
        )
        accepted = jnp.ones((current_paths.shape[0],), dtype=jnp.float32)
        applied_step_size = jnp.full(
            (current_paths.shape[0],), step_size, dtype=current_paths.dtype
        )

    current_physical = maybe_unnormalize_actions(current_paths, metric_tensors)
    next_physical = maybe_unnormalize_actions(next_paths, metric_tensors)
    path_change = _rms_per_path(next_physical - current_physical)
    cap_scale = jnp.minimum(
        1.0,
        max_direction_rms / jnp.maximum(raw_rms, 1.0e-6),
    )
    return next_paths, predicted_direction, {
        "raw_direction_finite": raw_finite.astype(jnp.float32),
        "raw_direction_rms": raw_rms,
        "field_direction_rms": _rms_per_path(predicted_direction),
        "field_uncapped_rms": uncapped_rms,
        "direction_cap_scale": cap_scale,
        "path_change_m": path_change,
        "moved": (path_change > 1.0e-6).astype(jnp.float32),
        "update_applied": accepted,
        "applied_step_size": applied_step_size,
    }


def oracle_projected_step(
    current_paths: at.Array,
    oracle_direction: at.Array,
    metric_tensors: dict[str, at.Array],
    energy_config: ObstacleEnergyConfig,
    *,
    step_size: float,
    step_size_start: float | None = None,
) -> tuple[at.Array, dict[str, at.Array]]:
    next_paths = project_bounded_actions(
        current_paths
        + waypoint_step_sizes(current_paths, step_size, step_size_start) * oracle_direction,
        metric_tensors,
        energy_config,
    )
    current_physical = maybe_unnormalize_actions(current_paths, metric_tensors)
    next_physical = maybe_unnormalize_actions(next_paths, metric_tensors)
    direction_rms = _rms_per_path(oracle_direction)
    path_change = _rms_per_path(next_physical - current_physical)
    return next_paths, {
        "oracle_direction_rms": direction_rms,
        "path_change_m": path_change,
        "moved": (path_change > 1.0e-6).astype(jnp.float32),
        "update_applied": (direction_rms > 1.0e-8).astype(jnp.float32),
        "applied_step_size": step_size * direction_rms,
    }


def field_oracle_diagnostics(
    predicted_direction: at.Array,
    oracle_direction: at.Array,
) -> dict[str, at.Array]:
    error = predicted_direction[:, 1:, :] - oracle_direction[:, 1:, :]
    target_rms = _rms_per_path(oracle_direction)
    return {
        "field_oracle_cosine": _cosine_per_path(
            predicted_direction,
            oracle_direction,
        ),
        "field_oracle_rmse": jnp.sqrt(jnp.mean(jnp.square(error), axis=(-2, -1))),
        "oracle_direction_rms": target_rms,
        "oracle_target_valid": (target_rms > 1.0e-6).astype(jnp.float32),
    }


def build_kernels(
    values: Mapping[str, Any],
    model_config: Any,
    energy_config: ObstacleEnergyConfig,
    *,
    prior_seed: int,
    safety_sample_spacing_m: float,
) -> EvaluationKernels:
    batch_size = int(model_config.batch_size)
    particles = int(values["train_particles"])
    if particles != 1:
        raise NotImplementedError(
            "This evaluator currently requires train_particles=1 so one path maps to one example."
        )

    def initialize_per_example(identity_words, metric_tensors):
        def initialize_one(words, one_metrics):
            key = jax.random.fold_in(jax.random.key(prior_seed), words[0])
            key = jax.random.fold_in(key, words[1])
            batched_metrics = jax.tree.map(lambda value: value[None], one_metrics)
            paths = initialize_prior_paths(
                key,
                batched_metrics,
                energy_config,
                base_batch_size=1,
                particles=particles,
                action_horizon=int(values["action_horizon"]),
                action_dim=int(values["action_dim"]),
            )
            return paths[0]

        return jax.vmap(initialize_one)(identity_words, metric_tensors)

    initialize = jax.jit(initialize_per_example)
    oracle = jax.jit(
        functools.partial(
            query_energy_oracle,
            energy_config=energy_config,
            base_batch_size=batch_size,
            particles=particles,
            gradient_floor=float(values["oracle_gradient_floor"]),
            max_direction_rms=float(values["max_field_direction_rms"]),
            diversity_direction_weight=float(values["diversity_direction_weight"]),
            step_size=float(values["aggregation_step_size"]),
            step_size_start=values.get("aggregation_step_size_start"),
            backtracks=int(values["rollout_backtracks"]),
            backtrack_factor=float(values["backtrack_factor"]),
            energy_tolerance=float(values["energy_increase_tolerance"]),
        )
    )
    score = jax.jit(
        functools.partial(score_paths_per_example, energy_config=energy_config)
    )
    dense_segment_samples = max(
        energy_config.segment_samples,
        math.ceil(energy_config.max_step_length_m / safety_sample_spacing_m),
    )
    dense_safety_config = dataclasses.replace(
        energy_config,
        segment_samples=dense_segment_samples,
        strict_esdf_coverage=True,
    )
    dense_score = jax.jit(
        functools.partial(
            score_paths_per_example,
            energy_config=dense_safety_config,
        )
    )
    oracle_step = jax.jit(
        functools.partial(
            oracle_projected_step,
            energy_config=energy_config,
            step_size=float(values["aggregation_step_size"]),
            step_size_start=values.get("aggregation_step_size_start"),
        )
    )
    decode = jax.jit(
        functools.partial(decode_bounded_waypoint_path, energy_config=energy_config)
    )
    return EvaluationKernels(
        initialize_prior=initialize,
        query_oracle=oracle,
        score_paths=score,
        score_dense_safety=dense_score,
        oracle_step=oracle_step,
        decode_paths=decode,
    )


def model_template(
    model_config: Any,
    mesh: jax.sharding.Mesh,
    *,
    seed: int,
    parameter_sources: Sequence[ParameterSource],
) -> tuple[Any, dict[ParameterSource, Any], dict[ParameterSource, Any]]:
    """Build model graph/parameter shapes with the trainer's exact dtype policy."""
    state_shape, state_sharding = init_train_state(
        model_config,
        jax.random.key(seed),
        mesh,
        resume=True,
    )
    parameter_shapes: dict[ParameterSource, Any] = {"raw": state_shape.params}
    parameter_shardings: dict[ParameterSource, Any] = {
        "raw": state_sharding.params
    }
    if "ema" in parameter_sources:
        if state_shape.ema_params is None or state_sharding.ema_params is None:
            raise ValueError("EMA parameters were requested but training did not use EMA")
        parameter_shapes["ema"] = state_shape.ema_params
        parameter_shardings["ema"] = state_sharding.ema_params
    return state_shape.model_def, parameter_shapes, parameter_shardings


def restore_checkpoint_params(
    checkpoint_root: Path,
    *,
    step: int,
    source: ParameterSource,
    params_shape: Any,
    params_sharding: Any,
) -> Any:
    """Restore only one parameter tree, never optimizer state or the other weights."""
    step_dir = checkpoint_root / str(step)
    source_dir = step_dir / ("train_state" if source == "raw" else "params")
    item = {"params": params_shape}
    item_sharding = {"params": params_sharding}
    restore_args = jax.tree.map(
        lambda _, leaf_sharding: ocp.ArrayRestoreArgs(
            restore_type=jax.Array,
            sharding=leaf_sharding,
        ),
        item,
        item_sharding,
    )
    kwargs: dict[str, Any] = {"item": item, "restore_args": restore_args}
    if source == "raw":
        # The train_state item also contains step/optimizer leaves.  Empty
        # transforms requests an intentional partial restore of only `params`.
        kwargs["transforms"] = {}
    with ocp.PyTreeCheckpointer() as checkpointer:
        restored = checkpointer.restore(
            source_dir,
            args=ocp.args.PyTreeRestore(**kwargs),
        )["params"]
    jax.block_until_ready(restored)
    if jax.tree.structure(restored) != jax.tree.structure(params_shape):
        raise ValueError(f"Restored {source} parameter structure does not match the model")
    restored_leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(restored)
    expected_leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(params_shape)
    mismatches = []
    for (restored_path, restored_leaf), (_, expected_leaf) in zip(
        restored_leaves_with_paths,
        expected_leaves_with_paths,
        strict=True,
    ):
        actual_shape = tuple(getattr(restored_leaf, "shape", ()))
        expected_shape = tuple(getattr(expected_leaf, "shape", ()))
        actual_dtype = str(getattr(restored_leaf, "dtype", ""))
        expected_dtype = str(getattr(expected_leaf, "dtype", ""))
        if actual_shape != expected_shape or actual_dtype != expected_dtype:
            mismatches.append(
                f"{jax.tree_util.keystr(restored_path)}: "
                f"restored {actual_shape}/{actual_dtype}, "
                f"expected {expected_shape}/{expected_dtype}"
            )
    if mismatches:
        preview = "; ".join(mismatches[:5])
        raise ValueError(
            f"Restored {source} parameter leaves do not match the model: {preview}"
        )
    return restored


def build_model_kernels(
    model_def: Any,
    values: Mapping[str, Any],
    energy_config: ObstacleEnergyConfig,
) -> ModelKernels:
    common = {
        "energy_config": energy_config,
        "field_time": float(values["field_time"]),
        "max_direction_rms": float(values["max_field_direction_rms"]),
        "step_size": float(values["aggregation_step_size"]),
        "step_size_start": values.get("aggregation_step_size_start"),
        "backtracks": int(values["rollout_backtracks"]),
        "backtrack_factor": float(values["backtrack_factor"]),
        "energy_tolerance": float(values["energy_increase_tolerance"]),
    }
    return ModelKernels(
        model_only_step=jax.jit(
            functools.partial(
                learned_field_step,
                model_def,
                guarded=False,
                **common,
            )
        ),
        guarded_step=jax.jit(
            functools.partial(
                learned_field_step,
                model_def,
                guarded=True,
                **common,
            )
        ),
        alignment=jax.jit(field_oracle_diagnostics),
    )


def _host_vectors(tree: Mapping[str, Any], valid_count: int) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key, value in jax.device_get(tree).items():
        array = np.asarray(value)
        if array.ndim == 0:
            array = np.full((valid_count,), array.item())
        elif array.shape[0] < valid_count:
            raise ValueError(f"Metric {key} has shape {array.shape}; expected batch axis")
        result[key] = np.asarray(array[:valid_count])
    return result


def _numeric(value: Any) -> int | float:
    scalar = np.asarray(value).item()
    if isinstance(scalar, (bool, np.bool_)):
        return int(scalar)
    if isinstance(scalar, (int, np.integer)):
        return int(scalar)
    return float(scalar)


def records_for_depth(
    *,
    checkpoint: str,
    parameter_source: str,
    mode: Mode,
    depth: int,
    identities: Sequence[ExampleIdentity],
    scores: Mapping[str, np.ndarray],
    transition: Mapping[str, np.ndarray] | None,
    previous_energy: np.ndarray | None,
    rollout_valid: np.ndarray,
    progress_threshold: float,
) -> list[dict[str, Any]]:
    rows = []
    for row_index, identity in enumerate(identities):
        row: dict[str, Any] = {
            "checkpoint": checkpoint,
            "parameter_source": parameter_source,
            "mode": mode,
            "depth": depth,
            **dataclasses.asdict(identity),
        }
        for key, values in scores.items():
            row[key] = _numeric(values[row_index])
        if transition is not None:
            for key, values in transition.items():
                row[key] = _numeric(values[row_index])
        if previous_energy is not None:
            energy_delta = float(row["obstacle_energy"] - previous_energy[row_index])
            row["energy_delta"] = energy_delta
            row["energy_nonincrease"] = float(energy_delta <= 1.0e-6)

        row["rollout_valid"] = float(bool(rollout_valid[row_index]))
        collision_free = float(row["dense_collision_rate"]) < 0.5
        safe = float(row["dense_unsafe_rate"]) < 0.5
        valid_geometry = (
            bool(rollout_valid[row_index])
            and float(row["action_finite_rate"]) >= 1.0 - 1.0e-6
            and float(row["dense_all_invalid_traj_rate"]) < 0.5
            and float(row["dense_invalid_esdf_rate"]) <= 1.0e-6
        )
        sufficient_progress = (
            float(row["achieved_required_progress_ratio"]) >= progress_threshold
        )
        row["progress_success"] = float(sufficient_progress)
        row["success"] = float(
            valid_geometry and collision_free and sufficient_progress
        )
        row["safe_success"] = float(valid_geometry and safe and sufficient_progress)
        rows.append(row)
    return rows


def _identity_words(
    identities: Sequence[ExampleIdentity],
    *,
    batch_size: int,
) -> np.ndarray:
    words = []
    for identity in identities:
        digest = bytes.fromhex(identity.digest_hex)
        words.append(np.frombuffer(digest[:8], dtype="<u4"))
    while len(words) < batch_size:
        words.append(words[-1].copy())
    return np.asarray(words, dtype=np.uint32)


def evaluate_modes(
    *,
    modes: Sequence[Mode],
    checkpoint: str,
    parameter_source: str,
    params: Any | None,
    model_kernels: ModelKernels | None,
    dataset: OracleOnlyNavigationDataset,
    batcher: OracleOnlyNavigationDataLoader,
    selected_examples: Sequence[ExampleIdentity],
    kernels: EvaluationKernels,
    mesh: jax.sharding.Mesh,
    seed: int,
    rollout_rounds: int,
    progress_threshold: float,
    writer: JsonlWriter,
    records: list[dict[str, Any]],
    final_paths: dict[tuple[str, str, str, int], np.ndarray],
) -> None:
    """Evaluate compatible modes for one checkpoint/source combination."""
    batch_size = batcher.batch_size
    total_batches = math.ceil(len(selected_examples) / batch_size)
    for batch_index, observation, metrics, identities, valid_count in iter_evaluation_batches(
        dataset,
        batcher,
        selected_examples,
        batch_size=batch_size,
        seed=seed,
    ):
        words = jax.make_array_from_process_local_data(
            batcher._sharding,
            _identity_words(identities, batch_size=batch_size),
        )
        with sharding.set_mesh(mesh):
            initial_paths = kernels.initialize_prior(words, metrics)

        for mode in modes:
            current_paths = initial_paths
            pending_transition: dict[str, np.ndarray] | None = None
            previous_energy: np.ndarray | None = None
            rollout_valid = np.ones((valid_count,), dtype=bool)
            max_depth = 0 if mode == "prior" else rollout_rounds
            for depth in range(max_depth + 1):
                scores = _host_vectors(kernels.score_paths(current_paths, metrics), valid_count)
                dense_scores = _host_vectors(
                    kernels.score_dense_safety(current_paths, metrics),
                    valid_count,
                )
                for dense_key in (
                    "collision_rate",
                    "unsafe_rate",
                    "invalid_esdf_rate",
                    "all_invalid_traj_rate",
                    "min_esdf_m",
                    "min_clearance_m",
                ):
                    scores[f"dense_{dense_key}"] = dense_scores[dense_key]
                depth_rows = records_for_depth(
                    checkpoint=checkpoint,
                    parameter_source=parameter_source,
                    mode=mode,
                    depth=depth,
                    identities=identities,
                    scores=scores,
                    transition=pending_transition,
                    previous_energy=previous_energy,
                    rollout_valid=rollout_valid,
                    progress_threshold=progress_threshold,
                )
                for row in depth_rows:
                    writer.write(row)
                records.extend(depth_rows)

                if depth == max_depth:
                    decoded, _ = kernels.decode_paths(current_paths, metrics)
                    physical = np.asarray(jax.device_get(decoded), dtype=np.float32)
                    for item_index, identity in enumerate(identities):
                        final_paths[
                            (checkpoint, parameter_source, mode, identity.dataset_index)
                        ] = physical[item_index]
                    break

                with sharding.set_mesh(mesh):
                    if mode == "oracle":
                        oracle_direction, oracle_info = kernels.query_oracle(
                            current_paths, metrics
                        )
                        next_paths, transition_info = kernels.oracle_step(
                            current_paths,
                            oracle_direction,
                            metrics,
                        )
                    elif mode == "model_only":
                        if params is None or model_kernels is None:
                            raise ValueError("model_only evaluation requires a model state")
                        next_paths, predicted_direction, transition_info = (
                            model_kernels.model_only_step(
                                params,
                                observation,
                                current_paths,
                                metrics,
                            )
                        )
                        # This oracle query is diagnostic-only and occurs after
                        # the unguarded model transition has been determined.
                        oracle_direction, oracle_info = kernels.query_oracle(
                            current_paths, metrics
                        )
                        alignment = model_kernels.alignment(
                            predicted_direction, oracle_direction
                        )
                    elif mode == "guarded":
                        if params is None or model_kernels is None:
                            raise ValueError("guarded evaluation requires a model state")
                        oracle_direction, oracle_info = kernels.query_oracle(
                            current_paths, metrics
                        )
                        next_paths, predicted_direction, transition_info = (
                            model_kernels.guarded_step(
                                params,
                                observation,
                                current_paths,
                                metrics,
                            )
                        )
                        alignment = model_kernels.alignment(
                            predicted_direction, oracle_direction
                        )
                    else:  # pragma: no cover - prior has max_depth zero
                        raise ValueError(f"Unsupported rollout mode: {mode}")

                pending_transition = _host_vectors(transition_info, valid_count)
                if mode in MODEL_MODES:
                    pending_transition.update(_host_vectors(alignment, valid_count))
                    rollout_valid &= pending_transition["raw_direction_finite"].astype(
                        bool
                    )
                else:
                    oracle_finite = np.all(
                        np.isfinite(
                            np.asarray(jax.device_get(oracle_direction))[:valid_count]
                        ),
                        axis=(-2, -1),
                    )
                    rollout_valid &= oracle_finite
                oracle_gradient = np.asarray(
                    jax.device_get(oracle_info["raw_oracle_grad_rms_per_path"])
                )[:valid_count]
                pending_transition["oracle_gradient_rms"] = oracle_gradient
                pending_transition["rollout_valid"] = rollout_valid.astype(np.float32)
                previous_energy = scores["obstacle_energy"].copy()
                current_paths = next_paths

        logging.info(
            "Evaluated %s/%s %s batch %d/%d",
            checkpoint,
            parameter_source or "none",
            ",".join(modes),
            batch_index + 1,
            total_batches,
        )


def _is_numeric_metric(key: str, values: Sequence[Any]) -> bool:
    if key in IDENTITY_KEYS:
        return False
    return bool(values) and all(
        isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating))
        for value in values
    )


def _bootstrap_clip_mean_ci(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    samples: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if samples <= 0:
        return None, None
    by_clip: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if isinstance(value, (bool, int, float)) and math.isfinite(float(value)):
            by_clip[str(row["clip_id"])].append(float(value))
    clips = sorted(by_clip)
    if not clips:
        return None, None
    clip_means = np.asarray([np.mean(by_clip[clip]) for clip in clips], dtype=np.float64)
    rng = np.random.default_rng(seed)
    selections = rng.integers(0, len(clips), size=(samples, len(clips)))
    bootstrap_means = np.mean(clip_means[selections], axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(low), float(high)


def _clip_means(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> dict[str, float]:
    by_clip: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if isinstance(value, (bool, int, float)) and math.isfinite(float(value)):
            by_clip[str(row["clip_id"])].append(float(value))
    return {clip: float(np.mean(values)) for clip, values in by_clip.items()}


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        group = (
            str(row["checkpoint"]),
            str(row["parameter_source"]),
            str(row["mode"]),
            int(row["depth"]),
        )
        groups[group].append(row)

    summary = []
    for group, rows in sorted(groups.items()):
        candidate_keys = sorted(set().union(*(row.keys() for row in rows)))
        for metric in candidate_keys:
            metric_values = [row[metric] for row in rows if metric in row]
            if not _is_numeric_metric(metric, metric_values):
                continue
            finite = np.asarray(
                [float(value) for value in metric_values if math.isfinite(float(value))],
                dtype=np.float64,
            )
            if finite.size == 0:
                continue
            clip_means = _clip_means(rows, metric)
            digest = hashlib.sha256(f"{group}:{metric}".encode()).digest()
            metric_seed = seed ^ int.from_bytes(digest[:4], "little")
            if metric in PRIMARY_METRICS:
                ci_low, ci_high = _bootstrap_clip_mean_ci(
                    rows,
                    metric,
                    samples=bootstrap_samples,
                    seed=metric_seed,
                )
            else:
                ci_low, ci_high = None, None
            summary.append(
                {
                    "checkpoint": group[0],
                    "parameter_source": group[1],
                    "mode": group[2],
                    "depth": group[3],
                    "metric": metric,
                    "count": int(finite.size),
                    "num_clips": len(clip_means),
                    "mean": float(np.mean(finite)),
                    "clip_macro_mean": float(np.mean(list(clip_means.values()))),
                    "std": float(np.std(finite)),
                    "median": float(np.median(finite)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )
    return summary


def _paired_comparison_specs(
    variants: set[tuple[str, str, str, int]],
) -> list[
    tuple[
        str,
        tuple[str, str, str, int],
        tuple[str, str, str, int],
    ]
]:
    """Choose interpretable, matched comparisons with right-minus-left signs."""
    specs: set[
        tuple[
            str,
            tuple[str, str, str, int],
            tuple[str, str, str, int],
        ]
    ] = set()

    # Training continuation: each later retained checkpoint minus its immediate
    # predecessor, holding source, mode, and depth fixed.
    checkpoint_groups: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for checkpoint, source, mode, depth in variants:
        if checkpoint.isdigit():
            checkpoint_groups[(source, mode, depth)].append(int(checkpoint))
    for (source, mode, depth), checkpoints in checkpoint_groups.items():
        ordered = sorted(set(checkpoints))
        for left_step, right_step in zip(ordered, ordered[1:], strict=False):
            specs.add(
                (
                    "checkpoint",
                    (str(left_step), source, mode, depth),
                    (str(right_step), source, mode, depth),
                )
            )

    # EMA minus raw at the same checkpoint/mode/depth.
    for checkpoint, source, mode, depth in variants:
        if source != "raw":
            continue
        right = (checkpoint, "ema", mode, depth)
        if right in variants:
            specs.add(("parameter_source", (checkpoint, source, mode, depth), right))

    # Guarded minus model-only at the same checkpoint/source/depth.
    for checkpoint, source, mode, depth in variants:
        if mode != "model_only":
            continue
        right = (checkpoint, source, "guarded", depth)
        if right in variants:
            specs.add(("mode", (checkpoint, source, mode, depth), right))

    # Privileged oracle minus each learned mode at matching rollout depth.
    for variant in variants:
        checkpoint, source, mode, depth = variant
        if mode not in MODEL_MODES:
            continue
        oracle = ("baseline", "", "oracle", depth)
        if oracle in variants:
            specs.add(("oracle_reference", variant, oracle))

    # Final learned rollout minus the common initialization.
    prior = ("baseline", "", "prior", 0)
    if prior in variants:
        model_groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        for checkpoint, source, mode, depth in variants:
            if mode in MODEL_MODES:
                model_groups[(checkpoint, source, mode)].append(depth)
        for (checkpoint, source, mode), depths in model_groups.items():
            final_variant = (checkpoint, source, mode, max(depths))
            specs.add(("prior_reference", prior, final_variant))

    return sorted(specs)


def summarize_paired_differences(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Summarize matched right-minus-left differences at the clip level."""
    rows_by_variant: dict[
        tuple[str, str, str, int], dict[str, Mapping[str, Any]]
    ] = defaultdict(dict)
    for row in records:
        variant = (
            str(row["checkpoint"]),
            str(row["parameter_source"]),
            str(row["mode"]),
            int(row["depth"]),
        )
        example_id = str(row["example_id"])
        if example_id in rows_by_variant[variant]:
            raise ValueError(f"Duplicate result for variant {variant}: {example_id}")
        rows_by_variant[variant][example_id] = row

    result: list[dict[str, Any]] = []
    for comparison_type, left_variant, right_variant in _paired_comparison_specs(
        set(rows_by_variant)
    ):
        left_rows = rows_by_variant[left_variant]
        right_rows = rows_by_variant[right_variant]
        matched_ids = sorted(set(left_rows).intersection(right_rows))
        if not matched_ids:
            continue
        for metric in sorted(PRIMARY_METRICS):
            differences = []
            for example_id in matched_ids:
                left_row = left_rows[example_id]
                right_row = right_rows[example_id]
                left_value = left_row.get(metric)
                right_value = right_row.get(metric)
                if not isinstance(left_value, (bool, int, float)) or not isinstance(
                    right_value, (bool, int, float)
                ):
                    continue
                if not math.isfinite(float(left_value)) or not math.isfinite(
                    float(right_value)
                ):
                    continue
                if str(left_row["clip_id"]) != str(right_row["clip_id"]):
                    raise ValueError(f"Clip identity changed for paired row {example_id}")
                differences.append(
                    {
                        "clip_id": str(left_row["clip_id"]),
                        "delta": float(right_value) - float(left_value),
                    }
                )
            if not differences:
                continue
            values = np.asarray(
                [float(item["delta"]) for item in differences], dtype=np.float64
            )
            clip_means = _clip_means(differences, "delta")
            digest = hashlib.sha256(
                f"{comparison_type}:{left_variant}:{right_variant}:{metric}".encode()
            ).digest()
            metric_seed = seed ^ int.from_bytes(digest[:4], "little")
            ci_low, ci_high = _bootstrap_clip_mean_ci(
                differences,
                "delta",
                samples=bootstrap_samples,
                seed=metric_seed,
            )
            result.append(
                {
                    "comparison_type": comparison_type,
                    "difference": "right_minus_left",
                    "left_checkpoint": left_variant[0],
                    "left_parameter_source": left_variant[1],
                    "left_mode": left_variant[2],
                    "left_depth": left_variant[3],
                    "right_checkpoint": right_variant[0],
                    "right_parameter_source": right_variant[1],
                    "right_mode": right_variant[2],
                    "right_depth": right_variant[3],
                    "metric": metric,
                    "count": int(values.size),
                    "num_clips": len(clip_means),
                    "mean_delta": float(np.mean(values)),
                    "clip_macro_mean_delta": float(
                        np.mean(list(clip_means.values()))
                    ),
                    "std_delta": float(np.std(values)),
                    "median_delta": float(np.median(values)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )
    return result


def write_paired_summary_csv(
    path: Path,
    summary: Sequence[Mapping[str, Any]],
) -> None:
    fieldnames = (
        "comparison_type",
        "difference",
        "left_checkpoint",
        "left_parameter_source",
        "left_mode",
        "left_depth",
        "right_checkpoint",
        "right_parameter_source",
        "right_mode",
        "right_depth",
        "metric",
        "count",
        "num_clips",
        "mean_delta",
        "clip_macro_mean_delta",
        "std_delta",
        "median_delta",
        "ci95_low",
        "ci95_high",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def write_summary_csv(path: Path, summary: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = (
        "checkpoint",
        "parameter_source",
        "mode",
        "depth",
        "metric",
        "count",
        "num_clips",
        "mean",
        "clip_macro_mean",
        "std",
        "median",
        "ci95_low",
        "ci95_high",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def write_per_clip_summary_csv(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    groups: dict[tuple[str, str, str, int, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in records:
        groups[
            (
                str(row["checkpoint"]),
                str(row["parameter_source"]),
                str(row["mode"]),
                int(row["depth"]),
                str(row["clip_id"]),
            )
        ].append(row)

    fieldnames = (
        "checkpoint",
        "parameter_source",
        "mode",
        "depth",
        "clip_id",
        "metric",
        "count",
        "mean",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group, rows in sorted(groups.items()):
            candidate_keys = sorted(set().union(*(row.keys() for row in rows)))
            for metric in candidate_keys:
                values = [row[metric] for row in rows if metric in row]
                if not _is_numeric_metric(metric, values):
                    continue
                finite = [float(value) for value in values if math.isfinite(float(value))]
                if not finite:
                    continue
                writer.writerow(
                    {
                        "checkpoint": group[0],
                        "parameter_source": group[1],
                        "mode": group[2],
                        "depth": group[3],
                        "clip_id": group[4],
                        "metric": metric,
                        "count": len(finite),
                        "mean": float(np.mean(finite)),
                    }
                )


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "example"


def write_worst_case_plots(
    *,
    output_dir: Path,
    dataset: OracleOnlyNavigationDataset,
    records: Sequence[Mapping[str, Any]],
    final_paths: Mapping[tuple[str, str, str, int], np.ndarray],
    checkpoint: str,
    parameter_source: str,
    rollout_rounds: int,
    num_plots: int,
) -> list[Path]:
    if num_plots <= 0:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib unavailable; skipping worst-case path plots")
        return []

    candidates = [
        row
        for row in records
        if row["checkpoint"] == checkpoint
        and row["parameter_source"] == parameter_source
        and row["mode"] == "model_only"
        and int(row["depth"]) == rollout_rounds
    ]
    candidates.sort(
        key=lambda row: (
            float(row["safe_success"]),
            -float(row["dense_unsafe_rate"]),
            -float(row["dense_collision_rate"]),
            float(row["achieved_required_progress_ratio"]),
            -float(row["obstacle_energy"]),
        )
    )
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for rank, row in enumerate(candidates[:num_plots], start=1):
        dataset_index = int(row["dataset_index"])
        sample = dataset[dataset_index]
        esdf = np.asarray(sample["esdf"], dtype=np.float32)
        resolution = float(np.asarray(sample["esdf_resolution"]))
        x_min = float(np.asarray(sample["esdf_x_min"]))
        y_min = float(np.asarray(sample["esdf_y_min"]))
        extent = [
            x_min,
            x_min + esdf.shape[1] * resolution,
            y_min,
            y_min + esdf.shape[0] * resolution,
        ]
        goal = np.asarray(sample["goal_xy"], dtype=np.float32)
        path_keys = (
            ("prior", ("baseline", "", "prior", dataset_index)),
            ("oracle", ("baseline", "", "oracle", dataset_index)),
            (
                "model_only",
                (checkpoint, parameter_source, "model_only", dataset_index),
            ),
            ("guarded", (checkpoint, parameter_source, "guarded", dataset_index)),
        )

        fig, ax = plt.subplots(figsize=(8.0, 6.5), dpi=130)
        color_scale = esdf_display_scale(esdf)
        background = ax.imshow(
            np.where(np.isfinite(esdf), esdf, np.nan),
            origin="lower",
            extent=extent,
            cmap="coolwarm",
            vmin=-color_scale,
            vmax=color_scale,
            aspect="equal",
        )
        styles = {
            "prior": ("white", "--"),
            "oracle": ("tab:green", "-"),
            "model_only": ("tab:red", "-"),
            "guarded": ("tab:blue", "-"),
        }
        for label, key in path_keys:
            path = final_paths.get(key)
            if path is None:
                continue
            color, linestyle = styles[label]
            ax.plot(
                path[:, 0],
                path[:, 1],
                color=color,
                linestyle=linestyle,
                linewidth=2.2,
                label=label,
            )
        ax.scatter([0.0], [0.0], color="white", edgecolor="black", s=50, label="robot")
        ax.scatter([goal[0]], [goal[1]], marker="*", color="yellow", s=120, label="goal")
        ax.set_title(
            f"{row['clip_id']} / {row['sample_id']} — "
            f"collision={float(row['dense_collision_rate']):.0f}, "
            f"progress={float(row['achieved_required_progress_ratio']):.3f}"
        )
        ax.set_xlabel("local x [m]")
        ax.set_ylabel("local y [m]")
        ax.legend(fontsize=8, loc="best")
        fig.colorbar(background, ax=ax, label="ESDF [m]")
        fig.tight_layout()
        filename = (
            f"{rank:02d}_{_safe_filename(str(row['clip_id']))}_"
            f"{_safe_filename(str(row['sample_id']))}.png"
        )
        plot_path = plot_dir / filename
        fig.savefig(plot_path)
        plt.close(fig)
        written.append(plot_path)
    return written


def save_final_paths(
    path: Path,
    final_paths: Mapping[tuple[str, str, str, int], np.ndarray],
) -> None:
    arrays: dict[str, np.ndarray] = {}
    metadata = []
    for array_index, (key, value) in enumerate(sorted(final_paths.items())):
        array_key = f"path_{array_index:07d}"
        arrays[array_key] = np.asarray(value, dtype=np.float32)
        metadata.append(
            {
                "array_key": array_key,
                "checkpoint": key[0],
                "parameter_source": key[1],
                "mode": key[2],
                "dataset_index": key[3],
            }
        )
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **arrays)


def maybe_log_wandb(
    args: argparse.Namespace,
    run_metadata: Mapping[str, Any],
    summary: Sequence[Mapping[str, Any]],
    plots: Sequence[Path],
) -> None:
    if not args.wandb_enabled:
        return
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        job_type="evaluation",
        config=dict(run_metadata),
    )
    by_depth: dict[int, dict[str, float]] = defaultdict(dict)
    for row in summary:
        if row["metric"] not in PRIMARY_METRICS:
            continue
        depth = int(row["depth"])
        prefix = (
            f"eval/{row['checkpoint']}/{row['parameter_source'] or 'none'}/"
            f"{row['mode']}"
        )
        by_depth[depth][f"{prefix}/{row['metric']}"] = float(row["clip_macro_mean"])
    for depth, metrics in sorted(by_depth.items()):
        wandb.log(metrics, step=depth)
    if plots:
        wandb.log({"eval/worst_cases": [wandb.Image(str(path)) for path in plots]})
    run.finish()


def _prepare_output_dir(path: Path) -> Path:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Evaluation output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_args(args: argparse.Namespace, values: Mapping[str, Any]) -> None:
    if jax.process_count() != 1:
        raise RuntimeError(
            "This evaluator supports one JAX process only; multi-process launches "
            "would duplicate examples and race on result files"
        )
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.rollout_rounds < 0:
        raise ValueError("--rollout-rounds must be non-negative")
    if args.fsdp_devices <= 0:
        raise ValueError("--fsdp-devices must be positive")
    if int(values["train_particles"]) != 1:
        raise ValueError("This evaluator currently supports only single-particle checkpoints")
    if not 0.0 <= args.progress_threshold <= 1.0:
        raise ValueError("--progress-threshold must be in [0, 1]")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be non-negative")
    if args.safety_sample_spacing_m <= 0.0:
        raise ValueError("--safety-sample-spacing-m must be positive")
    for option, entries in (
        ("--steps", args.steps),
        ("--parameter-sources", args.parameter_sources),
        ("--modes", args.modes),
    ):
        if len(entries) != len(set(entries)):
            raise ValueError(f"{option} contains duplicate entries")
    if not args.dry_run:
        if args.batch_size % jax.device_count() != 0:
            raise ValueError(
                "--batch-size must be divisible by the visible JAX device count"
            )
        if jax.device_count() % args.fsdp_devices != 0:
            raise ValueError("Visible JAX devices must be divisible by --fsdp-devices")
        if MODEL_MODES.intersection(args.modes):
            for step in args.steps:
                checkpoint = args.checkpoint_root / str(step)
                if not (checkpoint / "_CHECKPOINT_METADATA").exists():
                    raise FileNotFoundError(
                        f"Committed full checkpoint not found: {checkpoint}"
                    )
    if args.goal_kind != "sampled":
        logging.warning(
            "This run trained with sampled_goal_fraction=1.0; %s goals are out of distribution.",
            args.goal_kind,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=_path, required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[30_000])
    parser.add_argument(
        "--parameter-sources",
        nargs="+",
        choices=("ema", "raw"),
        default=["raw"],
        help="Raw exactly matches training rollouts; EMA is a separately labeled ablation.",
    )
    parser.add_argument(
        "--training-config",
        type=_path,
        default=None,
        help="Defaults to <checkpoint-root>/monitoring/configuration.json.",
    )
    parser.add_argument("--data-root", type=_path, default=REPOSITORY_ROOT / "data")
    parser.add_argument("--manifest-path", type=_path, default=None)
    parser.add_argument("--eval-split-ids", type=_path, default=DEFAULT_EVAL_SPLIT)
    parser.add_argument(
        "--goal-kind",
        choices=("sampled", "object", "all"),
        default="sampled",
        help="Sampled waypoint goals match this run's training distribution.",
    )
    parser.add_argument(
        "--object-goal-modality",
        choices=("training", "text", "image", "waypoint", "all"),
        default="training",
    )
    parser.add_argument("--examples-per-clip", type=int, default=128)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=4)
    parser.add_argument("--rollout-rounds", type=int, default=12)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=VALID_MODES,
        default=["prior", "model_only", "guarded", "oracle"],
    )
    parser.add_argument("--progress-threshold", type=float, default=0.9)
    parser.add_argument(
        "--safety-sample-spacing-m",
        type=float,
        default=0.05,
        help="Maximum spacing for the independent dense collision/safety scorer.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--num-plots", type=int, default=12)
    parser.add_argument("--save-final-paths", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output-dir",
        type=_path,
        default=None,
        help="Defaults to outputs/evaluation/eval_<timestamp>.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and sample the held-out index without loading a checkpoint.",
    )
    parser.add_argument("--wandb-enabled", action="store_true")
    parser.add_argument("--wandb-project", default="astar-energy-eval")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> None:
    init_logging()
    args.checkpoint_root = _path(args.checkpoint_root)
    args.training_config = _path(
        args.training_config
        or args.checkpoint_root / "monitoring" / "configuration.json"
    )
    args.data_root = _path(args.data_root)
    args.eval_split_ids = _path(args.eval_split_ids)
    if args.manifest_path is not None:
        args.manifest_path = _path(args.manifest_path)

    values = load_training_config(args.training_config)
    model_args = build_training_namespace(
        values,
        batch_size=args.batch_size,
        fsdp_devices=args.fsdp_devices,
    )
    model_config = create_train_config(model_args)
    energy_config = build_energy_config(values)
    _validate_args(args, values)
    train_split_path, train_clip_ids, eval_clip_ids = validate_held_out_split(
        values,
        args.eval_split_ids,
    )

    dataset = build_dataset(args, values, model_config)
    groups = group_dataset_examples_by_clip(dataset, goal_kind=args.goal_kind)
    selected_examples = select_clip_balanced_examples(
        groups,
        examples_per_clip=args.examples_per_clip,
        seed=args.seed,
        max_examples=args.max_examples,
    )
    selected_counts: dict[str, int] = defaultdict(int)
    for example in selected_examples:
        selected_counts[example.clip_id] += 1
    logging.info(
        "Selected %d held-out %s examples across %d clips: %s",
        len(selected_examples),
        args.goal_kind,
        len(selected_counts),
        dict(sorted(selected_counts.items())),
    )
    if args.dry_run:
        dataset.close()
        return

    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(sharding.DATA_AXIS),
    )
    batcher = build_batcher(args, values, model_config, data_sharding, dataset)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = _prepare_output_dir(
        args.output_dir or DEFAULT_RESULTS_ROOT / f"eval_{timestamp}"
    )
    if args.wandb_run_name is None:
        args.wandb_run_name = f"eval_{args.checkpoint_root.name}_{timestamp}"

    run_metadata = {
        "checkpoint_root": str(args.checkpoint_root),
        "steps": args.steps,
        "parameter_sources": args.parameter_sources,
        "training_config": str(args.training_config),
        "train_split_ids": str(train_split_path),
        "eval_split_ids": str(args.eval_split_ids),
        "train_clip_count": len(train_clip_ids),
        "eval_clip_count": len(eval_clip_ids),
        "goal_kind": args.goal_kind,
        "object_goal_modality": args.object_goal_modality,
        "selected_examples": len(selected_examples),
        "selected_examples_by_clip": dict(sorted(selected_counts.items())),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "fsdp_devices": args.fsdp_devices,
        "rollout_rounds": args.rollout_rounds,
        "modes": args.modes,
        "progress_threshold": args.progress_threshold,
        "safety_sample_spacing_m": args.safety_sample_spacing_m,
        "safety_semantics": "all_segment_samples_conservative_esdf_coverage_v3",
        "safety_requires_full_esdf_coverage": True,
        "bootstrap_samples": args.bootstrap_samples,
        "jax_devices": [str(device) for device in jax.devices()],
        "provenance": {
            "astar_git_commit": _git_commit(REPOSITORY_ROOT),
            "eval_split_sha256": _sha256_file(args.eval_split_ids),
            "train_split_sha256": _sha256_file(train_split_path),
            "training_config_sha256": _sha256_file(args.training_config),
            "evaluator_sha256": _sha256_file(Path(__file__).resolve()),
            "aggregation_trainer_sha256": _sha256_file(
                PACKAGE_ROOT / "training" / "aggregation.py"
            ),
            "waypoint_steps_sha256": _sha256_file(
                PACKAGE_ROOT / "training" / "steps.py"
            ),
            "energy_source_sha256": _sha256_file(
                PACKAGE_ROOT / "waypoint_energy.py"
            ),
            "esdf_source_sha256": _sha256_file(PACKAGE_ROOT / "esdf.py"),
            "selection_sha256": hashlib.sha256(
                "\n".join(item.example_id for item in selected_examples).encode("utf-8")
            ).hexdigest(),
        },
        "training_values": values,
    }
    with (output_dir / "evaluation_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with (output_dir / "selected_examples.json").open("w", encoding="utf-8") as handle:
        json.dump(
            [dataclasses.asdict(item) for item in selected_examples],
            handle,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")

    kernels = build_kernels(
        values,
        model_config,
        energy_config,
        prior_seed=args.seed,
        safety_sample_spacing_m=args.safety_sample_spacing_m,
    )
    records: list[dict[str, Any]] = []
    final_paths: dict[tuple[str, str, str, int], np.ndarray] = {}
    examples_path = output_dir / "examples.jsonl"
    try:
        with JsonlWriter(examples_path) as writer:
            baseline_modes = [mode for mode in args.modes if mode in BASELINE_MODES]
            if baseline_modes:
                evaluate_modes(
                    modes=baseline_modes,
                    checkpoint="baseline",
                    parameter_source="",
                    params=None,
                    model_kernels=None,
                    dataset=dataset,
                    batcher=batcher,
                    selected_examples=selected_examples,
                    kernels=kernels,
                    mesh=mesh,
                    seed=args.seed,
                    rollout_rounds=args.rollout_rounds,
                    progress_threshold=args.progress_threshold,
                    writer=writer,
                    records=records,
                    final_paths=final_paths,
                )

            model_modes = [mode for mode in args.modes if mode in MODEL_MODES]
            if model_modes:
                model_def, parameter_shapes, parameter_shardings = model_template(
                    model_config,
                    mesh,
                    seed=int(getattr(model_config, "seed", args.seed)),
                    parameter_sources=args.parameter_sources,
                )
                model_kernels = build_model_kernels(
                    model_def,
                    values,
                    energy_config,
                )
                for step in args.steps:
                    for source in args.parameter_sources:
                        params = restore_checkpoint_params(
                            args.checkpoint_root,
                            step=step,
                            source=source,
                            params_shape=parameter_shapes[source],
                            params_sharding=parameter_shardings[source],
                        )
                        evaluate_modes(
                            modes=model_modes,
                            checkpoint=str(step),
                            parameter_source=source,
                            params=params,
                            model_kernels=model_kernels,
                            dataset=dataset,
                            batcher=batcher,
                            selected_examples=selected_examples,
                            kernels=kernels,
                            mesh=mesh,
                            seed=args.seed,
                            rollout_rounds=args.rollout_rounds,
                            progress_threshold=args.progress_threshold,
                            writer=writer,
                            records=records,
                            final_paths=final_paths,
                        )
                        del params
                        gc.collect()

        summary = summarize_records(
            records,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        write_summary_csv(output_dir / "summary.csv", summary)
        paired_summary = summarize_paired_differences(
            records,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        with (output_dir / "paired_summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                paired_summary,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        write_paired_summary_csv(output_dir / "paired_summary.csv", paired_summary)
        write_per_clip_summary_csv(output_dir / "per_clip_summary.csv", records)
        if args.save_final_paths:
            save_final_paths(output_dir / "final_paths.npz", final_paths)

        primary_checkpoint = str(args.steps[-1])
        primary_source = str(args.parameter_sources[0])
        plots = write_worst_case_plots(
            output_dir=output_dir,
            dataset=dataset,
            records=records,
            final_paths=final_paths,
            checkpoint=primary_checkpoint,
            parameter_source=primary_source,
            rollout_rounds=args.rollout_rounds,
            num_plots=args.num_plots,
        )
        maybe_log_wandb(args, run_metadata, summary, plots)
        logging.info("Evaluation complete. Results: %s", output_dir)
    finally:
        dataset.close()


def cli() -> None:
    main(parse_args())


if __name__ == "__main__":
    cli()
