"""Fixed validation examples, model-only rollouts, and focused training metrics.

No training entry point is imported here. Geometry is privileged scoring context;
the model-only transition never queries energy or an acceptance guard.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np

from astar.training.steps import waypoint_step_sizes
from astar.waypoint_energy import compute_obstacle_energy
from astar.waypoint_energy import decode_bounded_waypoint_path

ENERGY_TERMS = (
    "collision",
    "clearance",
    "goal",
    "progress",
    "early_heading",
    "smoothness",
)
ENERGY_EQUATION_NAMES = {
    "collision": "Ecollision",
    "clearance": "Eclearance",
    "goal": "Egoal",
    "progress": "Ebackward",
    "early_heading": "Eheading",
    "smoothness": "Esmoothness",
}
VALIDATION_METRICS = (
    "success_rate",
    "safe_success_rate",
    "progress_ratio",
    "collision_rate",
    "clearance_violation_rate",
    "invalid_coverage_rate",
    "goal_retreat_segment_rate",
)


def host_numpy(value):
    """Materialize a process-spanning JAX array identically on every host."""
    if isinstance(value, jax.Array) and value.is_fully_replicated:
        return np.asarray(jax.device_get(value.addressable_data(0)))
    if isinstance(value, jax.Array) and not value.is_fully_addressable:
        return np.asarray(multihost_utils.process_allgather(value))
    return np.asarray(jax.device_get(value))


def esdf_display_scale(grid) -> float:
    """Match the regenerated-ESDF sheets' robust, zero-centered color scale."""
    values = np.asarray(grid)
    finite_values = values[np.isfinite(values)]
    return max(
        float(np.percentile(np.abs(finite_values), 99))
        if finite_values.size
        else 1.0,
        1.0,
    )


def learning_metrics(info):
    """Use the ratio of means over the SAME logging window, not mean ratios."""
    baseline = float(info["zero_predictor_loss"])
    return {
        "train/loss": float(info["loss"]),
        "train/loss_vs_zero": float(info["loss"]) / baseline
        if baseline > 1e-12
        else None,
    }


def round_wandb_metrics(round_metrics, energy_config):
    """Log raw equation terms and their weighted contributions after each round."""
    payload = {
        "aggregation/field_oracle_cosine": round_metrics["field_oracle_cosine"],
        "aggregation/learned_step_acceptance": round_metrics["rollout_acceptance_rate"],
        "aggregation/collision_rate": round_metrics["collision_rate"],
        "aggregation/clearance_violation_rate": round_metrics[
            "clearance_violation_rate"
        ],
        "energy/total": round_metrics["energy_after"],
        "energy/E": round_metrics["energy_after"],
    }
    for term in ENERGY_TERMS:
        payload[f"energy/{ENERGY_EQUATION_NAMES[term]}"] = round_metrics[f"{term}_energy"]
        if getattr(energy_config, f"{term}_weight") != 0.0:
            payload[f"energy/{term}"] = round_metrics[f"weighted_{term}_energy"]
    return payload


def select_validation_examples(dataset, count, seed):
    """Choose unique frames, balanced across clips, without loading image/ESDF arrays."""

    def rank(value):
        return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()

    by_frame = {}
    previous_offset, record = None, None
    for index, ref in enumerate(dataset.goal_references):
        if ref.byte_offset != previous_offset:
            record = dataset._read_record_at(ref.byte_offset)
            previous_offset = ref.byte_offset
        clip, sample = str(record["clip_id"]), str(record["sample_id"])
        identity = f"{clip}/{sample}/{ref.goal_kind}/{ref.goal_index}"
        item = {
            "dataset_index": index,
            "clip_id": clip,
            "sample_id": sample,
            "id": identity,
        }
        frame = (clip, sample)
        if frame not in by_frame or rank(identity) < rank(by_frame[frame]["id"]):
            by_frame[frame] = item
    by_clip = defaultdict(list)
    for item in by_frame.values():
        by_clip[item["clip_id"]].append(item)
    for items in by_clip.values():
        items.sort(key=lambda item: rank(item["id"]))
    selected = []
    clips = sorted(by_clip, key=rank)
    for index in range(max(map(len, by_clip.values()), default=0)):
        for clip in clips:
            if index < len(by_clip[clip]):
                selected.append(by_clip[clip][index])
                if len(selected) == count:
                    return selected
    return selected


@dataclass
class ValidationBatch:
    observation: object
    metrics: dict
    initial_paths: object
    identities: list[dict]

    @property
    def valid_count(self):
        return len(self.identities)


def cache_validation_pool(
    loader, train_clip_ids, *, num_batches, seed, sanitize, fingerprint, initialize
):
    dataset = loader.dataset
    overlap = set(train_clip_ids or ()) & set(dataset._split_ids or ())
    if not train_clip_ids or not dataset._split_ids:
        raise ValueError(
            "Validation requires explicit, nonempty training and validation clip splits."
        )
    if overlap:
        raise ValueError(f"Validation split overlaps training clips: {sorted(overlap)}")
    process_count = jax.process_count()
    process_index = jax.process_index()
    global_batch_size = loader.batch_size * process_count
    selected = select_validation_examples(dataset, num_batches * global_batch_size, seed)
    if not selected:
        raise ValueError("No held-out validation examples are available.")
    rng = np.random.default_rng(seed)
    batches, fingerprints = [], []
    for batch_index, start in enumerate(range(0, len(selected), global_batch_size)):
        global_identities = selected[start : start + global_batch_size]
        local_start = process_index * loader.batch_size
        identities = global_identities[local_start : local_start + loader.batch_size]
        samples = [dataset[item["dataset_index"]] for item in identities]
        if not samples:
            # A short final global batch can leave a host with padding only.
            samples = [dataset[global_identities[-1]["dataset_index"]]]
        samples.extend([samples[-1]] * (loader.batch_size - len(samples)))
        observation, actions, metrics = loader._openpi_batch(samples, rng)
        placeholders = host_numpy(actions)
        if not np.all(np.isfinite(placeholders)) or np.any(placeholders != 0):
            raise AssertionError("Validation received ground-truth trajectory actions.")
        metrics = sanitize(metrics)
        initial = initialize(
            jax.random.fold_in(jax.random.key(seed), batch_index), metrics
        )
        batches.append(ValidationBatch(observation, metrics, initial, global_identities))
        fingerprints.append(fingerprint(observation, metrics))
    return batches, fingerprints, selected


def rollout_model_only(
    initial_paths, field, project, *, steps, step_size, max_direction_rms,
    step_size_start=None,
):
    """Fixed-length integration using model output and kinematics only.

    Track nonfinite model output before sanitization; an invalid rollout cannot
    later become a success by projecting NaNs into a plausible path.
    """

    waypoint_steps = waypoint_step_sizes(initial_paths, step_size, step_size_start)

    def body(_, carry):
        paths, valid = carry
        raw = field(paths)
        finite = jnp.all(jnp.isfinite(raw), axis=(-2, -1))
        direction = jnp.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        direction = direction.at[:, 0].set(0.0)
        rms = jnp.sqrt(jnp.mean(jnp.square(direction[:, 1:]), axis=(-2, -1)) + 1e-12)
        scale = jnp.minimum(1.0, max_direction_rms / jnp.maximum(rms, 1e-6))
        direction = jnp.where(finite[:, None, None], direction, 0.0)
        proposed = project(paths + waypoint_steps * direction * scale[:, None, None])
        return proposed, valid & finite & jnp.all(jnp.isfinite(proposed), axis=(-2, -1))

    return jax.lax.fori_loop(
        0, steps, body, (initial_paths, jnp.ones(initial_paths.shape[0], dtype=bool))
    )


def score_validation_paths(
    paths, metrics, rollout_valid, *, energy_config, progress_threshold
):
    """Dense geometry scoring with the standalone evaluator's success semantics."""

    def score_one(path, context, valid):
        context = jax.tree.map(lambda value: value[None], context)
        _, info = compute_obstacle_energy(path[None], context, energy_config)
        xy, _ = decode_bounded_waypoint_path(path[None], context, energy_config)
        rows, cols = context["esdf"].shape[-2:]
        resolution = context["esdf_resolution"].reshape(-1)[0]
        lower = jnp.stack(
            [context["esdf_x_min"].reshape(-1)[0], context["esdf_y_min"].reshape(-1)[0]]
        )
        goal = jnp.clip(
            context["goal_xy"][0],
            lower + 0.5 * resolution,
            lower + jnp.asarray([cols - 1, rows - 1]) * resolution,
        )
        distances = jnp.linalg.norm(xy[0] - goal, axis=-1)
        retreat = jnp.maximum(distances[1:] - distances[:-1], 0.0)
        geometry_ok = (
            valid
            & (info["action_finite_rate"] >= 1.0 - 1e-6)
            & (info["invalid_esdf_rate"] <= 1e-6)
            & (info["all_invalid_traj_rate"] < 0.5)
        )
        progress = info["achieved_required_progress_ratio"]
        reached = progress >= progress_threshold
        return {
            "success_rate": (
                geometry_ok & reached & (info["collision_rate"] < 0.5)
            ).astype(jnp.float32),
            "safe_success_rate": (
                geometry_ok & reached & (info["unsafe_rate"] < 0.5)
            ).astype(jnp.float32),
            "progress_ratio": progress,
            "bounded_progress_ratio": jnp.clip(progress, -1.0, 1.0),
            "collision_rate": info["collision_rate"],
            "clearance_violation_rate": info["unsafe_rate"],
            "invalid_coverage_rate": info["invalid_esdf_rate"],
            "goal_retreat_segment_rate": jnp.mean((retreat > 1e-6).astype(jnp.float32)),
            "goal_retreat_m": jnp.sum(retreat),
            "rollout_valid": valid.astype(jnp.float32),
        }

    return jax.vmap(score_one)(paths, metrics, rollout_valid)


def summarize_validation(records):
    """Equal-clip means; padding is already excluded from the input records."""
    by_clip = defaultdict(list)
    for record in records:
        by_clip[record["clip_id"]].append(record)
    keys = (
        *VALIDATION_METRICS,
        "bounded_progress_ratio",
        "goal_retreat_m",
        "rollout_valid",
    )
    return {
        key: float(
            np.mean(
                [
                    np.mean([record[key] for record in group])
                    for group in by_clip.values()
                ]
            )
        )
        for key in keys
    }


def evaluate_validation_pool(
    state, batches, rollout, score, *, output_dir, step, plot_count, rollout_steps=None
):
    """Restart all examples from their fixed priors; persist diagnostics and PNGs."""
    primary = jax.process_index() == 0
    directory = Path(output_dir) / f"step_{step:08d}"
    if primary:
        directory.mkdir(parents=True, exist_ok=True)
    records, plot_paths = [], []
    plotted_clips = set()
    for batch in batches:
        paths, rollout_valid = rollout(
            state, batch.observation, batch.initial_paths, batch.metrics
        )
        scores = jax.tree.map(host_numpy, score(paths, batch.metrics, rollout_valid))
        physical = host_numpy(paths)
        initial = host_numpy(batch.initial_paths)
        # Every process participates in gathers, even though only rank zero plots.
        plot_context = jax.tree.map(host_numpy, batch.metrics) if plot_count else None
        for index, identity in enumerate(batch.identities):
            record = {
                **identity,
                **{key: float(value[index]) for key, value in scores.items()},
            }
            if rollout_steps is not None:
                record["rollout_steps"] = int(rollout_steps)
            records.append(record)
            if primary and (
                len(plot_paths) < plot_count
                and identity["clip_id"] not in plotted_clips
            ):
                context = jax.tree.map(lambda value: value[index], plot_context)
                figure = path_overlay_figure(
                    initial[index], physical[index], context, record
                )
                plot_path = directory / f"path_{len(plot_paths):02d}.png"
                figure.savefig(plot_path, dpi=130)
                import matplotlib.pyplot as plt

                plt.close(figure)
                plot_paths.append(plot_path)
                plotted_clips.add(identity["clip_id"])
    summary = summarize_validation(records)
    payload = {"optimizer_step": step, "summary": summary, "examples": records}
    if rollout_steps is not None:
        payload["rollout_steps"] = int(rollout_steps)
    if primary:
        (directory / "metrics.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n"
        )
    return summary, plot_paths


def path_overlay_figure(initial, final, context, record):
    """ESDF overlay; pass ``final=None`` to render only the starting prior."""
    from matplotlib.patches import Circle
    import matplotlib.pyplot as plt

    grid = np.asarray(context["esdf"])
    resolution = float(np.asarray(context["esdf_resolution"]).item())
    x_min, y_min = (
        float(np.asarray(context[key]).item()) for key in ("esdf_x_min", "esdf_y_min")
    )
    extent = [
        x_min,
        x_min + grid.shape[1] * resolution,
        y_min,
        y_min + grid.shape[0] * resolution,
    ]
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    color_scale = esdf_display_scale(grid)
    backdrop = ax.imshow(
        grid,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        vmin=-color_scale,
        vmax=color_scale,
    )
    fig.colorbar(backdrop, ax=ax, label="ESDF [m]")
    ax.plot(initial[:, 0], initial[:, 1], "w--", linewidth=2, label="prior")
    if final is not None:
        ax.plot(
            final[:, 0],
            final[:, 1],
            "o-",
            color="tab:red",
            markersize=3,
            label=(
                f"model only ({record['rollout_steps']} adjustments)"
                if "rollout_steps" in record else "model only"
            ),
        )
        delta = np.diff(final, axis=0)
        ax.quiver(
            final[:-1, 0],
            final[:-1, 1],
            delta[:, 0],
            delta[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="tab:red",
            width=0.004,
        )
    radius = float(np.asarray(context.get("robot_radius_m", 0.5)).item())
    ax.add_patch(Circle((0, 0), radius, fill=False, color="white", linewidth=1))
    ax.plot(0, 0, "wo", markeredgecolor="black", label="robot")
    goal = np.asarray(context["goal_xy"])
    ax.plot(goal[0], goal[1], "*", color="yellow", markersize=14, label="goal")
    ax.set(
        xlim=(min(extent[0], -2.0), extent[1]),
        ylim=extent[2:],
        aspect="equal",
        xlabel="local x [m]",
        ylabel="local y [m]",
        title=f"{record['clip_id']} / {record['sample_id']}\n"
        f"progress={record['progress_ratio']:.2f}, collision={record['collision_rate']:.0f}, "
        f"invalid={record['invalid_coverage_rate']:.0%}",
    )
    ax.legend(loc="upper right")
    return fig
