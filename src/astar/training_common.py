"""Shared helpers for ASTAR pi0.5 navigation training scripts."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import logging

import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import wandb

from astar.adapters import waypoint_goal_projector_factory
from astar.dataloader import create_navigation_data_loader as _create_astar_navigation_data_loader
from astar.esdf import robot_radius_per_path
from astar.esdf import sample_batched_esdf as sample_batched_esdf
from astar.esdf import sample_batched_esdf_with_valid as sample_batched_esdf_with_valid
from astar.heads import pi05_waypoint_action_adapter_factory
from astar.path_sampler import maybe_unnormalize_actions

PI05_BASE_PARAMS = "./checkpoints/pi05_base/params"


def create_navigation_data_loader(config: _config.TrainConfig, **kwargs):
    """Create an ASTAR navigation loader used by flow and energy training."""
    return _create_astar_navigation_data_loader(config, **kwargs)


def navigation_loader_kwargs(args: argparse.Namespace, data_sharding) -> dict[str, object]:
    """Shared CLI-to-dataloader wiring for the ASTAR navigation scripts."""
    kwargs = {
        "data_root": args.data_root,
        "manifest_path": args.manifest_path,
        "sharding": data_sharding,
        "sampled_goal_fraction": args.sampled_goal_fraction,
        "object_text_goal_prob": args.object_text_goal_prob,
        "object_image_goal_prob": args.object_image_goal_prob,
        "object_waypoint_goal_prob": args.object_waypoint_goal_prob,
    }
    if hasattr(args, "path_stride"):
        kwargs["path_stride"] = args.path_stride
    return kwargs


def init_logging() -> None:
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s "
        "(%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers[0].setFormatter(formatter)


def freeze_filter(train_scope: str) -> nnx.filterlib.Filter:
    """Build the freeze filter for common ASTAR fine-tuning scopes."""
    match train_scope:
        case "all":
            return nnx.Nothing
        case "adapter":
            return nnx.Not(nnx_utils.PathRegex(".*(action_adapter|goal_adapter).*"))
        case "action_stack":
            return nnx.Not(nnx_utils.PathRegex(".*(action_adapter|goal_adapter|llm.*_1).*"))
        case _:
            raise ValueError(f"Unknown train scope: {train_scope}")


def split_batch(
    batch: tuple[_model.Observation, _model.Actions] | tuple[
        _model.Observation,
        _model.Actions,
        dict[str, at.Array],
    ],
) -> tuple[_model.Observation, _model.Actions, dict[str, at.Array]]:
    if len(batch) == 2:
        observation, actions = batch
        return observation, actions, {}
    observation, actions, metric_tensors = batch
    return observation, actions, metric_tensors


def wrap_angle(angle: at.Array) -> at.Array:
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def goal_condition_metrics(metric_tensors: dict[str, at.Array]) -> dict[str, at.Array]:
    """Summarize which goal modalities the batch exposed to pi0.5."""
    metrics = {}
    for key, output_name in (
        ("goal_text_condition_mask", "goal_text_rate"),
        ("goal_image_condition_mask", "goal_image_rate"),
        ("goal_waypoint_condition_mask", "goal_waypoint_rate"),
    ):
        if key in metric_tensors:
            metrics[output_name] = jnp.mean(jnp.asarray(metric_tensors[key], dtype=jnp.float32))
    return metrics


def compute_navigation_eval_metrics(
    pred_actions: _model.Actions,
    target_actions: _model.Actions,
    metric_tensors: dict[str, at.Array],
    *,
    esdf_blindspot_x_m: float = 2.0,
) -> dict[str, at.Array]:
    """Compute held-out trajectory metrics for 4D waypoint actions."""
    pred_actions = maybe_unnormalize_actions(pred_actions, metric_tensors)
    target_actions = maybe_unnormalize_actions(target_actions, metric_tensors)

    pred_xy = pred_actions[..., :2]
    target_xy = target_actions[..., :2]
    path_mask = metric_tensors.get("path_step_mask")
    if path_mask is None:
        path_mask = jnp.ones(pred_xy.shape[:2], dtype=bool)
    else:
        path_mask = jnp.asarray(path_mask, dtype=bool)

    waypoint_l2 = jnp.linalg.norm(pred_xy - target_xy, axis=-1)
    valid_count = jnp.maximum(jnp.sum(path_mask, axis=-1), 1)
    waypoint_l2_mean = jnp.sum(jnp.where(path_mask, waypoint_l2, 0.0)) / jnp.maximum(
        jnp.sum(path_mask),
        1,
    )

    batch_indices = jnp.arange(pred_xy.shape[0])
    final_indices = valid_count.astype(jnp.int32) - 1
    pred_final_xy = pred_xy[batch_indices, final_indices]
    target_final_xy = target_xy[batch_indices, final_indices]
    final_waypoint_error = jnp.linalg.norm(pred_final_xy - target_final_xy, axis=-1)

    pred_heading = jnp.arctan2(pred_actions[..., 3], pred_actions[..., 2])
    target_heading = jnp.arctan2(target_actions[..., 3], target_actions[..., 2])
    heading_error = jnp.abs(wrap_angle(pred_heading - target_heading))
    heading_error_mean = jnp.sum(jnp.where(path_mask, heading_error, 0.0)) / jnp.maximum(
        jnp.sum(path_mask),
        1,
    )

    metrics = {
        "waypoint_l2": waypoint_l2_mean,
        "final_goal_error": jnp.mean(final_waypoint_error),
        "final_waypoint_error": jnp.mean(final_waypoint_error),
        "heading_error": heading_error_mean,
        "final_heading_error": jnp.mean(heading_error[batch_indices, final_indices]),
    }
    if "goal_xy" in metric_tensors:
        goal_xy = metric_tensors["goal_xy"]
        metrics["full_goal_error"] = jnp.mean(jnp.linalg.norm(pred_final_xy - goal_xy, axis=-1))
        metrics["target_full_goal_error"] = jnp.mean(jnp.linalg.norm(target_final_xy - goal_xy, axis=-1))

    if "collision_esdf_energy" in metric_tensors:
        metrics["collision_esdf_energy"] = jnp.mean(metric_tensors["collision_esdf_energy"])
    if {"esdf", "esdf_x_min", "esdf_y_min", "esdf_resolution"}.issubset(metric_tensors):
        pred_esdf, esdf_valid = sample_batched_esdf_with_valid(
            metric_tensors["esdf"],
            pred_xy,
            metric_tensors["esdf_x_min"],
            metric_tensors["esdf_y_min"],
            metric_tensors["esdf_resolution"],
        )
        esdf_finite = jnp.isfinite(pred_esdf)
        esdf_sample_ok = esdf_valid & esdf_finite
        robot_radius = robot_radius_per_path(
            metric_tensors, pred_xy.shape[0], pred_xy.dtype
        )[:, None]
        pred_esdf_for_metrics = jnp.where(esdf_sample_ok, pred_esdf, -jnp.inf)
        clearance = pred_esdf_for_metrics - robot_radius
        after_blindspot = pred_xy[..., 0] > esdf_blindspot_x_m
        after_blindspot_ok = esdf_sample_ok & after_blindspot
        clearance_after_blindspot = jnp.where(after_blindspot_ok, clearance, jnp.inf)
        esdf_after_blindspot = jnp.where(after_blindspot_ok, pred_esdf, jnp.inf)
        has_after_blindspot = jnp.any(after_blindspot_ok, axis=-1)
        after_count = jnp.maximum(jnp.sum(has_after_blindspot.astype(jnp.float32)), 1.0)

        collision_per_point = clearance < 0.0
        collision_after_blindspot_per_point = after_blindspot_ok & (clearance < 0.0)
        metrics["collision_esdf_energy"] = jnp.mean(jnp.square(jnp.maximum(-clearance, 0.0)))
        metrics["collision_rate"] = jnp.mean(jnp.any(collision_per_point, axis=-1).astype(jnp.float32))
        metrics["collision_rate_all"] = metrics["collision_rate"]
        metrics["collision_rate_after_blindspot"] = jnp.mean(
            jnp.any(collision_after_blindspot_per_point, axis=-1).astype(jnp.float32)
        )
        metrics["min_esdf_m"] = jnp.mean(jnp.min(pred_esdf_for_metrics, axis=-1))
        metrics["min_clearance_m"] = jnp.mean(jnp.min(clearance, axis=-1))
        metrics["min_esdf_after_blindspot_m"] = (
            jnp.sum(jnp.where(has_after_blindspot, jnp.min(esdf_after_blindspot, axis=-1), 0.0))
            / after_count
        )
        metrics["min_clearance_after_blindspot_m"] = (
            jnp.sum(
                jnp.where(
                    has_after_blindspot,
                    jnp.min(clearance_after_blindspot, axis=-1),
                    0.0,
                )
            )
            / after_count
        )
        metrics["esdf_valid_rate"] = jnp.mean(esdf_valid.astype(jnp.float32))
        metrics["esdf_finite_rate"] = jnp.mean(esdf_finite.astype(jnp.float32))
    elif "pred_esdf_along_path_m" in metric_tensors:
        pred_esdf = metric_tensors["pred_esdf_along_path_m"]
        metrics["collision_esdf_energy"] = jnp.mean(jnp.square(jnp.maximum(-pred_esdf, 0.0)))
        metrics["collision_rate"] = jnp.mean(jnp.any(pred_esdf < 0.0, axis=-1).astype(jnp.float32))
        metrics["min_esdf_m"] = jnp.mean(jnp.min(pred_esdf, axis=-1))

    return metrics


def create_train_config(args: argparse.Namespace) -> _config.TrainConfig:
    use_goal_waypoint_adapter = getattr(args, "use_goal_waypoint_adapter", True)
    model = pi0_config.Pi0Config(
        pi05=True,
        action_dim=getattr(args, "action_dim", 4),
        action_horizon=args.action_horizon,
        action_adapter_factory=pi05_waypoint_action_adapter_factory,
        goal_adapter_factory=waypoint_goal_projector_factory if use_goal_waypoint_adapter else None,
        goal_waypoint_dim=getattr(args, "goal_waypoint_dim", 2),
        max_goal_waypoints=getattr(args, "max_goal_waypoints", 1),
        discrete_state_input=args.discrete_state_input,
    )
    return _config.TrainConfig(
        name=args.config_name,
        project_name=args.project_name,
        exp_name=args.exp_name,
        model=model,
        weight_loader=_weight_loaders.CheckpointWeightLoader(args.pretrained_params),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=args.warmup_steps,
            peak_lr=args.peak_lr,
            decay_steps=args.decay_steps,
            decay_lr=args.decay_lr,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=args.clip_gradient_norm),
        ema_decay=args.ema_decay,
        freeze_filter=freeze_filter(args.train_scope),
        batch_size=args.batch_size,
        num_train_steps=args.num_train_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        keep_period=args.keep_period,
        overwrite=args.overwrite,
        resume=args.resume,
        wandb_enabled=args.wandb_enabled,
        fsdp_devices=args.fsdp_devices,
        assets_base_dir=args.assets_base_dir,
        checkpoint_base_dir=args.checkpoint_base_dir,
    )


ACTION_STACK_PRETRAIN_SKIP_PATTERNS = (
    "PaliGemma/llm/final_norm_1/",
    "PaliGemma/llm/layers/attn/attn_vec_einsum_1/",
    "PaliGemma/llm/layers/attn/kv_einsum_1/",
    "PaliGemma/llm/layers/attn/q_einsum_1/",
    "PaliGemma/llm/layers/mlp_1/",
    "PaliGemma/llm/layers/pre_attention_norm_1/",
    "PaliGemma/llm/layers/pre_ffw_norm_1/",
)


def _param_path_to_string(path: tuple[object, ...]) -> str:
    return "/".join(str(part) for part in path) + "/"


def _load_weights_and_validate(
    loader: _weight_loaders.WeightLoader,
    params_shape: at.Params,
    *,
    skip_patterns: tuple[str, ...] = (),
) -> at.Params:
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(
        expected=params_shape,
        got=loaded_params,
        check_shapes=True,
        check_dtypes=True,
    )
    return traverse_util.unflatten_dict(
        {
            k: v
            for k, v in traverse_util.flatten_dict(loaded_params).items()
            if not isinstance(v, jax.ShapeDtypeStruct)
            and not any(pattern in _param_path_to_string(k) for pattern in skip_patterns)
        }
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
    skip_pretrained_load_patterns: tuple[str, ...] = (),
) -> tuple[training_utils.TrainState, object]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(
        rng: at.KeyArrayLike,
        partial_params: at.Params | None = None,
    ) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(
            params,
            config.freeze_filter,
            lambda p: p.replace(p.value.astype(jnp.bfloat16)),
        )

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(
        config.weight_loader,
        train_state_shape.params.to_pure_dict(),
        skip_patterns=skip_pretrained_load_patterns,
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def run_eval(
    peval_step,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    eval_iter: Iterator,
    *,
    num_batches: int,
    step: int,
) -> dict[str, float]:
    infos = []
    for batch_index in range(num_batches):
        eval_rng = jax.random.fold_in(rng, step * num_batches + batch_index)
        infos.append(peval_step(eval_rng, state, next(eval_iter)))

    stacked_infos = common_utils.stack_forest(infos)
    reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
    return {f"eval/{key}": float(value) for key, value in reduced_info.items()}


def first_n_sample_tree(tree, batch_size: int, num_samples: int):
    return jax.tree.map(
        lambda x: x[:num_samples]
        if hasattr(x, "shape") and len(x.shape) > 0 and x.shape[0] == batch_size
        else x,
        tree,
    )


def first_sample_tree(tree, batch_size: int):
    return first_n_sample_tree(tree, batch_size, 1)


def make_eval_path_wandb_image(
    pred_actions: at.Array,
    metric_tensors: dict[str, at.Array],
    *,
    title: str,
    front_image: at.Array | None = None,
    prior_actions: at.Array | None = None,
    target_actions: at.Array | None = None,
    show_target_path: bool = True,
):
    if not {"esdf", "esdf_x_min", "esdf_y_min", "esdf_resolution"}.issubset(metric_tensors):
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib is not installed; skipping eval path plot.")
        return None

    pred_xy = np.asarray(pred_actions[0, :, :2], dtype=np.float32)
    prior_xy = (
        np.asarray(prior_actions[0, :, :2], dtype=np.float32)
        if prior_actions is not None
        else None
    )
    target_xy = None
    if show_target_path:
        if "path_xy" in metric_tensors:
            target_xy = np.asarray(metric_tensors["path_xy"][0], dtype=np.float32)
        elif target_actions is not None:
            target_xy = np.asarray(target_actions[0, :, :2], dtype=np.float32)

    if show_target_path and target_xy is not None and "path_step_mask" in metric_tensors:
        path_mask = np.asarray(metric_tensors["path_step_mask"][0], dtype=bool)
        if path_mask.any():
            pred_xy = pred_xy[path_mask]
            if prior_xy is not None:
                prior_xy = prior_xy[path_mask]
            target_xy = target_xy[path_mask]
    goal_xy = (
        np.asarray(metric_tensors["goal_xy"][0], dtype=np.float32)
        if "goal_xy" in metric_tensors
        else None
    )
    esdf = np.asarray(metric_tensors["esdf"][0], dtype=np.float32)
    x_min = float(np.asarray(metric_tensors["esdf_x_min"])[0])
    y_min = float(np.asarray(metric_tensors["esdf_y_min"])[0])
    resolution = float(np.asarray(metric_tensors["esdf_resolution"])[0])
    robot_radius = float(
        robot_radius_per_path(metric_tensors, pred_actions.shape[0], jnp.float32)[0]
    )

    rows, cols = esdf.shape[-2:]
    extent = [x_min, x_min + cols * resolution, y_min, y_min + rows * resolution]
    finite = np.isfinite(esdf)
    esdf_for_plot = np.where(finite, esdf, np.nan)

    if front_image is not None:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12.5, 5.8),
            dpi=140,
            gridspec_kw={"width_ratios": [1.05, 1.35]},
        )
        image_ax, ax = axes
        front_image = np.asarray(front_image[0], dtype=np.float32)
        if front_image.ndim == 3 and front_image.shape[0] in (1, 3) and front_image.shape[-1] not in (1, 3):
            front_image = np.moveaxis(front_image, 0, -1)
        if front_image.dtype.kind == "f":
            if front_image.min() < 0.0:
                front_image = front_image * 0.5 + 0.5
            front_image = np.clip(front_image, 0.0, 1.0)
        image_ax.imshow(front_image)
        image_ax.set_title("front camera")
        image_ax.axis("off")
    else:
        fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=140)

    finite_values = esdf[finite]
    color_scale = max(
        float(np.percentile(np.abs(finite_values), 99))
        if finite_values.size
        else 1.0,
        1.0,
    )
    image = ax.imshow(
        esdf_for_plot,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        vmin=-color_scale,
        vmax=color_scale,
        interpolation="nearest",
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="ESDF phi(x, y) [m]")
    if finite.any():
        ax.contour(
            esdf_for_plot,
            levels=[robot_radius],
            origin="lower",
            extent=extent,
            colors="black",
            linewidths=0.9,
        )

    if show_target_path and target_xy is not None:
        ax.plot(target_xy[:, 0], target_xy[:, 1], color="#39d353", linewidth=2.2, label="dataset path")
        ax.scatter(
            target_xy[0, 0],
            target_xy[0, 1],
            color="white",
            edgecolor="black",
            s=40,
            zorder=5,
            label="start",
        )
        ax.scatter(
            target_xy[-1, 0],
            target_xy[-1, 1],
            color="#39d353",
            edgecolor="black",
            s=48,
            zorder=5,
            label="target end",
        )
    else:
        ax.scatter(0.0, 0.0, color="white", edgecolor="black", s=40, zorder=5, label="start")
    if goal_xy is not None:
        ax.scatter(
            goal_xy[0],
            goal_xy[1],
            color="#39d353",
            edgecolor="black",
            s=54,
            zorder=5,
            label="goal",
        )
    if prior_xy is not None:
        ax.plot(
            prior_xy[:, 0],
            prior_xy[:, 1],
            color="#8b949e",
            linewidth=1.8,
            linestyle="--",
            label="sampled prior",
        )
    ax.plot(pred_xy[:, 0], pred_xy[:, 1], color="#ff4d6d", linewidth=2.2, label="predicted path")
    ax.scatter(pred_xy[-1, 0], pred_xy[-1, 1], color="#ff4d6d", edgecolor="black", s=48, zorder=5, label="pred end")
    ax.set_title("ESDF + path")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.18)
    fig.suptitle(title)
    fig.tight_layout()

    image = wandb.Image(fig)
    plt.close(fig)
    return image


def eval_path_plot_for_wandb(
    ppredict_eval_actions_step,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions] | tuple[
        _model.Observation,
        _model.Actions,
        dict[str, at.Array],
    ],
    *,
    step: int,
    title_prefix: str,
    show_target_path: bool = True,
    num_images: int = 1,
):
    predicted = ppredict_eval_actions_step(rng, state, batch)
    if isinstance(predicted, tuple):
        pred_actions, prior_actions = predicted
    else:
        pred_actions = predicted
        prior_actions = None
    observation, target_actions, metric_tensors = split_batch(batch)
    if target_actions is not None:
        batch_size = int(target_actions.shape[0])
    else:
        batch_size = int(metric_tensors["goal_xy"].shape[0])
    num_images = max(1, min(int(num_images), batch_size))
    observation = first_n_sample_tree(observation, batch_size, num_images)
    metric_tensors = first_n_sample_tree(metric_tensors, batch_size, num_images)
    pred_actions = maybe_unnormalize_actions(pred_actions[:num_images], metric_tensors)
    prior_actions = (
        maybe_unnormalize_actions(prior_actions[:num_images], metric_tensors)
        if prior_actions is not None
        else None
    )
    target_actions = (
        maybe_unnormalize_actions(target_actions[:num_images], metric_tensors)
        if show_target_path
        else None
    )

    pred_actions = jax.device_get(pred_actions)
    prior_actions = jax.device_get(prior_actions) if prior_actions is not None else None
    metric_tensors = jax.device_get(metric_tensors)
    front_images = jax.device_get(observation.images.get("base_0_rgb"))
    target_actions = jax.device_get(target_actions) if target_actions is not None else None

    images = []
    for sample_index in range(num_images):
        sample_metrics = jax.tree.map(
            lambda x: x[sample_index : sample_index + 1]
            if hasattr(x, "shape") and len(x.shape) > 0 and x.shape[0] == num_images
            else x,
            metric_tensors,
        )
        image = make_eval_path_wandb_image(
            pred_actions[sample_index : sample_index + 1],
            sample_metrics,
            title=f"{title_prefix} step {step} sample {sample_index}",
            front_image=front_images[sample_index : sample_index + 1] if front_images is not None else None,
            prior_actions=prior_actions[sample_index : sample_index + 1] if prior_actions is not None else None,
            target_actions=target_actions[sample_index : sample_index + 1] if target_actions is not None else None,
            show_target_path=show_target_path,
        )
        if image is not None:
            images.append(image)
    return images or None
