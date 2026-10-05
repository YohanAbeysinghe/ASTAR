"""Continue a checkpoint on the full train split with a frozen, bounded-depth teacher.

Every optimizer update draws a fresh batch from a shuffled full-partition epoch.
The source model generates a candidate at a sampled depth in [0, depth_cap).
An analytic oracle labels that candidate; SGD changes only the student. Neither
the teacher nor the maximum rollout depth changes. Deterministic batch seeds
allow regeneration after interruption without caching millions of images/maps.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import gc
import json
import logging
import math
from pathlib import Path
import signal
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi.shared import array_typing as at
import openpi.training.checkpoints as checkpoints
import openpi.training.sharding as sharding
import optax
import orbax.checkpoint as ocp
import wandb

from astar.energy_checkpoints import initialize_bounded_checkpoint_dir
from astar.evaluation.aggregation import build_energy_config
from astar.evaluation.aggregation import restore_checkpoint_params
from astar.training import aggregation as agg
from astar.training.continuation_data import FullPartitionBatches
from astar.training.continuation_data import dataset_resume_metadata
from astar.training.continuation_data import deterministic_batch_seed
from astar.training.continuation_data import metadata_signature
from astar.training.continuation_wandb import init_continuation_wandb
from astar.training.monitoring import cache_validation_pool
from astar.training.monitoring import evaluate_validation_pool
from astar.training.monitoring import host_numpy
from astar.training.monitoring import learning_metrics
from astar.training.monitoring import rollout_model_only
from astar.training.monitoring import score_validation_paths


def restore_train_state(root, step, shape, state_sharding):
    """Restore raw model, Adam state/counter and EMA, without opening a writer."""
    # The abstract shape and sharding carry ShapeDtypeStruct/NamedSharding
    # leaves, which are deliberately outside TrainState's runtime annotations.
    with at.disable_typechecking():
        raw_shape = dataclasses.replace(shape, ema_params=None)
        raw_sharding = dataclasses.replace(state_sharding, ema_params=None)
        if shape.ema_params is None:
            raw_shape = dataclasses.replace(raw_shape, params={})
            raw_sharding = dataclasses.replace(raw_sharding, params={})
    restore_args = jax.tree.map(
        lambda _, s: ocp.ArrayRestoreArgs(restore_type=jax.Array, sharding=s),
        raw_shape, raw_sharding,
    )
    with ocp.PyTreeCheckpointer() as reader:
        raw = reader.restore(
            Path(root) / str(step) / "train_state",
            args=ocp.args.PyTreeRestore(item=raw_shape, restore_args=restore_args),
        )
    params = restore_checkpoint_params(
        Path(root), step=step, source="ema",
        params_shape=shape.ema_params if shape.ema_params is not None else shape.params,
        params_sharding=state_sharding.ema_params if shape.ema_params is not None else state_sharding.params,
    )
    with at.disable_typechecking():
        restored = dataclasses.replace(raw, ema_params=params) if shape.ema_params is not None else (
            dataclasses.replace(raw, params=params)
        )
    jax.block_until_ready(restored)
    if int(restored.step) != step:
        raise ValueError(f"Checkpoint directory {step} contains optimizer step {int(restored.step)}")
    return restored


def depth_for_batch(seed, batch_index, depth_cap):
    """Equal frequency of initial/intermediate depths; reproducible after resume."""
    block, position = divmod(batch_index, depth_cap)
    rng = np.random.default_rng(deterministic_batch_seed(seed, block, stream=3))
    return int(rng.permutation(depth_cap)[position])


def goal_weights(is_object, valid, *, object_fraction, target_sampled_fraction):
    """Enumerate every goal while retaining the source run's goal-type objective."""
    if not 0 < object_fraction < 1:
        return jnp.asarray(valid, jnp.float32)
    weights = jnp.where(
        is_object,
        (1 - target_sampled_fraction) / object_fraction,
        target_sampled_fraction / (1 - object_fraction),
    )
    return weights * valid


def train_step(config, field_time, state, observation, candidates, directions, weights, valid):
    """Weighted field regression; oracle computation is outside this graph."""
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def objective(model):
        predicted = agg.predict_descent_field(model, observation, candidates, field_time=field_time)
        predicted = predicted.at[:, 0].set(0)
        per_path = 0.5 * jnp.mean(jnp.square(predicted[:, 1:] - directions[:, 1:]), axis=(-2, -1))
        baseline = 0.5 * jnp.mean(jnp.square(directions[:, 1:]), axis=(-2, -1))
        count = jnp.maximum(jnp.sum(valid), 1)
        return jnp.sum(weights * per_path) / count, {
            "zero_predictor_loss": jnp.sum(weights * baseline) / count,
            "field_oracle_cosine": agg._field_oracle_cosine(predicted * valid[:, None, None],
                                                            directions * valid[:, None, None]),
        }

    (loss, info), grads = nnx.value_and_grad(
        objective, argnums=nnx.DiffState(0, config.trainable_filter), has_aux=True,
    )(model)
    params = state.params.filter(config.trainable_filter)
    updates, opt_state = state.tx.update(grads, state.opt_state, params)
    nnx.update(model, optax.apply_updates(params, updates))
    params = nnx.state(model)
    ema = None if state.ema_params is None else jax.tree.map(
        lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
        state.ema_params, params,
    )
    return dataclasses.replace(state, step=state.step + 1, params=params, opt_state=opt_state, ema_params=ema), {
        **info, "loss": loss, "grad_norm": optax.global_norm(grads),
    }


def make_rollout(model_def, values, energy_config):
    def rollout(params, observation, initial, metrics, depth):
        model = nnx.merge(model_def, params)
        model.eval()
        return rollout_model_only(
            initial,
            lambda paths: agg.predict_descent_field(model, observation, paths, field_time=values["field_time"]),
            lambda paths: agg.project_bounded_actions(paths, metrics, energy_config),
            steps=depth, step_size=values["aggregation_step_size"],
            step_size_start=values["aggregation_step_size_start"],
            max_direction_rms=values["max_field_direction_rms"],
        )
    return rollout


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def source_configuration(args):
    source = args.source_root.resolve()
    output = args.output_dir.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Continuation output must be separate from the immutable source checkpoint")
    values = json.loads((source / "monitoring/configuration.json").read_text())
    if values["train_particles"] != 1:
        raise ValueError("Full-partition continuation currently supports one particle per condition")
    if not (source / str(args.source_step) / "_CHECKPOINT_METADATA").is_file():
        raise FileNotFoundError(f"Source checkpoint {args.source_step} is not committed")
    archive_path = source / f"aggregation_state_{args.source_step:08d}.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata_json"].item()))
        if meta["update_in_round"] or meta["replay_updates"]:
            raise ValueError("Select a source checkpoint saved after a completed aggregation round")
        if meta["round_index"] * values["updates_per_round"] != args.source_step:
            raise ValueError("Source replay and checkpoint step disagree")
        if meta["round_index"] != args.depth_cap:
            raise ValueError("Depth cap must match the selected completed source round")
    return values


def main(args):
    agg.init_logging()
    if jax.process_count() != 1:
        raise ValueError("This continuation supports a single process with multiple GPUs")
    values = source_configuration(args)
    run_dir = args.output_dir.resolve()
    source_root = args.source_root.resolve()
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"{run_dir} already exists; use --resume")
    # Preserve optimizer, LR, architecture and oracle settings from the source.
    train_args = argparse.Namespace(**values)
    train_args.config_name = run_dir.parent.name
    train_args.checkpoint_base_dir = str(run_dir.parent.parent)
    train_args.exp_name = run_dir.name
    train_args.batch_size = args.batch_size
    train_args.fsdp_devices = args.fsdp_devices
    train_args.wandb_enabled = args.wandb_enabled
    train_args.resume = args.resume
    train_args.overwrite = False
    train_args.save_interval = args.save_interval
    train_args.log_interval = args.log_interval
    train_args.eval_batches = args.eval_batches or values["eval_batches"]
    config = agg.create_train_config(train_args)
    if config.batch_size % jax.device_count():
        raise ValueError("Batch size must be divisible by the device count")
    energy = build_energy_config(values)
    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    shape, state_sharding = agg.init_train_state(config, jax.random.key(config.seed), mesh, resume=True)
    train_loader = agg.create_oracle_only_data_loader(config, train_args, data_sharding, split="train", num_batches=1)
    eval_loader = agg.create_oracle_only_data_loader(config, train_args, data_sharding, split="eval", num_batches=1)
    dataset = train_loader.dataset
    sampler = FullPartitionBatches(len(dataset), args.batch_size, args.seed)
    object_count = len(dataset.object_indices)
    object_fraction = object_count / len(dataset)
    # Original batch16 used round(16/6)=3 sampled goals. Preserve that objective
    # with importance weights, without discarding any goals from the full split.
    target_sampled_fraction = round(values["batch_size"] * values["sampled_goal_fraction"]) / values["batch_size"]
    dataset_metadata = dataset_resume_metadata(
        manifest_path=dataset.samples_path,
        train_split_ids_path=dataset._split_ids_path,
        eval_split_ids_path=eval_loader.dataset._split_ids_path,
        dataset_size=len(dataset),
    )
    source_digest = metadata_signature(values)
    critical = {
        "algorithm": "frozen_teacher_full_partition_v1",
        "source_root": str(source_root), "source_step": args.source_step,
        "source_config_signature": source_digest, "depth_cap": args.depth_cap,
        "teacher_parameter_source": "raw", "teacher_rollout": "model_only",
        "training_depths": list(range(args.depth_cap)),
        "batch_size": args.batch_size, "fsdp_devices": args.fsdp_devices, "seed": args.seed,
        "dataset": dataset_metadata, "target_sampled_fraction": target_sampled_fraction,
        "eval_batches": train_args.eval_batches, "eval_depth": args.depth_cap,
    }
    signature = metadata_signature(critical)
    metadata_path = run_dir / "continuation.json"
    if args.resume:
        if not metadata_path.is_file():
            raise FileNotFoundError("Resume requires the continuation metadata")
        if json.loads(metadata_path.read_text())["signature"] != signature:
            raise ValueError("Source, dataset, depth or continuation settings changed on resume")
    manager, resuming = initialize_bounded_checkpoint_dir(run_dir, overwrite=False, resume=args.resume)
    target_updates = sampler.steps_per_epoch if args.one_epoch else args.additional_steps
    target_step = args.source_step + target_updates
    metadata = {
        **critical, "signature": signature, "source_wandb_run": args.source_wandb_run,
        "source_checkpoint": str(source_root / str(args.source_step)),
        "object_goal_count": object_count, "sampled_goal_count": len(dataset) - object_count,
        "steps_per_epoch": sampler.steps_per_epoch, "target_optimizer_step": target_step,
        "learning_rate": values["decay_lr"], "source_training_config": values,
    }
    _atomic_json(metadata_path, metadata)
    monitoring_dir = run_dir / "monitoring"
    monitoring_dir.mkdir(exist_ok=True)
    _atomic_json(monitoring_dir / "configuration.json", {**vars(train_args), "continuation": metadata})
    # Restore the teacher separately, so student donation cannot invalidate it.
    with sharding.set_mesh(mesh):
        teacher_params = restore_checkpoint_params(
            source_root, step=args.source_step, source="raw",
            params_shape=shape.params, params_sharding=state_sharding.params,
        )
        restore_root = run_dir if resuming else source_root
        restore_step = int(manager.latest_step()) if resuming else args.source_step
        state = restore_train_state(restore_root, restore_step, shape, state_sharding)
    start_step = int(state.step)
    if start_step < args.source_step or start_step > target_step:
        raise ValueError(f"Restored step {start_step} is outside [{args.source_step}, {target_step}]")
    logging.info("Restored optimizer step %d; frozen teacher=%d; depth cap=%d; train goals=%d; target=%d",
                 start_step, args.source_step, args.depth_cap, len(dataset), target_step)
    initialize = jax.jit(functools.partial(
        agg.initialize_prior_paths, energy_config=energy, base_batch_size=args.batch_size,
        particles=1, action_horizon=values["action_horizon"], action_dim=values["action_dim"],
    ), in_shardings=(replicated, data_sharding), out_shardings=data_sharding)
    rollout = jax.jit(make_rollout(shape.model_def, values, energy),
                      in_shardings=(state_sharding.params, data_sharding, data_sharding, data_sharding, replicated),
                      out_shardings=(data_sharding, data_sharding))
    oracle = jax.jit(functools.partial(
        agg.query_energy_oracle, energy_config=energy, base_batch_size=args.batch_size, particles=1,
        gradient_floor=values["oracle_gradient_floor"], max_direction_rms=values["max_field_direction_rms"],
        diversity_direction_weight=0, step_size=values["aggregation_step_size"],
        step_size_start=values["aggregation_step_size_start"], backtracks=values["rollout_backtracks"],
        backtrack_factor=values["backtrack_factor"], energy_tolerance=values["energy_increase_tolerance"],
    ), in_shardings=(data_sharding, data_sharding), out_shardings=(data_sharding, replicated))
    update = jax.jit(functools.partial(train_step, config, values["field_time"]),
                     in_shardings=(state_sharding, data_sharding, data_sharding, data_sharding,
                                   data_sharding, data_sharding),
                     out_shardings=(state_sharding, replicated), donate_argnums=(0,))
    dense = dataclasses.replace(energy, strict_esdf_coverage=True,
                                segment_samples=max(energy.segment_samples,
                                                    math.ceil(energy.max_step_length_m / values["eval_safety_spacing_m"])))
    score = jax.jit(functools.partial(score_validation_paths, energy_config=dense,
                                    progress_threshold=values["eval_progress_threshold"]),
                    in_shardings=(data_sharding, data_sharding, data_sharding), out_shardings=data_sharding)
    with sharding.set_mesh(mesh):
        validation, _, selected = cache_validation_pool(
            eval_loader, dataset._split_ids, num_batches=train_args.eval_batches, seed=values["eval_seed"],
            sanitize=agg.sanitize_metric_tensors, fingerprint=agg.condition_fingerprint, initialize=initialize,
        )
    _atomic_json(monitoring_dir / "selected_examples.json", selected)
    run = init_continuation_wandb(
        run_dir, metadata, enabled=args.wandb_enabled, source_run_path=args.source_wandb_run,
        source_optimizer_step=args.source_step, name=run_dir.name,
        resume=args.resume and (run_dir / "wandb_id.txt").is_file(),
    )
    stop_requested = False

    def stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        logging.warning("Signal %s: saving after the current update", signum)

    previous_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGUSR1, signal.SIGTERM)}

    def log(payload):
        payload = {"optimizer_step": int(state.step), "aggregation/round": args.depth_cap, **payload}
        run.log(payload)
        with (monitoring_dir / "training_diagnostics.jsonl").open("a") as handle:
            handle.write(json.dumps(payload, allow_nan=False) + "\n")

    def save():
        step = int(state.step)
        if step not in manager.all_steps():
            # Sidecar precedes the committed checkpoint; resume uses only a
            # sidecar belonging to a checkpoint with a completed Orbax marker.
            _atomic_json(run_dir / f"continuation_state_{step:08d}.json", {
                "signature": signature, "optimizer_step": step,
                "next_batch_index": step - args.source_step,
            })
            checkpoints.save_state(manager, state, train_loader, step)
            manager.wait_until_finished()
        logging.info("Saved continuation at optimizer step %d", step)

    def evaluate():
        with sharding.set_mesh(mesh):
            summary, plots = evaluate_validation_pool(
                state, validation,
                lambda s, o, p, m: rollout(s.params, o, p, m, jnp.asarray(args.depth_cap, jnp.int32)),
                score, output_dir=monitoring_dir, step=int(state.step),
                plot_count=values["eval_plot_count"], rollout_steps=args.depth_cap,
            )
        payload = {f"eval/model_only/{key}": value for key, value in summary.items()}
        payload["eval/model_only/rollout_steps"] = args.depth_cap
        log(payload)
        if plots:
            run.log({"optimizer_step": int(state.step),
                     "eval/model_only/paths": [wandb.Image(str(path)) for path in plots]})
        logging.info("Fixed-depth validation at step %d: %s", int(state.step), summary)

    try:
        if resuming:
            saved = json.loads((run_dir / f"continuation_state_{start_step:08d}.json").read_text())
            if saved != {"signature": signature, "optimizer_step": start_step,
                         "next_batch_index": start_step - args.source_step}:
                raise ValueError("Checkpoint and continuation sampler state disagree")
        save()
        evaluate()
        window, window_start = [], time.monotonic()
        for batch_index in range(start_step - args.source_step, target_updates):
            if stop_requested:
                break
            indices, valid_np = sampler.indices(batch_index)
            samples = [dataset[int(i)] for i in indices]
            rng = np.random.default_rng(deterministic_batch_seed(args.seed, batch_index, stream=0))
            with sharding.set_mesh(mesh):
                observation, placeholder, metrics = train_loader._openpi_batch(samples, rng)
                if np.any(host_numpy(placeholder)):
                    raise ValueError("Unexpected trajectory labels in the oracle-only loader")
                metrics = agg.sanitize_metric_tensors(metrics)
                key = jax.random.key(deterministic_batch_seed(args.seed, batch_index, stream=1))
                initial = initialize(key, metrics)
                depth = depth_for_batch(args.seed, batch_index, args.depth_cap)
                candidates, finite = rollout(teacher_params, observation, initial, metrics, jnp.asarray(depth, jnp.int32))
                if not np.all(host_numpy(finite)):
                    raise FloatingPointError(f"Nonfinite teacher rollout at batch {batch_index}")
                directions, _ = oracle(candidates, metrics)
                valid = jax.device_put(valid_np.astype(np.float32), data_sharding)
                weights = goal_weights(metrics["is_object_goal"], valid, object_fraction=object_fraction,
                                       target_sampled_fraction=target_sampled_fraction)
                state, info = update(state, observation, candidates, directions, weights, valid)
            values_host = jax.device_get(info)
            if not all(np.all(np.isfinite(v)) for v in values_host.values()):
                raise FloatingPointError(f"Nonfinite training diagnostics at optimizer step {int(state.step)}")
            window.append(values_host)
            step = int(state.step)
            if len(window) >= args.log_interval or step == target_step or batch_index == start_step - args.source_step:
                averaged = {k: float(np.mean([x[k] for x in window])) for k in window[0]}
                log({**learning_metrics(averaged), "train/grad_norm": averaged["grad_norm"],
                     "train/field_oracle_cosine": averaged["field_oracle_cosine"],
                     "data/epoch": (batch_index + 1) / sampler.steps_per_epoch,
                     "data/conditions_seen": min((batch_index + 1) * args.batch_size, len(dataset))
                     if batch_index < sampler.steps_per_epoch else (batch_index // sampler.steps_per_epoch) * len(dataset)
                     + min((batch_index % sampler.steps_per_epoch + 1) * args.batch_size, len(dataset)),
                     "teacher/sampled_depth": depth,
                     "performance/seconds_per_update": (time.monotonic() - window_start) / len(window)})
                logging.info("Train step %d loss=%.6f grad=%.4f depth=%d epoch=%.4f", step,
                             averaged["loss"], averaged["grad_norm"], depth,
                             (batch_index + 1) / sampler.steps_per_epoch)
                window, window_start = [], time.monotonic()
            if stop_requested or step % args.save_interval == 0 or step == target_step:
                save()
            if not stop_requested and (step % args.eval_interval == 0 or step == target_step):
                evaluate()
                window_start = time.monotonic()
            del samples, observation, metrics, placeholder, candidates, directions, initial
        save()
        logging.info("Continuation stopped at step %d (target %d); resume with the same output directory",
                     int(state.step), target_step)
    finally:
        manager.close()
        run.finish()
        dataset.close()
        eval_loader.dataset.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        del teacher_params
        gc.collect()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-step", type=int, default=35000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth-cap", type=int, default=14)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--additional-steps", type=int, default=50000)
    budget.add_argument("--one-epoch", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fsdp-devices", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-interval", type=int, default=2500)
    parser.add_argument("--eval-interval", type=int, default=2500)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--source-wandb-run", default="yohanab/astar/4wzq1hru")
    parser.add_argument("--wandb-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    for key in ("source_step", "depth_cap", "additional_steps", "batch_size", "fsdp_devices",
                "save_interval", "eval_interval", "log_interval"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.eval_batches is not None and args.eval_batches <= 0:
        parser.error("--eval-batches must be positive")
    return args


if __name__ == "__main__":
    main(parse_args())
