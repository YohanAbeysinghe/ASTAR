"""Infer and render recurrent model paths beyond their trained rollout depth."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np


def trained_depth_from_replay(path: Path) -> int:
    """Read the number of completed aggregation rounds from a replay archive."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
    depth = int(metadata["round_index"])
    if depth <= 0:
        raise ValueError(f"Replay has no completed aggregation rounds: {path}")
    return depth


def resolve_capture_depths(trained_depth: int, extra_depths: list[int]) -> tuple[int, ...]:
    depths = tuple(sorted({trained_depth, *extra_depths}))
    if not depths or any(depth <= 0 for depth in depths):
        raise ValueError("Capture depths must be positive")
    return depths


def _path_key(depth: int) -> str:
    return f"paths_{depth}"


def _checkpoint_array_name(path: tuple[Any, ...]) -> str:
    """Convert a JAX key path into the dotted name used by Orbax/TensorStore."""
    parts = ["params"]
    for key in path:
        if hasattr(key, "key"):
            parts.append(str(key.key))
        elif hasattr(key, "name"):
            parts.append(str(key.name))
        elif hasattr(key, "idx"):
            parts.append(str(key.idx))
        else:
            raise TypeError(f"Unsupported JAX tree key in checkpoint path: {key!r}")
    return ".".join(parts)


def _stream_restore_params(source_dir: Path, params_shape: Any, params_sharding: Any) -> Any:
    """Restore a parameter tree in storage-sized chunks directly to one GPU.

    Orbax normally stages an entire array on the host. The largest fused tensor
    in this model is 2.4 GB, which cannot be restored inside a 4 GB Slurm host
    allocation once JAX and the model graph are loaded. Reading the checkpoint's
    native chunks and concatenating them on the accelerator bounds host usage.
    """
    import jax
    import jax.numpy as jnp
    import tensorstore as ts

    leaves_with_paths, tree_def = jax.tree_util.tree_flatten_with_path(params_shape)
    sharding_leaves, sharding_tree_def = jax.tree_util.tree_flatten(params_sharding)
    if tree_def != sharding_tree_def:
        raise ValueError("Parameter shapes and shardings have different tree structures")

    restored_leaves = []
    total = len(leaves_with_paths)
    for leaf_index, ((path, expected), leaf_sharding) in enumerate(
        zip(leaves_with_paths, sharding_leaves, strict=True),
        start=1,
    ):
        name = _checkpoint_array_name(path)
        spec = {
            "driver": "zarr",
            "kvstore": {
                "driver": "ocdbt",
                "base": {"driver": "file", "path": str(source_dir)},
                "path": name,
            },
            "recheck_cached_data": False,
            "recheck_cached_metadata": False,
        }
        store = ts.open(spec, open=True).result()
        shape = tuple(int(size) for size in store.shape)
        expected_shape = tuple(int(size) for size in expected.shape)
        if shape != expected_shape or str(store.dtype.numpy_dtype) != str(expected.dtype):
            raise ValueError(
                f"Checkpoint array {name} is {shape}/{store.dtype}; expected {expected_shape}/{expected.dtype}"
            )

        chunk_shape = tuple(int(size) for size in store.chunk_layout.read_chunk.shape)
        split_axes = [axis for axis, (chunk, size) in enumerate(zip(chunk_shape, shape, strict=True)) if chunk < size]
        if len(split_axes) > 1:
            raise ValueError(f"Checkpoint array {name} is chunked over multiple axes: {chunk_shape}")

        if not split_axes:
            host = store.read().result()
            restored = jax.device_put(host, leaf_sharding)
            jax.block_until_ready(restored)
            del host
        else:
            axis = split_axes[0]
            parts = []
            for start in range(0, shape[axis], chunk_shape[axis]):
                selection = [slice(None)] * len(shape)
                selection[axis] = slice(start, min(start + chunk_shape[axis], shape[axis]))
                host = store[tuple(selection)].read().result()
                part = jax.device_put(host, leaf_sharding)
                jax.block_until_ready(part)
                parts.append(part)
                del host
                gc.collect()
            restored = jnp.concatenate(parts, axis=axis)
            jax.block_until_ready(restored)
            del parts
        restored_leaves.append(restored)
        gc.collect()
        print(f"Stream-restored parameter {leaf_index}/{total}: {name}", flush=True)
    return jax.tree_util.tree_unflatten(tree_def, restored_leaves)


def infer_paths(args: argparse.Namespace) -> None:
    import jax
    from openpi.training import sharding

    from astar.evaluation import aggregation as agg

    checkpoint_root = args.checkpoint_root.resolve()
    training_config = (args.training_config or checkpoint_root / "monitoring/configuration.json").resolve()
    replay_path = (args.replay_path or checkpoint_root / f"aggregation_state_{args.step:08d}.npz").resolve()
    trained_depth = trained_depth_from_replay(replay_path)
    capture_depths = resolve_capture_depths(trained_depth, args.extra_depths)
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}")

    values = agg.load_training_config(training_config)
    model_args = agg.build_training_namespace(
        values,
        batch_size=args.batch_size,
        fsdp_devices=args.fsdp_devices,
    )
    model_config = agg.create_train_config(model_args)
    energy_config = agg.build_energy_config(values)
    dataset_args = argparse.Namespace(
        data_root=args.data_root.resolve(),
        manifest_path=args.manifest_path.resolve() if args.manifest_path else None,
        eval_split_ids=args.eval_split_ids.resolve(),
        goal_kind="sampled",
        object_goal_modality="training",
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print(f"Capture depths: {capture_depths}", flush=True)
    print(f"Building {args.fsdp_devices}-device model mesh", flush=True)
    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(sharding.DATA_AXIS),
    )
    model_def, parameter_shapes, parameter_shardings = agg.model_template(
        model_config,
        mesh,
        seed=int(getattr(model_config, "seed", args.seed)),
        parameter_sources=[args.parameter_source],
    )
    model_kernels = agg.build_model_kernels(model_def, values, energy_config)
    print(f"Restoring checkpoint {args.step} ({args.parameter_source})", flush=True)
    params_shape = parameter_shapes[args.parameter_source]
    params_sharding = parameter_shardings[args.parameter_source]
    source_dir = checkpoint_root / str(args.step) / ("train_state" if args.parameter_source == "raw" else "params")
    if args.stream_restore:
        params = _stream_restore_params(source_dir, params_shape, params_sharding)
    else:
        restore_item = {"params": params_shape}
        restore_sharding = {"params": params_sharding}
        restore_args = jax.tree.map(
            lambda _, leaf_sharding: agg.ocp.ArrayRestoreArgs(
                restore_type=jax.Array,
                sharding=leaf_sharding,
            ),
            restore_item,
            restore_sharding,
        )
        restore_kwargs: dict[str, Any] = {
            "item": restore_item,
            "restore_args": restore_args,
        }
        if args.parameter_source == "raw":
            restore_kwargs["transforms"] = {}
        restore_handler = agg.ocp.PyTreeCheckpointHandler(restore_concurrent_gb=args.restore_concurrent_gb)
        with agg.ocp.Checkpointer(restore_handler) as checkpointer:
            params = checkpointer.restore(
                source_dir,
                args=agg.ocp.args.PyTreeRestore(**restore_kwargs),
            )["params"]
    jax.block_until_ready(params)

    dataset = agg.build_dataset(dataset_args, values, model_config)
    try:
        selected_raw = json.loads(args.selected_examples.read_text(encoding="utf-8"))
        selected = [agg.ExampleIdentity(**item) for item in selected_raw]
        if not selected:
            raise ValueError(f"No selected examples in {args.selected_examples}")
        batcher = agg.build_batcher(
            dataset_args,
            values,
            model_config,
            data_sharding,
            dataset,
        )
        kernels = agg.build_kernels(
            values,
            model_config,
            energy_config,
            prior_seed=args.seed,
            safety_sample_spacing_m=args.safety_sample_spacing_m,
        )

        captured: dict[int, list[np.ndarray]] = {depth: [] for depth in capture_depths}
        finite_by_depth: dict[int, list[np.ndarray]] = {depth: [] for depth in capture_depths}
        total_batches = math.ceil(len(selected) / args.batch_size)
        for batch_index, observation, metrics, identities, valid_count in agg.iter_evaluation_batches(
            dataset,
            batcher,
            selected,
            batch_size=args.batch_size,
            seed=args.seed,
        ):
            words = jax.make_array_from_process_local_data(
                batcher._sharding,
                agg._identity_words(identities, batch_size=args.batch_size),
            )
            with sharding.set_mesh(mesh):
                current_paths = kernels.initialize_prior(words, metrics)
                for depth in range(1, max(capture_depths) + 1):
                    current_paths, _predicted_direction, transition = model_kernels.model_only_step(
                        params,
                        observation,
                        current_paths,
                        metrics,
                    )
                    if depth not in captured:
                        continue
                    decoded, _ = kernels.decode_paths(current_paths, metrics)
                    captured[depth].append(np.asarray(jax.device_get(decoded), dtype=np.float32)[:valid_count])
                    finite_by_depth[depth].append(
                        np.asarray(
                            jax.device_get(transition["raw_direction_finite"]),
                            dtype=np.float32,
                        )[:valid_count]
                    )
            print(f"Inferred batch {batch_index + 1}/{total_batches}", flush=True)
    finally:
        dataset.close()

    arrays = {depth: np.concatenate(chunks, axis=0) for depth, chunks in captured.items()}
    finite_arrays = {depth: np.concatenate(chunks, axis=0) for depth, chunks in finite_by_depth.items()}
    expected_shape = (len(selected), int(values["action_horizon"]), 2)
    for depth, paths in arrays.items():
        if paths.shape != expected_shape:
            raise RuntimeError(f"Depth {depth} has shape {paths.shape}; expected {expected_shape}")
        if not np.all(np.isfinite(paths)):
            raise RuntimeError(f"Depth {depth} contains non-finite path coordinates")
        if not np.all(finite_arrays[depth] == 1.0):
            raise RuntimeError(f"Depth {depth} contains a non-finite raw model direction")

    metadata = {
        "checkpoint_root": str(checkpoint_root),
        "checkpoint": args.step,
        "parameter_source": args.parameter_source,
        "mode": "model_only",
        "trained_depth": trained_depth,
        "capture_depths": list(capture_depths),
        "selected_examples": str(args.selected_examples.resolve()),
        "data_root": str(args.data_root.resolve()),
        "seed": args.seed,
    }
    payload: dict[str, Any] = {_path_key(depth): paths for depth, paths in arrays.items()}
    payload.update(
        {
            "goals": np.asarray(
                [[item.goal_x_m, item.goal_y_m] for item in selected],
                dtype=np.float32,
            ),
            "dataset_indices": np.asarray([item.dataset_index for item in selected], dtype=np.int64),
            "clip_ids": np.asarray([item.clip_id for item in selected]),
            "sample_ids": np.asarray([item.sample_id for item in selected]),
            "goal_indices": np.asarray([item.goal_index for item in selected], dtype=np.int32),
            "capture_depths": np.asarray(capture_depths, dtype=np.int32),
            "trained_depth": np.asarray(trained_depth, dtype=np.int32),
            "checkpoint": np.asarray(args.step, dtype=np.int32),
            "parameter_source": np.asarray(args.parameter_source),
            "mode": np.asarray("model_only"),
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    print(f"Wrote {len(selected)} examples at depths {capture_depths} to {output}", flush=True)


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "example"


def render_paths(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    from PIL import Image

    archive_path = args.archive.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    with np.load(archive_path, allow_pickle=False) as archive:
        capture_depths = tuple(int(value) for value in archive["capture_depths"])
        trained_depth = int(archive["trained_depth"])
        checkpoint = int(archive["checkpoint"])
        parameter_source = str(archive["parameter_source"])
        mode = str(archive["mode"])
        dataset_indices = np.asarray(archive["dataset_indices"], dtype=np.int64)
        clip_ids = np.asarray(archive["clip_ids"]).astype(str)
        sample_ids = np.asarray(archive["sample_ids"]).astype(str)
        goal_indices = np.asarray(archive["goal_indices"], dtype=np.int32)
        goals = np.asarray(archive["goals"], dtype=np.float32)
        paths_by_depth = {depth: np.asarray(archive[_path_key(depth)], dtype=np.float32) for depth in capture_depths}

    count = len(dataset_indices)
    expected_prefix = (count,)
    if any(len(values) != count for values in (clip_ids, sample_ids, goal_indices, goals, *paths_by_depth.values())):
        raise ValueError(f"Archive arrays do not share prefix {expected_prefix}: {archive_path}")

    manifest_path = args.manifest_path or args.data_root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    esdf_configs = {str(clip["clip_id"]): clip["esdf_config"] for clip in manifest["clips"]}
    colors = {trained_depth: "#00E5FF", 20: "#FF9800", 30: "#E040FB"}
    linestyles = {trained_depth: "-", 20: "--", 30: "-."}
    markers = {trained_depth: "o", 20: "s", 30: "^"}
    fallback_colors = plt.get_cmap("tab10")
    index_rows = []

    for offset in range(count):
        rank = offset + 1
        clip_id = str(clip_ids[offset])
        sample_id = str(sample_ids[offset])
        goal_index = int(goal_indices[offset])
        goal = goals[offset]
        image_path = args.data_root / clip_id / "images" / f"{sample_id}.jpg"
        esdf_path = args.data_root / clip_id / "esdf_fix" / f"{sample_id}.npy"
        with Image.open(image_path) as source:
            frame = np.asarray(source.convert("RGB"))
        esdf = np.load(esdf_path).astype(np.float32, copy=False)

        grid = esdf_configs[clip_id]
        x_min, x_max = map(float, grid["x_range_m"])
        y_min, y_max = map(float, grid["y_range_m"])
        finite = esdf[np.isfinite(esdf)]
        color_scale = max(
            float(np.percentile(np.abs(finite), 99)) if finite.size else 1.0,
            1.0,
        )
        fig, (frame_ax, esdf_ax) = plt.subplots(
            1,
            2,
            figsize=(16, 7.5),
            dpi=120,
            gridspec_kw={"width_ratios": [1.05, 1.0]},
        )
        frame_ax.imshow(frame)
        frame_ax.set_title("Source camera frame", fontsize=14)
        frame_ax.axis("off")

        background = esdf_ax.imshow(
            np.where(np.isfinite(esdf), esdf, np.nan),
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            cmap="coolwarm",
            vmin=-color_scale,
            vmax=color_scale,
            interpolation="nearest",
            aspect="equal",
        )
        # Draw the most extrapolated paths first so the trained-depth path stays visible.
        for color_index, depth in enumerate(reversed(capture_depths)):
            path = paths_by_depth[depth][offset]
            label = (
                f"{depth} refinements (trained depth)"
                if depth == trained_depth
                else f"{depth} refinements (extrapolated)"
            )
            line = esdf_ax.plot(
                path[:, 0],
                path[:, 1],
                color=colors.get(depth, fallback_colors(color_index)),
                linestyle=linestyles.get(depth, "-"),
                linewidth=2.8,
                marker=markers.get(depth, "o"),
                markersize=3.8,
                markeredgecolor="black",
                markeredgewidth=0.4,
                label=label,
                zorder=5 + color_index,
            )[0]
            line.set_path_effects([path_effects.Stroke(linewidth=4.8, foreground="black"), path_effects.Normal()])
        esdf_ax.scatter(
            [goal[0]],
            [goal[1]],
            marker="*",
            s=260,
            color="#FFD600",
            edgecolor="black",
            linewidth=1.2,
            label="Goal",
            zorder=10,
        )
        esdf_ax.scatter(
            [0.0],
            [0.0],
            marker="o",
            s=80,
            color="white",
            edgecolor="black",
            linewidth=1.2,
            label="Robot start",
            zorder=10,
        )
        esdf_ax.set_xlim(x_min, x_max)
        esdf_ax.set_ylim(y_min, y_max)
        esdf_ax.set_xlabel("Local x (m)")
        esdf_ax.set_ylabel("Local y (m)")
        esdf_ax.set_title("ESDF with recurrent model paths", fontsize=14)
        esdf_ax.grid(color="black", alpha=0.15, linewidth=0.5)
        esdf_ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        colorbar = fig.colorbar(background, ax=esdf_ax, fraction=0.046, pad=0.04)
        colorbar.set_label("Signed ESDF distance (m)")

        fig.suptitle(
            f"{clip_id} / {sample_id}   sampled goal {goal_index}   "
            f"checkpoint {checkpoint}   trained depth {trained_depth}",
            fontsize=15,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
        filename = f"{rank:03d}_{_safe_filename(clip_id)}_{_safe_filename(sample_id)}_goal{goal_index:02d}.png"
        fig.savefig(output_dir / filename, bbox_inches="tight")
        plt.close(fig)
        index_rows.append(
            {
                "plot_file": filename,
                "dataset_index": int(dataset_indices[offset]),
                "clip_id": clip_id,
                "sample_id": sample_id,
                "goal_index": goal_index,
                "goal_x_m": float(goal[0]),
                "goal_y_m": float(goal[1]),
                "checkpoint": checkpoint,
                "trained_depth": trained_depth,
                "capture_depths": ";".join(map(str, capture_depths)),
                "parameter_source": parameter_source,
                "mode": mode,
            }
        )
        if rank % 25 == 0 or rank == count:
            print(f"Rendered {rank}/{count}", flush=True)

    with (output_dir / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"Wrote {count} plots to {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser("infer", help="Run recurrent model-only inference")
    infer.add_argument("--checkpoint-root", type=Path, required=True)
    infer.add_argument("--step", type=int, required=True)
    infer.add_argument("--training-config", type=Path, default=None)
    infer.add_argument("--replay-path", type=Path, default=None)
    infer.add_argument("--data-root", type=Path, required=True)
    infer.add_argument("--manifest-path", type=Path, default=None)
    infer.add_argument("--eval-split-ids", type=Path, required=True)
    infer.add_argument("--selected-examples", type=Path, required=True)
    infer.add_argument("--extra-depths", nargs="+", type=int, default=[20, 30])
    infer.add_argument("--parameter-source", choices=("raw", "ema"), default="raw")
    infer.add_argument("--batch-size", type=int, default=8)
    infer.add_argument("--fsdp-devices", type=int, default=4)
    infer.add_argument(
        "--stream-restore",
        action="store_true",
        help="Stream checkpoint chunks to one GPU to fit a low host-memory allocation.",
    )
    infer.add_argument(
        "--restore-concurrent-gb",
        type=float,
        default=2.6,
        help="Orbax host-memory limiter; 2.6 GB admits this model's largest shard.",
    )
    infer.add_argument("--safety-sample-spacing-m", type=float, default=0.05)
    infer.add_argument("--seed", type=int, default=0)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--overwrite", action="store_true")
    infer.set_defaults(handler=infer_paths)

    render = subparsers.add_parser("render", help="Render camera/ESDF path overlays")
    render.add_argument("--archive", type=Path, required=True)
    render.add_argument("--data-root", type=Path, required=True)
    render.add_argument("--manifest-path", type=Path, default=None)
    render.add_argument("--output-dir", type=Path, required=True)
    render.add_argument("--overwrite", action="store_true")
    render.set_defaults(handler=render_paths)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
