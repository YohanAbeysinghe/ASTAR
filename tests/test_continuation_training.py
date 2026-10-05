"""Tiny CPU models exercise weighted SGD and actual Orbax continuation resumes."""

import functools
import json
from types import SimpleNamespace

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi.shared import array_typing as at
from openpi.training import utils as training_utils
import optax
import pytest

from astar.training import continuation as trainer
from astar.waypoint_energy import ObstacleEnergyConfig


class TinyField(nnx.Module):
    def __init__(self, offset=0.25):
        self.offset = nnx.Param(jnp.asarray(offset, jnp.float32))


def tiny_state(*, step=4, ema=True, tx=None):
    graph, params = nnx.split(TinyField())
    tx = optax.adam(0.01) if tx is None else tx
    opt_state = tx.init(params)
    # A nonzero optimizer counter/moment makes restoration observable.
    _, opt_state = tx.update(jax.tree.map(jnp.ones_like, params), opt_state, params)
    with at.disable_typechecking():
        return training_utils.TrainState(
            step=jnp.asarray(step, jnp.int32), params=params, model_def=graph,
            opt_state=opt_state, tx=tx, ema_decay=0.9 if ema else None,
            ema_params=jax.tree.map(lambda x: jnp.full_like(x, -0.5), params) if ema else None,
        )


def state_template(state, mesh):
    shape = jax.eval_shape(lambda: state)
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    return shape, jax.tree.map(lambda _: replicated, shape)


def assert_state_equal(left, right):
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)


def predict(model, observation, paths, **kwargs):
    return jnp.broadcast_to(model.offset.value * observation[:, :1, None], paths.shape)


class AssetLoader:
    def data_config(self):
        return SimpleNamespace(norm_stats=None, asset_id=None)


def save_checkpoint(root, state):
    manager, _ = trainer.initialize_bounded_checkpoint_dir(root, overwrite=False, resume=False)
    try:
        trainer.checkpoints.save_state(manager, state, AssetLoader(), int(state.step))
        manager.wait_until_finished()
    finally:
        manager.close()


@pytest.mark.parametrize("ema", [True, False])
def test_restore_preserves_raw_ema_optimizer_and_step_without_source_writes(tmp_path, ema):
    source = tmp_path / "source"
    original = tiny_state(ema=ema)
    save_checkpoint(source, original)
    before = {str(p.relative_to(source)): (p.stat().st_size, p.stat().st_mtime_ns)
              for p in source.rglob("*") if p.is_file()}
    mesh = trainer.sharding.make_mesh(1)
    shape, shardings = state_template(original, mesh)
    restored = trainer.restore_train_state(source, 4, shape, shardings)
    assert_state_equal(original, restored)
    after = {str(p.relative_to(source)): (p.stat().st_size, p.stat().st_mtime_ns)
             for p in source.rglob("*") if p.is_file()}
    assert after == before
    assert float(restored.params["offset"].value) == 0.25
    if ema:
        assert float(restored.ema_params["offset"].value) == -0.5


def test_weighted_sgd_excludes_padding_and_origin(monkeypatch):
    monkeypatch.setattr(trainer.agg, "predict_descent_field", predict)
    state = tiny_state(tx=optax.sgd(0.1))
    config = SimpleNamespace(trainable_filter=nnx.Param)
    observations = jnp.ones((3, 1))
    candidates = jnp.zeros((3, 4, 2))
    directions = jnp.broadcast_to(jnp.asarray([1.0, 3.0, 1000.0])[:, None, None], candidates.shape)
    directions = directions.at[:, 0].set(-9999)
    weights = jnp.asarray([0.5, 1.5, 0.0])
    valid = jnp.asarray([1.0, 1.0, 0.0])
    updated, metrics = jax.jit(functools.partial(trainer.train_step, config, 1.0))(
        state, observations, candidates, directions, weights, valid,
    )
    expected_loss = 0.5 * (0.5 * (0.25 - 1) ** 2 + 1.5 * (0.25 - 3) ** 2) / 2
    expected_gradient = (0.5 * (0.25 - 1) + 1.5 * (0.25 - 3)) / 2
    np.testing.assert_allclose(metrics["loss"], expected_loss, rtol=1e-6)
    np.testing.assert_allclose(metrics["grad_norm"], abs(expected_gradient), rtol=1e-6)
    np.testing.assert_allclose(updated.params["offset"].value, 0.25 - 0.1 * expected_gradient)
    np.testing.assert_allclose(updated.ema_params["offset"].value,
                               0.9 * -0.5 + 0.1 * (0.25 - 0.1 * expected_gradient))
    assert int(updated.step) == 5


def test_goal_weights_preserve_source_mixture_and_zero_padding():
    is_object = jnp.asarray([True, True, False, False, False, False, False])
    valid = jnp.asarray([1, 1, 1, 1, 1, 1, 0], jnp.float32)
    weights = np.asarray(trainer.goal_weights(
        is_object, valid, object_fraction=2 / 6, target_sampled_fraction=3 / 16,
    ))
    np.testing.assert_allclose(weights[:2].sum() / 6, 13 / 16)
    np.testing.assert_allclose(weights[2:].sum() / 6, 3 / 16)
    assert weights[-1] == 0


def test_depths_are_balanced_and_resume_independent():
    full = [trainer.depth_for_batch(42, i, 14) for i in range(140)]
    for start in range(0, 140, 14):
        assert sorted(full[start:start + 14]) == list(range(14))
    assert [trainer.depth_for_batch(42, i, 14) for i in range(57, 140)] == full[57:]
    assert [trainer.depth_for_batch(43, i, 14) for i in range(140)] != full


def test_restored_teacher_survives_donated_student_update(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer.agg, "predict_descent_field", predict)
    source = tmp_path / "source"
    original = tiny_state()
    save_checkpoint(source, original)
    mesh = trainer.sharding.make_mesh(1)
    shape, shardings = state_template(original, mesh)
    teacher = trainer.restore_checkpoint_params(
        source, step=4, source="raw", params_shape=shape.params, params_sharding=shardings.params,
    )
    student = trainer.restore_train_state(source, 4, shape, shardings)
    update = jax.jit(functools.partial(
        trainer.train_step, SimpleNamespace(trainable_filter=nnx.Param), 1.0,
    ), donate_argnums=(0,))
    student, _ = update(student, jnp.ones((2, 1)), jnp.zeros((2, 4, 2)),
                        jnp.ones((2, 4, 2)), jnp.ones(2), jnp.ones(2))
    jax.block_until_ready(student)
    assert not teacher["offset"].value.is_deleted()
    np.testing.assert_array_equal(teacher["offset"].value, 0.25)
    assert float(student.params["offset"].value) != 0.25


def test_main_full_epoch_matches_interrupted_resume(tmp_path, monkeypatch):
    """Real SGD, teacher rollout and Orbax save/restore over a padded tiny epoch."""
    source = tmp_path / "source"
    initial_state = tiny_state()
    save_checkpoint(source, initial_state)
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text("tiny stable manifest\n")
    split_paths = {}
    for split in ("train", "eval"):
        split_paths[split] = tmp_path / f"{split}.txt"
        split_paths[split].write_text(split + "\n")
    values = {
        "batch_size": 2, "sampled_goal_fraction": 0.5, "train_particles": 1,
        "eval_batches": 1, "eval_seed": 123, "eval_plot_count": 0,
        "action_horizon": 4, "action_dim": 2, "field_time": 1.0,
        "aggregation_step_size": 0.15, "aggregation_step_size_start": 0.1,
        "max_field_direction_rms": 1.0, "oracle_gradient_floor": 1e-4,
        "rollout_backtracks": 1, "backtrack_factor": 0.5,
        "energy_increase_tolerance": 1e-6, "eval_safety_spacing_m": 0.1,
        "eval_progress_threshold": 0.9, "decay_lr": 0.01,
    }
    active_run = [None]
    seen_batches = {}
    saved_states = {}
    teachers = []
    handlers = {}
    interrupted = [False]
    original_save = trainer.checkpoints.save_state
    original_restore_params = trainer.restore_checkpoint_params

    class Dataset:
        def __init__(self, split):
            self._split_ids = {split}
            self._split_ids_path = split_paths[split]
            self.samples_path = manifest
            self.object_indices = np.asarray([0, 1, 2])

        def __len__(self):
            return 7

        def __getitem__(self, index):
            return int(index)

        def close(self):
            pass

    class Loader(AssetLoader):
        def __init__(self, split, data_sharding):
            self.dataset = Dataset(split)
            self.data_sharding = data_sharding

        def _openpi_batch(self, samples, rng):
            seen_batches.setdefault(active_run[0], []).append(tuple(samples))
            noise = rng.uniform(size=(2, 1)).astype(np.float32)
            observation = noise + np.asarray(samples, np.float32)[:, None] / 10
            data = (observation, np.zeros((2, 4, 2), np.float32), {
                "is_object_goal": np.asarray(samples) < 3,
                "goal_xy": np.stack([np.asarray(samples) / 10 + 0.5, np.zeros(2)], axis=-1).astype(np.float32),
            })
            return jax.tree.map(lambda x: jax.device_put(x, self.data_sharding), data)

    def make_config(args):
        active_run[0] = args.exp_name
        return SimpleNamespace(seed=42, batch_size=2, fsdp_devices=1,
                               trainable_filter=nnx.Param, ema_decay=0.9)

    def init_state(config, rng, mesh, **kwargs):
        return state_template(initial_state, mesh)

    def restore_params(*args, **kwargs):
        restored = original_restore_params(*args, **kwargs)
        if kwargs["source"] == "raw":
            teachers.append(restored)
        return restored

    def save(manager, state, loader, step):
        saved_states[active_run[0]] = jax.tree.map(lambda x: np.asarray(x).copy(), state)
        original_save(manager, state, loader, step)
        if active_run[0] == "resumed" and step == 5 and not interrupted[0]:
            interrupted[0] = True
            handlers[trainer.signal.SIGUSR1](trainer.signal.SIGUSR1, None)

    def signal_handler(number, handler):
        old = handlers.get(number)
        handlers[number] = handler
        return old

    monkeypatch.setattr(trainer, "source_configuration", lambda args: dict(values))
    monkeypatch.setattr(trainer.agg, "create_train_config", make_config)
    monkeypatch.setattr(trainer.agg, "init_train_state", init_state)
    monkeypatch.setattr(trainer.agg, "create_oracle_only_data_loader",
                        lambda config, args, data_sharding, *, split, num_batches: Loader(split, data_sharding))
    monkeypatch.setattr(trainer.agg, "predict_descent_field", predict)
    monkeypatch.setattr(trainer.agg, "project_bounded_actions", lambda p, m, e: p.at[:, 0].set(0))
    monkeypatch.setattr(trainer.agg, "initialize_prior_paths",
                        lambda key, metrics, **kwargs: jax.random.normal(key, (2, 4, 2)) * 0.01)
    monkeypatch.setattr(trainer.agg, "query_energy_oracle",
                        lambda paths, metrics, **kwargs: (metrics["goal_xy"][:, None] - paths, {}))
    monkeypatch.setattr(trainer.agg, "sanitize_metric_tensors", lambda metrics: metrics)
    monkeypatch.setattr(trainer, "build_energy_config", lambda values: ObstacleEnergyConfig())
    monkeypatch.setattr(trainer, "cache_validation_pool", lambda *a, **kw: ([], [], []))
    monkeypatch.setattr(trainer, "evaluate_validation_pool", lambda *a, **kw: ({"progress_ratio": 0.5}, []))
    monkeypatch.setattr(trainer, "init_continuation_wandb",
                        lambda *a, **kw: SimpleNamespace(log=lambda payload: None, finish=lambda: None))
    monkeypatch.setattr(trainer, "restore_checkpoint_params", restore_params)
    monkeypatch.setattr(trainer.checkpoints, "save_state", save)
    monkeypatch.setattr(trainer.signal, "signal", signal_handler)

    def arguments(name, *, resume=False):
        argv = ["--source-root", str(source), "--source-step", "4", "--depth-cap", "2",
                "--output-dir", str(tmp_path / name), "--batch-size", "2", "--fsdp-devices", "1",
                "--one-epoch", "--save-interval", "1", "--eval-interval", "99", "--log-interval", "1",
                "--no-wandb-enabled"]
        return trainer.parse_args(argv + (["--resume"] if resume else []))

    trainer.main(arguments("uninterrupted"))
    trainer.main(arguments("resumed"))
    assert int(saved_states["resumed"].step) == 5
    trainer.main(arguments("resumed", resume=True))
    assert int(saved_states["resumed"].step) == 8
    assert_state_equal(saved_states["resumed"], saved_states["uninterrupted"])
    assert seen_batches["resumed"] == seen_batches["uninterrupted"]
    flattened = [index for batch in seen_batches["resumed"] for index in batch]
    assert sorted(flattened[:7]) == list(range(7))
    assert len(flattened) == 8  # One explicitly masked pad, no dropped tail.
    sidecar = json.loads((tmp_path / "resumed/continuation_state_00000008.json").read_text())
    assert sidecar["next_batch_index"] == 4
    for teacher in teachers:
        assert not teacher["offset"].value.is_deleted()
        np.testing.assert_array_equal(teacher["offset"].value, 0.25)
