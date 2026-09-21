"""Exercise the actual outer loop, SGD, and Orbax/replay resume with a tiny field."""

from types import SimpleNamespace

from etils import epath
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import astar.training.aggregation as trainer
from astar.training.replay import AggregationReplay


class TinyField(nnx.Module):
    def __init__(self):
        self.offset = nnx.Param(jnp.zeros((1, 4, 2), dtype=jnp.float32))


class TinyDataset:
    def __init__(self, split):
        self._split_ids = {split}
        self.goal_references = [
            SimpleNamespace(byte_offset=0, goal_kind="sampled", goal_index=0)
        ]

    def _read_record_at(self, offset):
        return {"clip_id": next(iter(self._split_ids)), "sample_id": "frame"}

    def __getitem__(self, index):
        return {}


class TinyLoader:
    batch_size = 1

    def __init__(self, split):
        self.dataset = TinyDataset(split)

    def _openpi_batch(self, samples, rng):
        return (
            jnp.zeros((1, 1)),
            jnp.zeros((1, 4, 2)),
            {
                "esdf": jnp.full((1, 20, 30), 5.0),
                "esdf_x_min": jnp.asarray([-1.0]),
                "esdf_y_min": jnp.asarray([-1.0]),
                "esdf_resolution": jnp.asarray([0.1]),
                "goal_xy": jnp.asarray([[0.6, 0.0]]),
                "robot_radius_m": jnp.asarray([0.5]),
            },
        )

    def __iter__(self):
        yield self._openpi_batch(None, None)

    def data_config(self):
        return SimpleNamespace(norm_stats=None, asset_id=None)


@pytest.mark.parametrize(
    "resume_mode", ["completed", "mid_round", "freeze_mid_round", "extend_aggregation", "cap_relabel"]
)
def test_training_resume_preserves_optimizer_and_cached_labels(
    tmp_path, monkeypatch, resume_mode
):
    managers, saved, payloads = [], {}, []
    current_root = [None]
    signal_handlers = {}
    interruption_step = [None if resume_mode in {"completed", "extend_aggregation"} else 1]
    updates_per_round = 1 if resume_mode in {"completed", "extend_aggregation"} else 2
    initialize_manager = trainer.initialize_bounded_checkpoint_dir
    save_state = trainer._checkpoints.save_state

    def manager_factory(*args, **kwargs):
        for manager in managers:
            manager.close()
        managers.clear()
        manager, resuming = initialize_manager(*args, **kwargs)
        managers.append(manager)
        return manager, resuming

    def make_config(args):
        directory = epath.Path(tmp_path / args.exp_name)
        current_root[0] = args.exp_name
        return SimpleNamespace(
            seed=42,
            fsdp_devices=1,
            checkpoint_dir=directory,
            overwrite=args.overwrite,
            resume=args.resume,
            batch_size=1,
            model=SimpleNamespace(action_horizon=4, action_dim=2),
            trainable_filter=nnx.Everything(),
            ema_decay=None,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            wandb_enabled=False,
        )

    def initialize_state(config, rng, mesh, **kwargs):
        graph, params = nnx.split(TinyField())
        tx = optax.adam(0.01)
        with trainer.at.disable_typechecking():
            state = trainer.training_utils.TrainState(
                step=jnp.asarray(0, jnp.int32),
                params=params,
                model_def=graph,
                opt_state=tx.init(params),
                tx=tx,
                ema_decay=None,
                ema_params=None,
            )
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        return state, jax.tree.map(lambda _: replicated, state)

    def save(manager, state, loader, step):
        saved[current_root[0]] = state
        save_state(manager, state, loader, step)
        if current_root[0] == "resumed" and step == interruption_step[0]:
            interruption_step[0] = None
            signal_handlers[trainer.signal.SIGUSR1](trainer.signal.SIGUSR1, None)

    monkeypatch.setattr(trainer, "create_train_config", make_config)
    monkeypatch.setattr(trainer, "initialize_bounded_checkpoint_dir", manager_factory)
    monkeypatch.setattr(trainer, "init_train_state", initialize_state)
    monkeypatch.setattr(
        trainer,
        "create_oracle_only_data_loader",
        lambda config, args, sharding, *, split, num_batches: TinyLoader(split),
    )
    monkeypatch.setattr(
        trainer,
        "predict_descent_field",
        lambda model, observation, paths, **kwargs: jnp.broadcast_to(
            model.offset.value, paths.shape
        ),
    )
    monkeypatch.setattr(trainer._checkpoints, "save_state", save)
    monkeypatch.setattr(trainer, "init_aggregation_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        trainer.wandb, "log", lambda payload, **kwargs: payloads.append(payload)
    )
    monkeypatch.setattr(trainer.wandb, "finish", lambda: None)
    monkeypatch.setattr(
        trainer.signal,
        "signal",
        lambda number, handler: signal_handlers.update({number: handler}),
    )
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(tmp_path / "jax_cache"))

    def args_for(name, updates, resume=False, replay_only=False, continue_until=0):
        argv = [
            "trainer",
            "--exp-name",
            name,
            "--aggregation-rounds",
            "2",
            "--updates-per-round",
            str(updates_per_round),
            "--aggregation-batches",
            "1",
            "--train-particles",
            "1",
            "--action-horizon",
            "4",
            "--batch-size",
            "1",
            "--fsdp-devices",
            "1",
            "--post-aggregation-updates",
            str(updates),
            "--eval-batches",
            "1",
            "--eval-plot-count",
            "0",
            "--aggregation-plot-interval",
            "0",
            "--replay-eval-interval",
            "2",
            "--benefit-min-rounds",
            "99",
            "--convergence-min-rounds",
            "99",
            "--segment-samples",
            "2",
            "--save-interval",
            "1",
            "--log-interval",
            "2",
            "--trainable-snapshot-interval",
            "0",
            "--no-wandb-enabled",
        ]
        if resume:
            argv.append("--resume")
        if replay_only:
            argv.append("--replay-only")
        if continue_until:
            argv.extend(["--continue-aggregation-until-round", str(continue_until)])
        monkeypatch.setattr("sys.argv", argv)
        return trainer.parse_args()

    try:
        if resume_mode == "cap_relabel":
            interruption_step[0] = None
            first = args_for("resumed", 0)
            first.aggregation_rounds = 1
            trainer.main(first)
            root = tmp_path / "resumed"
            source = tmp_path / "source_config.json"
            source.write_bytes((root / "monitoring/configuration.json").read_bytes())
            before = AggregationReplay.load(
                root / "aggregation_state_00000002.npz",
                expected_config_signature=trainer._resume_signature(first),
                expected_condition_fingerprints=trainer.cache_condition_pool(TinyLoader("train"), 1)[1],
            )
            continued = args_for("resumed", 0, resume=True, continue_until=2)
            continued.max_step_length_m = 2.0
            continued.relabel_max_step_from_config = str(source)
            trainer.main(continued)
            assert int(saved["resumed"].step) == 4
            after = AggregationReplay.load(
                root / "aggregation_state_00000004.npz",
                expected_config_signature=trainer._resume_signature(continued),
                expected_condition_fingerprints=before.condition_fingerprints,
            )
            np.testing.assert_array_equal(after.visited_paths[0], before.visited_paths[0])
            assert not np.allclose(after.oracle_directions[0], before.oracle_directions[0])
            assert (root / "before_cap_relabel_00000002.npz").exists()
            assert (root / "cap_relabel.json").exists()
            # A normal restart of the migrated run must never relabel again.
            monkeypatch.setattr(trainer, "relabel_max_step_replay", lambda *a: pytest.fail("relabel repeated"))
            trainer.main(continued)
            assert int(saved["resumed"].step) == 4
            return
        if resume_mode == "extend_aggregation":
            # Force a plateau so the integration test exercises reopening it,
            # ignoring subsequent plateaus, and stopping at the absolute cap.
            monkeypatch.setattr(trainer.BenefitTracker, "observe", lambda *a: True)
            first = args_for("resumed", 0)
            trainer.main(first)
            assert int(saved["resumed"].step) == 1
            fingerprints = trainer.cache_condition_pool(TinyLoader("train"), 1)[1]

            def load_replay(step):
                return AggregationReplay.load(
                    tmp_path / "resumed" / f"aggregation_state_{step:08d}.npz",
                    expected_config_signature=trainer._resume_signature(first),
                    expected_condition_fingerprints=fingerprints,
                )

            before = load_replay(1)
            assert before.aggregation_stop_reason == "validation_plateau"
            continued = args_for("resumed", 0, resume=True, continue_until=3)
            assert trainer._resume_signature(continued) == trainer._resume_signature(first)
            trainer.main(continued)
            resumed = saved["resumed"]
            assert int(resumed.step) == 3
            after = load_replay(3)
            assert after.round_index == after.labelled_rounds == 3
            assert after.aggregation_stop_reason == "resource_limit"
            np.testing.assert_array_equal(after.visited_paths[0], before.visited_paths[0])
            np.testing.assert_array_equal(after.oracle_directions[0], before.oracle_directions[0])
            # Reusing the absolute target must not add more rounds.
            trainer.main(continued)
            assert int(saved["resumed"].step) == 3
            monkeypatch.setattr(trainer.BenefitTracker, "observe", lambda *a: False)
            reference_args = args_for("uninterrupted", 0)
            reference_args.aggregation_rounds = 3
            trainer.main(reference_args)
            reference = saved["uninterrupted"]
            for left, right in zip(
                jax.tree.leaves((resumed.params, resumed.opt_state)),
                jax.tree.leaves((reference.params, reference.opt_state)),
                strict=True,
            ):
                np.testing.assert_allclose(left, right, rtol=0, atol=1e-7)
            return
        first = args_for("resumed", 3)
        trainer.main(first)
        first_step = 5 if resume_mode == "completed" else 1
        assert int(saved["resumed"].step) == first_step
        archive = tmp_path / "resumed" / f"aggregation_state_{first_step:08d}.npz"
        replay = AggregationReplay.load(
            archive,
            expected_config_signature=trainer._resume_signature(first),
            expected_condition_fingerprints=trainer.cache_condition_pool(
                TinyLoader("train"), 1
            )[1],
        )
        assert replay.replay_updates == (3 if resume_mode == "completed" else 0)
        assert replay.round_index == (2 if resume_mode == "completed" else 0)
        if resume_mode != "completed":
            assert replay.update_in_round == 1
            assert replay.labelled_rounds == 1
        original_query = trainer.query_energy_oracle

        def forbidden_query(*args, **kwargs):
            raise AssertionError(
                "Completed aggregation was queried again during replay continuation"
            )

        if resume_mode != "mid_round":
            monkeypatch.setattr(trainer, "query_energy_oracle", forbidden_query)
        trainer.main(
            args_for(
                "resumed", 5, resume=True, replay_only=resume_mode == "freeze_mid_round"
            )
        )
        resumed = saved["resumed"]
        expected_step = (
            6 if resume_mode == "freeze_mid_round" else 2 * updates_per_round + 5
        )
        assert int(resumed.step) == expected_step
        restored_replay = AggregationReplay.load(
            tmp_path / "resumed" / f"aggregation_state_{expected_step:08d}.npz",
            expected_config_signature=trainer._resume_signature(first),
            expected_condition_fingerprints=replay.condition_fingerprints,
        )
        assert restored_replay.replay_updates == 5
        for record in restored_replay.validation_records:
            assert record["rollout_steps"] == record["aggregation_round"]
        validation_payloads = [
            payload for payload in payloads
            if "eval/model_only/rollout_steps" in payload
        ]
        assert validation_payloads
        assert validation_payloads[0]["eval/model_only/rollout_steps"] == 0
        assert all(
            payload["eval/model_only/rollout_steps"] == payload["aggregation/round"]
            for payload in validation_payloads
        )
        assert validation_payloads[-1]["eval/model_only/rollout_steps"] == (
            restored_replay.round_index
        )
        np.testing.assert_array_equal(
            restored_replay.oracle_directions[0], replay.oracle_directions[0]
        )
        if resume_mode == "freeze_mid_round":
            assert restored_replay.round_index == 0
            assert restored_replay.update_in_round == 1
            assert restored_replay.aggregation_stop_reason == "manual_replay"
            np.testing.assert_array_equal(
                restored_replay.current_paths, replay.current_paths
            )
            # A later resume retains the frozen phase without requiring --replay-only again.
            trainer.main(args_for("resumed", 7, resume=True))
            assert int(saved["resumed"].step) == 8
            return
        monkeypatch.setattr(trainer, "query_energy_oracle", original_query)
        trainer.main(args_for("uninterrupted", 5))
        reference = saved["uninterrupted"]
        for left, right in zip(
            jax.tree.leaves((resumed.params, resumed.opt_state)),
            jax.tree.leaves((reference.params, reference.opt_state)),
            strict=True,
        ):
            np.testing.assert_allclose(left, right, rtol=0, atol=1e-7)
        assert any("train/loss_vs_zero" in payload for payload in payloads)
        assert any("eval/model_only/success_rate" in payload for payload in payloads)
        assert all("optimizer_step" in payload for payload in payloads)
        assert not any("diversity" in key for payload in payloads for key in payload)
    finally:
        for manager in managers:
            manager.close()
