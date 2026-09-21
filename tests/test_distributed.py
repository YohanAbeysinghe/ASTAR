"""Two real JAX controllers: global replay, validation plots, save and resume."""

import os
from pathlib import Path
import socket
import subprocess
import sys


def test_two_controller_training_and_resume(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{sock.getsockname()[1]}"
    env = dict(os.environ, JAX_PLATFORMS="cpu", WANDB_MODE="disabled",
               MPLBACKEND="Agg", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               XLA_FLAGS="--xla_force_host_platform_device_count=1")
    workers, logs = [], []
    try:
        for rank in range(2):
            log = (tmp_path / f"rank{rank}.log").open("w")
            logs.append(log)
            workers.append(subprocess.Popen(
                [sys.executable, __file__, address, str(rank), str(tmp_path)],
                env=env, stdout=log, stderr=subprocess.STDOUT,
            ))
        for worker in workers:
            assert worker.wait(timeout=240) == 0, "\n".join(
                path.read_text() for path in tmp_path.glob("rank*.log")
            )
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
            worker.wait()
        for log in logs:
            log.close()


def worker(address, rank, directory):
    # Limit CPU thread pools when simulating two hosts on one machine.
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, allowed[rank * 2:rank * 2 + 2] or allowed[:2])
    import jax

    jax.distributed.initialize(
        coordinator_address=address, num_processes=2, process_id=rank,
        local_device_ids=[0], initialization_timeout=60,
    )
    import json
    from types import SimpleNamespace

    from etils import epath
    import flax.nnx as nnx
    import jax.numpy as jnp
    import numpy as np
    import optax
    import pytest
    from test_training_resume import TinyField
    from test_training_resume import TinyLoader

    import astar.training.aggregation as trainer

    root = Path(directory)
    managers = []
    original_manager = trainer.initialize_bounded_checkpoint_dir

    class DistributedLoader(TinyLoader):
        def __init__(self, split, sharding):
            super().__init__(split)
            self.sharding = sharding

        def _openpi_batch(self, samples, rng):
            batch = super()._openpi_batch(samples, rng)
            # Distinct rank-local goals verify that a global batch is assembled.
            batch[2]["goal_xy"] = jnp.asarray([[0.6 + 0.1 * rank, 0.0]])
            return jax.tree.map(
                lambda x: jax.make_array_from_process_local_data(
                    self.sharding, np.asarray(x)
                ), batch,
            )

    def make_config(args):
        return SimpleNamespace(
            seed=42, fsdp_devices=1, checkpoint_dir=epath.Path(root / "run"),
            overwrite=False, resume=args.resume, batch_size=2,
            model=SimpleNamespace(action_horizon=4, action_dim=2),
            trainable_filter=nnx.Everything(), ema_decay=None,
            log_interval=1, save_interval=1, wandb_enabled=False,
        )

    def init_state(config, rng, mesh, **kwargs):
        graph, params = nnx.split(TinyField())
        tx = optax.adam(0.01)
        with trainer.at.disable_typechecking():
            state = trainer.training_utils.TrainState(
                step=jnp.asarray(0, jnp.int32), params=params, model_def=graph,
                opt_state=tx.init(params), tx=tx, ema_decay=None, ema_params=None,
            )
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        return jax.device_put(state, replicated), jax.tree.map(lambda _: replicated, state)

    def make_manager(*args, **kwargs):
        for manager in managers:
            manager.close()
        managers.clear()
        result = original_manager(*args, **kwargs)
        managers.append(result[0])
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(trainer, "create_train_config", make_config)
        patch.setattr(trainer, "init_train_state", init_state)
        patch.setattr(trainer, "initialize_bounded_checkpoint_dir", make_manager)
        patch.setattr(trainer, "create_oracle_only_data_loader",
                      lambda config, args, sharding, *, split, num_batches:
                      DistributedLoader(split, sharding))
        patch.setattr(trainer, "predict_descent_field",
                      lambda model, observation, paths, **kwargs:
                      jnp.broadcast_to(model.offset.value, paths.shape))
        patch.setenv("JAX_COMPILATION_CACHE_DIR", str(root / f"cache{rank}"))
        for resume, replay_updates in ((False, 0), (True, 0)):
            argv = [
                "trainer", "--exp-name", "distributed_smoke", "--batch-size", "2",
                "--fsdp-devices", "1", "--action-horizon", "4", "--train-particles", "1",
                "--aggregation-rounds", "1", "--updates-per-round", "1",
                "--aggregation-batches", "1", "--eval-batches", "1",
                "--eval-plot-count", "1", "--aggregation-plot-interval", "1",
                "--post-aggregation-updates", str(replay_updates),
                "--save-interval", "1", "--trainable-snapshot-interval", "1",
                "--segment-samples", "2", "--benefit-min-rounds", "99",
                "--convergence-min-rounds", "99", "--no-wandb-enabled",
            ] + (["--resume"] if resume else [])
            if resume:
                argv += ["--max-step-length-m", "2.0", "--continue-aggregation-until-round", "2",
                         "--relabel-max-step-from-config", str(root / "source_configuration.json")]
            patch.setattr(sys, "argv", argv)
            trainer.main(trainer.parse_args())
            if not resume:
                if rank == 0:
                    (root / "source_configuration.json").write_bytes(
                        (root / "run" / "monitoring/configuration.json").read_bytes())
                jax.experimental.multihost_utils.sync_global_devices("source_config_saved")
        for manager in managers:
            manager.close()
    if rank == 0:
        for step, expected_depth in ((0, 0), (1, 1), (2, 2)):
            artifact = root / "run" / "monitoring" / f"step_{step:08d}"
            metrics = json.loads((artifact / "metrics.json").read_text())
            assert metrics["rollout_steps"] == expected_depth
            assert len(metrics["examples"]) == 1  # rank one was padding only
            assert (artifact / "path_00.png").exists()
        with np.load(root / "run" / "aggregation_state_00000002.npz") as saved:
            assert saved["current_paths"].shape == (1, 2, 4, 2)
            assert saved["visited_paths"].shape[0] == 2
        assert json.loads((root / "run/cap_relabel.json").read_text())["retained_rounds"] == 1
        lines = (root / "run" / "monitoring" / "training_diagnostics.jsonl").read_text().splitlines()
        assert len(lines) == 2  # Exactly one writer, once per optimizer update.
    jax.experimental.multihost_utils.sync_global_devices("test_done")
    jax.distributed.shutdown()


if __name__ == "__main__":
    worker(sys.argv[1], int(sys.argv[2]), sys.argv[3])
