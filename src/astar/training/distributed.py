"""Start one JAX controller per Slurm node, exposing all eight local GPUs."""

import os
import resource
import socket

import jax


def main():
    print(
        f"Distributed startup: host={socket.gethostname()}, "
        f"memlock_soft_hard_bytes={resource.getrlimit(resource.RLIMIT_MEMLOCK)}, "
        f"NCCL_IB_DISABLE={os.environ.get('NCCL_IB_DISABLE', 'unset')}",
        flush=True,
    )
    # Initialize before importing the trainer or creating any JAX arrays.
    jax.distributed.initialize(
        coordinator_address=os.environ["ASTAR_JAX_COORDINATOR"],
        num_processes=int(os.environ["SLURM_NTASKS"]),
        process_id=int(os.environ["SLURM_PROCID"]),
        local_device_ids=list(range(8)),
        initialization_timeout=600,
    )
    if (
        jax.process_count() != 2
        or jax.local_device_count() != 8
        or jax.device_count() != 16
        or any(device.platform != "gpu" for device in jax.local_devices())
    ):
        raise RuntimeError(f"Expected 2 hosts x 8 GPUs; got {jax.devices()}")
    print(
        f"Distributed allocation: rank={jax.process_index()}, "
        f"local={jax.local_devices()}, global_count={jax.device_count()}",
        flush=True,
    )
    from astar.training.aggregation import cli

    cli()
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
