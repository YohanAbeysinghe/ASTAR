"""Verify ASTAR's package, OpenPI dependency, and optional base checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import jax
import openpi

from astar.tools.download_pi05 import DEFAULT_OUTPUT
from astar.tools.download_pi05 import validate_checkpoint


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--require-checkpoint",
        action="store_true",
        help="Fail instead of warning when the pi0.5 checkpoint is absent.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    print(f"OpenPI package: {Path(openpi.__file__).resolve()}")
    print(f"JAX devices ({jax.device_count()}): {jax.devices()}")

    if checkpoint.exists():
        validate_checkpoint(checkpoint)
        print(f"pi0.5 checkpoint: {checkpoint}")
    elif args.require_checkpoint:
        raise FileNotFoundError(
            f"pi0.5 checkpoint not found at {checkpoint}; run astar-download-pi05"
        )
    else:
        print(f"pi0.5 checkpoint not installed yet: {checkpoint}")


if __name__ == "__main__":
    main()
