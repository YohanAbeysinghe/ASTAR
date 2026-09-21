"""Download and expose the official pi0.5 base parameters for ASTAR."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import Sequence

from openpi.shared import download

PI05_PARAMS_URI = "gs://openpi-assets/checkpoints/pi05_base/params"
DEFAULT_OUTPUT = Path("checkpoints/pi05_base/params")
REQUIRED_FILES = (
    "_CHECKPOINT_METADATA",
    "_METADATA",
    "_sharding",
    "commit_success.txt",
    "manifest.ocdbt",
)


def validate_checkpoint(path: Path) -> None:
    """Raise if ``path`` is not a complete Orbax parameter checkpoint."""
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete pi0.5 checkpoint at {path}: missing {', '.join(missing)}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy the cached 12 GB checkpoint instead of creating a local symlink.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()

    if output.exists() or output.is_symlink():
        validate_checkpoint(output)
        print(f"pi0.5 checkpoint already available: {output}")
        return

    cached = Path(download.maybe_download(PI05_PARAMS_URI)).resolve()
    validate_checkpoint(cached)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.copy:
        shutil.copytree(cached, output)
    else:
        output.symlink_to(cached, target_is_directory=True)

    validate_checkpoint(output)
    method = "copied" if args.copy else "linked"
    print(f"pi0.5 checkpoint {method} at {output}")
    print(f"OpenPI cache: {cached}")


if __name__ == "__main__":
    main()
