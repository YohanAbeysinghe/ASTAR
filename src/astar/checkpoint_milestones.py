"""Archive periodic checkpoints from an already-running bounded training job.

The trainer's checkpoint manager may already have been constructed with a
one-checkpoint retention policy. This watcher preserves selected future steps
without interrupting that process. It uses hard links on the shared filesystem,
so archiving is fast and the data remains available after Orbax unlinks its
rolling copy.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
import shutil
import subprocess
import time

CHECKPOINT_MARKER = "_CHECKPOINT_METADATA"


def _log(message: str) -> None:
    timestamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def _replay_path(run_dir: Path, step: int) -> Path:
    return run_dir / f"aggregation_state_{step:08d}.npz"


def completed_milestones(run_dir: Path, interval: int) -> list[int]:
    """Return completed checkpoint steps that have matching replay archives."""
    if interval <= 0:
        raise ValueError("interval must be positive")
    result = []
    for child in run_dir.iterdir():
        if not child.is_dir() or not child.name.isdigit():
            continue
        step = int(child.name)
        if step % interval:
            continue
        if not (child / CHECKPOINT_MARKER).is_file():
            continue
        if not _replay_path(run_dir, step).is_file():
            continue
        result.append(step)
    return sorted(result)


def _link_tree(source: Path, destination: Path) -> None:
    """Create a directory tree whose regular files hard-link to source."""
    shutil.copytree(source, destination, copy_function=os.link, symlinks=True)


def _archive_monitoring(run_dir: Path, archive_root: Path, step: int) -> None:
    source_root = run_dir / "monitoring"
    destination_root = archive_root / "monitoring"
    destination_root.mkdir(parents=True, exist_ok=True)
    for name in ("configuration.json", "selected_examples.json"):
        source = source_root / name
        destination = destination_root / name
        if source.is_file() and not destination.exists():
            os.link(source, destination)

    source_step = source_root / f"step_{step:08d}"
    destination_step = destination_root / source_step.name
    if source_step.is_dir() and not destination_step.exists():
        temporary = destination_root / f".{source_step.name}.tmp-{os.getpid()}"
        if temporary.exists():
            shutil.rmtree(temporary)
        _link_tree(source_step, temporary)
        temporary.rename(destination_step)


def archive_milestone(run_dir: Path, archive_root: Path, step: int) -> bool:
    """Archive one committed checkpoint and replay; return whether work occurred."""
    source_checkpoint = run_dir / str(step)
    source_replay = _replay_path(run_dir, step)
    if not (source_checkpoint / CHECKPOINT_MARKER).is_file():
        raise FileNotFoundError(f"Checkpoint {step} is not committed")
    if not source_replay.is_file():
        raise FileNotFoundError(f"Missing replay archive for checkpoint {step}")

    archive_root.mkdir(parents=True, exist_ok=True)
    destination_checkpoint = archive_root / str(step)
    destination_replay = archive_root / source_replay.name
    changed = False

    if not destination_checkpoint.exists():
        temporary = archive_root / f".{step}.tmp-{os.getpid()}"
        if temporary.exists():
            shutil.rmtree(temporary)
        _link_tree(source_checkpoint, temporary)
        if not (temporary / CHECKPOINT_MARKER).is_file():
            shutil.rmtree(temporary)
            raise RuntimeError(f"Archived checkpoint {step} is incomplete")
        temporary.rename(destination_checkpoint)
        changed = True

    if not destination_replay.exists():
        temporary_replay = archive_root / f".{source_replay.name}.tmp-{os.getpid()}"
        if temporary_replay.exists():
            temporary_replay.unlink()
        os.link(source_replay, temporary_replay)
        temporary_replay.rename(destination_replay)
        changed = True

    _archive_monitoring(run_dir, archive_root, step)
    return changed


def slurm_job_active(job_id: str) -> bool | None:
    """Return job liveness, or None when Slurm cannot be queried."""
    result = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        _log(f"squeue failed ({result.returncode}): {result.stderr.strip()}")
        return None
    return bool(result.stdout.strip())


def watch(
    *,
    run_dir: Path,
    archive_root: Path,
    interval: int,
    training_job_id: str,
    poll_seconds: float,
) -> None:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Training directory does not exist: {run_dir}")
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")

    _log(f"Watching {run_dir} for {interval}-step milestones from Slurm job {training_job_id}; archive={archive_root}")
    while True:
        for step in completed_milestones(run_dir, interval):
            if archive_milestone(run_dir, archive_root, step):
                _log(f"Archived checkpoint and replay at step {step}")

        active = slurm_job_active(training_job_id)
        if active is False:
            # One final scan covers metadata becoming visible as the job exits.
            time.sleep(min(poll_seconds, 10.0))
            for step in completed_milestones(run_dir, interval):
                if archive_milestone(run_dir, archive_root, step):
                    _log(f"Archived checkpoint and replay at step {step}")
            _log(f"Training job {training_job_id} is no longer active; exiting")
            return
        time.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, default=None)
    parser.add_argument("--interval", type=int, default=10_000)
    parser.add_argument("--training-job-id", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    archive_root = args.archive_root.expanduser().resolve() if args.archive_root is not None else run_dir / "milestones"
    watch(
        run_dir=run_dir,
        archive_root=archive_root,
        interval=args.interval,
        training_job_id=args.training_job_id,
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    main()
