import os
import shutil

from astar.checkpoint_milestones import archive_milestone
from astar.checkpoint_milestones import completed_milestones


def _make_checkpoint(run_dir, step):
    checkpoint = run_dir / str(step)
    checkpoint.mkdir(parents=True)
    (checkpoint / "_CHECKPOINT_METADATA").write_text("committed")
    (checkpoint / "weights").write_bytes(b"weights")
    (run_dir / f"aggregation_state_{step:08d}.npz").write_bytes(b"replay")
    monitoring = run_dir / "monitoring"
    monitoring.mkdir(exist_ok=True)
    (monitoring / "configuration.json").write_text("{}")
    step_monitoring = monitoring / f"step_{step:08d}"
    step_monitoring.mkdir()
    (step_monitoring / "path_00.png").write_bytes(b"plot")
    return checkpoint


def test_archive_milestone_hard_links_checkpoint_replay_and_monitoring(tmp_path):
    run_dir = tmp_path / "run"
    source = _make_checkpoint(run_dir, 40_000)
    _make_checkpoint(run_dir, 42_500)
    archive = run_dir / "milestones"

    assert completed_milestones(run_dir, 10_000) == [40_000]
    assert archive_milestone(run_dir, archive, 40_000)
    assert not archive_milestone(run_dir, archive, 40_000)

    assert (archive / "40000" / "weights").read_bytes() == b"weights"
    assert os.stat(source / "weights").st_ino == os.stat(archive / "40000" / "weights").st_ino
    replay = run_dir / "aggregation_state_00040000.npz"
    assert os.stat(replay).st_ino == os.stat(archive / replay.name).st_ino
    assert (archive / "monitoring" / "step_00040000" / "path_00.png").is_file()

    shutil.rmtree(source)
    replay.unlink()
    assert (archive / "40000" / "weights").read_bytes() == b"weights"
    assert (archive / "aggregation_state_00040000.npz").read_bytes() == b"replay"


def test_incomplete_checkpoint_is_not_discovered(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "50000"
    checkpoint.mkdir(parents=True)
    (run_dir / "aggregation_state_00050000.npz").write_bytes(b"replay")
    assert completed_milestones(run_dir, 10_000) == []
