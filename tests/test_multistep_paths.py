import json

import numpy as np
import pytest

from astar.evaluation.multistep_paths import resolve_capture_depths
from astar.evaluation.multistep_paths import trained_depth_from_replay


def test_trained_depth_comes_from_replay_round_index(tmp_path):
    replay = tmp_path / "aggregation_state_00035000.npz"
    np.savez_compressed(
        replay,
        metadata_json=np.asarray(json.dumps({"round_index": 14})),
    )
    assert trained_depth_from_replay(replay) == 14


def test_capture_depths_include_trained_depth_and_are_sorted():
    assert resolve_capture_depths(14, [30, 20, 20]) == (14, 20, 30)


def test_capture_depths_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        resolve_capture_depths(14, [0, 20])
