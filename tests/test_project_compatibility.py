"""The renamed package keeps existing imports and dataset configuration usable."""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

from gear_nav import dataloader


def test_legacy_package_path_resolves_to_same_source():
    legacy = import_module("astar.esdf")
    current = import_module("gear_nav.esdf")
    assert Path(legacy.__file__).resolve() == Path(current.__file__).resolve()


@pytest.mark.parametrize(
    ("new_root", "old_root", "expected"),
    [("/new", "/old", "/new"), (None, "/old", "/old")],
)
def test_dataset_root_supports_new_and_legacy_environment(monkeypatch, new_root, old_root, expected):
    if new_root is None:
        monkeypatch.delenv("GEAR_NAV_DATA_ROOT", raising=False)
    else:
        monkeypatch.setenv("GEAR_NAV_DATA_ROOT", new_root)
    monkeypatch.setenv("ASTAR_DATA_ROOT", old_root)
    monkeypatch.setattr(dataloader, "AstarNavigationDataset", lambda **kwargs: kwargs)
    monkeypatch.setattr(dataloader, "AstarNavigationDataLoader", lambda dataset, **kwargs: dataset)
    config = SimpleNamespace(model=SimpleNamespace(action_horizon=16, max_token_len=48), batch_size=1)
    result = dataloader.create_navigation_data_loader(config)
    assert result["data_root"] == Path(expected)
