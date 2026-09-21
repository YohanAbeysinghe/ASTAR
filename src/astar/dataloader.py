"""ASTAR navigation dataset and batching utilities.

The data-generation pipeline writes one JSONL row per video frame. Each row can
contain several object goals and several sampled free-space goals; each goal
points at arrays in a per-frame ``paths/<sample_id>.npz`` file.

This module keeps the base dataset close to those semantics:

* one dataset item is one goal/path training example;
* object and sampled goals are both fully indexed;
* the full ESDF grid is loaded for energy-based training;
* the object bbox crop is exposed as a goal image, not as a fake robot camera.

An optional OpenPI output mode is kept for current training experiments, but it
is deliberately an adapter layer over the ASTAR sample format.
"""

from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import json
import os
from pathlib import Path
from typing import Any
from typing import Literal

import numpy as np

GoalKind = Literal["object", "sampled"]
OutputFormat = Literal["astar", "openpi"]

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_SAMPLED_GOAL_FRACTION = 1.0 / 6.0
DEFAULT_OBJECT_TEXT_GOAL_PROB = 1.0 / 3.0
DEFAULT_OBJECT_IMAGE_GOAL_PROB = 1.0 / 3.0
DEFAULT_OBJECT_WAYPOINT_GOAL_PROB = 1.0 / 6.0
OPENPI_IMAGE_SIZE = (224, 224)


@dataclasses.dataclass(frozen=True, slots=True)
class GoalReference:
    """Pointer from a flattened goal index back into the sample JSONL file."""

    byte_offset: int
    goal_kind: GoalKind
    goal_index: int


@dataclasses.dataclass(frozen=True)
class AstarNavigationDatasetConfig:
    """Configuration for :class:`AstarNavigationDataset`.

    ``image_size`` and ``goal_image_size`` are optional. When left as ``None``,
    the dataset returns the original image/crop size from disk. Batch loaders
    that stack arrays should set fixed sizes.
    """

    data_root: Path = DEFAULT_DATA_ROOT
    manifest_path: Path | None = None
    split: Literal["train", "eval", "val", "test", "all"] = "train"
    split_ids_path: Path | None = None
    include_object_goals: bool = True
    include_sampled_goals: bool = True
    action_horizon: int = 50
    path_stride: int = 10
    path_start_index: int = 0
    image_size: tuple[int, int] | None = None
    goal_image_size: tuple[int, int] | None = None
    current_pose_state: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.0)
    load_esdf: bool = True
    max_records: int | None = None


@dataclasses.dataclass(frozen=True)
class GoalModalityConfig:
    """Training-time goal-conditioning mask probabilities.

    Object goals may expose multiple modalities at once. Sampled goals always
    expose only the waypoint, because they do not have object text or crops.
    """

    object_text_prob: float = DEFAULT_OBJECT_TEXT_GOAL_PROB
    object_image_prob: float = DEFAULT_OBJECT_IMAGE_GOAL_PROB
    object_waypoint_prob: float = DEFAULT_OBJECT_WAYPOINT_GOAL_PROB

    def __post_init__(self) -> None:
        for name, value in (
            ("object_text_prob", self.object_text_prob),
            ("object_image_prob", self.object_image_prob),
            ("object_waypoint_prob", self.object_waypoint_prob),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")


@dataclasses.dataclass(frozen=True, slots=True)
class GoalModalityMask:
    text: bool
    image: bool
    waypoint: bool


class AstarNavigationDataset:
    """Random-access dataset over all planned object and sampled goals."""

    def __init__(self, config: AstarNavigationDatasetConfig | None = None, **kwargs: Any):
        if config is None:
            config = AstarNavigationDatasetConfig(**kwargs)
        elif kwargs:
            config = dataclasses.replace(config, **kwargs)

        if config.action_horizon <= 0:
            raise ValueError("action_horizon must be positive.")
        if config.path_stride <= 0:
            raise ValueError("path_stride must be positive.")

        self.config = config
        self.data_root = Path(config.data_root).expanduser().resolve()
        self.manifest_path = (
            Path(config.manifest_path).expanduser().resolve()
            if config.manifest_path is not None
            else self.data_root / "dataset_manifest.json"
        )
        self.samples_path, manifest = self._resolve_samples_path(self.manifest_path)
        self._clip_manifests = self._clip_manifest_map(manifest)
        self._split_ids_path = self._resolve_split_ids_path(config.split_ids_path, config.split)
        self._split_ids = self._load_split_ids(self._split_ids_path)
        self._index = self._build_index()
        self._samples_handle = None

        if not self._index:
            raise ValueError("No usable ASTAR goals were found for the requested split.")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self._index[int(index)]
        record = self._read_record_at(ref.byte_offset)
        goals_key = "object_goals" if ref.goal_kind == "object" else "sampled_goals"
        goal = record[goals_key][ref.goal_index]
        is_object_goal = ref.goal_kind == "object"

        front_image_path = self._resolve_path(record["image_path"], record)
        front_image = self._load_image(front_image_path, self.config.image_size)

        bbox = np.zeros((4,), dtype=np.float32)
        goal_image = None
        if is_object_goal:
            bbox = np.asarray(goal.get("bbox") or [0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            goal_image = self._crop_goal_image(front_image_path, bbox, self.config.goal_image_size)

        plan_arrays = self._load_goal_path(record, goal)
        path_xytheta, path_mask = self._downsample_path(plan_arrays["path_xytheta"])
        path_xy, xy_mask = self._downsample_path(plan_arrays["path_xy"])
        actions_vw, vw_mask = self._downsample_path(plan_arrays["actions_vw"])
        actions_xycossin = self._xytheta_to_xycossin(path_xytheta)

        esdf_config = self._esdf_config(record)
        sample: dict[str, Any] = {
            "front_image": front_image,
            "front_image_path": front_image_path.as_posix(),
            "goal_image": goal_image,
            "goal_image_mask": np.bool_(goal_image is not None),
            "state": np.asarray(self.config.current_pose_state, dtype=np.float32),
            "prompt": self._prompt_for_goal(goal, ref.goal_kind),
            "actions": actions_xycossin.astype(np.float32, copy=False),
            "actions_vw": actions_vw.astype(np.float32, copy=False),
            "path_xytheta": path_xytheta.astype(np.float32, copy=False),
            "path_xy": path_xy.astype(np.float32, copy=False),
            "path_step_mask": (path_mask & xy_mask & vw_mask).astype(bool, copy=False),
            "goal_xy": np.asarray(goal["goal_xy_m"][:2], dtype=np.float32),
            "is_object_goal": np.bool_(is_object_goal),
            "goal_kind": ref.goal_kind,
            "goal_index": np.asarray(ref.goal_index, dtype=np.int32),
            "bbox": bbox,
            "sample_id": str(record.get("sample_id", "")),
            "clip_id": str(record.get("clip_id", "")),
            "label": str(goal.get("label") or ""),
            "raw_label": str(goal.get("raw_label") or ""),
            "caption": str(goal.get("caption") or ""),
            "path_plan_status": str((goal.get("path_plan") or {}).get("status") or ""),
            "esdf_path": self._resolve_esdf_path(record).as_posix(),
            "esdf_x_min": np.asarray(esdf_config["x_min"], dtype=np.float32),
            "esdf_y_min": np.asarray(esdf_config["y_min"], dtype=np.float32),
            "esdf_resolution": np.asarray(esdf_config["resolution"], dtype=np.float32),
            "robot_radius_m": np.asarray(esdf_config["robot_radius"], dtype=np.float32),
        }
        if self.config.load_esdf:
            sample["esdf"] = np.load(sample["esdf_path"]).astype(np.float32, copy=False)
        return sample

    @property
    def goal_references(self) -> Sequence[GoalReference]:
        return self._index

    @property
    def object_indices(self) -> np.ndarray:
        return np.asarray(
            [index for index, ref in enumerate(self._index) if ref.goal_kind == "object"],
            dtype=np.int64,
        )

    @property
    def sampled_indices(self) -> np.ndarray:
        return np.asarray(
            [index for index, ref in enumerate(self._index) if ref.goal_kind == "sampled"],
            dtype=np.int64,
        )

    def close(self) -> None:
        if self._samples_handle is not None:
            self._samples_handle.close()
            self._samples_handle = None

    def _resolve_samples_path(self, manifest_path: Path) -> tuple[Path, dict[str, Any] | None]:
        if not manifest_path.exists():
            raise FileNotFoundError(f"ASTAR manifest not found: {manifest_path}")

        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except json.JSONDecodeError:
            return manifest_path, None

        samples_path = manifest.get("samples_path")
        if samples_path is None:
            return manifest_path, manifest
        return self._resolve_manifest_relative_path(samples_path, manifest_path.parent), manifest

    def _resolve_manifest_relative_path(self, path: str | os.PathLike[str], root: Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return (root / candidate).resolve()

    def _clip_manifest_map(self, manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
        clips = (manifest or {}).get("clips") or []
        return {
            str(clip["clip_id"]): clip
            for clip in clips
            if isinstance(clip, Mapping) and clip.get("clip_id") is not None
        }

    def _resolve_split_ids_path(
        self,
        split_ids_path: Path | None,
        split: str,
    ) -> Path | None:
        if split == "all":
            return None
        if split_ids_path is not None:
            candidate = Path(split_ids_path).expanduser()
            if not candidate.is_absolute():
                candidate = self.data_root / candidate
            candidate = candidate.resolve()
            if not candidate.exists():
                raise FileNotFoundError(f"ASTAR split IDs file not found: {candidate}")
            return candidate

        candidate = self.data_root / "splits" / f"{split}_clip_ids.txt"
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(
            "ASTAR split IDs file is required for split "
            f"{split!r}. Pass split_ids_path or create {candidate}."
        )

    def _load_split_ids(self, split_ids_path: Path | None) -> set[str] | None:
        if split_ids_path is None:
            return None
        split_ids = set()
        with split_ids_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                value = line.strip()
                if not value or value.startswith("#"):
                    continue
                split_ids.add(value)
        if not split_ids:
            raise ValueError(f"ASTAR split IDs file is empty: {split_ids_path}")
        return split_ids

    def _build_index(self) -> list[GoalReference]:
        references: list[GoalReference] = []
        with self.samples_path.open("rb") as handle:
            records_seen = 0
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue

                record = json.loads(line)
                records_seen += 1
                if self.config.max_records is not None and records_seen > self.config.max_records:
                    break
                if not self._record_in_split(record):
                    continue

                if self.config.include_object_goals:
                    for goal_index, goal in enumerate(record.get("object_goals") or []):
                        if self._usable_goal(goal):
                            references.append(GoalReference(offset, "object", goal_index))

                if self.config.include_sampled_goals:
                    for goal_index, goal in enumerate(record.get("sampled_goals") or []):
                        if self._usable_goal(goal):
                            references.append(GoalReference(offset, "sampled", goal_index))

        return references

    def _record_in_split(self, record: Mapping[str, Any]) -> bool:
        split = self.config.split
        if split == "all":
            return True
        if split in {"train", "eval", "val", "test"}:
            assert self._split_ids is not None
            return str(record.get("clip_id", "")) in self._split_ids
        raise ValueError(f"Unknown split: {split!r}")

    def _usable_goal(self, goal: Any) -> bool:
        if not isinstance(goal, Mapping):
            return False

        goal_xy = goal.get("goal_xy_m")
        if not isinstance(goal_xy, Sequence) or isinstance(goal_xy, str) or len(goal_xy) < 2:
            return False
        if goal_xy[0] is None or goal_xy[1] is None:
            return False

        plan = goal.get("path_plan")
        if not isinstance(plan, Mapping):
            return False
        if plan.get("path_data_path") is None:
            return False

        required_arrays = ("path_xytheta_m", "path_xy_m", "actions_vw")
        return all(_plan_key_available(plan, key) for key in required_arrays)

    def _read_record_at(self, byte_offset: int) -> dict[str, Any]:
        if self._samples_handle is None:
            self._samples_handle = self.samples_path.open("rb")
        self._samples_handle.seek(byte_offset)
        return json.loads(self._samples_handle.readline())

    def _resolve_path(self, path: str | os.PathLike[str], record: Mapping[str, Any]) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate

        direct = self.data_root / candidate
        if direct.exists():
            return direct

        clip_id = record.get("clip_id")
        if clip_id:
            in_clip = self.data_root / str(clip_id) / candidate
            if in_clip.exists():
                return in_clip
        return direct

    def _resolve_esdf_path(self, record: Mapping[str, Any]) -> Path:
        path_generation = record.get("path_generation")
        if isinstance(path_generation, Mapping):
            esdf_path_used = path_generation.get("esdf_path_used")
            if esdf_path_used is not None:
                resolved = self._resolve_path(esdf_path_used, record)
                if resolved.exists():
                    return resolved

        return self._resolve_path(str(record["esdf_path"]), record)

    def _load_image(self, image_path: Path, output_size: tuple[int, int] | None) -> np.ndarray:
        with _open_rgb_image(image_path) as image:
            if output_size is not None:
                image = _resize_with_pad(image, output_size)
            return _pil_to_uint8(image)

    def _crop_goal_image(
        self,
        image_path: Path,
        bbox: np.ndarray,
        output_size: tuple[int, int] | None,
    ) -> np.ndarray | None:
        with _open_rgb_image(image_path) as image:
            crop_box = _bbox_to_pixel_box(bbox, image.size)
            if crop_box is None:
                return None
            crop = image.crop(crop_box)
            if output_size is not None:
                crop = _resize_with_pad(crop, output_size)
            return _pil_to_uint8(crop)

    def _load_goal_path(
        self,
        record: Mapping[str, Any],
        goal: Mapping[str, Any],
    ) -> dict[str, np.ndarray]:
        plan = goal["path_plan"]
        path_file = self._resolve_path(plan["path_data_path"], record)
        if not path_file.exists():
            raise FileNotFoundError(f"Path data file not found: {path_file}")

        with np.load(path_file) as data:
            return {
                "path_xytheta": np.asarray(
                    data[_plan_array_key(plan, "path_xytheta_m")],
                    dtype=np.float32,
                ),
                "path_xy": np.asarray(data[_plan_array_key(plan, "path_xy_m")], dtype=np.float32),
                "actions_vw": np.asarray(
                    data[_plan_array_key(plan, "actions_vw")],
                    dtype=np.float32,
                ),
            }

    def _downsample_path(self, array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        if array.shape[0] == 0:
            raise ValueError("Cannot downsample an empty path array.")

        indices = (
            self.config.path_start_index
            + np.arange(self.config.action_horizon) * self.config.path_stride
        )
        mask = indices < array.shape[0]
        clipped = np.clip(indices, 0, array.shape[0] - 1)
        return array[clipped], mask.astype(bool, copy=False)

    def _xytheta_to_xycossin(self, path_xytheta: np.ndarray) -> np.ndarray:
        theta = path_xytheta[:, 2]
        return np.stack(
            (
                path_xytheta[:, 0],
                path_xytheta[:, 1],
                np.cos(theta),
                np.sin(theta),
            ),
            axis=-1,
        )

    def _esdf_config(self, record: Mapping[str, Any]) -> dict[str, float]:
        clip_id = str(record.get("clip_id", ""))
        clip_manifest = self._clip_manifests.get(clip_id)
        if clip_manifest is None:
            manifest_path = self._resolve_path(
                record.get("clip_manifest_path", "manifest.json"),
                record,
            )
            if manifest_path.exists():
                with manifest_path.open("r", encoding="utf-8") as handle:
                    clip_manifest = json.load(handle)
                self._clip_manifests[clip_id] = clip_manifest
            else:
                clip_manifest = {}

        esdf_config = clip_manifest.get("esdf_config") or {}
        x_range = esdf_config.get("x_range_m") or [0.0, 25.0]
        y_range = esdf_config.get("y_range_m") or [-10.0, 10.0]
        robot_radius = float(esdf_config.get("robot_radius_m", 0.5))
        if not np.isfinite(robot_radius) or robot_radius < 0.0:
            raise ValueError(f"Invalid robot_radius_m for clip {clip_id}: {robot_radius}")
        return {
            "x_min": float(x_range[0]),
            "y_min": float(y_range[0]),
            "resolution": float(esdf_config.get("resolution_m", 0.02)),
            "robot_radius": robot_radius,
        }

    def _prompt_for_goal(self, goal: Mapping[str, Any], goal_kind: GoalKind) -> str:
        if goal_kind == "object":
            label = str(goal.get("label") or goal.get("raw_label") or "object")
            caption = str(goal.get("caption") or "").strip()
            if caption:
                return f"go to {label} described by {caption}"
            return f"go to {label}"

        return "go to the waypoint goal"


class AstarNavigationDataLoader:
    """Batch loader with optional 5/6 object and 1/6 sampled-goal batches."""

    def __init__(
        self,
        dataset: AstarNavigationDataset,
        *,
        batch_size: int,
        sampled_goal_fraction: float | None = DEFAULT_SAMPLED_GOAL_FRACTION,
        goal_modality_config: GoalModalityConfig | None = None,
        output_format: OutputFormat = "astar",
        max_token_len: int = 200,
        discrete_state_input: bool = True,
        shuffle: bool = True,
        seed: int = 0,
        sharding: Any | None = None,
        num_batches: int | None = None,
        return_metric_tensors: bool = True,
        make_jax_arrays: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if sampled_goal_fraction is not None and not 0.0 <= sampled_goal_fraction <= 1.0:
            raise ValueError("sampled_goal_fraction must be in [0, 1].")
        if output_format not in {"astar", "openpi"}:
            raise ValueError(f"Unknown output_format: {output_format!r}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.sampled_goal_fraction = sampled_goal_fraction
        self.goal_modality_config = goal_modality_config or GoalModalityConfig()
        self.output_format = output_format
        self.shuffle = shuffle
        self.seed = seed
        self.num_batches = num_batches
        self.return_metric_tensors = return_metric_tensors
        self.make_jax_arrays = make_jax_arrays
        self._object_indices = dataset.object_indices
        self._sampled_indices = dataset.sampled_indices
        self._all_indices = np.arange(len(dataset), dtype=np.int64)
        self._model = None
        self._config = None
        self._tokenizer = None
        self._jax = None
        self._sharding = sharding

        if output_format == "openpi":
            if dataset.config.image_size is None or dataset.config.goal_image_size is None:
                raise ValueError(
                    "OpenPI batches require fixed image_size and goal_image_size "
                    "so arrays can be stacked."
                )
            self._model, self._config, tokenizer_module = _import_openpi_modules()
            self._tokenizer = tokenizer_module.PaligemmaTokenizer(max_token_len)
            self._discrete_state_input = discrete_state_input
            if make_jax_arrays:
                self._jax = _import_jax()
                if self._sharding is None:
                    self._sharding = self._jax.sharding.NamedSharding(
                        self._jax.sharding.Mesh(self._jax.devices(), ("B",)),
                        self._jax.sharding.PartitionSpec("B"),
                    )

    def data_config(self):
        if self._config is None:
            _, self._config, _ = _import_openpi_modules()
        return self._config.DataConfig(repo_id=None, asset_id=None, norm_stats=None)

    def __iter__(self) -> Iterator[Any]:
        rng = np.random.default_rng(self.seed)
        yielded = 0
        index_iter = self._batch_indices(rng)
        while True:
            if self.num_batches is not None and yielded >= self.num_batches:
                return
            samples = [self.dataset[int(index)] for index in next(index_iter)]
            yielded += 1
            if self.output_format == "openpi":
                yield self._openpi_batch(samples, rng)
            else:
                yield _stack_tree(samples)

    def _batch_indices(self, rng: np.random.Generator) -> Iterator[np.ndarray]:
        if self.sampled_goal_fraction is None:
            yield from self._plain_batch_indices(rng)
            return
        yield from self._mixed_batch_indices(rng)

    def _plain_batch_indices(self, rng: np.random.Generator) -> Iterator[np.ndarray]:
        order = self._all_indices.copy()
        position = len(order)
        while True:
            if position + self.batch_size > len(order):
                order = self._all_indices.copy()
                if self.shuffle:
                    rng.shuffle(order)
                position = 0
            batch = order[position : position + self.batch_size]
            position += self.batch_size
            yield batch

    def _mixed_batch_indices(self, rng: np.random.Generator) -> Iterator[np.ndarray]:
        sampled_count = int(round(self.batch_size * float(self.sampled_goal_fraction)))
        sampled_count = min(max(sampled_count, 0), self.batch_size)
        if self._sampled_indices.size == 0:
            sampled_count = 0
        if self._object_indices.size == 0:
            sampled_count = self.batch_size
        object_count = self.batch_size - sampled_count

        object_stream = (
            _IndexStream(self._object_indices, rng, shuffle=self.shuffle) if object_count else None
        )
        sampled_stream = (
            _IndexStream(self._sampled_indices, rng, shuffle=self.shuffle)
            if sampled_count
            else None
        )
        while True:
            parts = []
            if object_count:
                assert object_stream is not None
                parts.append(object_stream.take(object_count))
            if sampled_count:
                assert sampled_stream is not None
                parts.append(sampled_stream.take(sampled_count))
            batch = np.concatenate(parts)
            if self.shuffle:
                rng.shuffle(batch)
            yield batch

    def _openpi_batch(self, samples: Sequence[dict[str, Any]], rng: np.random.Generator) -> Any:
        assert self._model is not None
        assert self._tokenizer is not None

        data_items = []
        metric_items = []
        for sample in samples:
            goal_mask = self._sample_goal_modality_mask(sample, rng)
            data, metrics = self._openpi_item(sample, goal_mask)
            data_items.append(data)
            metric_items.append(metrics)

        data = _stack_tree(data_items)
        metric_tensors = _stack_tree(metric_items)
        if self.make_jax_arrays:
            data = self._to_sharded_arrays(data)
            metric_tensors = self._to_sharded_arrays(metric_tensors)

        observation = self._model.Observation.from_dict(data)
        actions = data["actions"]
        if self.return_metric_tensors:
            return observation, actions, metric_tensors
        return observation, actions

    def _sample_goal_modality_mask(
        self,
        sample: Mapping[str, Any],
        rng: np.random.Generator,
    ) -> GoalModalityMask:
        if not bool(sample["is_object_goal"]):
            return GoalModalityMask(text=False, image=False, waypoint=True)

        config = self.goal_modality_config
        text = bool(rng.random() < config.object_text_prob)
        image = bool(rng.random() < config.object_image_prob) and bool(sample["goal_image_mask"])
        waypoint = bool(rng.random() < config.object_waypoint_prob)

        if not (text or image or waypoint):
            text = True
        return GoalModalityMask(text=text, image=image, waypoint=waypoint)

    def _prompt_for_goal_mask(
        self,
        sample: Mapping[str, Any],
        goal_mask: GoalModalityMask,
    ) -> str:
        if not bool(sample["is_object_goal"]):
            return "go to the provided waypoint"

        clauses = []
        if goal_mask.text:
            label = str(sample.get("label") or sample.get("raw_label") or "object").strip()
            caption = str(sample.get("caption") or "").strip()
            if caption:
                clauses.append(f"go to {label} described by {caption}")
            else:
                clauses.append(f"go to {label}")
        else:
            clauses.append("go to the object")

        if goal_mask.image:
            clauses.append("shown in the goal image")
        if goal_mask.waypoint:
            clauses.append("at the provided waypoint")
        return " ".join(clauses)

    def _openpi_item(
        self,
        sample: Mapping[str, Any],
        goal_mask: GoalModalityMask,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        assert self._tokenizer is not None
        front_image = np.asarray(sample["front_image"], dtype=np.uint8)
        state = np.asarray(sample["state"], dtype=np.float32)
        tokenizer_state = state if self._discrete_state_input else None
        prompt = self._prompt_for_goal_mask(sample, goal_mask)
        tokens, token_mask = self._tokenizer.tokenize(prompt, tokenizer_state)

        zero_camera = np.zeros_like(front_image)
        goal_image = sample.get("goal_image")
        goal_image_available = goal_image is not None and bool(sample["goal_image_mask"])
        if goal_image is None:
            goal_image = np.zeros_like(front_image)
        goal_xy = np.asarray(sample["goal_xy"], dtype=np.float32)

        data = {
            "image": {
                "base_0_rgb": front_image,
                "left_wrist_0_rgb": zero_camera,
                "right_wrist_0_rgb": zero_camera,
            },
            "image_mask": {
                "base_0_rgb": np.bool_(True),
                "left_wrist_0_rgb": np.bool_(False),
                "right_wrist_0_rgb": np.bool_(False),
            },
            "state": state,
            "tokenized_prompt": tokens.astype(np.int32, copy=False),
            "tokenized_prompt_mask": token_mask.astype(bool, copy=False),
            "goal_waypoints": goal_xy[None, :],
            "goal_waypoint_mask": np.asarray([goal_mask.waypoint], dtype=bool),
            "goal_image": {
                "crop": np.asarray(goal_image, dtype=np.uint8),
            },
            "goal_image_mask": {
                "crop": np.asarray(goal_mask.image, dtype=bool),
            },
            "actions": np.asarray(sample["actions"], dtype=np.float32),
        }

        metric_tensors = {
            "goal_xy": goal_xy,
            "actions_vw": np.asarray(sample["actions_vw"], dtype=np.float32),
            "path_xy": np.asarray(sample["path_xy"], dtype=np.float32),
            "path_xytheta": np.asarray(sample["path_xytheta"], dtype=np.float32),
            "path_step_mask": np.asarray(sample["path_step_mask"], dtype=bool),
            "is_object_goal": np.asarray(sample["is_object_goal"], dtype=bool),
            "goal_image": np.asarray(goal_image, dtype=np.uint8),
            "goal_image_available_mask": np.asarray(goal_image_available, dtype=bool),
            "goal_text_condition_mask": np.asarray(goal_mask.text, dtype=bool),
            "goal_image_condition_mask": np.asarray(goal_mask.image, dtype=bool),
            "goal_waypoint_condition_mask": np.asarray(goal_mask.waypoint, dtype=bool),
            "bbox": np.asarray(sample["bbox"], dtype=np.float32),
            "goal_index": np.asarray(sample["goal_index"], dtype=np.int32),
            "esdf_x_min": np.asarray(sample["esdf_x_min"], dtype=np.float32),
            "esdf_y_min": np.asarray(sample["esdf_y_min"], dtype=np.float32),
            "esdf_resolution": np.asarray(sample["esdf_resolution"], dtype=np.float32),
            "robot_radius_m": np.asarray(sample["robot_radius_m"], dtype=np.float32),
        }
        if "esdf" in sample:
            metric_tensors["esdf"] = np.asarray(sample["esdf"], dtype=np.float32)
        return data, metric_tensors

    def _to_sharded_arrays(self, tree: Any) -> Any:
        assert self._jax is not None
        return self._jax.tree.map(
            lambda array: self._jax.make_array_from_process_local_data(self._sharding, array),
            tree,
        )


NavigationDataLoader = AstarNavigationDataLoader


def create_navigation_data_loader(
    config: Any,
    *,
    split: Literal["train", "eval", "val", "test", "all"] = "train",
    data_root: str | os.PathLike[str] | None = None,
    manifest_path: str | os.PathLike[str] | None = None,
    split_ids_path: str | os.PathLike[str] | None = None,
    sharding: Any | None = None,
    shuffle: bool | None = None,
    num_batches: int | None = None,
    output_format: OutputFormat = "openpi",
    return_metric_tensors: bool = True,
    load_esdf: bool = True,
    sampled_goal_fraction: float | None = DEFAULT_SAMPLED_GOAL_FRACTION,
    object_text_goal_prob: float = DEFAULT_OBJECT_TEXT_GOAL_PROB,
    object_image_goal_prob: float = DEFAULT_OBJECT_IMAGE_GOAL_PROB,
    object_waypoint_goal_prob: float = DEFAULT_OBJECT_WAYPOINT_GOAL_PROB,
    path_stride: int = 10,
    image_size: tuple[int, int] | None = None,
    goal_image_size: tuple[int, int] | None = None,
) -> AstarNavigationDataLoader:
    """Factory for the current training scripts.

    ``output_format="openpi"`` adapts the ASTAR sample to OpenPI's current
    observation tuple. The underlying dataset still indexes all goals and loads
    the full ESDF.
    """

    model_config = config.model
    if output_format == "openpi":
        image_size = image_size or OPENPI_IMAGE_SIZE
        goal_image_size = goal_image_size or OPENPI_IMAGE_SIZE

    dataset = AstarNavigationDataset(
        data_root=Path(data_root or os.environ.get("ASTAR_DATA_ROOT", DEFAULT_DATA_ROOT)),
        manifest_path=Path(manifest_path).expanduser().resolve() if manifest_path else None,
        split=split,
        split_ids_path=Path(split_ids_path).expanduser() if split_ids_path else None,
        action_horizon=int(model_config.action_horizon),
        path_stride=path_stride,
        image_size=image_size,
        goal_image_size=goal_image_size,
        load_esdf=load_esdf,
    )
    return AstarNavigationDataLoader(
        dataset,
        batch_size=int(config.batch_size),
        sampled_goal_fraction=sampled_goal_fraction,
        goal_modality_config=GoalModalityConfig(
            object_text_prob=object_text_goal_prob,
            object_image_prob=object_image_goal_prob,
            object_waypoint_prob=object_waypoint_goal_prob,
        ),
        output_format=output_format,
        max_token_len=int(model_config.max_token_len),
        discrete_state_input=bool(getattr(model_config, "discrete_state_input", False)),
        shuffle=(split == "train") if shuffle is None else shuffle,
        seed=int(getattr(config, "seed", 0)),
        sharding=sharding,
        num_batches=num_batches,
        return_metric_tensors=return_metric_tensors,
        make_jax_arrays=(output_format == "openpi"),
    )


class _IndexStream:
    def __init__(self, indices: np.ndarray, rng: np.random.Generator, *, shuffle: bool):
        if indices.size == 0:
            raise ValueError("Cannot create an index stream from an empty index set.")
        self._base_indices = np.asarray(indices, dtype=np.int64)
        self._rng = rng
        self._shuffle = shuffle
        self._order = np.empty((0,), dtype=np.int64)
        self._position = 0

    def take(self, count: int) -> np.ndarray:
        chunks = []
        remaining = count
        while remaining:
            if self._position >= self._order.size:
                self._order = self._base_indices.copy()
                if self._shuffle:
                    self._rng.shuffle(self._order)
                self._position = 0
            available = min(remaining, self._order.size - self._position)
            chunks.append(self._order[self._position : self._position + available])
            self._position += available
            remaining -= available
        return np.concatenate(chunks)


def _plan_key_available(plan: Mapping[str, Any], array_name: str) -> bool:
    return plan.get(f"{array_name}_key") is not None or plan.get("path_data_key") is not None


def _plan_array_key(plan: Mapping[str, Any], array_name: str) -> str:
    explicit = plan.get(f"{array_name}_key")
    if explicit is not None:
        return str(explicit)
    path_data_key = plan.get("path_data_key")
    if path_data_key is None:
        raise KeyError(f"Missing {array_name}_key and path_data_key in path_plan.")
    return f"{path_data_key}_{array_name}"


def _bbox_to_pixel_box(
    bbox: Sequence[float],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    width, height = image_size
    if len(bbox) < 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height

    left = int(np.floor(np.clip(min(x1, x2), 0, width - 1)))
    top = int(np.floor(np.clip(min(y1, y2), 0, height - 1)))
    right = int(np.ceil(np.clip(max(x1, x2), left + 1, width)))
    bottom = int(np.ceil(np.clip(max(y1, y2), top + 1, height)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _resize_with_pad(image: Any, output_size: tuple[int, int]) -> Any:
    from PIL import Image

    target_h, target_w = output_size
    image = image.copy()
    image.thumbnail((target_w, target_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    left = (target_w - image.width) // 2
    top = (target_h - image.height) // 2
    canvas.paste(image, (left, top))
    return canvas


def _open_rgb_image(path: Path) -> Any:
    from PIL import Image

    return Image.open(path).convert("RGB")


def _pil_to_uint8(image: Any) -> np.ndarray:
    return np.asarray(image, dtype=np.uint8)


def _stack_tree(items: Sequence[Any]) -> Any:
    first = items[0]
    if isinstance(first, Mapping):
        return {key: _stack_tree([item[key] for item in items]) for key in first}
    if first is None:
        return list(items)

    arrays = [np.asarray(item) for item in items]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        return list(items)
    return np.stack(arrays, axis=0)


def _import_openpi_modules():
    try:
        from openpi.models import model as _model
        from openpi.models import tokenizer as _tokenizer
        import openpi.training.config as _config
    except ImportError as exc:
        raise ImportError(
            "OpenPI output requires OpenPI to be installed. Run the ASTAR environment "
            "setup so the local third_party/openpi package is available."
        ) from exc
    return _model, _config, _tokenizer


def _import_jax():
    try:
        import jax
    except ImportError as exc:
        raise ImportError("make_jax_arrays=True requires jax.") from exc
    return jax
