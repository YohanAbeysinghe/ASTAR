"""Small, model-independent state container for energy-oracle aggregation.

The large observation/ESDF condition pool is deliberately not stored here.
It is reconstructed deterministically on resume and validated with stable
fingerprints.  Only visited paths and their once-queried oracle directions are
aggregated, which keeps the replay archive compact.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ARCHIVE_VERSION = 2


def stable_config_signature(values: dict[str, Any]) -> str:
    """Return a deterministic signature for resume-critical configuration."""
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deterministic_replay_slot(
    *, seed: int, global_step: int, num_rounds: int, num_condition_batches: int
) -> tuple[int, int]:
    """Select one replay round/batch without mutable RNG state.

    Deriving the draw from the optimizer step makes an interrupted/resumed run
    choose the same sequence as an uninterrupted run.
    """
    if num_rounds <= 0:
        raise ValueError("num_rounds must be positive.")
    if num_condition_batches <= 0:
        raise ValueError("num_condition_batches must be positive.")
    rng = np.random.default_rng(np.random.SeedSequence([seed, global_step]))
    return int(rng.integers(num_rounds)), int(rng.integers(num_condition_batches))


@dataclass(frozen=True)
class ConvergenceConfig:
    min_rounds: int = 3
    patience: int = 2
    median_path_change_m: float = 0.01
    p95_path_change_m: float = 0.03
    relative_energy_change: float = 1.0e-3
    oracle_grad_rms: float = 1.0e-4
    max_clearance_violation_rate: float = 0.05
    min_progress_ratio: float = 0.9

    def __post_init__(self) -> None:
        if self.min_rounds <= 0:
            raise ValueError("min_rounds must be positive.")
        if self.patience <= 0:
            raise ValueError("patience must be positive.")
        for name in (
            "median_path_change_m",
            "p95_path_change_m",
            "relative_energy_change",
            "oracle_grad_rms",
            "max_clearance_violation_rate",
            "min_progress_ratio",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if self.max_clearance_violation_rate > 1.0:
            raise ValueError("max_clearance_violation_rate must be at most 1.")
        if self.min_progress_ratio > 1.0:
            raise ValueError("min_progress_ratio must be at most 1.")


@dataclass
class ConvergenceTracker:
    config: ConvergenceConfig
    stable_rounds: int = 0

    def observe(self, round_number: int, metrics: dict[str, float]) -> tuple[bool, str]:
        """Update patience and return ``(converged, status)``.

        Motion alone is insufficient: a field that collapsed to zero would
        otherwise look converged.  We also require stable energy, a small raw
        oracle gradient, and acceptable task/safety metrics.
        """
        motion_stable = (
            metrics["path_change_median"] <= self.config.median_path_change_m
            and metrics["path_change_p95"] <= self.config.p95_path_change_m
        )
        energy_stable = (
            abs(metrics["relative_energy_improvement"])
            <= self.config.relative_energy_change
        )
        oracle_stationary = metrics["oracle_grad_rms"] <= self.config.oracle_grad_rms
        task_satisfactory = (
            metrics.get("clearance_violation_rate", metrics["collision_rate"])
            <= self.config.max_clearance_violation_rate
            and metrics.get("invalid_esdf_rate", 0.0) <= 1.0e-6
            and metrics["progress_ratio"] >= self.config.min_progress_ratio
        )
        eligible = round_number >= self.config.min_rounds
        stable = (
            eligible
            and motion_stable
            and energy_stable
            and oracle_stationary
            and task_satisfactory
        )

        if stable:
            self.stable_rounds += 1
        else:
            self.stable_rounds = 0

        if self.stable_rounds >= self.config.patience:
            return True, "converged"
        if (
            eligible
            and motion_stable
            and energy_stable
            and not (oracle_stationary and task_satisfactory)
        ):
            return False, "stalled"
        return False, "running"


@dataclass
class BenefitTracker:
    """Stop collection after no meaningful validation improvement for several rounds.

    Each metric has its own best-so-far reference. Small improvements accumulate
    until they exceed its threshold. A plateau can be unsafe or stalled; it is
    never classified as successful convergence. State is saved with replay.
    """

    min_rounds: int = 6
    patience: int = 4
    rate_delta: float = 0.01
    progress_delta: float = 0.02
    best: dict[str, float] = field(default_factory=dict)
    bad_rounds: int = 0

    def __post_init__(self) -> None:
        if self.min_rounds < 1 or self.patience < 1:
            raise ValueError("Benefit minimum rounds and patience must be positive.")
        if not (0.0 < self.rate_delta <= 1.0 and 0.0 < self.progress_delta <= 1.0):
            raise ValueError("Benefit thresholds must be in (0, 1].")

    def observe(self, round_number: int, metrics: dict[str, float]) -> bool:
        values = {
            "success_rate": metrics["success_rate"],
            "safe_success_rate": metrics["safe_success_rate"],
            # Bound the influence of pathological short-goal overshoots.
            "progress_ratio": metrics["bounded_progress_ratio"],
            "collision_rate": -metrics["collision_rate"],
            "clearance_violation_rate": -metrics.get(
                "clearance_violation_rate", metrics["collision_rate"]
            ),
            "invalid_coverage_rate": -metrics["invalid_coverage_rate"],
            "goal_retreat_segment_rate": -metrics["goal_retreat_segment_rate"],
        }
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError(
                "Nonfinite validation metrics cannot drive aggregation stopping."
            )
        improved = not self.best
        for key, value in values.items():
            delta = self.progress_delta if key == "progress_ratio" else self.rate_delta
            if key not in self.best or value >= self.best[key] + delta:
                self.best[key] = float(value)
                improved = True
        if improved or round_number < self.min_rounds:
            self.bad_rounds = 0
        else:
            self.bad_rounds += 1
        return self.bad_rounds >= self.patience


@dataclass
class AggregationReplay:
    """Factorized replay state for the outer DAgger-style loop."""

    config_signature: str
    condition_fingerprints: list[str]
    current_paths: np.ndarray
    round_index: int = 0
    update_in_round: int = 0
    convergence_streak: int = 0
    aggregation_stop_reason: str = ""
    replay_updates: int = 0
    validation_fingerprints: list[str] = field(default_factory=list)
    validation_records: list[dict[str, Any]] = field(default_factory=list)
    benefit_state: dict[str, Any] = field(default_factory=dict)
    visited_paths: list[np.ndarray] = field(default_factory=list)
    oracle_directions: list[np.ndarray] = field(default_factory=list)
    oracle_gradient_rms: list[np.ndarray] = field(default_factory=list)
    oracle_records: list[dict[str, float]] = field(default_factory=list)
    round_records: list[dict[str, float | int | str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.current_paths = np.asarray(self.current_paths, dtype=np.float32)
        self.validate()

    @property
    def labelled_rounds(self) -> int:
        return len(self.visited_paths)

    @property
    def num_condition_batches(self) -> int:
        return int(self.current_paths.shape[0])

    @property
    def converged(self) -> bool:
        return bool(
            self.round_records and self.round_records[-1].get("status") == "converged"
        )

    def append_current_round(
        self,
        oracle_directions: np.ndarray,
        oracle_gradient_rms: np.ndarray,
        oracle_record: dict[str, float],
    ) -> None:
        """Cache the current frontier and its oracle answer exactly once."""
        if self.aggregation_stop_reason:
            raise RuntimeError(
                "Aggregation is stopped; cached-replay training cannot add labels."
            )
        if self.labelled_rounds != self.round_index:
            raise RuntimeError(
                "Current aggregation round is already labelled; refusing to query/cache it twice."
            )
        labels = np.asarray(oracle_directions, dtype=np.float32)
        if labels.shape != self.current_paths.shape:
            raise ValueError(
                f"Oracle labels have shape {labels.shape}, expected {self.current_paths.shape}."
            )
        gradient_rms = np.asarray(oracle_gradient_rms, dtype=np.float32)
        expected_gradient_shape = self.current_paths.shape[:2]
        if gradient_rms.shape != expected_gradient_shape:
            raise ValueError(
                f"Oracle gradient RMS has shape {gradient_rms.shape}, "
                f"expected {expected_gradient_shape}."
            )
        self.visited_paths.append(self.current_paths.copy())
        self.oracle_directions.append(labels.copy())
        self.oracle_gradient_rms.append(gradient_rms.copy())
        self.oracle_records.append(
            {key: float(value) for key, value in oracle_record.items()}
        )
        self.validate()

    def replay_batch(
        self, round_index: int, condition_batch: int
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.visited_paths[round_index][condition_batch],
            self.oracle_directions[round_index][condition_batch],
        )

    def finish_round(
        self,
        next_paths: np.ndarray,
        record: dict[str, float | int | str],
        *,
        convergence_streak: int,
    ) -> None:
        if self.labelled_rounds != self.round_index + 1:
            raise RuntimeError(
                "Cannot finish a round before its oracle labels are cached."
            )
        next_paths = np.asarray(next_paths, dtype=np.float32)
        if next_paths.shape != self.current_paths.shape:
            raise ValueError(
                f"Next paths have shape {next_paths.shape}, expected {self.current_paths.shape}."
            )
        self.current_paths = next_paths.copy()
        self.round_records.append(dict(record))
        self.round_index += 1
        self.update_in_round = 0
        self.convergence_streak = int(convergence_streak)
        self.validate()

    def validate(self) -> None:
        if self.current_paths.ndim != 4:
            raise ValueError(
                "current_paths must have shape "
                "[condition_batch, batch*particles, horizon, action_dim]."
            )
        if self.current_paths.shape[1] == 0:
            raise ValueError("current_paths contains an empty particle batch.")
        if not np.isfinite(self.current_paths).all():
            raise ValueError("current_paths contains NaN or infinity.")
        if len(self.condition_fingerprints) != self.current_paths.shape[0]:
            raise ValueError(
                "Condition fingerprint count does not match the condition pool."
            )
        replay_lengths = {
            len(self.visited_paths),
            len(self.oracle_directions),
            len(self.oracle_gradient_rms),
            len(self.oracle_records),
        }
        if len(replay_lengths) != 1:
            raise ValueError(
                "Visited-path, oracle-label and oracle-diagnostic lengths differ."
            )
        if len(self.visited_paths) not in {self.round_index, self.round_index + 1}:
            raise ValueError("Replay phase is inconsistent with round_index.")
        if len(self.round_records) != self.round_index:
            raise ValueError("Completed round records do not match round_index.")
        if self.update_in_round < 0:
            raise ValueError("update_in_round must be non-negative.")
        if self.replay_updates < 0:
            raise ValueError("replay_updates must be non-negative.")
        if self.replay_updates and not self.aggregation_stop_reason:
            raise ValueError("Post-aggregation updates require a recorded stop reason.")
        if self.labelled_rounds == self.round_index and self.update_in_round != 0:
            raise ValueError(
                "An unlabelled frontier cannot have completed replay updates."
            )
        for states, labels, gradient_rms in zip(
            self.visited_paths,
            self.oracle_directions,
            self.oracle_gradient_rms,
            strict=True,
        ):
            if (
                states.shape != self.current_paths.shape
                or labels.shape != self.current_paths.shape
            ):
                raise ValueError(
                    "A replay round has a shape inconsistent with current_paths."
                )
            if gradient_rms.shape != self.current_paths.shape[:2]:
                raise ValueError(
                    "An oracle-gradient diagnostic has an inconsistent shape."
                )
            if (
                not np.isfinite(states).all()
                or not np.isfinite(labels).all()
                or not np.isfinite(gradient_rms).all()
            ):
                raise ValueError("Replay contains NaN or infinity.")

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Atomically save the replay/frontier state without pickle."""
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "archive_version": ARCHIVE_VERSION,
            "config_signature": self.config_signature,
            "condition_fingerprints": self.condition_fingerprints,
            "round_index": self.round_index,
            "update_in_round": self.update_in_round,
            "convergence_streak": self.convergence_streak,
            "aggregation_stop_reason": self.aggregation_stop_reason,
            "replay_updates": self.replay_updates,
            "validation_fingerprints": self.validation_fingerprints,
            "validation_records": self.validation_records,
            "benefit_state": self.benefit_state,
            "oracle_records": self.oracle_records,
            "round_records": self.round_records,
        }
        visited = (
            np.stack(self.visited_paths, axis=0)
            if self.visited_paths
            else np.empty((0, *self.current_paths.shape), dtype=np.float32)
        )
        labels = (
            np.stack(self.oracle_directions, axis=0)
            if self.oracle_directions
            else np.empty((0, *self.current_paths.shape), dtype=np.float32)
        )
        gradient_rms = (
            np.stack(self.oracle_gradient_rms, axis=0)
            if self.oracle_gradient_rms
            else np.empty((0, *self.current_paths.shape[:2]), dtype=np.float32)
        )
        temporary = output.with_name(f".{output.name}.tmp.npz")
        np.savez_compressed(
            temporary,
            current_paths=self.current_paths,
            visited_paths=visited,
            oracle_directions=labels,
            oracle_gradient_rms=gradient_rms,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        os.replace(temporary, output)
        return output

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        expected_config_signature: str,
        expected_condition_fingerprints: list[str],
    ) -> "AggregationReplay":
        source = Path(path)
        with np.load(source, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata.get("archive_version") != ARCHIVE_VERSION:
                raise ValueError(
                    f"Unsupported replay archive version {metadata.get('archive_version')!r}."
                )
            if metadata["config_signature"] != expected_config_signature:
                raise ValueError(
                    "Resume configuration differs from the replay archive."
                )
            if metadata["condition_fingerprints"] != expected_condition_fingerprints:
                raise ValueError("Condition pool differs from the replay archive.")
            visited = np.asarray(archive["visited_paths"], dtype=np.float32)
            labels = np.asarray(archive["oracle_directions"], dtype=np.float32)
            gradient_rms = np.asarray(archive["oracle_gradient_rms"], dtype=np.float32)
            replay = cls(
                config_signature=metadata["config_signature"],
                condition_fingerprints=list(metadata["condition_fingerprints"]),
                current_paths=np.asarray(archive["current_paths"], dtype=np.float32),
                round_index=int(metadata["round_index"]),
                update_in_round=int(metadata["update_in_round"]),
                convergence_streak=int(metadata["convergence_streak"]),
                aggregation_stop_reason=str(
                    metadata.get("aggregation_stop_reason", "")
                ),
                replay_updates=int(metadata.get("replay_updates", 0)),
                validation_fingerprints=list(
                    metadata.get("validation_fingerprints", [])
                ),
                validation_records=list(metadata.get("validation_records", [])),
                benefit_state=dict(metadata.get("benefit_state", {})),
                visited_paths=[
                    visited[index].copy() for index in range(visited.shape[0])
                ],
                oracle_directions=[
                    labels[index].copy() for index in range(labels.shape[0])
                ],
                oracle_gradient_rms=[
                    gradient_rms[index].copy() for index in range(gradient_rms.shape[0])
                ],
                oracle_records=list(metadata["oracle_records"]),
                round_records=list(metadata["round_records"]),
            )
        replay.validate()
        return replay
