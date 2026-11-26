"""CSV replay utilities for hardware-free ESKF validation."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Iterable

import numpy as np

from modules.navigation.eskf import ESKF


@dataclass
class SensorSample:
    timestamp: float
    acceleration_body: np.ndarray
    angular_velocity_body: np.ndarray
    attitude_zyx_degrees: np.ndarray
    dvl_velocity_body: np.ndarray
    dvl_velocity_valid: bool
    depth: float
    depth_valid: bool
    dvl_position: np.ndarray | None = None


@dataclass
class ReplayState:
    timestamp: float
    position: np.ndarray
    velocity: np.ndarray
    attitude_zyx_degrees: np.ndarray
    covariance_trace: float
    valid: bool


CSV_FIELDS = [
    "timestamp",
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z",
    "yaw", "pitch", "roll",
    "dvl_vx", "dvl_vy", "dvl_vz", "dvl_valid",
    "depth", "depth_valid",
    "dvl_x", "dvl_y", "dvl_z", "dvl_position_valid",
]


def _float(row: dict[str, str], name: str, default: float = 0.0) -> float:
    value = row.get(name, "")
    return default if value in (None, "") else float(value)


def _bool(row: dict[str, str], name: str, default: bool = False) -> bool:
    value = row.get(name, "")
    return default if value in (None, "") else value.strip().lower() in {"1", "true", "yes"}


def read_samples(path: str | Path) -> Iterable[SensorSample]:
    """Read the documented CSV schema and yield normalized sensor samples."""
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = {field for field in ("timestamp", "yaw", "pitch", "roll") if field not in (reader.fieldnames or [])}
        if missing:
            raise ValueError(f"Replay CSV is missing required fields: {sorted(missing)}")
        for row in reader:
            position = None
            if _bool(row, "dvl_position_valid"):
                position = np.array([_float(row, "dvl_x"), _float(row, "dvl_y"), _float(row, "dvl_z")])
            yield SensorSample(
                timestamp=_float(row, "timestamp"),
                acceleration_body=np.array([_float(row, "accel_x"), _float(row, "accel_y"), _float(row, "accel_z")]),
                angular_velocity_body=np.array([_float(row, "gyro_x"), _float(row, "gyro_y"), _float(row, "gyro_z")]),
                attitude_zyx_degrees=np.array([_float(row, "yaw"), _float(row, "pitch"), _float(row, "roll")]),
                dvl_velocity_body=np.array([_float(row, "dvl_vx"), _float(row, "dvl_vy"), _float(row, "dvl_vz")]),
                dvl_velocity_valid=_bool(row, "dvl_valid"),
                depth=_float(row, "depth"),
                depth_valid=_bool(row, "depth_valid"),
                dvl_position=position,
            )


def run_replay(samples: Iterable[SensorSample]) -> list[ReplayState]:
    """Run samples through the ESKF and return one state per sample."""
    estimator = ESKF()
    results: list[ReplayState] = []
    initialized = False
    for sample in samples:
        if not initialized:
            if sample.dvl_position is not None:
                estimator.initialize_position(sample.dvl_position)
            estimator.initialize_attitude(sample.attitude_zyx_degrees)
            initialized = True
        estimator.predict(sample.acceleration_body, sample.angular_velocity_body, sample.timestamp)
        estimator.update_ahrs(sample.attitude_zyx_degrees)
        if sample.dvl_velocity_valid:
            estimator.update_dvl_velocity(sample.dvl_velocity_body)
        if sample.depth_valid:
            estimator.update_depth(sample.depth)
        estimator.state.valid = initialized
        results.append(ReplayState(
            timestamp=sample.timestamp,
            position=estimator.state.position.copy(),
            velocity=estimator.state.velocity.copy(),
            attitude_zyx_degrees=estimator.state.euler_zyx_degrees.copy(),
            covariance_trace=float(np.trace(estimator.state.covariance)),
            valid=estimator.state.valid,
        ))
    return results


def write_states(path: str | Path, states: Iterable[ReplayState]) -> None:
    """Write replay output in a plotting-friendly CSV format."""
    fields = [
        "timestamp", "x", "y", "z", "vx", "vy", "vz",
        "yaw", "pitch", "roll", "covariance_trace", "valid",
    ]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for state in states:
            writer.writerow({
                "timestamp": state.timestamp,
                "x": state.position[0], "y": state.position[1], "z": state.position[2],
                "vx": state.velocity[0], "vy": state.velocity[1], "vz": state.velocity[2],
                "yaw": state.attitude_zyx_degrees[0],
                "pitch": state.attitude_zyx_degrees[1],
                "roll": state.attitude_zyx_degrees[2],
                "covariance_trace": state.covariance_trace,
                "valid": int(state.valid),
            })
