"""Sensor-only logging for TRAX2, DVL, and depth replay data."""

from __future__ import annotations

import csv
from multiprocessing import Process
from pathlib import Path
import time

from modules.navigation.replay import CSV_FIELDS


def shared_memory_row(shared_memory_object, timestamp: float) -> dict[str, object]:
    """Convert the latest shared-memory sensor values to replay CSV fields."""
    shm = shared_memory_object
    return {
        "timestamp": timestamp,
        "accel_x": shm.imu_lin_acc[0],
        "accel_y": shm.imu_lin_acc[1],
        "accel_z": shm.imu_lin_acc[2],
        "gyro_x": shm.imu_ang_vel[0],
        "gyro_y": shm.imu_ang_vel[1],
        "gyro_z": shm.imu_ang_vel[2],
        "yaw": shm.trax_yaw.value,
        "pitch": shm.trax_pitch.value,
        "roll": shm.trax_roll.value,
        "dvl_vx": shm.dvl_x_velocity.value,
        "dvl_vy": shm.dvl_y_velocity.value,
        "dvl_vz": shm.dvl_z_velocity.value,
        "dvl_valid": int(bool(shm.dvl_velocity_valid.value)),
        "depth": shm.depth.value,
        "depth_valid": int(bool(shm.depth_data_valid.value)),
        "dvl_x": shm.dvl_x.value,
        "dvl_y": shm.dvl_y.value,
        "dvl_z": shm.dvl_z.value,
        "dvl_position_valid": int(shm.dvl_pose_timestamp.value > 0),
    }


class SensorLogger:
    """Run sensor interfaces without starting any motor or mission process."""

    def __init__(self, shared_memory_object, sensors: list[object], rate_hz: float = 50.0) -> None:
        self.shared_memory_object = shared_memory_object
        self.sensors = sensors
        self.period = 1.0 / rate_hz

    def run(self, output: str | Path, duration: float | None = None) -> int:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processes = [Process(target=sensor.run_loop, name=type(sensor).__name__) for sensor in self.sensors]
        sample_count = 0
        self.shared_memory_object.running.value = 1
        try:
            for process in processes:
                process.start()
            start = time.monotonic()
            with output_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
                writer.writeheader()
                while duration is None or time.monotonic() - start < duration:
                    sample_started = time.monotonic()
                    writer.writerow(shared_memory_row(self.shared_memory_object, sample_started))
                    stream.flush()
                    sample_count += 1
                    remaining = self.period - (time.monotonic() - sample_started)
                    if remaining > 0:
                        time.sleep(remaining)
        finally:
            self.shared_memory_object.running.value = 0
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=2)
        return sample_count
