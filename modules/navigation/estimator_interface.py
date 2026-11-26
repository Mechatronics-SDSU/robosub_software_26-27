"""Runtime bridge between sensor shared memory and the ESKF."""

from __future__ import annotations

import time
import numpy as np

from modules.navigation.eskf import ESKF


class EstimatorInterface:
    """Runs the ESKF and publishes a coherent estimated-state snapshot."""

    def __init__(self, shared_memory_object, rate_hz: float = 50.0) -> None:
        self.shared_memory_object = shared_memory_object
        self.filter = ESKF()
        self.period = 1.0 / rate_hz
        self.initialized = False
        self.last_trax_timestamp = 0.0
        self.last_dvl_timestamp = 0.0
        self.last_depth_timestamp = 0.0

    def _publish(self) -> None:
        state = self.filter.state
        with self.shared_memory_object.estimated_state_lock:
            self.shared_memory_object.estimated_position[:] = state.position
            self.shared_memory_object.estimated_velocity[:] = state.velocity
            self.shared_memory_object.estimated_orientation[:] = state.euler_zyx_degrees
            self.shared_memory_object.estimated_state_timestamp.value = state.timestamp
            self.shared_memory_object.estimated_state_valid.value = bool(state.valid)
            self.shared_memory_object.estimated_covariance_trace.value = float(np.trace(state.covariance))

    def update_once(self) -> None:
        shm = self.shared_memory_object
        trax_timestamp = shm.trax_timestamp.value
        if trax_timestamp > self.last_trax_timestamp and shm.trax_data_valid.value:
            if not self.initialized:
                if shm.dvl_pose_timestamp.value > 0:
                    self.filter.initialize_position(np.array([
                        shm.dvl_x.value, shm.dvl_y.value, shm.dvl_z.value
                    ]))
                self.filter.initialize_attitude(np.array([
                    shm.trax_yaw.value, shm.trax_pitch.value, shm.trax_roll.value
                ]))
                self.initialized = True
            self.filter.predict(
                np.array(shm.imu_lin_acc[:], dtype=float),
                np.array(shm.imu_ang_vel[:], dtype=float),
                trax_timestamp,
            )
            self.filter.update_ahrs(np.array([
                shm.trax_yaw.value, shm.trax_pitch.value, shm.trax_roll.value
            ]))
            self.last_trax_timestamp = trax_timestamp

        if shm.dvl_velocity_timestamp.value > self.last_dvl_timestamp and shm.dvl_velocity_valid.value:
            self.filter.update_dvl_velocity(np.array([
                shm.dvl_x_velocity.value,
                shm.dvl_y_velocity.value,
                shm.dvl_z_velocity.value,
            ]))
            self.last_dvl_timestamp = shm.dvl_velocity_timestamp.value

        if shm.depth_timestamp.value > self.last_depth_timestamp and shm.depth_data_valid.value:
            self.filter.update_depth(shm.depth.value)
            self.last_depth_timestamp = shm.depth_timestamp.value

        self.filter.state.valid = self.initialized
        self._publish()

    def run_loop(self) -> None:
        while self.shared_memory_object.running.value:
            started = time.monotonic()
            try:
                self.update_once()
            except Exception as exc:
                self.shared_memory_object.estimated_state_valid.value = False
                print(f"ESTIMATOR ERROR: {exc}")
            remaining = self.period - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
