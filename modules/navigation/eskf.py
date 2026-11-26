"""Small dependency-free error-state EKF for the submarine.

The filter uses a scalar-first quaternion and a world frame whose Z axis is
positive downward.  Sensor adapters are responsible for converting units and
vehicle-frame conventions before measurements reach this class.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


GRAVITY = 9.80665


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _quat_from_rotation_vector(rotation_vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotation_vector))
    if angle < 1e-10:
        return _quat_normalize(np.array([1.0, *(0.5 * rotation_vector)]))
    axis = rotation_vector / angle
    half = 0.5 * angle
    return np.array([math.cos(half), *(axis * math.sin(half))])


def _rotation_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _euler_zyx_degrees(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    pitch_arg = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(pitch_arg)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    return np.degrees([yaw, pitch, roll])


def _wrap_degrees(values: np.ndarray) -> np.ndarray:
    return (values + 180.0) % 360.0 - 180.0


@dataclass
class EstimatedState:
    position: np.ndarray
    velocity: np.ndarray
    quaternion: np.ndarray
    covariance: np.ndarray
    timestamp: float = 0.0
    valid: bool = False

    @property
    def euler_zyx_degrees(self) -> np.ndarray:
        return _euler_zyx_degrees(self.quaternion)


class ESKF:
    """Error-state EKF with IMU prediction and DVL/depth/AHRS updates."""

    def __init__(self, gravity: float = GRAVITY) -> None:
        self.gravity = gravity
        self.state = EstimatedState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            covariance=np.eye(15) * 1e-3,
        )
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        self.last_imu_timestamp: float | None = None

    def initialize_position(self, position: np.ndarray) -> None:
        self.state.position = np.asarray(position, dtype=float).reshape(3).copy()

    def initialize_attitude(self, yaw_pitch_roll_degrees: np.ndarray) -> None:
        yaw, pitch, roll = np.radians(yaw_pitch_roll_degrees)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        self.state.quaternion = _quat_normalize(np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]))

    def predict(self, acceleration_body: np.ndarray, angular_velocity_body: np.ndarray,
                timestamp: float) -> None:
        if self.last_imu_timestamp is None:
            self.last_imu_timestamp = timestamp
            self.state.timestamp = timestamp
            return
        dt = float(timestamp - self.last_imu_timestamp)
        self.last_imu_timestamp = timestamp
        if not 1e-5 <= dt <= 0.25:
            return

        accel = np.asarray(acceleration_body, dtype=float) - self.accel_bias
        gyro = np.asarray(angular_velocity_body, dtype=float) - self.gyro_bias
        rotation = _rotation_matrix(self.state.quaternion)
        world_accel = rotation @ accel + np.array([0.0, 0.0, self.gravity])

        self.state.position += self.state.velocity * dt + 0.5 * world_accel * dt * dt
        self.state.velocity += world_accel * dt
        dq = _quat_from_rotation_vector(gyro * dt)
        self.state.quaternion = _quat_normalize(_quat_multiply(self.state.quaternion, dq))

        f = np.eye(15)
        f[0:3, 3:6] = np.eye(3) * dt
        f[3:6, 6:9] = -rotation @ _skew(accel) * dt
        f[3:6, 12:15] = -rotation * dt
        f[6:9, 9:12] = -np.eye(3) * dt
        process_noise = np.eye(15) * 1e-6
        process_noise[3:6, 3:6] *= 10.0
        process_noise[9:15, 9:15] *= 0.1
        self.state.covariance = f @ self.state.covariance @ f.T + process_noise * dt
        self.state.timestamp = timestamp

    def _correct(self, residual: np.ndarray, h: np.ndarray, measurement_covariance: np.ndarray) -> None:
        covariance = self.state.covariance
        innovation = h @ covariance @ h.T + measurement_covariance
        gain = covariance @ h.T @ np.linalg.pinv(innovation)
        error = gain @ residual
        self.state.position += error[0:3]
        self.state.velocity += error[3:6]
        self.state.quaternion = _quat_normalize(
            _quat_multiply(self.state.quaternion, _quat_from_rotation_vector(error[6:9]))
        )
        self.gyro_bias += error[9:12]
        self.accel_bias += error[12:15]
        identity = np.eye(15)
        self.state.covariance = (identity - gain @ h) @ covariance

    def update_dvl_velocity(self, velocity_body: np.ndarray, variance: float = 0.04) -> None:
        rotation = _rotation_matrix(self.state.quaternion)
        predicted = rotation.T @ self.state.velocity
        h = np.zeros((3, 15))
        h[:, 3:6] = rotation.T
        h[:, 6:9] = -rotation.T @ _skew(self.state.velocity)
        residual = np.asarray(velocity_body, dtype=float).reshape(3) - predicted
        self._correct(residual, h, np.eye(3) * variance)

    def update_depth(self, depth: float, variance: float = 0.01) -> None:
        h = np.zeros((1, 15))
        h[0, 2] = 1.0
        self._correct(np.array([float(depth) - self.state.position[2]]), h, np.array([[variance]]))

    def update_ahrs(self, yaw_pitch_roll_degrees: np.ndarray, variance_degrees: float = 4.0) -> None:
        measured = np.asarray(yaw_pitch_roll_degrees, dtype=float).reshape(3)
        residual = np.radians(_wrap_degrees(measured - self.state.euler_zyx_degrees))
        h = np.zeros((3, 15))
        h[:, 6:9] = np.eye(3)
        self._correct(residual, h, np.eye(3) * math.radians(variance_degrees) ** 2)

