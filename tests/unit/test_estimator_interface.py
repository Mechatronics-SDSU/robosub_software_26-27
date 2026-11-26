import numpy as np

from modules.navigation.estimator_interface import EstimatorInterface
from shared_memory import SharedMemoryWrapper


def test_estimator_bridge_fuses_trax_dvl_and_depth_sample():
    shared = SharedMemoryWrapper()
    shared.dvl_x.value = 1.0
    shared.dvl_y.value = -0.5
    shared.dvl_z.value = 0.25
    shared.dvl_pose_timestamp.value = 10.0
    shared.dvl_x_velocity.value = 0.4
    shared.dvl_y_velocity.value = 0.0
    shared.dvl_z_velocity.value = 0.0
    shared.dvl_velocity_valid.value = True
    shared.dvl_velocity_timestamp.value = 10.0
    shared.depth.value = 0.25
    shared.depth_data_valid.value = True
    shared.depth_timestamp.value = 10.0
    shared.trax_yaw.value = 15.0
    shared.trax_pitch.value = 0.0
    shared.trax_roll.value = 0.0
    shared.imu_lin_acc[:] = np.array([0.0, 0.0, -9.80665])
    shared.imu_ang_vel[:] = np.zeros(3)
    shared.trax_data_valid.value = True
    shared.trax_timestamp.value = 10.0

    estimator = EstimatorInterface(shared)
    estimator.update_once()

    assert shared.estimated_state_valid.value
    np.testing.assert_allclose(shared.estimated_position[:2], [1.0, -0.5], atol=1e-6)
    np.testing.assert_allclose(shared.estimated_orientation[0], 15.0, atol=1e-6)
