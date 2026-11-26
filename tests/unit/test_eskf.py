import numpy as np

from modules.navigation.eskf import ESKF


def test_eskf_initializes_and_publishes_attitude():
    estimator = ESKF()
    estimator.initialize_attitude(np.array([30.0, 0.0, 0.0]))

    np.testing.assert_allclose(estimator.state.euler_zyx_degrees, [30.0, 0.0, 0.0], atol=1e-6)


def test_eskf_depth_update_moves_estimate_to_measurement():
    estimator = ESKF()
    estimator.update_depth(2.0)

    assert estimator.state.position[2] > 0.0
    assert estimator.state.position[2] < 2.0


def test_eskf_dvl_velocity_update_reduces_velocity_error():
    estimator = ESKF()
    estimator.update_dvl_velocity(np.array([1.0, 0.0, 0.0]))

    assert estimator.state.velocity[0] > 0.0
    assert estimator.state.velocity[0] < 1.0


def test_eskf_rejects_unreasonable_time_step():
    estimator = ESKF()
    estimator.predict(np.zeros(3), np.zeros(3), 1.0)
    position_before = estimator.state.position.copy()
    estimator.predict(np.array([10.0, 0.0, 0.0]), np.zeros(3), 2.0)

    np.testing.assert_allclose(estimator.state.position, position_before)
