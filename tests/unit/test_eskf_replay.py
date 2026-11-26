import numpy as np

from modules.navigation.replay import SensorSample, run_replay


def make_sample(timestamp: float, dvl_valid: bool = True) -> SensorSample:
    return SensorSample(
        timestamp=timestamp,
        acceleration_body=np.array([0.0, 0.0, -9.80665]),
        angular_velocity_body=np.zeros(3),
        attitude_zyx_degrees=np.zeros(3),
        dvl_velocity_body=np.array([0.5, 0.0, 0.0]),
        dvl_velocity_valid=dvl_valid,
        depth=0.75,
        depth_valid=True,
        dvl_position=np.array([0.0, 0.0, 0.75]) if timestamp == 0.0 else None,
    )


def test_replay_produces_one_estimate_per_sample():
    states = run_replay([make_sample(index * 0.02) for index in range(20)])

    assert len(states) == 20
    assert all(state.valid for state in states)
    assert all(np.isfinite(state.position).all() for state in states)


def test_replay_survives_dvl_dropout():
    samples = [make_sample(index * 0.02, dvl_valid=index < 10) for index in range(30)]
    states = run_replay(samples)

    assert len(states) == 30
    assert all(np.isfinite(state.velocity).all() for state in states)
    assert states[-1].velocity[0] > 0.0
