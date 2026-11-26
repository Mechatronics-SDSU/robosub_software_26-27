import numpy as np

from modules.navigation.replay import SensorSample, run_replay, summarize_replay


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
        dvl_position=np.zeros(3) if timestamp == 0.0 else None,
    )


def test_replay_metrics_report_sensor_availability_and_depth_error():
    samples = [make_sample(index * 0.02, dvl_valid=index < 5) for index in range(10)]
    states = run_replay(samples)
    metrics = summarize_replay(samples, states)

    assert metrics.sample_count == 10
    assert metrics.dvl_valid_fraction == 0.5
    assert metrics.depth_valid_fraction == 1.0
    assert metrics.valid_fraction == 1.0
    assert metrics.max_depth_residual < 1e-6
