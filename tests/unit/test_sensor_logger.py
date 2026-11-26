from modules.navigation.sensor_logger import shared_memory_row
from shared_memory import SharedMemoryWrapper


def test_sensor_logger_row_uses_current_sensor_values():
    shared = SharedMemoryWrapper()
    shared.imu_lin_acc[:] = (1.0, 2.0, 3.0)
    shared.imu_ang_vel[:] = (4.0, 5.0, 6.0)
    shared.trax_yaw.value = 7.0
    shared.trax_pitch.value = 8.0
    shared.trax_roll.value = 9.0
    shared.dvl_velocity_valid.value = True
    shared.dvl_x_velocity.value = 10.0
    shared.depth.value = 11.0
    shared.depth_data_valid.value = True

    row = shared_memory_row(shared, 12.0)

    assert row["timestamp"] == 12.0
    assert row["accel_x"] == 1.0
    assert row["gyro_z"] == 6.0
    assert row["yaw"] == 7.0
    assert row["dvl_vx"] == 10.0
    assert row["depth"] == 11.0
    assert row["depth_valid"] == 1
