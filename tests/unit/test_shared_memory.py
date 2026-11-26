"""Hardware-independent tests for the shared memory container."""

from shared_memory import SharedMemoryWrapper


def test_numeric_values_start_at_zero():
    shared = SharedMemoryWrapper()

    assert shared.distance_from_object.value == 0
    assert shared.imu_yaw.value == 0
    assert shared.dvl_x.value == 0
    assert shared.target_z.value == 0
    assert shared.motor_values.value == 0


def test_flags_have_expected_initial_values():
    shared = SharedMemoryWrapper()

    assert shared.color_enable.value == 0
    assert shared.yolo_enable.value == 0
    assert shared.gate_enable.value == 0
    assert shared.dvl_velocity_valid.value == 0
    assert shared.dvl_status.value == 0
    assert shared.running.value == 1
    assert shared.display_on.value == 0


def test_sensor_and_offset_arrays_have_expected_lengths():
    shared = SharedMemoryWrapper()

    assert len(shared.imu_lin_acc) == 3
    assert len(shared.imu_ang_vel) == 3
    assert len(shared.imu_orientation) == 3
    assert len(shared.yolo_offset) == 2
    assert len(shared.color_offset) == 2
    assert len(shared.gate_offset) == 2


def test_shared_values_and_arrays_can_be_updated():
    shared = SharedMemoryWrapper()

    shared.dvl_x.value = 1.25
    shared.yolo_enable.value = True
    shared.imu_orientation[1] = -12.5

    assert shared.dvl_x.value == 1.25
    assert shared.yolo_enable.value == 1
    assert shared.imu_orientation[1] == -12.5
