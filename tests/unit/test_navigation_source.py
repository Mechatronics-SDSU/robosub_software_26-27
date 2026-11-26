from fsm.fsm import FSM_Template
from shared_memory import SharedMemoryWrapper


def test_fsm_uses_raw_dvl_until_estimator_is_valid():
    shared = SharedMemoryWrapper()
    shared.dvl_x.value = 1.0
    shared.dvl_y.value = 2.0
    shared.dvl_z.value = 3.0
    fsm = FSM_Template(shared, [])

    assert fsm.navigation_position() == (1.0, 2.0, 3.0)


def test_fsm_uses_estimated_position_when_enabled_and_valid():
    shared = SharedMemoryWrapper()
    shared.dvl_x.value = 1.0
    shared.dvl_y.value = 2.0
    shared.dvl_z.value = 3.0
    shared.use_estimated_state.value = True
    shared.estimated_state_valid.value = True
    shared.estimated_position[:] = (4.0, 5.0, 6.0)
    fsm = FSM_Template(shared, [])

    assert fsm.navigation_position() == (4.0, 5.0, 6.0)
