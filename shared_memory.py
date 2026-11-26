from multiprocessing import Value, Array, Lock

"""
discord: @kialli
github: @kchan5071

This file initializes shared memory values used throughout the code base

"""

class SharedMemoryWrapper:
    def __init__(self):
        # imu values
        self.imu_lin_acc            = Array('d', 3)
        self.imu_ang_vel            = Array('d', 3) #p y r
        self.imu_orientation        = Array('d', 3)
        self.distance_from_object   = Value('d', 0)
        self.imu_yaw                = Value('d', 0)
        self.imu_x                  = Value('d', 0)
        self.imu_y                  = Value('d', 0)
        self.imu_z                  = Value('d', 0)

        # offsets
        self.yolo_offset            = Array('d', 2)
        self.color_offset           = Array('d', 2)
        self.gate_offset            = Array('d', 2)
        # vision enables
        self.color_enable           = Value('b', False)
        self.yolo_enable            = Value('b', False)
        self.gate_enable            = Value('b', False)
        self.current_color_index    = Value('i', 0)

        # Latest YOLO detections. Each index in these arrays is one detection;
        # vision_detection_count is the number of populated entries.
        self.vision_lock            = Lock()
        self.vision_detection_count = Value('i', 0)
        self.vision_class_id        = Array('i', 4)
        self.vision_confidence      = Array('d', 4)
        self.vision_x               = Array('d', 4)
        self.vision_y               = Array('d', 4)
        self.vision_width           = Array('d', 4)
        self.vision_height          = Array('d', 4)
        self.vision_depth           = Array('d', 4)
        self.vision_target_valid    = Value('b', False)
        self.vision_target_class_id = Value('i', -1)
        self.vision_target_confidence = Value('d', 0)
        self.vision_target_x        = Value('d', 0)
        self.vision_target_y        = Value('d', 0)
        self.vision_target_depth    = Value('d', -1)
        self.vision_target_centered = Value('b', False)
        self.depth                  = Value('d', 0) 
        self.running                = Value('i', 1)
        self.x_hard_deadzone        = Value('i', 0) # TESTME

        # dvl/orientation
        self.dvl_yaw                = Value('d', 0)
        self.dvl_pitch              = Value('d', 0)
        self.dvl_roll               = Value('d', 0)
        self.dvl_x                  = Value('d', 0)
        self.dvl_y                  = Value('d', 0)
        self.dvl_z                  = Value('d', 0)
        
        # trax/orientation
        self.trax_yaw               = Value('d', 0)
        self.trax_pitch             = Value('d', 0)
        self.trax_roll              = Value('d', 0)
        self.trax_x                 = Value('d', 0)
        self.trax_y                 = Value('d', 0)
        self.trax_z                 = Value('d', 0)

        # velocities/position
        self.dvl_x_velocity         = Value('d', 0)
        self.dvl_y_velocity         = Value('d', 0)
        self.dvl_z_velocity         = Value('d', 0)
        self.dvl_velocity_valid     = Value('b', False)
        self.dvl_status             = Value('b', False)
        self.dvl_altitude           = Value('d', 0)

        ### change these values for PID testing ###
        self.target_x               = Value('d', 0)
        self.target_y               = Value('d', 0)
        self.target_z               = Value('d', 0)
        self.target_yaw             = Value('d', 0) 
        self.target_pitch           = Value('d', 0)
        self.target_roll            = Value('d', 0)

        # gps
        self.gps_latitude           = Value('d', 0)
        self.gps_longitude          = Value('d', 0)

        # motor values
        self.motor_values           = Value('d', 0)
        
        # whether or not to run display code
        self.display_on             = Value('b', False)
    
