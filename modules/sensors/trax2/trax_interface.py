import sys, os, time, math
import numpy                                as np
from multiprocessing                        import Process, Value
from modules.sensors.trax2.trax_fxns        import TRAX

G_TO_MS2 = 9.80665 # gravity conversion

class Trax_Interface(TRAX):

    """
    discord: @.kech
    github: @rsunderr
    """

    def __init__(self, shared_memory_object) -> None:
        """
        Trax interface constructor
        """
        super().__init__()
        self.shared_memory_object = shared_memory_object
        self.interval: float = 0
        self.acq_params: tuple = (False, False, 0, self.interval) # poll mode false, flush filter false, PNI reserved, interval
        # Acceleration is in g, angular velocity is in degrees/sec, and
        # attitude is in degrees according to the TRAX component protocol.
        self.data_components: tuple = (
            9, 0x15, 0x16, 0x17, 0x4A, 0x4B, 0x4C, 0x05, 0x18, 0x19
        )

        # positional values
    
    def setup(self) -> None:
        """
        Setup function to initialize Trax interface
        Call this function once ever after continuous mode is stopped, it is remembered by non-volatile memory (no need to run every time)
        """
        self.connect() # connect to trax
        self.send_packet("kStopContinuousMode") # kStopContinuousMode (ensure not running)
        self.send_packet("kSetAcqParams", self.acq_params) # kSetAcqParams - set acquisition parameters
        self.send_packet("kSetDataComponents", self.data_components) # kSetDataComponents - set data components
        self.send_packet("kSave") # kSave - save settings
        resp = self.recv_packet() # save response
        print(resp)
        
    def run_loop(self) -> None:
        """
        Start the Trax interface process
        """
        #self.setup() # only run once ever
        self.connect()
        # Apply the runtime stream configuration on every launch.  This keeps
        # the estimator independent of whatever component list was last saved
        # in the TRAX's non-volatile settings.
        self.send_packet("kStopContinuousMode")
        self.send_packet("kSetAcqParams", self.acq_params)
        self.send_packet("kSetDataComponents", self.data_components)
        self.send_packet("kStartContinuousMode") # kStartContinuousMode - start continuous mode
        while self.shared_memory_object.running.value:
            self.update()

    def adjust_accel(self, accel_x: float, accel_y: float, acel_z: float, yw: float, ptch: float, rll: float) -> tuple:
        """
        TODO make this adjust acceleration based on orientation (ie gravity pulling on it at an angle) and convert to m/s^2
        """
        # ADJUST VALUES ---------------------------------------------------------------------------------------------------------------
        # convert from g to m/s^2
        ay = accel_y * G_TO_MS2
        ax = accel_x * G_TO_MS2
        az = acel_z * G_TO_MS2
        # convert from degrees to radians
        yaw = math.radians(yw)
        pitch = math.radians(ptch)
        roll = math.radians(rll)
        
        
        
        
        return (ax, ay, az)

    def update(self) -> None:
        """
        Function targeted by looping multiprocessing calls, called only once
        """
        timestamp = time.monotonic()
        try:
            # READ DATA
            data:       tuple = self.recv_packet(self.data_components)
            components = {data[index]: data[index + 1] for index in range(3, len(data) - 2, 2)}
            accel = np.array([components[0x15], components[0x16], components[0x17]]) * G_TO_MS2
            gyro = np.radians(np.array([components[0x4A], components[0x4B], components[0x4C]]))
            yaw = components[0x05]
            pitch = components[0x18]
            roll = components[0x19]

            self.shared_memory_object.trax_yaw.value = yaw
            self.shared_memory_object.trax_pitch.value = pitch
            self.shared_memory_object.trax_roll.value = roll
            self.shared_memory_object.imu_lin_acc[:] = accel
            self.shared_memory_object.imu_ang_vel[:] = gyro
            self.shared_memory_object.imu_orientation[:] = (yaw, pitch, roll)
            self.shared_memory_object.trax_timestamp.value = timestamp
            self.shared_memory_object.trax_data_valid.value = True
        except KeyboardInterrupt:
            self.send_packet("kStopContinuousMode")
            self.close()
        except Exception as e:
            self.shared_memory_object.trax_data_valid.value = False
            print(f"INVALID TRAX DATA: {e}") # errors are expected
