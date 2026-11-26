"""Record TRAX2, DVL, and depth data without starting the submarine motors."""

from __future__ import annotations

import argparse

from modules.navigation.sensor_logger import SensorLogger
from modules.sensors.a50_dvl.dvl_interface import DVL_Interface
from modules.sensors.depth_sensor.depth_sensor_interface import DepthSensorInterface
from modules.sensors.trax2.trax_interface import Trax_Interface
from shared_memory import SharedMemoryWrapper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="Output sensor CSV")
    parser.add_argument("--duration", type=float, default=30.0, help="Seconds to record")
    parser.add_argument("--rate", type=float, default=50.0, help="CSV sample rate")
    args = parser.parse_args()

    shared = SharedMemoryWrapper()
    sensors = [
        DVL_Interface(shared),
        DepthSensorInterface(shared),
        Trax_Interface(shared),
    ]
    logger = SensorLogger(shared, sensors, rate_hz=args.rate)
    count = logger.run(args.output, duration=args.duration)
    print(f"Recorded {count} samples to {args.output}")


if __name__ == "__main__":
    main()
