"""Generate a synthetic TRAX2/DVL/depth CSV for ESKF replay."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from modules.navigation.replay import CSV_FIELDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="Output sensor CSV")
    parser.add_argument("--samples", type=int, default=250)
    parser.add_argument("--rate", type=float, default=50.0)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for index in range(args.samples):
            timestamp = index / args.rate
            # Specific force at rest, forward DVL velocity, and a constant
            # depth. DVL is intentionally unavailable for the final 20%.
            dvl_valid = index < int(args.samples * 0.8)
            writer.writerow({
                "timestamp": timestamp,
                "accel_x": 0.0, "accel_y": 0.0, "accel_z": -9.80665,
                "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0,
                "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
                "dvl_vx": 0.5, "dvl_vy": 0.0, "dvl_vz": 0.0,
                "dvl_valid": int(dvl_valid),
                "depth": 0.75, "depth_valid": 1,
                "dvl_x": 0.0, "dvl_y": 0.0, "dvl_z": 0.75,
                "dvl_position_valid": int(index == 0),
            })
    print(f"Generated {args.samples} samples at {output}")


if __name__ == "__main__":
    main()
