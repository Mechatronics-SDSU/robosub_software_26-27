"""Run the vision worker by itself, without starting vehicle control."""

from __future__ import annotations

import argparse
import time
from multiprocessing import Process
from pathlib import Path

from modules.vision.vision_worker import DEFAULT_CONFIG, VisionWorker
from shared_memory import SharedMemoryWrapper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"vision YAML config (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args()

    shared = SharedMemoryWrapper()
    worker = VisionWorker(shared, config_path=args.config)
    process = Process(target=worker.run_loop, name="vision-worker")
    process.start()
    print("Vision-only mode running; press Ctrl+C to stop. No mission or motor loop is started.", flush=True)

    try:
        while process.is_alive():
            with shared.vision_lock:
                count = int(shared.vision_detection_count.value)
                target_valid = bool(shared.vision_target_valid.value)
                target_class_id = int(shared.vision_target_class_id.value)
                confidence = float(shared.vision_target_confidence.value)
                x = float(shared.vision_target_x.value)
                y = float(shared.vision_target_y.value)
                depth = float(shared.vision_target_depth.value)
                centered = bool(shared.vision_target_centered.value)

            if target_valid:
                print(
                    f"detections={count}; target_class_id={target_class_id}; "
                    f"confidence={confidence:.2f}; center=({x:.3f}, {y:.3f}); "
                    f"depth_m={depth:.2f}; centered={centered}",
                    flush=True,
                )
            else:
                print(f"detections={count}; target=not confirmed", flush=True)
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping vision-only mode...", flush=True)
    finally:
        shared.running.value = 0
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join()


if __name__ == "__main__":
    main()
