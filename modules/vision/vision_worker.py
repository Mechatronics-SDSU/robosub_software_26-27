"""Run the current camera/YOLO API as a mission-owned process."""

from __future__ import annotations

from pathlib import Path

import yaml

from shared_memory import SharedMemoryWrapper
from modules.vision.target_tracker import TargetTracker

VISION_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = VISION_DIR / "config.yaml"
MAX_DETECTIONS = 4


class VisionWorker:
    """Publish frame detections and a confirmed configured target."""

    def __init__(self, shared_memory_object: SharedMemoryWrapper,
                 config_path: Path | str = DEFAULT_CONFIG, ready_event=None):
        self.shared_memory_object = shared_memory_object
        self.config_path = Path(config_path)
        self.ready_event = ready_event

    def _load_config(self) -> dict:
        with self.config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        if not isinstance(config, dict):
            raise ValueError(f"Vision config must contain a YAML mapping: {self.config_path}")
        return config

    @staticmethod
    def _resolve_weights(config_path: Path, weights: str) -> Path:
        path = Path(weights).expanduser()
        if path.is_absolute() and path.is_file():
            return path
        candidates = [config_path.parent / path, VISION_DIR / path.name,
                      VISION_DIR / "models" / path.name]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"YOLO weights not found: {weights}. Checked: "
            + ", ".join(str(candidate) for candidate in candidates)
        )

    def _publish(self, detections: dict) -> None:
        shared = self.shared_memory_object
        fields = ("vision_class_id", "vision_confidence", "vision_x", "vision_y",
                  "vision_width", "vision_height", "vision_depth")
        parsed = []
        for values in detections.values():
            if len(values) < 8:
                continue
            _, class_id, confidence, x, y, width, height, depth = values[:8]
            parsed.append((int(class_id), float(confidence), float(x), float(y),
                           float(width), float(height), float(depth)))
            if len(parsed) == MAX_DETECTIONS:
                break

        with shared.vision_lock:
            for field in fields:
                array = getattr(shared, field)
                for index in range(MAX_DETECTIONS):
                    array[index] = 0
            for index, values in enumerate(parsed):
                for field, value in zip(fields, values):
                    getattr(shared, field)[index] = value
            shared.vision_detection_count.value = len(parsed)

    def _publish_target(self, target: dict | None) -> None:
        shared = self.shared_memory_object
        with shared.vision_lock:
            shared.vision_target_valid.value = target is not None
            if target is None:
                shared.vision_target_class_id.value = -1
                shared.vision_target_confidence.value = 0
                shared.vision_target_x.value = 0
                shared.vision_target_y.value = 0
                shared.vision_target_depth.value = -1
                shared.vision_target_centered.value = False
                return

            shared.vision_target_class_id.value = target["class_id"]
            shared.vision_target_confidence.value = target["confidence"]
            shared.vision_target_x.value = target["x"]
            shared.vision_target_y.value = target["y"]
            shared.vision_target_depth.value = target["depth"]
            shared.vision_target_centered.value = (
                abs(target["x"] - 0.5) <= self._center_tolerance_x
                and abs(target["y"] - 0.5) <= self._center_tolerance_y
            )

    def run_loop(self) -> None:
        """Load camera/model once, then publish the latest frame repeatedly."""
        from modules.vision.vision_model_main import camera, yolo

        cam = None
        try:
            config = self._load_config()
            model_config = config.get("model", {})
            yolo_config = config.get("yolo", {})
            source = str(config.get("source", "zed"))
            camera_config = config.get(source, {})
            cam_kwargs = dict(camera_config) if isinstance(camera_config, dict) else {}
            if source == "webcam":
                cam_kwargs.setdefault("index", 0)

            cam = camera(source, **cam_kwargs)
            weights = self._resolve_weights(self.config_path, model_config["weights"])
            model = yolo(
                str(weights), device=model_config.get("device"),
                conf=float(yolo_config.get("conf", 0.5)),
                iou=float(yolo_config.get("iou", 0.5)),
                imgsz=int(yolo_config.get("imgsz", 640)),
                max_det=min(int(yolo_config.get("max_det", MAX_DETECTIONS)), MAX_DETECTIONS),
                augment=bool(yolo_config.get("augment", False)),
            )
            targeting = config.get("targeting", {})
            target_tracker = None
            if targeting.get("enabled", False):
                self._center_tolerance_x = float(targeting.get("center_tolerance_x", 0.05))
                self._center_tolerance_y = float(targeting.get("center_tolerance_y", 0.05))
                target_tracker = TargetTracker(
                    target_class=str(targeting["target_class"]),
                    min_confidence=float(targeting.get("min_confidence", 0.7)),
                    stable_frames=int(targeting.get("stable_frames", 4)),
                    smoothing_alpha=float(targeting.get("smoothing_alpha", 0.35)),
                )
            else:
                self._center_tolerance_x = self._center_tolerance_y = 0.05

            if self.ready_event is not None:
                self.ready_event.set()

            classes = config.get("classes")
            while self.shared_memory_object.running.value:
                detections = model.infer(cam, headless=True, classes=classes)
                self._publish(detections)
                target = target_tracker.update(detections) if target_tracker else None
                self._publish_target(target)
        except Exception as exc:
            self._publish({})
            self._publish_target(None)
            print(f"Vision worker stopped: {exc}", flush=True)
        finally:
            if self.ready_event is not None:
                self.ready_event.clear()
            if cam is not None:
                cam.close()
            self._publish({})
            self._publish_target(None)
