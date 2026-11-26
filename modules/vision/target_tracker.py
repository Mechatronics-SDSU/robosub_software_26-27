"""Select and smooth one configured target from frame detections."""

from __future__ import annotations


class TargetTracker:
    """Confirm a configured class across frames and smooth its position."""

    def __init__(self, target_class: str, min_confidence: float = 0.7,
                 stable_frames: int = 4, smoothing_alpha: float = 0.35):
        if not target_class.strip():
            raise ValueError("target_class must be a non-empty class name")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if stable_frames < 1:
            raise ValueError("stable_frames must be at least 1")
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")

        self.target_class = target_class.strip().casefold()
        self.min_confidence = min_confidence
        self.stable_frames = stable_frames
        self.smoothing_alpha = smoothing_alpha
        self._consecutive_frames = 0
        self._smoothed = None

    def update(self, detections: dict) -> dict | None:
        """Return a confirmed smoothed target record, or None until confirmed."""
        matches = []
        for values in detections.values():
            if len(values) < 8:
                continue
            label, class_id, confidence, x, y, width, height, depth = values[:8]
            if str(label).casefold() != self.target_class:
                continue
            if float(confidence) < self.min_confidence:
                continue
            matches.append({
                "class_id": int(class_id),
                "confidence": float(confidence),
                "x": float(x),
                "y": float(y),
                "width": float(width),
                "height": float(height),
                "depth": float(depth),
            })

        if not matches:
            self._consecutive_frames = 0
            self._smoothed = None
            return None

        detection = max(matches, key=lambda item: item["confidence"])
        self._consecutive_frames += 1
        if self._smoothed is None:
            self._smoothed = detection
        else:
            alpha = self.smoothing_alpha
            self._smoothed = {
                key: detection[key] if key == "class_id" else
                self._smoothed[key] + alpha * (detection[key] - self._smoothed[key])
                for key in detection
            }

        if self._consecutive_frames < self.stable_frames:
            return None
        return dict(self._smoothed)
