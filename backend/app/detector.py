"""YuNet-backed face detection and annotation utilities."""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np

MODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


class FaceDetector:
    """Lazy, thread-safe wrapper around OpenCV FaceDetectorYN."""

    def __init__(self) -> None:
        default_path = Path(__file__).resolve().parents[1] / "models" / "face_detection_yunet_2023mar.onnx"
        self.model_path = Path(os.getenv("FACE_MODEL_PATH", str(default_path)))
        self.score_threshold = float(os.getenv("FACE_SCORE_THRESHOLD", "0.75"))
        self.nms_threshold = float(os.getenv("FACE_NMS_THRESHOLD", "0.3"))
        self._detector: Any | None = None
        self._lock = Lock()

    def _ensure_model(self) -> None:
        if self.model_path.exists() and self.model_path.stat().st_size > 0:
            return
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.model_path.with_suffix(self.model_path.suffix + ".part")
        try:
            urllib.request.urlretrieve(MODEL_URL, temporary_path)
            if temporary_path.stat().st_size < 10_000:
                raise RuntimeError("Downloaded YuNet model appears to be incomplete.")
            temporary_path.replace(self.model_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def load(self) -> None:
        """Download (if needed) and initialize the model."""
        with self._lock:
            if self._detector is not None:
                return
            self._ensure_model()
            self._detector = cv2.FaceDetectorYN.create(
                str(self.model_path),
                "",
                (320, 320),
                self.score_threshold,
                self.nms_threshold,
                5000,
            )

    @property
    def ready(self) -> bool:
        return self._detector is not None

    def detect(self, image: np.ndarray, blur_faces: bool = False) -> dict[str, Any]:
        if image is None or image.size == 0:
            raise ValueError("The supplied image is empty.")

        self.load()
        height, width = image.shape[:2]
        with self._lock:
            self._detector.setInputSize((width, height))
            _, raw_faces = self._detector.detect(image)

        annotated = image.copy()
        detections: list[dict[str, Any]] = []

        if raw_faces is not None:
            for index, face in enumerate(raw_faces, start=1):
                x, y, box_width, box_height = [int(round(value)) for value in face[:4]]
                x = max(0, x)
                y = max(0, y)
                x2 = min(width, x + max(0, box_width))
                y2 = min(height, y + max(0, box_height))
                confidence = float(face[-1])

                if x2 <= x or y2 <= y:
                    continue

                if blur_faces:
                    roi = annotated[y:y2, x:x2]
                    if roi.size:
                        kernel = max(15, (min(roi.shape[:2]) // 7) | 1)
                        annotated[y:y2, x:x2] = cv2.GaussianBlur(roi, (kernel, kernel), 0)

                cv2.rectangle(annotated, (x, y), (x2, y2), (40, 220, 120), 2)
                cv2.putText(
                    annotated,
                    f"Face {index}  {confidence:.0%}",
                    (x, max(22, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (40, 220, 120),
                    2,
                    cv2.LINE_AA,
                )
                detections.append({
                    "box": {"x": x, "y": y, "width": x2 - x, "height": y2 - y},
                    "confidence": round(confidence, 4),
                })

        return {"faces": detections, "annotated_image": annotated}
