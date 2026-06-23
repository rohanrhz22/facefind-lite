"""
Face engine for FaceFind Lite.

Primary backend: OpenCV YuNet (detector) + SFace (recognizer).
 - Free, open-source, ~99% LFW accuracy.
 - Installs cleanly on Windows via the prebuilt opencv-python wheel.
 - Models are small ONNX files auto-downloaded from the OpenCV Zoo on first run.

Embeddings are L2-normalized 128-d float32 vectors, so cosine similarity is
just a dot product (matching the brute-force NumPy strategy in the docs).
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

MODELS_DIR = os.environ.get(
    "FACEFIND_MODELS_DIR", os.path.join(os.path.dirname(__file__), "models"))

# OpenCV Zoo model weights (small, free).
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"

_BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODEL_URLS = {
    YUNET_FILE: f"{_BASE}/face_detection_yunet/{YUNET_FILE}",
    SFACE_FILE: f"{_BASE}/face_recognition_sface/{SFACE_FILE}",
}

EMBED_DIM = 128


@dataclass
class DetectedFace:
    bbox: list[int]          # [x, y, w, h]
    score: float
    embedding: np.ndarray    # (128,) float32, L2-normalized


def _ensure_model(filename: str) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        url = MODEL_URLS[filename]
        print(f"[face_engine] downloading {filename} ...")
        urllib.request.urlretrieve(url, path)
        print(f"[face_engine] saved {path} ({os.path.getsize(path)} bytes)")
    return path


class FaceEngine:
    """Wraps YuNet detection + SFace embedding."""

    # Detection works best when the longest image side is in this range.
    MAX_SIDE = 1280
    MIN_SIDE = 320

    def __init__(self, det_size: tuple[int, int] = (640, 640),
                 score_threshold: float = 0.6):
        yunet_path = _ensure_model(YUNET_FILE)
        sface_path = _ensure_model(SFACE_FILE)

        self.det_size = det_size
        self.detector = cv2.FaceDetectorYN.create(
            model=yunet_path,
            config="",
            input_size=det_size,
            score_threshold=score_threshold,
            nms_threshold=0.3,
            top_k=5000,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=sface_path,
            config="",
        )

    def _detect_at(self, img_bgr: np.ndarray) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img_bgr)
        return faces if faces is not None else np.empty((0, 15), dtype=np.float32)

    def _detect(self, img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """Multi-scale detection. Returns faces in *resized* coords + the scale
        used (resized = original * scale), so callers can map bboxes back."""
        h, w = img_bgr.shape[:2]
        longest = max(h, w)

        # Candidate scales: normalize big photos down, and upscale tiny ones.
        scales: list[float] = [1.0]
        if longest > self.MAX_SIDE:
            scales.insert(0, self.MAX_SIDE / longest)
        if longest < self.MIN_SIDE:
            scales.insert(0, self.MIN_SIDE / longest)
        # Fallback retries if nothing is found at the preferred scale.
        scales += [0.75, 1.5, 2.0]

        seen: set[float] = set()
        for s in scales:
            s = round(s, 4)
            if s in seen:
                continue
            seen.add(s)
            scaled = (img_bgr if s == 1.0
                      else cv2.resize(img_bgr, (max(1, int(w * s)),
                                                max(1, int(h * s)))))
            faces = self._detect_at(scaled)
            if len(faces) > 0:
                return faces, s
        return np.empty((0, 15), dtype=np.float32), 1.0

    def embed_faces(self, img_bgr: np.ndarray) -> list[DetectedFace]:
        """Detect every face, align + embed each into a normalized vector."""
        results: list[DetectedFace] = []
        faces, scale = self._detect(img_bgr)
        # Align/crop on the same scaled image the detection landmarks refer to.
        scaled = (img_bgr if scale == 1.0
                  else cv2.resize(img_bgr,
                                  (max(1, int(img_bgr.shape[1] * scale)),
                                   max(1, int(img_bgr.shape[0] * scale)))))
        for face in faces:
            aligned = self.recognizer.alignCrop(scaled, face)
            feat = self.recognizer.feature(aligned).flatten().astype("float32")
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm
            x, y, w, h = (face[:4] / scale).astype(int).tolist()
            results.append(DetectedFace(
                bbox=[x, y, w, h],
                score=float(face[-1]),
                embedding=feat,
            ))
        return results


_engine: FaceEngine | None = None


def get_engine() -> FaceEngine:
    global _engine
    if _engine is None:
        _engine = FaceEngine()
    return _engine
