"""Face detection (YuNet and the Haar cascade baseline) and face recognition (SFace).

Detection answers "where is a face?". Recognition answers "whose face is it?":
1. YuNet returns a box, a score, and five landmarks (two eyes, nose tip, two mouth corners).
2. SFace uses the landmarks to rotate and crop the face to a standard 112 x 112 pose (alignment).
3. SFace turns the aligned face into 128 numbers (an embedding). Photos of the same person give
   embeddings that point in nearly the same direction, so cosine similarity measures identity.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import config


@dataclass
class Face:
    box: tuple[int, int, int, int]      # x1, y1, x2, y2
    score: float
    row: np.ndarray | None = None       # raw YuNet row (box, 5 landmarks, score) used for alignment
    name: str = "Unknown"
    similarity: float = 0.0


class YuNetDetector:
    """YuNet face detector. Large frames are shrunk to max_side pixels first (an optimization: detection time
    grows with the pixel count), and the boxes and landmarks are scaled back to the original frame."""

    def __init__(self, model_path=config.YUNET_MODEL, score: float = config.FACE_SCORE, nms: float = 0.3,
                 max_side: int = 640):
        self.net = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score, nms, 5000)
        self.max_side = max_side

    def detect(self, img: np.ndarray) -> list[Face]:
        h, w = img.shape[:2]
        f = min(1.0, self.max_side / max(h, w)) if self.max_side else 1.0
        small = cv2.resize(img, (int(round(w * f)), int(round(h * f))), interpolation=cv2.INTER_AREA) if f < 1 else img
        self.net.setInputSize((small.shape[1], small.shape[0]))
        _, rows = self.net.detect(small)
        faces = []
        if rows is not None:
            for r in rows:
                r = r.copy()
                r[:14] /= f                      # box (4 values) and 5 landmarks (10 values) back to full size
                x, y, bw, bh = r[:4]
                faces.append(Face((int(x), int(y), int(x + bw), int(y + bh)), float(r[-1]), r))
        return faces


class HaarDetector:
    """Classical Viola-Jones style cascade, kept as a baseline for the evaluation."""

    def __init__(self, model_path=config.HAAR_MODEL, scale: float = 1.1, neighbors: int = 5,
                 min_size: tuple[int, int] = (30, 30)):
        self.cascade = cv2.CascadeClassifier(str(model_path))
        if self.cascade.empty():
            raise RuntimeError(f"Could not load {model_path}")
        self.scale, self.neighbors, self.min_size = scale, neighbors, min_size

    def detect(self, img: np.ndarray) -> list[Face]:
        gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        boxes = self.cascade.detectMultiScale(gray, self.scale, self.neighbors, minSize=self.min_size)
        return [Face((int(x), int(y), int(x + w), int(y + h)), 1.0) for (x, y, w, h) in boxes]


class SFaceEmbedder:
    def __init__(self, model_path=config.SFACE_MODEL):
        self.net = cv2.FaceRecognizerSF.create(str(model_path), "")

    def embed(self, img: np.ndarray, face: Face) -> np.ndarray:
        aligned = self.net.alignCrop(img, face.row)
        feat = self.net.feature(aligned).flatten().astype(np.float32)
        return feat / (np.linalg.norm(feat) + 1e-12)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between unit vectors (a dot product)."""
    return np.asarray(a) @ np.asarray(b).T
