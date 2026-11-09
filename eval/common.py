"""Shared helpers for the evaluation scripts (data set locations, LFW embeddings, result files)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recognizer import config  # noqa: E402
from recognizer.faces import Face, SFaceEmbedder, YuNetDetector  # noqa: E402

DATA_DIR = Path(os.environ.get("FR_DATA", ROOT.parent / "datasets"))   # override with the FR_DATA variable
LFW_DIR = DATA_DIR / "lfw_funneled"
LFW_PAIRS = DATA_DIR / "pairs.txt"
COCO_IMAGES = DATA_DIR / "val2017"
COCO_ANN = DATA_DIR / "annotations" / "instances_val2017.json"
IMAGENETV2 = DATA_DIR / "imagenetv2-matched-frequency-format-val"
RESULTS = config.RESULTS
RESULTS.mkdir(exist_ok=True)


def lfw_path(name: str, n: int) -> Path:
    return LFW_DIR / name / f"{name}_{n:04d}.jpg"


def central_face(faces: list[Face], shape) -> Face | None:
    """LFW photos are centered on the labeled person, so keep the face nearest the image center."""
    if not faces:
        return None
    cy, cx = shape[0] / 2, shape[1] / 2
    return min(faces, key=lambda f: ((f.box[0] + f.box[2]) / 2 - cx) ** 2 + ((f.box[1] + f.box[3]) / 2 - cy) ** 2)


class LFWEmbedder:
    """Detects, aligns, and embeds LFW images, with a disk cache so repeated runs are fast."""

    def __init__(self, cache: Path = DATA_DIR / "lfw_sface_cache.npz"):
        self.det, self.low_det, self.emb = YuNetDetector(), YuNetDetector(score=0.5), SFaceEmbedder()
        self.cache_path, self.cache, self.failed = cache, {}, set()
        if cache.exists():
            z = np.load(cache, allow_pickle=False)
            self.cache = {str(k): v for k, v in zip(z["keys"], z["emb"])}
            self.failed = set(str(k) for k in z["failed"])

    def get(self, path: Path) -> np.ndarray | None:
        key = f"{path.parent.name}/{path.name}"
        if key in self.cache:
            return self.cache[key]
        if key in self.failed:
            return None
        img = cv2.imread(str(path))
        face = central_face(self.det.detect(img), img.shape) if img is not None else None
        if face is None and img is not None:            # second chance with a lower face score
            face = central_face(self.low_det.detect(img), img.shape)
        if face is None:
            self.failed.add(key)
            return None
        self.cache[key] = self.emb.embed(img, face)
        return self.cache[key]

    def save(self) -> None:
        keys = list(self.cache)
        np.savez(self.cache_path, keys=np.array(keys), emb=np.vstack([self.cache[k] for k in keys]),
                 failed=np.array(sorted(self.failed), dtype=str))


def write_json(name: str, obj) -> Path:
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return p
