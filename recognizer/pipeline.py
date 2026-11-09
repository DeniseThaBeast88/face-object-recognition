"""One frame in, labeled results out: objects, school bus refinement, faces, and identities."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config
from .detector import Detection, ObjectDetector
from .faces import Face, HaarDetector, SFaceEmbedder, YuNetDetector
from .gallery import FaceGallery
from .refiner import SchoolBusRefiner

PALETTE = {"person": (255, 160, 0), "bus": (0, 140, 255), "school bus": (0, 215, 255),
           "cell phone": (255, 0, 200), "face": (0, 200, 0), "unknown": (0, 0, 255), "haar": (200, 200, 200)}


def color_for(label: str) -> tuple[int, int, int]:
    if label in PALETTE:
        return PALETTE[label]
    h = (sum(ord(ch) * 31 ** i for i, ch in enumerate(label)) * 7) % 180   # same color every run
    hsv = np.uint8([[[h, 200, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


@dataclass
class FrameResult:
    objects: list[Detection] = field(default_factory=list)
    faces: list[Face] = field(default_factory=list)
    haar_faces: list[Face] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)


class Recognizer:
    def __init__(self, imgsz: int = 640, conf: float = config.OBJECT_CONF, use_refiner: bool = True,
                 use_faces: bool = True, gallery: FaceGallery | None = None, model_path=config.YOLO_MODEL):
        self.detector = ObjectDetector(model_path, imgsz=imgsz, conf=conf)
        self.refiner = SchoolBusRefiner() if use_refiner else None
        self.face_detector = YuNetDetector() if use_faces else None
        self.embedder = SFaceEmbedder() if use_faces else None
        self.haar = None
        self.gallery = gallery if gallery is not None else FaceGallery.load()

    def enable_haar(self, on: bool) -> None:
        self.haar = HaarDetector() if on else None

    def process(self, frame: np.ndarray, run_objects: bool = True, previous: FrameResult | None = None) -> FrameResult:
        res, t = FrameResult(), {}
        t0 = time.perf_counter()
        if run_objects:
            res.objects = self.detector.detect(frame)
        elif previous is not None:
            res.objects = previous.objects               # frame skipping: reuse last object boxes
        t["objects"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        if self.refiner and run_objects:
            self.refiner.refine(frame, res.objects)
        t["refiner"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        if self.face_detector:
            res.faces = self.face_detector.detect(frame)
        t["face_detect"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        if self.embedder and self.gallery.people:
            for f in res.faces:
                m = self.gallery.identify(self.embedder.embed(frame, f))
                f.name, f.similarity = m.name, m.similarity
        t["face_recognize"] = (time.perf_counter() - t0) * 1000
        if self.haar:
            t0 = time.perf_counter()
            res.haar_faces = self.haar.detect(frame)
            t["haar"] = (time.perf_counter() - t0) * 1000
        t["total"] = sum(t.values())
        res.timings_ms = t
        return res


def draw(frame: np.ndarray, res: FrameResult, fps: float | None = None, show_people_boxes: bool = True) -> np.ndarray:
    out = frame.copy()
    for d in res.objects:
        if d.label == "person" and not show_people_boxes:
            continue
        c = color_for(d.label)
        x1, y1, x2, y2 = d.box
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        _label(out, f"{d.label} {d.confidence:.2f}", (x1, y1), c)
    for f in res.haar_faces:
        x1, y1, x2, y2 = f.box
        cv2.rectangle(out, (x1, y1), (x2, y2), PALETTE["haar"], 1)
        _label(out, "Haar face", (x1, y2 + 18), PALETTE["haar"])
    for f in res.faces:
        known = f.name != "Unknown"
        c = PALETTE["face"] if known else PALETTE["unknown"]
        x1, y1, x2, y2 = f.box
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        if f.row is not None:
            for k in range(5):
                cv2.circle(out, (int(f.row[4 + 2 * k]), int(f.row[5 + 2 * k])), 2, c, -1)
        text = f"{f.name} ({f.similarity:.2f})" if f.similarity else f"face {f.score:.2f}"
        _label(out, text, (x1, y1), c)
    if fps is not None:
        n_obj = len([d for d in res.objects if d.label != "person"])
        _label(out, f"FPS {fps:.1f} | objects {n_obj} | faces {len(res.faces)}", (8, 26), (40, 40, 40), scale=0.6)
    return out


def _label(img, text, org, color, scale=0.55):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    y = max(th + 4, y)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), color, -1)
    lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0) if lum > 140 else (255, 255, 255), 1, cv2.LINE_AA)
