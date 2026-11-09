"""Fine-grained refinement: decide whether a detected bus is a school bus.

COCO, the data set the detector was trained on, has a single "bus" class. ImageNet, the data set the
MobileNetV3 classifier was trained on, has a separate "school bus" class. The refiner crops every bus
box, classifies the crop, and relabels the box when the school bus probability is high enough.
"""
from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

from . import config

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def crop(img: np.ndarray, box, margin: float = 0.05) -> np.ndarray:
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    h, w = img.shape[:2]
    x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    x2, y2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
    return img[y1:y2, x1:x2]


class SchoolBusRefiner:
    def __init__(self, model_path=config.CLASSIFIER_MODEL, labels_path=config.IMAGENET_LABELS,
                 threshold: float = config.SCHOOL_BUS_PROB, labels: tuple = config.REFINE_LABELS):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.labels = labels_path.read_text(encoding="utf-8").splitlines()
        self.threshold = threshold
        self.candidate_labels = tuple(labels)

    def preprocess(self, patch: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(cv2.resize(patch, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        return x.transpose(2, 0, 1)[None]

    def school_bus_probability(self, patch: np.ndarray) -> tuple[float, str]:
        probs = softmax(self.session.run(None, {self.input_name: self.preprocess(patch)})[0][0])
        return float(probs[config.IMAGENET_SCHOOL_BUS]), self.labels[int(probs.argmax())]

    def refine(self, img: np.ndarray, detections) -> None:
        """Relabel bus (and truck) detections in place when the classifier says the vehicle is a school bus."""
        for d in detections:
            if d.label not in self.candidate_labels:
                continue
            patch = crop(img, d.box)
            if patch.size == 0:
                continue
            p, top1 = self.school_bus_probability(patch)
            d.extra.update(school_bus_prob=round(p, 3), imagenet_top1=top1)
            if p >= self.threshold:
                d.label = "school bus"
