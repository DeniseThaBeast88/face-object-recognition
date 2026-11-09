"""Object detection with YOLOv10-N (COCO, 80 classes) through ONNX Runtime.

Steps for every frame:
1. Letterbox: resize the frame so its longer side equals imgsz, keep the aspect ratio,
   and pad the rest with gray (114) so the network sees a square image.
2. Run the network. The output has 84 rows for each of the candidate boxes:
   4 box values (center x, center y, width, height) and 80 class scores.
3. Keep boxes whose best class score is above the confidence threshold.
4. Undo the letterbox so the boxes line up with the original frame.
5. Class-aware non-maximum suppression removes duplicate boxes of the same class.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import onnxruntime as ort

from . import config

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


@dataclass
class Detection:
    class_id: int
    label: str
    confidence: float
    box: tuple[int, int, int, int]          # x1, y1, x2, y2 in original image pixels
    extra: dict = field(default_factory=dict)


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize with unchanged aspect ratio and pad to size x size. Returns image, scale, (pad_x, pad_y)."""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - nw) / 2, (size - nh) / 2
    top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
    left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, scale, (left, top)


def decode(output: np.ndarray, scale: float, pad: tuple[float, float], img_shape: tuple[int, int],
           conf: float, iou: float, max_det: int = 300) -> list[Detection]:
    """Turn the raw (84, N) network output into a list of detections in original image coordinates."""
    pred = output.T                          # (N, 84)
    scores = pred[:, 4:]
    class_ids = scores.argmax(axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]
    keep = confidences >= conf
    if not np.any(keep):
        return []
    pred, class_ids, confidences = pred[keep], class_ids[keep], confidences[keep]
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    x1 = (cx - bw / 2 - pad[0]) / scale
    y1 = (cy - bh / 2 - pad[1]) / scale
    x2 = (cx + bw / 2 - pad[0]) / scale
    y2 = (cy + bh / 2 - pad[1]) / scale
    h, w = img_shape
    x1, x2 = np.clip(x1, 0, w - 1), np.clip(x2, 0, w - 1)
    y1, y2 = np.clip(y1, 0, h - 1), np.clip(y2, 0, h - 1)
    boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
    idx = cv2.dnn.NMSBoxesBatched(boxes_xywh.tolist(), confidences.tolist(), class_ids.tolist(), conf, iou)
    idx = np.array(idx).flatten()[:max_det] if len(idx) else []
    dets = []
    for i in idx:
        dets.append(Detection(int(class_ids[i]), COCO_CLASSES[int(class_ids[i])], float(confidences[i]),
                              (int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i]))))
    dets.sort(key=lambda d: d.confidence, reverse=True)
    return dets


class ObjectDetector:
    def __init__(self, model_path=config.YOLO_MODEL, imgsz: int = 640, conf: float = config.OBJECT_CONF,
                 iou: float = config.OBJECT_IOU, threads: int = 0):
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz, self.conf, self.iou = imgsz, conf, iou

    def detect(self, img: np.ndarray, conf: float | None = None, max_det: int = 300) -> list[Detection]:
        boxed, scale, pad = letterbox(img, self.imgsz)
        blob = cv2.dnn.blobFromImage(boxed, 1 / 255.0, swapRB=True)   # BGR to RGB, 0 to 1, NCHW
        output = self.session.run(None, {self.input_name: blob})[0][0]
        return decode(output, scale, pad, img.shape[:2], self.conf if conf is None else conf, self.iou, max_det)
