"""Object detection accuracy and speed on COCO val2017 (5,000 images) with the official pycocotools metric.

Runs each configuration (model and input size), writes COCO-format detections, and reports mAP@[.5:.95],
mAP@.5, per-class AP for the classes the assignment names (bus, cell phone, person, and others), and the
mean time per image on this computer.

    python eval_coco.py                         all configurations, all 5,000 images
    python eval_coco.py --limit 500             quicker check on the first 500 images
    python eval_coco.py --configs yolov10n:640
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import time

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from common import COCO_ANN, COCO_IMAGES, DATA_DIR, RESULTS, write_json
from recognizer import config
from recognizer.detector import ObjectDetector

FOCUS = ["person", "bus", "cell phone", "car", "truck", "laptop", "cup", "book", "chair", "bottle"]
MODEL_FILES = {"yolov10n": config.YOLO_MODEL, "yolov10s": DATA_DIR / "yolov10s.onnx"}


def run_config(coco, img_ids, model, imgsz):
    det = ObjectDetector(MODEL_FILES[model], imgsz=imgsz, conf=0.001)
    cat_ids = sorted(coco.getCatIds())                 # COCO ids are not contiguous: map 0..79 to them
    out, times = [], []
    for k, iid in enumerate(img_ids):
        info = coco.loadImgs(iid)[0]
        img = cv2.imread(str(COCO_IMAGES / info["file_name"]))
        t0 = time.perf_counter()
        dets = det.detect(img, max_det=300)
        times.append(time.perf_counter() - t0)
        for d in dets:
            x1, y1, x2, y2 = d.box
            out.append({"image_id": iid, "category_id": cat_ids[d.class_id],
                        "bbox": [x1, y1, x2 - x1, y2 - y1], "score": round(d.confidence, 4)})
        if k % 500 == 0:
            print(f"  {model}@{imgsz}: {k}/{len(img_ids)}")
    return out, np.array(times[5:] if len(times) > 10 else times)


def evaluate(coco, img_ids, dets):
    if not dets:
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        dt = coco.loadRes(dets)
        ev = COCOeval(coco, dt, "bbox")
        ev.params.imgIds = img_ids
        ev.evaluate(), ev.accumulate(), ev.summarize()
    per_class = {}
    prec = ev.eval["precision"]                       # [iou, recall, class, area, maxdet]
    for name in FOCUS:
        cid = coco.getCatIds(catNms=[name])[0]
        k = ev.params.catIds.index(cid)
        p = prec[:, :, k, 0, -1]
        p50 = prec[0, :, k, 0, -1]
        per_class[name] = {"ap": float(np.mean(p[p > -1])) if np.any(p > -1) else None,
                           "ap50": float(np.mean(p50[p50 > -1])) if np.any(p50 > -1) else None}
    s = ev.stats
    return {"map": float(s[0]), "map50": float(s[1]), "map75": float(s[2]), "map_small": float(s[3]),
            "map_medium": float(s[4]), "map_large": float(s[5]), "ar100": float(s[8]), "per_class": per_class}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="yolov10n:320,yolov10n:480,yolov10n:640,yolov10s:640")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(COCO_ANN))
    img_ids = sorted(coco.getImgIds())
    if a.limit:
        img_ids = img_ids[:a.limit]
    results_file = RESULTS / "coco_results.json"
    results = json.loads(results_file.read_text()) if results_file.exists() else {}
    for cfg in a.configs.split(","):
        model, size = cfg.split(":")
        dets, times = run_config(coco, img_ids, model, int(size))
        r = evaluate(coco, img_ids, dets)
        r.update(images=len(img_ids), ms_mean=float(times.mean() * 1000), ms_p95=float(np.percentile(times, 95) * 1000),
                 fps=float(1 / times.mean()))
        results[cfg] = r
        write_json("coco_results.json", results)
        print(f"{cfg}: mAP {r['map']:.3f}  mAP50 {r['map50']:.3f}  bus {r['per_class']['bus']['ap']:.3f}  "
              f"cell phone {r['per_class']['cell phone']['ap']:.3f}  {r['ms_mean']:.1f} ms/image")


if __name__ == "__main__":
    main()
