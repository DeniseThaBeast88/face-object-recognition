"""School bus and cell phone recognition on ImageNetV2 photos (10 new photos per class, never used in training).

Positive class: school bus (ImageNet class 779). Negative classes: 10 look-alike vehicles (trolleybus, minibus,
ambulance, fire engine, police van, minivan, moving van, passenger car, streetcar, trailer truck).
The refiner is tested on bus boxes only and on bus plus truck boxes. Also checks that the detector finds a "cell phone" in the 10 cellular telephone photos (class 487).

For every photo the full application pipeline runs (YOLOv10-N, then the MobileNetV3 refiner on bus boxes).
The script sweeps the refiner probability threshold and reports precision, recall, and F1, and compares with
two simpler designs: the detector alone (every "bus" is called a school bus) and the classifier alone on the
whole photo (top-1 class is "school bus").
Outputs results/school_bus.json.
"""
from __future__ import annotations

import cv2
import numpy as np

from common import IMAGENETV2, write_json
from recognizer.detector import ObjectDetector
from recognizer.refiner import SchoolBusRefiner

SCHOOL_BUS, PHONE = 779, 487
NEGATIVES = {874: "trolleybus", 654: "minibus", 407: "ambulance", 555: "fire engine", 734: "police van",
             656: "minivan", 675: "moving van", 705: "passenger car", 829: "streetcar", 867: "trailer truck"}


def prf(pred, truth):
    pred, truth = np.array(pred), np.array(truth)
    tp, fp, fn = np.sum(pred & truth), np.sum(pred & ~truth), np.sum(~pred & truth)
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": float(p), "recall": float(r), "f1": float(f), "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def main():
    det, ref = ObjectDetector(conf=0.25), SchoolBusRefiner(threshold=0.0, labels=("bus", "truck"))
    records = []
    for cls in [SCHOOL_BUS, *NEGATIVES]:
        for p in sorted((IMAGENETV2 / str(cls)).glob("*")):
            img = cv2.imread(str(p))
            if img is None:
                continue
            dets = det.detect(img)
            bus_probs, truck_probs = [], []
            for d in dets:
                if d.label in ("bus", "truck"):
                    was = d.label
                    ref.refine(img, [d])
                    (bus_probs if was == "bus" else truck_probs).append(d.extra["school_bus_prob"])
            whole_p, whole_top1 = ref.school_bus_probability(img)
            records.append({"cls": cls, "file": p.name, "is_school_bus": cls == SCHOOL_BUS,
                            "bus_found": bool(bus_probs), "max_bus_prob": max(bus_probs) if bus_probs else 0.0,
                            "max_bus_or_truck_prob": max(bus_probs + truck_probs) if bus_probs + truck_probs else 0.0,
                            "whole_top1": whole_top1, "labels": sorted({d.label for d in dets})})
    truth = [r["is_school_bus"] for r in records]
    sweep = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        m = prf([r["bus_found"] and r["max_bus_prob"] >= t for r in records], truth)
        m2 = prf([r["max_bus_or_truck_prob"] >= t for r in records], truth)
        m.update(threshold=float(t), bus_truck_precision=m2["precision"], bus_truck_recall=m2["recall"],
                 bus_truck_f1=m2["f1"])
        sweep.append(m)
    phone_hits = []
    for p in sorted((IMAGENETV2 / str(PHONE)).glob("*")):
        img = cv2.imread(str(p))
        phone_hits.append(any(d.label == "cell phone" for d in det.detect(img)))
    out = {
        "images": len(records), "positives": int(sum(truth)),
        "pipeline_at_0.30": prf([r["bus_found"] and r["max_bus_prob"] >= 0.30 for r in records], truth),
        "pipeline_bus_and_truck_at_0.30": prf([r["max_bus_or_truck_prob"] >= 0.30 for r in records], truth),
        "detector_only": prf([r["bus_found"] for r in records], truth),
        "classifier_only_whole_image": prf([r["whole_top1"] == "school bus" for r in records], truth),
        "school_bus_photos_with_bus_box": float(np.mean([r["bus_found"] for r in records if r["is_school_bus"]])),
        "negatives_relabeled": {NEGATIVES[c]: int(sum(r["bus_found"] and r["max_bus_prob"] >= 0.30
                                                       for r in records if r["cls"] == c)) for c in NEGATIVES},
        "negatives_with_bus_box": {NEGATIVES[c]: int(sum(r["bus_found"] for r in records if r["cls"] == c))
                                   for c in NEGATIVES},
        "cell_phone_photos_detected": float(np.mean(phone_hits)), "cell_phone_photos": len(phone_hits),
        "sweep": sweep, "records": records,
    }
    write_json("school_bus.json", out)
    for k in ["pipeline_at_0.30", "pipeline_bus_and_truck_at_0.30", "detector_only", "classifier_only_whole_image"]:
        print(k, out[k])
    print("school bus photos with a bus box", out["school_bus_photos_with_bus_box"])
    print("negatives with bus box", out["negatives_with_bus_box"])
    print("negatives relabeled", out["negatives_relabeled"])
    print("cell phone detected", out["cell_phone_photos_detected"])


if __name__ == "__main__":
    main()
