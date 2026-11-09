"""Measure how fast each part of the application runs on this computer.

    python benchmark_speed.py                    uses a sample photo with a face, 30 timed runs per setting
    python benchmark_speed.py --runs 50

Prints a table and writes results/speed_benchmark.json. Timings depend on the processor, so run this on the
computer used for the demonstration and compare with the numbers in the paper.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np

from recognizer import config
from recognizer.detector import ObjectDetector
from recognizer.faces import HaarDetector, SFaceEmbedder, YuNetDetector
from recognizer.refiner import SchoolBusRefiner


def timeit(fn, runs):
    for _ in range(3):
        fn()                                   # warm-up runs are not timed
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(t)), float(np.percentile(t, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--image", type=Path, default=None, help="test image (default: samples/person_with_cell_phone.jpg)")
    a = ap.parse_args()
    path = a.image or config.ROOT / "samples" / "person_with_cell_phone.jpg"
    img = cv2.resize(cv2.imread(str(path)), (1280, 720))          # a typical webcam frame size
    rows = {}
    for size in (320, 480, 640):
        det = ObjectDetector(imgsz=size)
        rows[f"YOLOv10-N objects, input {size}"] = timeit(lambda: det.detect(img), a.runs)
    ref = SchoolBusRefiner()
    patch = img[100:500, 200:800]
    rows["MobileNetV3 school bus check (per bus box)"] = timeit(lambda: ref.school_bus_probability(patch), a.runs)
    yn_full, yn_small = YuNetDetector(max_side=0), YuNetDetector(max_side=640)
    rows["YuNet faces, full 1280 x 720"] = timeit(lambda: yn_full.detect(img), a.runs)
    rows["YuNet faces, shrunk to 640 wide"] = timeit(lambda: yn_small.detect(img), a.runs)
    haar = HaarDetector()
    rows["Haar cascade faces, full 1280 x 720"] = timeit(lambda: haar.detect(img), max(5, a.runs // 3))
    faces = yn_small.detect(img)
    if faces:
        emb = SFaceEmbedder()
        rows["SFace embedding (per face)"] = timeit(lambda: emb.embed(img, faces[0]), a.runs)
    print(f"\nComputer: {platform.processor() or platform.machine()} | OpenCV {cv2.__version__} | image {path.name}")
    print(f"{'Step':48s} {'mean ms':>9s} {'p95 ms':>9s} {'max FPS':>9s}")
    for k, (m, p95) in rows.items():
        print(f"{k:48s} {m:9.1f} {p95:9.1f} {1000 / m:9.1f}")
    config.RESULTS.mkdir(exist_ok=True)
    out = {k: {"mean_ms": m, "p95_ms": p} for k, (m, p) in rows.items()}
    out["_computer"] = {"processor": platform.processor(), "machine": platform.machine(),
                        "system": platform.platform(), "opencv": cv2.__version__}
    (config.RESULTS / "speed_benchmark.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nSaved results/speed_benchmark.json")


if __name__ == "__main__":
    main()
