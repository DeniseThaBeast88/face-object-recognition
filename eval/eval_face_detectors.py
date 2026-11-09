"""Haar cascade versus YuNet face detection under controlled image changes, plus false alarms.

1. Detection rate: 300 LFW photos (one per person, fixed random seed), each changed in 12 ways
   (rotation, downscaling, brightness, blur, occlusion). A face counts as found when a detected box center lies
   within 20 percent of the image size from the image center, where LFW places the labeled person.
2. False alarms: 300 COCO val2017 photos that contain no person; every face box reported is a false positive.
3. Speed: mean milliseconds per 640 x 480 image for each detector.
Outputs results/face_detectors.json (used by make_figures.py for the heatmap).
"""
from __future__ import annotations

import contextlib
import io
import random
import time

import cv2
import numpy as np
from pycocotools.coco import COCO

from common import COCO_ANN, COCO_IMAGES, LFW_DIR, write_json
from recognizer.faces import HaarDetector, YuNetDetector


def rotate(img, deg):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def scale(img, f):
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)


def bright(img, f):
    return np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)


def occlude(img, part):
    out, (h, w) = img.copy(), img.shape[:2]
    if part == "eyes":        # dark band like sunglasses over the eye region of a centered LFW face
        cv2.rectangle(out, (int(w * 0.33), int(h * 0.38)), (int(w * 0.67), int(h * 0.47)), (20, 20, 20), -1)
    else:                     # lower face covered, like a mask
        cv2.rectangle(out, (int(w * 0.33), int(h * 0.52)), (int(w * 0.67), int(h * 0.72)), (200, 200, 200), -1)
    return out


CONDITIONS = {
    "original": lambda x: x,
    "rotate 15": lambda x: rotate(x, 15), "rotate 30": lambda x: rotate(x, 30), "rotate 45": lambda x: rotate(x, 45),
    "scale 0.5": lambda x: scale(x, 0.5), "scale 0.3": lambda x: scale(x, 0.3),
    "dark 0.4": lambda x: bright(x, 0.4), "bright 1.8": lambda x: bright(x, 1.8),
    "blur 3": lambda x: cv2.GaussianBlur(x, (0, 0), 3), "blur 6": lambda x: cv2.GaussianBlur(x, (0, 0), 6),
    "eyes covered": lambda x: occlude(x, "eyes"), "mask": lambda x: occlude(x, "mask"),
}


def found(faces, shape):
    h, w = shape[:2]
    tol = 0.2 * min(h, w)
    return any(abs((f.box[0] + f.box[2]) / 2 - w / 2) < tol and abs((f.box[1] + f.box[3]) / 2 - h / 2) < tol
               for f in faces)


def main():
    rng = random.Random(7)
    people = sorted(p for p in LFW_DIR.iterdir() if p.is_dir())
    sample = [sorted(p.glob("*.jpg"))[0] for p in rng.sample(people, 300)]
    imgs = [cv2.imread(str(p)) for p in sample]
    # a 20 x 20 pixel minimum lets the Haar cascade try the smallest (downscaled) faces too
    detectors = {"Haar cascade": HaarDetector(min_size=(20, 20)), "YuNet": YuNetDetector(max_side=0)}

    rates = {name: {} for name in detectors}
    for cond, fn in CONDITIONS.items():
        changed = [fn(im) for im in imgs]
        for name, det in detectors.items():
            rates[name][cond] = float(np.mean([found(det.detect(im), im.shape) for im in changed]))
        print(cond, {n: round(rates[n][cond], 3) for n in detectors})

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(COCO_ANN))
    person = coco.getCatIds(catNms=["person"])[0]
    with_person = set(coco.getImgIds(catIds=[person]))
    no_person = sorted(set(coco.getImgIds()) - with_person)
    picks = rng.sample(no_person, 300)
    fp, ms = {n: [] for n in detectors}, {n: [] for n in detectors}
    for iid in picks:
        img = cv2.imread(str(COCO_IMAGES / coco.loadImgs(iid)[0]["file_name"]))
        img = cv2.resize(img, (640, 480))
        for name, det in detectors.items():
            t0 = time.perf_counter()
            n = len(det.detect(img))
            ms[name].append((time.perf_counter() - t0) * 1000)
            fp[name].append(n)
    out = {"conditions": list(CONDITIONS), "lfw_images": len(imgs), "detection_rate": rates,
           "coco_no_person_images": len(picks),
           "false_positives_total": {n: int(np.sum(v)) for n, v in fp.items()},
           "images_with_false_positive": {n: int(np.sum(np.array(v) > 0)) for n, v in fp.items()},
           "ms_per_640x480": {n: float(np.mean(v[5:])) for n, v in ms.items()}}
    write_json("face_detectors.json", out)
    print(out["false_positives_total"], out["images_with_false_positive"], out["ms_per_640x480"])


if __name__ == "__main__":
    main()
