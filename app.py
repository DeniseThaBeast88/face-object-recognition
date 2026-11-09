"""Face and Object Recognition application.

Recognizes 80 kinds of everyday objects (for example person, bus, cell phone, laptop, cup, book), relabels
school buses, finds faces, and names enrolled people.

Usage (from the project folder, with the virtual environment active):
    python app.py                         webcam 0
    python app.py --source 1              a different camera
    python app.py --source samples        every image in the samples folder (press any key for the next one)
    python app.py --source photo.jpg      one image
    python app.py --source clip.mp4       a video file

Keys while the window is active:
    q or Esc  quit
    s         save a screenshot to the screenshots folder (evidence for the assignment)
    e         enroll the largest face in view (type the name in the terminal)
    h         show or hide the Haar cascade baseline (gray boxes) next to YuNet
    p         show or hide person boxes (faces stay visible)
    +  -      raise or lower the object confidence threshold
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import time
from pathlib import Path

import cv2
import numpy as np

from recognizer import config
from recognizer.gallery import FaceGallery
from recognizer.pipeline import Recognizer, draw

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WINDOW = "Face and Object Recognition"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="camera number, image, folder of images, or video file")
    ap.add_argument("--imgsz", type=int, default=640, choices=[320, 416, 480, 640], help="detector input size")
    ap.add_argument("--conf", type=float, default=config.OBJECT_CONF, help="object confidence threshold")
    ap.add_argument("--detect-every", type=int, default=1, help="run the object detector every N frames")
    ap.add_argument("--no-refiner", action="store_true", help="turn off school bus refinement")
    ap.add_argument("--width", type=int, default=1280, help="requested camera width")
    ap.add_argument("--height", type=int, default=720, help="requested camera height")
    return ap.parse_args()


class SessionLog:
    """Writes one CSV row per processed frame or image, as evidence of testing."""

    def __init__(self):
        config.RESULTS.mkdir(exist_ok=True)
        self.path = config.RESULTS / f"session_{dt.datetime.now():%Y%m%d_%H%M%S}.csv"
        self.f = open(self.path, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        self.w.writerow(["time", "source", "objects", "faces", "names", "total_ms", "objects_ms", "faces_ms"])

    def write(self, source, res):
        t = res.timings_ms
        self.w.writerow([f"{dt.datetime.now():%H:%M:%S.%f}"[:-3], source,
                         "; ".join(f"{d.label}:{d.confidence:.2f}" for d in res.objects),
                         len(res.faces), "; ".join(f"{f.name}:{f.similarity:.2f}" for f in res.faces),
                         round(t.get("total", 0), 1), round(t.get("objects", 0) + t.get("refiner", 0), 1),
                         round(t.get("face_detect", 0) + t.get("face_recognize", 0), 1)])

    def close(self):
        self.f.close()


def save_screenshot(img) -> Path:
    config.SCREENSHOTS.mkdir(exist_ok=True)
    path = config.SCREENSHOTS / f"screenshot_{dt.datetime.now():%Y%m%d_%H%M%S_%f}.png"
    cv2.imwrite(str(path), img)
    print(f"Saved {path}")
    return path


def enroll_from_frame(rec: Recognizer, frame, res) -> None:
    if not res.faces:
        print("No face in view. Look at the camera and press e again.")
        return
    face = max(res.faces, key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]))
    name = input("Name for this face (press Enter to cancel): ").strip()
    if not name:
        return
    rec.gallery.add(name, rec.embedder.embed(frame, face)[None])
    background = np.load(config.BACKGROUND_FILE) if config.BACKGROUND_FILE.exists() else None
    cal = rec.gallery.calibrate(background)
    rec.gallery.save()
    print(f"Enrolled one sample of {name}. Samples stored: {len(rec.gallery.people[name])}. "
          f"Threshold now {rec.gallery.threshold:.3f}. For better accuracy run enroll.py to capture 15 samples.")
    if "note" in cal:
        print(cal["note"])


def handle_key(key, rec, frame, shown, res, state) -> bool:
    if key in (ord("q"), 27):
        return False
    if key == ord("s"):
        save_screenshot(shown)
    elif key == ord("e"):
        enroll_from_frame(rec, frame, res)
    elif key == ord("h"):
        state["haar"] = not state["haar"]
        rec.enable_haar(state["haar"])
        print("Haar baseline", "on" if state["haar"] else "off")
    elif key == ord("p"):
        state["people"] = not state["people"]
    elif key in (ord("+"), ord("=")):
        rec.detector.conf = min(0.95, rec.detector.conf + 0.05)
        print(f"Object confidence threshold {rec.detector.conf:.2f}")
    elif key in (ord("-"), ord("_")):
        rec.detector.conf = max(0.05, rec.detector.conf - 0.05)
        print(f"Object confidence threshold {rec.detector.conf:.2f}")
    return True


def run_images(rec, paths, log, state):
    for p in paths:
        frame = cv2.imread(str(p))
        if frame is None:
            print(f"Could not read {p}")
            continue
        res = rec.process(frame)
        log.write(p.name, res)
        shown = draw(frame, res, show_people_boxes=state["people"])
        print(f"{p.name}: " + ", ".join(f"{d.label} {d.confidence:.2f}" for d in res.objects if d.label != "person")
              + f" | people {sum(d.label == 'person' for d in res.objects)} | faces {len(res.faces)}"
              + f" | {res.timings_ms['total']:.0f} ms")
        h, w = shown.shape[:2]
        scale = min(1.0, 1200 / w, 800 / h)
        cv2.imshow(WINDOW, cv2.resize(shown, (int(w * scale), int(h * scale))) if scale < 1 else shown)
        key = cv2.waitKey(0) & 0xFF
        if not handle_key(key, rec, frame, shown, res, state):
            break


def run_stream(rec, source, log, args, state):
    if source.isdigit():
        idx = int(source)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {source}. Close other apps that use the camera, or try --source 1.")
    fps, last, prev, n = 0.0, time.perf_counter(), None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = rec.process(frame, run_objects=(n % max(1, args.detect_every) == 0), previous=prev)
        prev, n = res, n + 1
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6)) if fps else 1.0 / max(now - last, 1e-6)
        last = now
        if n % 5 == 0:
            log.write(f"frame {n}", res)
        shown = draw(frame, res, fps=fps, show_people_boxes=state["people"])
        cv2.imshow(WINDOW, shown)
        if not handle_key(cv2.waitKey(1) & 0xFF, rec, frame, shown, res, state):
            break
    cap.release()


def main():
    args = parse_args()
    gallery = FaceGallery.load()
    print(f"Loaded {len(gallery.people)} enrolled people: {', '.join(gallery.people) or 'none yet'} "
          f"(threshold {gallery.threshold:.3f})")
    rec = Recognizer(imgsz=args.imgsz, conf=args.conf, use_refiner=not args.no_refiner, gallery=gallery)
    state = {"haar": False, "people": True}
    log = SessionLog()
    src = args.source
    try:
        p = Path(src)
        if p.is_dir():
            run_images(rec, sorted(x for x in p.iterdir() if x.suffix.lower() in IMAGE_EXT), log, state)
        elif p.suffix.lower() in IMAGE_EXT:
            run_images(rec, [p], log, state)
        else:
            run_stream(rec, src, log, args, state)
    finally:
        log.close()
        cv2.destroyAllWindows()
        print(f"Session log written to {log.path}")


if __name__ == "__main__":
    main()
