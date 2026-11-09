"""Enroll a person so the application can recognize them by name.

Only 128-number face embeddings are stored (data/face_gallery.npz). No photos are saved.

Examples:
    python enroll.py --name "Margaret" --camera          capture 15 samples from the webcam
    python enroll.py --name "Margaret" --images my_photos  use every face photo in a folder
    python enroll.py --list                              show enrolled people and the threshold
    python enroll.py --remove "Margaret"                 delete a person

While capturing from the camera, turn your head slightly and change expression between samples. Samples
are taken automatically about three times per second when exactly one face is in view. Press q to stop early.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np

from recognizer import config
from recognizer.faces import SFaceEmbedder, YuNetDetector
from recognizer.gallery import FaceGallery


def embeddings_from_images(folder: Path, det, emb) -> list[np.ndarray]:
    out = []
    for p in sorted(folder.iterdir()):
        img = cv2.imread(str(p))
        if img is None:
            continue
        faces = det.detect(img)
        if len(faces) != 1:
            print(f"  skipped {p.name}: found {len(faces)} faces (need exactly 1)")
            continue
        out.append(emb.embed(img, faces[0]))
        print(f"  used {p.name}")
    return out


def embeddings_from_camera(n: int, camera: int, det, emb) -> list[np.ndarray]:
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera.")
    out, last = [], 0.0
    while len(out) < n:
        ok, frame = cap.read()
        if not ok:
            break
        faces = det.detect(frame)
        shown = frame.copy()
        for f in faces:
            cv2.rectangle(shown, f.box[:2], f.box[2:], (0, 200, 0), 2)
        if len(faces) == 1 and time.time() - last > 0.33:
            out.append(emb.embed(frame, faces[0]))
            last = time.time()
        msg = f"Samples {len(out)}/{n}" + ("" if len(faces) == 1 else "  (need exactly one face)")
        cv2.putText(shown, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Enrollment", shown)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name")
    ap.add_argument("--camera", action="store_true")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--images", type=Path)
    ap.add_argument("--samples", type=int, default=15)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--remove")
    a = ap.parse_args()

    gallery = FaceGallery.load()
    background = np.load(config.BACKGROUND_FILE) if config.BACKGROUND_FILE.exists() else None
    if a.list:
        for name, emb in gallery.people.items():
            print(f"{name}: {len(emb)} samples")
        print(f"Threshold: {gallery.threshold:.3f}")
        for k, v in gallery.calibration.items():
            print(f"  {k}: {v}")
        return
    if a.remove:
        gallery.remove(a.remove)
        gallery.calibrate(background)
        gallery.save()
        print(f"Removed {a.remove}.")
        return
    if not a.name or not (a.camera or a.images):
        ap.error("give --name and either --camera or --images (or use --list / --remove)")

    det, emb = YuNetDetector(), SFaceEmbedder()
    samples = embeddings_from_camera(a.samples, a.camera_index, det, emb) if a.camera \
        else embeddings_from_images(a.images, det, emb)
    if not samples:
        raise SystemExit("No usable face samples were captured.")
    gallery.add(a.name, np.vstack(samples))
    cal = gallery.calibrate(background)
    gallery.save()
    print(f"\nEnrolled {a.name} with {len(samples)} samples ({len(gallery.people[a.name])} total).")
    if "genuine_mean" in cal:
        print(f"Genuine similarity: mean {cal['genuine_mean']:.3f}, sd {cal['genuine_sd']:.3f}, "
              f"lower limit {cal['genuine_lcl']:.3f}")
        print(f"Impostor similarity: mean {cal['impostor_mean']:.3f}, sd {cal['impostor_sd']:.3f}, "
              f"upper limit {cal['impostor_ucl']:.3f}")
    print(f"Decision threshold (three-sigma impostor upper limit): {gallery.threshold:.3f}")


if __name__ == "__main__":
    main()
