"""Integration tests: run the real models on the sample photos.   Run:  python -m pytest -q"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recognizer.faces import SFaceEmbedder, YuNetDetector  # noqa: E402
from recognizer.gallery import FaceGallery  # noqa: E402
from recognizer.pipeline import Recognizer, draw  # noqa: E402

SAMPLES = ROOT / "samples"


@pytest.fixture(scope="module")
def rec():
    return Recognizer(gallery=FaceGallery())


def labels(rec, name):
    return {d.label for d in rec.process(cv2.imread(str(SAMPLES / name))).objects}


def test_city_bus_is_bus_not_school_bus(rec):
    found = labels(rec, "city_bus_front.jpg")
    assert "bus" in found and "school bus" not in found


def test_school_bus_is_relabeled(rec):
    assert "school bus" in labels(rec, "school_bus.jpg")


def test_cell_phone_and_person(rec):
    found = labels(rec, "person_with_cell_phone.jpg")
    assert {"cell phone", "person"} <= found


def test_face_detected_with_landmarks(rec):
    res = rec.process(cv2.imread(str(SAMPLES / "person_with_cell_phone.jpg")))
    assert len(res.faces) >= 1
    assert res.faces[0].row is not None and len(res.faces[0].row) == 15


def test_enroll_then_recognize_same_person_and_reject_other():
    det, emb = YuNetDetector(), SFaceEmbedder()
    a = cv2.imread(str(SAMPLES / "person_with_cell_phone.jpg"))
    b = cv2.imread(str(SAMPLES / "selfie_cell_phone.jpg"))
    ea = emb.embed(a, det.detect(a)[0])
    flipped = cv2.flip(a, 1)                          # a second view of the same person
    ea2 = emb.embed(flipped, det.detect(flipped)[0])
    eb = emb.embed(b, det.detect(b)[0])
    assert abs(np.linalg.norm(ea) - 1) < 1e-4
    g = FaceGallery()
    g.add("Sample Person", ea[None])
    assert g.identify(ea2).name == "Sample Person"
    assert g.identify(eb).name == "Unknown"


def test_draw_returns_same_size_image(rec):
    img = cv2.imread(str(SAMPLES / "double_decker_bus.jpg"))
    out = draw(img, rec.process(img), fps=12.3)
    assert out.shape == img.shape and not np.array_equal(out, img)


def test_frame_skipping_reuses_previous_objects(rec):
    img = cv2.imread(str(SAMPLES / "city_bus_front.jpg"))
    first = rec.process(img)
    second = rec.process(img, run_objects=False, previous=first)
    assert second.objects == first.objects
    assert second.timings_ms["objects"] < first.timings_ms["objects"]
