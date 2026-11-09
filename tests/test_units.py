"""Unit tests for the math and the data handling (no camera needed).   Run:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recognizer.detector import COCO_CLASSES, decode, letterbox  # noqa: E402
from recognizer.gallery import FaceGallery, control_limits, three_sigma_threshold  # noqa: E402
from recognizer.refiner import crop, softmax  # noqa: E402


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


# ---------- letterbox and box decoding --------------------------------------------------------
def test_letterbox_keeps_aspect_ratio_and_pads():
    img = np.zeros((480, 640, 3), np.uint8)
    out, scale, (px, py) = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert scale == pytest.approx(1.0)
    assert (px, py) == (0, 80)                       # 160 rows of padding split evenly top and bottom
    assert out[0, 0, 0] == 114 and out[100, 0, 0] == 0


def test_letterbox_downscale():
    out, scale, (px, py) = letterbox(np.zeros((1080, 1920, 3), np.uint8), 320)
    assert out.shape[:2] == (320, 320)
    assert scale == pytest.approx(320 / 1920)
    assert px == 0 and py == pytest.approx((320 - 180) / 2, abs=1)


def _fake_output(boxes_scores):
    """Build a (84, N) output like the network gives: cx, cy, w, h, then 80 class scores."""
    out = np.zeros((84, len(boxes_scores)), np.float32)
    for j, (cx, cy, w, h, cls, s) in enumerate(boxes_scores):
        out[:4, j] = (cx, cy, w, h)
        out[4 + cls, j] = s
    return out


def test_decode_maps_back_to_original_pixels():
    # a 100 x 50 box centered at (320, 320) in a 640 letterbox of a 640 x 480 image padded 80 rows on top
    out = _fake_output([(320, 320, 100, 50, 5, 0.9)])
    dets = decode(out, 1.0, (0, 80), (480, 640), conf=0.25, iou=0.5)
    assert len(dets) == 1 and dets[0].label == "bus"
    assert dets[0].box == (270, 215, 370, 265)


def test_decode_nms_removes_duplicates_but_keeps_other_classes():
    out = _fake_output([(300, 300, 100, 100, 67, 0.9),     # cell phone
                        (305, 302, 100, 100, 67, 0.6),     # duplicate cell phone (IoU about 0.9)
                        (300, 300, 100, 100, 0, 0.8),      # person in the same place: a different class
                        (100, 100, 20, 20, 67, 0.1)])      # below the confidence threshold
    dets = decode(out, 1.0, (0, 0), (640, 640), conf=0.25, iou=0.5)
    assert sorted(d.label for d in dets) == ["cell phone", "person"]


def test_coco_classes_include_assignment_targets():
    assert len(COCO_CLASSES) == 80
    assert COCO_CLASSES[5] == "bus" and COCO_CLASSES[67] == "cell phone" and COCO_CLASSES[0] == "person"


# ---------- three-sigma control limits ----------------------------------------------------------
def test_control_limits_formula():
    s = [0.2, 0.4, 0.6]
    mu, sd, lcl, ucl = control_limits(s)
    assert mu == pytest.approx(0.4)
    assert sd == pytest.approx(0.2)                  # sample standard deviation
    assert lcl == pytest.approx(-0.2) and ucl == pytest.approx(1.0)


def test_three_sigma_threshold_uses_impostor_upper_limit():
    rng = np.random.default_rng(0)
    gen, imp = rng.normal(0.65, 0.05, 500), rng.normal(0.1, 0.05, 500)
    r = three_sigma_threshold(gen, imp)
    assert r["threshold"] == pytest.approx(imp.mean() + 3 * imp.std(ddof=1))
    assert r["separated"] is True


# ---------- gallery ---------------------------------------------------------------------------
def _gallery():
    g = FaceGallery(threshold=0.5)
    g.add("Ana", np.vstack([unit([1, 0.05, 0]), unit([1, -0.05, 0])]))
    g.add("Ben", np.vstack([unit([0, 1, 0.05]), unit([0, 1, -0.05])]))
    return g


def test_identify_known_and_unknown():
    g = _gallery()
    assert g.identify(unit([1, 0, 0])).name == "Ana"
    assert g.identify(unit([0, 1, 0])).name == "Ben"
    m = g.identify(unit([0, 0, 1]))
    assert m.name == "Unknown" and not m.known


def test_calibrate_best_match_sets_threshold_between_people():
    g = _gallery()
    background = np.vstack([unit([0.3, 0.3, 1]), unit([-0.2, 0.3, 1]), unit([0.2, -0.3, 1])])
    cal = g.calibrate(background)
    assert cal["mode"] == "best_match"
    assert cal["impostor_ucl"] < cal["genuine_mean"]
    assert g.identify(unit([1, 0, 0])).name == "Ana"


def test_calibrate_needs_enough_samples():
    g = FaceGallery()
    g.add("Solo", unit([1, 0, 0])[None])
    assert "note" in g.calibrate(None)


def test_gallery_save_and_load_round_trip(tmp_path):
    g = _gallery()
    g.calibrate(np.vstack([unit([0, 0, 1]), unit([0.1, 0, 1])]))
    p = tmp_path / "g.npz"
    g.save(p)
    h = FaceGallery.load(p)
    assert list(h.people) == ["Ana", "Ben"]
    assert np.allclose(h.people["Ben"], g.people["Ben"])
    assert h.threshold == pytest.approx(g.threshold)
    z = np.load(p)
    assert set(z.files) == {"names", "counts", "embeddings", "threshold", "calibration"}   # no images stored


def test_remove_person():
    g = _gallery()
    g.remove("Ana")
    assert list(g.people) == ["Ben"]


# ---------- refiner helpers --------------------------------------------------------------------
def test_softmax_sums_to_one():
    p = softmax(np.array([[1.0, 2.0, 3.0]]))
    assert p.sum() == pytest.approx(1.0) and p.argmax() == 2


def test_crop_adds_margin_and_stays_inside_image():
    img = np.zeros((100, 200, 3), np.uint8)
    assert crop(img, (10, 10, 110, 60), margin=0.1).shape[:2] == (60, 120)
    assert crop(img, (0, 0, 200, 100), margin=0.1).shape[:2] == (100, 200)
