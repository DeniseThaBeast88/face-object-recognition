"""Draw every graph and heatmap used in the paper from the JSON results (results/figures/*.png, 300 dpi)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)
W = 3.4                                                    # IEEE column width in inches
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "savefig.dpi": 300, "axes.linewidth": 0.6})


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", name)


def pipeline():
    fig, ax = plt.subplots(figsize=(W, 3.3))
    ax.set_xlim(0, 100), ax.set_ylim(0, 100), ax.axis("off")
    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6", fc=color, ec="#333333", lw=0.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.3)
    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=6, lw=0.6, color="#333333"))
    box(22, 89, 56, 8, "Camera frame or image (BGR)", "#eeeeee")
    left = [("Letterbox to 320, 480,\nor 640 pixels", "#dbe9f6"), ("YOLOv10-N (ONNX Runtime)\n80 COCO classes", "#dbe9f6"),
            ("Confidence filter and\nclass-wise NMS", "#dbe9f6"), ("MobileNetV3 school bus\ncheck on bus, truck boxes", "#fde8c8")]
    right = [("Shrink to 640 pixels\non the longest side", "#dcefdc"), ("YuNet face box and\n5 landmarks", "#dcefdc"),
             ("Align to 112 x 112;\nSFace 128-D embedding", "#dcefdc"), ("Cosine match to gallery;\nthree-sigma threshold", "#f6dada")]
    ys = [70, 53, 36, 19]
    for (t, c), y in zip(left, ys):
        box(2, y, 44, 10, t, c)
    for (t, c), y in zip(right, ys):
        box(54, y, 44, 10, t, c)
    for y1, y2 in zip(ys[:-1], ys[1:]):
        arrow(24, y1 - 0.8, 24, y2 + 10.8), arrow(76, y1 - 0.8, 76, y2 + 10.8)
    arrow(30, 88.2, 12, 80.8), arrow(70, 88.2, 88, 80.8)
    box(25, 2, 50, 8, "Labeled boxes, names, FPS, CSV log", "#eeeeee")
    arrow(24, 18.2, 40, 10.8), arrow(76, 18.2, 60, 10.8)
    ax.text(24, 83.5, "Object branch (every N frames)", ha="center", fontsize=6.3, style="italic")
    ax.text(76, 83.5, "Face branch (every frame)", ha="center", fontsize=6.3, style="italic")
    save(fig, "fig1_pipeline.png")


def coco():
    r = load("coco_results.json")
    if not r:
        return
    fig, ax = plt.subplots(figsize=(W, 2.1))
    for k, v in r.items():
        m = "o" if "10n" in k else "s"
        ax.plot(v["ms_mean"], v["map"] * 100, m, color="#1f4e79" if "10n" in k else "#c55a11", ms=5)
        ax.annotate(k.replace("yolov10", "YOLOv10-").replace(":", " @ ").upper().replace("-N", "-N").replace("-S", "-S"),
                    (v["ms_mean"], v["map"] * 100), textcoords="offset points", xytext=(5, -3), fontsize=6)
    n = sorted([(int(k.split(":")[1]), v) for k, v in r.items() if "10n" in k])
    ax.plot([v["ms_mean"] for _, v in n], [v["map"] * 100 for _, v in n], "-", color="#1f4e79", lw=0.8)
    ax.set_xlabel("CPU time per image (ms, 2-core virtual machine)")
    ax.set_ylabel("COCO val2017 mAP@[.5:.95] (%)")
    ax.grid(alpha=0.3, lw=0.4)
    ax.set_xlim(0, max(v["ms_mean"] for v in r.values()) * 1.35)
    save(fig, "fig2_coco_speed_accuracy.png")

    classes = ["person", "bus", "cell phone", "car", "truck", "laptop", "cup", "book"]
    keys = [k for k in ["yolov10n:320", "yolov10n:640", "yolov10s:640"] if k in r]
    fig, ax = plt.subplots(figsize=(W, 1.9))
    x = np.arange(len(classes))
    wbar = 0.8 / len(keys)
    colors = ["#9dc3e6", "#1f4e79", "#c55a11"]
    for i, k in enumerate(keys):
        ax.bar(x + (i - (len(keys) - 1) / 2) * wbar, [r[k]["per_class"][c]["ap"] * 100 for c in classes], wbar,
               label=k.replace("yolov10", "YOLOv10-").replace(":", " @ ").upper(), color=colors[i])
    ax.set_xticks(x, classes, rotation=25, ha="right")
    ax.set_ylabel("AP@[.5:.95] (%)")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="y", alpha=0.3, lw=0.4)
    save(fig, "fig3_per_class_ap.png")


def lfw():
    r, z = load("lfw_verification.json"), RES / "lfw_scores.npz"
    if not r or not z.exists():
        return
    d = np.load(z)
    s, l = d["scores"], d["labels"]
    fig, ax = plt.subplots(figsize=(W, 1.95))
    bins = np.linspace(-0.3, 1.0, 66)
    ax.hist(s[l == 0], bins, alpha=0.65, color="#c55a11", label="Different people (impostor)")
    ax.hist(s[l == 1], bins, alpha=0.65, color="#1f4e79", label="Same person (genuine)")
    for val, txt, st in [(r["impostor"]["ucl"], "impostor UCL", "--"), (r["genuine"]["lcl"], "genuine LCL", ":"),
                         (0.363, "SFace default", "-.")]:
        ax.axvline(val, color="k", ls=st, lw=0.8, label=f"{txt} = {val:.3f}")
    ax.set_xlabel("Cosine similarity of SFace embeddings")
    ax.set_ylabel("Number of LFW pairs")
    ax.legend(frameon=False, fontsize=5.8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    save(fig, "fig4_lfw_score_distributions.png")

    fig, ax = plt.subplots(figsize=(W, 1.8))
    ax.semilogx(np.maximum(d["fpr"], 1e-4), d["tpr"] * 100, color="#1f4e79", lw=1)
    ax.set_xlabel("False accept rate (log scale)")
    ax.set_ylabel("True accept rate (%)")
    ax.set_xlim(3e-4, 1), ax.set_ylim(90, 100.3)
    ax.grid(alpha=0.3, lw=0.4, which="both")
    ax.text(1e-3, 91, f"AUC = {r['auc']:.4f}", fontsize=7)
    save(fig, "fig5_lfw_roc.png")


def detectors():
    r = load("face_detectors.json")
    if not r:
        return
    conds = r["conditions"]
    names = list(r["detection_rate"])
    m = np.array([[r["detection_rate"][n][c] * 100 for n in names] for c in conds])
    fig, ax = plt.subplots(figsize=(W * 0.8, 2.9))
    im = ax.imshow(m, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_yticks(range(len(conds)), conds)
    ax.set_xticks(range(len(names)), names)
    ax.xaxis.tick_top()
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if m[i, j] < 35 else "black")
    cb = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.06)
    cb.set_label("Faces found (%)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    save(fig, "fig6_face_detector_heatmap.png")


def openset():
    r, z = load("openset.json"), RES / "openset_confusion.npz"
    if not r:
        return
    sw = r["sweep"]
    t = [s["threshold"] for s in sw]
    fig, ax = plt.subplots(figsize=(W, 1.9))
    ax.plot(t, [s["identification_rate"] * 100 for s in sw], "-o", ms=2.5, color="#1f4e79", lw=0.9,
            label="Enrolled people named correctly")
    ax.plot(t, [s["false_accept_rate"] * 100 for s in sw], "-s", ms=2.5, color="#c55a11", lw=0.9,
            label="Strangers wrongly accepted")
    for val, txt, st in [(0.363, "default", "-."), (r["calibration_pairwise"]["threshold"], "pairwise 3-sigma", ":"),
                         (r["calibration"]["threshold"], "best-match 3-sigma", "--")]:
        ax.axvline(val, color="k", ls=st, lw=0.8, label=f"{txt} ({val:.3f})")
    ax.set_xlabel("Decision threshold (cosine similarity)")
    ax.set_ylabel("Rate (%)")
    ax.legend(frameon=False, fontsize=5.8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    ax.grid(alpha=0.3, lw=0.4)
    save(fig, "fig7_openset_threshold_sweep.png")
    if z.exists():
        d = np.load(z)
        c, names = d["confusion"], [str(n) for n in d["names"]]
        fig, ax = plt.subplots(figsize=(W, W * 0.92))
        im = ax.imshow(np.log1p(c), cmap="Blues")
        short = [n if len(n) < 16 else n.split(" ")[0][0] + ". " + n.split(" ")[-1] for n in names]
        ax.set_xticks(range(len(names)), short, rotation=90, fontsize=5)
        ax.set_yticks(range(len(names)), short, fontsize=5)
        for i in range(len(names)):
            for j in range(len(names)):
                if c[i, j]:
                    ax.text(j, i, int(c[i, j]), ha="center", va="center", fontsize=4.5,
                            color="white" if c[i, j] > 20 else "black")
        ax.set_xlabel("Predicted label"), ax.set_ylabel("True label")
        save(fig, "fig8_openset_confusion.png")


def school_bus():
    r = load("school_bus.json")
    if not r:
        return
    sw = r["sweep"]
    t = [s["threshold"] for s in sw]
    fig, ax = plt.subplots(figsize=(W, 1.8))
    ax.plot(t, [s["bus_truck_precision"] * 100 for s in sw], "-o", ms=2.5, color="#1f4e79", lw=0.9, label="Precision")
    ax.plot(t, [s["bus_truck_recall"] * 100 for s in sw], "-s", ms=2.5, color="#c55a11", lw=0.9, label="Recall")
    ax.plot(t, [s["bus_truck_f1"] * 100 for s in sw], "-^", ms=2.5, color="#548235", lw=0.9, label="F1")
    ax.axvline(0.30, color="k", ls="--", lw=0.8, label="chosen threshold 0.30")
    ax.set_xlabel("School bus probability threshold (MobileNetV3)")
    ax.set_ylabel("Score (%)")
    ax.legend(frameon=False, fontsize=5.8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    ax.grid(alpha=0.3, lw=0.4)
    save(fig, "fig9_school_bus_threshold.png")


def examples():
    import cv2
    sys.path.insert(0, str(ROOT))
    from recognizer.gallery import FaceGallery
    from recognizer.pipeline import Recognizer, draw
    rec = Recognizer(gallery=FaceGallery())
    tiles = []
    out_dir = RES / "annotated_samples"
    out_dir.mkdir(exist_ok=True)
    for name in ["school_bus_street.jpg", "city_bus_front.jpg", "person_with_cell_phone.jpg", "person_texting.jpg"]:
        img = cv2.imread(str(ROOT / "samples" / name))
        shown = draw(img, rec.process(img))
        cv2.imwrite(str(out_dir / name.replace(".jpg", "_annotated.jpg")), shown)
        h = 300
        tiles.append(cv2.resize(shown, (int(shown.shape[1] * h / shown.shape[0]), h)))
    def fit(t):
        f = min(400 / t.shape[1], 300 / t.shape[0])
        t = cv2.resize(t, (int(t.shape[1] * f), int(t.shape[0] * f)))
        out = np.full((300, 400, 3), 255, np.uint8)
        y, x = (300 - t.shape[0]) // 2, (400 - t.shape[1]) // 2
        out[y:y + t.shape[0], x:x + t.shape[1]] = t
        return out
    tiles = [fit(t) for t in tiles]
    gap_v, gap_h = np.full((300, 6, 3), 255, np.uint8), np.full((6, 806, 3), 255, np.uint8)
    grid = np.vstack([np.hstack([tiles[0], gap_v, tiles[1]]), gap_h, np.hstack([tiles[2], gap_v, tiles[3]])])
    cv2.imwrite(str(FIG / "fig10_annotated_examples.png"), grid)
    print("wrote fig10_annotated_examples.png")


if __name__ == "__main__":
    pipeline(), coco(), lfw(), detectors(), openset(), school_bus(), examples()
