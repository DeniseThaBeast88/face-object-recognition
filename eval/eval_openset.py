"""Open-set identification test: can the application name enrolled people and reject strangers?

Protocol (fixed random seed, no person appears in more than one group):
  * 20 enrolled people from LFW, 5 enrollment photos each, up to 5 different probe photos each
  * 200 strangers (one photo each) who are never enrolled: they must be labeled "Unknown"
  * 500 background people (one photo each) used only to calibrate the three-sigma threshold;
    their embeddings are also saved as data/background_embeddings.npy for enroll.py
Outputs results/openset.json and results/openset_confusion.npz.
"""
from __future__ import annotations

import random

import numpy as np

from common import LFW_DIR, ROOT, LFWEmbedder, write_json
from recognizer import config
from recognizer.gallery import FaceGallery

N_ENROLL, N_PER_ENROLL, N_PROBE, N_STRANGERS, N_BACKGROUND = 20, 5, 5, 200, 500


def embs(emb, paths):
    out = [emb.get(p) for p in paths]
    return [e for e in out if e is not None]


def score(gallery, known_probes, stranger_embs):
    names = list(gallery.people)
    idx = {n: i for i, n in enumerate(names + ["Unknown"])}
    conf = np.zeros((len(idx), len(idx)), dtype=int)
    correct = rejected = wrong = 0
    for true, e in known_probes:
        m = gallery.identify(e)
        conf[idx[true], idx[m.name]] += 1
        correct += m.name == true
        rejected += m.name == "Unknown"
        wrong += m.name not in (true, "Unknown")
    fa = 0
    for e in stranger_embs:
        m = gallery.identify(e)
        conf[idx["Unknown"], idx[m.name]] += 1
        fa += m.name != "Unknown"
    n = len(known_probes)
    return {"threshold": gallery.threshold, "identification_rate": correct / n, "false_reject_rate": rejected / n,
            "misidentification_rate": wrong / n, "false_accept_rate": fa / len(stranger_embs),
            "overall_accuracy": (correct + len(stranger_embs) - fa) / (n + len(stranger_embs))}, conf, names


def main():
    rng = random.Random(42)
    emb = LFWEmbedder()
    people = {p.name: sorted(p.glob("*.jpg")) for p in LFW_DIR.iterdir() if p.is_dir()}
    rich = sorted(n for n, v in people.items() if len(v) >= N_PER_ENROLL + N_PROBE)
    single = sorted(n for n, v in people.items() if len(v) == 1)
    enrolled = rng.sample(rich, N_ENROLL)
    rest = rng.sample(single, N_STRANGERS + N_BACKGROUND)
    strangers, background_people = rest[:N_STRANGERS], rest[N_STRANGERS:]

    gallery, probes = FaceGallery(), []
    for name in enrolled:
        files = people[name][:]
        rng.shuffle(files)
        gallery.add(name.replace("_", " "), np.vstack(embs(emb, files[:N_PER_ENROLL])))
        probes += [(name.replace("_", " "), e) for e in embs(emb, files[N_PER_ENROLL:N_PER_ENROLL + N_PROBE])]
    stranger_embs = embs(emb, [people[n][0] for n in strangers])
    background = np.vstack(embs(emb, [people[n][0] for n in background_people]))
    emb.save()
    config.DATA.mkdir(exist_ok=True)
    np.save(config.BACKGROUND_FILE, background.astype(np.float32))

    results = {}
    gallery.threshold = config.SFACE_DEFAULT_COSINE
    results["default_0.363"], _, _ = score(gallery, probes, stranger_embs)
    cal_pair = gallery.calibrate(background, mode="pairwise")
    results["three_sigma_pairwise"], _, _ = score(gallery, probes, stranger_embs)
    cal = gallery.calibrate(background, mode="best_match")
    results["three_sigma_best_match"], conf, names = score(gallery, probes, stranger_embs)
    sweep = []
    for t in np.round(np.arange(0.15, 0.61, 0.025), 3):
        gallery.threshold = float(t)
        sweep.append(score(gallery, probes, stranger_embs)[0])
    out = {"enrolled_people": N_ENROLL, "enroll_photos_each": N_PER_ENROLL, "known_probes": len(probes),
           "strangers": len(stranger_embs), "background": len(background),
           "calibration": cal, "calibration_pairwise": cal_pair, "results": results, "sweep": sweep}
    write_json("openset.json", out)
    np.savez(ROOT / "results" / "openset_confusion.npz", confusion=conf, names=np.array(names + ["Unknown"]))
    for k, v in results.items():
        print(k, {a: round(b, 4) for a, b in v.items()})
    print("calibration", {a: (round(b, 4) if isinstance(b, float) else b) for a, b in cal.items()})


if __name__ == "__main__":
    main()
