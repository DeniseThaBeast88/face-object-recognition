"""Face verification on the LFW benchmark (6,000 pairs in 10 folds): YuNet, alignment, SFace, cosine score.

Compares three ways to choose the decision threshold:
  1. the default SFace threshold (0.363),
  2. the three-sigma impostor upper control limit, computed on the 9 training folds,
  3. the threshold with the best accuracy on the 9 training folds (the usual LFW protocol).
Outputs results/lfw_verification.json and results/lfw_scores.npz (used by make_figures.py).
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from common import LFW_PAIRS, RESULTS, LFWEmbedder, lfw_path, write_json
from recognizer import config
from recognizer.gallery import control_limits


def read_pairs():
    lines = LFW_PAIRS.read_text().strip().splitlines()
    folds, per = map(int, lines[0].split())
    pairs = []
    for i, line in enumerate(lines[1:]):
        p = line.split("\t")
        fold = i // (2 * per)
        if len(p) == 3:
            pairs.append((fold, 1, lfw_path(p[0], int(p[1])), lfw_path(p[0], int(p[2]))))
        else:
            pairs.append((fold, 0, lfw_path(p[0], int(p[1])), lfw_path(p[2], int(p[3]))))
    return pairs, folds


def accuracy(scores, labels, t):
    return float(np.mean((scores >= t) == (labels == 1)))


def best_threshold(scores, labels):
    cands = np.unique(np.round(scores, 4))
    accs = [accuracy(scores, labels, t) for t in cands]
    return float(cands[int(np.argmax(accs))])


def main():
    pairs, n_folds = read_pairs()
    emb = LFWEmbedder()
    t0 = time.perf_counter()
    scores, labels, folds, missing = [], [], [], 0
    for k, (fold, same, a, b) in enumerate(pairs):
        ea, eb = emb.get(a), emb.get(b)
        if ea is None or eb is None:
            missing += 1
            s = -1.0                      # no face found: the pair is rejected (counted as "different")
        else:
            s = float(ea @ eb)
        scores.append(s), labels.append(same), folds.append(fold)
        if k % 1000 == 0:
            print(f"{k}/{len(pairs)} pairs")
    emb.save()
    elapsed = time.perf_counter() - t0
    scores, labels, folds = np.array(scores), np.array(labels), np.array(folds)

    rows = {"default_0.363": [], "three_sigma_ucl": [], "best_cv": []}
    th = {"three_sigma_ucl": [], "best_cv": []}
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        valid_imp = tr & (labels == 0) & (scores > -1)
        ucl = control_limits(scores[valid_imp])[3]
        bt = best_threshold(scores[tr], labels[tr])
        th["three_sigma_ucl"].append(ucl), th["best_cv"].append(bt)
        rows["default_0.363"].append(accuracy(scores[te], labels[te], config.SFACE_DEFAULT_COSINE))
        rows["three_sigma_ucl"].append(accuracy(scores[te], labels[te], ucl))
        rows["best_cv"].append(accuracy(scores[te], labels[te], bt))

    fpr, tpr, thr = roc_curve(labels, scores)
    def tar_at(far):
        return float(tpr[np.searchsorted(fpr, far, side="right") - 1])
    gen, imp = scores[(labels == 1) & (scores > -1)], scores[(labels == 0) & (scores > -1)]
    g_mu, g_sd, g_lcl, _ = control_limits(gen)
    i_mu, i_sd, _, i_ucl = control_limits(imp)
    out = {
        "pairs": len(pairs), "pairs_with_missing_face": missing, "seconds": round(elapsed, 1),
        "auc": float(roc_auc_score(labels, scores)),
        "tar_at_far_1e-2": tar_at(1e-2), "tar_at_far_1e-3": tar_at(1e-3),
        "accuracy": {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))} for k, v in rows.items()},
        "thresholds": {"default": config.SFACE_DEFAULT_COSINE,
                       "three_sigma_ucl_mean": float(np.mean(th["three_sigma_ucl"])),
                       "best_cv_mean": float(np.mean(th["best_cv"]))},
        "far_frr": {},
        "genuine": {"mean": g_mu, "sd": g_sd, "lcl": g_lcl},
        "impostor": {"mean": i_mu, "sd": i_sd, "ucl": i_ucl},
    }
    for name, t in [("default", config.SFACE_DEFAULT_COSINE), ("three_sigma", out["thresholds"]["three_sigma_ucl_mean"]),
                    ("best_cv", out["thresholds"]["best_cv_mean"])]:
        out["far_frr"][name] = {"threshold": t, "far": float(np.mean(scores[labels == 0] >= t)),
                                "frr": float(np.mean(scores[labels == 1] < t))}
    write_json("lfw_verification.json", out)
    np.savez(RESULTS / "lfw_scores.npz", scores=scores, labels=labels, folds=folds, fpr=fpr, tpr=tpr)
    import json
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
