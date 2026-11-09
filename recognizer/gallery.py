"""Face gallery (enrolled people) and the three-sigma decision threshold.

The threshold method adapts the three-sigma control limits that Dileep and Danti used to accept or reject
neural network outputs for age and gender classes. Here the outputs are cosine similarity scores:

    impostor upper control limit  UCL = mean(impostor scores) + 3 * sd(impostor scores)
    genuine lower control limit   LCL = mean(genuine scores)  - 3 * sd(genuine scores)

A score above the impostor UCL is very unlikely to come from a different person, so the face is accepted as
the best-matching enrolled person. Otherwise the face is labeled "Unknown".
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from . import config


def control_limits(scores) -> tuple[float, float, float, float]:
    """Return mean, sample standard deviation, lower limit, and upper limit (mean -/+ 3 sd)."""
    s = np.asarray(scores, dtype=np.float64)
    mu = float(s.mean())
    sd = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    return mu, sd, mu - 3 * sd, mu + 3 * sd


def three_sigma_threshold(genuine, impostor) -> dict:
    g_mu, g_sd, g_lcl, _ = control_limits(genuine)
    i_mu, i_sd, _, i_ucl = control_limits(impostor)
    return {"threshold": i_ucl, "impostor_mean": i_mu, "impostor_sd": i_sd, "impostor_ucl": i_ucl,
            "genuine_mean": g_mu, "genuine_sd": g_sd, "genuine_lcl": g_lcl,
            "separated": bool(g_lcl > i_ucl)}


@dataclass
class Match:
    name: str
    similarity: float
    known: bool


class FaceGallery:
    def __init__(self, threshold: float = config.SFACE_DEFAULT_COSINE):
        self.people: dict[str, np.ndarray] = {}
        self.threshold = threshold
        self.calibration: dict = {}

    # ---- enrollment -------------------------------------------------------------------------
    def add(self, name: str, embeddings: np.ndarray) -> None:
        emb = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
        self.people[name] = np.vstack([self.people[name], emb]) if name in self.people else emb

    def remove(self, name: str) -> None:
        self.people.pop(name, None)

    def genuine_scores(self) -> np.ndarray:
        """Similarity between different samples of the same enrolled person."""
        out = []
        for emb in self.people.values():
            if len(emb) > 1:
                s = emb @ emb.T
                out.extend(s[np.triu_indices(len(emb), 1)])
        return np.array(out)

    def impostor_scores(self, background: np.ndarray | None = None, mode: str = "best_match") -> np.ndarray:
        """Similarity between enrolled people and other people (other enrolled people and a background set).

        mode="best_match" (used by the application) scores each other face the same way identify() does: its
        highest similarity to any stored sample of any enrolled person. This is the score an unknown visitor
        would actually receive, so its three-sigma limit matches the open-set decision.
        mode="pairwise" keeps every single comparison, as in a one-to-one verification test.
        """
        names = list(self.people)
        if mode == "pairwise":
            out = []
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    out.extend((self.people[a] @ self.people[b].T).ravel())
                if background is not None and len(background):
                    out.extend((self.people[a] @ background.T).ravel())
            return np.array(out)
        out = []
        for a in names:                                   # each enrolled sample against the other people
            others = [self.people[b] for b in names if b != a]
            if others:
                out.extend((self.people[a] @ np.vstack(others).T).max(axis=1))
        if background is not None and len(background) and names:
            out.extend((background @ np.vstack([self.people[n] for n in names]).T).max(axis=1))
        return np.array(out)

    def calibrate(self, background: np.ndarray | None = None, mode: str = "best_match") -> dict:
        g, i = self.genuine_scores(), self.impostor_scores(background, mode)
        if len(g) < 2 or len(i) < 2:
            self.calibration = {"threshold": self.threshold, "note": "not enough samples; default kept"}
            return self.calibration
        self.calibration = three_sigma_threshold(g, i)
        self.calibration.update(mode=mode, genuine_n=int(len(g)), impostor_n=int(len(i)))
        self.threshold = float(self.calibration["threshold"])
        return self.calibration

    # ---- recognition ------------------------------------------------------------------------
    def identify(self, embedding: np.ndarray) -> Match:
        best_name, best = "Unknown", -1.0
        for name, emb in self.people.items():
            s = float((emb @ embedding).max())
            if s > best:
                best_name, best = name, s
        if best >= self.threshold:
            return Match(best_name, best, True)
        return Match("Unknown", best, False)

    # ---- storage (embeddings only; no face images are saved) --------------------------------
    def save(self, path=config.GALLERY_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        names = list(self.people)
        np.savez(path, names=np.array(names), counts=np.array([len(self.people[n]) for n in names]),
                 embeddings=np.vstack([self.people[n] for n in names]) if names else np.zeros((0, 128)),
                 threshold=self.threshold, calibration=json.dumps(self.calibration))

    @classmethod
    def load(cls, path=config.GALLERY_FILE) -> "FaceGallery":
        g = cls()
        if not path.exists():
            return g
        z = np.load(path, allow_pickle=False)
        start = 0
        for name, n in zip(z["names"], z["counts"]):
            g.people[str(name)] = z["embeddings"][start:start + int(n)]
            start += int(n)
        g.threshold = float(z["threshold"])
        g.calibration = json.loads(str(z["calibration"]))
        return g
