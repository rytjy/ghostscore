"""Ghost detector: TabPFN when installed, gradient boosting otherwise.

The fallback exists so the demo runs before the torch/TabPFN install lands.
Both paths expose fit / predict_proba; the report says which one ran.
"""
from __future__ import annotations

import numpy as np


def backend_available() -> str:
    try:
        import tabpfn  # noqa: F401
        return "tabpfn"
    except Exception:
        return "sklearn-hgb"


class GhostModel:
    def __init__(self, backend: str | None = None, random_state: int = 0):
        self.backend = backend or backend_available()
        self.random_state = random_state
        self.clf = None

    def fit(self, X, y):
        if self.backend == "tabpfn":
            from tabpfn import TabPFNClassifier
            try:
                self.clf = TabPFNClassifier(device="cpu", random_state=self.random_state)
            except TypeError:
                self.clf = TabPFNClassifier()
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.clf = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.06, random_state=self.random_state
            )
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]


def calibration_table(y, p, bins: int = 10) -> dict:
    """Fixed-width probability bins, not quantiles, plus ECE."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if m.sum():
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                         "pred": float(p[m].mean()), "obs": float(y[m].mean())})
            ece += m.sum() / len(y) * abs(p[m].mean() - y[m].mean())
    return {"bins": rows, "ece": float(ece)}


def evaluate(y, p) -> dict:
    from sklearn.metrics import brier_score_loss, roc_auc_score
    y = np.asarray(y)
    p = np.asarray(p)
    out = {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
    }
    cal = calibration_table(y, p)
    out["ece"] = cal["ece"]
    out["calibration"] = cal["bins"]
    return out
