#!/usr/bin/env python3
"""Render the reliability diagram from a run JSON.

Usage: python scripts/plot_calibration.py results/run_seed7.json
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(path: str) -> None:
    d = json.load(open(path))
    bins = d["calibration"]
    xs = [b["pred"] for b in bins]
    ys = [b["obs"] for b in bins]
    ns = [b["n"] for b in bins]

    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=160)
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1, label="perfect")
    ax.plot(xs, ys, "o-", color="#7a3aa8", lw=1.8, ms=6, label="ghostscore")
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(6, -10),
                    fontsize=7, color="#555")
    ax.set_xlabel("predicted P(edge survives)")
    ax.set_ylabel("observed frequency")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Reliability · AUC {d['auc']:.3f} · Brier {d['brier']:.3f} · ECE {d['ece']:.3f}")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = path.replace(".json", "_calibration.png")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/run_seed7.json")
