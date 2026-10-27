"""Command line entry point.

  python -m ghostscore.cli run  --n 1500 --seed 7        # synthetic only
  python -m ghostscore.cli real --n 1200 --seed 7        # score real configs
  python -m ghostscore.cli generate --n 5000 --out data.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def cmd_generate(args):
    import pandas as pd
    from .synth import build_dataset

    d = build_dataset(args.n, seed=args.seed, bars=args.bars)
    df = pd.DataFrame(d["X"], columns=d["names"])
    df["label"] = d["y"]
    for k in ("regime", "k", "cost_bps", "oos_sharpe", "oos_path_std", "bars"):
        df[k] = [m[k] for m in d["meta"]]
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows -> {args.out}")
    print(f"base_rate={df['label'].mean():.3f}")


def cmd_train(args):
    import pandas as pd
    from .model import GhostModel, evaluate

    df = pd.read_csv(args.data)
    drop = ("label", "regime", "k", "cost_bps", "oos_sharpe", "oos_path_std", "bars")
    names = [c for c in df.columns if c not in drop]
    X, y = df[names].to_numpy(float), df["label"].to_numpy(int)
    cut = int(len(df) * 0.7)
    m = GhostModel(random_state=args.seed).fit(X[:cut], y[:cut])
    p = m.predict_proba(X[cut:])
    rep = evaluate(y[cut:], p)
    rep["backend"] = m.backend
    print(json.dumps(rep, indent=2))


def cmd_run(args):
    from .model import GhostModel, evaluate
    from .synth import build_dataset

    d = build_dataset(args.n, seed=args.seed, bars=args.bars)
    X, y = d["X"], d["y"]
    rng = np.random.default_rng(args.seed + 1)
    idx = rng.permutation(len(X))
    cut = int(len(idx) * 0.7)
    tr, te = idx[:cut], idx[cut:]
    m = GhostModel(random_state=args.seed).fit(X[tr], y[tr])
    p = m.predict_proba(X[te])
    rep = evaluate(y[te], p)
    rep["backend"] = m.backend
    rep["train_rows"] = int(len(tr))
    rep["bars"] = args.bars
    print(json.dumps(rep, indent=2))


def cmd_real(args):
    from .model import GhostModel, evaluate
    from .realconfigs import score_configs
    from .synth import build_dataset

    bars = (900, 1900)
    d = build_dataset(args.n, seed=args.seed, bars=bars)
    X, y = d["X"], d["y"]
    rng = np.random.default_rng(args.seed + 1)
    idx = rng.permutation(len(X))
    cut = int(len(idx) * 0.7)
    m = GhostModel(random_state=args.seed).fit(X[idx[:cut]], y[idx[:cut]])
    rep = evaluate(y[idx[cut:]], m.predict_proba(X[idx[cut:]]))
    rep["backend"] = m.backend
    rep["bars"] = list(bars)
    rep["train_rows"] = int(len(X))
    os.makedirs("results", exist_ok=True)
    with open("results/holdout.json", "w") as fh:
        json.dump(rep, fh, indent=2)
    print("synthetic holdout: AUC %.3f  Brier %.3f  ECE %.3f  base %.3f  backend %s" % (
        rep["auc"], rep["brier"], rep["ece"], rep["base_rate"], rep["backend"]))

    # Refit on everything before touching real data.
    m = GhostModel(random_state=args.seed).fit(X, y)
    df = score_configs(m, d["names"])
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("wrote results/real_configs.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ghostscore")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--n", type=int, default=2000)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--bars", type=int, default=1200)
    g.add_argument("--out", default="data.csv")
    g.set_defaults(func=cmd_generate)

    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=cmd_train)

    r = sub.add_parser("run")
    r.add_argument("--n", type=int, default=1500)
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--bars", type=int, default=1200)
    r.set_defaults(func=cmd_run)

    q = sub.add_parser("real")
    q.add_argument("--n", type=int, default=1200)
    q.add_argument("--seed", type=int, default=7)
    q.set_defaults(func=cmd_real)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
