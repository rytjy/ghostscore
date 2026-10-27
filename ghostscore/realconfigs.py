"""Score the real MACD / indicator-family configs with the synthetic model.

Data: trading_research/data/proxy_1h_v5_fullhist.csv (6 coins, 1h bars),
resampled to daily. IS = years <= 2023, OOS = years >= 2024, cost 0.095%/edge,
equal weight across coins — the same accounting as bt_macd_daily.py and
bt_indicator_family.py, so the numbers are comparable with the archive.

Outputs results/real_configs.csv and prints a ranked table. Scores are
P(edge survives) from a model trained on synthetic markets only; they are a
relative ranking of these configs against each other, not a market forecast.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

TR = "/root/.openclaw/workspace/trading_research"
sys.path.insert(0, TR)

from bt_macd_sensitivity import CONFIGS, COINS, load  # noqa: E402
from bt_indicator_family import FAMILIES, p_ao, p_boll, p_donchian, p_rsi2, p_tsmom  # noqa: E402

COST = 0.00095


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def daily_ohlc():
    df = load()
    out = {}
    for coin in COINS:
        d = df[df["coin"] == coin].copy()
        d["t"] = pd.to_datetime(d["ts_end"], unit="ms", utc=True)
        out[coin] = d.set_index("t").resample("1D").agg(
            {"high": "max", "low": "min", "close": "last"}).dropna()
    return out


def macd_pos(c, f, s, sig):
    macd = ema(c, f) - ema(c, s)
    sg = ema(macd, sig)
    return pd.Series(np.where(macd > sg, 1.0, -1.0), index=c.index)


def macd_pnl(closes, f, s, sig, gate):
    ports, turns = [], []
    for c in closes.values():
        pos = macd_pos(c, f, s, sig)
        if gate:
            trend = pd.Series(np.where(c > ema(c, 200), 1.0, -1.0), index=c.index)
            pos = pos.where(pos == trend, 0.0)
        pnl = (pos.shift(1) * c.pct_change() - pos.diff().abs() * COST).fillna(0.0)
        ports.append(pnl)
        turns.append(pos.diff().abs().fillna(0.0))
    return pd.concat(ports, axis=1).mean(axis=1), pd.concat(turns, axis=1).mean(axis=1)


def fam_pnl(ohlc, fn, params, gate=True):
    ports, turns = [], []
    for g in ohlc.values():
        pos = fn(g, *params)
        if gate:
            trend = pd.Series(np.where(g["close"] > ema(g["close"], 200), 1.0, -1.0),
                              index=g.index)
            pos = pos.where(pos == trend, 0.0)
        pnl = (pos.shift(1) * g["close"].pct_change() - pos.diff().abs() * COST).fillna(0.0)
        ports.append(pnl)
        turns.append(pos.diff().abs().fillna(0.0))
    return pd.concat(ports, axis=1).mean(axis=1), pd.concat(turns, axis=1).mean(axis=1)


def score_configs(model, X_cols_order):
    from .features import features_from_pnl

    ohlc = daily_ohlc()
    closes = {k: v["close"] for k, v in ohlc.items()}
    rows = []

    def add(family, label, pnl, turn):
        is_mask = pnl.index.year <= 2023
        oos_mask = pnl.index.year >= 2024
        is_pnl, oos_pnl = pnl[is_mask], pnl[oos_mask]
        if is_pnl.std() == 0 or oos_pnl.std() == 0:
            return
        feat = features_from_pnl(is_pnl.values, turn[is_mask].values)
        rows.append({"family": family, "config": label, "feat": feat,
                     "is_sharpe": feat[0],
                     "oos_sharpe": float(oos_pnl.mean() / oos_pnl.std() * np.sqrt(252)),
                     "is_bars": len(is_pnl)})

    for label, f, s, sig in CONFIGS:
        for gate in (False, True):
            pnl, turn = macd_pnl(closes, f, s, sig, gate)
            add("MACD", label + (" +gate" if gate else ""), pnl, turn)

    fam_fns = {"Donchian": p_donchian, "Bollinger": p_boll, "RSI反转": p_rsi2,
               "时序动量": p_tsmom, "AO": p_ao}
    for fam, grids, _ in FAMILIES:
        fn = fam_fns[fam]
        for params in grids:
            pnl, turn = fam_pnl(ohlc, fn, params)
            add(fam, str(params), pnl, turn)

    X = np.asarray([r["feat"] for r in rows], float)
    p = model.predict_proba(X)
    for r, pi in zip(rows, p):
        r["ghost_score"] = float(pi)
    df = pd.DataFrame(rows).drop(columns=["feat"])
    df = df.sort_values("ghost_score", ascending=False).reset_index(drop=True)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/real_configs.csv", index=False)
    return df
