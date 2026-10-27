"""Synthetic markets and a time-series-momentum strategy population.

Three regimes with known ground truth:
  noise  — iid returns, no exploitable structure (costs only)
  trend  — positively autocorrelated returns (momentum pays)
  revert — negatively autocorrelated returns (momentum bleeds)

A strategy is sign(sum of last k returns), filled at the close and earning the
next bar. Features come from ONE in-sample window; the label is the **mean
out-of-sample Sharpe across several independent continuation paths**, so the
label itself is not a single noisy draw. The regime is never a feature.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

REGIMES = ("noise", "trend", "revert")


def market_returns(regime: str, n: int, rng: np.random.Generator,
                   sigma: float = 0.01, phi: float = 0.35,
                   df: float = 5.0, jump_p: float = 0.02,
                   jump_mult: float = 8.0) -> np.ndarray:
    """Fat-tailed innovations plus occasional negative jumps.

    Real daily crypto portfolios in the archive show skew around -6 and
    kurtosis in the 10-200 range; Gaussian innovations (kurtosis 3) put every
    real config outside the training support and the classifier answered 0.000
    for all of them. df and the jump term exist to cover that tail shape.
    """
    z = rng.standard_t(df, size=n)
    z = z / np.sqrt(df / (df - 2.0)) if df > 2.0 else z
    eps = z * sigma
    jumps = rng.random(n) < jump_p
    if jumps.any():
        eps[jumps] -= jump_mult * sigma * rng.random(jumps.sum())
    if regime == "noise":
        return eps
    sign = 1.0 if regime == "trend" else -1.0
    return lfilter([1.0], [1.0, -sign * phi], eps)


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    if w <= 0 or w > len(x):
        return out
    c = np.cumsum(np.insert(x, 0, 0.0))
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def tsmom(returns: np.ndarray, k: int = 10, cost_bps: float = 2.0,
          gate: bool = True, ema_span: int = 200):
    """Position = sign of the trailing k-bar return, held only above the trend.

    The gate is what the real configs use (EMA200), and it is also what makes
    the real PnL series full of zeros: long flat stretches produce the high
    kurtosis and negative skew seen in the archive. Without it the synthetic
    distribution had kurtosis ~5 against the real 10-240.
    """
    trail = np.convolve(returns, np.ones(k), mode="full")[:len(returns)]
    sig = np.sign(trail)
    pos = np.concatenate([[0.0], sig[:-1]])
    if gate:
        price = np.cumprod(1.0 + returns)
        alpha = 2.0 / (ema_span + 1.0)
        trend = lfilter([alpha], [1.0, -(1.0 - alpha)], price)
        pos = np.where(price > trend, pos, 0.0)
    pnl = pos * returns
    turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
    return pnl - turn * (cost_bps / 1e4), turn


def _sharpe(pnl: np.ndarray) -> float:
    s = pnl.std(ddof=1)
    return float(pnl.mean() / s * np.sqrt(252)) if s > 0 else 0.0


def build_dataset(n: int, seed: int = 0, bars=1200,
                  is_frac: float = 0.6, paths: int = 3,
                  label_threshold: float = 0.3) -> dict:
    """bars may be an int or a (low, high) tuple sampled per strategy.

    The tuple exists so the training distribution covers the in-sample length
    of the real daily configs (~1,800 bars), not just one fixed length.
    """
    from .features import FEATURE_NAMES, features_from_pnl

    rng = np.random.default_rng(seed)
    rows, labels, meta = [], [], []
    lo, hi = (bars, bars) if isinstance(bars, (int, float)) else (bars[0], bars[1])

    for i in range(n):
        regime = REGIMES[i % len(REGIMES)]
        k = int(rng.integers(5, 21))
        cost = float(rng.choice([1.0, 2.0, 5.0]))
        n_bars = int(rng.integers(lo, hi + 1))
        gate = bool(rng.integers(0, 2))
        split = int(n_bars * is_frac)
        path_sharpes = []
        feat = None
        for p in range(paths):
            rets = market_returns(regime, n_bars, rng)
            pnl, turn = tsmom(rets, k=k, cost_bps=cost, gate=gate)
            if p == 0:
                is_pnl = pnl[:split]
                if is_pnl.std() == 0:
                    break
                feat = features_from_pnl(is_pnl, turn[:split])
            path_sharpes.append(_sharpe(pnl[split:]))
        if feat is None or len(path_sharpes) != paths:
            continue
        rows.append(feat)
        mean_oos = float(np.mean(path_sharpes))
        labels.append(int(mean_oos > label_threshold))
        meta.append({"regime": regime, "k": k, "cost_bps": cost,
                     "oos_sharpe": mean_oos,
                     "oos_path_std": float(np.std(path_sharpes)),
                     "bars": n_bars, "gate": gate})

    return {"X": np.asarray(rows, float), "y": np.asarray(labels, int),
            "names": FEATURE_NAMES, "meta": meta}
