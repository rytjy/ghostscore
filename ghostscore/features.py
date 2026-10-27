"""In-sample-only features. Nothing here may touch the out-of-sample slice.

Scale-free by construction: absolute mean and volatility are deliberately NOT
features, because synthetic markets (1% daily vol) and real crypto (3-5%) differ
by a factor that a tree model would read as a regime. Sharpe, drawdown, turnover,
skew, kurtosis, hit rate and profit factor are all comparable across scales.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "is_sharpe", "skew", "kurtosis", "hit_rate", "turnover", "max_dd",
    "profit_factor", "t_stat", "autocorr1", "sharpe_se", "n_bars",
]


def _max_drawdown(pnl: np.ndarray) -> float:
    """Multiplicative equity, so the value is scale-free and bounded in [-1, 0].

    The earlier cumsum version divided by a peak that can sit near zero and
    produced drawdowns of -4.8e11; that single bug flattened every real score.
    """
    r = np.clip(np.asarray(pnl, float), -0.999, None)
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def features_from_pnl(pnl: np.ndarray, turnover: np.ndarray) -> list[float]:
    n = len(pnl)
    mean = float(pnl.mean())
    std = float(pnl.std(ddof=1)) if n > 1 else 0.0
    sharpe = mean / std * np.sqrt(252) if std > 0 else 0.0
    gains, losses = pnl[pnl > 0], pnl[pnl < 0]
    hit = float((pnl > 0).mean())
    pf = float(gains.sum() / abs(losses.sum())) if losses.size else 10.0
    t_stat = mean / (std / np.sqrt(n)) if std > 0 else 0.0
    ac1 = float(np.corrcoef(pnl[:-1], pnl[1:])[0, 1]) if n > 2 else 0.0
    if not np.isfinite(ac1):
        ac1 = 0.0
    skew = float(((pnl - mean) ** 3).mean() / (std ** 3)) if std > 0 else 0.0
    kurt = float(((pnl - mean) ** 4).mean() / (std ** 4)) if std > 0 else 0.0
    se = float(np.sqrt((1 + 0.5 * sharpe ** 2) / max(n, 1)) * np.sqrt(252))
    return [
        float(sharpe), skew, kurt, hit,
        float(np.mean(turnover)) if len(turnover) else 0.0,
        _max_drawdown(pnl), pf, float(t_stat), ac1, se, float(n),
    ]
