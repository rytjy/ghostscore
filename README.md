# ghostscore

A calibrated **ghost detector** for backtest edges: given a strategy's in-sample
statistics, estimate the probability that its edge survives out of sample.

Built for **Hacktoberfest Open-Source AI Challenge · Week 4: Ghost in the Machine**
(deadline 2026-10-31 23:59 PDT / 2026-11-01 14:59 Beijing).

## What it is (and is not)

- **Is**: a small, reproducible experiment. Synthetic markets with known ground
  truth (`noise` / `trend` / `revert`), a time-series-momentum population with
  and without an EMA200 gate, in-sample-only features, a tabular classifier, and
  fixed-bin reliability curves.
- **Is not**: a market predictor. Labels come from a generator, so calibration is
  only claimed **against that generator**.

## Results

### Synthetic (holdout, n=270, 900–1900 bars)

```
AUC 0.850 · Brier 0.180 · ECE 0.141 · base rate 0.400 · backend sklearn-hgb
```

Reliability is the interesting part: the model is **sharp but overconfident at
both tails** — 0.9–1.0 bin predicted 0.988, observed 0.803; 0.0–0.1 predicted
0.013, observed 0.136.

### Real configs from the archive (29 configs, 6 coins, daily)

```
spearman(ghost_score, OOS Sharpe) = 0.512   p = 0.0045   n = 29
top-5 mean OOS Sharpe 0.482 · bottom-5 mean OOS Sharpe 0.220
```

The clearest ghost: **Donchian (20)** has the highest in-sample Sharpe of the set
(0.981) and a ghost score of **0.101**. The archive's previous favourite
**MACD 19/39/9** sits mid-table (0.511 / 0.536 with and without the gate).

Caveat: the OOS Sharpe here is used only to *evaluate* the ranking; the model
never sees it. n=29 is small and the configs were not pre-registered.

## Two failures that shaped it (kept on purpose)

1. **Single OOS window as label** → AUC **0.489**, chance. A 480-bar OOS Sharpe
   has a standard error near 1.0, so the label was mostly noise
   (corr(IS Sharpe, OOS Sharpe) = −0.02). Fix: mean OOS Sharpe over **3
   independent continuation paths**, stronger regimes (phi 0.15 → 0.35).
2. **Real data scored 0.000 on every config.** Two causes: `max_dd` divided by a
   peak that can sit near zero (values like −4.8e11), and Gaussian innovations
   (kurtosis ~3) against real crypto kurtosis 10–240. Fix: multiplicative
   drawdown, Student-t innovations with negative jumps, and an EMA200 gate in
   half the synthetic population so flat stretches are represented.

## Status

- [x] synthetic market + strategy population (gated and ungated)
- [x] scale-free in-sample features (no OOS leakage)
- [x] fixed-bin calibration + ECE
- [x] TabPFN path with gradient-boosting fallback
- [x] real-config scoring from the archive (29 configs)
- [ ] TabPFN installed on this host (CPU torch download in progress)
- [ ] article draft

## Run

```bash
python3 -m ghostscore.cli run  --n 1500 --seed 7     # synthetic only
python3 -m ghostscore.cli real --n 900  --seed 7     # score real configs
python3 scripts/plot_calibration.py results/holdout.json
PYTHONPATH=. python3 tests/test_smoke.py
```

## Why TabPFN

TabPFN is a tabular foundation model: few-shot classification without a training
loop, on CPU, with no data leaving the machine. It also puts the project in the
**Best Use of TabPFN** partner category. Until its install lands, every reported
number says `backend=sklearn-hgb` rather than pretending.

## License

MIT.
