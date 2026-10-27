"""Plain-python smoke test (no pytest required)."""
from ghostscore.model import GhostModel, evaluate
from ghostscore.synth import build_dataset


def test_smoke():
    d = build_dataset(120, seed=3, bars=250)
    assert d["X"].shape[0] == d["y"].shape[0] > 0
    assert d["X"].shape[1] == len(d["names"])
    cut = int(len(d["X"]) * 0.7)
    m = GhostModel(random_state=0).fit(d["X"][:cut], d["y"][:cut])
    p = m.predict_proba(d["X"][cut:])
    rep = evaluate(d["y"][cut:], p)
    assert 0.0 <= rep["brier"] <= 1.0
    assert rep["n"] > 0


if __name__ == "__main__":
    test_smoke()
    print("smoke ok")
