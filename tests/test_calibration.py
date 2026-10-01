import numpy as np

from childsafe.classifier.calibration import ece, fit_temperature, softmax
from childsafe.classifier.engine import summarize


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(0)
    n, k = 4000, 4
    y = rng.integers(0, k, n)
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), y] += 1.5
    over = logits * 4                      # overconfident model
    t = fit_temperature(over, y)
    assert t > 2.0  # the scaled-up model needs T well above 1
    p_over, p_cal = softmax(over), softmax(over, t)
    e_over = ece(p_over.max(1), (p_over.argmax(1) == y).astype(float))
    e_cal = ece(p_cal.max(1), (p_cal.argmax(1) == y).astype(float))
    assert e_cal < e_over / 2


def test_summarize_binary_view():
    classes = ["knife", "spoon", "toy"]
    bmap = {"knife": "unsafe", "spoon": "safe", "toy": "safe"}
    logits = np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5], [0, 5, 0.]])
    y = np.array([0, 1, 2, 0])
    r = summarize(logits, y, classes, bmap, reject_threshold=0.7)
    assert r["binary"]["unsafe_recall"] == 0.5
    assert r["binary"]["tn"] == 2 and r["binary"]["fn"] == 1
    assert r["reject"]["coverage"] == 1.0
