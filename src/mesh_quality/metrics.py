"""Contest metrics and calibration-only threshold fitting."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

DEFECTS = [
    "abstract",
    "artifacts",
    "intersection",
    "lowpoly",
    "noisy",
    "open",
    "partial",
    "scale",
    "set",
    "simple",
]


def scores(y, p, thresholds=0.5, ranking=True):
    y = np.asarray(y)
    p = np.asarray(p)
    assert y.shape == (len(p), 11) and p.shape[1] == 10 and len(p) > 0
    assert np.isfinite(p).all()
    pred = p >= thresholds
    q = ~pred.any(1)
    qf = f1_score(y[:, 10], q, zero_division=0)
    df = f1_score(y[:, :10], pred, average="weighted", zero_division=0)
    result = {
        "f1_quality": float(qf),
        "f1_defects_weighted": float(df),
        "f1_final": float(10 * (qf + df)),
        "good_rejected": int(((y[:, 10] == 1) & ~q).sum()),
        "bad_accepted": int(((y[:, 10] == 0) & q).sum()),
    }
    if ranking:
        result["weighted_ap"] = float(
            average_precision_score(y[:, :10], p, average="weighted")
        )
    return result


def tune_thresholds(y, p, passes=5):
    # Vectorized F1 updates: AP is never recomputed inside threshold search.
    y = np.asarray(y)
    p = np.asarray(p)
    truth = y[:, :10].astype(bool)
    support = truth.sum(0).astype(float)
    weights = support / support.sum()
    grid = np.unique(np.r_[np.geomspace(0.001, 0.1, 25), np.arange(1, 100) / 100, 0.5])
    t = np.full(10, 0.5)
    pred = p >= t
    history = []

    def defect_f1(pred):
        tp = (pred & truth).sum(0)
        return np.divide(
            2 * tp,
            support + pred.sum(0),
            out=np.zeros(10),
            where=(support + pred.sum(0)) > 0,
        )

    best = scores(y, p, t, False)["f1_final"]
    for pass_id in range(passes):
        changed = False
        for j in range(10):
            candidates = p[:, j, None] >= grid[None, :]
            other = np.delete(pred, j, axis=1).any(1)
            q = ~(other[:, None] | candidates)
            qy = y[:, 10].astype(bool)
            tpq = (q & qy[:, None]).sum(0)
            denq = q.sum(0) + qy.sum()
            qf = np.divide(2 * tpq, denq, out=np.zeros(len(grid)), where=denq > 0)
            tp = (candidates & truth[:, j, None]).sum(0)
            den = support[j] + candidates.sum(0)
            cf = np.divide(2 * tp, den, out=np.zeros(len(grid)), where=den > 0)
            f = defect_f1(pred)
            base = float(np.dot(weights, f) - weights[j] * f[j])
            values = 10 * (qf + base + weights[j] * cf)
            ix = int(values.argmax())
            if values[ix] > best + 1e-12:
                old = t[j]
                t[j] = grid[ix]
                pred[:, j] = candidates[:, ix]
                best = float(values[ix])
                changed = True
                history.append(
                    {
                        "pass": pass_id + 1,
                        "defect": DEFECTS[j],
                        "old": old,
                        "threshold": t[j],
                        "f1_final": best,
                    }
                )
        if not changed:
            break
    return t, pd.DataFrame(history)
