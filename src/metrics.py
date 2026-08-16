"""
Metrics for a 4% base rate. Shared by baselines.py, train.py and evaluate.py so
every table in the repo is computed the same way.

The rules this file encodes:

- Never report accuracy. Predicting "no blunder" always scores 96%.
- Brier alone is useless here. A constant predictor scores 0.038 at a 4% base
  rate, which looks excellent and means nothing. Report the SKILL SCORE against
  that constant: 1 - brier/brier_base. Zero means you matched the base rate,
  negative means you did worse than a constant.
- PR-AUC's baseline is the positive rate, not 0.5. Always print it alongside.
- ROC-AUC is invariant to class balance, so imbalance does NOT inflate it. It
  is worth reporting, but never alone: 0.85 AUC coexists with 8% precision at
  this base rate.
- ECE measures whether "12%" means 12%. That is the actual deliverable.
"""

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)

EPS = 1e-15


def expected_calibration_error(y, p, n_bins=20):
    """Quantile-binned |predicted - observed|, weighted by bin size."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return float("nan")
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    ece = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum():
            ece += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return ece


def evaluate(y_true, y_prob, base_rate=None) -> dict:
    """All metrics for one model on one split."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), EPS, 1 - EPS)
    pos = y.mean()
    base = pos if base_rate is None else base_rate

    const = np.full_like(p, base)
    brier = brier_score_loss(y, p)
    brier_base = brier_score_loss(y, const)
    ll = log_loss(y, p, labels=[0, 1])
    ll_base = log_loss(y, const, labels=[0, 1])

    return {
        "n": len(y),
        "pos_rate": pos,
        "log_loss": ll,
        "ll_skill": 1 - ll / ll_base if ll_base else float("nan"),
        "brier": brier,
        "brier_skill": 1 - brier / brier_base if brier_base else float("nan"),
        "pr_auc": average_precision_score(y, p) if 0 < pos < 1 else float("nan"),
        "pr_baseline": pos,
        "pr_lift": (average_precision_score(y, p) / pos
                    if 0 < pos < 1 else float("nan")),
        "roc_auc": roc_auc_score(y, p) if 0 < pos < 1 else float("nan"),
        "ece": expected_calibration_error(y, p),
        "mean_pred": p.mean(),
    }


COLUMNS = ["model", "n", "log_loss", "ll_skill", "brier", "brier_skill",
           "pr_auc", "pr_baseline", "pr_lift", "roc_auc", "ece", "mean_pred"]


def format_table(results: dict) -> str:
    """results: {model_name: metrics dict} -> aligned text table."""
    rows = []
    for name, m in results.items():
        rows.append([name] + [
            f"{m['n']:,}",
            f"{m['log_loss']:.5f}", f"{m['ll_skill']:+.4f}",
            f"{m['brier']:.5f}", f"{m['brier_skill']:+.4f}",
            f"{m['pr_auc']:.4f}", f"{m['pr_baseline']:.4f}",
            f"{m['pr_lift']:.2f}x", f"{m['roc_auc']:.4f}",
            f"{m['ece']:.4f}", f"{m['mean_pred']:.4f}",
        ])
    widths = [max(len(str(r[i])) for r in ([COLUMNS] + rows))
              for i in range(len(COLUMNS))]
    out = ["  ".join(c.ljust(w) for c, w in zip(COLUMNS, widths)),
           "  ".join("-" * w for w in widths)]
    for r in rows:
        out.append("  ".join(str(v).ljust(w) for v, w in zip(r, widths)))
    return "\n".join(out)
