#!/usr/bin/env python3
"""
Figures and tables for the trained models.

Four things this produces that the training output cannot.

1. RELIABILITY WITHIN RATING BANDS. Aggregate calibration is easy to get right
   by accident: over-predicting weak players and under-predicting strong ones
   averages out to a straight line. The claim "a 1400 blunders here 23% of the
   time" only holds if calibration holds inside each band.

2. THE ELIGIBLE-SUBSET COMPARISON. 12.5% of rows have winpct_before <= 22.46,
   where a 20-point drop is arithmetically impossible, so the label can never
   fire. engine_assisted sees winpct_before and gets those rows free.
   Re-scoring on eligible rows only removes that freebie, and the gap that
   SURVIVES is the honest answer to "how much does an engine help".

3. PR curves with the positive rate drawn in, since PR-AUC's baseline is the
   base rate and not 0.5.

4. A calibration table by predicted-probability decile: mean prediction against
   observed frequency, with counts, so you can see where the model is wrong
   rather than just how much.

Usage:
    python evaluate.py --data data/features_blitz.parquet \
        --splits data/splits_blitz.parquet \
        --models models/blitz_free models/blitz_assisted \
        --out figures/ --metrics-out metrics/evaluation_blitz.json
"""

import argparse
import json
import os

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from metrics import evaluate, format_table  # noqa: E402

# Below this win%, a 20-point drop cannot happen (cp is clamped at -1000, so
# win% bottoms out at 2.46). Derived, not tuned. See FINDINGS.md.
ELIGIBLE_MIN_WINPCT = 22.46

RATING_BANDS = [(0, 1200, "<1200"), (1200, 1600, "1200-1600"),
                (1600, 2000, "1600-2000"), (2000, 9999, "2000+")]

STYLE = dict(free="#4c8dd8", assisted="#d8734c", other="#7ba05b")


def load_model(prefix: str):
    booster = lgb.Booster(model_file=f"{prefix}.txt")
    meta = json.load(open(f"{prefix}_meta.json"))
    iso_path = f"{prefix}_iso.npz"
    if os.path.exists(iso_path):
        d = np.load(iso_path)
        iso = lambda v: np.interp(v, d["x"], d["y"])  # noqa: E731
    else:
        iso = lambda v: v  # noqa: E731
    return booster, meta, iso


def reliability(y, p, n_bins=20):
    """Quantile bins -> (mean predicted, observed frequency, count) per bin."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return np.array([]), np.array([]), np.array([])
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    pred, obs, cnt = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() >= 30:
            pred.append(p[m].mean())
            obs.append(y[m].mean())
            cnt.append(int(m.sum()))
    return np.array(pred), np.array(obs), np.array(cnt)


def colour(name):
    if "free" in name:
        return STYLE["free"]
    if "assisted" in name:
        return STYLE["assisted"]
    return STYLE["other"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--out", default="figures/")
    ap.add_argument("--metrics-out", default=None)
    ap.add_argument("--bins", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_parquet(args.data)
    sp = pd.read_parquet(args.splits)[["game_id", "split", "split_gamehash"]]
    df = df.merge(sp, on="game_id", how="inner")
    df = df[df.label_valid].copy()
    df["blunder"] = df.blunder.astype(int)

    train = df[df[args.split_col] == "train"]
    test = df[df[args.split_col] == "test"].reset_index(drop=True)
    base = train.blunder.mean()
    y = test.blunder.to_numpy()

    eligible = (test.winpct_before > ELIGIBLE_MIN_WINPCT).to_numpy()
    print(f"test {len(test):,} rows | base rate {y.mean():.4%}")
    print(f"eligible (winpct_before > {ELIGIBLE_MIN_WINPCT}): "
          f"{eligible.mean():.1%} of rows, base rate {y[eligible].mean():.4%}")
    impossible_positives = int(y[~eligible].sum())
    if impossible_positives:
        print(f"  WARNING: {impossible_positives} blunders found in rows where "
              f"the label\n  cannot fire. winpct_before and the label disagree; "
              f"the parquet is inconsistent.")
    else:
        print("  structurally impossible rows contain 0 blunders, as required")
    print()

    preds, metas = {}, {}
    for prefix in args.models:
        name = os.path.basename(prefix)
        booster, meta, iso = load_model(prefix)
        X = test[meta["features"]].to_numpy(dtype=np.float32)
        raw = booster.predict(X, num_iteration=meta["best_iteration"])
        preds[name] = {"raw": raw, "cal": iso(raw)}
        metas[name] = meta
        print(f"  scored {name}: {len(meta['features'])} features, "
              f"{meta['feature_set']}")

    # ---------------- tables ------------------------------------------------
    full, elig = {}, {}
    full["B0 constant"] = evaluate(y, np.full(len(y), base), base)
    elig["B0 constant"] = evaluate(y[eligible],
                                   np.full(eligible.sum(), base), base)
    for name, p in preds.items():
        full[name] = evaluate(y, p["cal"], base)
        elig[name] = evaluate(y[eligible], p["cal"][eligible], base)

    print("\n=== ALL TEST ROWS ===")
    print(format_table(full))
    print(f"\n=== ELIGIBLE ROWS ONLY (winpct_before > {ELIGIBLE_MIN_WINPCT}) ===")
    print("Removes the rows where the label cannot fire, so the engine's")
    print("knowledge of that boundary stops counting as skill.")
    print(format_table(elig))

    free = [n for n in preds if "free" in n and "noclock" not in n]
    asst = [n for n in preds if "assisted" in n and "noclock" not in n]
    if free and asst:
        f, a = free[0], asst[0]
        gf, ga = full[f]["brier_skill"], full[a]["brier_skill"]
        ef, ea = elig[f]["brier_skill"], elig[a]["brier_skill"]
        print(f"\nWHAT THE ENGINE BUYS")
        print(f"  all rows:      {gf:+.4f} -> {ga:+.4f}  "
              f"({ga/gf if gf else float('nan'):.2f}x)")
        print(f"  eligible only: {ef:+.4f} -> {ea:+.4f}  "
              f"({ea/ef if ef else float('nan'):.2f}x)")
        print("  The eligible-only ratio is the number to quote. The gap "
              "between the two\n  ratios is the structural-floor freebie.")

    # ---------------- figure 1: reliability ---------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(7, 8),
                             gridspec_kw={"height_ratios": [3, 1]})
    ax, axh = axes
    lim = 0.0
    for name, p in preds.items():
        for key, ls, lab in (("raw", "--", "raw"), ("cal", "-", "calibrated")):
            pr, ob, _ = reliability(y, p[key], args.bins)
            if not len(pr):
                continue
            ax.plot(pr, ob, ls, marker="o", ms=3, color=colour(name),
                    alpha=0.9 if key == "cal" else 0.45,
                    label=f"{name} ({lab})")
            lim = max(lim, pr.max(), ob.max())
    lim *= 1.05
    ax.plot([0, lim], [0, lim], color="#888", lw=1, ls=":", label="perfect")
    ax.set_xlim(0, lim), ax.set_ylim(0, lim)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed blunder frequency")
    ax.set_title("Reliability, all test rows")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    for name, p in preds.items():
        axh.hist(p["cal"], bins=60, range=(0, lim), histtype="step",
                 color=colour(name), label=name)
    axh.set_yscale("log")
    axh.set_xlabel("predicted probability")
    axh.set_ylabel("rows (log)")
    axh.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "reliability.png"), dpi=140)
    plt.close(fig)

    # ---------------- figure 2: reliability by rating band -------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))  # each band spans a
    # different probability range, so autoscale per panel rather than sharing
    band_rows = []
    for axb, (lo, hi, lab) in zip(axes.ravel(), RATING_BANDS):
        m = (test.mover_elo >= lo) & (test.mover_elo < hi)
        m = m.to_numpy()
        if m.sum() < 500:
            axb.set_title(f"{lab} (too few rows)")
            continue
        top = 0.0
        for name, p in preds.items():
            pr, ob, _ = reliability(y[m], p["cal"][m], 12)
            if not len(pr):
                continue
            axb.plot(pr, ob, "-o", ms=3, color=colour(name), label=name)
            top = max(top, pr.max(), ob.max())
            band_rows.append([lab, name, f"{m.sum():,}",
                              f"{y[m].mean():.4f}", f"{p['cal'][m].mean():.4f}",
                              f"{evaluate(y[m], p['cal'][m], base)['ece']:.4f}"])
        top = max(top, 0.01) * 1.05
        axb.plot([0, top], [0, top], ":", color="#888", lw=1)
        axb.set_title(f"{lab}  (n={m.sum():,}, base {y[m].mean():.2%})")
        axb.grid(alpha=0.2)
        axb.legend(fontsize=7)
    for axb in axes[-1]:
        axb.set_xlabel("predicted")
    for axb in axes[:, 0]:
        axb.set_ylabel("observed")
    fig.suptitle("Reliability within rating bands "
                 "(aggregate calibration can hide band-level error)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "reliability_by_rating.png"), dpi=140)
    plt.close(fig)

    if band_rows:
        print("\n=== CALIBRATION BY RATING BAND ===")
        w = [max(len(str(r[i])) for r in band_rows) for i in range(6)]
        hdr = ["band", "model", "rows", "observed", "predicted", "ECE"]
        w = [max(a, len(b)) for a, b in zip(w, hdr)]
        print("  ".join(h.ljust(x) for h, x in zip(hdr, w)))
        print("  ".join("-" * x for x in w))
        for r in band_rows:
            print("  ".join(str(v).ljust(x) for v, x in zip(r, w)))

    # ---------------- figure 3: PR curves -----------------------------------
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, p in preds.items():
        pre, rec, _ = precision_recall_curve(y, p["cal"])
        ax.plot(rec, pre, color=colour(name),
                label=f"{name} (AP={full[name]['pr_auc']:.3f})")
    ax.axhline(y.mean(), color="#888", ls=":", lw=1,
               label=f"base rate = {y.mean():.3f}")
    ax.set_xlabel("recall"), ax.set_ylabel("precision")
    ax.set_title("Precision-recall (baseline is the base rate, not 0.5)")
    ax.legend(fontsize=8), ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "precision_recall.png"), dpi=140)
    plt.close(fig)

    # ---------------- decile table ------------------------------------------
    print("\n=== CALIBRATION BY PREDICTED-PROBABILITY DECILE ===")
    for name, p in preds.items():
        q = pd.qcut(p["cal"], 10, labels=False, duplicates="drop")
        t = pd.DataFrame({"decile": q, "y": y, "p": p["cal"]}).groupby("decile")
        agg = t.agg(rows=("y", "size"), predicted=("p", "mean"),
                    observed=("y", "mean")).reset_index()
        agg["ratio"] = agg.observed / agg.predicted.replace(0, np.nan)
        print(f"\n  {name}")
        print("    " + agg.to_string(index=False,
                                     float_format=lambda v: f"{v:.4f}"))

    if args.metrics_out:
        os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
        with open(args.metrics_out, "w") as fh:
            json.dump({"base_rate_train": float(base),
                       "test_rows": int(len(test)),
                       "eligible_fraction": float(eligible.mean()),
                       "eligible_min_winpct": ELIGIBLE_MIN_WINPCT,
                       "all_rows": full, "eligible_rows": elig}, fh, indent=2)
        print(f"\nwrote {args.metrics_out}")

    print(f"wrote 3 figures to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
