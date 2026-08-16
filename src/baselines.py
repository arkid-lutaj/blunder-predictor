#!/usr/bin/env python3
"""
The bar the real model has to clear.

B0  constant, always the training base rate. Brier skill 0 by construction.
B1  logistic regression on mover_elo alone. THIS is the bar. If LightGBM with
    60 features cannot beat "how good is this player", the project has no
    result worth writing up.
B2  logistic on rating + eval + clock + legal-move count. Four features chosen
    by hand. If B2 is close to the full model, most of your feature
    engineering did nothing and you should say so.

Usage:
    python baselines.py --data data/features_blitz.parquet \
        --splits data/splits_blitz.parquet --out metrics/baselines_blitz.json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from metrics import evaluate, format_table

B1_FEATURES = ["mover_elo"]
B2_FEATURES = ["mover_elo", "winpct_before", "clock_frac", "n_legal"]


def load(data_path: str, splits_path: str) -> pd.DataFrame:
    df = pd.read_parquet(data_path)
    sp = pd.read_parquet(splits_path)[["game_id", "split", "split_gamehash"]]
    df = df.merge(sp, on="game_id", how="inner")
    before = len(df)
    df = df[df.label_valid].copy()
    df["blunder"] = df.blunder.astype(int)
    print(f"{before:,} rows joined, {len(df):,} label_valid, "
          f"base rate {df.blunder.mean():.4%}")
    return df


def fit_logistic(train, test, features, seed=0):
    cols = [c for c in features if c in train.columns]
    missing = set(features) - set(cols)
    if missing:
        print(f"    (missing, skipped: {sorted(missing)})")
    med = train[cols].median()
    Xtr = train[cols].fillna(med).to_numpy(dtype=float)
    Xte = test[cols].fillna(med).to_numpy(dtype=float)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed))
    model.fit(Xtr, train.blunder.to_numpy())
    return model.predict_proba(Xte)[:, 1], model, cols


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load(args.data, args.splits)
    train = df[df[args.split_col] == "train"]
    test = df[df[args.split_col] == "test"]
    if not len(train) or not len(test):
        print(f"FATAL: empty split. Values present: "
              f"{df[args.split_col].unique().tolist()}")
        return 1

    base = train.blunder.mean()
    y = test.blunder.to_numpy()
    print(f"train {len(train):,} rows / test {len(test):,} rows | "
          f"train base rate {base:.4%} | test base rate {y.mean():.4%}\n")

    results, coefs = {}, {}

    results["B0 constant"] = evaluate(y, np.full(len(y), base), base)

    for name, feats in (("B1 rating only", B1_FEATURES),
                        ("B2 rating+eval+clock+moves", B2_FEATURES)):
        print(f"  fitting {name}")
        p, model, cols = fit_logistic(train, test, feats, args.seed)
        results[name] = evaluate(y, p, base)
        lr = model[-1]
        coefs[name] = dict(zip(cols, lr.coef_[0].round(4).tolist()))

    print()
    print(format_table(results))

    print("\nstandardised coefficients (sign and magnitude, not effect sizes):")
    for name, c in coefs.items():
        print(f"  {name}: {c}")

    b1 = results["B1 rating only"]["brier_skill"]
    print(f"\nTHE BAR: B1 Brier skill = {b1:+.4f}")
    if b1 <= 0:
        print("  B1 does not beat a constant. Rating carries no signal in this")
        print("  join, which contradicts the decile tables. Check the merge.")
        return 1
    print("  The full model must beat this, on the same split, to be worth")
    print("  anything. Quote both numbers in the README.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"base_rate_train": base,
                       "split_col": args.split_col,
                       "results": results, "coefficients": coefs}, fh, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
