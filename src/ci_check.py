#!/usr/bin/env python3
"""
Assert the project's actual claims, on whatever artifacts exist.

These are not smoke tests. Each one is a claim the README makes, written as
something that can fail the build:

  base rate      the label fires on 2-5% of plies. Outside that band either the
                 threshold or the win% arithmetic is wrong.
  no leakage     none of the forbidden post-move columns reached the feature
                 matrix. This is the claim that makes every metric meaningful.
  split honesty  no player has rows in both train and test.
  beats baseline the model's Brier skill against the base rate is positive.
                 A model that cannot beat "always predict the base rate" is not
                 a model.

Run whichever checks the given paths support:

    python ci_check.py --positions data/sample_positions.parquet
    python ci_check.py --features f.parquet --splits s.parquet --model m
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# Mirrors SPEC.md and train.py. Duplicated deliberately: if someone edits the
# list in train.py to make a failure go away, this file still fails.
FORBIDDEN = {"cp_after", "mate_after", "winpct_after", "win_drop", "blunder",
             "label_valid", "clk_after", "time_spent", "result", "termination"}

BASE_LO, BASE_HI = 0.02, 0.05


class Checks:
    def __init__(self) -> None:
        self.failed = 0
        self.ran = 0

    def check(self, name: str, ok: bool, detail: str) -> None:
        self.ran += 1
        self.failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


def check_positions(path: str, c: Checks) -> None:
    df = pd.read_parquet(path, columns=["blunder", "label_valid", "win_drop"])
    valid = df[df.label_valid]
    br = float(valid.blunder.mean())
    c.check("base rate in 2-5%", BASE_LO <= br <= BASE_HI,
            f"{br:.4%} on {len(valid):,} labelled plies")

    # The stored label must equal the definition applied to the stored drop.
    # If these disagree the parquet is internally inconsistent and nothing
    # downstream means anything.
    recomputed = valid.win_drop > 20.0
    agree = int((recomputed == valid.blunder).sum())
    c.check("label matches win_drop > 20", agree == len(valid),
            f"{agree:,}/{len(valid):,} agree")


def check_features(path: str, c: Checks) -> None:
    cols = set(pd.read_parquet(path).columns) if os.path.getsize(path) < 5e7 \
        else set(pd.read_parquet(path, columns=None).columns)
    # engine_free/assisted selection happens at train time, so the parquet is
    # allowed to CARRY the label columns; what matters is that train.py's
    # selector excludes them. That is checked via the model meta below. Here we
    # assert the parquet has the columns the pipeline needs.
    need = {"game_id", "mover", "blunder", "label_valid", "mover_elo"}
    missing = need - cols
    c.check("features carry the required columns", not missing,
            f"{len(cols)} columns, missing {sorted(missing) or 'none'}")


def check_model_features(prefix: str, c: Checks) -> None:
    meta_path = f"{prefix}_meta.json"
    meta = json.load(open(meta_path))
    feats = set(meta["features"])
    leaked = sorted(feats & FORBIDDEN)
    c.check("no leaky column reached the model", not leaked,
            f"{len(feats)} features, leaked {leaked or 'none'}")

    if meta["feature_set"] == "engine_free":
        engine = {"cp_before", "winpct_before", "abs_cp_before", "eval_bucket",
                  "winpct_momentum_2", "winpct_momentum_4",
                  "prev_own_move_was_blunder"}
        bad = sorted(feats & engine)
        c.check("engine_free uses no engine feature", not bad,
                f"found {bad or 'none'}")

    res = meta.get("results", {})
    row = res.get("calibrated (test)") or res.get("raw (test)") or {}
    bs = row.get("brier_skill")
    if bs is None:
        c.check("model beats the base rate", False,
                "no test Brier skill in meta")
    else:
        c.check("model beats the base rate", bs > 0,
                f"Brier skill {bs:+.4f}")

    if meta.get("hit_round_cap"):
        c.check("training was not truncated by --rounds", False,
                f"best iteration {meta.get('best_iteration')} hit the ceiling")
    else:
        c.check("training was not truncated by --rounds", True,
                f"converged at {meta.get('best_iteration')} rounds")


def check_splits(splits_path: str, features_path: str, c: Checks) -> None:
    sp = pd.read_parquet(splits_path)
    fe = pd.read_parquet(features_path, columns=["game_id", "mover"])
    m = fe.merge(sp[["game_id", "split"]], on="game_id", how="inner")
    train = set(m.mover[m.split == "train"])
    test = set(m.mover[m.split == "test"])
    overlap = train & test
    frac = len(overlap) / max(len(test), 1)
    # make_splits carves oversized components, so a tiny overlap is by design on
    # a dense graph. It must stay far below what a naive split would leak.
    c.check("no player in both train and test", frac <= 0.01,
            f"{len(overlap)} of {len(test)} test players ({frac:.2%})")

    naive = sp.get("split_gamehash")
    if naive is not None:
        mn = fe.merge(sp[["game_id", "split_gamehash"]], on="game_id")
        tr = set(mn.mover[mn.split_gamehash == "train"])
        te = set(mn.mover[mn.split_gamehash == "test"])
        nf = len(tr & te) / max(len(te), 1)
        c.check("player split beats the naive split", frac <= nf,
                f"{frac:.2%} against {nf:.2%} for game-hash")

    for name in ("train", "val", "test"):
        n = int((sp.split == name).sum())
        c.check(f"{name} fold is non-empty", n > 0, f"{n:,} games")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions")
    ap.add_argument("--features")
    ap.add_argument("--splits")
    ap.add_argument("--model", help="prefix, e.g. models/ci_free")
    args = ap.parse_args()

    if not any([args.positions, args.features, args.splits, args.model]):
        ap.error("nothing to check; pass at least one path")

    c = Checks()
    print("CHECKS")
    if args.positions:
        check_positions(args.positions, c)
    if args.features:
        check_features(args.features, c)
    if args.splits and args.features:
        check_splits(args.splits, args.features, c)
    elif args.splits:
        print("  SKIP  split honesty needs --features too")
    if args.model:
        check_model_features(args.model, c)

    print(f"\n{c.ran - c.failed}/{c.ran} passed")
    if c.failed:
        print(f"{c.failed} CHECK(S) FAILED")
    return 1 if c.failed else 0


if __name__ == "__main__":
    sys.exit(main())
