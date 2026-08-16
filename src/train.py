#!/usr/bin/env python3
"""
LightGBM on the tabular features, with isotonic calibration.

Two decisions worth knowing about.

is_unbalance stays FALSE. The deliverable is a calibrated probability, not a
balanced decision rule. Rebalancing inflates every prediction and destroys the
calibration that the whole project is about. The base rate is not a problem to
be fixed, it is the thing being modelled.

Calibration is fit on the VALIDATION fold, never on train (the model is
overconfident on data it memorised) and never on test (that is fitting the
thing you are reporting).

The --feature-set flag is the headline experiment:
    engine_free       nothing that came from a Stockfish eval. A model on this
                      set scores any FEN with python-chess alone, so the static
                      demo works with no engine anywhere.
    engine_assisted   adds cp_before, winpct_before, eval buckets, momentum and
                      prev_own_move_was_blunder.
The gap between them answers "how much of human blunder prediction needs an
engine at all". Run both and put the number in the README.

Usage:
    python train.py --data data/features_blitz.parquet \
        --splits data/splits_blitz.parquet \
        --feature-set engine_free --out models/blitz_free
"""

import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from metrics import evaluate, format_table

# Never features. Mirrors SPEC.md; asserted at load time, not trusted.
LEAK_COLS = {"cp_after", "mate_after", "winpct_after", "win_drop", "blunder",
             "label_valid", "clk_after", "time_spent", "result", "termination"}

# Derived from a Stockfish evaluation, directly or through a lag.
ENGINE_COLS = {"cp_before", "winpct_before", "abs_cp_before", "eval_bucket",
               "winpct_momentum_2", "winpct_momentum_4",
               "prev_own_move_was_blunder"}

CLOCK_COLS = {"clk_before", "clock_frac", "log_clk_before", "tc_base", "tc_inc",
              "own_time_prev1", "own_time_prev2", "own_time_prev3"}

# Identifiers and bookkeeping, not predictive input.
NON_FEATURES = {"game_id", "ply", "fen", "move", "mover", "date", "tc_category",
                "split", "split_gamehash", "component", "component_games", "eco"}

def say(*a, **kw) -> None:
    """print that actually reaches the terminal.

    flush=True is REQUIRED, not decoration. Python only line-buffers stdout
    when it detects a tty and Git Bash / MinTTY does not present as one, so
    without this every line below sits in a 4-8 KB buffer and the job looks
    hung. That has already cost this project a killed run; see FINDINGS.md.
    """
    kw.setdefault("flush", True)
    print(*a, **kw)


def progress(period: int, label: str, t0: float):
    """Per-iteration progress for one LightGBM trial.

    LightGBM is silent here by construction: verbose=-1 plus
    early_stopping(verbose=False). On the week sample a trial took ~30s and
    nobody noticed. On the full month train is 4.4M x 56 and a trial runs for
    minutes with nothing printed at all, which is indistinguishable from a
    hang -- and gets Ctrl-C'd, correctly, by anyone sensible.
    """
    def _cb(env) -> None:
        if period <= 0:
            return
        i = env.iteration + 1
        if i % period and i != env.end_iteration:
            return
        ll = ""
        if env.evaluation_result_list:
            _, metric, val = env.evaluation_result_list[0][:3]
            ll = f" val_{metric}={val:.5f}"
        say(f"    {label} round {i}/{env.end_iteration}{ll} "
            f"({time.time()-t0:.0f}s elapsed)")
    _cb.order = 25
    return _cb


# ORDER IS LOAD-BEARING: --n-trials N takes the FIRST N entries, so the
# best-known configurations go first and a cheap 3-trial run covers both
# winners. Measured winners: 63/0.03 took engine_free on the week sample and
# all three models on the full month; 255/0.02 took engine_assisted on the week
# sample and has never been tried at full-month scale.
#
# Two consequences of reordering, both real:
#   - "--n-trials 3" means something different before and after this change, so
#     every run records the configs it actually tried in its meta.json
#     (`trials_tried`). Do not compare two models by trial count alone.
#   - Never reorder this list while a multi-command `&&` chain is in flight.
#     Each command re-imports this module, so an edit mid-chain silently gives
#     the later models a different search than the earlier ones -- which is
#     exactly what would confound a leakage comparison.
SEARCH_SPACE = [
    dict(num_leaves=63, min_child_samples=500, learning_rate=0.03, feature_fraction=0.7),
    dict(num_leaves=255, min_child_samples=500, learning_rate=0.02, feature_fraction=0.6),
    dict(num_leaves=127, min_child_samples=300, learning_rate=0.03, feature_fraction=0.7),
    dict(num_leaves=31, min_child_samples=200, learning_rate=0.05, feature_fraction=0.8),
    dict(num_leaves=63, min_child_samples=100, learning_rate=0.05, feature_fraction=0.8),
    dict(num_leaves=31, min_child_samples=1000, learning_rate=0.05, feature_fraction=0.9),
]


def select_features(df: pd.DataFrame, feature_set: str, no_clock: bool):
    drop = LEAK_COLS | NON_FEATURES
    if feature_set == "engine_free":
        drop |= ENGINE_COLS
    if no_clock:
        drop |= CLOCK_COLS

    cols = [c for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    cols = [c for c in cols if df[c].nunique(dropna=True) > 1]

    leaked = sorted(set(cols) & LEAK_COLS)
    if leaked:
        raise SystemExit(f"FATAL: leak columns survived selection: {leaked}")
    if feature_set == "engine_free":
        eng = sorted(set(cols) & ENGINE_COLS)
        if eng:
            raise SystemExit(f"FATAL: engine columns in engine_free set: {eng}")
    return cols


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--feature-set", default="engine_assisted",
                    choices=["engine_free", "engine_assisted"])
    ap.add_argument("--no-clock", action="store_true")
    ap.add_argument("--out", required=True, help="prefix, e.g. models/blitz_free")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-trials", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=0, help="0 = all cores")
    ap.add_argument("--log-every", type=int, default=50,
                    help="print val log loss every N boosting rounds; 0 "
                         "silences it. Do not silence it on a full-month run.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    df = pd.read_parquet(args.data)
    sp = pd.read_parquet(args.splits)[["game_id", "split", "split_gamehash"]]
    df = df.merge(sp, on="game_id", how="inner")
    df = df[df.label_valid].copy()
    df["blunder"] = df.blunder.astype(int)

    feats = select_features(df, args.feature_set, args.no_clock)
    tag = f"{args.feature_set}{'_noclock' if args.no_clock else ''}"
    say(f"{len(df):,} label_valid rows | {len(feats)} features | {tag}")
    say(f"  base rate {df.blunder.mean():.4%}")

    parts = {k: df[df[args.split_col] == k] for k in ("train", "val", "test")}
    for k, v in parts.items():
        if not len(v):
            raise SystemExit(f"FATAL: split '{k}' is empty in {args.split_col}")
    base = parts["train"].blunder.mean()

    X = {k: v[feats].to_numpy(dtype=np.float32) for k, v in parts.items()}
    y = {k: v.blunder.to_numpy() for k, v in parts.items()}
    # feature_pre_filter=False is REQUIRED to reuse one Dataset across trials
    # with different min_child_samples. Without it LightGBM raises
    # "Reducing min_data_in_leaf with feature_pre_filter=true..." on trial 2.
    dtrain = lgb.Dataset(X["train"], y["train"], feature_name=feats,
                         params={"feature_pre_filter": False},
                         free_raw_data=False)
    dval = lgb.Dataset(X["val"], y["val"], reference=dtrain,
                       free_raw_data=False)

    best = None
    trial_scores = []
    n_trials = min(max(1, args.n_trials), len(SEARCH_SPACE))
    t_all = time.time()
    for i, params in enumerate(SEARCH_SPACE[:n_trials], 1):
        p = dict(objective="binary", metric="binary_logloss", verbose=-1,
                 seed=args.seed, num_threads=args.threads,
                 is_unbalance=False, **params)
        t0 = time.time()
        label = f"trial {i}/{n_trials}"
        # Announce BEFORE the work, not after. A line that only appears on
        # completion cannot tell you the difference between slow and stuck.
        say(f"  {label} starting: leaves={params['num_leaves']} "
            f"lr={params['learning_rate']} min_child={params['min_child_samples']} "
            f"ff={params['feature_fraction']} (up to {args.rounds} rounds, "
            f"early stop 50)")
        booster = lgb.train(p, dtrain, num_boost_round=args.rounds,
                            valid_sets=[dval],
                            callbacks=[lgb.early_stopping(50, verbose=False),
                                       progress(args.log_every, label, t0)])
        score = booster.best_score["valid_0"]["binary_logloss"]
        cap = booster.best_iteration >= args.rounds
        say(f"  {label} done: iters={booster.best_iteration:<5} "
            f"val_ll={score:.5f} ({time.time()-t0:.0f}s)"
            f"{'  [hit --rounds cap]' if cap else ''}")
        trial_scores.append({"trial": i, "val_logloss": float(score),
                             "best_iteration": int(booster.best_iteration),
                             "hit_round_cap": bool(cap), "seconds": round(time.time()-t0),
                             **params})
        if best is None or score < best[0]:
            best = (score, booster, p, i)

    _, booster, params, best_i = best
    spread = (max(t["val_logloss"] for t in trial_scores)
              - min(t["val_logloss"] for t in trial_scores))
    say(f"\nbest: trial {best_i}, {params['num_leaves']} leaves, "
        f"lr {params['learning_rate']}, {booster.best_iteration} rounds "
        f"({time.time()-t_all:.0f}s for {n_trials} trial(s))")
    say(f"  val log loss spread across trials: {spread:.5f}"
        f"{'  <-- tuning is buying almost nothing' if spread < 0.001 else ''}")
    if best_i == n_trials and n_trials < len(SEARCH_SPACE):
        say(f"  NOTE: the winner is the LAST config tried, so the search may be "
            f"hitting its\n  ceiling. Compare the spread above against what a "
            f"wider search could plausibly\n  buy before spending the time.")
    say("scoring train/val/test ...")

    raw = {k: booster.predict(X[k], num_iteration=booster.best_iteration)
           for k in parts}

    # Isotonic on VAL only. Fitting on train would learn the model's
    # overconfidence on memorised rows; fitting on test would be cheating.
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw["val"], y["val"])
    cal = {k: iso.predict(v) for k, v in raw.items()}

    results = {
        "B0 constant": evaluate(y["test"], np.full(len(y["test"]), base), base),
        "raw (test)": evaluate(y["test"], raw["test"], base),
        "calibrated (test)": evaluate(y["test"], cal["test"], base),
    }

    # Same model, naive game-hash split, to quantify what leakage buys you.
    if "split_gamehash" in df.columns and args.split_col == "split":
        gh_test = df[df.split_gamehash == "test"]
        if len(gh_test):
            pg = iso.predict(booster.predict(
                gh_test[feats].to_numpy(dtype=np.float32),
                num_iteration=booster.best_iteration))
            results["calibrated (game-hash test)"] = evaluate(
                gh_test.blunder.to_numpy(), pg, base)

    say()
    say(format_table(results))

    imp = pd.Series(booster.feature_importance("gain"), index=feats)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    say("\ntop 15 features by gain share:")
    for k, v in imp.head(15).items():
        say(f"  {k:<28} {v:6.2%}")

    booster.save_model(f"{args.out}.txt", num_iteration=booster.best_iteration)
    np.savez(f"{args.out}_iso.npz", x=iso.X_thresholds_, y=iso.y_thresholds_)
    meta = {"feature_set": args.feature_set, "no_clock": args.no_clock,
            "features": feats, "params": params,
            "best_iteration": booster.best_iteration,
            "hit_round_cap": booster.best_iteration >= args.rounds,
            "base_rate_train": float(base), "split_col": args.split_col,
            # Self-describing: SEARCH_SPACE can be reordered, so "n_trials=3"
            # alone does not identify which configurations were compared.
            "n_trials": n_trials,
            "trials_tried": SEARCH_SPACE[:n_trials],
            "trial_scores": trial_scores,
            "results": results,
            "importance": imp.round(5).to_dict()}
    with open(f"{args.out}_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    cb = results["calibrated (test)"]["brier_skill"]
    say(f"\nwrote {args.out}.txt, {args.out}_iso.npz, {args.out}_meta.json")
    say(f"Brier skill (calibrated) = {cb:+.4f}. Compare against B1 from "
        f"baselines.py on the SAME split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
