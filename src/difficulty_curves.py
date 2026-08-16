#!/usr/bin/env python3
"""
Phase 8. The deliverable figure: predicted blunder rate against rating, for
four positions of visibly different character.

This is the picture the whole project exists to draw. Each curve answers "how
often does a player of rating R blunder HERE", and the four separate because
position difficulty is not one number.

THE CAVEAT IS IMPLEMENTED, NOT MENTIONED. A rating sweep is an extrapolation:
a 2200 rarely reaches the position a 1200 blunders in, so parts of every curve
sit outside the training distribution and the model is guessing there. Drawing
a confident line through that region would be the most dishonest thing this
repo could do.

So for every position and every rating point we count how many TRAINING rows
actually live in that region of feature space -- same rating bin, similar legal
move count, similar material balance, similar phase -- and grey out the stretch
where support falls below a floor. The greyed part of a curve is the model
extrapolating.

WHY THE NO-CLOCK MODEL. These are bare FENs with no game around them: no clock,
no time control, no opponent. `full_free_noclock`'s only clock-like feature is
`halfmove_clock`, a FEN field, so it is exactly computable here with nothing
imputed. Using the with-clock model would mean inventing six features carrying
~9% of its gain. The clock ablation priced that substitution at 1.4% of Brier
skill, which is what makes this an easy trade. Same reasoning as
validate_puzzles.py.

The positions are REAL, taken from the dataset rather than composed, so their
descriptions match their measured features and they are guaranteed to be
in-distribution somewhere. Composed FENs kept failing to have the property
their caption claimed.

Usage:
    python difficulty_curves.py --data data/features_blitz_full.parquet \
        --model models/full_free_noclock --out figures/
"""

import argparse
import json
import os

import chess
import numpy as np
import pandas as pd

from build_features import check_sweep_span, position_features, rating_grid

RATINGS = np.arange(800, 2401, 50)

# Real positions from features_blitz_full.parquet, chosen so each one's caption
# is true of its measured features. All sit within winpct_before 35-65 so the
# curve is about difficulty rather than about converting a won game.
POSITIONS = [
    {"name": "Quiet endgame",
     "fen": "8/p7/1b3pp1/5p2/3k4/5P1P/P2NK1P1/8 b - - 0 45",
     "note": "12 pieces, no tension, nothing hanging, 12 legal moves"},
    {"name": "Sharp middlegame",
     "fen": "r3k1nr/pp1bq1pp/2pb1p2/3p4/3N1P2/PP1nP1P1/1B1P2BP/RN1Q1K1R b kq - 2 14",
     "note": "30 pieces, tension 7, 45 legal moves"},
    {"name": "Opening",
     "fen": "rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
     "note": "move 2, everything still on the board"},
    {"name": "Piece hanging",
     "fen": "8/2p1rp1k/1p2qnpp/p2Qp3/P3P3/2P4P/1P2NPPK/3R4 w - - 5 31",
     "note": "own queen loose, 43 legal moves"},
]

# Support cell. Deliberately coarse: the question is "did the model ever see
# anything like this at this rating", not "did it see this exact position".
RATING_BIN = 100        # +/- this many Elo
LEGAL_TOL = 6           # +/- legal moves
MATERIAL_TOL = 150      # +/- centipawns of material balance
PIECE_TOL = 6           # +/- total pieces

# What counts as "enough support" is set by EXPECTED BLUNDER EVENTS, not by a
# round number of rows. At a ~3.95% base rate a cell of 200 rows holds about 8
# blunders, which cannot pin down a probability; 50 events is the usual
# rule-of-thumb floor for a stable rate estimate. So the row threshold is
# derived from the measured base rate rather than picked.
MIN_EVENTS = 50


def features_for(fen: str, feat_names: list[str]) -> np.ndarray:
    """Feature vector for a bare FEN. Fails loudly rather than imputing."""
    board = chess.Board(fen)
    f = position_features(fen)
    f["mover_is_white"] = int(board.turn == chess.WHITE)
    f["move_number"] = board.fullmove_number
    f["ply"] = (board.fullmove_number - 1) * 2 + (0 if board.turn else 1)
    for n in ("mover_elo", "opp_elo", "mean_elo"):
        f[n] = 1500.0
    f["elo_gap"] = 0.0
    missing = set(feat_names) - set(f)
    if missing:
        raise SystemExit(
            f"FATAL: cannot build {sorted(missing)} from a bare FEN.\n"
            f"Use a *_noclock model, or add the feature deliberately rather "
            f"than imputing it.")
    return np.array([f[n] for n in feat_names], dtype=np.float32)


def support_curve(train: pd.DataFrame, fen: str) -> np.ndarray:
    """Training rows per rating point in this position's neighbourhood.

    Counted on the four axes that actually move the model here: how many legal
    moves, how much material, how many pieces (phase) and the rating itself.
    A finer cell would return zero everywhere and a coarser one would never
    warn, so this is calibrated to leave the middle of the rating range
    supported for typical positions.
    """
    p = position_features(fen)
    m = (train.n_legal.sub(p["n_legal"]).abs() <= LEGAL_TOL) & \
        (train.material_balance.sub(p["material_balance"]).abs() <= MATERIAL_TOL) & \
        (train.total_pieces.sub(p["total_pieces"]).abs() <= PIECE_TOL)
    elos = train.mover_elo[m].to_numpy()
    return np.array([np.sum(np.abs(elos - r) <= RATING_BIN) for r in RATINGS])


def load_iso(prefix: str):
    p = f"{prefix}_iso.npz"
    if not os.path.exists(p):
        return lambda v: v
    d = np.load(p)
    return lambda v: np.interp(v, d["x"], d["y"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="features parquet, for the support density")
    ap.add_argument("--model", default="models/full_free_noclock")
    ap.add_argument("--out", default="figures/")
    ap.add_argument("--metrics-out", default="metrics/difficulty_curves.json")
    ap.add_argument("--min-support", type=int, default=None,
                    help="row floor for 'supported'; default derives it from "
                         "the base rate so the cell holds ~50 blunders")
    ap.add_argument("--min-events", type=int, default=MIN_EVENTS)
    args = ap.parse_args()

    import lightgbm as lgb
    booster = lgb.Booster(model_file=f"{args.model}.txt")
    meta = json.load(open(f"{args.model}_meta.json"))
    feat_names = meta["features"]
    iso = load_iso(args.model)

    clockish = [f for f in feat_names
                if any(k in f.lower() for k in ("clk", "clock", "tc_", "time"))
                and f != "halfmove_clock"]
    if clockish:
        print(f"FATAL: {args.model} needs clock features {clockish}, which a "
              f"bare FEN cannot supply. Use a *_noclock model.")
        return 1
    print(f"model {os.path.basename(args.model)} "
          f"({meta['feature_set']}, {len(feat_names)} features)")

    train = pd.read_parquet(
        args.data, columns=["mover_elo", "n_legal", "material_balance",
                            "total_pieces", "label_valid", "blunder"])
    train = train[train.label_valid]
    base = float(train.blunder.fillna(False).mean())
    if args.min_support is None:
        args.min_support = int(round(args.min_events / max(base, 1e-9)))
    print(f"{len(train):,} training rows for the support density")
    print(f"base rate {base:.4%}, so {args.min_events} expected blunders needs "
          f"{args.min_support:,} rows -- that is the support floor\n")

    curves, supports = [], []
    for p in POSITIONS:
        x = features_for(p["fen"], feat_names)
        curve = np.asarray(iso(booster.predict(
            rating_grid(x, feat_names, RATINGS))), dtype=float)
        sup = support_curve(train, p["fen"])
        curves.append(curve)
        supports.append(sup)
        ok = sup >= args.min_support
        span = curve.max() / max(curve.min(), 1e-12)
        print(f"  {p['name']:18s} {curve[0]:6.2%} at 800 -> "
              f"{curve[-1]:6.2%} at 2400   span {span:5.2f}x")
        print(f"  {'':18s} supported over "
              f"{RATINGS[ok].min() if ok.any() else 0}-"
              f"{RATINGS[ok].max() if ok.any() else 0} Elo "
              f"({ok.sum()}/{len(RATINGS)} points, median {np.median(sup):,.0f} rows)")

    # The same gate the other consumers use. A flat set of curves here would
    # mean the sweep is broken, and this figure is the one people will look at.
    span = check_sweep_span(curves, "difficulty curves")
    print(f"\nsweep span check: median {span:.2f}x")

    # ----- figure ---------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 6))
    colours = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    for p, curve, sup, c in zip(POSITIONS, curves, supports, colours):
        ok = sup >= args.min_support
        # full curve faint, supported stretch solid on top
        ax.plot(RATINGS, curve * 100, color=c, lw=1.2, alpha=0.30,
                zorder=2, solid_capstyle="round")
        seg = np.where(ok, curve * 100, np.nan)
        ax.plot(RATINGS, seg, color=c, lw=2.8, zorder=3,
                label=f"{p['name']} — {p['note']}")

    # shade every rating region unsupported for ALL FOUR, which is where the
    # figure as a whole stops being evidence
    none_ok = np.all([s < args.min_support for s in supports], axis=0)
    if none_ok.any():
        d = np.diff(np.concatenate([[0], none_ok.view(np.int8), [0]]))
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            ax.axvspan(RATINGS[s], RATINGS[min(e, len(RATINGS) - 1)],
                       color="#999999", alpha=0.16, zorder=1, lw=0)

    ax.plot([], [], color="#666666", lw=1.2, alpha=0.5,
            label=f"faint = sparse training support (<{args.min_support:,} rows)")
    ax.set_xlabel("player rating (Elo), all rating features swept together")
    ax.set_ylabel("predicted P(blunder) at this position (%)")
    ax.set_title("Position difficulty is not one number:\n"
                 "predicted blunder rate against rating, four real positions",
                 fontsize=13)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.93)
    ax.set_xlim(RATINGS[0], RATINGS[-1])
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    dest = os.path.join(args.out, "difficulty_curves.png")
    fig.savefig(dest, dpi=150)
    print(f"wrote {dest}")

    if args.metrics_out:
        os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
        with open(args.metrics_out, "w") as fh:
            json.dump({"model": os.path.basename(args.model),
                       "ratings": RATINGS.tolist(),
                       "min_support": args.min_support,
                       "positions": [
                           {**p, "curve": c.tolist(), "support": s.tolist()}
                           for p, c, s in zip(POSITIONS, curves, supports)]},
                      fh, indent=2, default=float)
        print(f"wrote {args.metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
