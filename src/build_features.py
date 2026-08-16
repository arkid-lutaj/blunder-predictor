#!/usr/bin/env python3
"""
Second pass: positions parquet -> features parquet.

Two feature sets are produced in one file and selected by name at train time:

  engine_free      everything python-chess can compute from the FEN alone,
                   plus rating and clock. A model on this set needs NO engine
                   at inference, so a paste-a-FEN demo works with nothing but
                   python-chess.
  engine_assisted  the above plus cp_before, winpct_before, the eval bucket,
                   eval momentum, and prev_own_move_was_blunder. All of these
                   trace back to a Stockfish evaluation, so a model using them
                   requires an engine call before it can score a position.

Reporting both answers a question worth asking: how much of human blunder
prediction needs an engine at all? The gap between the two is the answer, and
it is a single number.

NAMING: pool suffix is carried through. positions_blitz_YYYY-MM.parquet ->
features_blitz.parquet. Blitz and rapid are separate Glicko pools and must
never be mixed.

Usage:
    python build_features.py --data data/positions_blitz_2026-06.parquet \
        --out data/features_blitz.parquet --workers 8
"""

import argparse
import os
import sys
from multiprocessing import Pool

import chess
import numpy as np
import pandas as pd

from tactical import VAL, compute_tactical_state

# ---------------------------------------------------------------------------
# Columns that must never become features
# ---------------------------------------------------------------------------

LEAK_COLS = [
    "cp_after", "mate_after", "winpct_after", "win_drop", "blunder",
    "label_valid", "clk_after", "time_spent", "result", "termination",
]
# Carried through for labelling and grouping, but not features.
CARRY_COLS = ["game_id", "ply", "fen", "move", "mover", "blunder",
              "label_valid", "win_drop", "tc_category", "date"]

ENGINE_COLS = [
    "cp_before", "winpct_before", "abs_cp_before", "eval_bucket",
    "winpct_momentum_2", "winpct_momentum_4", "prev_own_move_was_blunder",
]

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
PIECE_NAMES = {chess.PAWN: "p", chess.KNIGHT: "n", chess.BISHOP: "b",
               chess.ROOK: "r", chess.QUEEN: "q"}


# ---------------------------------------------------------------------------
# Position features, from the FEN alone
# ---------------------------------------------------------------------------


def pawn_structure(board: chess.Board, color: bool) -> tuple[int, int, int]:
    """passed, isolated, doubled counts for `color`."""
    own = board.pieces(chess.PAWN, color)
    opp = board.pieces(chess.PAWN, not color)
    own_files = [0] * 8
    for sq in own:
        own_files[chess.square_file(sq)] += 1

    passed = isolated = 0
    for sq in own:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if not any(own_files[g] for g in (f - 1, f + 1) if 0 <= g <= 7):
            isolated += 1
        blocked = False
        for osq in opp:
            of, orank = chess.square_file(osq), chess.square_rank(osq)
            if abs(of - f) <= 1:
                if (color == chess.WHITE and orank > r) or \
                   (color == chess.BLACK and orank < r):
                    blocked = True
                    break
        if not blocked:
            passed += 1
    doubled = sum(c - 1 for c in own_files if c > 1)
    return passed, isolated, doubled


def position_features(fen: str) -> dict:
    board = chess.Board(fen)
    mover = board.turn
    f: dict[str, float] = {}

    legal = list(board.legal_moves)
    f["n_legal"] = len(legal)
    f["n_captures"] = sum(1 for m in legal if board.is_capture(m))
    f["n_checks"] = sum(1 for m in legal if board.gives_check(m))
    f["n_quiet"] = f["n_legal"] - f["n_captures"] - f["n_checks"]
    f["n_promotions"] = sum(1 for m in legal if m.promotion)
    f["in_check"] = int(board.is_check())

    f.update(compute_tactical_state(board))

    own_mat = opp_mat = 0
    for pt in PIECE_TYPES:
        no = len(board.pieces(pt, mover))
        nx = len(board.pieces(pt, not mover))
        f[f"n_{PIECE_NAMES[pt]}_own"] = no
        f[f"n_{PIECE_NAMES[pt]}_opp"] = nx
        own_mat += no * VAL[pt]
        opp_mat += nx * VAL[pt]
    f["material_own"] = own_mat
    f["material_opp"] = opp_mat
    f["material_balance"] = own_mat - opp_mat
    f["non_pawn_material"] = (
        own_mat - len(board.pieces(chess.PAWN, mover)) * VAL[chess.PAWN]
        + opp_mat - len(board.pieces(chess.PAWN, not mover)) * VAL[chess.PAWN]
    )
    f["total_pieces"] = chess.popcount(board.occupied)

    for color, tag in ((mover, "own"), (not mover, "opp")):
        k = board.king(color)
        if k is None:
            f[f"king_ring_atk_{tag}"] = 0
            f[f"king_ring_def_{tag}"] = 0
            f[f"king_escapes_{tag}"] = 0
            continue
        ring = chess.SquareSet(chess.BB_KING_ATTACKS[k])
        f[f"king_ring_atk_{tag}"] = sum(
            chess.popcount(board.attackers_mask(not color, s)) for s in ring)
        f[f"king_ring_def_{tag}"] = sum(
            chess.popcount(board.attackers_mask(color, s)) for s in ring)
        f[f"king_escapes_{tag}"] = sum(
            1 for s in ring
            if not board.piece_at(s) and not board.is_attacked_by(not color, s))

    for color, tag in ((mover, "own"), (not mover, "opp")):
        pas, iso, dbl = pawn_structure(board, color)
        f[f"passed_pawns_{tag}"] = pas
        f[f"isolated_pawns_{tag}"] = iso
        f[f"doubled_pawns_{tag}"] = dbl

    f["can_castle_own"] = int(bool(board.castling_rights & (
        chess.BB_A1 | chess.BB_H1 if mover == chess.WHITE
        else chess.BB_A8 | chess.BB_H8)))
    f["can_castle_opp"] = int(bool(board.castling_rights & (
        chess.BB_A8 | chess.BB_H8 if mover == chess.WHITE
        else chess.BB_A1 | chess.BB_H1)))
    f["ep_available"] = int(board.ep_square is not None)
    f["halfmove_clock"] = board.halfmove_clock
    return f


def _worker(fens: list[str]) -> list[dict]:
    return [position_features(x) for x in fens]


# ---------------------------------------------------------------------------
# Context and lag features
# ---------------------------------------------------------------------------


def context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Everything derived from the row context. Lags reach BACKWARD only."""
    out = pd.DataFrame(index=df.index)

    out["mover_elo"] = df.mover_elo
    out["opp_elo"] = df.opp_elo
    out["elo_gap"] = df.mover_elo - df.opp_elo
    out["mean_elo"] = (df.mover_elo + df.opp_elo) / 2
    out["mover_is_white"] = df.mover_is_white.astype(int)
    out["ply"] = df.ply
    out["move_number"] = df.ply // 2 + 1

    out["tc_base"] = df.tc_base
    out["tc_inc"] = df.tc_inc
    out["clk_before"] = df.clk_before
    out["clock_frac"] = df.clk_before / df.tc_base.replace(0, np.nan)
    out["log_clk_before"] = np.log1p(df.clk_before.clip(lower=0))

    # Sort once; every lag below assumes this order.
    order = df.sort_values(["game_id", "ply"]).index
    g = df.loc[order].groupby("game_id", sort=False)

    # The mover alternates each ply, so a player's own previous move is 2 plies
    # back. time_spent of the CURRENT move is decided simultaneously with the
    # move and is on the leak list; only its lags are admissible.
    for k, lag in ((1, 2), (2, 4), (3, 6)):
        out.loc[order, f"own_time_prev{k}"] = g["time_spent"].shift(lag).values

    out.loc[order, "winpct_momentum_2"] = (
        df.loc[order, "winpct_before"].values
        - g["winpct_before"].shift(2).values)
    out.loc[order, "winpct_momentum_4"] = (
        df.loc[order, "winpct_before"].values
        - g["winpct_before"].shift(4).values)
    out.loc[order, "prev_own_move_was_blunder"] = (
        g["blunder"].shift(2).fillna(False).astype(int).values)

    out["cp_before"] = df.cp_before
    out["winpct_before"] = df.winpct_before
    out["abs_cp_before"] = df.cp_before.abs()
    out["eval_bucket"] = pd.cut(
        df.winpct_before, [-0.1, 20, 40, 60, 80, 100.1],
        labels=[0, 1, 2, 3, 4]).astype("float")
    return out


# ---------------------------------------------------------------------------
# Rating sweeps
# ---------------------------------------------------------------------------


RATING_FEATURES = ("mover_elo", "opp_elo", "mean_elo", "elo_gap")


def rating_grid(base, feat_names, ratings):
    """Replicate one feature row across a range of ratings.

    MOVES ALL FOUR RATING FEATURES TOGETHER. This is the whole point of the
    function and the reason it exists rather than being written inline.

    mean_elo = (mover_elo + opp_elo) / 2 and elo_gap = mover_elo - opp_elo, so
    the four are algebraically linked. Sweeping mover_elo alone and leaving the
    other three at the values of whichever position supplied the base row asks
    the model about a 2400 playing a 900 while their mean rating is 1650 --
    incoherent, absent from training data, and not the question a rating sweep
    is meant to answer.

    Setting opp_elo = mean_elo = R and elo_gap = 0 asks the question that was
    intended: a player of rating R against a peer.

    Any rating feature the model does not carry is skipped, so this is safe on
    reduced feature sets.
    """
    ratings = np.asarray(ratings, dtype=np.float32)
    grid = np.repeat(np.asarray(base, dtype=np.float32)[None, :],
                     len(ratings), axis=0)
    idx = {n: feat_names.index(n) for n in RATING_FEATURES if n in feat_names}
    if "mover_elo" not in idx:
        raise ValueError("model has no mover_elo feature; a rating sweep is "
                         "meaningless")
    for name in ("mover_elo", "opp_elo", "mean_elo"):
        if name in idx:
            grid[:, idx[name]] = ratings
    if "elo_gap" in idx:
        grid[:, idx["elo_gap"]] = 0.0
    return grid


# A rating sweep that moves all four features produces a curve spanning
# several-fold from the weakest to the strongest rating. A bare mover_elo sweep
# cannot: the other three rating features contradict it and the signal is
# diluted. Measured on 400 real positions with the engine_free no-clock model:
#
#     correct   median 6.28x   p10 3.26x   min 1.66x
#     broken    median 1.83x   p10 1.42x   min 1.12x
#
# So the gate is on the MEDIAN across many curves, not on each curve. A
# per-curve threshold of 2x would fire on genuinely flat positions, which
# exist -- the correct sweep's own minimum is 1.66x. The median separates the
# two regimes with a wide margin either side.
MIN_MEDIAN_SWEEP_SPAN = 3.0


def check_sweep_span(curves, where: str, threshold: float = MIN_MEDIAN_SWEEP_SPAN):
    """Fail loudly if a batch of rating curves is too flat to be a real sweep.

    This exists because the bare-mover_elo bug was documented as fixed while
    the code still had it, and nothing executed the fixed path to notice. A
    comment cannot catch that; an assertion on the output can.
    """
    c = np.asarray(curves, dtype=float)
    if c.ndim != 2 or c.shape[0] == 0:
        raise ValueError(f"{where}: expected a 2-D batch of curves")
    lo = np.maximum(c.min(axis=1), 1e-12)
    span = np.median(c.max(axis=1) / lo)
    if span < threshold:
        raise SystemExit(
            f"FATAL: {where} produced rating curves with median span "
            f"{span:.2f}x, below {threshold:.1f}x.\n"
            f"That is the signature of sweeping mover_elo alone while opp_elo, "
            f"mean_elo and\nelo_gap stay fixed. Use "
            f"build_features.rating_grid, which moves all four.")
    return float(span)


def self_test_rating_grid() -> int:
    """Check rating_grid moves the rating features together. No data needed."""
    names = ["material_balance", "mover_elo", "opp_elo", "elo_gap", "mean_elo"]
    base = np.array([3.0, 1234.0, 1600.0, -366.0, 1417.0], dtype=np.float32)
    ratings = [800, 1500, 2200]
    g = rating_grid(base, names, ratings)

    fail = 0

    def check(label, cond):
        nonlocal fail
        fail += not cond
        print(f"  {'OK  ' if cond else 'FAIL'}  {label}")

    for i, r in enumerate(ratings):
        check(f"mover_elo = {r}", g[i, 1] == r)
        check(f"opp_elo   = {r}  (not left at 1600)", g[i, 2] == r)
        check(f"mean_elo  = {r}  (not left at 1417)", g[i, 4] == r)
        check(f"elo_gap   = 0    (not left at -366)", g[i, 3] == 0)
    check("non-rating features untouched",
          bool((g[:, 0] == base[0]).all()))

    # reduced feature sets must not raise
    small = ["material_balance", "mover_elo"]
    gs = rating_grid(np.array([3.0, 1234.0], dtype=np.float32), small, ratings)
    check("works when only mover_elo is present",
          bool((gs[:, 1] == np.asarray(ratings, dtype=np.float32)).all()))

    try:
        rating_grid(np.array([3.0], dtype=np.float32), ["material_balance"],
                    ratings)
        check("raises when mover_elo is absent", False)
    except ValueError:
        check("raises when mover_elo is absent", True)

    # the span gate must reject a flat batch and accept a spread one
    flat = np.tile(np.array([0.05, 0.048, 0.046]), (20, 1))
    try:
        check_sweep_span(flat, "self-test")
        check("check_sweep_span rejects a flat batch", False)
    except SystemExit:
        check("check_sweep_span rejects a flat batch", True)
    wide = np.tile(np.array([0.15, 0.07, 0.03]), (20, 1))
    try:
        check_sweep_span(wide, "self-test")
        check("check_sweep_span accepts a real sweep", True)
    except SystemExit:
        check("check_sweep_span accepts a real sweep", False)

    print(f"\n{'rating_grid OK' if not fail else f'{fail} FAILED'}")
    return 1 if fail else 0


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true",
                    help="check rating_grid and the sweep-span gate, no data")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--chunk-size", type=int, default=20_000)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.self_test:
        return self_test_rating_grid()
    if not args.data or not args.out:
        ap.error("--data and --out are required (or use --self-test)")

    df = pd.read_parquet(args.data)
    if args.limit:
        df = df.head(args.limit)
    print(f"{len(df):,} rows in from {args.data}")

    fens = df.fen.tolist()
    chunks = [fens[i:i + args.chunk_size]
              for i in range(0, len(fens), args.chunk_size)]

    # Convert each chunk to a float32 frame IMMEDIATELY and drop the dicts.
    # Holding one dict per row costs ~2.8 KB against 200 B for the same row in
    # a float32 frame, i.e. 14x. At 9M rows that is 26 GB of dicts versus
    # 1.9 GB of frame, which is the difference between running and swapping.
    frames: list[pd.DataFrame] = []
    seen = 0

    def absorb(part):
        nonlocal seen
        seen += len(part)
        frames.append(pd.DataFrame(part).astype("float32"))

    if args.workers > 1:
        with Pool(args.workers) as pool:
            for i, part in enumerate(pool.imap(_worker, chunks), 1):
                absorb(part)
                print(f"  {i}/{len(chunks)} chunks  {seen:,} positions",
                      end="\r", file=sys.stderr)
    else:
        for i, c in enumerate(chunks, 1):
            absorb(_worker(c))
            print(f"  {i}/{len(chunks)} chunks  {seen:,} positions",
                  end="\r", file=sys.stderr)
    print(file=sys.stderr)

    pos = pd.concat(frames, ignore_index=True)
    del frames
    pos.index = df.index
    ctx = context_features(df)
    # float32 halves the frame and is far more precision than any of these
    # features carry. Keeps the parquet and the LightGBM Dataset small.
    for c in ctx.columns:
        if pd.api.types.is_float_dtype(ctx[c]):
            ctx[c] = ctx[c].astype("float32")
    feats = pd.concat([pos, ctx], axis=1)
    del pos, ctx

    engine_free = [c for c in feats.columns if c not in ENGINE_COLS]
    engine_assisted = list(feats.columns)

    # The assertion this script exists to guarantee.
    bad = sorted(set(engine_assisted) & set(LEAK_COLS))
    if bad:
        print(f"FATAL: leak columns present in feature set: {bad}", file=sys.stderr)
        return 1

    # `ply` is legitimately both a carry column and a feature; keep one copy.
    carry = [c for c in CARRY_COLS if c not in feats.columns]
    out = pd.concat([df[carry], feats], axis=1)
    out.to_parquet(args.out, index=False)

    print(f"\nfeatures: {len(engine_free)} engine-free, "
          f"{len(engine_assisted)} engine-assisted "
          f"(+{len(ENGINE_COLS)}: {', '.join(ENGINE_COLS)})")
    print(f"leak check: 0 of {len(LEAK_COLS)} leak columns in either set. OK")

    nulls = feats.isna().mean().sort_values(ascending=False)
    high = nulls[nulls > 0.005]
    if len(high):
        print("\ncolumns with >0.5% nulls (lags at game start are expected):")
        for k, v in high.items():
            print(f"  {k:<28} {v:.2%}")

    num = feats.select_dtypes("number")
    const = [c for c in num.columns if num[c].nunique(dropna=True) <= 1]
    if const:
        print(f"\nconstant columns, drop them at train time: {', '.join(const)}")

    valid = out[out.label_valid]
    if len(valid):
        corr = (num.drop(columns=const)
                .loc[valid.index]
                .corrwith(valid.blunder.astype(float))
                .abs().sort_values(ascending=False))
        print("\ntop 12 |correlation| with the label:")
        for k, v in corr.head(12).items():
            flag = "  <-- LEAK?" if v > 0.5 else ""
            print(f"  {k:<28} {v:.3f}{flag}")
        if (corr > 0.5).any():
            print("\nFATAL: correlation above 0.5 means a leak survived.",
                  file=sys.stderr)
            return 1

    print(f"\nwrote {args.out} ({len(out):,} rows x {len(out.columns)} cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
