#!/usr/bin/env python3
"""
Phase 9. External validation against Lichess puzzle ratings.

Every metric in this project so far says the model fits ITS OWN data: a test
split of the same dump, labelled by the same annotation, split by the same
code. That is necessary and it is not sufficient. This is the one check that
uses a difficulty scale measured somewhere else entirely.

Lichess publishes ~6M puzzles, each with a Glicko2 rating derived from real
human solve attempts. Nothing about that rating came from our dump, our label,
or our features. If the model's notion of "this position is dangerous" tracks
the humans' notion of "this puzzle is hard", the model has learned something
about human error rather than about this parquet.

THE MODEL'S DIFFICULTY SCALE. Predicted P(blunder) is not comparable to a
puzzle rating -- different units, different meaning. So convert the model onto
the rating scale instead:

    D10 := the rating at which predicted P(blunder) crosses 10%

A position where even a 2400 blunders 10% of the time is hard; one where only
a 900 does is easy. D10 is in Elo, so it can be correlated with the puzzle's
own Elo directly. Positions whose curve never crosses 10% anywhere in the
sweep are CENSORED and reported separately -- never silently dropped, because
they are systematically the easy ones and dropping them would bias the
correlation upward.

WHICH MODEL, AND WHY IT MATTERS HERE. A puzzle is a FEN. It has no clock, no
time control and no opponent, so an engine_free model still needs six clock
features imputed, and imputing them puts a made-up number in front of a
feature carrying ~9% of the model's gain.

The *_noclock model needs none of that: its only clock-like feature is
halfmove_clock, which is a FEN field (the fifty-move counter), not a wall
clock. So it is exactly computable on a puzzle. The clock ablation measured
the cost of that substitution at 1.4% of Brier skill, which is why this is a
cheap trade rather than a compromise. --model defaults accordingly, and the
script warns loudly if pointed at a model needing imputation.

THE CAVEAT THAT MUST BE PRINTED, not buried. Puzzles are SELECTED so that a
blunder is available and the correct move is unique. That is a biased sample
of positions -- but it is biased in a specific and useful direction: it is
close to the conditional "a mistake is on offer here" population, which is the
regime the Phase 7 selection term describes. So a positive correlation is
evidence about selection, not about availability, and the summary says so.

Usage:
    python validate_puzzles.py --puzzles data/lichess_db_puzzle.csv.zst \
        --model models/full_free_noclock --out figures/ --n 20000
"""

import argparse
import io
import json
import os
import sys

import chess
import lightgbm as lgb
import numpy as np
import pandas as pd

from build_features import check_sweep_span, position_features, rating_grid

# The sweep. Wide enough to bracket the whole Lichess range; 25-point steps
# because D10 is interpolated between grid points, so finer buys nothing.
RATINGS = np.arange(600, 2601, 25)
CROSS = 0.10                       # the "10%" in D10

# Mate puzzles carry forced-mate evals, which the win% label cannot represent
# (see SPEC.md: cp is clamped, mate scores are label_valid=False). Excluded
# for the same reason those rows are excluded from training.
MATE_THEMES = ("mate", "mateIn1", "mateIn2", "mateIn3", "mateIn4", "mateIn5",
               "smotheredMate", "backRankMate", "anastasiaMate", "arabianMate",
               "bodenMate", "doubleBishopMate", "dovetailMate", "hookMate",
               "killBoxMate", "vukovicMate")

THEME_GROUPS = ("fork", "pin", "hangingPiece", "deflection", "skewer",
                "discoveredAttack", "endgame", "middlegame", "opening")

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def open_maybe_zstd(path: str):
    """Read .zst directly. Same magic-byte sniffing as parse_lichess.py.

    pzstd writes a 12-byte SKIPPABLE frame ahead of the data frame, so a real
    Lichess archive starts 0x184D2A5* rather than the data-frame magic. Testing
    only the data magic is the bug that cost an evening on the main dump; do
    not reintroduce it here by assuming this file is plain CSV.
    """
    with open(path, "rb") as fh:
        head = fh.read(4)
    magic = int.from_bytes(head, "little") if len(head) == 4 else 0
    if head == ZSTD_MAGIC or 0x184D2A50 <= magic <= 0x184D2A5F:
        import zstandard as zstd
        fh = open(path, "rb")
        reader = zstd.ZstdDecompressor(max_window_size=2 ** 27).stream_reader(fh)
        return io.TextIOWrapper(reader, encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def spearman(a, b) -> float:
    """Pearson on ranks. Avoids a scipy import for one number, and ties are
    handled by pandas' average-rank default, which is what we want."""
    ra = pd.Series(a).rank()
    rb = pd.Series(b).rank()
    return float(ra.corr(rb))


def pearson(a, b) -> float:
    return float(pd.Series(a).corr(pd.Series(b)))


def solver_position(fen: str, moves: str):
    """The position actually shown to the solver.

    The CSV's FEN is BEFORE the opponent's move; the puzzle starts after the
    first move in Moves is played. Getting this wrong shifts every position by
    one ply and silently weakens the correlation, so it is asserted rather than
    assumed.
    """
    board = chess.Board(fen)
    first = moves.split()[0]
    board.push(chess.Move.from_uci(first))
    return board


def d10(curve: np.ndarray, ratings: np.ndarray) -> float:
    """Rating at which the curve crosses CROSS, linearly interpolated.

    The curve falls with rating, so we want the LAST point at or above CROSS
    and the first below it. Returns nan when it never crosses -- censored, and
    the caller must report those rather than dropping them.
    """
    above = curve >= CROSS
    if not above.any():
        return np.nan          # never dangerous enough, even for the weakest
    if above.all():
        return np.nan          # dangerous even for the strongest: censored high
    i = int(np.flatnonzero(above)[-1])
    if i + 1 >= len(curve):
        return np.nan
    y0, y1 = curve[i], curve[i + 1]
    if y0 == y1:
        return float(ratings[i])
    t = (y0 - CROSS) / (y0 - y1)
    return float(ratings[i] + t * (ratings[i + 1] - ratings[i]))


def load_iso(prefix: str):
    p = f"{prefix}_iso.npz"
    if not os.path.exists(p):
        return lambda v: v
    d = np.load(p)
    return lambda v: np.interp(v, d["x"], d["y"])


def build_rows(df: pd.DataFrame, feat_names: list[str], say):
    """FEN -> feature matrix. Returns (X, keep_index, n_failed)."""
    needed = set(feat_names)
    rows, keep = [], []
    failed = 0
    for r in df.itertuples():
        try:
            board = solver_position(r.FEN, r.Moves)
            f = position_features(board.fen())
        except Exception:
            failed += 1
            continue

        # Context the FEN can supply. move_number and ply are real; everything
        # else a game would provide (clock, time control, opponent) does not
        # exist for a puzzle, which is why the no-clock model is the default.
        f["mover_is_white"] = int(board.turn == chess.WHITE)
        f["move_number"] = board.fullmove_number
        f["ply"] = (board.fullmove_number - 1) * 2 + (0 if board.turn else 1)
        # placeholders; rating_grid overwrites all four on every sweep
        for n in ("mover_elo", "opp_elo", "mean_elo"):
            f[n] = 1500.0
        f["elo_gap"] = 0.0

        missing = needed - set(f)
        if missing:
            raise SystemExit(
                f"FATAL: cannot build {sorted(missing)} from a FEN.\n"
                f"That model needs game context a puzzle does not have. Use a "
                f"*_noclock model, or extend this function deliberately rather "
                f"than imputing.")
        rows.append([f[n] for n in feat_names])
        keep.append(r.Index)
    return np.asarray(rows, dtype=np.float32), keep, failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzles", required=True)
    ap.add_argument("--model", default="models/full_free_noclock")
    ap.add_argument("--out", default="figures/")
    ap.add_argument("--metrics-out", default="metrics/puzzle_validation.json")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rd", type=float, default=80.0,
                    help="keep puzzles whose rating is well measured")
    ap.add_argument("--min-plays", type=int, default=200)
    args = ap.parse_args()

    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    booster = lgb.Booster(model_file=f"{args.model}.txt")
    meta = json.load(open(f"{args.model}_meta.json"))
    feat_names = meta["features"]
    iso = load_iso(args.model)

    clockish = [f for f in feat_names
                if any(k in f.lower() for k in ("clk", "clock", "tc_", "time"))
                and f != "halfmove_clock"]
    if clockish:
        say(f"WARNING: {args.model} carries clock features {clockish}, which a")
        say("  puzzle cannot supply. Prefer a *_noclock model.")
        return 1

    say(f"model {os.path.basename(args.model)} "
        f"({meta['feature_set']}, {len(feat_names)} features)")

    # ----- load and filter -------------------------------------------------
    with open_maybe_zstd(args.puzzles) as fh:
        raw = pd.read_csv(fh)
    say(f"\n{len(raw):,} puzzles in file")

    df = raw.copy()
    df["Themes"] = df.Themes.fillna("")
    n0 = len(df)
    df = df[df.RatingDeviation < args.max_rd]
    n1 = len(df)
    df = df[df.NbPlays > args.min_plays]
    n2 = len(df)
    theme_set = set(MATE_THEMES)
    is_mate = df.Themes.str.split().apply(lambda t: bool(theme_set & set(t)))
    df = df[~is_mate]
    n3 = len(df)

    say(f"  RatingDeviation < {args.max_rd:g}   {n0:,} -> {n1:,}")
    say(f"  NbPlays > {args.min_plays}          {n1:,} -> {n2:,}")
    say(f"  drop mate themes             {n2:,} -> {n3:,}")
    say("  Mate puzzles are excluded because forced-mate evals are exactly")
    say("  what label_valid=False marks in training. Same rule, same reason.")
    if not len(df):
        say("FATAL: no puzzles survived filtering")
        return 1

    df = df.sample(min(args.n, len(df)), random_state=args.seed)
    df = df.reset_index(drop=True)
    say(f"\nsampled {len(df):,} puzzles  "
        f"(rating {df.Rating.min():.0f}-{df.Rating.max():.0f}, "
        f"median {df.Rating.median():.0f})")

    # ----- featurise and sweep --------------------------------------------
    say("\nbuilding features from FEN (no engine needed for this model) ...")
    X, keep, failed = build_rows(df, feat_names, say)
    df = df.loc[keep].reset_index(drop=True)
    if failed:
        say(f"  {failed:,} puzzles unparseable, dropped")
    say(f"  {len(df):,} feature rows")

    say(f"\nsweeping rating {RATINGS[0]}-{RATINGS[-1]} in {RATINGS[1]-RATINGS[0]}"
        f"-point steps, all four rating features moving together ...")
    d10s = np.empty(len(X))
    p1500 = np.empty(len(X))
    i1500 = int(np.argmin(np.abs(RATINGS - 1500)))
    # Batched: one predict call per CHUNK of puzzles rather than per puzzle.
    # Each puzzle contributes len(RATINGS) rows, so a chunk of 400 is ~32k
    # rows -- large enough to amortise the call overhead, small enough that
    # the stacked grid stays trivial in memory.
    CHUNK = 400
    span = None
    for s in range(0, len(X), CHUNK):
        block = X[s:s + CHUNK]
        grids = np.concatenate(
            [rating_grid(row, feat_names, RATINGS) for row in block])
        preds = iso(booster.predict(grids)).reshape(len(block), len(RATINGS))
        if span is None:
            # D10 is defined entirely by this curve, so a flattened sweep would
            # compress D10 and attenuate the correlation -- producing exactly
            # the weak-but-positive result this script is meant to measure.
            # Gate on the first chunk so a bad run dies in seconds, not after
            # the full sweep.
            span = check_sweep_span(preds, "puzzle rating sweep")
            say(f"  sweep span check: median {span:.2f}x across the rating "
                f"range (gate {check_sweep_span.__defaults__[0]:.1f}x)")
        for j in range(len(block)):
            d10s[s + j] = d10(preds[j], RATINGS)
            p1500[s + j] = preds[j][i1500]
        if s and s % (CHUNK * 10) == 0:
            print(f"  {s:,}/{len(X):,}", flush=True)
    df["d10"] = d10s
    df["p_at_1500"] = p1500

    # ----- censoring, reported before any correlation ----------------------
    cens = ~np.isfinite(d10s)
    n_cens = int(cens.sum())
    say(f"\ncensored (curve never crosses {CROSS:.0%} in range): "
        f"{n_cens:,} of {len(df):,} ({n_cens/len(df):.1%})")
    if n_cens:
        say(f"  their mean puzzle rating {df.Rating[cens].mean():.0f} against "
            f"{df.Rating[~cens].mean():.0f} for the uncensored")
        say("  Reported, not dropped: censored positions are systematically")
        say("  the easy ones, so discarding them would bias the correlation.")
    ok = df[~cens]
    if len(ok) < 100:
        say("FATAL: too few uncensored puzzles to correlate")
        return 1

    # ----- the headline ----------------------------------------------------
    rho = spearman(ok.d10, ok.Rating)
    r = pearson(ok.d10, ok.Rating)
    say("\n" + "=" * 70)
    say("HEADLINE: model difficulty (D10) vs human puzzle rating")
    say("=" * 70)
    say(f"  n = {len(ok):,} uncensored puzzles")
    say(f"  Spearman rho = {rho:+.4f}")
    say(f"  Pearson  r   = {r:+.4f}")
    say(f"  D10 range {ok.d10.min():.0f}-{ok.d10.max():.0f}, "
        f"median {ok.d10.median():.0f}")

    results = {"model": os.path.basename(args.model),
               "n_puzzles": int(len(ok)), "n_censored": n_cens,
               "spearman": rho, "pearson": r,
               "filters": {"max_rd": args.max_rd, "min_plays": args.min_plays,
                           "mate_themes_dropped": True}}

    # ----- by theme --------------------------------------------------------
    say("\nby theme (a global correlation can hide a single dominant theme):")
    rows, per_theme = [], {}
    for t in THEME_GROUPS:
        m = ok.Themes.str.split().apply(lambda x: t in x)
        sub = ok[m]
        if len(sub) < 200:
            continue
        rt = spearman(sub.d10, sub.Rating)
        per_theme[t] = {"n": int(len(sub)), "spearman": rt}
        rows.append([t, f"{len(sub):,}", f"{rt:+.4f}"])
    if rows:
        w = [max(len(r[i]) for r in rows + [["theme", "n", "spearman"]])
             for i in range(3)]
        say("  " + "  ".join(h.ljust(x) for h, x in
                             zip(["theme", "n", "spearman"], w)))
        say("  " + "  ".join("-" * x for x in w))
        for rr in rows:
            say("  " + "  ".join(c.ljust(x) for c, x in zip(rr, w)))
    results["by_theme"] = per_theme

    # ----- deciles ---------------------------------------------------------
    say("\nmean predicted P(blunder) at a fixed 1500 rating, by puzzle-rating")
    say("decile. Should rise monotonically if the model tracks difficulty:")
    ok = ok.assign(dec=pd.qcut(ok.Rating, 10, labels=False, duplicates="drop"))
    dec = ok.groupby("dec").agg(n=("Rating", "size"),
                                rating=("Rating", "mean"),
                                p=("p_at_1500", "mean"),
                                d10=("d10", "mean")).reset_index()
    say("")
    say(f"  {'decile':<7}{'n':>8}{'mean rating':>13}{'P@1500':>10}{'mean D10':>10}")
    for rr in dec.itertuples():
        say(f"  {rr.dec:<7}{rr.n:>8,}{rr.rating:>13.0f}"
            f"{rr.p:>10.2%}{rr.d10:>10.0f}")
    ups = int((dec.p.diff().dropna() > 0).sum())
    say(f"\n  monotone up-steps {ups} of {len(dec)-1}")
    results["deciles"] = dec.to_dict("records")
    results["decile_up_steps"] = ups

    # ----- figures ---------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    hb = ax1.hexbin(ok.Rating, ok.d10, gridsize=45, cmap="viridis",
                    mincnt=1, bins="log")
    fig.colorbar(hb, ax=ax1, label="puzzles (log)")
    b, a = np.polyfit(ok.Rating.to_numpy(dtype=float),
                      ok.d10.to_numpy(dtype=float), 1)
    xs = np.linspace(ok.Rating.min(), ok.Rating.max(), 50)
    ax1.plot(xs, a + b * xs, "r-", lw=2, label=f"fit, slope {b:.2f}")
    ax1.set_xlabel("Lichess puzzle rating (human solve attempts)")
    ax1.set_ylabel("model difficulty D10 (Elo at which P(blunder)=10%)")
    ax1.set_title("Model difficulty vs human difficulty")
    ax1.text(0.03, 0.95, f"Spearman $\\rho$ = {rho:+.3f}\nn = {len(ok):,}",
             transform=ax1.transAxes, va="top", fontsize=11,
             bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax1.legend(loc="lower right")

    ax2.plot(dec.rating, dec.p * 100, "o-", lw=2)
    ax2.set_xlabel("mean puzzle rating in decile")
    ax2.set_ylabel("mean predicted P(blunder) at 1500 (%)")
    ax2.set_title("Predicted danger rises with human-rated difficulty")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    dest = os.path.join(args.out, "puzzle_validation.png")
    fig.savefig(dest, dpi=140)
    say(f"\nwrote {dest}")

    # ----- the caveat, printed rather than buried --------------------------
    say("\n" + "=" * 70)
    say("CAVEAT, and why it does not sink the result")
    say("=" * 70)
    say("  Puzzles are SELECTED so a blunder exists and the best move is")
    say("  unique. They are therefore a biased sample of positions, and this")
    say("  is not a claim about all chess positions.")
    say("")
    say("  But the bias runs in a useful direction. Conditioning on 'a mistake")
    say("  is on offer here' is close to the regime the Phase 7 SELECTION term")
    say("  describes, and Phase 7 found selection carries ~100% of the rating")
    say("  effect while availability carries none. So this correlation is")
    say("  evidence about the selection channel specifically, which is exactly")
    say("  the channel that mattered.")
    say("")
    say("  It is still external: the puzzle rating comes from human solve")
    say("  attempts on lichess.org, not from this dump, this label or these")
    say("  features.")

    if args.metrics_out:
        os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
        with open(args.metrics_out, "w") as fh:
            json.dump(results, fh, indent=2, default=float)
        print(f"wrote {args.metrics_out}")
    report = os.path.join("reports", "puzzle_validation.txt")
    os.makedirs("reports", exist_ok=True)
    with open(report, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
