#!/usr/bin/env python3
"""
Validation report for a positions parquet produced by parse_lichess.py.

Reads only. Prints plain-text tables; writes nothing unless --out is given,
in which case the same report is teed to that path.

    python src/verify_labels.py --data data/positions_2026-06_blitz.parquet
    python src/verify_labels.py --data data/smoke.parquet --out report.txt
    python src/verify_labels.py --self-test        # no --data needed

Exit codes
----------
0   every gating check passed
1   a gating check failed, or the parquet is empty / missing columns

Gating checks are the ones that can only fail because of a parser bug:

    6   cp_before / cp_after / blunder null on a label_valid row
    2b  white and black blunder rates more than --colour-tol apart
    3c  a blunder recorded in a structurally impossible position

Everything else is reported and interpreted but does not gate, because it can
legitimately vary with the sample. Pass --strict to make the rating-decile
monotonicity verdict (check 2) gate as well.

Rating deciles are always computed WITHIN a tc_category: blitz and rapid are
separate Glicko2 pools and a 1500 in one is not a 1500 in the other.

Only the columns this report actually uses are read off disk. On a 6.7M-row
month that leaves `fen` and `move` in the file and saves a couple of GB.
"""

import argparse
import calendar
import contextlib
import io
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Label arithmetic, mirrored from SPEC.md
# ---------------------------------------------------------------------------

WIN_K = 0.00368208          # win% = 50 + 50 * (2 / (1 + exp(-k*cp)) - 1)
CP_CLAMP = 1000.0           # cp is clamped to +/-1000 before the transform

# ---------------------------------------------------------------------------
# Reference distribution for the selection-bias probe (check 5)
# ---------------------------------------------------------------------------

# Approximate marginal distribution of Lichess *rated game players* per pool,
# i.e. what you would see if analysis were not opt-in. These are eyeballed off
# the public rating-distribution page and are good to maybe +/- 50 Elo; they
# are a sanity yardstick, not ground truth. Replace with real numbers pulled
# from the ratings API before quoting any of this in writing.
LICHESS_REFERENCE = {
    "bullet": {"p10": 1050, "p25": 1290, "median": 1500, "p75": 1720, "p90": 1930},
    "blitz": {"p10": 1080, "p25": 1300, "median": 1500, "p75": 1710, "p90": 1900},
    "rapid": {"p10": 1090, "p25": 1300, "median": 1500, "p75": 1700, "p90": 1880},
    "classical": {"p10": 1120, "p25": 1320, "median": 1520, "p75": 1720, "p90": 1900},
}

# Overall result split across all rated games in a pool, white's point of view.
LICHESS_RESULT_REFERENCE = {
    "bullet": {"white_win": 0.500, "draw": 0.030, "black_win": 0.470},
    "blitz": {"white_win": 0.495, "draw": 0.045, "black_win": 0.460},
    "rapid": {"white_win": 0.490, "draw": 0.055, "black_win": 0.455},
    "classical": {"white_win": 0.485, "draw": 0.090, "black_win": 0.425},
}

DEFAULT_THRESHOLD = 20.0
THRESHOLDS = (15.0, 20.0, 25.0)
N_DECILES = 10
COLOUR_TOL = 1.20           # max white/black blunder-rate ratio, see check 2b

# Columns that must never reach a model. Kept here so the report can shout if
# a future parquet quietly drops or renames one of them.
LEAKY = [
    "cp_after", "mate_after", "winpct_after", "win_drop", "blunder",
    "label_valid", "clk_after", "time_spent", "result", "termination",
]

# Columns the report cannot run without.
REQUIRED = [
    "game_id", "mover", "mover_is_white", "mover_elo", "tc_category",
    "cp_before", "win_drop", "blunder", "label_valid",
]

# Columns the report uses if they are there. Anything not in REQUIRED or here
# is left on disk; `fen` alone is several hundred MB on a full month.
OPTIONAL = [
    "ply", "cp_after", "mate_before", "mate_after", "winpct_before",
    "clk_before", "clk_after", "time_spent", "tc_base", "result", "date",
]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_LINES: list[str] = []


def p(line: str = "") -> None:
    print(line)
    _LINES.append(line)


def rule(title: str = "") -> None:
    p()
    p("=" * 78)
    if title:
        p(title)
        p("=" * 78)


def loud(msg: str) -> None:
    """Impossible-to-skim-past warning block."""
    p()
    p("!" * 78)
    for line in msg.splitlines():
        p("!! " + line)
    p("!" * 78)
    p()


def table(headers: list[str], rows: list[list], aligns: str | None = None) -> None:
    """Minimal fixed-width table. aligns is a string of 'l'/'r' per column."""
    body = [[("" if c is None else str(c)) for c in r] for r in rows]
    widths = [len(h) for h in headers]
    for r in body:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    if aligns is None:
        aligns = "l" + "r" * (len(headers) - 1)

    def fmt(cells):
        out = []
        for i, c in enumerate(cells):
            out.append(c.ljust(widths[i]) if aligns[i] == "l" else c.rjust(widths[i]))
        return "  ".join(out).rstrip()

    p(fmt(headers))
    p("  ".join("-" * w for w in widths))
    for r in body:
        p(fmt(r))


def pct(x: float, nd: int = 2) -> str:
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100*x:.{nd}f}%"


# ---------------------------------------------------------------------------
# Shared derivations
# ---------------------------------------------------------------------------


def winpct(cp):
    """SPEC.md's win% transform, with the same +/-1000 clamp."""
    c = np.clip(np.asarray(cp, dtype=float), -CP_CLAMP, CP_CLAMP)
    return 50.0 + 50.0 * (2.0 / (1.0 + np.exp(-WIN_K * c)) - 1.0)


def floor_cp(threshold: float) -> float:
    """Largest cp_before at which a `threshold`-point win% drop is impossible.

    win% bottoms out at winpct(-1000) = 2.46 because of the clamp, so a drop of
    `threshold` cannot be recorded at all once winpct_before <= 2.46 + t. This
    is a definitional boundary, not a property of chess, and everything
    downstream that compares engine_free to engine_assisted has to know about
    it.
    """
    target = winpct(-CP_CLAMP) + threshold
    s = (target - 50.0) / 50.0
    return float(-np.log(2.0 / (s + 1.0) - 1.0) / WIN_K)


def decile_frame(valid: pd.DataFrame, pool: str) -> pd.DataFrame | None:
    """Rating deciles within one pool. Returns None if the pool is too thin."""
    sub = valid[valid.tc_category == pool]
    if len(sub) < N_DECILES * 10:
        return None
    try:
        bins = pd.qcut(sub.mover_elo, N_DECILES, labels=False, duplicates="drop")
    except ValueError:
        return None
    return sub.assign(decile=bins)


def decile_rates(sub: pd.DataFrame, label: pd.Series) -> pd.DataFrame:
    """Per-decile n / rate / Elo range for an arbitrary boolean label."""
    g = pd.DataFrame({
        "decile": sub["decile"].to_numpy(),
        "elo": sub["mover_elo"].to_numpy(),
        "y": np.asarray(label).astype(bool),
    }).groupby("decile")
    out = g.agg(n=("y", "size"), k=("y", "sum"),
                elo_lo=("elo", "min"), elo_hi=("elo", "max"))
    out["rate"] = out.k / out.n
    out["se"] = np.sqrt(out.rate * (1 - out.rate) / out.n)
    return out.reset_index()


def monotonicity(rates: pd.DataFrame) -> tuple[list[tuple[int, float]], list[int]]:
    """Deciles where the rate RISES with rating. Returns (all, significant).

    'Significant' means the rise exceeds 2 standard errors of the difference,
    i.e. it is not just small-sample noise in a 2-5% base rate.
    """
    r = rates.rate.to_numpy()
    se = rates.se.to_numpy()
    viol, sig = [], []
    for i in range(len(r) - 1):
        d = r[i + 1] - r[i]
        if d > 0:
            viol.append((i, d))
            if d > 2 * np.sqrt(se[i] ** 2 + se[i + 1] ** 2):
                sig.append(i)
    return viol, sig


# ---------------------------------------------------------------------------
# Check 1 - counts and base rate
# ---------------------------------------------------------------------------


def check_counts(df: pd.DataFrame, valid: pd.DataFrame, schema: list[str]) -> None:
    rule("1. COUNTS AND BASE RATE")

    n_games = df.game_id.nunique(dropna=False)
    n_players = df.mover.nunique(dropna=False)
    n_sides = df.groupby(["game_id", "mover_is_white"], dropna=False).ngroups

    table(
        ["quantity", "value"],
        [
            ["rows (plies)", f"{len(df):,}"],
            ["games", f"{n_games:,}"],
            ["player-games (sides)", f"{n_sides:,}"],
            ["distinct players", f"{n_players:,}"],
            ["rows / game", f"{len(df)/max(n_games,1):.1f}"],
            ["games / player", f"{n_sides/max(n_players,1):.2f}"],
            ["label_valid rows", f"{len(valid):,}"],
            ["label_valid=False rows", f"{len(df)-len(valid):,}"],
            ["label_valid=False fraction", pct(1 - len(valid) / max(len(df), 1))],
            ["blunders (valid rows)", f"{int(valid.blunder.sum()):,}"],
            ["BASE RATE (valid rows)", pct(valid.blunder.mean(), 3)],
        ],
    )

    br = valid.blunder.mean()
    if not (0.02 <= br <= 0.05):
        loud(f"Base rate {pct(br,3)} is outside the 2-5% band SPEC.md expects.\n"
             "Either the threshold is not 20 win% or the eval source changed.")

    # The stored label must be exactly win_drop > 20. If parse_lichess.py was
    # run with --threshold something else, everything downstream is off.
    recomputed = valid.win_drop > DEFAULT_THRESHOLD
    disagree = int((recomputed != valid.blunder).sum())
    p()
    p(f"stored `blunder` vs (win_drop > {DEFAULT_THRESHOLD:g}): "
      f"{len(valid)-disagree:,} / {len(valid):,} agree, {disagree:,} disagree")
    if disagree:
        loud(f"{disagree:,} rows disagree with the documented threshold of "
             f"{DEFAULT_THRESHOLD:g}.\nThis parquet was probably parsed with a "
             "different --threshold.")

    # Presence is checked against the parquet SCHEMA, not the loaded frame:
    # this report deliberately leaves most of these columns on disk.
    missing = [c for c in LEAKY if c not in schema]
    present = [c for c in LEAKY if c in schema]
    p()
    p(f"leak-listed columns present in this parquet ({len(present)}/{len(LEAKY)}): "
      + ", ".join(present))
    if missing:
        p(f"  absent: {', '.join(missing)}")
    p("  (label-side columns. features_*.parquet legitimately carries some of")
    p("   them - `blunder` IS the target and `label_valid` IS the row mask. What")
    p("   must never happen is one of them entering a FEATURE SET, which is what")
    p("   LEAK_COLS in train.py and build_features.py guard.)")


# ---------------------------------------------------------------------------
# Check 2 - base rate by tc_category and rating decile
# ---------------------------------------------------------------------------


def check_breakdowns(valid: pd.DataFrame) -> list[str]:
    rule("2. BASE RATE BY TIME CONTROL AND BY RATING DECILE")

    rows = []
    for tc, sub in valid.groupby("tc_category", dropna=False):
        rows.append([
            tc, len(sub), f"{sub.game_id.nunique():,}",
            f"{int(sub.blunder.sum()):,}", pct(sub.blunder.mean(), 3),
            f"{sub.mover_elo.median():.0f}",
        ])
    rows.sort(key=lambda r: -r[1])
    for r in rows:
        r[1] = f"{r[1]:,}"
    p("by tc_category (each is a separate rating pool):")
    p()
    table(["tc_category", "rows", "games", "blunders", "base rate", "med elo"], rows)

    failed_pools = []
    for pool in sorted(valid.tc_category.dropna().unique()):
        sub = decile_frame(valid, pool)
        p()
        p(f"rating deciles within pool = {pool}  (deciles are pool-local)")
        p()
        if sub is None:
            p("  too few rows to decile.")
            continue

        rates = decile_rates(sub, sub.blunder)
        table(
            ["decile", "elo range", "rows", "blunders", "base rate", "+/- 1se"],
            [[int(r.decile) + 1, f"{int(r.elo_lo)}-{int(r.elo_hi)}",
              f"{int(r.n):,}", f"{int(r.k):,}", pct(r.rate, 3), pct(r.se, 3)]
             for r in rates.itertuples()],
        )

        viol, sig = monotonicity(rates)
        first, last = rates.rate.iloc[0], rates.rate.iloc[-1]
        ratio = f"  (ratio {first/last:.2f}x)" if last > 0 else ""
        p()
        p(f"  D1 -> D{len(rates)}: {pct(first,3)} -> {pct(last,3)}{ratio}")

        if not viol:
            p("  MONOTONIC: blunder rate falls at every decile step. OK.")
        elif not sig:
            p(f"  {len(viol)} non-monotonic step(s), all within 2se of noise: "
              + ", ".join(f"D{i+1}->D{i+2} +{pct(d,3)}" for i, d in viol))
            p("  Treat as sampling noise, not a data problem.")
        else:
            failed_pools.append(pool)
            loud(
                f"NON-MONOTONIC RATING CURVE in pool '{pool}'.\n"
                + "\n".join(
                    f"  D{i+1} -> D{i+2}: rate RISES by {pct(d,3)}"
                    + ("   <-- exceeds 2se, not noise" if i in sig else "")
                    for i, d in viol
                )
                + "\n\nStronger players should blunder less at every step. A real "
                  "rise means\nthe pool is contaminated (mixed time controls?), "
                  "mover_elo is wrong, or\nthe annotated subset is selected "
                  "differently at different ratings."
            )
    return failed_pools


# ---------------------------------------------------------------------------
# Check 2b - colour symmetry
# ---------------------------------------------------------------------------


def check_colour_symmetry(valid: pd.DataFrame, tol: float = COLOUR_TOL) -> bool:
    """White and Black must blunder at similar RATES.

    win_drop is computed from White-POV evals flipped to the mover. If that
    flip were wrong for Black, Black's label would become "the eval moved 20+
    points in Black's favour during Black's own move", which happens in well
    under 0.1% of rows. Black would read ~0.01% against White's ~4%, a gap of
    several hundred x. That is what this check exists to catch.

    IMPORTANT: gate on the RATIO, not on standard errors. An earlier version
    used a 6-se threshold and fired at 6.3M rows on a 2.8% relative difference
    that had been present, and harmless, at 1.5M rows too. At large n any real
    effect is significant, so significance is not evidence of a bug. The
    effect size is. See --self-test for the fixture that pins both directions.
    """
    rule("2b. COLOUR SYMMETRY (sign-flip check)")
    rows, rates = [], {}
    for is_white, name in ((True, "white"), (False, "black")):
        sub = valid[valid.mover_is_white == is_white]
        if not len(sub):
            continue
        r = sub.blunder.mean()
        se = np.sqrt(r * (1 - r) / len(sub))
        rates[name] = (r, se, len(sub))
        rows.append([name, f"{len(sub):,}", f"{int(sub.blunder.sum()):,}",
                     pct(r), f"{se*100:.3f}%", f"{sub.mover_elo.median():.0f}"])
    table(["mover", "rows", "blunders", "base rate", "1se", "med elo"], rows)

    if len(rates) != 2:
        p()
        p("  only one colour present, nothing to compare. Skipped.")
        return True

    (rw, sw, _), (rb, sb, _) = rates["white"], rates["black"]
    ratio = max(rw, rb) / max(min(rw, rb), 1e-12)
    se = np.sqrt(sw ** 2 + sb ** 2)
    z = (rw - rb) / se if se else 0.0
    p()
    p(f"white - black = {(rw-rb)*100:+.3f}pp | ratio {ratio:.3f}x | "
      f"{abs(z):.1f} standard errors")
    p("the RATIO is the test, tolerance "
      f"{tol:.2f}x. Standard errors are shown for")
    p("information only: at several million rows any real effect is significant,")
    p("so significance on its own is not evidence of a bug.")

    # The direct smoking gun. Under a broken flip this column would be huge
    # for one colour and tiny for the other.
    t = DEFAULT_THRESHOLD
    p()
    neg = []
    for is_white, name in ((True, "white"), (False, "black")):
        sub = valid[valid.mover_is_white == is_white]
        neg.append([name, pct((sub.win_drop < -t).mean(), 3),
                    pct((sub.win_drop > t).mean(), 3)])
    table(["mover", f"win_drop < -{t:g}", f"win_drop > {t:g}"], neg)
    p("under a broken sign flip these two columns would be swapped for one")
    p("colour, so a colour showing ~0.0% positive and a few percent negative is")
    p("the unambiguous signature.")

    if ratio > tol:
        loud(f"COLOUR ASYMMETRY: {ratio:.2f}x apart, past the {tol:.2f}x "
             f"tolerance.\nThat is far beyond any plausible chess effect. "
             f"Suspect the mover-POV\nsign flip in parse_lichess.py.")
        return False

    p()
    p(f"  {ratio:.3f}x apart, within the {tol:.2f}x tolerance. The sign flip is "
      f"behaving. OK")
    if ratio > 1.05:
        p("  Larger than expected for a pure chess effect, so worth a look, but")
        p("  not a sign-flip failure.")
    else:
        p("  A small White excess is expected: White scores better overall, so")
        p("  White sits above the structural floor more often (check 3c) and")
        p("  simply has more positions in which a blunder can be recorded.")
    return True


# ---------------------------------------------------------------------------
# Checks 2c / 2d - coverage and clocks
# ---------------------------------------------------------------------------


def _expected_days(dates: pd.Series) -> tuple[int, str]:
    """Calendar days in the months the data actually touches.

    The old version hardcoded 30, which quietly mis-scores February and any
    parquet spanning more than one month.
    """
    months = sorted({d[:7] for d in dates if isinstance(d, str) and len(d) >= 7})
    if not months:
        return 30, "assumed 30-day month (dates unparseable)"
    total = 0
    for m in months:
        try:
            y, mo = int(m[:4]), int(m[5:7])
            total += calendar.monthrange(y, mo)[1]
        except ValueError:
            total += 30
    return total, ", ".join(months)


def check_coverage(df: pd.DataFrame) -> None:
    """The dump is chronological and streamed, so --max-games truncates in TIME.

    A run that stops early has not sampled the month, it has sampled the
    beginning of the month, and weekday and weekend populations differ.
    """
    rule("2c. CALENDAR COVERAGE (front-loading check)")
    if "date" not in df.columns or df.date.isna().all():
        p("no date column, skipping")
        return
    days = df.date.dropna()
    uniq = sorted(days.unique())
    span = len(uniq)
    expected, months = _expected_days(uniq)
    counts = days.value_counts().sort_index()
    table(["quantity", "value"],
          [["distinct days present", f"{span}"],
           ["first day", uniq[0]], ["last day", uniq[-1]],
           ["months touched", months],
           ["calendar days in those months", f"{expected}"],
           ["coverage", f"{span/expected:.0%}"],
           ["busiest day share", pct(counts.max() / len(days))],
           ["uniform share would be", pct(1 / span)]])
    p()
    if span < 0.8 * expected:
        loud(f"FRONT-LOADED: only {span} distinct days of {expected}. --max-games "
             f"stopped the stream early.\nRaise --every so the same game count "
             f"spreads over the whole file, or state\nthe window as a limitation "
             f"and do not call this dataset a full month.")
        p("  Before quoting this as a defect, check WHICH days: a prefix that")
        p("  happens to land on a whole Monday-to-Sunday week is a clean sample")
        p("  of a week, and this banner only counts days.")
    else:
        p("  spans the month. OK")


def check_clocks(df: pd.DataFrame) -> None:
    rule("2d. CLOCK AVAILABILITY")
    rows = []
    for c in ("clk_before", "clk_after", "time_spent"):
        if c in df.columns:
            rows.append([c, f"{int(df[c].isna().sum()):,}", pct(df[c].isna().mean())])
    if not rows:
        p("no clock columns, skipping")
        return
    table(["column", "null rows", "null share"], rows)
    if "time_spent" in df.columns:
        neg = int((df.time_spent < 0).sum())
        p()
        p(f"negative time_spent rows: {neg:,} ({pct(neg/max(len(df),1), 3)})")
        if neg:
            p("  Lichess lag compensation gives time back, so the clock can")
            p("  legitimately rise by more than the increment. A handful of these")
            p("  is expected; a few percent means clock parsing drifted.")
    if "clk_before" in df.columns and "tc_base" in df.columns:
        frac = df.clk_before / df.tc_base.replace(0, np.nan)
        p(f"clk_before / tc_base: p50 {frac.median():.2f}, "
          f"p99 {frac.quantile(0.99):.2f} (values >1 are increment accrual, "
          f"not an error)")


# ---------------------------------------------------------------------------
# Check 3 - win_drop distribution and threshold-adjacent mass
# ---------------------------------------------------------------------------


def check_win_drop(valid: pd.DataFrame) -> None:
    rule("3. WIN_DROP DISTRIBUTION (label_valid rows only)")

    d = valid.win_drop
    table(
        ["stat", "win% drop"],
        [["min", f"{d.min():.2f}"], ["p50", f"{d.median():.2f}"],
         ["p90", f"{d.quantile(0.90):.2f}"], ["p99", f"{d.quantile(0.99):.2f}"],
         ["p99.9", f"{d.quantile(0.999):.2f}"], ["max", f"{d.max():.2f}"],
         ["mean", f"{d.mean():.2f}"], ["negative (move helped)", pct((d < 0).mean())]],
    )

    edges = [-np.inf, -10, -5, -1, 0, 1, 5, 10, 15, 20, 25, 30, 40, 50, 75, np.inf]
    cut = pd.cut(d, edges, right=False)
    counts = cut.value_counts().sort_index()
    p()
    p("histogram:")
    p()
    total = len(d)
    rows = []
    for interval, n in counts.items():
        lo = "-inf" if interval.left == -np.inf else f"{interval.left:g}"
        hi = "+inf" if interval.right == np.inf else f"{interval.right:g}"
        bar = "#" * int(round(60 * n / max(counts.max(), 1)))
        rows.append([f"[{lo}, {hi})", f"{n:,}", pct(n / total), bar])
    table(["bin", "rows", "share", ""], rows, aligns="lrrl")

    p()
    p("A fifth of moves having a NEGATIVE win_drop is not a fifth of the evals")
    p("being broken: the mass sits in [-1, 0), which is annotation jitter at the")
    p("depth Lichess analyses at. What would be alarming is mass below -5.")

    # Split the band exactly where the label does: <= 20 is a negative,
    # > 20 is a positive.
    n_near = int(d.between(15, 25, inclusive="both").sum())
    n_pos = int((d > DEFAULT_THRESHOLD).sum())
    below = int(d.between(15, DEFAULT_THRESHOLD, inclusive="both").sum())
    above = int(((d > DEFAULT_THRESHOLD) & (d <= 25)).sum())

    rule("3b. THRESHOLD-ADJACENT MASS (label noise budget)")
    table(
        ["quantity", "rows", "of all valid", "vs positive count"],
        [
            ["win_drop in [15, 25]", f"{n_near:,}", pct(n_near / total), pct(n_near / max(n_pos, 1))],
            [f"  in [15, {DEFAULT_THRESHOLD:g}] - near-miss negatives", f"{below:,}",
             pct(below / total), pct(below / max(n_pos, 1))],
            [f"  in ({DEFAULT_THRESHOLD:g}, 25] - marginal positives", f"{above:,}",
             pct(above / total), pct(above / max(n_pos, 1))],
            ["win_drop > 20 (all positives)", f"{n_pos:,}", pct(n_pos / total), "100.00%"],
        ],
    )
    p()
    p(f"{pct(above/max(n_pos,1))} of positives sit within 5 win% of the cut, and "
      f"{below:,} negatives sit\njust under it. Depth-dependent eval jitter in the "
      "Lichess annotations moves rows\nacross the line in BOTH directions, so the "
      "irreducible label noise is roughly\nthis band. It caps how well ANY model can "
      "score and attenuates measured\neffects; it does not by itself make a smaller "
      "gain fake. The test that\nmatters is whether model ranking is stable at "
      "thresholds 15, 20 and 25.")


# ---------------------------------------------------------------------------
# Check 3c - the structural floor
# ---------------------------------------------------------------------------


def check_structural_floor(valid: pd.DataFrame) -> bool:
    """The label is arithmetically incapable of firing in lost positions.

    cp is clamped to +/-1000 so win% bottoms out at 2.46. A 20-point drop is
    therefore impossible once winpct_before <= 22.46, i.e. cp_before < -337.
    Those rows must contain exactly zero blunders. If any of them is labelled a
    blunder, winpct_before and the label were computed from different numbers
    and the parquet is internally inconsistent.

    This also sizes the `eligible` subset that evaluate.py re-scores on, so the
    engine_free vs engine_assisted comparison can be reported without the
    structural-floor freebie.
    """
    rule("3c. STRUCTURAL FLOOR (label cannot fire in lost positions)")

    wmin = float(winpct(-CP_CLAMP))
    p(f"cp clamped to +/-{CP_CLAMP:g} => win% is clamped to "
      f"[{wmin:.2f}, {100-wmin:.2f}]")
    p(f"so a t-point drop needs winpct_before > {wmin:.2f} + t")
    p()
    p("POV matters here: cp_before is WHITE-POV as Lichess writes it, while")
    p("winpct_before and win_drop are MOVER-POV. The floor is a statement about")
    p("the mover, so the cp cut-offs below are mover-POV too (negate them to")
    p("read them off cp_before for a Black move).")
    p()

    mover_cp = np.where(valid.mover_is_white.to_numpy(),
                        valid.cp_before.to_numpy(), -valid.cp_before.to_numpy())
    if "winpct_before" in valid.columns:
        w = valid.winpct_before.to_numpy()
        p("  winpct_before source: winpct_before column (mover-POV)")
        # Cross-check the stored column against SPEC.md's formula applied to
        # the mover-POV cp. Drift means the parser and the spec have diverged.
        max_dev = float(np.nanmax(np.abs(winpct(mover_cp) - w))) if len(w) else 0.0
        p(f"  max |stored - recomputed from cp_before| win%: {max_dev:.4f} "
          f"({'OK' if max_dev < 0.01 else 'DRIFT - parser and SPEC.md disagree'})")
        if max_dev >= 0.01:
            loud("winpct_before does not match the SPEC.md formula applied to "
                 "the\nmover-POV cp_before. Either the constant, the clamp or "
                 "the sign flip\nchanged in parse_lichess.py.")
    else:
        w = winpct(mover_cp)
        p("  winpct_before source: recomputed from cp_before (mover-POV)")

    rows, ok = [], True
    for t in THRESHOLDS:
        # The positive class has to be the one THIS threshold defines. Testing
        # the stored t=20 label against the t=25 impossibility region compares
        # two different questions and reports phantom violations.
        y = (valid.win_drop > t).to_numpy()
        impossible = w <= wmin + t
        n_imp = int(impossible.sum())
        n_bad = int((y & impossible).sum())
        if n_bad:
            ok = False
        elig = ~impossible
        n_elig = int(elig.sum())
        rows.append([
            f"{t:g}", f"mover cp < {floor_cp(t):.0f}", f"{n_imp:,}",
            pct(n_imp / max(len(valid), 1)), f"{n_bad:,}",
            pct(float(y[elig].mean()) if n_elig else float("nan"), 3),
        ])
    table(
        ["threshold", "impossible when", "impossible rows", "share",
         "positives there", "eligible base rate"],
        rows,
    )

    p()
    p("`positives there` is (win_drop > t) counted inside the region where a")
    p("t-point drop is arithmetically impossible, so it MUST be 0 on every row.")
    p("A non-zero count is a parser inconsistency, not a chess fact.")
    p()
    p("Two consequences worth carrying into the README:")
    p("  1. The label is blind to blunders in already-lost positions. Dropping a")
    p("     rook while down a queen does not register. That is a deliberate")
    p("     consequence of using win probability rather than centipawn loss.")
    p("  2. engine_assisted gets winpct_before, so part of its advantage is just")
    p("     learning 'below the floor, predict zero'. Always report that")
    p("     comparison ALSO on the eligible subset, where the free win vanishes.")

    if not ok:
        loud("STRUCTURAL FLOOR VIOLATED: a blunder is labelled in a position "
             "where a\n20-point win% drop is arithmetically impossible. "
             "winpct_before and the\nlabel were computed from different numbers.")
    return ok


# ---------------------------------------------------------------------------
# Check 4 - threshold robustness
# ---------------------------------------------------------------------------


def check_thresholds(valid: pd.DataFrame) -> None:
    rule("4. THRESHOLD ROBUSTNESS "
         f"({' / '.join(f'{t:g}' for t in THRESHOLDS)} win% drop)")

    raw = {}
    for t in THRESHOLDS:
        y = valid.win_drop > t
        raw[t] = (int(y.sum()), float(y.mean()))
    base20 = raw[DEFAULT_THRESHOLD][1]
    p("base rate on all valid rows:")
    p()
    table(
        ["threshold", "positives", "base rate", f"vs t={DEFAULT_THRESHOLD:g}"],
        [[f"{t:g}", f"{raw[t][0]:,}", pct(raw[t][1], 3),
          f"{raw[t][1]/base20:.2f}x" if base20 else "-"] for t in THRESHOLDS],
    )

    t_lo, t_hi = THRESHOLDS[0], THRESHOLDS[-1]
    for pool in sorted(valid.tc_category.dropna().unique()):
        sub = decile_frame(valid, pool)
        if sub is None:
            continue
        p()
        p(f"decile base rate by threshold, pool = {pool}:")
        p()
        per_t = {t: decile_rates(sub, sub.win_drop > t) for t in THRESHOLDS}
        ref = per_t[DEFAULT_THRESHOLD]
        table(
            ["decile", "elo range"] + [f"t={t:g}" for t in THRESHOLDS]
            + [f"t{t_lo:g}/t{t_hi:g}"],
            [[i + 1, f"{int(ref.elo_lo[i])}-{int(ref.elo_hi[i])}"]
             + [pct(per_t[t].rate[i], 3) for t in THRESHOLDS]
             + [f"{per_t[t_lo].rate[i]/per_t[t_hi].rate[i]:.2f}x"
                if per_t[t_hi].rate[i] > 0 else "-"]
             for i in range(len(ref))],
        )

        summary = []
        for t in THRESHOLDS:
            rates = per_t[t]
            viol, sig = monotonicity(rates)
            first, last = rates.rate.iloc[0], rates.rate.iloc[-1]
            summary.append([
                f"{t:g}", pct(rates.k.sum() / rates.n.sum(), 3),
                f"{first/last:.2f}x" if last > 0 else "-",
                len(viol), len(sig),
                "yes" if not viol else ("noise only" if not sig else "NO"),
            ])
        p()
        table(
            ["threshold", "base rate", "D1/D10 ratio", "up-steps", "signif.", "monotonic"],
            summary,
        )

    p()
    p("Read this as: if the D1/D10 ratio and the monotonicity verdict hold across")
    p("all three thresholds, the rating effect is a property of the players, not an")
    p("artefact of where the cut was placed. If they flip, the label is doing the")
    p("work and the threshold choice needs justifying.")


# ---------------------------------------------------------------------------
# Check 5 - selection bias probe
# ---------------------------------------------------------------------------


def check_selection_bias(df: pd.DataFrame) -> None:
    rule("5. SELECTION BIAS PROBE (annotated subset vs all of Lichess)")

    if "result" not in df.columns:
        p("no result column, skipping the result-mix half of this check")

    # One row per player-game, so long games stop out-voting short ones.
    agg = {"mover_elo": ("mover_elo", "first"), "tc_category": ("tc_category", "first")}
    if "result" in df.columns:
        agg["result"] = ("result", "first")
    sides = df.groupby(["game_id", "mover_is_white"], dropna=False).agg(**agg).reset_index()

    p(f"unit of analysis: player-game ({len(sides):,} sides, "
      f"{sides.game_id.nunique():,} games)")
    p("row-level Elo is game-length-weighted and would overstate whoever plays long")
    p("games, so the percentiles below are computed per player-game.")

    for pool in sorted(sides.tc_category.dropna().unique()):
        ref = LICHESS_REFERENCE.get(pool)
        sub = sides[sides.tc_category == pool]
        p()
        p(f"pool = {pool}  ({len(sub):,} player-games)")
        p()
        if ref is None:
            p(f"  no reference distribution hardcoded for '{pool}'; skipping.")
            continue
        qs = {"p10": 0.10, "p25": 0.25, "median": 0.50, "p75": 0.75, "p90": 0.90}
        table(
            ["percentile", "annotated", "lichess ref", "delta"],
            [[k, f"{sub.mover_elo.quantile(q):.0f}", f"{ref[k]}",
              f"{sub.mover_elo.quantile(q) - ref[k]:+.0f}"] for k, q in qs.items()],
        )
        med_delta = sub.mover_elo.median() - ref["median"]
        iqr = sub.mover_elo.quantile(0.75) - sub.mover_elo.quantile(0.25)
        ref_iqr = ref["p75"] - ref["p25"]
        p(f"  median shift {med_delta:+.0f} Elo | "
          f"IQR {iqr:.0f} vs ref {ref_iqr} ({iqr/ref_iqr:.2f}x)")

        notes = []
        if abs(med_delta) >= 75:
            notes.append(
                f"median Elo is {abs(med_delta):.0f} points "
                f"{'ABOVE' if med_delta > 0 else 'BELOW'} the pool as a whole - the "
                "annotated\n  subset is not a random sample of players")
        elif abs(med_delta) >= 25:
            notes.append(f"median Elo is {med_delta:+.0f} vs the pool: mild skew")
        else:
            notes.append("median Elo is within 25 points of the pool: no obvious "
                         "rating skew")

        # Result mix. Note this is white-POV: at the player-game level the
        # win and loss rates are symmetric BY CONSTRUCTION (every decisive
        # game contributes one winner and one loser), so the informative
        # numbers are the DRAW rate and the white-win rate.
        rref = LICHESS_RESULT_REFERENCE.get(pool, {})
        if "result" in sub.columns:
            games = sub.drop_duplicates("game_id")
            obs = {
                "white_win": (games.result == "1-0").mean(),
                "draw": (games.result == "1/2-1/2").mean(),
                "black_win": (games.result == "0-1").mean(),
            }
            p()
            table(
                ["result (white POV, per game)", "annotated", "lichess ref", "delta"],
                [[k, pct(v, 2), pct(rref[k], 2) if k in rref else "-",
                  f"{100*(v-rref[k]):+.2f}pp" if k in rref else "-"]
                 for k, v in obs.items()],
            )
            if "draw" in rref:
                dd = obs["draw"] - rref["draw"]
                if dd <= -0.01:
                    notes.append(
                        f"draw rate is {abs(100*dd):.1f}pp LOW ({pct(obs['draw'])} vs "
                        f"{pct(rref['draw'])}). Decisive games are\n  over-represented, "
                        "which is what opt-in analysis looks like: people request\n  it "
                        "after a loss, not after a draw")
                elif dd >= 0.01:
                    notes.append(f"draw rate is {100*dd:+.1f}pp HIGH vs the pool - "
                                 "unexpected, worth a look")
                else:
                    notes.append("draw rate matches the pool")
            if abs(obs["white_win"] - rref.get("white_win", obs["white_win"])) > 0.03:
                notes.append("white win rate is off by >3pp, which no selection story "
                             "explains - suspect a\n  parsing bug in `result`")

        p()
        p("  note:")
        for n in notes:
            p("  - " + n.replace("\n  ", "\n    "))

    p()
    p("Caveat: analysis is requested per GAME and both players' moves then enter")
    p("the data, so a loser-driven request pulls in the winner's moves too. Win and")
    p("loss rates therefore cannot show the bias; only the draw rate and the rating")
    p("distribution can. The reference dict is approximate - see the top of this")
    p("file before quoting any delta.")


# ---------------------------------------------------------------------------
# Check 6 - hard integrity assertion
# ---------------------------------------------------------------------------


def check_nulls(valid: pd.DataFrame, n_blunder_null: int) -> bool:
    """`n_blunder_null` is counted in main() BEFORE `blunder` is coerced to bool.

    Counting it here would be vacuous: astype(bool) maps NaN to True, so the
    nulls would already be gone (and would have been silently counted as
    blunders, inflating the base rate).
    """
    rule("6. INTEGRITY: cp / mate / blunder nulls on label_valid rows")

    rows, ok = [], True
    for col in ("cp_before", "cp_after"):
        n = int(valid[col].isna().sum()) if col in valid.columns else -1
        if n != 0:
            ok = False
        rows.append([col, f"{n:,}" if n >= 0 else "COLUMN MISSING",
                     "OK" if n == 0 else "FAIL"])
    # mate_before / mate_after must be null exactly when the label is valid.
    for col in ("mate_before", "mate_after"):
        if col in valid.columns:
            n = int(valid[col].notna().sum())
            if n != 0:
                ok = False
            rows.append([f"{col} non-null on valid", f"{n:,}", "OK" if n == 0 else "FAIL"])
    if n_blunder_null:
        ok = False
    rows.append(["blunder null on valid (pre-coercion)", f"{n_blunder_null:,}",
                 "OK" if n_blunder_null == 0 else "FAIL"])

    table(["check", "offending rows", "verdict"], rows)

    if not ok:
        loud("NULL CHECK FAILED - see the table above.\n"
             "A label_valid row with a missing cp is a parser bug, not a data "
             "quirk.")
    return ok


# ---------------------------------------------------------------------------
# Self-test: pin the colour gate in BOTH directions
# ---------------------------------------------------------------------------


def _fixture(n: int = 2_000_000, seed: int = 0, broken: bool = False) -> pd.DataFrame:
    """Synthetic valid-rows frame matching the real win_drop marginals.

    The three constants are fitted to what the June 2026 blitz parquet actually
    looks like, because a fixture with NO negative tail would give Black a zero
    blunder rate and an infinite ratio, and would then pass any tolerance you
    cared to set. The tail is the whole test, so it has to be the real size:

                            fixture   real (blitz, full month)
        P(win_drop > 20)      3.8%      3.952%
        P(win_drop < 0)      19.5%     20.08%
        P(win_drop < -20)  0.00067%   0.00070%

    The last row is what sets the magnitude of the broken case: a broken flip
    turns Black's blunder rate into P(win_drop < -20), so the fixture's ratio
    lands near the ~5,600x the real data implies.

    `broken=True` simulates the bug check 2b exists to catch: the mover-POV
    sign flip is not applied to Black, so Black's win_drop keeps White's sign.
    """
    rng = np.random.default_rng(seed)
    is_white = rng.random(n) < 0.5
    drop = (rng.standard_exponential(n) * 6.2 - 1.0
            + rng.laplace(0.0, 2.0, n))
    if broken:
        drop = np.where(is_white, drop, -drop)
    return pd.DataFrame({
        "mover_is_white": is_white,
        "mover_elo": rng.integers(800, 2400, n),
        "win_drop": drop,
        "blunder": drop > DEFAULT_THRESHOLD,
    })


def self_test(tol: float = COLOUR_TOL) -> int:
    """Run check 2b against a healthy and a sign-flipped fixture."""
    failures = []
    results = {}
    for name, broken, want in (("healthy", False, True), ("sign-flipped", True, False)):
        frame = _fixture(broken=broken)
        buf = io.StringIO()
        saved = list(_LINES)
        with contextlib.redirect_stdout(buf):
            got = check_colour_symmetry(frame, tol)
        _LINES[:] = saved
        ratio = next((ln for ln in buf.getvalue().splitlines() if "ratio" in ln), "")
        results[name] = (got, ratio.strip(), frame)
        if got is not want:
            failures.append(
                f"{name} fixture: expected the gate to "
                f"{'PASS' if want else 'FAIL'}, it did the opposite")

    rule("SELF-TEST: colour-symmetry gate")
    healthy = results["healthy"][2]
    p(f"fixture: {len(healthy):,} rows, "
      f"P(win_drop>{DEFAULT_THRESHOLD:g})={pct((healthy.win_drop>DEFAULT_THRESHOLD).mean(),3)}, "
      f"P(<0)={pct((healthy.win_drop<0).mean(),1)}, "
      f"P(<-{DEFAULT_THRESHOLD:g})={pct((healthy.win_drop<-DEFAULT_THRESHOLD).mean(),4)}")
    p(f"gate tolerance: {tol:.2f}x")
    p()
    for name, (got, ratio, _) in results.items():
        p(f"{name:14s} -> {'pass' if got else 'FAIL'}")
        p(f"               {ratio}")
    p()
    if failures:
        loud("SELF-TEST FAILED:\n" + "\n".join(failures))
        return 1
    p("Both directions behave: healthy data passes, a simulated sign flip is")
    p("caught with more than three orders of magnitude of headroom. This is the")
    p("fixture behind the claim in FINDINGS.md, so re-run it rather than taking")
    p("it on trust.")
    return 0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def load(path: str) -> tuple[pd.DataFrame, list[str]]:
    """Read only the columns the report uses; return them plus the full schema."""
    import pyarrow.parquet as pq

    schema = list(pq.ParquetFile(path).schema_arrow.names)
    missing = [c for c in REQUIRED if c not in schema]
    if missing:
        raise SystemExit(
            f"{path} is missing required column(s): {', '.join(missing)}\n"
            f"present: {', '.join(schema)}"
        )
    want = REQUIRED + [c for c in OPTIONAL if c in schema]
    return pd.read_parquet(path, columns=want), schema


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="positions parquet to validate")
    ap.add_argument("--out", default=None,
                    help="also tee the report to this text file")
    ap.add_argument("--colour-tol", type=float, default=COLOUR_TOL,
                    help=f"max white/black blunder-rate ratio (default {COLOUR_TOL})")
    ap.add_argument("--strict", action="store_true",
                    help="also gate on rating-decile monotonicity (check 2)")
    ap.add_argument("--self-test", action="store_true",
                    help="run the colour-gate fixtures instead of reading --data")
    args = ap.parse_args()

    if args.self_test:
        rc = self_test(args.colour_tol)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write("\n".join(_LINES) + "\n")
        return rc
    if not args.data:
        ap.error("--data is required unless --self-test is given")

    df, schema = load(args.data)
    rule(f"LABEL VALIDATION REPORT: {args.data}")
    p(f"{len(df):,} rows x {len(schema)} columns in file, "
      f"{len(df.columns)} read")

    if not len(df):
        p("empty parquet, nothing to check.")
        return 1

    label_valid = df.label_valid.fillna(False).astype(bool)
    valid = df[label_valid].copy()
    if not len(valid):
        loud("no label_valid rows at all. Every eval on every row was a forced "
             "mate,\nwhich means the parser is not reading cp comments.")
        return 1

    # Count blunder nulls BEFORE coercing, and fill with False rather than
    # astype(bool): astype maps NaN to True and would silently promote a
    # missing label to a positive.
    n_blunder_null = int(valid.blunder.isna().sum())
    valid["blunder"] = valid.blunder.fillna(False).astype(bool)

    check_counts(df, valid, schema)
    failed_pools = check_breakdowns(valid)
    colour_ok = check_colour_symmetry(valid, args.colour_tol)
    check_coverage(df)
    check_clocks(df)
    check_win_drop(valid)
    floor_ok = check_structural_floor(valid)
    check_thresholds(valid)
    check_selection_bias(df)
    nulls_ok = check_nulls(valid, n_blunder_null)

    mono_ok = not failed_pools
    gates = {
        "2b colour symmetry": colour_ok,
        "3c structural floor": floor_ok,
        "6 null integrity": nulls_ok,
    }
    if args.strict:
        gates["2 rating monotonicity"] = mono_ok

    rule("SUMMARY")
    table(
        ["check", "verdict", "gates?"],
        [
            ["1 counts / base rate", "see table", "no"],
            ["2 rating monotonicity",
             "OK" if mono_ok else "FAILED in: " + ", ".join(failed_pools),
             "yes" if args.strict else "no (--strict)"],
            ["2b colour symmetry",
             "OK" if colour_ok else "FAILED - check sign flip", "yes"],
            ["2c calendar coverage", "see table", "no"],
            ["2d clocks", "see table", "no"],
            ["3 win_drop distribution", "see table", "no"],
            ["3c structural floor",
             "OK" if floor_ok else "FAILED - impossible blunder", "yes"],
            ["4 threshold robustness", "see table", "no"],
            ["5 selection bias", "see notes", "no"],
            ["6 null integrity", "OK" if nulls_ok else "FAILED", "yes"],
        ],
    )

    failed = [k for k, v in gates.items() if not v]
    p()
    if failed:
        p("RESULT: FAILED -> " + "; ".join(failed))
    else:
        p("RESULT: all gating checks passed.")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(_LINES) + "\n")
        print(f"\nreport written to {args.out}", file=sys.stderr)

    # Deliberately not `assert`: `python -O` strips asserts, and a gate that
    # silently disappears under an optimisation flag is worse than no gate.
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
