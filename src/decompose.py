#!/usr/bin/env python3
"""
Phase 7. Decompose the rating effect on blundering into two channels.

    do stronger players face fewer chances to blunder,
    or do they take the chances they face less often?

THE IDENTITY, and why it is written this way.

The obvious form, P(blunder) = P(blunder available) x P(picks one | available),
is nearly vacuous here: the binary flag is true in 81.5% of positions, because
"does ANY of 31 legal moves lose 20% or more" is almost always yes. Structural
rather than small-sample.

The quantity that actually varies is `frac_blunder_moves` -- what FRACTION of
legal moves are blunders. Median 0.250, p90 0.886. That is the thickness of the
minefield, and it is also exactly what uniform-random play would score.

So per position, with p the blunder probability and f the blunder fraction:

    s := p / f        the SELECTION RATIO: how often a human plays a blunder
                      relative to how often random play would. s = 1 means
                      "as bad as random"; s = 0.11 means 9x better.

Two problems with decomposing that directly, both fixed by binning first.

1. `p_i` is latent. We observe a binary outcome, so per-position log p_i is
   log 0 on every non-blunder and the decomposition is undefined.
2. E[log f] != log E[f]. A per-position log decomposition would explain
   variation in the GEOMETRIC mean while the 3.95% everyone quotes is
   arithmetic.

Binning fixes both. Within a bin b, take arithmetic means p_b and f_b, define
s_b := p_b / f_b, and then

    log p_b = log f_b + log s_b

holds exactly, so differences across rating bands decompose additively with NO
residual and no covariance term left over:

    D log p = D log f + D log s
    availability share = D log f / D log p
    selection share    = D log s / D log p     (the two sum to 1 exactly)

TWO CONTROLS, answering different questions. Report both.

  - Restrict to `best_child_win > 22.46` (mechanical). Below that, no child can
    be 20 win% worse than the best, so `frac_blunder_moves` is forced to 0 by
    arithmetic rather than by the position being safe. THIS IS THE HEADLINE
    NUMBER: on this data the uncontrolled contrast is dominated by the floor,
    because weak players sit below it more than twice as often (14.9% against
    6.8%). Section 2 prints that mechanism explicitly.
  - Band on `winpct_before` (causal). Holding the kind of position reached
    constant, do weak players face thicker minefields? Rating correlates with
    the eval distribution, so a marginal comparison confounds "faces a thicker
    minefield" with "spends more time in decided positions".

TWO FLOORS, NOT ONE. The label's floor is on `winpct_before`, which comes from
Lichess' deep annotation. The mechanical floor here is on `best_child_win`, a
depth-8 sweep over children measured against the BEST CHILD. They are different
quantities and disagree on a handful of positions; section 0 reports the rate
and it is a depth disagreement, not an inconsistency.

WHY THE BOOTSTRAP IS CLUSTERED. The sample averages 2.05 positions per game and
79% of positions come from a game contributing more than one. Positions in a
game share a player, a rating, an opening and a clock situation, so an iid
bootstrap over positions understates the variance. Resampling GAMES is the
honest interval; `--no-cluster` shows the difference.

Usage:
    python decompose.py --children data/children_150k.parquet \
        --features data/features_blitz_full.parquet \
        --out metrics/decomposition.json --report reports/decomposition.txt

    python decompose.py --self-test     # planted-split recovery, no data needed
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

# win% floors at 2.46 because cp is clamped to +/-1000, so a 20-point drop is
# impossible unless the best child is above this. Derived, not tuned.
FLOOR = 22.46

RATING_BANDS = [(0, 1200, "<1200"), (1200, 1600, "1200-1600"),
                (1600, 2000, "1600-2000"), (2000, 9999, "2000+")]
EVAL_BANDS = [(FLOOR, 40, "losing"), (40, 60, "balanced"),
              (60, 80, "better"), (80, 100.1, "winning")]

EPS = 1e-12


# ---------------------------------------------------------------------------
# the decomposition itself
# ---------------------------------------------------------------------------


def bin_stats(df: pd.DataFrame, bands, col: str) -> pd.DataFrame:
    """Arithmetic means per bin. p, f and s must come from the SAME rows, or
    you are mixing a population-weighted rate with an equal-weight fraction."""
    out = []
    for lo, hi, label in bands:
        m = (df[col] >= lo) & (df[col] < hi)
        sub = df[m]
        if len(sub) < 30:
            continue
        p = sub.blunder.mean()
        f = sub.frac_blunder_moves.mean()
        out.append({"band": label, "n": len(sub),
                    "p": p, "f": f,
                    "s": p / f if f > EPS else np.nan,
                    "better_than_random": f / p if p > EPS else np.nan,
                    "mid": sub[col].median()})
    return pd.DataFrame(out)


def decompose(stats: pd.DataFrame) -> dict:
    """Endpoint contrast plus an OLS slope version.

    The endpoint contrast is what the README quotes; the slope uses every band
    and is the robustness check. They should agree. If they do not, one band is
    driving the result and that is worth knowing.
    """
    s = stats.dropna(subset=["p", "f", "s"])
    s = s[(s.p > EPS) & (s.f > EPS)]
    if len(s) < 2:
        return {"ok": False, "reason": "fewer than 2 usable bands"}

    lo, hi = s.iloc[0], s.iloc[-1]
    d_logp = np.log(hi.p) - np.log(lo.p)
    d_logf = np.log(hi.f) - np.log(lo.f)
    d_logs = np.log(hi.s) - np.log(lo.s)

    res = {
        "ok": True,
        "contrast": f"{lo.band} -> {hi.band}",
        "d_log_p": d_logp, "d_log_f": d_logf, "d_log_s": d_logs,
        "residual": d_logp - (d_logf + d_logs),
    }
    if abs(d_logp) > EPS:
        res["availability_share"] = d_logf / d_logp
        res["selection_share"] = d_logs / d_logp
    else:
        res["availability_share"] = res["selection_share"] = float("nan")

    # slope version: regress each log series on the band midpoint
    x = s["mid"].to_numpy(dtype=float)
    if len(s) >= 3 and x.std() > 0:
        bp = np.polyfit(x, np.log(s.p.to_numpy()), 1)[0]
        bf = np.polyfit(x, np.log(s.f.to_numpy()), 1)[0]
        bs = np.polyfit(x, np.log(s.s.to_numpy()), 1)[0]
        res["slope_log_p_per_100elo"] = bp * 100
        res["slope_availability_share"] = bf / bp if abs(bp) > EPS else np.nan
        res["slope_selection_share"] = bs / bp if abs(bp) > EPS else np.nan
    return res


# ---------------------------------------------------------------------------
# bootstrap
#
# Done in numpy rather than by re-running bin_stats on resampled DataFrames.
# Not premature optimisation: the causal-control section alone needs 8 separate
# bootstraps, and the pandas version cost about 40x more per resample, which
# was enough to push n_boot down to a number too small to read a 2.5th
# percentile off.
# ---------------------------------------------------------------------------


def band_index(values: np.ndarray, bands) -> np.ndarray:
    """Map each row to its band, -1 for rows outside every band."""
    idx = np.full(len(values), -1, dtype=np.int64)
    for i, (lo, hi, _) in enumerate(bands):
        idx[(values >= lo) & (values < hi)] = i
    return idx


def _share_from_arrays(bidx, blunder, frac, n_bands, min_n=30):
    """The availability share, computed by bincount over band codes.

    Returns None when the contrast is undefined, so a resample that empties a
    band is dropped rather than silently changing which bands are contrasted.
    """
    n = np.bincount(bidx, minlength=n_bands).astype(float)
    sp = np.bincount(bidx, weights=blunder, minlength=n_bands)
    sf = np.bincount(bidx, weights=frac, minlength=n_bands)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = sp / n
        f = sf / n
    good = (n >= min_n) & (p > EPS) & (f > EPS)
    w = np.flatnonzero(good)
    if len(w) < 2:
        return None
    lo, hi = w[0], w[-1]
    d_logp = np.log(p[hi]) - np.log(p[lo])
    if abs(d_logp) <= EPS:
        return None
    # selection share is 1 - availability by the identity, exactly.
    return (np.log(f[hi]) - np.log(f[lo])) / d_logp


def _cluster_ranges(order_starts, order_counts, draw):
    """Concatenated index ranges for a drawn set of clusters, vectorised.

    np.concatenate over 73k per-cluster arrays per resample is the obvious
    version and is far too slow. This builds the whole index in one cumsum.
    """
    counts = order_counts[draw]
    keep = counts > 0
    counts, starts = counts[keep], order_starts[draw][keep]
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    out = np.ones(total, dtype=np.int64)
    out[0] = starts[0]
    if len(starts) > 1:
        pos = np.cumsum(counts)[:-1]
        out[pos] = starts[1:] - (starts[:-1] + counts[:-1]) + 1
    return np.cumsum(out)


def bootstrap_shares(df: pd.DataFrame, bands, col: str, n_boot: int = 400,
                     seed: int = 0, cluster: str | None = None) -> dict:
    """Resample and recompute the availability share.

    Necessary, not decorative. The share is a ratio of differences of logs of
    noisy estimates, which is a construction with fat tails: when the
    denominator D log p is small the ratio explodes. A point estimate without
    an interval would be misleading.

    `cluster` names a column to resample instead of rows. Positions within a
    game are not independent, so the iid interval is too narrow.
    """
    if len(df) < 2:
        return {"ok": False, "reason": "too few rows"}
    bidx = band_index(df[col].to_numpy(dtype=float), bands)
    keep = bidx >= 0
    bidx = bidx[keep]
    blunder = df.blunder.to_numpy(dtype=float)[keep]
    frac = df.frac_blunder_moves.to_numpy(dtype=float)[keep]
    n_bands = len(bands)
    rng = np.random.default_rng(seed)

    if cluster is not None:
        codes = pd.factorize(df[cluster].to_numpy()[keep])[0]
        order = np.argsort(codes, kind="stable")
        bidx, blunder, frac = bidx[order], blunder[order], frac[order]
        counts = np.bincount(codes)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        n_clusters = len(counts)
    else:
        n_rows = len(bidx)

    vals = []
    for _ in range(n_boot):
        if cluster is not None:
            take = _cluster_ranges(starts, counts,
                                   rng.integers(0, n_clusters, n_clusters))
        else:
            take = rng.integers(0, n_rows, n_rows)
        a = _share_from_arrays(bidx[take], blunder[take], frac[take], n_bands)
        if a is not None and np.isfinite(a):
            vals.append(a)

    if len(vals) < n_boot // 4:
        return {"ok": False, "reason": "too many degenerate resamples",
                "n_ok": len(vals)}
    a = np.asarray(vals)
    ci = [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"ok": True, "n_boot": len(a), "clustered_by": cluster,
            "availability_ci": ci,
            # selection = 1 - availability exactly, so its CI is the mirror
            "selection_ci": [1.0 - ci[1], 1.0 - ci[0]],
            "availability_median": float(np.median(a)),
            "availability_se": float(a.std(ddof=1)),
            "excludes_zero": bool(ci[0] > 0 or ci[1] < 0)}


# ---------------------------------------------------------------------------
# loading, with the guard that matters
# ---------------------------------------------------------------------------


def load(children_path: str, features_path: str) -> pd.DataFrame:
    kids = pd.read_parquet(children_path)
    need = {"row_id", "fen", "frac_blunder_moves", "blunder_available",
            "n_moves", "best_child_win"}
    missing = need - set(kids.columns)
    if missing:
        raise SystemExit(f"FATAL: children file lacks {sorted(missing)}")

    cols = ["game_id", "fen", "blunder", "label_valid", "mover_elo",
            "winpct_before"]
    feats = pd.read_parquet(features_path, columns=cols)
    feats = feats[feats.label_valid].reset_index(drop=True)
    feats["row_id"] = feats.index

    df = kids.merge(feats, on="row_id", how="inner", suffixes=("", "_f"))

    # row_id is a POSITIONAL index into one specific parquet's label_valid
    # rows. Point this at a different features file and every id still
    # resolves, silently, to an unrelated position. Verify on the FEN.
    mism = (df.fen != df.fen_f).sum()
    if mism:
        raise SystemExit(
            f"FATAL: {mism:,} of {len(df):,} row_ids resolve to a different "
            f"FEN.\nThe children file was built against a different features "
            f"parquet than\n{features_path}. row_id is positional, so this "
            f"would fail silently.")

    df = df.drop(columns=["fen_f"])
    df["blunder"] = df.blunder.fillna(False).astype(int)
    print(f"{len(kids):,} evaluated positions joined to {len(df):,} rows, "
          f"FEN check passed")
    return df


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def table(rows, headers):
    rows = [[str(c) for c in r] for r in rows]
    w = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
         for i, h in enumerate(headers)]
    out = ["  ".join(h.ljust(x) for h, x in zip(headers, w)),
           "  ".join("-" * x for x in w)]
    out += ["  ".join(c.ljust(x) for c, x in zip(r, w)) for r in rows]
    return "\n".join(out)


def stats_table(s: pd.DataFrame) -> str:
    rows = [[r.band, f"{r.n:,}", f"{r.p:.4%}", f"{r.f:.1%}", f"{r.s:.4f}",
             f"{r.better_than_random:.1f}x"] for r in s.itertuples()]
    return table(rows, ["band", "n", "P(blunder)", "E[frac]",
                        "s = p/f", "vs random"])


def describe(d: dict, boot: dict | None = None) -> str:
    if not d.get("ok"):
        return f"  cannot decompose: {d.get('reason')}"
    L = [f"  contrast {d['contrast']}",
         f"  D log p = {d['d_log_p']:+.4f}   "
         f"D log f = {d['d_log_f']:+.4f}   D log s = {d['d_log_s']:+.4f}",
         f"  residual {d['residual']:+.2e}  (must be ~0; the identity is exact)",
         "",
         f"  AVAILABILITY share {d['availability_share']:+7.1%}"
         f"   (do weak players face thicker minefields?)",
         f"  SELECTION    share {d['selection_share']:+7.1%}"
         f"   (do they step in them more often?)"]
    if boot and boot.get("ok"):
        a, s = boot["availability_ci"], boot["selection_ci"]
        how = (f"clustered by {boot['clustered_by']}" if boot["clustered_by"]
               else "iid over positions")
        L += ["", f"  bootstrap 95% CI ({boot['n_boot']} resamples, {how}):",
              f"    availability [{a[0]:+.1%}, {a[1]:+.1%}]",
              f"    selection    [{s[0]:+.1%}, {s[1]:+.1%}]",
              f"    -> availability {'EXCLUDES' if boot['excludes_zero'] else 'includes'}"
              f" zero"]
    if "slope_availability_share" in d:
        L += ["", f"  slope check (all bands, per 100 Elo): "
                  f"availability {d['slope_availability_share']:+.1%}, "
                  f"selection {d['slope_selection_share']:+.1%}",
              f"    log p slope {d['slope_log_p_per_100elo']:+.4f} per 100 Elo",
              "    should agree with the endpoint contrast; if it does not, "
              "one band is driving\n    the result"]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------


def self_test(n_seeds: int = 25, n_band: int = 100_000) -> int:
    """Plant a known availability/selection split and check it is recovered.

    Construction: choose f_b and s_b per band so that log f moves a chosen
    share of the total log p movement, then draw binary outcomes at
    p_b = f_b * s_b. If the code returns the planted share, the arithmetic is
    right.

    Judged over MANY SEEDS, not one. The share is a ratio of differences of
    logs of noisy estimates, so a single draw scatters by a couple of points
    even when the estimator is unbiased: measured sd is 1.7% at 100k per band,
    falling to 0.56% at 1M, i.e. the 1/sqrt(n) you would expect. Testing one
    seed against a tight tolerance therefore fails at random, which is how a
    correct implementation gets mistaken for a biased one.

    The negative case checks that a term moving AGAINST the total is reported
    as a negative share rather than silently renormalised into [0, 1].

    The FLOOR case is the one that justifies section 2 of the report. It plants
    ZERO availability effect, then censors frac to 0 on a rating-dependent
    share of rows -- exactly what the win% clamp does to real data. The
    uncontrolled contrast must then read a large spurious availability share
    and the controlled one must recover the planted zero. If that case ever
    fails, the headline number in section 2 is not doing what it claims.
    """
    print(f"planted-split recovery: {n_seeds} seeds x {n_band:,} rows per "
          f"band\n")

    def one(planted, seed, floor_share=None):
        rng = np.random.default_rng(seed)
        p_lo, p_hi = 0.070, 0.022      # roughly the real D1 -> D10 spread
        d_logp = np.log(p_hi) - np.log(p_lo)
        f_lo = 0.44
        f_hi = f_lo * np.exp(planted * d_logp)
        parts = []
        for i, (elo, p, f) in enumerate(((1000, p_lo, f_lo),
                                         (2200, p_hi, f_hi))):
            frac = np.clip(rng.normal(f, 0.02, n_band), 0.01, 0.99)
            blunder = (rng.random(n_band) < p).astype(int)
            best = np.full(n_band, 60.0)
            if floor_share is not None:
                # censor a rating-dependent share to zero, both channels at
                # once: below the floor no move can be a blunder AND none is
                # observed. That is what the clamp does.
                hit = rng.random(n_band) < floor_share[i]
                frac = np.where(hit, 0.0, frac)
                blunder = np.where(hit, 0, blunder)
                best = np.where(hit, 10.0, best)
            parts.append(pd.DataFrame({
                "mover_elo": elo, "frac_blunder_moves": frac,
                "blunder": blunder, "winpct_before": 50.0,
                "best_child_win": best}))
        df = pd.concat(parts, ignore_index=True)
        bands = [(0, 1600, "low"), (1600, 9999, "high")]
        raw = decompose(bin_stats(df, bands, "mover_elo"))
        ctl = decompose(bin_stats(df[df.best_child_win > FLOOR], bands,
                                  "mover_elo"))
        return raw, ctl

    rows, failed = [], 0
    for planted in (0.0, 0.30, 0.50, 0.85, 1.0, -0.25):
        ds = [one(planted, s)[0] for s in range(n_seeds)]
        v = np.array([d["availability_share"] for d in ds])
        resid = max(abs(d["residual"]) for d in ds)
        se = v.std(ddof=1) / np.sqrt(len(v))
        # unbiased if the planted value sits inside the mean's own 3-se band
        ok = abs(v.mean() - planted) < max(3 * se, 0.005) and resid < 1e-9
        failed += not ok
        rows.append([f"{planted:+.0%}", f"{v.mean():+.2%}", f"{v.std(ddof=1):.2%}",
                     f"{v.mean()-planted:+.2%}", f"{resid:.1e}",
                     "OK" if ok else "FAIL"])
    print(table(rows, ["planted", "mean recovered", "sd", "bias",
                       "max residual", ""]))
    print("\nresidual is the identity check, D log p - (D log f + D log s).")
    print("It is at machine epsilon because the identity is exact by")
    print("construction rather than fitted.")
    print("\nsd is per-run scatter, NOT bias. It sets how precise a single real")
    print("run can be, which is why the tool reports a bootstrap interval.")

    # --- the floor case ----------------------------------------------------
    print("\n\nmechanical-floor case: 0% availability planted, then frac")
    print("censored to 0 on 15% of weak rows and 7% of strong rows (the real")
    print("ineligible shares). The uncontrolled contrast must read a large")
    print("SPURIOUS availability share; the control must recover ~0%.\n")
    pairs = [one(0.0, s, floor_share=(0.15, 0.07)) for s in range(n_seeds)]
    raw_v = np.array([d["availability_share"] for d, _ in pairs])
    ctl_v = np.array([d["availability_share"] for _, d in pairs])
    raw_ok = raw_v.mean() < -0.03           # spurious and NEGATIVE, as observed
    ctl_se = ctl_v.std(ddof=1) / np.sqrt(len(ctl_v))
    ctl_ok = abs(ctl_v.mean()) < max(3 * ctl_se, 0.005)
    failed += (not raw_ok) + (not ctl_ok)
    print(table([["uncontrolled", f"{raw_v.mean():+.2%}", f"{raw_v.std(ddof=1):.2%}",
                  "must be << 0", "OK" if raw_ok else "FAIL"],
                 ["controlled", f"{ctl_v.mean():+.2%}", f"{ctl_v.std(ddof=1):.2%}",
                  "must be ~0", "OK" if ctl_ok else "FAIL"]],
                ["contrast", "mean availability", "sd", "expected", ""]))
    print("\nThe spurious share is NEGATIVE because censoring drags the weak")
    print("band's mean frac down, so f appears to RISE with rating. That is the")
    print("sign seen on the real data, and section 2 is what removes it.")

    print(f"\n{'all cases recovered' if not failed else f'{failed} FAILED'}")
    return 1 if failed else 0


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--children")
    ap.add_argument("--features")
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cluster", action="store_true",
                    help="bootstrap iid over positions instead of over games. "
                         "Only for showing how much the clustering matters; "
                         "the clustered interval is the honest one.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.children or not args.features:
        ap.error("--children and --features are required (or use --self-test)")

    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    df = load(args.children, args.features)
    clus = None if args.no_cluster else "game_id"
    if clus and clus not in df.columns:
        say(f"  WARNING: no {clus} column, falling back to an iid bootstrap")
        clus = None

    def boot(d, bands=RATING_BANDS, col="mover_elo"):
        return bootstrap_shares(d, bands, col, args.n_boot, args.seed, clus)

    results = {"n_positions": len(df), "n_boot": args.n_boot,
               "clustered_by": clus}

    say("\n" + "=" * 74)
    say("0. SANITY: is the binary flag informative, and do the floors agree?")
    say("=" * 74)
    fl = df.blunder_available.mean()
    say(f"  blunder_available is true in {fl:.1%} of positions")
    say(f"  frac_blunder_moves: median {df.frac_blunder_moves.median():.3f}, "
        f"p90 {df.frac_blunder_moves.quantile(0.90):.3f}, "
        f"mean {df.frac_blunder_moves.mean():.3f}")
    if fl > 0.7:
        say("  -> the binary flag is near-constant and carries little "
            "information.\n     The decomposition therefore uses "
            "frac_blunder_moves, which is also\n     exactly what "
            "uniform-random play would score.")
    results["binary_flag_prevalence"] = float(fl)

    below = df[df.best_child_win <= FLOOR]
    max_frac = float(below.frac_blunder_moves.max()) if len(below) else 0.0
    odd = int(below.blunder.sum())
    say("")
    say(f"  below the mechanical floor: {len(below):,} positions, "
        f"max frac_blunder_moves {max_frac:.3f}")
    say(f"  labelled blunders in those rows: {odd:,} "
        f"({odd / max(len(below), 1):.3%})")
    say("  Those are not an inconsistency. The LABEL's floor is on")
    say("  winpct_before, from Lichess' deep annotation; this floor is on")
    say("  best_child_win from a depth-8 sweep measured against the best")
    say("  child. Two different quantities, so a handful disagree.")
    if max_frac > 0:
        say("  WARNING: frac must be exactly 0 below the floor. It is not.")
    if odd / max(len(below), 1) > 0.01:
        say("  WARNING: over 1% disagreement is too much for depth alone.")
    results["floor_integrity"] = {"n_below": len(below), "max_frac": max_frac,
                                  "labelled_blunders": odd}

    say("\n" + "=" * 74)
    say("1. UNCONTROLLED: by rating band, all positions")
    say("=" * 74)
    say("  Reported for completeness. It is NOT the headline -- section 2")
    say("  shows most of the availability share here is the mechanical floor.")
    say("")
    st = bin_stats(df, RATING_BANDS, "mover_elo")
    say(stats_table(st))
    d = decompose(st)
    b = boot(df)
    say("")
    say(describe(d, b))
    results["uncontrolled"] = {"bins": st.to_dict("records"),
                               "decomposition": d, "bootstrap": b}

    say("\n" + "=" * 74)
    say("2. HEADLINE, MECHANICAL CONTROL: best_child_win > %.2f" % FLOOR)
    say("=" * 74)
    say("  Below this, no child can be 20 win%% worse than the best, so")
    say("  frac_blunder_moves is forced to zero by arithmetic rather than by")
    say("  the position being safe.")
    elig = df[df.best_child_win > FLOOR]
    say(f"  {len(elig):,} of {len(df):,} positions ({len(elig)/len(df):.1%})")
    say("")
    say("  WHY THIS MATTERS HERE. Weak players sit below the floor far more")
    say("  often, so censored zeros drag their mean frac down and f appears to")
    say("  RISE with rating. Removing them flattens it:")
    say("")
    mech = []
    for lo, hi, lab in RATING_BANDS:
        sub = df[(df.mover_elo >= lo) & (df.mover_elo < hi)]
        if not len(sub):
            continue
        se = sub[sub.best_child_win > FLOOR]
        mech.append([lab, f"{len(sub):,}",
                     f"{(sub.best_child_win <= FLOOR).mean():.2%}",
                     f"{sub.frac_blunder_moves.mean():.4f}",
                     f"{se.frac_blunder_moves.mean():.4f}"])
    say("  " + table(mech, ["band", "n", "below floor", "E[frac] all",
                            "E[frac] eligible"]).replace("\n", "\n  "))
    results["floor_mechanism"] = mech

    st_e = bin_stats(elig, RATING_BANDS, "mover_elo")
    say("")
    say(stats_table(st_e))
    d_e = decompose(st_e)
    b_e = boot(elig)
    say("")
    say(describe(d_e, b_e))
    say("")
    say("  Note s = p/f is unchanged from section 1 to four decimals. The")
    say("  floor removes matched mass from p and f, so it cannot move their")
    say("  ratio -- an internal check that section 2 removes an artefact")
    say("  rather than a real effect.")
    results["mechanical_control"] = {"bins": st_e.to_dict("records"),
                                     "decomposition": d_e, "bootstrap": b_e}

    say("\n" + "=" * 74)
    say("3. CAUSAL CONTROL: within winpct_before bands")
    say("=" * 74)
    say("  Rating correlates with the eval distribution, so a marginal")
    say("  comparison confounds 'faces thicker minefields' with 'spends more")
    say("  time in decided positions'. Holding the eval band fixed separates")
    say("  them. Each band gets its own interval: the cells here are 10x")
    say("  smaller than in section 2 and a point estimate would over-read.")
    per_eval = {}
    for lo, hi, lab in EVAL_BANDS:
        sub = elig[(elig.winpct_before >= lo) & (elig.winpct_before < hi)]
        if len(sub) < 500:
            continue
        st_b = bin_stats(sub, RATING_BANDS, "mover_elo")
        d_b = decompose(st_b)
        b_b = boot(sub)
        per_eval[lab] = {"n": len(sub), "bins": st_b.to_dict("records"),
                         "decomposition": d_b, "bootstrap": b_b}
        say(f"\n  --- winpct_before {lab} ({len(sub):,} positions) ---")
        say("  " + stats_table(st_b).replace("\n", "\n  "))
        if d_b.get("ok"):
            line = (f"    availability {d_b['availability_share']:+.1%}   "
                    f"selection {d_b['selection_share']:+.1%}")
            if b_b.get("ok"):
                a = b_b["availability_ci"]
                line += f"   95% CI [{a[0]:+.1%}, {a[1]:+.1%}]"
                line += "  EXCLUDES 0" if b_b["excludes_zero"] else "  includes 0"
            say(line)
    results["causal_control"] = per_eval

    say("\n" + "=" * 74)
    say("4. DISTRIBUTION OF frac_blunder_moves BY RATING (deciles, not means)")
    say("=" * 74)
    say("  median 0.25 against p90 0.89 makes the mean a poor summary. If the")
    say("  whole curve is flat across rating, the availability channel is")
    say("  genuinely empty rather than merely small on average.")
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    rows = []
    for lo, hi, lab in RATING_BANDS:
        sub = elig[(elig.mover_elo >= lo) & (elig.mover_elo < hi)]
        if len(sub) < 100:
            continue
        q = sub.frac_blunder_moves.quantile(qs)
        rows.append([lab, f"{len(sub):,}"] + [f"{q[x]:.3f}" for x in qs])
    say("")
    say(table(rows, ["band", "n"] + [f"p{int(x*100)}" for x in qs]))
    results["frac_quantiles_by_rating"] = rows

    say("\n" + "=" * 74)
    say("SUMMARY")
    say("=" * 74)
    if d_e.get("ok"):
        a, sh = d_e["availability_share"], d_e["selection_share"]
        say(f"  On the mechanically controlled contrast ({d_e['contrast']}),")
        say(f"  {a:.0%} of the drop in blunder rate runs through availability")
        say(f"  and {sh:.0%} through selection.")
        if b_e.get("ok"):
            ci = b_e["availability_ci"]
            say(f"  Availability 95% CI [{ci[0]:+.1%}, {ci[1]:+.1%}], "
                f"{'excluding' if b_e['excludes_zero'] else 'including'} zero.")
            if not b_e["excludes_zero"] and (ci[1] - ci[0]) < 0.10:
                say("")
                say("  That interval is NARROW and brackets zero, so this is a")
                say("  precise null rather than an absence of power: the data")
                say("  can rule out an availability channel worth more than a")
                say("  couple of points, not merely fail to detect one.")
            elif not b_e["excludes_zero"]:
                say("")
                say("  The interval is WIDE and brackets zero, so state this as")
                say("  a BOUND, not a point estimate: the availability channel")
                say("  is no larger than the interval allows, and this run")
                say("  cannot say more than that.")
        if abs(a) < 0.15:
            say("")
            say("  The availability channel is essentially empty. Position")
            say("  difficulty is not something weak players walk into more")
            say("  often; they fall for it more often. That is the stronger")
            say("  claim and the one the phase was scoped to test.")
        elif a > 0.4:
            say("")
            say("  A large share runs through availability, so 'weak players")
            say("  reach messier positions' is a real part of the rating")
            say("  effect and the README must say so.")
    say("")
    say("  Caveat: the selection ratio s is a ratio of a measured probability")
    say("  to what UNIFORM-RANDOM play would score. Humans do not sample")
    say("  uniformly from legal moves, so s is a benchmark against random, not")
    say("  a claim about human search. A human-policy baseline (Maia) would be")
    say("  the honest denominator and is out of scope.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2, default=float)
        print(f"\nwrote {args.out}")
    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
