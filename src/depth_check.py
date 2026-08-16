#!/usr/bin/env python3
"""
Depth sensitivity for eval_children.py, run BEFORE committing a night to it.

eval_children evaluates every legal move in every sampled position. Cost is
linear in depth-ish and the full run is hours, so the only question worth
answering first is: does a cheaper depth give the same ANSWER?

It deliberately imports `_eval_position` and `_init` from eval_children rather
than reimplementing them. A depth check that runs its own copy of the
evaluation would be measuring its own copy, and the conclusion would not
transfer to the job you actually run.

    python src/depth_check.py --data data/features_blitz_full.parquet \
        --out reports/depth_check.txt --n-positions 500 --depths 8,10,12

Reports, per depth pair:
  - agreement on the `blunder_available` flag, WITH the flag's prevalence and
    Cohen's kappa. Raw agreement on a lopsided boolean is the same mistake as
    reporting accuracy on a 4% base rate, which SPEC.md forbids: if the flag
    is true 92% of the time then 92% agreement is what you get for free.
  - Pearson and Spearman correlation on frac_blunder_moves.
  - seconds per position, and what the full run would cost.
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_children import _eval_position, _init, find_engine  # noqa: E402

_LINES: list[str] = []


def p(line: str = "") -> None:
    print(line, flush=True)
    _LINES.append(line)


def rule(title: str = "") -> None:
    p()
    p("=" * 74)
    if title:
        p(title)
        p("=" * 74)


def table(headers, rows) -> None:
    body = [[str(c) for c in r] for r in rows]
    w = [len(h) for h in headers]
    for r in body:
        for i, c in enumerate(r):
            w[i] = max(w[i], len(c))
    fmt = lambda cs: "  ".join(  # noqa: E731
        c.ljust(w[i]) if i == 0 else c.rjust(w[i]) for i, c in enumerate(cs)).rstrip()
    p(fmt(headers))
    p("  ".join("-" * x for x in w))
    for r in body:
        p(fmt(r))


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa: agreement corrected for agreement by chance.

    1.0 = perfect, 0.0 = no better than two independent coins with the same
    bias. This is the number that survives a lopsided flag.
    """
    a, b = a.astype(bool), b.astype(bool)
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return 1.0 if pe >= 1.0 else (po - pe) / (1 - pe)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def run_depth(jobs, engine: str, depth: int, workers: int) -> tuple[dict, float]:
    t0 = time.time()
    out = {}
    with Pool(workers, initializer=_init, initargs=(engine, depth)) as pool:
        for i, res in enumerate(pool.imap_unordered(_eval_position, jobs, 8), 1):
            if res is not None:
                out[res["row_id"]] = res
            if i % 100 == 0:
                p(f"    depth {depth}: {i}/{len(jobs)} positions "
                  f"({time.time()-t0:.0f}s)")
    return out, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None, help="tee the report here")
    ap.add_argument("--n-positions", type=int, default=500)
    ap.add_argument("--depths", default="8,10,12")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--engine", default="stockfish")
    ap.add_argument("--threshold", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full-run-positions", type=int, default=40000,
                    help="used only to extrapolate the full eval_children cost")
    ap.add_argument("--agree-tol", type=float, default=0.97,
                    help="flag agreement vs the deepest depth needed to accept "
                         "a cheaper one")
    ap.add_argument("--kappa-tol", type=float, default=0.85)
    args = ap.parse_args()

    depths = sorted({int(d) for d in args.depths.split(",") if d.strip()})
    if len(depths) < 2:
        print("FATAL: need at least two --depths to compare")
        return 1

    engine = find_engine(args.engine)
    if engine is None:
        print(f"FATAL: could not resolve engine '{args.engine}'.")
        return 1

    rule(f"DEPTH SENSITIVITY: {args.data}")
    p(f"engine: {engine}")
    p(f"depths: {depths} | threshold {args.threshold:g} win% | "
      f"{args.workers} workers")

    df = pd.read_parquet(args.data, columns=["fen", "mover_elo", "label_valid",
                                             "winpct_before", "tc_category"])
    df = df[df.label_valid].reset_index(drop=True)
    df["row_id"] = df.index

    # Same stratification as eval_children, for the same reason: an unstratified
    # sample is mostly 1500-rated middlegames and would not tell you how depth
    # behaves in the sharp positions where it matters.
    strata = (pd.cut(df.mover_elo, [0, 1200, 1600, 2000, 4000], labels=False)
              .astype(str) + "_" + df.tc_category.astype(str))
    per = max(1, args.n_positions // max(strata.nunique(), 1))
    sample = (df.assign(_s=strata).groupby("_s", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), per), random_state=args.seed))
              .head(args.n_positions))
    p(f"{len(df):,} valid rows -> {len(sample):,} sampled across "
      f"{strata.nunique()} strata")

    jobs = [(int(r.row_id), r.fen, float(r.winpct_before), args.threshold)
            for r in sample.itertuples()]

    results, timings = {}, {}
    for d in depths:
        p(f"\n  running depth {d} over {len(jobs)} positions ...")
        results[d], timings[d] = run_depth(jobs, engine, d, args.workers)
        p(f"  depth {d} done in {timings[d]:.0f}s")

    # Only positions every depth managed (engines occasionally die mid-run).
    common = sorted(set.intersection(*(set(r) for r in results.values())))
    if not common:
        p("FATAL: no position completed at every depth.")
        return 1
    p(f"\n{len(common):,} positions completed at every depth "
      f"({len(common)/len(jobs):.1%} of the sample)")

    flags = {d: np.array([results[d][i]["blunder_available"] for i in common])
             for d in depths}
    fracs = {d: np.array([results[d][i]["frac_blunder_moves"] for i in common])
             for d in depths}
    nmoves = np.array([results[depths[0]][i]["n_moves"] for i in common])

    rule("1. COST")
    n_full = args.full_run_positions
    table(["depth", "wall clock", "s / position", f"est. {n_full:,} positions",
           "vs deepest"],
          [[d, f"{timings[d]:.0f}s", f"{timings[d]/len(jobs):.3f}",
            f"{timings[d]/len(jobs)*n_full/3600:.1f} h",
            f"{timings[d]/timings[depths[-1]]:.2f}x"] for d in depths])
    p()
    p(f"mean legal moves per position: {nmoves.mean():.1f} -- that is how many "
      f"engine calls\neach position costs, which is why depth is expensive here.")

    rule("2. AGREEMENT ON `blunder_available`")
    p("prevalence of the flag at each depth (read this FIRST):")
    p()
    table(["depth", "flag true", "n"],
          [[d, f"{flags[d].mean():.1%}", f"{len(common):,}"] for d in depths])
    p()
    prev = flags[depths[-1]].mean()
    if prev > 0.85 or prev < 0.15:
        p(f"  NOTE: the flag is {prev:.0%} at the deepest setting, so it is "
          f"lopsided.\n  Raw agreement is inflated by that alone -- a constant "
          f"predictor would score\n  {max(prev, 1-prev):.0%}. Judge on kappa, "
          f"not on agreement.")
    p()
    rows = []
    for i, d1 in enumerate(depths):
        for d2 in depths[i+1:]:
            both = int((flags[d1] & flags[d2]).sum())
            neither = int((~flags[d1] & ~flags[d2]).sum())
            only1 = int((flags[d1] & ~flags[d2]).sum())
            only2 = int((~flags[d1] & flags[d2]).sum())
            rows.append([f"d{d1} vs d{d2}",
                         f"{(flags[d1]==flags[d2]).mean():.2%}",
                         f"{kappa(flags[d1], flags[d2]):.3f}",
                         f"{both:,}", f"{neither:,}", f"{only1:,}", f"{only2:,}"])
    table(["pair", "agreement", "kappa", "both", "neither",
           "only 1st", "only 2nd"], rows)

    rule("3. CORRELATION ON `frac_blunder_moves`")
    rows = []
    for i, d1 in enumerate(depths):
        for d2 in depths[i+1:]:
            x, y = fracs[d1], fracs[d2]
            pear = float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else float("nan")
            rows.append([f"d{d1} vs d{d2}", f"{pear:.4f}", f"{spearman(x, y):.4f}",
                         f"{np.abs(x-y).mean():.4f}", f"{(y-x).mean():+.4f}"])
    table(["pair", "pearson", "spearman", "mean |diff|", "mean bias (2nd-1st)"],
          rows)
    p()
    table(["depth", "mean frac_blunder_moves", "median", "p90"],
          [[d, f"{fracs[d].mean():.4f}", f"{np.median(fracs[d]):.4f}",
            f"{np.percentile(fracs[d],90):.4f}"] for d in depths])

    rule("RECOMMENDATION")
    deepest = depths[-1]
    ok = []
    for d in depths[:-1]:
        agr = float((flags[d] == flags[deepest]).mean())
        k = kappa(flags[d], flags[deepest])
        speed = timings[deepest] / timings[d]
        passes = agr >= args.agree_tol and k >= args.kappa_tol
        ok.append((d, agr, k, speed, passes))
        p(f"  depth {d} vs {deepest}: agreement {agr:.2%} "
          f"(need {args.agree_tol:.0%}), kappa {k:.3f} "
          f"(need {args.kappa_tol:.2f}), {speed:.2f}x faster -> "
          f"{'ACCEPT' if passes else 'REJECT'}")
    p()
    winners = [o for o in ok if o[4]]
    if winners:
        d, agr, k, speed, _ = winners[0]
        hours = timings[d] / len(jobs) * n_full / 3600
        p(f"USE --depth {d}. It reproduces depth {deepest}'s answer "
          f"({agr:.1%} agreement,\nkappa {k:.3f}) at {speed:.2f}x the speed, "
          f"putting the {n_full:,}-position run at\nabout {hours:.1f} hours.")
    else:
        hours = timings[deepest] / len(jobs) * n_full / 3600
        p(f"USE --depth {deepest}. No cheaper depth reproduced it within "
          f"tolerance, so the\nsaving is not real -- a faster run that "
          f"disagrees is not a faster run, it is a\ndifferent experiment. "
          f"Budget about {hours:.1f} hours.")
    p()
    p("Caveat: this measures agreement on THIS sample of positions. It says")
    p("nothing about whether the deepest depth here is itself deep enough in an")
    p("absolute sense; it only says where the answer stops moving.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(_LINES) + "\n")
        print(f"\nreport written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
