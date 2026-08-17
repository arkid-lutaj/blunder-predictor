#!/usr/bin/env python3
"""
Cut the committed sample data, so a fresh clone can run the whole pipeline.

Nobody downloads 28 GB to look at a repo. Without a committed sample every
script here is unrunnable by anyone but me, which makes the code unreviewable
and the CI impossible. Two outputs:

  data/sample_positions.parquet   stratified rows, preserving the base rate, so
                                  features / splits / baselines / train all run
  data/sample_games.pgn           annotated games straight from the dump, so the
                                  PARSER is covered too. Sampling the parquet
                                  cannot test the parser, and the parser is
                                  where the nastiest bugs in this project were.

Both are sized for git. Stratification is across rating band and time control
because the base rate moves with both, and a sample that quietly shifted it
would make the CI assertion on base rate meaningless.

Usage:
    python make_sample.py --data data/positions_2026-06_blitz.parquet \
        --pgn data/lichess_db_standard_rated_2026-06.pgn.zst \
        --out-parquet data/sample_positions.parquet \
        --out-pgn data/sample_games.pgn
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_lichess import iter_game_blocks, parse_headers  # noqa: E402

RATING_BANDS = [0, 1200, 1600, 2000, 4000]


def sample_positions(path: str, out: str, n: int, seed: int) -> None:
    df = pd.read_parquet(path)
    print(f"{len(df):,} rows in {os.path.basename(path)}")
    base_all = df.blunder[df.label_valid].mean()

    # Sample whole GAMES, not rows. A game split across the boundary would hand
    # make_splits a player with a partial game, which the real pipeline never
    # produces, so testing against it would be testing a fiction. Sampling rows
    # and then widening to their games is the same mistake in reverse: 20k rows
    # touched 17,809 distinct games and pulling those games back gave 1.36M rows
    # and a 38 MB file.
    per_game = df.groupby("game_id").first()
    band = pd.cut(per_game.mover_elo, RATING_BANDS, labels=False).fillna(0)
    strata = band.astype(str) + "|" + per_game.tc_category.astype(str)
    rows_per_game = len(df) / max(df.game_id.nunique(), 1)
    n_games = max(1, int(round(n / max(rows_per_game, 1))))
    per = max(1, n_games // max(strata.nunique(), 1))
    picked = per_game.groupby(strata, group_keys=False).apply(
        lambda g: g.sample(min(len(g), per), random_state=seed))
    out_df = df[df.game_id.isin(set(picked.index))].reset_index(drop=True)

    base_s = out_df.blunder[out_df.label_valid].mean()
    out_df.to_parquet(out, index=False, compression="zstd")
    mb = os.path.getsize(out) / 1e6
    print(f"wrote {out}")
    print(f"  {len(out_df):,} rows, {out_df.game_id.nunique():,} games, "
          f"{mb:.1f} MB")
    print(f"  base rate {base_s:.4%} against {base_all:.4%} in the full file "
          f"(drift {abs(base_s-base_all)*100:+.3f}pp)")
    if not 0.02 <= base_s <= 0.05:
        print("  WARNING: sample base rate is outside the 2-5% band the CI "
              "asserts on")
    return base_s


def sample_pgn(pgn: str, out: str, n_games: int, categories: set) -> None:
    """Copy the first n annotated games of the wanted categories, verbatim.

    Verbatim matters: the point is to exercise the real parser against real
    text, including the header quirks and the eval comment format. A
    reconstructed PGN would only test my idea of what Lichess writes.
    """
    kept = 0
    seen = 0
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for block in iter_game_blocks(pgn):
            seen += 1
            if "[%eval" not in block:
                continue
            meta = parse_headers(block)
            if not meta or meta.get("tc_category") not in categories:
                continue
            fh.write(block.strip() + "\n\n")
            kept += 1
            if kept >= n_games:
                break
            if kept % 100 == 0:
                print(f"  {kept}/{n_games} annotated games "
                      f"({seen:,} scanned)", flush=True)
    mb = os.path.getsize(out) / 1e6
    print(f"wrote {out}")
    print(f"  {kept} annotated games from {seen:,} scanned, {mb:.1f} MB")
    if kept < n_games:
        print(f"  WARNING: only found {kept} of {n_games} requested")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="full positions parquet")
    ap.add_argument("--pgn", help="raw dump, .zst or plain")
    ap.add_argument("--out-parquet", default="data/sample_positions.parquet")
    ap.add_argument("--out-pgn", default="data/sample_games.pgn")
    ap.add_argument("--n-rows", type=int, default=20000)
    ap.add_argument("--n-games", type=int, default=500)
    ap.add_argument("--categories", default="blitz,rapid")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.data and not args.pgn:
        ap.error("give --data, --pgn, or both")

    if args.data:
        os.makedirs(os.path.dirname(args.out_parquet) or ".", exist_ok=True)
        sample_positions(args.data, args.out_parquet, args.n_rows, args.seed)
    if args.pgn:
        print()
        os.makedirs(os.path.dirname(args.out_pgn) or ".", exist_ok=True)
        sample_pgn(args.pgn, args.out_pgn, args.n_games,
                   set(args.categories.split(",")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
