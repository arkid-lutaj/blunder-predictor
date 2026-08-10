#!/usr/bin/env python3
"""
Stream a Lichess monthly PGN dump and emit one row per (position, move played)
with a win-probability-drop blunder label.

Deliberately does NOT compute position features. It emits the FEN before the
move plus everything needed for the label and the context variables. Features
go in a second pass over the parquet so you can iterate on them without
re-reading 30 GB.

Usage:
    python parse_lichess.py lichess_db_standard_rated_2026-06.pgn.zst \
        --out positions.parquet --max-games 200000

    # sanity check on a plain .pgn
    python parse_lichess.py sample.pgn --out sample.parquet
"""

import argparse
import io
import math
import os
import re
import sys
import time

import chess
import chess.pgn
import pandas as pd

# ---------------------------------------------------------------------------
# Label
# ---------------------------------------------------------------------------

# Lichess' own logistic fit mapping centipawns to win probability.
# Confirm against https://lichess.org/page/accuracy before you quote it.
WIN_K = 0.00368208
CP_CLAMP = 1000  # Lichess clamps eval to +/- 10 pawns before converting


def win_pct(cp: float) -> float:
    """Centipawns (from the mover's point of view) -> win% in [0, 100]."""
    cp = max(-CP_CLAMP, min(CP_CLAMP, cp))
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-WIN_K * cp)) - 1.0)


# ---------------------------------------------------------------------------
# Header handling
# ---------------------------------------------------------------------------

TC_RE = re.compile(r"^(\d+)\+(\d+)$")
SITE_RE = re.compile(r"lichess\.org/(\w+)")


def tc_category(base: int, inc: int) -> str:
    """Lichess' speed buckets: estimated duration = base + 40 * increment."""
    total = base + 40 * inc
    if total < 29:
        return "ultrabullet"
    if total < 179:
        return "bullet"
    if total < 479:
        return "blitz"
    if total < 1499:
        return "rapid"
    return "classical"


def parse_headers(block: str) -> dict | None:
    """Cheap regex header scan. Returns None if the game fails the filters."""
    h = dict(re.findall(r'^\[(\w+) "(.*)"\]$', block, flags=re.M))

    tc = h.get("TimeControl", "-")
    m = TC_RE.match(tc)
    if not m:
        return None  # correspondence or malformed
    base, inc = int(m.group(1)), int(m.group(2))

    try:
        white_elo = int(h["WhiteElo"])
        black_elo = int(h["BlackElo"])
    except (KeyError, ValueError):
        return None  # provisional or unrated

    site = SITE_RE.search(h.get("Site", ""))

    return {
        "game_id": site.group(1) if site else None,
        "white": h.get("White"),
        "black": h.get("Black"),
        "white_elo": white_elo,
        "black_elo": black_elo,
        "tc_base": base,
        "tc_inc": inc,
        "tc_category": tc_category(base, inc),
        "eco": h.get("ECO"),
        "result": h.get("Result"),
        "termination": h.get("Termination"),
        "date": h.get("UTCDate"),
    }


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def iter_game_blocks(path: str):
    """Yield raw text of one game at a time, streaming, without extracting.

    Pass "-" to read a .zst stream from stdin, e.g.
        curl -sL <url> | python parse_lichess.py - --out out.parquet
    which lets --max-games stop you after a few GB instead of downloading 30.
    """
    if path == "-" or path.endswith(".zst"):
        import zstandard as zstd

        fh = sys.stdin.buffer if path == "-" else open(path, "rb")
        # Lichess compresses with long-distance matching; the default 8 MB
        # window raises "frame requires too much memory".
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        reader = dctx.stream_reader(fh)
        stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    else:
        stream = open(path, "r", encoding="utf-8", errors="replace")

    buf: list[str] = []
    seen_moves = False
    for line in stream:
        if line.startswith("[Event ") and seen_moves:
            yield "".join(buf)
            buf, seen_moves = [], False
        if line.strip() and not line.startswith("["):
            seen_moves = True
        buf.append(line)
    if buf:
        yield "".join(buf)
    stream.close()


# ---------------------------------------------------------------------------
# Per-game extraction
# ---------------------------------------------------------------------------


def score_to_cp(score, mate_cp: int = 1200):
    """PovScore(white POV) -> (cp, mate_in). cp is None-safe for mate scores."""
    if score is None:
        return None, None
    w = score.white()
    if w.is_mate():
        m = w.mate()
        return (mate_cp if m > 0 else -mate_cp), m
    return w.score(), None


def extract_game(block: str, meta: dict, blunder_threshold: float):
    """Yield one row dict per ply that has a usable before/after eval pair."""
    game = chess.pgn.read_game(io.StringIO(block))
    if game is None:
        return

    nodes = list(game.mainline())
    if not nodes:
        return

    # evals[i] / clocks[i] describe the position AFTER ply i.
    evals = [n.eval() for n in nodes]
    clocks = [n.clock() for n in nodes]
    root_eval = game.eval()

    board = game.board()
    for i, node in enumerate(nodes):
        before = root_eval if i == 0 else evals[i - 1]
        after = evals[i]

        if before is None or after is None:
            board.push(node.move)
            continue

        mover = board.turn  # True = white
        cp_b, mate_b = score_to_cp(before)
        cp_a, mate_a = score_to_cp(after)

        # Flip to the mover's point of view.
        sign = 1 if mover == chess.WHITE else -1
        wp_b = win_pct(sign * cp_b)
        wp_a = win_pct(sign * cp_a)
        drop = wp_b - wp_a

        # cp is undefined inside forced-mate scores; keep the row, flag it.
        label_valid = mate_b is None and mate_a is None

        # The mover's clock going into this decision is their clock after
        # their own previous move, i.e. two plies back.
        clk_before = clocks[i - 2] if i >= 2 and clocks[i - 2] is not None else (
            float(meta["tc_base"]) if i < 2 else None
        )
        clk_after = clocks[i]
        if clk_before is not None and clk_after is not None:
            time_spent = clk_before - clk_after + meta["tc_inc"]
        else:
            time_spent = None

        yield {
            "game_id": meta["game_id"],
            "ply": i,
            "fen": board.fen(),
            "move": node.move.uci(),
            "mover_is_white": bool(mover),
            "mover": meta["white"] if mover else meta["black"],
            "mover_elo": meta["white_elo"] if mover else meta["black_elo"],
            "opp_elo": meta["black_elo"] if mover else meta["white_elo"],
            "cp_before": cp_b,
            "cp_after": cp_a,
            "mate_before": mate_b,
            "mate_after": mate_a,
            "winpct_before": wp_b,
            "winpct_after": wp_a,
            "win_drop": drop,
            "blunder": bool(drop > blunder_threshold) if label_valid else None,
            "label_valid": label_valid,
            "clk_before": clk_before,
            "clk_after": clk_after,
            "time_spent": time_spent,
            "tc_base": meta["tc_base"],
            "tc_inc": meta["tc_inc"],
            "tc_category": meta["tc_category"],
            "eco": meta["eco"],
            "result": meta["result"],
            "termination": meta["termination"],
            "date": meta["date"],
        }

        board.push(node.move)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--out", default="positions.parquet")
    ap.add_argument("--max-games", type=int, default=None,
                    help="stop after this many ACCEPTED (eval-annotated) games")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=20.0,
                    help="win%% drop that counts as a blunder")
    ap.add_argument("--categories", default="blitz,rapid",
                    help="comma-separated tc buckets to keep, or 'all'")
    ap.add_argument("--every", type=int, default=1,
                    help="keep every Kth eligible game. The dump is in "
                         "chronological order, so --max-games alone samples "
                         "only the first days of the month.")
    ap.add_argument("--chunk", type=int, default=250_000,
                    help="rows buffered before flushing a parquet part")
    args = ap.parse_args()

    keep = None if args.categories == "all" else set(args.categories.split(","))

    stats = dict(seen=0, no_eval=0, bad_header=0, wrong_tc=0, eligible=0, accepted=0, rows=0)
    rows: list[dict] = []
    parts: list[str] = []
    t0 = time.time()

    def flush():
        nonlocal rows
        if not rows:
            return
        p = f"{args.out}.part{len(parts):04d}"
        pd.DataFrame(rows).to_parquet(p, index=False)
        parts.append(p)
        rows = []

    for block in iter_game_blocks(args.pgn):
        stats["seen"] += 1

        # The cheap filter that makes this tractable: skip the ~85% of games
        # with no engine annotation before python-chess ever sees them.
        if "%eval" not in block:
            stats["no_eval"] += 1
            continue

        meta = parse_headers(block)
        if meta is None:
            stats["bad_header"] += 1
            continue
        if keep is not None and meta["tc_category"] not in keep:
            stats["wrong_tc"] += 1
            continue

        stats["eligible"] += 1
        if args.every > 1 and stats["eligible"] % args.every:
            continue

        stats["accepted"] += 1
        for row in extract_game(block, meta, args.threshold):
            rows.append(row)
            stats["rows"] += 1

        if len(rows) >= args.chunk:
            flush()

        if stats["seen"] % 200_000 == 0:
            el = time.time() - t0
            print(f"  {stats['seen']:,} games scanned | {stats['accepted']:,} kept "
                  f"| {stats['rows']:,} rows | {stats['seen']/el:,.0f} games/s",
                  file=sys.stderr)

        if args.max_games and stats["accepted"] >= args.max_games:
            break
        if args.max_rows and stats["rows"] >= args.max_rows:
            break

    flush()

    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) \
        if parts else pd.DataFrame()
    df.to_parquet(args.out, index=False)
    for p in parts:
        os.remove(p)

    print("\n".join(f"{k:>12}: {v:,}" for k, v in stats.items()), file=sys.stderr)
    if len(df):
        valid = df[df.label_valid]
        print(f"{'base rate':>12}: {valid.blunder.mean():.4f} "
              f"({valid.blunder.sum():,} / {len(valid):,} labelled plies)",
              file=sys.stderr)
        print(f"{'mate-scored':>12}: {(~df.label_valid).mean():.4f} of rows",
              file=sys.stderr)
        print(f"{'wrote':>12}: {args.out} ({len(df):,} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()