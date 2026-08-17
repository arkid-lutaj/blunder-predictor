#!/usr/bin/env python3
"""
Evaluate EVERY legal move in a sample of positions with Stockfish.

Two consumers:

1. The decomposition (Phase 7).
       P(blunder) = P(a blunder is available) x P(player picks one | available)
   The first term is position geometry and has nothing to do with rating. If
   you do not separate them, your "rating effect" is partly just "weak players
   reach messier positions".

2. The challenge game (Phase 11). To tell a player whether the move they chose
   was a blunder, you need the eval of that specific move. That is per-move
   detail, so this script stores it with --per-move rather than only the
   aggregates the decomposition needs.

Checkpoints every --checkpoint positions. --resume picks up where it stopped,
so thermal throttling or a closed laptop costs wall-clock, never work.

Usage:
    python eval_children.py --data data/features_blitz.parquet \
        --out data/children_40k.parquet \
        --n-positions 40000 --depth 8 --workers 7 --per-move
"""

import argparse
import glob
import math
import os
import shutil
import sys
import time
from multiprocessing import Pool

import chess
import chess.engine
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# The centipawn -> win%% logistic below is LICHESS'S, not this
# project's: the constant and the form are documented at
# https://lichess.org/page/accuracy. See THIRD_PARTY.md.
WIN_K = 0.00368208
CP_CLAMP = 1000
MATE_CP = 1200

_ENGINE = None
_DEPTH = 8
_PATH = "stockfish"


_TTY = sys.stdout.isatty()


def say(msg: str = "", transient: bool = False) -> None:
    """Progress that survives being redirected to a log.

    Two failure modes this exists to avoid, both already paid for once on this
    project (see FINDINGS.md):

    - no flush: Git Bash is not a tty, so Python block-buffers stdout and a
      multi-hour job shows nothing at all until it ends.
    - a bare `end="\\r"`: lovely in a terminal, but a redirected log becomes one
      single enormous line that `tail -f` cannot usefully display.

    So carriage-return updates happen only when stdout really is a terminal;
    otherwise every update is its own flushed line.
    """
    if transient and _TTY:
        print(msg, end="\r", flush=True)
    else:
        print(msg, flush=True)


def win_pct(cp: float) -> float:
    cp = max(-CP_CLAMP, min(CP_CLAMP, cp))
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-WIN_K * cp)) - 1.0)


def find_engine(name: str = "stockfish") -> str | None:
    """Resolve a Stockfish binary, PATH or not.

    On this machine `shutil.which("stockfish")` is None: the Windows build is
    `stockfish-windows-x86-64-avx2.exe`, so the bare name never matches even
    though the binary is sitting in tools/. A multi-hour overnight job that
    dies one second after launch on a name lookup is the worst possible way to
    find that out, so look in the obvious places before giving up.
    """
    if os.path.sep in name or (os.path.altsep and os.path.altsep in name):
        return name if os.path.exists(name) else None
    direct = shutil.which(name)
    if direct:
        return direct
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    patterns = [
        os.path.join(here, "tools", "stockfish", "stockfish*.exe"),
        os.path.join(here, "tools", "stockfish", "stockfish*"),
        os.path.join(here, "tools", "stockfish*"),
        os.path.expanduser("~/AppData/Local/Microsoft/WinGet/Packages/"
                           "Stockfish*/stockfish/stockfish*.exe"),
    ]
    for pat in patterns:
        hits = sorted(g for g in glob.glob(pat)
                      if os.path.isfile(g) and not g.endswith(
                          (".md", ".cff", ".txt")))
        if hits:
            return hits[0]
    return None


def _init(engine_path: str, depth: int):
    """One engine per worker process, single-threaded so workers don't fight."""
    global _ENGINE, _DEPTH, _PATH
    _PATH, _DEPTH = engine_path, depth
    _ENGINE = chess.engine.SimpleEngine.popen_uci(engine_path)
    _ENGINE.configure({"Threads": 1, "Hash": 64})


def _score_cp(pov) -> float:
    if pov.is_mate():
        return MATE_CP if pov.mate() > 0 else -MATE_CP
    return pov.score()


def _eval_position(job):
    """job = (row_id, fen, winpct_before, threshold). Returns one dict."""
    global _ENGINE
    row_id, fen, wp_before, threshold = job
    board = chess.Board(fen)
    mover = board.turn
    moves = list(board.legal_moves)
    if not moves:
        return None

    try:
        results = []
        for mv in moves:
            board.push(mv)
            info = _ENGINE.analyse(board, chess.engine.Limit(depth=_DEPTH))
            board.pop()
            cp = _score_cp(info["score"].pov(mover))
            results.append((mv.uci(), win_pct(cp)))
    except (chess.engine.EngineTerminatedError, BrokenPipeError):
        # Engines do die. Restart and drop this position rather than the run.
        try:
            _ENGINE = chess.engine.SimpleEngine.popen_uci(_PATH)
            _ENGINE.configure({"Threads": 1, "Hash": 64})
        except Exception:
            pass
        return None

    wins = np.array([w for _, w in results])
    # Reference point: the best available move. Using the pre-move eval instead
    # would fold engine disagreement (Lichess' annotation depth vs ours) into
    # the "is a blunder available" question.
    best = wins.max()
    drops = best - wins
    is_blunder = drops > threshold

    out = {
        "row_id": row_id,
        "fen": fen,
        "n_moves": len(results),
        "best_child_win": float(best),
        "worst_child_win": float(wins.min()),
        "mean_child_win": float(wins.mean()),
        "std_child_win": float(wins.std()),
        "n_blunder_moves": int(is_blunder.sum()),
        "frac_blunder_moves": float(is_blunder.mean()),
        "blunder_available": bool(is_blunder.any()),
        "winpct_before": wp_before,
    }
    out["_moves"] = [{"uci": u, "win": round(float(w), 2),
                      "drop": round(float(best - w), 2),
                      "blunder": bool(best - w > threshold)}
                     for u, w in results]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-positions", type=int, default=40000)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--engine", default="stockfish")
    ap.add_argument("--threshold", type=float, default=20.0)
    ap.add_argument("--checkpoint", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--per-move", action="store_true",
                    help="also write moves.jsonl with every legal move's eval. "
                         "Required by build_challenge.py.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--yes", action="store_true",
                    help="skip the 10s pause when the ETA exceeds --warn-hours")
    ap.add_argument("--warn-hours", type=float, default=4.0,
                    help="pause 10s if the projected runtime exceeds this. "
                         "It never aborts; it only gives you a window.")
    args = ap.parse_args()

    engine = find_engine(args.engine)
    if engine is None:
        print(f"FATAL: could not resolve engine '{args.engine}'. Looked on "
              f"PATH, in tools/stockfish/ and in the winget package dir. "
              f"Pass --engine with an explicit path to the binary.")
        return 1
    if engine != args.engine:
        print(f"engine: {engine}")
    args.engine = engine

    # Only the columns the sampler and the workers touch. The full features
    # parquet is 73 columns x 6.7M rows and would sit in RAM for the entire
    # multi-hour run alongside seven Stockfish processes, for the sake of
    # sampling a few hundred thousand FENs out of it.
    need = ["fen", "mover_elo", "label_valid", "winpct_before", "tc_category"]
    have = set(pq.ParquetFile(args.data).schema_arrow.names)
    missing = [c for c in need if c not in have and c != "tc_category"]
    if missing:
        print(f"FATAL: {args.data} is missing column(s): {', '.join(missing)}")
        return 1
    df = pd.read_parquet(args.data, columns=[c for c in need if c in have])
    df = df[df.label_valid].reset_index(drop=True)
    # row_id is POSITIONAL within label_valid rows of THIS file. Anything that
    # joins on it later (build_challenge.py) must be given the same --features
    # parquet, or it will silently line up against different positions.
    df["row_id"] = df.index

    # Stratify so the sample is not all 1500-rated blitz middlegames.
    strata = (pd.cut(df.mover_elo, [0, 1200, 1600, 2000, 4000], labels=False)
              .astype(str) + "_" + df.get("tc_category", "x").astype(str))
    per = max(1, args.n_positions // max(strata.nunique(), 1))
    sample = (df.assign(_s=strata).groupby("_s", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), per), random_state=args.seed))
              .head(args.n_positions))
    say(f"{len(df):,} valid rows -> {len(sample):,} sampled across "
          f"{strata.nunique()} strata")

    ckpt = args.out + ".ckpt.parquet"
    done: set[int] = set()
    prior: list[dict] = []
    resumed = args.resume and os.path.exists(ckpt)
    if args.resume and not resumed:
        say("  --resume given but no checkpoint found; starting from scratch.")
    if resumed:
        prev = pd.read_parquet(ckpt)
        done = set(prev.row_id.tolist())
        prior = prev.to_dict("records")
        say(f"resuming: {len(done):,} positions already done")

    jobs = [(int(r.row_id), r.fen, float(r.winpct_before), args.threshold)
            for r in sample.itertuples() if int(r.row_id) not in done]
    if not jobs:
        say("nothing left to do")
        return 0

    moves_path = os.path.splitext(args.out)[0] + "_moves.jsonl"
    # Append ONLY when a checkpoint was actually loaded. `--resume` with no
    # checkpoint (killed before the first one) re-runs every position, so
    # appending would silently duplicate every row already in the file while
    # the parquet stayed correct -- a desync nothing downstream would notice.
    if args.per_move and not resumed and os.path.exists(moves_path):
        say(f"  overwriting existing {moves_path} "
            f"({os.path.getsize(moves_path)/1e6:.1f} MB) -- starting clean")
    mfh = open(moves_path, "a" if resumed else "w") if args.per_move else None

    results, t0 = list(prior), time.time()
    with Pool(args.workers, initializer=_init,
              initargs=(args.engine, args.depth)) as pool:
        for i, res in enumerate(pool.imap_unordered(_eval_position, jobs, 16), 1):
            if res is None:
                continue
            if mfh is not None:
                import json
                mfh.write(json.dumps({"row_id": res["row_id"], "fen": res["fen"],
                                      "moves": res["_moves"]}) + "\n")
            res.pop("_moves", None)
            results.append(res)

            if i == 100:
                rate = i / (time.time() - t0)
                eta = (len(jobs) - i) / rate / 3600
                say(f"\n  {rate:.1f} positions/s -> ETA {eta:.1f} h "
                    f"for the remaining {len(jobs)-i:,}")
                # Report, do not order. The old text said "Ctrl-C now and lower
                # --n-positions or --depth", which reads as an instruction --
                # and got an intentional 4.0 h overnight run killed at position
                # 100 by someone correctly doing as they were told. A long run
                # is the plan here, not an accident, so say what will happen and
                # let the operator decide.
                if not args.yes and eta > args.warn_hours:
                    say(f"  Longer than --warn-hours ({args.warn_hours:g} h). "
                        f"CONTINUING ANYWAY in 10s.")
                    say(f"  Interrupt only if you did not intend a run this "
                        f"long. It checkpoints every {args.checkpoint:,} "
                        f"positions and --resume is safe.")
                    time.sleep(10)
            if i % args.checkpoint == 0:
                tck = time.time()
                pd.DataFrame(results).to_parquet(ckpt, index=False)
                el = time.time() - t0
                say(f"  {i:,}/{len(jobs):,} | {i/el:.1f} pos/s | "
                    f"{(len(jobs)-i)/(i/el)/60:.0f} min left | "
                    f"ckpt {time.time()-tck:.1f}s", transient=True)

    if mfh is not None:
        mfh.close()

    out = pd.DataFrame(results)
    out.to_parquet(args.out, index=False)
    if os.path.exists(ckpt):
        os.remove(ckpt)

    print(f"\n\n{len(out):,} positions evaluated at depth {args.depth}")
    say(f"  blunder available in       {out.blunder_available.mean():.2%}")
    say(f"  mean legal moves           {out.n_moves.mean():.1f}")
    say(f"  mean frac blundering moves {out.frac_blunder_moves.mean():.2%}")
    say(f"wrote {args.out}" + (f" and {moves_path}" if args.per_move else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
