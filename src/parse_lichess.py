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
import json
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

# LICHESS'S own logistic fit mapping centipawns to win probability, not
# this project's. Constant and form documented at
# https://lichess.org/page/accuracy. See THIRD_PARTY.md.
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


# Byte-level filters. These run on EVERY annotated game (~2.2M for a full
# month), so they must not touch the full header regex. Same accept/reject
# semantics as parse_headers: TimeControl must be "N+M" and both Elos must be
# integers, so a provisional "?" rating fails here exactly as it did there.
TC_B = re.compile(rb'\[TimeControl "(\d+)\+(\d+)"\]')
WE_B = re.compile(rb'\[WhiteElo "(\d+)"\]')
BE_B = re.compile(rb'\[BlackElo "(\d+)"\]')
SEP = b"\n[Event "


ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"  # 0xFD2FB528, a data frame


def is_zstd(head: bytes) -> bool:
    """True for any zstd frame, data or skippable.

    Sniffing ONLY the data-frame magic is a real bug and it bit hard. Lichess'
    HTTP file is written by pzstd, which prefixes each compressed frame with a
    12-byte SKIPPABLE frame holding that frame's size so the archive can be
    decompressed in parallel. Such a file begins 50 2a 4d 18, not 28 b5 2f fd.

    The old sniff saw no match, concluded "plain PGN text", and handed raw
    compressed bytes to the block splitter. It then searched for b"\\n[Event "
    inside binary, never found it, never yielded a block, and grew its buffer
    forever. Symptom: "seen: 0" no matter how long you wait. Deterministic, so
    waiting could never have helped.

    Skippable magic is a RANGE, 0x184D2A50 to 0x184D2A5F, so match all sixteen.
    """
    if len(head) < 4:
        return False
    if head[:4] == ZSTD_MAGIC:
        return True
    return head[1:4] == b"\x2a\x4d\x18" and 0x50 <= head[0] <= 0x5F


class _Prepend(io.RawIOBase):
    """Re-attach bytes already consumed while sniffing the stream header."""

    def __init__(self, head: bytes, rest):
        self._head, self._rest = head, rest

    def readable(self) -> bool:
        return True

    def readinto(self, buf) -> int:
        if self._head:
            n = min(len(buf), len(self._head))
            buf[:n], self._head = self._head[:n], self._head[n:]
            return n
        return self._rest.readinto(buf)


def open_stream(path: str):
    """Return a binary stream of PGN text, transparently un-zstd-ing if needed.

    Sniffs the magic number rather than trusting the extension, so all of these
    work and can be mixed freely:
        parse_lichess.py file.pgn.zst          in-process decompression
        curl -sL URL | parse_lichess.py -      compressed stream over stdin
        zstd -dc file.pgn.zst | ... -          decompression on its own core
    """
    raw = sys.stdin.buffer if path == "-" else open(path, "rb")
    head = raw.read(4)
    stream = io.BufferedReader(_Prepend(head, raw))
    if is_zstd(head):
        import zstandard as zstd
        # Lichess compresses with long-distance matching; the default 8 MB
        # window raises "frame requires too much memory".
        return zstd.ZstdDecompressor(max_window_size=2**31).stream_reader(stream)
    return stream


def quick_filter(block: bytes):
    """Cheap accept/reject before anything expensive. None means reject."""
    tc = TC_B.search(block)
    if tc is None:
        return None
    we = WE_B.search(block)
    if we is None:
        return None
    be = BE_B.search(block)
    if be is None:
        return None
    base, inc = int(tc.group(1)), int(tc.group(2))
    return base, inc, tc_category(base, inc), int(we.group(1)), int(be.group(1))


def iter_game_blocks_bytes(path: str, chunk: int = 8 << 20):
    """Yield raw game blocks as BYTES, splitting at C speed.

    The line-oriented reader below decodes ~200 GB of UTF-8 and runs a Python
    loop over ~1.5 billion lines. This one reads 8 MB at a time and lets
    bytes.split do the work, decoding only the handful of games that survive
    the filters. Measured 2.3x faster on the reader stage.
    """
    reader = open_stream(path)

    buf = b"\n"  # so the first [Event is preceded by a separator
    while True:
        data = reader.read(chunk)
        if not data:
            break
        buf += data
        parts = buf.split(SEP)
        buf = SEP + parts.pop()  # incomplete tail, may span chunks
        for part in parts:
            if part:
                yield b"[Event " + part
    tail = buf[len(SEP):]
    if tail.strip():
        yield b"[Event " + tail
    reader.close()


def iter_game_blocks(path: str):
    """Yield raw text of one game at a time, streaming, without extracting.

    Pass "-" to read a .zst stream from stdin, e.g.
        curl -sL <url> | python parse_lichess.py - --out out.parquet
    which lets --max-games stop you after a few GB instead of downloading 30.
    """
    stream = io.TextIOWrapper(open_stream(path), encoding="utf-8",
                              errors="replace")

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


# Standard rated games in June 2026, from database.lichess.org. Only used to
# turn the scan counter into an ETA.
TOTAL_GAMES_HINT = 86_483_328
REPORT_EVERY = 500_000


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
    ap.add_argument("--legacy-reader", action="store_true",
                    help="use the original line-oriented reader. Slower; kept "
                         "so the fast path can be diffed against it.")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run. Re-scans to the last "
                         "checkpoint WITHOUT parsing (the reader alone is ~4x "
                         "faster than the full loop), then carries on. "
                         "Checkpoints are written every --report games.")
    ap.add_argument("--report", type=int, default=REPORT_EVERY,
                    help="games between progress lines and checkpoints")
    ap.add_argument("--split-by-category", action="store_true",
                    help="write one parquet per tc_category, inserting the "
                         "category before the extension of --out. Lets a "
                         "single 2.5 h pass over the dump feed every pool "
                         "instead of one stream per pool.")
    args = ap.parse_args()

    keep = None if args.categories == "all" else set(args.categories.split(","))
    state_path = args.out + ".state.json"

    stats = dict(seen=0, no_eval=0, bad_header=0, wrong_tc=0, eligible=0, accepted=0, rows=0)
    # Per-category state. The stride is applied WITHIN each category, so one
    # pass with --every 200 yields a month-wide sample of every pool at once.
    rows: dict[str, list[dict]] = {}
    parts: dict[str, list[str]] = {}
    per_cat = {}
    t0 = time.time()

    def out_path(cat: str) -> str:
        if not args.split_by_category:
            return args.out
        stem, dot, ext = args.out.rpartition(".")
        return f"{stem}_{cat}{dot}{ext}" if dot else f"{args.out}_{cat}"

    def flush(cat: str):
        buf = rows.get(cat)
        if not buf:
            return
        parts.setdefault(cat, [])
        # The PID is not decoration. Two runs sharing an --out used to write
        # the SAME .partNNNN paths, clobbering each other's chunks, and the
        # first to finish then os.remove()d those shared paths out from under
        # the other. Silent, and it cost a full pass. Per-PID names make
        # concurrent runs merely wasteful instead of corrupting. A --resume run
        # has a new PID, but the checkpoint stores full paths, so the old parts
        # are still found and numbering continues past them.
        pth = f"{out_path(cat)}.{os.getpid()}.part{len(parts[cat]):04d}"
        pd.DataFrame(buf).to_parquet(pth, index=False)
        parts[cat].append(pth)
        rows[cat] = []

    print(f"reading {args.pgn}", file=sys.stderr, flush=True)
    print(f"  categories={args.categories} every={args.every} "
          f"split_by_category={args.split_by_category}", file=sys.stderr, flush=True)
    print(f"  progress every {args.report:,} games scanned; "
          f"June 2026 has {TOTAL_GAMES_HINT:,}", file=sys.stderr, flush=True)

    blocks = (iter_game_blocks(args.pgn) if args.legacy_reader
              else iter_game_blocks_bytes(args.pgn))
    eval_tag = "%eval" if args.legacy_reader else b"%eval"

    # ---- resume ---------------------------------------------------------
    skip_to = 0
    if args.resume and os.path.exists(state_path):
        st = json.load(open(state_path))
        skip_to = st["seen"]
        stats.update(st["stats"])
        per_cat.update({k: dict(v) for k, v in st["per_cat"].items()})
        parts.update({k: list(v) for k, v in st["parts"].items()})
        missing = [x for v in parts.values() for x in v if not os.path.exists(x)]
        if missing:
            print(f"FATAL: checkpoint references {len(missing)} missing part "
                  f"files, e.g. {missing[0]}. Delete {state_path} and start "
                  f"over.", file=sys.stderr, flush=True)
            return
        print(f"resuming from {skip_to:,} games "
              f"({skip_to/TOTAL_GAMES_HINT:.1%}), {stats['rows']:,} rows already "
              f"parsed in {sum(len(v) for v in parts.values())} part files",
              file=sys.stderr, flush=True)
    elif args.resume:
        print(f"no checkpoint at {state_path}, starting fresh",
              file=sys.stderr, flush=True)

    def save_state(seen):
        for c in list(rows):
            flush(c)          # so no buffered row is lost or duplicated
        tmp = state_path + ".tmp"
        # stats["seen"] is incremented at the TOP of the loop, so by the time
        # this runs it already counts the in-flight block, while no_eval /
        # bad_header / wrong_tc still reflect only the `seen` games that were
        # fully handled. Pinning seen to `seen` keeps the whole dict consistent
        # with "blocks 1..seen processed"; otherwise every resume re-counts one
        # block and the coverage percentage drifts.
        with open(tmp, "w") as fh:
            json.dump({"seen": seen, "stats": {**stats, "seen": seen},
                       "per_cat": per_cat, "parts": parts,
                       "args": vars(args)}, fh)
        os.replace(tmp, state_path)   # atomic, so a crash mid-write is safe

    interrupted = False
    next_report = (skip_to // args.report + 1) * args.report
    skipped = 0
    committed = skip_to     # games fully written; what the checkpoint records
    t_skip = time.time()
    try:
      for block in blocks:
        # Fast-forward without parsing. The reader alone runs several times
        # faster than the full loop, so re-reaching a checkpoint is cheap
        # compared with re-parsing everything.
        if skipped < skip_to:
            skipped += 1
            if skipped % 2_000_000 == 0:
                r = skipped / (time.time() - t_skip)
                print(f"  skipping... {skipped:,}/{skip_to:,} "
                      f"({r:,.0f} games/s)", file=sys.stderr, flush=True)
            continue

        stats["seen"] += 1

        # MUST be here, at the top. Previously this sat below eight `continue`
        # statements, so only ACCEPTED games could reach it, and it also
        # required seen to be an exact multiple of the interval at that
        # instant. With ~0.4% of games accepted that fired essentially never
        # and the job looked hung. Counter-based, not modulo, so it cannot be
        # skipped over.
        if stats["seen"] >= next_report:
            next_report += args.report
            save_state(committed)
            el = time.time() - t0
            rate = stats["seen"] / el if el else 0
            eta = (TOTAL_GAMES_HINT - stats["seen"]) / rate / 60 if rate else 0
            # flush=True is REQUIRED. Python block-buffers stderr when it does
            # not detect a tty, and Git Bash / MinTTY trips that.
            print(f"  {stats['seen']:>11,} scanned | {stats['accepted']:>7,} kept "
                  f"| {stats['rows']:>10,} rows | {rate:>7,.0f} games/s "
                  f"| ETA {eta:>4.0f} min", file=sys.stderr, flush=True)

        # ~89% of games carry no engine annotation. Rejecting them with a
        # substring scan before any parsing is what makes this tractable.
        if eval_tag not in block:
            stats["no_eval"] += 1
            committed = stats["seen"]
            continue

        if args.legacy_reader:
            meta = parse_headers(block)
            if meta is None:
                stats["bad_header"] += 1
                committed = stats["seen"]
                continue
            category = meta["tc_category"]
        else:
            quick = quick_filter(block)
            if quick is None:
                stats["bad_header"] += 1
                committed = stats["seen"]
                continue
            category = quick[2]
            meta = None  # full parse deferred until the game is actually kept

        if keep is not None and category not in keep:
            stats["wrong_tc"] += 1
            committed = stats["seen"]
            continue

        cat = category if args.split_by_category else "_all"
        c = per_cat.setdefault(cat, dict(eligible=0, accepted=0, rows=0))

        stats["eligible"] += 1
        c["eligible"] += 1
        if args.every > 1 and c["eligible"] % args.every:
            committed = stats["seen"]
            continue
        if args.max_games and c["accepted"] >= args.max_games:
            committed = stats["seen"]
            continue  # this pool is full; keep scanning for the others

        if meta is None:  # decode + full header parse ONLY for kept games
            block = block.decode("utf-8", errors="replace")
            meta = parse_headers(block)
            if meta is None:  # cannot happen; quick_filter is stricter-or-equal
                stats["bad_header"] += 1
                committed = stats["seen"]
                continue

        # Build this game's rows in a scratch list and commit them in one go.
        # A KeyboardInterrupt can land anywhere, including halfway through
        # extract_game. If rows went straight into the buffer, the checkpoint
        # could record a game as seen while holding only part of its rows, and
        # the resume would skip past it -- silently losing a game. Committing
        # atomically, and only then advancing `committed`, makes an interrupt
        # cost at most the game in flight, which the resume re-parses.
        game_rows = list(extract_game(block, meta, args.threshold))

        stats["accepted"] += 1
        c["accepted"] += 1
        rows.setdefault(cat, []).extend(game_rows)
        stats["rows"] += len(game_rows)
        c["rows"] += len(game_rows)
        committed = stats["seen"]

        if len(rows[cat]) >= args.chunk:
            flush(cat)


        wanted = keep if keep else set(per_cat)
        if args.max_games and per_cat and all(
                per_cat.get(c, {}).get("accepted", 0) >= args.max_games
                for c in (wanted if args.split_by_category else ["_all"])):
            break
        if args.max_rows and stats["rows"] >= args.max_rows:
            break
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n\ninterrupted at {stats['seen']:,} games scanned. Writing "
              f"what was parsed so far.", file=sys.stderr, flush=True)

    for cat in list(rows):
        flush(cat)

    print("\n".join(f"{k:>12}: {v:,}" for k, v in stats.items()),
          file=sys.stderr, flush=True)
    if interrupted:
        print(f"{'coverage':>12}: {stats['seen']/TOTAL_GAMES_HINT:.1%} of the "
              f"file. The sample is a PREFIX, not a month. Check section 2c of "
              f"verify_labels.py before quoting a date range.",
              file=sys.stderr, flush=True)
    print(f"{'annotated':>12}: {(stats['seen']-stats['no_eval'])/max(stats['seen'],1):.2%} "
          f"of games scanned", file=sys.stderr)

    for cat in sorted(parts):
        df = pd.concat([pd.read_parquet(x) for x in parts[cat]], ignore_index=True)
        dest = out_path(cat)
        df.to_parquet(dest, index=False)
        # Parts are the resume material. Deleting them after an interrupt
        # would strand the checkpoint that references them, so keep them and
        # let the next --resume append. On a clean finish they are redundant.
        if not interrupted:
            for x in parts[cat]:
                os.remove(x)
        valid = df[df.label_valid]
        print(f"\n  [{cat}] {len(df):,} rows, {per_cat[cat]['accepted']:,} games",
              file=sys.stderr)
        if len(valid):
            print(f"  {'base rate':>12}: {valid.blunder.mean():.4f} "
                  f"({int(valid.blunder.sum()):,} / {len(valid):,} labelled plies)",
                  file=sys.stderr)
        print(f"  {'mate-scored':>12}: {(~df.label_valid).mean():.4f} of rows",
              file=sys.stderr)
        if "date" in df and df.date.notna().any():
            days = df.date.dropna().unique()
            print(f"  {'date span':>12}: {min(days)} to {max(days)} "
                  f"({len(days)} distinct days)", file=sys.stderr)
        print(f"  {'wrote':>12}: {dest}", file=sys.stderr)

    if interrupted:
        save_state(committed)
        print(f"\ncheckpoint at {committed:,} games "
              f"({committed/TOTAL_GAMES_HINT:.1%}). Resume with the SAME "
              f"command plus --resume", file=sys.stderr, flush=True)
    elif os.path.exists(state_path):
        os.remove(state_path)


if __name__ == "__main__":
    main()