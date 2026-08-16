#!/usr/bin/env python3
"""
The parse produced no output at all. Find out which layer is stalling.

Each stage is timed separately, so instead of "it hangs" you get "raw disk read
took 90 seconds", which points at a cause. Reference numbers from a Linux box
on a 39 MB LDM-compressed file: open 0.00s, first 8 MB decompressed 0.07s,
first block 0.21s.

    python src/diagnose_read.py --file data/lichess_db_standard_rated_2026-06.pgn.zst
"""

import argparse
import os
import sys
import time

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def stage(label, fn):
    t = time.time()
    try:
        out = fn()
    except Exception as exc:
        print(f"  {label:<42} FAILED after {time.time()-t:6.2f}s: "
              f"{type(exc).__name__}: {exc}", flush=True)
        raise
    el = time.time() - t
    flag = "" if el < 5 else "   <-- SLOW"
    print(f"  {label:<42} {el:6.2f}s{flag}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    args = ap.parse_args()
    path = args.file

    print(f"diagnosing {path}\n", flush=True)

    if not os.path.exists(path):
        print("FATAL: file does not exist at that path")
        return 1

    size = os.path.getsize(path)
    expected = 28_225_942_769
    print(f"  size on disk: {size:,} bytes ({size/1e9:.2f} GB)")
    if abs(size - expected) > 1_000_000:
        print(f"  WARNING: June 2026 should be {expected:,} bytes. "
              f"Off by {size-expected:+,}. A truncated or still-downloading "
              f"file will stall or fail.")
    else:
        print("  size matches the published June 2026 file. OK")

    low = path.lower()
    for marker, why in (
            ("onedrive", "OneDrive Files On-Demand can leave a 28 GB stub that "
                         "must download on first read"),
            ("dropbox", "Dropbox smart sync behaves the same way"),
            ("google drive", "Drive streaming behaves the same way")):
        if marker in low:
            print(f"  WARNING: path contains '{marker}'. {why}.")
    print(flush=True)

    print("timings, each stage separately:", flush=True)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from parse_lichess import open_stream, iter_game_blocks_bytes, is_zstd

    fh = stage("open() the file", lambda: open(path, "rb"))
    head = stage("read first 4 bytes (magic number)", lambda: fh.read(4))
    if head == ZSTD_MAGIC:
        what = "(zstd data frame, correct)"
    elif is_zstd(head):
        # pzstd prefixes every compressed frame with a 12-byte skippable frame
        # holding that frame's size. Valid zstd, but the naive magic check
        # rejects it and the file gets parsed as plain text.
        n = int.from_bytes(fh.read(4), "little")
        what = f"(zstd SKIPPABLE frame, {n} byte payload -- pzstd multi-frame)"
    else:
        what = "(NOT ZSTD)"
    print(f"       magic = {head!r} {what}", flush=True)
    if not is_zstd(head):
        print("  FATAL: this is not a zstd file. Download is corrupt.")
        return 1

    stage("read 1 MB raw from disk", lambda: fh.read(1 << 20))
    stage("read 32 MB raw from disk", lambda: fh.read(32 << 20))
    fh.close()

    reader = stage("open_stream() (sniff + zstd init)",
                   lambda: open_stream(path))
    data = stage("decompress first 8 MB", lambda: reader.read(8 << 20))
    print(f"       got {len(data):,} bytes, starts with "
          f"{data[:40]!r}", flush=True)

    def first_block():
        for b in iter_game_blocks_bytes(path):
            return b
    blk = stage("yield the first game block", first_block)
    print(f"       block is {len(blk):,} bytes, starts "
          f"{blk[:60]!r}", flush=True)

    def scan_200k():
        n = 0
        for b in iter_game_blocks_bytes(path):
            n += 1
            if n >= 200_000:
                return n
        return n
    n = stage("scan 200,000 game blocks", scan_200k)

    print(f"\nIf every stage above was fast, the parse itself is fine and the "
          f"problem\nwas output buffering. If 'read 32 MB raw' was slow, it is "
          f"disk or antivirus:\nadd the folder to Windows Defender exclusions "
          f"and try again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
