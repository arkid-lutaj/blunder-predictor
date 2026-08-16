#!/usr/bin/env python3
"""
Where am I, and what do I run next?

Run this any time you sit down, come back after a week, or lose the thread.
It looks at what exists on disk, works out which phase you finished, and prints
the exact next command. It never changes anything.

    python src/status.py

A static guide goes stale the moment you deviate from it. This reads the actual
filesystem, so it cannot.
"""

import argparse
import glob
import os
import sys
from datetime import datetime

DATA = "data"
MODELS = "models"
METRICS = "metrics"
FIGURES = "figures"


def size(path: str) -> str:
    b = os.path.getsize(path)
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024 or unit == "GB":
            return f"{b:.0f}{unit}" if unit == "B" else f"{b:.1f}{unit}"
        b /= 1024
    return f"{b:.1f}GB"


def age(path: str) -> str:
    d = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
    if d.days:
        return f"{d.days}d ago"
    if d.seconds > 3600:
        return f"{d.seconds//3600}h ago"
    return f"{d.seconds//60}m ago"


def rows(path: str):
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        return None


def find(pattern: str):
    # NEWEST FIRST, not alphabetical. Both features_blitz.parquet (the one-week
    # sample) and features_blitz_full.parquet (the full month) match the same
    # glob, and alphabetically the stale one sorts first -- so this printed
    # `--features data/features_blitz.parquet` next to a children file built
    # against the full month. That pairing is caught by decompose.py's FEN
    # guard, but only after you have pasted and run it. Sorting by mtime picks
    # the artefact you are actually working with.
    #
    # glob returns OS-native separators. On Windows that means backslashes,
    # which Git Bash treats as escape characters, so a printed command like
    # `--data data\positions.parquet` fails when pasted. Normalise to forward
    # slashes: Python, pandas and Windows itself all accept them.
    hits = glob.glob(pattern)
    hits.sort(key=os.path.getmtime, reverse=True)
    return [p.replace("\\", "/") for p in hits]


def newest_model(pool: str):
    """Prefix of the shippable engine_free model, newest first.

    Excludes the experiment variants: _gamehash is the leakage comparison and
    _noclock is the ablation. Neither is the model to build the site from, and
    _gamehash happens to be the newest file on disk, so "just take the latest"
    would quietly pick it.
    """
    skip = ("_gamehash", "_noclock")
    for path in find("models/*_free.txt"):
        stem = os.path.basename(path)[: -len(".txt")]
        if not any(s in stem for s in skip):
            return f"models/{stem}"
    return f"models/{pool}_free"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="blitz")
    args = ap.parse_args()
    p = args.pool

    if not os.path.isdir("src"):
        print("Run this from the project root (the folder containing src/).")
        return 1

    print("=" * 74)
    print(f"  BLUNDER PREDICTOR STATUS   pool={p}   {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 74)

    dump = (find("data/lichess_db_standard_rated_*.pgn.zst")
            or find("*.pgn.zst") or find("../*.pgn.zst"))
    positions = find(f"data/positions*{p}*.parquet")
    features = find(f"data/features*{p}*.parquet")
    splits = find(f"data/splits*{p}*.parquet")
    children = find("data/children*.parquet")
    # The shippable model if there is one, else any model at all. A plain
    # models/*{pool}*.txt glob misses the current full-month models entirely,
    # because they are named full_free / full_assisted rather than by pool.
    model_files = find("models/*.txt")
    shippable = newest_model(p)
    models = find(shippable + ".txt") or model_files

    # The four variants are a FAMILY sharing a prefix, and the prefix is not
    # the pool: the full-month models are full_free / full_assisted while the
    # week-sample ones are blitz_*. Deriving the family from the shippable
    # model means the completeness check follows whichever generation you are
    # actually on, instead of forever checking a stale one.
    family = os.path.basename(shippable)
    family = family[: -len("_free")] if family.endswith("_free") else p
    want = [(f"{family}_free", "engine_free", ""),
            (f"{family}_free_noclock", "engine_free", "--no-clock"),
            (f"{family}_assisted", "engine_assisted", ""),
            (f"{family}_assisted_noclock", "engine_assisted", "--no-clock")]
    have = {os.path.basename(m) for m in model_files}
    todo = [w for w in want if f"{w[0]}.txt" not in have]
    base_json = find(f"metrics/baselines*{p}*.json")
    decomp = find("metrics/decomposition*.json")
    figs = find("figures/*.png")
    site = find("docs/index.html")
    challenge = find("docs/challenge.json")
    readme = find("README.md")

    print("\nARTEFACTS")
    checks = [
        ("raw dump", dump), ("positions", positions), ("features", features),
        ("splits", splits), ("children (Stockfish)", children),
        ("models", models), ("baseline metrics", base_json),
        ("decomposition", decomp),
        ("figures", figs), ("site index", site), ("challenge data", challenge),
        ("README", readme),
    ]
    for name, got in checks:
        if got:
            n = rows(got[0]) if got[0].endswith(".parquet") else None
            extra = f", {n:,} rows" if n else ""
            print(f"  [x] {name:<22} {os.path.basename(got[0])[:38]:<40}"
                  f" {size(got[0])}{extra}, {age(got[0])}")
        else:
            print(f"  [ ] {name:<22} -")

    # ----- decide the next action -------------------------------------------
    print("\nNEXT STEP")

    def step(title, cmd, note=None, mins=None):
        print(f"  {title}")
        if mins:
            print(f"  (about {mins})")
        print()
        for line in cmd.strip().split("\n"):
            print(f"      {line}")
        if note:
            print(f"\n  {note}")

    if not dump and not positions:
        step("Get the data. You are network-bound, so use the torrent.",
             "aria2c lichess_db_standard_rated_2026-06.pgn.zst.torrent\n"
             "# or open the .torrent in qBittorrent/Transmission",
             "Verify with: python src/bench_pipeline.py", "1-3 h, unattended")

    elif not positions:
        step("Parse the dump. One pass writes both pools.",
             f"python src/parse_lichess.py {dump[0]} \\\n"
             f"    --out data/positions_2026-06.parquet --split-by-category \\\n"
             f"    --categories blitz,rapid --every 200",
             "--every 200 spreads the sample across the whole month.",
             "20-35 min from local disk")

    elif not find(f"reports/report_{p}.txt") and not features:
        step("Validate the label BEFORE building anything on top.",
             f"python src/verify_labels.py --data {positions[0]} \\\n"
             f"    --out reports/report_{p}.txt",
             "Gate: base rate 2-5%, deciles monotonic, colour symmetry OK.",
             "2 min")

    elif not features:
        step("Build features (both sets land in one file).",
             f"python src/build_features.py --data {positions[0]} \\\n"
             f"    --out data/features_{p}.parquet --workers 7",
             "Aborts if any feature correlates >0.5 with the label.",
             "5-15 min")

    elif not splits:
        step("Assign player-disjoint splits.",
             f"python src/make_splits.py --data {features[0]} \\\n"
             f"    --out data/splits_{p}.parquet",
             "Must end with 'train/test player overlap: 0  OK'.", "1 min")

    elif not base_json:
        step("Baselines. B1 (rating only) is the bar the model must clear.",
             f"python src/baselines.py --data {features[0]} \\\n"
             f"    --splits {splits[0]} --out metrics/baselines_{p}.json",
             "Write down the B1 Brier skill number.", "2 min")

    elif todo:
        step(f"Train. {len(todo)} of 4 '{family}' model variants left.",
             "\n".join(
                 f"python src/train.py --data {features[0]} \\\n"
                 f"    --splits {splits[0]} \\\n"
                 f"    --feature-set {fs} {fl} --out models/{name}"
                 for name, fs, fl in todo),
             "engine_free vs engine_assisted is the headline experiment.",
             "5 min each")

    elif not figs:
        step("Evaluate: reliability diagrams, PR curves, calibration tables.",
             f"python src/evaluate.py --data {features[0]} \\\n"
             f"    --splits {splits[0]} --model models/{p}_assisted \\\n"
             f"    --out figures/",
             "Calibration must hold WITHIN rating bands, not just on average.")

    elif not children:
        step("Stockfish over every legal child. Check depth first.",
             f"python src/depth_check.py --data {features[0]} --n 500\n"
             f"python src/eval_children.py --data {features[0]} \\\n"
             f"    --out data/children_40k.parquet \\\n"
             f"    --n-positions 40000 --depth 8 --workers 7",
             "Checkpoints every 1000 positions; --resume is safe.",
             "1-3 h, start before bed")

    elif not decomp:
        step("Phase 7: split the rating effect into availability vs selection.",
             f"python src/decompose.py --self-test\n"
             f"python src/decompose.py --children {children[0]} \\\n"
             f"    --features {features[0]} \\\n"
             f"    --out metrics/decomposition.json \\\n"
             f"    --report reports/decomposition.txt",
             "Run the self-test first; it takes a minute and checks the\n"
             "  identity, the planted-split recovery and the floor control.",
             "3 min")

    elif not challenge:
        step("Build the challenge game (needs children data, which you have).",
             f"python src/build_challenge.py --children {children[0]} \\\n"
             f"    --features {features[0]} \\\n"
             f"    --model {newest_model(p)} --out docs/",
             "engine_free model so the page needs no server. --features is\n"
             "  required and must be the SAME parquet the children were built\n"
             "  against, or row_id resolves to unrelated positions.", "5 min")

    else:
        # These phases are specified in RUNBOOK.md but not yet implemented.
        # Printing a command for a script that does not exist is exactly the
        # staleness this tool exists to prevent, so check before suggesting.
        pending = [("src/difficulty_curves.py", "Phase 8, rating sweep",
                    "RUNBOOK.md L777"),
                   ("src/validate_puzzles.py", "external validation on "
                    "Lichess puzzles", "RUNBOOK.md L825"),
                   ("src/build_site.py", "docs/index.html landing page",
                    "RUNBOOK.md L992"),
                   ("README.md", "the writeup", "pull numbers from FINDINGS")]
        missing = [x for x in pending if not os.path.exists(x[0])]
        if missing:
            body = "\n".join(f"# {path:28} {what}  ({where})"
                             for path, what, where in missing)
            step(f"Core pipeline is complete. {len(missing)} script(s) left to "
                 f"WRITE, not run.",
                 body + "\n# then write README.md and enable GitHub Pages "
                        "on /docs",
                 "Pull every README number from FINDINGS.md rather than\n"
                 "  re-deriving it. The rating sweep in difficulty_curves.py\n"
                 "  must move mover_elo, opp_elo, mean_elo and elo_gap\n"
                 "  TOGETHER -- see the bug note in FINDINGS.md.")
        else:
            remote = os.popen("git remote").read().strip()
            if not remote:
                step("Everything is built. The work is LOCAL ONLY.",
                     "# create an empty repo on GitHub, then:\n"
                     "git remote add origin <url>\n"
                     "git push -u origin master\n"
                     "# then Settings -> Pages -> deploy from branch, /docs",
                     "Nothing here is backed up: no remote is configured and\n"
                     "  data/ and models/ are gitignored, so a disk failure\n"
                     "  loses every parquet and every trained model too.")
            else:
                step("Everything is built and a remote exists.",
                     "git push\n"
                     "# then Settings -> Pages -> deploy from branch, /docs",
                     "Re-run any stage freely; every script is idempotent.")

    print("\n" + "-" * 74)
    print("  full plan: RUNBOOK.md   |   this check: python src/status.py")
    print("  rerun any completed step freely; every script is idempotent")
    print("-" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
