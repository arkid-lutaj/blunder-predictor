# If you stopped, start here

You do not need to remember anything. Two commands:

```bash
cd ~/projects/blunder-predictor
source .venv/bin/activate        # Windows Git Bash: source .venv/Scripts/activate
python src/status.py
```

`status.py` reads the actual filesystem, works out which phase you finished, and
prints the exact next command with a time estimate. It never modifies anything,
so run it as often as you like. A written checklist goes stale the moment you
deviate; this cannot.

If `status.py` itself fails, you are probably in the wrong folder or the venv is
not active. That is the only failure mode it has.

---

`FINDINGS.md` is the running record of what the data actually turned out to
be. Read it if you have forgotten why a number is what it is.

## The whole pipeline, one line each

Every script is idempotent. Re-running a completed step overwrites its own
output and breaks nothing.

| # | command | produces | time |
|---|---|---|---|
| 1 | `aria2c <torrent>` | the 28.2 GB dump | 1-3 h |
| 2 | `python src/parse_lichess.py <dump> --out data/positions_2026-06.parquet --split-by-category --categories blitz,rapid --every 200` | positions per pool | 20-35 min |
| 3 | `python src/verify_labels.py --data data/positions_blitz_2026-06.parquet --out reports/report_blitz.txt` | validation report | 2 min |
| 4 | `python src/build_features.py --data data/positions_blitz_2026-06.parquet --out data/features_blitz.parquet --workers 7` | features | 5-15 min |
| 5 | `python src/make_splits.py --data data/features_blitz.parquet --out data/splits_blitz.parquet` | balanced, near-player-disjoint folds | 1-4 min |
| 6 | `python src/baselines.py --data data/features_blitz.parquet --splits data/splits_blitz.parquet --out metrics/baselines_blitz.json` | the bar to beat | 2 min |
| 7 | `python src/train.py --data data/features_blitz.parquet --splits data/splits_blitz.parquet --feature-set engine_free --out models/blitz_free` | model (x4 variants) | 5 min each |
| 8 | `python src/eval_children.py --data data/features_blitz_full.parquet --out data/children_150k.parquet --n-positions 150000 --depth 8 --workers 7 --per-move > logs/children.log 2>&1` | Stockfish over all legal moves | 3-4 h |
| 9 | `python src/build_challenge.py --children data/children_150k.parquet --features data/features_blitz_full.parquet --model models/full_free --out docs/` | the playable game | 5 min |

Steps 3, 5 and 6 are gates. If one fails, fix it before continuing; everything
downstream inherits the problem.

---

## The three gates, and what "pass" means

**After step 3.** Base rate between 2% and 5%. Blunder rate falling at every
rating decile. Colour symmetry OK. Your actual numbers on the full month were
3.952% (blitz) and 4.265% (rapid), both perfectly monotonic at thresholds 15,
20 and 25.

**After step 5.** The line `splits match on rating and base rate. OK`, and the
exit code is 0. Player overlap is NO LONGER expected to be zero: on the full
month the graph percolated and one 22% component is carved, so overlap is ~3.5%
of test players by design against 36.9% for a naive split. The script prints
both and warns if the carve ever costs more than half the naive protection.
Zero overlap is only required when nothing was carved.

**After step 6.** B1 (rating only) beats B0 (constant) on Brier skill. Write
that number down. The trained model has to beat it on the same split or you
have no result.

---

## Interrupting a long run

Steps 1, 2 and 8 are the long ones.

- **Step 1** is a torrent. Closing the client pauses it; reopening resumes.
- **Step 2** has no resume. If you kill it, re-run from scratch. It is 20-35
  minutes from local disk, so this is survivable. Do not kill it while it is
  writing the final parquet.
- **Step 8** checkpoints every 1000 positions. Ctrl-C, then add `--resume` and
  it picks up where it stopped. Throttling or a closed lid costs wall clock,
  never work.

---

## Recovering from a bad state

| symptom | fix |
|---|---|
| a parquet is half-written after a crash | delete it and re-run the step; nothing else reads it yet |
| `.part0000` files left behind | safe to delete; the parser writes them then concatenates |
| `status.py` suggests a step you already did | its output file is missing or misnamed. `ls data/` and compare against the table above |
| everything looks broken | `git status`, `git stash`, re-run `status.py`. Data files are gitignored so a stash never loses them |
| you want a clean re-run of just the model | delete `models/` and `metrics/`, keep the parquets, re-run steps 6-7 |

---

## Where things live

```
data/      parquets and the dump      gitignored except the two samples
models/    boosters + calibrators     gitignored except final_model.txt
metrics/   json, every README number  committed
figures/   png                        committed
reports/   verify_labels output       committed
docs/      the static site and game   committed, served by GitHub Pages
src/       every script               committed
```

`SPEC.md` holds the invariants: the label definition, the columns that must
never become features, and the split rule. Read it before adding a feature --
several of its lines exist because the opposite was tried first, and FINDINGS.md
records what went wrong.
