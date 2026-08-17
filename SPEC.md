# Project spec and invariants

The rules this codebase is built to. Anything here is load-bearing: several of
these were learned by getting them wrong first, and the reasons are in
FINDINGS.md.

## What this is
A human error model: P(blunder | position, rating). NOT an engine clone. The
output is a calibrated probability, e.g. "a 1400 blunders here 23% of the time,
a 2000 blunders 4%".

## Label
blunder = (win%_before - win%_after) > 20, from the MOVER's point of view.
win% = 50 + 50 * (2 / (1 + exp(-0.00368208 * cp)) - 1), cp clamped to +/-1000.

The win% transform and the constant 0.00368208 are LICHESS'S, documented at
https://lichess.org/page/accuracy, not this project's. The 20-point threshold
and the clamp are this project's choices. See THIRD_PARTY.md. Never restate
this formula without the attribution attached.

Rows where either side of the transition is a forced-mate score have
label_valid=False and must be excluded from training and metrics.

## NEVER use these as features (they leak the label)
cp_after, mate_after, winpct_after, win_drop, blunder, label_valid,
clk_after, time_spent (of the current move), result, termination.
Everything must be knowable BEFORE the move is played. Time spent on the
CURRENT move is decided simultaneously with the move, so it is not a feature.
Time spent on PREVIOUS moves is fine.

## Rules
- Split by PLAYER, never randomly by position. Positions in one game share a
  player, an opening and a clock situation.
- Base rate is 2-5%. Never report accuracy. Report log loss, Brier skill score
  against the base rate, PR-AUC against the positive-rate baseline, and a
  reliability diagram.
- ROC-AUC is invariant to class balance, so it is not "inflated" by imbalance.
  Report it, but never alone: high AUC coexists with terrible precision here.
- Every script takes argparse args and writes to an explicit --out path.
- Set seeds. Print row counts and base rates at every stage.
- Prefer plain pandas + numpy. No new dependencies without asking.
- A rating sweep must move mover_elo, opp_elo, mean_elo and elo_gap TOGETHER.
  They are algebraically linked, so moving one alone asks the model an
  incoherent question. Use build_features.rating_grid, never a bare sweep.
- A fix is not done until something executes the fixed path. Documentation
  claiming a fix is worse than a known-open bug.

## Naming
Blitz and rapid have SEPARATE Glicko2 rating pools, so a 1500 blitz and a 1500
rapid are different players. Never mix them. Every artefact carries a pool
suffix: positions_blitz_YYYY-MM.parquet -> features_blitz.parquet -> and so on.
Every script takes an explicit --data path. Never hardcode a filename.

## Layout
```
src/            scripts
src/tactical.py hanging pieces and tension, written and tested, do not rewrite
data/           parquet (gitignored)
models/         pickles (gitignored)
figures/        plots (committed)
```
