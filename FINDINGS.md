# Findings

Measured facts about **this** dataset, not general advice. Everything here came
out of an actual run and is traceable to a command. SPEC.md holds the
invariants; this is what the data turned out to be. Add to it as you go, and pull the
README numbers from here rather than re-deriving them.

Current dataset: `positions_blitz_2026-06.parquet`, 25,000 games, 1,667,618
rows, parsed with `--every 40 --max-games 25000` (a prefix sample, see below).

---

## The label is healthy

| quantity | value |
|---|---|
| base rate, all valid rows | **3.984%** |
| base rate, eligible rows only (see structural floor) | **4.555%** |
| label_valid=False (forced-mate scores) | 6.22% |
| stored `blunder` vs `win_drop > 20` | 1,563,826 / 1,563,826 agree |
| rating deciles D1 -> D10 | 6.450% -> 2.224% (**2.90x**) |
| monotonic at thresholds 15 / 20 / 25 | yes, yes, yes (0 up-steps each) |
| D1/D10 ratio at 15 / 20 / 25 | 2.53x / 2.90x / **3.28x** |

The ratio *strengthening* as the threshold rises is the useful part: the rating
effect is a property of the players, not an artefact of where the cut was
placed.

## Structural floor: the label cannot fire in lost positions

`cp` is clamped to +/-1000, so win% is clamped to **[2.46, 97.54]**. A 20-point
drop is therefore arithmetically impossible whenever `winpct_before <= 22.46`,
which is **`cp_before < -337`**.

- **12.5% of valid rows** are structurally incapable of being blunders.
- Those rows contain **exactly 0 blunders**, which independently confirms the
  parser's arithmetic.
- Equivalent cut-offs: threshold 15 -> cp > -422; threshold 25 -> cp > -264.

Two consequences.

1. **README limitation to state plainly.** The win%-drop label is blind to
   blunders in already-lost positions. Losing a rook when you are down a queen
   does not register. That is a deliberate consequence of using win probability
   instead of centipawn loss, but it must be said.
2. **It contaminates the engine_free vs engine_assisted comparison.**
   `engine_assisted` has `winpct_before`, so part of its advantage is just
   learning "below -337cp, predict zero" — a definitional boundary, not
   insight into human error. Always report that comparison **also on the
   eligible subset** (`winpct_before > 22.46`), where the free win vanishes.
   The gap that survives is the real answer to "how much does an engine help".

## Calendar coverage: better than the warning suggests

The run stopped after 20.6M of 86.5M games, so `verify_labels.py` fires the
FRONT-LOADED banner at 23% coverage. But the specific window is good:

- Days present: **2026-06-01 to 2026-06-07**, which is exactly **Monday to
  Sunday**. A complete calendar week.
- Busiest day share **14.69%** against a uniform 14.29%. Near-perfectly even.

So describe it as "one complete calendar week of June 2026", not "a full
month" and not "a biased prefix". The banner only counts distinct days and
cannot see that the window happens to be a clean week.

## Colour symmetry: passes, and why the check had to be rewritten

| sample | white | black | diff | se | ratio |
|---|---|---|---|---|---|
| one week, 1.5M rows | 4.03% | 3.94% | +0.088pp | 2.8 | 1.023x |
| full month, 6.3M rows | 4.01% | 3.90% | +0.110pp | **7.1** | **1.028x** |

**The effect did not change. The sample got 4x bigger.** The standard error
halved twice and the same harmless 2.8% relative difference crossed a 6-se
alarm threshold.

That was a design error in the check, and a textbook one: **gating on
statistical significance at large n**. At 6.3M rows any real effect is
significant, so significance is not evidence of a bug. The gate is now on the
RATIO, tolerance 1.20x, with standard errors reported for information only.

The check is still meaningful. Under a broken flip, Black's label would mean
"the eval moved 20+ points in Black's favour during Black's own move", which
occurs in under 0.08% of rows, so Black would read ~0.01% against White's 4% --
about **400x**. Verified on fixtures: healthy data gives 1.011x and passes, a
simulated broken flip gives 427.6x and fails. The report also now prints the
`win_drop < -20` rate by colour, which is the unambiguous signature.

- Effect size is tiny: ~2.8% relative, stable across a 4x change in sample size.
- Most likely composition, not psychology: White scores better overall (49.8%
  vs 46.4%), so White sits above the -337cp structural floor more often and
  simply has more positions in which a blunder can be recorded at all. Testable
  by re-running the comparison on eligible rows only.
- Partly mechanical: Black has 12,126 more rows than White, because White's
  ply 0 is dropped whenever a game carries no pre-move eval comment.

Not a bug. Worth one footnote if anyone asks why the rates differ.

## Small stuff that looks alarming and is not

- **215 negative `time_spent` rows** (0.013%). Lichess lag compensation gives
  time back; the clock legitimately goes up by more than the increment. Not a
  parsing drift.
- **20.05% of moves have negative `win_drop`** (the eval improved for the mover
  after their own move). 15.66% of all rows sit in [-1, 0) and only 0.08% below
  -5, so this is small near-symmetric annotation jitter, not a fifth of the
  evals being broken.
- **30.50% of positives sit within 5 win% of the threshold**, and 32,043
  negatives sit just under it. This is the irreducible label noise. It caps how
  well any model can score and attenuates measured effects. It does not by
  itself make a smaller gain fake; the test that matters is whether model
  ranking is stable at thresholds 15, 20 and 25.

## Selection bias in the annotated subset

Lichess analysis is opt-in, so the annotated games are not a random sample.

- Blitz median Elo **+135 above the pool** (1635 vs ~1500), IQR **1.52x wider**.
- Draw rate matches the pool, so the loser-driven-request effect is not visible
  in blitz results (it was -1.89pp in rapid).

Since rating is a dominant feature, the model sees fewer weak blitz players
than the real pool contains. Any claim about performance at 900 Elo blitz is
thinner than the row count suggests.

## Graph structure and splits

| quantity | value |
|---|---|
| distinct players | 42,376 across 50,000 sides |
| games per player | 1.18 |
| connected components | 17,425 |
| singletons | 13,650 (**78.3%**) |
| largest component | 400 games (**1.6%** of the dataset) |
| naive both-players-same-fold, K=10 | would keep **10.1%** of games |
| naive both-players-same-fold, K=5 | would keep 19.8% |
| component split | keeps **100%**, 0 train/test player overlap |
| game-hash split leakage | **1,853 players = 19.0% of its test players** |

That last row is the number to quote in the README: it is the leakage the
component split removes, measured rather than asserted.

## Bugs found and fixed during the build

| bug | symptom | fix |
|---|---|---|
| `PIECE_VALUES` missing `chess.KING` | `KeyError: 6` on ~12% of positions | KING added at 20000 in tactical.py |
| Greedy largest-first split packing | train median Elo **+114** above test, base rate 3.86% vs 4.26% | random-order greedy in make_splits.py, plus a balance check that warns |
| Feature rows held as Python dicts | 2,784 B/row vs 200 B as float32, **14x**. 26 GB at `--every 30` | chunk -> float32 frame immediately in build_features.py |
| LightGBM Dataset reuse | trial 2 dies on `min_data_in_leaf` | `feature_pre_filter=False` in train.py |
| `-` assumed stdin was always zstd | `zstd -dc \| parse -` crashed | magic-byte sniffing in parse_lichess.py |
| Sniff matched only the zstd DATA-frame magic | full-month parse reported `seen: 0` forever | `is_zstd()` now accepts skippable frames too |
| Progress print sat below eight `continue`s | no progress output at all; job looked hung | moved to the top of the loop, counter-based not modulo |
| No resume on a 70-minute parse | lost 8.4% of a run twice | checkpoint every `--report` games, `--resume` fast-forwards |
| Interrupt could land mid-game | checkpoint counted a game whose rows were only half saved | rows built in a scratch list, committed atomically |
| status.py printed Windows backslashes | pasted commands failed in Git Bash | normalise to forward slashes |

## Performance, measured on this machine

| stage | rate | note |
|---|---|---|
| download over HTTP | 3.6-4.5 MB/s | **network-bound by 4-6x**; parser can eat 30 MB/s |
| parser reader stage | ~87,000 games/s | 1.5x faster than the original line reader |
| `extract_game` (python-chess) | 344 games/s | only on accepted games, ~1.2 min per pass |
| feature build | ~1,800 rows/s on 4 workers | dominated by `gives_check` over all legal moves |
| LightGBM, 1.6M x 60, 400 rounds | 33 s on ONE core | tuning is not the bottleneck; data volume is |

Things tried that did **not** help: `zstd -dc | python -` (0.98x), GPU (nothing
here can use one), Python threads (GIL), parallelising `extract_game` (1.2 min
of a full pass).

## Sizing the full-month re-parse

Eligible blitz games in all of June: **4,188,831** (4.84% of games scanned).
Rows per game: 66.7.

| `--every` | games | rows | positives | features on disk | peak RAM | feature build |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 21k | 1.4M | 56k | 0.5 GB | 0.3 GB | 13 min |
| 50 | 84k | 5.6M | 223k | 2.1 GB | 1.1 GB | 52 min |
| 30 | 140k | 9.3M | 371k | 3.5 GB | 1.9 GB | 86 min |
| **20** | **209k** | **14M** | **557k** | **5.2 GB** | **2.8 GB** | **129 min** |

`--every 200` gives better *spread* but the same *volume* as the current
sample. For a stronger model you want more positives, so target **`--every 20`**
(32 GB machine, after the dict fix) or 30 if RAM is contended.

## Baselines (one-week sample, component split)

Test set 302,939 rows, test base rate 4.257%, train base rate 3.863%.

| model | log loss | Brier skill | PR-AUC | PR baseline | ROC-AUC | ECE |
|---|---|---|---|---|---|---|
| B0 constant | 0.17623 | +0.0000 | 0.0426 | 0.0426 | 0.5000 | - |
| B1 rating only | 0.17450 | **+0.0036** | 0.0565 | 0.0426 | 0.5763 | 0.0020 |
| B2 rating+eval+clock+moves | 0.16683 | +0.0130 | 0.0786 | 0.0426 | 0.7133 | 0.0137 |

**THE BAR is B1 Brier skill = +0.0036.** Note these were computed on the
*imbalanced* split (train Elo +114 above test) and should be regenerated after
the split fix.

Signs are all correct: rating negative, clock fraction negative (less time ->
more blunders), legal-move count positive (more options -> more ways to go
wrong), `winpct_before` positive (partly the structural floor).

B2's ECE is 7x worse than B1's, so plain logistic is not calibrated here. That
is what the isotonic step in `train.py` is for.

## Trained models (one-week sample, component split, 6 trials)

Test set 318,697 rows, base rate 3.885%.

| model | features | log loss | **Brier skill** | PR-AUC | PR lift | ROC-AUC | ECE |
|---|---|---|---|---|---|---|---|
| B0 constant | - | 0.16428 | +0.0000 | 0.0388 | 1.00x | 0.500 | - |
| B1 rating only | 1 | 0.16240 | +0.0039 | 0.0533 | 1.37x | 0.589 | 0.0017 |
| B2 hand-picked | 4 | 0.15470 | +0.0145 | 0.0761 | 1.96x | 0.721 | 0.0110 |
| **engine_free** | 56 | 0.14317 | **+0.0473** | 0.1201 | 3.09x | 0.788 | 0.0017 |
| **engine_assisted** | 63 | 0.12474 | **+0.1209** | 0.2315 | 5.96x | 0.869 | 0.0014 |

Headlines:

- engine_free beats rating-only by **12x** on Brier skill.
- engine_assisted beats engine_free by **2.6x**. Part of that is the structural
  floor freebie (it has `winpct_before`), so this must be re-measured on the
  eligible subset before being quoted as "what an engine buys you".
- **Calibration is excellent without isotonic.** Raw ECE 0.0018 / 0.0012,
  calibrated 0.0017 / 0.0014. LightGBM with `is_unbalance=False` is already
  calibrated; isotonic is doing nothing here and slightly hurts PR-AUC. Keep it
  for safety but say plainly in the README that it was not needed.
- Best hyperparameters differed by feature set: engine_free chose 63 leaves at
  lr 0.03, engine_assisted 255 leaves at lr 0.02. Trials spanned only 0.0003
  val log loss, so tuning is worth very little here. Data volume is the lever.

### Feature importance, and why individual numbers mislead

engine_free: material_balance 13.8%, clock_frac 10.5%, move_number 7.3%,
mean_elo 7.3%, tension 7.0%, mover_elo 4.4%.

engine_assisted: winpct_momentum_2 16.0%, winpct_before 9.5%,
winpct_momentum_4 8.8%, clock_frac 5.3%, abs_cp_before 4.5%.

**`mean_elo` outranks `mover_elo`.** That is not a finding about chess, it is
collinearity: `mean_elo = (mover_elo + opp_elo) / 2` and
`elo_gap = mover_elo - opp_elo`, so gain is split arbitrarily among four
features carrying the same information. Never quote a single rating feature's
importance. Group them, or use SHAP on the group.

## Bug: the rating sweep must move all rating features together

Found while reading the importances above. `build_challenge.py` swept
`mover_elo` alone, leaving `opp_elo`, `mean_elo` and `elo_gap` fixed. Setting
mover_elo to 2400 while mean_elo stays at 1650 asks the model about a 2400
playing a 900 — not a position that exists in training, and not the question
intended.

Fixed: `rating_grid()` sets `opp_elo = mean_elo = R` and `elo_gap = 0`, so the
sweep answers "a player of rating R against a peer". **The same fix is required
in `difficulty_curves.py` (Phase 8) and any SHAP rating analysis.**

**Correction, found later.** The paragraph above was written when the fix was
designed, not when it landed. `rating_grid()` did not exist anywhere in `src/`;
`build_challenge.py` was still doing `grid[:, elo_idx] = RATINGS`, a bare
`mover_elo` sweep, and the first `docs/challenge.json` was generated and
committed with wrong curves. A documented fix is not a shipped fix, and this
file recorded the intent as though it were the outcome.

Now genuinely fixed: `rating_grid()` lives in `build_features.py`, is imported
by `build_challenge.py` and `validate_puzzles.py`, raises if the model has no
`mover_elo`, and skips whichever of the four rating features a reduced feature
set omits. `challenge.json` was regenerated against it.

Worth noting how it was caught: writing a SECOND consumer of the same sweep
forced a look at the first. A fix with one caller is easy to believe in.

**The bug was material, not cosmetic.** Comparing the 60 committed challenge
curves before and after, median absolute change **0.0297** and max **0.1320**.
A representative position:

| rating | 800 | 1200 | 1600 | 2000 | 2400 |
|---|---|---|---|---|---|
| bare mover_elo sweep | 7.64% | 6.39% | 5.71% | 5.71% | 4.62% |
| all four together | **15.17%** | **11.74%** | 7.64% | 5.71% | **3.63%** |

The broken sweep spans **1.65x** from 800 to 2400; the correct one spans
**4.2x**. Holding `mean_elo` at the position's actual value while moving
`mover_elo` feeds the model a contradiction, and the rating signal gets diluted
across the four collinear features — the same collinearity documented above
under feature importance, biting in a second place.

So the challenge game was showing a badly flattened rating curve, which is
precisely the thing it exists to demonstrate. Any figure or demo built on a
rating sweep must be regenerated after this fix, not just re-read.

### The process rule this earns

**A fix is not recorded as done until something executes the fixed path.**
Documentation claiming a fix is worse than a known-open bug: an open bug is
still being looked for, while a false record stops the search. This one
survived precisely because nothing ever ran the code the note described.

**Distrust single-caller fixes.** A helper with one consumer is never
cross-checked; the bug was caught only when a second consumer forced a reading
of the first.

Both are now enforced mechanically rather than by intention:

- `python src/build_features.py --self-test` checks `rating_grid` moves
  `mover_elo`, `opp_elo` and `mean_elo` together, holds `elo_gap` at 0, leaves
  non-rating features alone, tolerates reduced feature sets, and raises when
  `mover_elo` is absent.
- `check_sweep_span()` gates both consumers on the OUTPUT: it kills the run if
  the median curve span falls below **3.0x**. Measured separation on 400 real
  positions — correct sweep median 6.28x (p10 3.26x), broken 1.83x (p10 1.42x).
  Verified to fire by feeding it the broken sweep.

Note the gate is on the MEDIAN across many curves, not per curve. The correct
sweep's own minimum is 1.66x, because genuinely flat positions exist, so a
per-curve 2x threshold would fail on valid data. Live values: challenge
**13.53x**, puzzles **6.63x**, both far clear of the gate.

## The leakage number needs a proper experiment

The `calibrated (game-hash test)` row in `train.py` output is NOT a clean
leakage measurement. It scores a model trained on *component*-train against
*gamehash*-test, and those two overlap, so the model has seen some of those
exact rows. It reports +0.0703 (engine_free) and +0.1989 (engine_assisted)
against +0.0473 and +0.1209 properly, i.e. 49% and 65% inflation — but that
mixes row memorisation with player leakage.

The honest experiment is a full retrain on the naive split:

    python src/train.py --data data/features_blitz.parquet \
        --splits data/splits_blitz.parquet --split-col split_gamehash \
        --feature-set engine_free --out models/blitz_free_gamehash

Compare that model's test score against the component model's test score. Only
then quote a leakage figure.


## Evaluation protocol

`evaluate.py` produces three figures and four tables. Two things it does that
the training output cannot:

- **Reliability within rating bands.** Aggregate calibration can be right by
  accident: over-predicting weak players and under-predicting strong ones
  averages to a straight line. The claim "a 1400 blunders here 23% of the time"
  only holds if calibration holds inside each band. Panels autoscale
  independently because each band occupies a different probability range.
- **The eligible-subset table.** Re-scores everything on
  `winpct_before > 22.46`, removing the 12.5% of rows where the label cannot
  fire. engine_assisted loses its free win there, so the engine_free ->
  engine_assisted ratio on eligible rows is the honest one to quote. The
  difference between the two ratios IS the structural-floor freebie, and
  reporting both is more interesting than reporting either alone.

It also prints an integrity check: any blunder found in a structurally
impossible row means `winpct_before` and the label disagree and the parquet is
inconsistent. Must be 0.


## Clock ablation (Phase 5): the clock explains almost nothing

Calibrated test Brier skill, with and without every clock feature. Measured
twice, on two different sample sizes.

**One-week sample** (1.67M rows), 56 -> 48 and 63 -> 55 features:

| model | with clock | no clock | cost |
|---|---|---|---|
| engine_free | +0.0473 | +0.0463 | **-2.1%** |
| engine_assisted | +0.1209 | +0.1191 | **-1.5%** |

**Full month** (6,265,721 label_valid rows, 1,257,458 test rows):

| model | with clock | no clock | cost |
|---|---|---|---|
| engine_free | +0.0496 | +0.0489 | **-1.4%** |
| engine_assisted | +0.1238 | +0.1241 | **+0.2%** |

**The finding strengthens on 4x the data.** engine_assisted is fractionally
*better* without the clock, i.e. the cost is zero to within noise. Do not
over-read the positive sign — read it as "no measurable cost".

The original scope predicted time pressure would dominate. It does not.
`clock_frac` carries 9.1% gain share in full-month engine_free, but removing
every clock feature costs ~1% of Brier skill because other features absorb it:
`move_number` importance rises **9.59% -> 11.83%** when the clock is dropped,
acting as a proxy for elapsed time. On the week sample the same shift was
7.33% -> 13.28%, so it is the same mechanism, less pronounced with more data.
`mover_elo` holds at 4.73% -> 4.83%, i.e. unmoved.

README claim, in its strong form: **rating predicts blunders essentially
undiminished after controlling for time pressure.** The full month supports the
strong form better than the week sample did.

Note the with-clock full-month numbers (+0.0496 / +0.1238) are slightly above
the week-sample ones (+0.0473 / +0.1209). That is the data-volume effect, not a
change in method. The headline evaluation tables have since been regenerated
from the full month and are the ones to quote.

## Leakage (Phase 3): a measured NULL result

Trained end to end on the naive game-hash split and compared like for like:

| split | Brier skill | ROC-AUC | PR-AUC |
|---|---|---|---|
| component (player-disjoint) | +0.0473 | 0.7876 | 0.1201 |
| game-hash (naive) | **+0.0473** | 0.7871 | 0.1230 |

**No difference.** The earlier +0.0703 was an artefact of scoring a
component-trained model on gamehash-test, which overlaps component-train, so
the model had memorised those exact rows. That was row memorisation, not player
leakage.

Why it is null here: **1.18 games per player**. Only 19% of test players appear
in train, each contributing roughly one extra game, so there is almost nothing
about a player to memorise. Player leakage bites when players recur often.

Do NOT drop the component split. It is principled, costs nothing, and this is a
property of this dataset rather than of the method. State it as: "player-
disjoint splits were used; on this dataset the leakage proved negligible
because players recur only 1.18 times, and on a denser graph it would matter."
A null result you went looking for is stronger evidence of care than a
convenient positive one.


## Evaluation results (FULL MONTH) -- the headline set

**These are the numbers the README quotes.** Test 1,257,458 rows, base rate
3.9086%. Eligible rows 87.9%, base rate 4.4485%, containing 0 impossible
positives as required.

| model | Brier skill (all) | Brier skill (eligible) | PR-AUC | PR lift | ROC-AUC | ECE |
|---|---|---|---|---|---|---|
| B0 constant | +0.0000 | +0.0000 | 0.0391 | 1.00x | 0.500 | - |
| B1 rating only | +0.0037 | - | 0.0536 | 1.37x | 0.587 | 0.0012 |
| B2 hand-picked | +0.0142 | - | 0.0752 | 1.92x | 0.720 | 0.0116 |
| engine_free | +0.0496 | +0.0520 | 0.1258 | 3.22x | 0.792 | 0.0012 |
| engine_free, no clock | +0.0489 | +0.0512 | 0.1247 | 3.19x | 0.791 | 0.0011 |
| engine_assisted | **+0.1238** | **+0.1194** | 0.2389 | 6.11x | 0.871 | 0.0010 |
| engine_assisted, no clock | +0.1241 | +0.1196 | 0.2389 | 6.11x | 0.871 | 0.0010 |

engine_free beats rating-only by **13.4x** on Brier skill (+0.0496 / +0.0037).

The one-week numbers this table replaced were +0.0473 / +0.1209 all-rows. Every
figure moved slightly UP with 4x the data and none changed direction, which is
the same stability the base rate showed (3.984% -> 3.952%).

### Calibration holds WITHIN rating bands

Full month, engine_assisted, on 1,257,458 test rows:

| band | rows | observed | predicted | ECE |
|---|---|---|---|---|
| <1200 | 172,939 | 5.94% | 6.11% | 0.0027 |
| 1200-1600 | 352,209 | 4.52% | 4.62% | 0.0015 |
| 1600-2000 | 385,198 | 3.66% | 3.65% | 0.0009 |
| 2000+ | 347,112 | 2.54% | 2.56% | 0.0011 |

This is the result the whole project rests on, and it now stands on 4x the
rows: each band alone is larger than the entire one-week test set. Aggregate
calibration can be correct by accident — over-predicting weak players
cancelling under-predicting strong ones. It is not happening here, so **"a 1400
blunders here 23% of the time" is a defensible sentence**, which is the
product.

Worst band ECE is 0.0027, in the <1200 band, where the model over-predicts by
0.17pp. That is the direction to expect given the selection bias in the
annotated subset (blitz median Elo +135 above the pool), so the model has seen
fewer weak players than the pool contains.

### What the engine buys, honestly

| comparison | engine_free -> engine_assisted |
|---|---|
| all rows | +0.0496 -> +0.1238 (**2.49x**) |
| eligible rows only | +0.0520 -> +0.1194 (**2.30x**) |

Quote the eligible-only figure. The gap between 2.49x and 2.30x is the
structural-floor freebie, and it is small: the engine advantage is real, not an
artefact of knowing where the label cannot fire. Week sample gave 2.55x / 2.38x,
so this is stable.

### The honest ceiling

Top predicted decile: **13.86% observed against a 3.91% base rate** for
engine_free (**3.5x**), **21.06%** for engine_assisted (**5.4x**). So the model
identifies positions roughly 3.5x to 5.4x more dangerous than average. That is
a useful **difficulty signal, not a blunder detector**, and the README should
say so plainly. PR-AUC of 0.126 / 0.239 against a 0.039 baseline is the same
fact stated differently.

Deciles are monotone in observed frequency for every model and the
predicted/observed ratio stays within about 1.04 across the middle deciles. Two
presentation notes:

- Decile 0 ratios (0.55 to 0.79) are on predicted probabilities near 0.0001,
  where a handful of events moves the ratio and the absolute error is
  negligible. Do not quote them.
- **The engine_assisted models show 9 deciles, not 10.** Its predictions pile
  up so tightly near zero that two decile edges coincide and `qcut` merges
  them, leaving a 278,635-row bottom bin. Not a bug — a consequence of a
  sharper model on a 3.9% base rate — but the table must not be presented as
  if a decile went missing.

## Windows / Git Bash gotchas

- **Ctrl+C is interrupt, not copy.** Use Ctrl+Insert or right-click to copy. It
  is very easy to kill a two-hour job by reaching for Ctrl+C out of habit.
- **stderr is block-buffered under MinTTY.** Python only line-buffers stderr
  when it detects a tty, and Git Bash does not present as one, so progress
  output vanished into a buffer for the entire run and the job looked hung.
  Every progress print now passes `flush=True`. If you add a progress line
  anywhere, it needs the same.
- Multiprocessing uses spawn rather than fork, so worker code must sit under
  `if __name__ == "__main__"` guards. `build_features.py` and
  `eval_children.py` already do; a script that hangs at "0/N chunks" is the
  symptom.


## The zstd skippable-frame bug (cost most of an evening)

**Symptom.** The full-month parse printed its banner and then nothing. Killed
after two minutes it reported `seen: 0`. Not slow — zero blocks yielded ever.

**Cause.** Lichess' HTTP file is written by **pzstd**, which prefixes every
compressed frame with a 12-byte *skippable* frame holding that frame's size, so
the archive can be decompressed in parallel. Such a file begins `50 2a 4d 18`
(0x184D2A50), not the data-frame magic `28 b5 2f fd` (0xFD2FB528).

`open_stream()` compared only against the data-frame magic. No match meant "this
must be plain PGN text", so raw compressed bytes went straight to the block
splitter, which searched for `\n[Event ` inside binary, never found it, never
yielded, and grew its buffer without bound. Deterministic: waiting longer could
never have helped.

**Fix.** `is_zstd()` accepts both. Skippable magic is a **range**,
0x184D2A50 to 0x184D2A5F, so all sixteen values must match — checking only
0x184D2A50 would still break on other writers. Verified against a synthetic
pzstd-style multi-frame archive parsed end to end, plus edge cases at 0x...4F
and 0x...60.

**Two legitimate file sizes.** The torrent metadata says 28,225,942,769 bytes;
the HTTP download is 28,241,946,492, about 16 MB (0.057%) larger. Same content,
different compression settings. A file matching *either* is fine; only one
smaller than both is suspect.

**Lesson for the writeup.** Four plausible hypotheses (antivirus, OneDrive
stubs, truncated download, "just slow") were all wrong, and each was ruled out
by a measurement rather than an argument. The diagnostic that timed every layer
separately found it in seconds. `src/diagnose_read.py` now also names the frame
type, so this exact failure identifies itself next time.

## Parse checkpointing

A 70-minute job with no resume is fragile, and two runs were lost to it. The
second was almost certainly a background shell dying with its parent session,
not anything the user did.

`parse_lichess.py --resume` now:

- writes `{out}.state.json` every `--report` games (default 500,000), holding
  the committed game count, all counters, per-pool stride state and the part
  file list. Written to a temp file and `os.replace`d, so a crash mid-write
  cannot corrupt it.
- keeps the `.partNNNN` files after an interrupt instead of deleting them, and
  still writes a usable partial parquet. On a clean finish the parts and the
  state file are removed.
- fast-forwards on resume by counting blocks WITHOUT parsing them. The reader
  alone is several times faster than the full loop, so re-reaching a checkpoint
  costs a fraction of what re-parsing would.

**The subtle part.** A KeyboardInterrupt can land anywhere, including halfway
through `extract_game`. Appending rows directly to the buffer meant a
checkpoint could record a game as seen while holding only part of its rows, and
the resume would skip past it, silently dropping a game. First test caught it:
resumed blitz had 68,204 rows against a reference of 68,248, same game count.
Rows are now built in a scratch list and committed in one step, with the
commit point advanced only afterwards, so an interrupt costs at most the game
in flight.

Verified at four different interrupt points (4.3k, 8.4k, 12.8k, 16.7k games);
every resumed output is byte-identical to an uninterrupted reference run.

## Full-month parse (the real dataset)

86,483,328 games scanned, counters reconcile exactly. `--every 40`.

| | blitz | rapid |
|---|---|---|
| rows | 6,675,423 | 4,535,677 |
| games | 100,484 | 69,942 |
| base rate | **3.952%** | 4.270% |
| blunders | 247,623 | 178,856 |
| mate-scored | 6.14% | 7.55% |
| date span | 2026-06-01 to 06-30, **30 days** | same |
| games per player | 1.45 | - |

Annotated fraction of the whole file: **10.37%**.

**The one-week sample was not a sampling artefact.** Base rate 3.984% then,
3.952% now, across a 4x larger sample spanning the whole month instead of one
week. Rapid 4.254% then, 4.270% now. Decile monotonicity holds at all three
thresholds with 0 up-steps, D1/D10 ratio 2.91x (was 2.90x). Busiest day is
3.68% of rows against a uniform 3.33%. That stability is worth its own short
README section: it is evidence the findings are properties of the data rather
than of the sample.

Note `games per player` rose from 1.18 to **1.45**, so the connected-component
split matters slightly more here than it did on the week sample. Worth
re-checking the leakage null result on this data.

## Phase 7 framing, settled before the code was written

The decomposition in the original scope,
`P(blunder) = P(available) x P(picks | available)`, is nearly vacuous here.
`blunder_available` is true in **82.8%** of positions at depth 8 (n=500),
because "does ANY of 31 legal moves lose 20%+" is almost always yes. Structural,
not small-sample. Confirmed at n=150,000: **81.5%**.

**`frac_blunder_moves` is the availability measure**: median 0.271, p90 0.895
on the n=500 probe; **0.250 and 0.886** on the full 150k. It is also exactly
what uniform-random play would score, which makes the selection term
interpretable.

**Decompose in logs, on BIN AGGREGATES.** Two reasons the obvious forms fail:

- A product-of-means form leaves a covariance term. `E[p] = E[f]E[s] + Cov(f,s)`
  and the covariance is real, since sharp positions are both blunder-dense and
  handled differently. Logs are additive, so no residual.
- A PER-POSITION log decomposition is undefined: `p_i` is latent, we observe a
  binary outcome, so `log p_i` is `log 0` on every non-blunder. It would also
  hit Jensen (`E[log f] != log E[f]`) and end up explaining the geometric mean
  while the 3.95% everyone quotes is arithmetic.

Binning first fixes both. Within bin b take arithmetic means, define
`s_b := p_b / f_b`, and `log p_b = log f_b + log s_b` holds exactly:

    availability share = D log f / D log p
    selection share    = D log s / D log p        (sum to 1, no residual)

**Weighting must match on both sides.** An early draft mixed the
population-weighted base rate (3.952%) with a `frac` from the equal-strata
sample, giving 9.2x better-than-random instead of the correct **8.7x**.

**The 5.7x -> 16.5x rating sweep is a PREDICTION, not a measurement.** It is
computed by holding `frac_blunder_moves` constant across rating, which is
precisely the hypothesis the run exists to test. Conditional until the data
lands. **It landed, and the hypothesis held — see "Phase 7 results" below.**

### Self-test: what it does and does not show

`decompose.py --self-test` plants a known split and checks recovery over
**25 seeds**, not one:

| planted | mean recovered | sd | bias | residual |
|---|---|---|---|---|
| 0% | +0.00% | 0.02% | +0.00% | 6.7e-16 |
| 30% | +30.04% | 0.61% | +0.04% | 6.7e-16 |
| 50% | +50.06% | 1.02% | +0.06% | 4.4e-16 |
| 85% | +85.11% | 1.73% | +0.11% | 4.4e-16 |
| 100% | +100.13% | 2.03% | +0.13% | 4.4e-16 |
| -25% | -25.03% | 0.51% | -0.03% | 4.4e-16 |

**There is no bias.** A first version judged on a single seed and reported an
apparent "+0.4 to +0.7pp mid-range bias"; across seeds it is +0.04% to +0.13%
and flips sign as n grows. What looked like bias is per-run SCATTER: sd 1.73%
at 100k rows per band, 0.56% at 1M, exactly 1/sqrt(n). Judging an unbiased
estimator on one draw against a tight tolerance fails at random.

Consequence for the real run: 150k positions across 4 bands is ~37.5k per band,
so expect scatter WIDER than 1.7%, before accounting for real data being
messier than synthetic. **The bootstrap interval is load-bearing, not
decoration.**

The negative case (-25%) confirms a term moving against the total is reported
as a negative share rather than renormalised into [0,1].

A sixth case was added after the real run, because the real data turned out to
need it: see "the floor case" below.

## Phase 7 results: the availability channel is empty

150,000 positions, depth 8, every legal move evaluated (30.7 on average, so
~4.6M child evaluations). Equal strata of 37.5k per rating band. 2.1 h on 7
workers at 20.1 pos/s. FEN guard passed on 150,000/150,000.

**The headline, on the mechanically controlled contrast:**

| band | n | P(blunder) | E[frac] | s = p/f | vs random |
|---|---:|---:|---:|---:|---:|
| <1200 | 31,798 | 7.192% | 38.4% | 0.1874 | 5.3x |
| 1200-1600 | 33,246 | 5.144% | 38.4% | 0.1340 | 7.5x |
| 1600-2000 | 33,995 | 3.880% | 38.6% | 0.1006 | 9.9x |
| 2000+ | 35,031 | 2.595% | 38.6% | 0.0671 | 14.9x |

    D log p = -1.0195   D log f = +0.0069   D log s = -1.0264

**Availability -0.7%, selection +100.7%.** Bootstrap 95% CI on availability
**[-1.9%, +0.4%]**, clustered by game. The slope version over all four bands
agrees at -0.8% / +100.8%, so no single band is driving it.

`E[frac]` is flat to within 0.7% relative across 800 Elo, and flat at every
quantile, not just at the mean (p50 0.278 -> 0.300, p75 0.667 -> 0.600). **A
2000 and a 1000 stand in front of minefields of the same thickness. The 2000
steps in a third as often.** Better-than-random rises 5.3x -> 14.9x.

That is the claim the phase was scoped to test, and it is the strong version.

### The uncontrolled number is an artefact, and it is worth showing

Without the floor control the same contrast reads **availability -10.6%**, CI
[-12.6%, -8.8%], which *excludes* zero. That is entirely mechanical:

| band | below floor | E[frac] all | E[frac] eligible |
|---|---:|---:|---:|
| <1200 | 14.90% | 0.3266 | 0.3838 |
| 1200-1600 | 11.41% | 0.3400 | 0.3838 |
| 1600-2000 | 9.40% | 0.3493 | 0.3856 |
| 2000+ | 6.79% | 0.3603 | 0.3865 |

Below `best_child_win <= 22.46` no child can be 20 win% worse than the best, so
`frac` is *forced* to 0. Weak players sit there **2.2x more often**, those
censored zeros drag their mean down, and `f` appears to RISE with rating. The
sign is negative, which reads as "strong players face THICKER minefields" if
taken at face value.

Two independent confirmations that it is an artefact and not an effect:

- **`s = p/f` is identical in both tables to four decimals** (0.1875/0.1874,
  0.1342/0.1340, 0.1008/0.1006, 0.0672/0.0671). The floor removes matched mass
  from `p` and from `f`, so it cannot move their ratio. If the control were
  removing a real effect, `s` would move.
- **The self-test reproduces it from censoring alone.** A new floor case plants
  *zero* availability effect, then censors `frac` to 0 on 15% of weak rows and
  7% of strong rows — the observed ineligible shares. The uncontrolled contrast
  reads **-8.53%**, against -10.6% on real data; the control recovers
  **-0.00%**. Almost the whole real -10.6% is accounted for by censoring.

So report the controlled -0.7%, and report -10.6% next to it as the thing the
control removes. The gap between them is the more interesting half.

### The CI is a precise null, not a weak one

The question worth asking of a null is whether the interval is wide enough that
"indistinguishable from zero" only means "underpowered". Here it is not:
**[-1.9%, +0.4%] is 2.3 points wide.** The data rules out an availability
channel worth more than a couple of points; it does not merely fail to find
one. State it as a null, not as a bound.

Consistent with the self-test scatter: sd was 1.73% at 100k rows per band for
an 85% planted split, and the share's scatter shrinks as the planted share
does, so a near-zero share at 33k per band is estimated tightly.

### The bootstrap is clustered by game, and it barely matters

The sample averages **2.05 positions per game**, and 79% of positions come from
a game contributing more than one. Positions in a game share a player, a rating
and an opening, so an iid bootstrap over positions is the wrong one.

Measured at 3,000 resamples, the **design effect is 1.07 to 1.11**, i.e. the
standard error is 3-6% larger than iid. Small, because the stratified sampler
takes ~2 of a game's 66 positions and they land far apart. The conclusion is
unchanged either way. Clustered is still the default: it costs nothing, and
"we checked" is worth more than "it probably does not matter".

### Within eval bands: three of four flat, one is not

Holding `winpct_before` fixed (the causal control, on eligible rows):

| eval band | n | availability | 95% CI |
|---|---:|---:|---|
| losing | 14,976 | -4.5% | [-12.6%, +1.1%] includes 0 |
| balanced | 71,374 | -2.4% | [-4.1%, -0.6%] |
| better | 21,880 | +3.0% | [+1.1%, +5.2%] |
| **winning** | 22,034 | **-28.0%** | [-41.1%, -18.6%] |

The first three are near zero and bracket it, so the main result survives the
causal control. **The `winning` band is a real exception and should not be
buried.** There `E[frac]` climbs 39.9% -> 46.4% with rating: strong players in
won positions reach positions where a larger fraction of moves throws away 20
win%, because there is more to throw away. It is also the band where the top
cell is thinnest (4,064) and where the rating composition is most skewed
(7,181 weak against 4,064 strong), so treat it as a signposted anomaly worth
one sentence, not a finding to lead with.

### Integrity: two floors, not one

15,930 positions sit below the mechanical floor. `frac_blunder_moves` is
exactly 0 in all of them (max 0.000) and `blunder_available` is False in all of
them, as the arithmetic requires.

**6 of them carry a positive label** (0.038%). Not a bug: the label's floor is
on `winpct_before` from Lichess' deep annotation, while this floor is on
`best_child_win` from our depth-8 sweep measured against the best child. Two
different quantities, so a handful disagree. The report prints the rate and
warns above 1%.

## Phase 9: external validation on puzzles — POSITIVE BUT WEAK

The one check that uses a difficulty scale measured outside this dump.
20,000 puzzles sampled from 1,902,527 surviving filters (RatingDeviation < 80,
NbPlays > 200, mate themes dropped) out of 6,100,960 published.

Model difficulty is **D10**: the rating at which predicted P(blunder) crosses
10%, so the model lands on the Elo scale and can be correlated with the
puzzle's own Elo directly.

| quantity | value |
|---|---|
| uncensored puzzles | 16,031 |
| censored (never crosses 10%) | 3,969 (**19.8%**) |
| **Spearman rho** | **+0.1126** |
| Pearson r | +0.1134 |
| decile monotone up-steps | 8 of 9 |

**This does not meet the acceptance bar.** Phase 9 was scoped to want 0.4-0.6
as a strong result. We got +0.11. At n=16,031 the standard error is ~0.008, so
the correlation is ~14 se from zero — **definitively positive and definitively
small**. Report it that way; do not round it up into a success.

Every theme is positive, so it is not one group carrying a spurious average:

| theme | n | rho | | theme | n | rho |
|---|---:|---:|---|---|---:|---:|
| pin | 1,236 | +0.187 | | fork | 2,781 | +0.139 |
| skewer | 476 | +0.163 | | middlegame | 7,884 | +0.130 |
| discoveredAttack | 1,200 | +0.150 | | deflection | 1,006 | +0.112 |
| endgame | 7,349 | +0.097 | | opening | 798 | +0.091 |
| hangingPiece | 650 | +0.051 | | | | |

### Contamination check: was this measuring the rating-sweep bug?

D10 is computed entirely from the rating sweep, so a flattened sweep would
compress D10 and attenuate the correlation — producing exactly the
weak-but-positive result observed. The bug was found in the same session, so
this had to be ruled out before the availability story could be believed.

**It was not contaminated.** `validate_puzzles.py` imported `rating_grid` from
the moment it was written (16:42 fix, 16:46 script, 16:50 run), but file order
is weak evidence. The decisive check re-ran the correlation both ways on the
same 6,000 puzzles:

| sweep | censored | Spearman |
|---|---|---|
| correct (`rating_grid`) | **19.8%** | **+0.1250** |
| broken (`mover_elo` alone) | 59.9% | +0.0931 |

**The censoring rate is the fingerprint.** The real run reported 19.8%, which
matches the correct sweep exactly and is nowhere near 59.9%. Censoring is a
property of the curve shape that no bookkeeping error could fake.

The mechanism was real, though: the bug attenuates the correlation by ~25%
(+0.1250 -> +0.0931) and more than triples censoring. It simply was not what
happened here. Worth recording because the reasoning was correct and would have
been decisive under slightly different timing.

### The compression is the real story

Mean D10 by puzzle-rating decile rises **1464 -> 1650**, i.e. **186 Elo**, while
the puzzle rating underneath it rises **1039 -> 2241**, i.e. **1202 Elo**. The
model's difficulty scale responds about **6x too weakly**. The ordering is
almost perfect (8 of 9 up-steps) but the magnitude is badly compressed.

So the honest sentence is: **the model ranks human difficulty in the right
order and dramatically understates its range.**

### Why, and which explanations are testable

1. **Puzzle selection removes the variance the model is good at.** Every puzzle
   has a blunder available and a unique best move. Much of the model's skill is
   separating dangerous positions from safe ones, and that axis is constant by
   construction here. This is the leading explanation and it is consistent with
   Phase 7: the availability channel is exactly what puzzle selection holds
   fixed, so what remains is the selection channel alone.
2. **Different tasks.** A puzzle rating measures "can you FIND the one best
   move". Our label measures "do you give away 20 win%". Related, not the same.
3. **No engine features.** Tactical depth is what makes puzzles hard, and the
   engine_free model never sees an eval.

Explanation 3 is the obvious next test and **it is not currently runnable**.
`full_assisted` needs `winpct_momentum_2`, `winpct_momentum_4` and
`prev_own_move_was_blunder`, all of which require the GAME HISTORY a standalone
puzzle does not have; `full_assisted_noclock` needs them too. Testing it would
mean training a static-assisted feature set (eval features, no history), which
is a new model, not a new script. Worth doing, but say plainly that the
assisted arm of Phase 9 has not been run rather than implying it was.

### Why the engine_free NO-CLOCK model is the right one here

A puzzle has no clock, no time control and no opponent. `full_free_noclock`'s
only clock-like feature is `halfmove_clock`, a FEN field, so it is **exactly**
computable on a puzzle with nothing imputed and no engine call. The clock
ablation put the cost of that substitution at 1.4% of Brier skill. Using
`full_free` instead would have meant inventing six clock features carrying ~9%
of the model's gain, so the ablation is what made this phase clean.

`validate_puzzles.py` refuses to run on a model carrying clock features rather
than silently imputing them.

### The caveat that has to travel with the headline

`s` is measured against **uniform-random play**, because `frac_blunder_moves`
is exactly what random move selection would score. Humans do not sample
uniformly from legal moves, so "9.9x better than random" is a benchmark against
random, not a statement about human search. The honest denominator is a
human-policy baseline (Maia); out of scope, and the README must say so rather
than let "selection" imply a cognitive mechanism.
## The challenge game, v2 (interactive board)

Rebuilt from multiple-choice buttons to a real board. The enabling fact:
`eval_children.py --per-move` already evaluated EVERY legal child, so the page
knows the truth about any move the player makes. Free move selection costs
nothing extra -- about 31 moves at ~40 bytes is 1.2 KB per position.

What it now does:

- **Click a piece, then its destination.** Legality comes from the stored move
  list, so the page needs no chess engine and no chess.js. Selecting a piece
  shows its legal destinations, with capture squares ringed.
- **Eval bar** driven by the exact Stockfish win% of the move played, with a
  marker at `best_child_win` so you can see what the position was worth before
  you touched it.
- **Take back.** Try any other move and see its evaluation. Only the FIRST
  attempt counts towards the score, so exploring is free but cannot be farmed.
- **Show best** reveals the best and worst legal moves and how many of the
  legal moves lose 20%+.
- Pieces are the twelve Cburnett SVG groups lifted from `python-chess` at build
  time and embedded once as a sprite, so the page has zero external
  dependencies.

Dropping the 60 pre-rendered board SVGs in favour of FEN + one sprite took
`challenge.json` from **1.9 MB to 0.1 MB**, a 19x cut.

### Verification

Board logic is hand-rolled, so it was checked against `python-chess` as ground
truth. The JS `apply()` was ported to Python and diffed square by square over
**759,608 positions from 4,000 random games**: **0 mismatches**, covering 302
castlings, 77 en passants and 5,126 promotions. An earlier pass had tested
eleven hand-picked cases; the exhaustive version is what should be quoted,
because the failure modes here are the positions nobody thinks to pick.

Data checks on the generated JSON, all passing on 60 positions: every stored
move legal, every SAN matching, stored move count equal to the true legal move
count, the `white` flag agreeing with the FEN, all twelve sprite ids present
and matching the names the page builds, and every position containing both a
blunder and a safe move.

Page checks: all 18 JS element references resolve to elements that exist, no
element is defined and unused, and there are zero external URLs, so the page is
genuinely self-contained.

### Three regressions the rewrite introduced, and one bug

The v2 rewrite arrived as a standalone file and silently dropped guards the
committed version had. All three are restored; the lesson is that a rewrite is
a diff against the current file, not a fresh draft.

1. **`rating_grid` was re-implemented locally** instead of imported from
   `build_features`. Semantically correct, but it recreates exactly the
   single-caller duplication that let the original sweep bug survive. Now
   imported.
2. **The `row_id`/FEN consistency guard was gone.** That is the check that
   stops a mismatched `--features` parquet resolving every id to an unrelated
   position, silently. Restored.
3. **The `check_sweep_span` gate was gone**, i.e. the regression test written
   specifically to stop the flattened-curve bug recurring. Restored, and it
   passes at 13.53x.

**The eval bar was inverted.** It converted the win% to White's perspective and
then drew `100 - w` upward from the bottom, so the light fill grew as the side
to move got *worse*: a position won 90% by the mover rendered as a nearly empty
bar. Fixed by dropping the conversion entirely -- the board already auto-flips
so the side to move is at the bottom, and every number in the data is already
from the mover's point of view, so the fill is now just the mover's win% and
there is no sign left to invert.

Known limitation, stated on the page: promotions auto-queen. The data carries
underpromotions and `apply()` handles them correctly, but the click-to-move UI
has no way to ask which piece, so it always picks the queen.

## Phase 8: the difficulty curves

Four REAL positions taken from the dataset rather than composed. Composed FENs
kept failing to have the property their caption claimed -- a hand-built
"hanging piece" position measured `hanging_own = 0` -- so the positions are
selected by their measured features and the captions are therefore true.

Predicted P(blunder), 800 -> 2400 Elo, `full_free_noclock`:

| position | at 800 | at 2400 | span |
|---|---:|---:|---:|
| Sharp middlegame (30 pieces, tension 7, 45 legal) | 22.67% | 2.75% | 8.2x |
| Piece hanging (own queen loose, 43 legal) | 22.39% | 2.70% | 8.3x |
| Quiet endgame (12 pieces, no tension, 12 legal) | 13.19% | 3.12% | 4.2x |
| Opening (move 2) | 0.57% | 0.04% | 14.6x |

**The two that start together do not stay together.** The sharp middlegame and
the hanging piece both open near 22.5%, but by 1200 they have separated by
about 7 points: a loose piece is something stronger players simply stop
missing, while a sharp middlegame stays dangerous well up the rating scale.
That divergence is the figure's actual content, and it is not visible from any
single-number difficulty measure.

### The extrapolation caveat is implemented, and it does not fire

A rating sweep is an extrapolation -- a 2200 rarely reaches the position a 1200
blundered in -- so the script counts TRAINING rows in each position's
neighbourhood (same rating +/-100, legal moves +/-6, material +/-150cp, total
pieces +/-6) and draws unsupported stretches faint.

**The support floor is derived, not picked.** At a 3.952% base rate, 50 blunder
events -- the usual floor for a stable rate estimate -- needs **1,265 rows**. An
arbitrary round number like 200 rows would mean ~8 events, which cannot pin
down a probability.

At that floor all four positions are supported across the whole 800-2400 range
(33/33 points, medians 23k to 168k rows), so nothing greys out. That is a real
result of having 6.27M training rows, not a broken check: support peaks at
1600-1800 and falls at both extremes exactly as it should, and the mechanism
was verified to fire by re-running at a 30,000-row floor, where the quiet
endgame correctly collapses to 9 of 33 supported points.

Stated plainly: **on this dataset these curves are interpolation, not
extrapolation.** On a smaller sample the same code would grey out the ends.

## The landing page generates its own numbers

`build_site.py` reads `metrics/*.json` at build time rather than carrying
hardcoded figures. A hand-written page drifts the moment a model is retrained,
and stale numbers beside a live demo are worse than no page. If a metrics file
is missing the section says so instead of inventing a value.

This forced one upstream change: `evaluate.py` printed the within-band
calibration table but only ever wrote it to a log, so it now also stores typed
`bands` records in its JSON. Re-parsing formatted strings back into numbers is
exactly how a site starts disagreeing with its data.

**One rendering bug worth recording.** Band labels are strings like `<1200`.
Written unescaped into a table cell the browser reads `<1200` as the start of a
malformed tag and swallows the rest of the cell, so the row vanishes from the
rendered page while looking perfectly fine in the HTML source. Caught by
counting rendered rows (3) against expected rows (4), not by reading the
output. Everything from the metrics files is escaped now.

Verified after generation: 0 missing-metric placeholders, 0 unformatted
template braces, balanced tags (41 `<td>`/41 `</td>`, 12 `<tr>`/12 `</tr>`),
and every headline number cross-checked to appear on the page exactly as it
appears in the JSON it came from.
