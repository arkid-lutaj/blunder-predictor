# Chess Blunder Predictor

**A human error model: P(blunder | position, rating).** Not an engine clone.
The output is a calibrated probability — *"a 1400 blunders here 23% of the
time, a 2000 blunders 4%"* — and the calibration holds inside every rating
band, not just on average.

Built on one full month of Lichess blitz: **86.5M games scanned, 6.68M
evaluated positions, 1.26M held-out test rows.**

![difficulty curves](figures/difficulty_curves.png)

Four real positions from the dataset. A sharp middlegame stays dangerous even
for strong players; a hanging piece is something they simply stop missing; the
opening is safe for everyone. Position difficulty is not one number.

---

## The headline result

**Stronger players do not face fewer chances to blunder. They decline them.**

Every legal move in 150,000 positions was evaluated by Stockfish — about 4.6M
child evaluations — to measure how *many* of the available moves were blunders.
Splitting the rating effect into two channels:

| channel | share of the rating effect | 95% CI |
|---|---:|---|
| **availability** — do weak players face thicker minefields? | **−0.7%** | [−1.9%, +0.4%] |
| **selection** — do they step in them more often? | **+100.7%** | — |

The fraction of legal moves that are blunders is **flat across 800 Elo** — 38.4%
at &lt;1200 against 38.6% at 2000+, and flat at every quantile, not just the
mean. What changes is how often a human picks one:

| rating band | P(blunder) | % of moves that blunder | better than random |
|---|---:|---:|---:|
| &lt;1200 | 7.19% | 38.4% | 5.3x |
| 1200–1600 | 5.14% | 38.4% | 7.5x |
| 1600–2000 | 3.88% | 38.6% | 9.9x |
| 2000+ | 2.59% | 38.6% | **14.9x** |

The interval is 2.3 points wide and brackets zero, so this is a **precise null**
— the data rules out an availability channel worth more than a couple of
points, rather than failing to detect one.

---

## Model performance

Test set 1,257,458 rows, base rate 3.909%. Player-disjoint split.

| model | Brier skill | PR-AUC | PR lift | ROC-AUC | ECE |
|---|---:|---:|---:|---:|---:|
| constant baseline | +0.0000 | 0.0391 | 1.00x | 0.500 | — |
| rating only | +0.0037 | 0.0536 | 1.37x | 0.587 | 0.0012 |
| **engine_free** (56 features, no engine at inference) | **+0.0496** | 0.1258 | 3.22x | 0.792 | 0.0012 |
| **engine_assisted** (+ Stockfish eval) | **+0.1238** | 0.2389 | 6.11x | 0.871 | 0.0010 |

`engine_free` beats rating-only by **13.4x** on Brier skill. Adding an engine
roughly doubles that again (2.30x on the eligible subset, which is the honest
comparison — see *the structural floor* below).

Accuracy is never reported: at a 3.9% base rate, predicting "no blunder" always
scores 96%.

### Calibration holds *within* rating bands

This is the result the product rests on.

| band | rows | observed | predicted | ECE |
|---|---:|---:|---:|---:|
| &lt;1200 | 172,939 | 5.94% | 6.11% | 0.0027 |
| 1200–1600 | 352,209 | 4.52% | 4.62% | 0.0015 |
| 1600–2000 | 385,198 | 3.66% | 3.65% | 0.0009 |
| 2000+ | 347,112 | 2.54% | 2.56% | 0.0011 |

Aggregate calibration can be right by accident — over-predicting weak players
cancels under-predicting strong ones. It isn't happening here, which is what
makes *"a 1400 blunders here 23% of the time"* a defensible sentence rather
than a slogan.

Calibration is native, not fitted: LightGBM with `is_unbalance=False` came out
calibrated and the isotonic step changes ECE by ~0.0001.

### The honest ceiling

Top predicted decile: **13.9% observed against a 3.9% base rate** (engine_free,
3.5x) and **21.1%** (engine_assisted, 5.4x). This is a **difficulty signal, not
a blunder detector**, and it should be described that way.

---

## What else was measured

**Time pressure explains almost nothing.** Removing every clock feature costs
engine_free 1.4% of its Brier skill and engine_assisted nothing at all
(+0.0496 → +0.0489, and +0.1238 → +0.1241). `move_number` absorbs the loss,
rising 9.59% → 11.83% in importance as a proxy for elapsed time, while
`mover_elo` is unmoved. The original hypothesis was that time pressure would
dominate; it does not. **Rating predicts blunders essentially undiminished
after controlling for time pressure.**

**Player leakage is a measured null.** Trained end-to-end on a naive game-hash
split and compared like for like: +0.0496 (player-disjoint) against +0.0499
(naive), ROC-AUC 0.7920 against 0.7910. No difference — because players recur
only 1.45 times in this dataset, so there is almost nothing about a player to
memorise. The player-disjoint split is kept anyway: it is principled, costs
nothing, and would matter on a denser graph. A null you went looking for is
better evidence of care than a convenient positive.

**External validation is positive but weak.** Against 20,000 Lichess puzzles
whose difficulty ratings come from real human solve attempts and were never
seen in training:

![puzzle validation](figures/puzzle_validation.png)

Spearman **+0.113** on 16,031 uncensored puzzles, monotone across 8 of 9
deciles, every theme positive. That is ~14 standard errors from zero and well
below the 0.4–0.6 that would have been a strong result. The real finding is
**compression**: the model's difficulty scale moves 186 Elo while the human
scale under it moves 1,202. **It ranks human difficulty in roughly the right
order and badly understates its range.**

The leading explanation is consistent with the headline result — puzzles are
selected so a blunder is always available, which holds the availability channel
fixed by construction and leaves only selection. That is an explanation, not a
tested claim, and it is flagged as such.

---

## Try it

An interactive board where every legal move was pre-evaluated by Stockfish, so
the page knows the truth about whatever you play — no engine, no server, no
dependencies.

```bash
python -m http.server -d docs 8000   # then open localhost:8000/challenge.html
```

Play a move and it tells you what it cost, what the best move was, and what the
model predicted a player of your rating would do. Play ten and it reports the
rating whose predicted blunder rate matches yours.

---

## Method, and the parts that bite

**The label.** `blunder = (win% before − win% after) > 20`, from the mover's
point of view, where
`win% = 50 + 50 · (2/(1 + exp(−0.00368208 · cp)) − 1)` and `cp` is clamped to
±1000. Positions where either side of the transition is a forced mate are
excluded (6.1% of rows).

**The structural floor — a real blind spot, stated plainly.** Because `cp` is
clamped, win% is clamped to [2.46, 97.54], so a 20-point drop is arithmetically
impossible once `winpct_before ≤ 22.46` (`cp < −337`). **12.5% of rows cannot
register a blunder at all**, and they contain exactly zero — which
independently confirms the arithmetic. Losing a rook when you are already down
a queen does not count. Every engine comparison is therefore also reported on
the eligible subset, where that free win disappears.

**Splitting by player, not position.** Positions in one game share a player, an
opening and a clock situation. Splitting randomly leaks all three. Connected
components of the player co-occurrence graph are assigned whole; a naive
game-hash split leaks **19.0% of its test players** into training.

**Never used as features:** anything knowable only after the move —
`cp_after`, `winpct_after`, `win_drop`, `clk_after`, `result`, and the time
spent on the *current* move, which is chosen simultaneously with the move
itself. Time spent on *previous* moves is fair game.

**Metrics.** Log loss, Brier skill against the base rate, PR-AUC against the
positive rate, ECE, and reliability diagrams within rating bands. ROC-AUC is
reported but never alone — it is invariant to class balance, so a high value
coexists happily with terrible precision here.

---

## Limitations

- **The label is blind in lost positions** (the structural floor above).
- **The annotated subset is not a random sample.** Lichess analysis is opt-in;
  these players sit ~135 Elo above the pool median with a 1.5x wider spread.
  Claims about 900-rated blitz are thinner than the row count suggests.
- **The selection ratio is measured against random play.** `frac_blunder_moves`
  is exactly what uniform-random move choice would score. Humans do not sample
  uniformly, so "14.9x better than random" is a benchmark against random, not a
  claim about human search. The honest denominator is a human-policy baseline
  (Maia); out of scope here.
- **External validation is weak** (+0.113), and the explanation for *why* has
  not itself been tested.
- **Blitz only.** Blitz and rapid are separate Glicko2 pools — a 1500 blitz and
  a 1500 rapid are different players — so they are never mixed. Rapid is parsed
  but not modelled here.
- **30.5% of positives sit within 5 win% of the threshold.** That is
  irreducible label noise and it caps what any model can score.

---

## Reproducing

```bash
python src/status.py     # reads the filesystem, prints the exact next command
```

Every script takes explicit `--data` / `--out` paths and hardcodes no
filenames. The pipeline, one line each:

| step | script | time |
|---|---|---|
| parse the 28 GB dump → positions | `parse_lichess.py` | 20–35 min |
| validate the label | `verify_labels.py` | 2 min |
| positions → features | `build_features.py` | ~90 min |
| player-disjoint folds | `make_splits.py` | 1–4 min |
| baselines (the bar to beat) | `baselines.py` | 2 min |
| train (4 variants) | `train.py` | 5 min each |
| evaluate + figures | `evaluate.py` | 5 min |
| Stockfish over every legal move | `eval_children.py` | 2–3 h |
| availability vs selection | `decompose.py` | 3 min |
| difficulty curves | `difficulty_curves.py` | 2 min |
| external validation | `validate_puzzles.py` | 5 min |
| the interactive demo | `build_challenge.py` | 1 min |

Several scripts carry `--self-test` and run without data:

```bash
python src/decompose.py --self-test        # planted-split recovery, 25 seeds
python src/build_features.py --self-test   # the rating sweep and its gate
```

**[FINDINGS.md](FINDINGS.md)** is the running record — every number here comes
from there, including the bugs, the dead ends, and the things that looked
alarming and were not.

## Layout

```
src/       every script            data/     parquet (gitignored)
figures/   plots                   models/   boosters (gitignored)
metrics/   json, every number      reports/  validation output
docs/      the playable demo
```
