# Chess Blunder Predictor

[![ci](https://github.com/arkid-lutaj/blunder-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/arkid-lutaj/blunder-predictor/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![demo](https://img.shields.io/badge/demo-live-6ea8fe.svg)](https://arkid-lutaj.github.io/blunder-predictor/challenge.html)

**How often does a human blunder here?** Not "what is the best move" - a
calibrated probability of human error, by rating. A 1400 blunders in this
position 23% of the time; a 2000 blunders 4%.

### [Play it here](https://arkid-lutaj.github.io/blunder-predictor/challenge.html)

Pick the move you would actually play. Every legal move was scored by Stockfish
in advance, so the page knows instantly what yours cost. After ten positions it
estimates your rating from how often you blundered in positions of known
difficulty.

---

![difficulty curves](figures/difficulty_curves.png)

Four real positions. The sharp middlegame stays dangerous even at 2400. A
hanging piece stops mattering past about 1400. The opening is safe for
everyone. One number cannot describe difficulty.

## The finding

**Stronger players are not handed easier positions. They get the same positions
and choose better inside them.**

Every legal move in 150,000 positions went through Stockfish, so "what fraction
of the available moves lose the game" is measured, not guessed.

| rating | blunders | share of legal moves that blunder | vs random play |
|---|---:|---:|---:|
| under 1200 | 7.19% | 38.4% | 5.3x |
| 1200-1600 | 5.14% | 38.4% | 7.5x |
| 1600-2000 | 3.88% | 38.6% | 9.9x |
| 2000+ | 2.59% | 38.6% | **14.9x** |

That middle column is flat. The minefield is the same thickness at every level;
only the stepping changes. Splitting the rating effect formally:
**availability -0.7%** (95% CI -1.9% to +0.4%), **selection +100.7%**.

## Does it work

1,257,458 held-out rows, no player on both sides of the split.

| model | Brier skill | ROC-AUC | ECE |
|---|---:|---:|---:|
| rating only | +0.0037 | 0.587 | 0.0012 |
| no engine, 56 features | +0.0496 | 0.792 | 0.0012 |
| with a Stockfish eval | **+0.1238** | 0.871 | 0.0010 |

13x better than knowing the rating alone. Accuracy is never reported: at a 3.9%
base rate you score 96% by always guessing "no blunder".

**Calibration holds inside each band**, which is what makes the headline claim
usable rather than decorative.

| band | observed | predicted |
|---|---:|---:|
| under 1200 | 5.94% | 6.11% |
| 1200-1600 | 4.52% | 4.62% |
| 1600-2000 | 3.66% | 3.65% |
| 2000+ | 2.54% | 2.56% |

The top predicted decile hits 21% against a 3.9% base rate. So it finds
dangerous positions; it does not catch individual blunders.

## Three other results

**Time pressure barely matters.** Removing every clock feature costs 1.4% of
Brier skill, and nothing at all for the engine-assisted model. `move_number`
quietly absorbs the job as a proxy for elapsed time. The original hypothesis
was that time trouble would dominate. It does not.

**Player leakage was a null.** +0.0496 with player-disjoint splits against
+0.0499 with a naive one. Players recur only 1.45 times here, so there is
nothing to memorise. The strict split stays anyway.

**External validation is weak, and that is reported as weak.** Against 20,000
Lichess puzzles rated by real human solve attempts, Spearman **+0.113** - far
short of the 0.4 to 0.6 that would have been a success. The model gets the
ordering of difficulty roughly right and understates its range by about 6x.

![puzzle validation](figures/puzzle_validation.png)

## Limits worth knowing

- **The label is blind in lost positions.** win% is clamped, so a 20-point drop
  is impossible below -337cp. 12.5% of rows can never register a blunder, and
  they contain exactly zero, which is a useful check that the arithmetic works.
  Dropping a rook while down a queen does not count.
- **"14.9x better than random" is a benchmark, not a claim about human search.**
  Random legal-move choice is the denominator. A human-policy baseline like Maia
  would be the right one.
- **The games are not a random sample.** Lichess analysis is opt-in, so these
  players sit ~135 Elo above the pool median.
- **Blitz only.** Blitz and rapid are separate rating pools and are never mixed.

## Running it

A 500-game annotated sample is committed, so the whole pipeline runs on a fresh
clone with no download:

```bash
make setup
make test     # parse, features, splits, train, then assert the claims
```

That last step is what CI runs. The assertions are the README's claims written
as things that can fail:

| assertion | on this sample |
|---|---|
| the label fires on 2-5% of plies | 4.09% |
| no post-move column reaches the model | 56 features, 0 leaked |
| no player is in both train and test | 0 of 200 test players |
| the player split beats a naive one | 0.00% against 4.27% leakage |
| the model beats the base rate | Brier skill +0.0322 |

Each one was checked by injecting the fault and confirming the build goes red.

Full data is roughly 6 hours end to end from the 28 GB dump, most of it
Stockfish. `python src/status.py` reads the filesystem and prints the exact next
command; `make help` lists every stage.

**[FINDINGS.md](FINDINGS.md)** has every number, plus the bugs and dead ends.
**[SPEC.md](SPEC.md)** has the invariants.

## Credit where it is due

Positions come from the [Lichess open database](https://database.lichess.org),
published under CC0. The centipawn-to-win% formula is
[Lichess's](https://lichess.org/page/accuracy), not mine, and is cited at every
point of use. Stockfish and python-chess are used as tools; neither is
redistributed here, and no third-party image, font or SVG is bundled anywhere
in this repo. Full detail in **[THIRD_PARTY.md](THIRD_PARTY.md)**.

The label design, the availability/selection decomposition, the D10 difficulty
measure and everything in `src/` are my own.
