# Chess Blunder Predictor
#
# `make test` runs the whole pipeline on the committed sample and asserts the
# project's claims. It needs no download and finishes in a couple of minutes,
# which is what CI runs.
#
# The full-data targets expect the 28 GB monthly dump in data/. Run `make help`
# for the map, or `python src/status.py`, which reads the filesystem and tells
# you exactly which command comes next.

PY      ?= python
POOL    ?= blitz
DATA    ?= data/features_$(POOL)_full.parquet
SPLITS  ?= data/splits_$(POOL)_full.parquet
MODEL   ?= models/full_free
CHILDREN?= data/children_150k.parquet
WORKERS ?= 7

# sample pipeline artefacts, all under build/ so they never shadow real data
B       := build
SAMPLE  := data/sample_positions.parquet
GAMES   := data/sample_games.pgn

.PHONY: help setup test sample features split baselines train evaluate \
        ablation children decompose curves puzzles challenge site all clean

help:
	@echo "make test        pipeline on the committed sample + assertions (~2 min)"
	@echo "make setup       create .venv and install pinned requirements"
	@echo ""
	@echo "full-data targets, need the monthly dump:"
	@echo "  features split baselines train evaluate ablation"
	@echo "  children decompose curves puzzles challenge site all"
	@echo ""
	@echo "make sample      re-cut the committed sample from the full data"
	@echo "make clean       remove build/"

setup:
	$(PY) -m venv .venv
	.venv/bin/pip install -r requirements.txt || .venv/Scripts/pip install -r requirements.txt

# ---------------------------------------------------------------------------
# the sample pipeline: this is the test
# ---------------------------------------------------------------------------

$(B)/pos.parquet: $(GAMES)
	@mkdir -p $(B)
	$(PY) src/parse_lichess.py $(GAMES) --out $@ --categories blitz,rapid

$(B)/feat.parquet: $(B)/pos.parquet
	$(PY) src/build_features.py --data $< --out $@ --workers 2

$(B)/splits.parquet: $(B)/feat.parquet
	$(PY) src/make_splits.py --data $< --out $@

$(B)/model.txt: $(B)/feat.parquet $(B)/splits.parquet
	$(PY) src/train.py --data $(B)/feat.parquet --splits $(B)/splits.parquet \
	  --feature-set engine_free --n-trials 1 --rounds 300 --out $(B)/model

test: $(B)/model.txt
	$(PY) src/decompose.py --self-test
	$(PY) src/build_features.py --self-test
	$(PY) src/ci_check.py --positions $(B)/pos.parquet \
	  --features $(B)/feat.parquet --splits $(B)/splits.parquet \
	  --model $(B)/model

sample:
	$(PY) src/make_sample.py --data data/positions_2026-06_$(POOL).parquet \
	  --n-rows 45000
	@echo "sample_games.pgn needs the raw dump; see src/make_sample.py --pgn"

# ---------------------------------------------------------------------------
# full data
# ---------------------------------------------------------------------------

features:
	$(PY) src/build_features.py --data data/positions_2026-06_$(POOL).parquet \
	  --out $(DATA) --workers $(WORKERS)

split:
	$(PY) src/make_splits.py --data $(DATA) --out $(SPLITS)

baselines:
	$(PY) src/baselines.py --data $(DATA) --splits $(SPLITS) \
	  --out metrics/baselines_$(POOL)_full.json

train:
	$(PY) src/train.py --data $(DATA) --splits $(SPLITS) \
	  --feature-set engine_free --out models/full_free
	$(PY) src/train.py --data $(DATA) --splits $(SPLITS) \
	  --feature-set engine_assisted --out models/full_assisted

ablation:
	$(PY) src/train.py --data $(DATA) --splits $(SPLITS) \
	  --feature-set engine_free --no-clock --out models/full_free_noclock
	$(PY) src/train.py --data $(DATA) --splits $(SPLITS) \
	  --feature-set engine_assisted --no-clock --out models/full_assisted_noclock

evaluate:
	$(PY) src/evaluate.py --data $(DATA) --splits $(SPLITS) \
	  --models models/full_free models/full_free_noclock \
	           models/full_assisted models/full_assisted_noclock \
	  --out figures/ --metrics-out metrics/evaluation_$(POOL)_full.json

children:
	$(PY) src/eval_children.py --data $(DATA) --out $(CHILDREN) \
	  --n-positions 150000 --depth 8 --workers $(WORKERS) --per-move --resume

decompose:
	$(PY) src/decompose.py --children $(CHILDREN) --features $(DATA) \
	  --out metrics/decomposition.json --report reports/decomposition.txt

curves:
	$(PY) src/difficulty_curves.py --data $(DATA) \
	  --model models/full_free_noclock --out figures/

puzzles:
	$(PY) src/validate_puzzles.py --puzzles data/lichess_db_puzzle.csv.zst \
	  --model models/full_free_noclock --out figures/

challenge:
	$(PY) src/build_challenge.py --children $(CHILDREN) --features $(DATA) \
	  --model $(MODEL) --out docs/

site:
	$(PY) src/build_site.py --metrics metrics/ --figures figures/ --out docs/

all: features split baselines train ablation evaluate decompose curves \
     puzzles challenge site

clean:
	rm -rf $(B)
