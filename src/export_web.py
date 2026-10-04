#!/usr/bin/env python3
"""
Export a trained model for the Blunderprint web app -- provenance layer.

WHY THIS SCRIPT EXISTS AT ALL.

A model trained on the committed 500-game sample and the production model are
the same shape: both are `engine_free` with 56 features, both are LightGBM text
dumps, and nothing in `*_meta.json` records which data trained them. Shipping
the sample model would silently make every probability in the app wrong, and
nothing downstream would notice. So the export refuses to run on anything that
is not on an explicit approved list.

THE THREE GUARDS, which are independent on purpose.

  1. identity  every file of the artefact must hash to the SHA-256 recorded in
               the approved list. This is the real check.
  2. sanity    the artefact's own results["B0 constant"]["n"] must be at least
               the list's floor, and must equal the count the list records. The
               sample model reports 6,237; full-data artefacts report
               1,250,376 or more, a gap of ~200x with nothing in between.
  3. path      the source must live in <model repo>/models/ and its name must
               start with the list's prefix.

Renaming a file defeats the path guard only. Moving it into models/ defeats
nothing. Re-training in place defeats nothing, because the hash changes.

A fourth refusal is not a guard but a product rule: artefacts whose role is
`ablation` are not shipping candidates. `full_free_gamehash` is trained on the
naive game-hash split, whose test set shares 19.0% of its players with train
(FINDINGS.md), so its scores are inflated. `--allow-ablation` overrides it
deliberately.

THE APPROVED LIST lives in the app repo, not here, because approving a model
is an app-side decision: `docs/model-approvals.json` in arkid-lutaj/blunderprint,
with the same list shown to people in `docs/MODEL_CARD.md`. Pass it with
`--approved`. This script never parses hashes out of markdown.

WHAT IT WRITES once the guards pass (see "the export payload" below):

    <out>/model.json         the trees, flattened to arrays
    <out>/calibration.json   the isotonic knots
    <out>/feature_spec.json  the features in model order, the evaluation
                             semantics in writing, and the provenance stamp

and, with --fixtures, the golden files that pin the TypeScript port.

USAGE

    # guards only; prints the stamp, writes nothing
    python src/export_web.py --model models/full_free \\
      --approved ../docs/model-approvals.json --check-only

    # the payload, verified against booster.predict, plus golden fixtures
    python src/export_web.py --model models/full_free \\
      --approved ../docs/model-approvals.json --out ../app/public/model \\
      --sample-positions <tmp>/pos_blitz.parquet \\
      --sample-features <tmp>/feat_blitz.parquet \\
      --fixtures ../fixtures/golden

    python src/export_web.py --self-test     # synthetic files only, no data

Nothing here writes to build/, and nothing it reads is modified.
"""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORTER_VERSION = 1


class Refused(Exception):
    """The guards said no. Carries every reason, not just the first."""

    def __init__(self, reasons):
        self.reasons = list(reasons)
        super().__init__("; ".join(self.reasons))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def artefact_files(model_base: Path) -> dict:
    """The three files that make up one artefact, keyed by file name."""
    name = model_base.name
    return {name + suffix: model_base.with_name(name + suffix)
            for suffix in (".txt", "_iso.npz", "_meta.json")}


def load_approved(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        approved = json.load(fh)
    for key in ("source_commit", "source_repo", "min_test_rows",
                "name_prefix", "artefacts"):
        if key not in approved:
            raise Refused([f"approved list {path} has no '{key}' field"])
    return approved


def read_test_rows(meta_path: Path):
    """results["B0 constant"]["n"], or None if the file cannot supply it."""
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        return meta["results"]["B0 constant"]["n"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# the guards
# ---------------------------------------------------------------------------


def check_provenance(model_base: Path, approved: dict, *,
                     allow_ablation: bool = False,
                     models_dir: Path = None,
                     verbose: bool = True) -> dict:
    """Run all three guards and return the provenance stamp.

    Raises Refused, listing every reason, if any guard says no. All three run
    even after one fails, so a refusal shows the whole picture rather than the
    first thing noticed.

    `models_dir` exists for the self-test, which must not put synthetic files
    in the real models/ directory. The CLI never sets it.
    """
    models_dir = (models_dir or (REPO_ROOT / "models")).resolve()
    model_base = Path(model_base)
    name = model_base.name
    reasons = []

    def say(line):
        if verbose:
            print(line)

    # --- guard 3: path ------------------------------------------------------
    # First, because the other two read files and the point is to read them
    # only where they are supposed to live.
    parent = model_base.resolve().parent
    if parent != models_dir:
        reasons.append(f"path guard: source is in {parent}, not {models_dir}")
    if not name.startswith(approved["name_prefix"]):
        reasons.append(f"path guard: name {name!r} does not start with "
                       f"{approved['name_prefix']!r}")
    say(f"path     : {name} in {parent}")

    entry = approved["artefacts"].get(name)
    if entry is None:
        known = ", ".join(sorted(approved["artefacts"])) or "(none)"
        reasons.append(f"identity guard: {name!r} is not on the approved list. "
                       f"Approved: {known}")

    # --- guard 1: identity --------------------------------------------------
    files = artefact_files(model_base)
    hashes = {}
    for fname, fpath in files.items():
        if not fpath.is_file():
            reasons.append(f"identity guard: missing file {fname}")
            continue
        hashes[fname] = sha256_file(fpath)

    if entry is not None:
        recorded = entry.get("files", {})
        extra = sorted(set(hashes) - set(recorded))
        missing = sorted(set(recorded) - set(hashes))
        for fname in missing:
            reasons.append(f"identity guard: approved file {fname} was not read")
        for fname in extra:
            reasons.append(f"identity guard: {fname} is not in the approved entry")
        for fname, actual in sorted(hashes.items()):
            want = recorded.get(fname)
            if want is None:
                continue
            if actual != want:
                reasons.append(f"identity guard: {fname} hashes to {actual}, "
                               f"approved is {want}")
            say(f"identity : {fname} {actual[:16]}... "
                f"{'ok' if actual == want else 'MISMATCH'}")

    # --- guard 2: sanity ----------------------------------------------------
    floor = approved["min_test_rows"]
    test_rows = read_test_rows(files[name + "_meta.json"])
    if test_rows is None:
        reasons.append("sanity guard: could not read "
                       'results["B0 constant"]["n"] from the meta file')
    else:
        say(f"sanity   : {test_rows:,} test rows (floor {floor:,})")
        if test_rows < floor:
            reasons.append(f"sanity guard: {test_rows:,} test rows is below the "
                           f"floor of {floor:,}. A model trained on the 500-game "
                           f"sample reports 6,237.")
        if entry is not None and entry.get("test_rows") != test_rows:
            reasons.append(f"sanity guard: the artefact reports {test_rows:,} test "
                           f"rows, the approved list records "
                           f"{entry.get('test_rows')}")

    # --- role: not a guard, a product rule ----------------------------------
    if entry is not None:
        role = entry.get("role")
        say(f"role     : {role}")
        if role != "shipping_candidate" and not allow_ablation:
            reasons.append(f"{name} has role {role!r}, not 'shipping_candidate'. "
                           f"{entry.get('note', '')} Pass --allow-ablation to "
                           f"export it anyway.")

    if reasons:
        raise Refused(reasons)

    # No timestamp on purpose: two exports of the same artefact must produce
    # byte-identical files, so a stamp can be compared rather than trusted.
    return {
        "exporter": "model/src/export_web.py",
        "exporter_version": EXPORTER_VERSION,
        "artefact": name,
        "role": entry["role"],
        "feature_set": entry.get("feature_set"),
        "no_clock": entry.get("no_clock"),
        "n_features": entry.get("n_features"),
        "split_col": entry.get("split_col"),
        "test_rows": test_rows,
        "sha256": dict(sorted(hashes.items())),
        "source_repo": approved["source_repo"],
        "source_commit": approved["source_commit"],
    }


# ---------------------------------------------------------------------------
# the export payload (M0.2)
# ---------------------------------------------------------------------------
#
# THREE FILES land in the --out directory:
#
#   model.json         the trees, flattened to parallel arrays per tree
#   calibration.json   the isotonic knots plus the clip bounds
#   feature_spec.json  the features in model order, the evaluation semantics
#                      in writing, and the provenance stamp
#
# WHY THE SEMANTICS ARE WRITTEN DOWN RATHER THAN LEFT TO BE INFERRED. The
# TypeScript port has to reproduce LightGBM's prediction path bit for bit, and
# three steps of that path are invisible in the artefact:
#
#   1. the features reach the model as float32 (build_features.py casts every
#      chunk, train.py calls .to_numpy(dtype=np.float32)), and LightGBM then
#      widens each value to double before comparing it with the threshold. A
#      port that keeps JavaScript doubles end to end will disagree on rows
#      where the float32 rounding lands on the other side of a split.
#   2. a missing value does NOT always mean "go the default way". LightGBM
#      replaces NaN with 0.0 first unless the node's missing_type is NaN, so
#      at a missing_type=None node a NaN row takes the ordinary
#      0.0 <= threshold comparison. full_free has both kinds of node.
#   3. the isotonic calibrator was fitted on PROBABILITIES, not raw scores
#      (train.py: iso.fit(raw["val"], ...) where raw = booster.predict(...)
#      without raw_score=True). So the chain is sum -> sigmoid -> isotonic,
#      and swapping the last two steps changes every number in the app.
#
# Floats are written with repr(), the shortest representation that round-trips
# to the same double, so no threshold and no leaf value can drift. Nothing is
# rounded anywhere in this file.

ZERO_THRESHOLD = 1e-35          # LightGBM's kZeroThreshold
MISSING_NONE, MISSING_ZERO, MISSING_NAN = 0, 1, 2
_MISSING_CODES = {"None": MISSING_NONE, "Zero": MISSING_ZERO, "NaN": MISSING_NAN}

PAYLOAD_SCHEMA_VERSION = 1

# name -> (unit, when it can be missing, one-line definition), from
# src/build_features.py. The keys are checked against the model's own feature
# list at export time, so a renamed or added feature fails the export rather
# than shipping an undocumented column.
FEATURE_DOCS = {
    "n_legal": ("count", "never", "legal moves for the mover"),
    "n_captures": ("count", "never", "legal moves that capture"),
    "n_checks": ("count", "never", "legal moves that give check"),
    "n_quiet": ("count", "never", "n_legal minus captures, checks and promotions"),
    "n_promotions": ("count", "never", "legal moves that promote"),
    "in_check": ("0/1", "never", "the mover is in check"),
    "hanging_own": ("count", "never", "mover's pieces attacked and not defended"),
    "hanging_opp": ("count", "never", "opponent's pieces attacked and not defended"),
    "hanging_val_own": ("centipawns", "never", "material value of hanging_own"),
    "hanging_val_opp": ("centipawns", "never", "material value of hanging_opp"),
    "tension": ("count", "never", "squares both sides attack"),
    "n_p_own": ("count", "never", "mover's pawns"),
    "n_p_opp": ("count", "never", "opponent's pawns"),
    "n_n_own": ("count", "never", "mover's knights"),
    "n_n_opp": ("count", "never", "opponent's knights"),
    "n_b_own": ("count", "never", "mover's bishops"),
    "n_b_opp": ("count", "never", "opponent's bishops"),
    "n_r_own": ("count", "never", "mover's rooks"),
    "n_r_opp": ("count", "never", "opponent's rooks"),
    "n_q_own": ("count", "never", "mover's queens"),
    "n_q_opp": ("count", "never", "opponent's queens"),
    "material_own": ("centipawns", "never", "mover's material"),
    "material_opp": ("centipawns", "never", "opponent's material"),
    "material_balance": ("centipawns", "never", "material_own minus material_opp"),
    "non_pawn_material": ("centipawns", "never",
                          "both sides' material excluding pawns"),
    "total_pieces": ("count", "never", "pieces on the board, kings included"),
    "king_ring_atk_own": ("count", "never",
                          "opponent attacks on the mover's king ring"),
    "king_ring_def_own": ("count", "never",
                          "mover's defenders of its own king ring"),
    "king_escapes_own": ("count", "never", "safe king moves for the mover"),
    "king_ring_atk_opp": ("count", "never",
                          "mover's attacks on the opponent's king ring"),
    "king_ring_def_opp": ("count", "never",
                          "opponent's defenders of its own king ring"),
    "king_escapes_opp": ("count", "never", "safe king moves for the opponent"),
    "passed_pawns_own": ("count", "never", "mover's passed pawns"),
    "isolated_pawns_own": ("count", "never", "mover's isolated pawns"),
    "doubled_pawns_own": ("count", "never", "mover's doubled pawns"),
    "passed_pawns_opp": ("count", "never", "opponent's passed pawns"),
    "isolated_pawns_opp": ("count", "never", "opponent's isolated pawns"),
    "doubled_pawns_opp": ("count", "never", "opponent's doubled pawns"),
    "can_castle_own": ("0/1", "never", "the mover has any castling right"),
    "can_castle_opp": ("0/1", "never", "the opponent has any castling right"),
    "ep_available": ("0/1", "never", "an en-passant target square is set"),
    "halfmove_clock": ("plies", "never", "FEN halfmove clock"),
    "mover_elo": ("rating", "never", "the mover's blitz rating for this game"),
    "opp_elo": ("rating", "never", "the opponent's blitz rating"),
    "elo_gap": ("rating", "never", "mover_elo minus opp_elo"),
    "mean_elo": ("rating", "never", "(mover_elo + opp_elo) / 2"),
    "mover_is_white": ("0/1", "never", "the mover plays White"),
    "move_number": ("count", "never", "ply // 2 + 1"),
    "tc_base": ("seconds", "never", "time control base"),
    "tc_inc": ("seconds", "never", "time control increment"),
    "clk_before": ("seconds", "never", "the mover's clock before the move"),
    "clock_frac": ("fraction", "tc_base == 0", "clk_before / tc_base"),
    "log_clk_before": ("log seconds", "never", "log1p(max(clk_before, 0))"),
    "own_time_prev1": ("seconds", "the mover's 1st move of the game",
                       "seconds the mover spent 2 plies ago"),
    "own_time_prev2": ("seconds", "the mover's first 2 moves",
                       "seconds the mover spent 4 plies ago"),
    "own_time_prev3": ("seconds", "the mover's first 3 moves",
                       "seconds the mover spent 6 plies ago"),
}

# The four rating features always move together (SPEC.md). Recorded in the
# spec so the app's rating-sweep code cannot forget it.
RATING_FEATURES = ["mover_elo", "opp_elo", "elo_gap", "mean_elo"]


def write_json(path: Path, obj) -> str:
    """Compact JSON, LF endings, shortest round-tripping floats. Returns sha256."""
    text = json.dumps(obj, separators=(",", ":"), allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.write("\n")
    return sha256_file(path)


def _flatten_tree(node, out, tree_index):
    """LightGBM's nested dump -> parallel arrays.

    Children follow LightGBM's own convention: a non-negative child is an
    internal node index, a negative child `v` is leaf number `-v - 1`.
    """
    if "split_index" not in node:
        if "leaf_value" not in node:
            raise Refused([f"tree {tree_index}: a node is neither a split nor a leaf"])
        idx = node.get("leaf_index", 0)
        while len(out["leaf_value"]) <= idx:
            out["leaf_value"].append(None)
        out["leaf_value"][idx] = float(node["leaf_value"])
        return -idx - 1

    i = node["split_index"]
    for key in ("feature", "threshold", "default_left", "missing_type",
                "left", "right"):
        while len(out[key]) <= i:
            out[key].append(None)

    if node["decision_type"] != "<=":
        raise Refused([f"tree {tree_index}: unsupported decision_type "
                       f"{node['decision_type']!r} at split {i}. The exporter and "
                       f"the TS port implement numerical '<=' splits only."])
    mt = node.get("missing_type", "None")
    if mt not in _MISSING_CODES:
        raise Refused([f"tree {tree_index}: unknown missing_type {mt!r} at split {i}"])

    out["feature"][i] = int(node["split_feature"])
    out["threshold"][i] = float(node["threshold"])
    out["default_left"][i] = 1 if node["default_left"] else 0
    out["missing_type"][i] = _MISSING_CODES[mt]
    out["left"][i] = _flatten_tree(node["left_child"], out, tree_index)
    out["right"][i] = _flatten_tree(node["right_child"], out, tree_index)
    return i


def build_model_payload(dump: dict) -> dict:
    """booster.dump_model() -> the flat form the browser reads."""
    objective = dump.get("objective", "")
    if objective.split()[:1] != ["binary"]:
        raise Refused([f"objective {objective!r} is not binary; the app's sigmoid "
                       f"step assumes it is"])
    sigmoid = 1.0
    for part in objective.split()[1:]:
        if part.startswith("sigmoid:"):
            sigmoid = float(part.split(":", 1)[1])
    if dump.get("num_class", 1) != 1:
        raise Refused([f"num_class is {dump.get('num_class')}, not 1"])
    if dump.get("num_tree_per_iteration", 1) != 1:
        raise Refused(["num_tree_per_iteration != 1"])
    if dump.get("pandas_categorical"):
        raise Refused(["the model carries pandas_categorical, so some input was a "
                       "pandas category; the TS port has no categorical path"])

    trees = []
    for t in dump["tree_info"]:
        if t.get("num_cat", 0):
            raise Refused([f"tree {t['tree_index']} has {t['num_cat']} categorical "
                           f"splits; the TS port implements numerical splits only"])
        flat = {k: [] for k in ("feature", "threshold", "default_left",
                                "missing_type", "left", "right", "leaf_value")}
        root = _flatten_tree(t["tree_structure"], flat, t["tree_index"])
        if root != 0:
            raise Refused([f"tree {t['tree_index']}: root flattened to node {root}, "
                           f"not 0"])
        if any(v is None for arr in flat.values() for v in arr):
            raise Refused([f"tree {t['tree_index']}: flattened with a gap, so the "
                           f"dump's split or leaf indices are not contiguous"])
        trees.append(flat)

    return {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "format": "lightgbm-flat-arrays",
        "objective": "binary",
        "sigmoid": sigmoid,
        "n_features": int(dump["max_feature_idx"]) + 1,
        "n_trees": len(trees),
        "feature_names": list(dump["feature_names"]),
        "child_encoding": "child >= 0 is an internal node index; "
                          "child < 0 is leaf number -child - 1",
        "trees": trees,
    }


def build_calibration_payload(iso_npz: Path):
    """The isotonic knots, plus what is needed to reproduce sklearn exactly."""
    import numpy as np

    with np.load(iso_npz) as z:
        missing = [k for k in ("x", "y") if k not in z]
        if missing:
            raise Refused([f"{iso_npz.name}: no {missing} array"])
        x = np.asarray(z["x"], dtype=np.float64)
        y = np.asarray(z["y"], dtype=np.float64)
    if x.ndim != 1 or x.shape != y.shape or x.size < 2:
        raise Refused([f"{iso_npz.name}: expected two 1-D arrays of equal length "
                       f">= 2, got {x.shape} and {y.shape}"])
    if not bool(np.all(np.diff(x) > 0)):
        raise Refused([f"{iso_npz.name}: the x knots are not strictly increasing, "
                       f"so the interpolation would divide by zero"])
    if not bool(np.all(np.diff(y) >= 0)):
        raise Refused([f"{iso_npz.name}: the y knots are not monotone"])
    if float(y.min()) < 0.0 or float(y.max()) > 1.0:
        raise Refused([f"{iso_npz.name}: the y knots leave [0, 1]"])

    payload = {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "kind": "isotonic",
        # The one fact that is not in the npz and cannot be guessed from it.
        "fit_on": "probability",
        "input": "sigmoid(sum of leaf values), not the raw score",
        "fitted_on_split": "val",
        "out_of_bounds": "clip",
        "clip_lo": float(x[0]),
        "clip_hi": float(x[-1]),
        "interpolation": "scipy interp1d kind='linear' after sklearn's clip: "
                         "i = clip(searchsorted(x, v, 'left'), 1, n - 1); "
                         "p = (v - x[i-1]) / (x[i] - x[i-1]) * y[i] + "
                         "(x[i] - v) / (x[i] - x[i-1]) * y[i-1]",
        "n_knots": int(x.size),
        "x": [float(v) for v in x],
        "y": [float(v) for v in y],
    }
    return payload, x, y


def build_feature_spec(stamp: dict, model_payload: dict, calib: dict,
                       meta: dict, file_hashes: dict) -> dict:
    names = model_payload["feature_names"]
    if names != list(meta.get("features", [])):
        raise Refused(["the booster's feature_names disagree with the order in "
                       "*_meta.json, so one of the two did not train this model"])
    if len(names) != model_payload["n_features"]:
        raise Refused([f"{len(names)} feature names for "
                       f"{model_payload['n_features']} feature slots"])
    undocumented = [n for n in names if n not in FEATURE_DOCS]
    if undocumented:
        raise Refused([f"no FEATURE_DOCS entry for {undocumented}. Document the "
                       f"feature rather than shipping an unexplained column."])
    absent = [n for n in RATING_FEATURES if n not in names]
    if absent:
        raise Refused([f"rating features {absent} are absent, so the "
                       f"move-together rule cannot be stated"])

    features = []
    for i, n in enumerate(names):
        unit, missing_when, definition = FEATURE_DOCS[n]
        features.append({"index": i, "name": n, "unit": unit,
                         "missing_when": missing_when, "definition": definition})

    return {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "model_version": "{}@{}".format(
            stamp["artefact"], stamp["sha256"][stamp["artefact"] + ".txt"][:12]),
        "time_control": "blitz",
        "time_control_note": "Blitz only. Never score another time control with "
                             "this model: blitz and rapid are separate rating "
                             "pools (SPEC.md).",
        "n_features": len(features),
        "features": features,
        "rating_features_move_together": RATING_FEATURES,
        "input_dtype": "float32",
        "pipeline": [
            "1. build the features in the order above",
            "2. round each value to float32 (Math.fround), then widen to float64",
            "3. walk every tree and sum the leaf values into one float64, in tree order",
            "4. p_raw = 1 / (1 + exp(-sigmoid * sum)), sigmoid from model.json",
            "5. p = isotonic(p_raw) using calibration.json",
        ],
        "split_semantics": {
            "order": [
                "if the value is NaN and missing_type != 2, replace it with 0.0",
                "if missing_type == 1 and -1e-35 < value <= 1e-35, take default_left",
                "if missing_type == 2 and the value is NaN, take default_left",
                "otherwise go left when value <= threshold, right otherwise",
            ],
            "zero_threshold": ZERO_THRESHOLD,
            "missing_type_codes": {"0": "None", "1": "Zero", "2": "NaN"},
            "missing_types_present": sorted({int(mt) for t in model_payload["trees"]
                                             for mt in t["missing_type"]}),
            "note": "LightGBM tree.h NumericalDecision. A port that treats every "
                    "missing value as 'take the default branch' is wrong at the "
                    "missing_type == 0 nodes, which this model has.",
        },
        "base_rate_train": meta.get("base_rate_train"),
        "label": "blunder as defined in the model repo's SPEC.md: a win% drop "
                 "greater than 20 points from the mover's point of view, with cp "
                 "clamped to +/-1000, using Lichess's win% formula "
                 "(lichess.org/page/accuracy, Lichess's formula, not ours). Mate "
                 "transitions are label-invalid rows.",
        "calibration": {k: v for k, v in calib.items() if k not in ("x", "y")},
        "files": file_hashes,
        "provenance": stamp,
    }


# ---------------------------------------------------------------------------
# the reference evaluator: reads ONLY the exported payload
# ---------------------------------------------------------------------------
#
# Two implementations, on purpose.
#
#   eval_rows_scalar   plain Python, one row at a time, written the way the
#                      TypeScript port will be written. This is the normative
#                      reference: if it and LightGBM agree, the exported format
#                      is lossless and the semantics above are right.
#   eval_rows_vector   the same rules in numpy, so a 50,000-row check takes a
#                      minute instead of an afternoon.
#
# --verify asserts the two agree bit for bit on a slice before trusting the
# fast one, so the fast one cannot drift into its own private semantics.


def _descend(tree, row):
    feature = tree["feature"]
    threshold = tree["threshold"]
    default_left = tree["default_left"]
    missing_type = tree["missing_type"]
    left = tree["left"]
    right = tree["right"]
    node = 0
    while True:
        v = row[feature[node]]
        mt = missing_type[node]
        if v != v and mt != MISSING_NAN:        # NaN, missing_type None or Zero
            v = 0.0
        if ((mt == MISSING_ZERO and -ZERO_THRESHOLD < v <= ZERO_THRESHOLD)
                or (mt == MISSING_NAN and v != v)):
            node = left[node] if default_left[node] else right[node]
        elif v <= threshold[node]:
            node = left[node]
        else:
            node = right[node]
        if node < 0:
            return tree["leaf_value"][-node - 1]


def eval_rows_scalar(model, rows):
    """Raw score per row, summed in tree order. Rows must already be float32."""
    trees = model["trees"]
    out = []
    for row in rows:
        total = 0.0
        for tree in trees:
            total += _descend(tree, row)
        out.append(total)
    return out


def eval_rows_vector(model, X):
    """The same rules, vectorised over rows. X must already be float32."""
    import numpy as np

    n = int(X.shape[0])
    total = np.zeros(n, dtype=np.float64)
    Xd = X.astype(np.float64)       # float32 widened, exactly as LightGBM does
    for tree in model["trees"]:
        feature = np.asarray(tree["feature"], dtype=np.int64)
        threshold = np.asarray(tree["threshold"], dtype=np.float64)
        default_left = np.asarray(tree["default_left"], dtype=bool)
        missing_type = np.asarray(tree["missing_type"], dtype=np.int64)
        left = np.asarray(tree["left"], dtype=np.int64)
        right = np.asarray(tree["right"], dtype=np.int64)
        leaf_value = np.asarray(tree["leaf_value"], dtype=np.float64)

        node = np.zeros(n, dtype=np.int64)
        active = np.arange(n, dtype=np.int64)
        while active.size:
            nd = node[active]
            v = Xd[active, feature[nd]]
            mt = missing_type[nd]
            v = np.where(np.isnan(v) & (mt != MISSING_NAN), 0.0, v)
            take_default = (((mt == MISSING_ZERO)
                             & (v > -ZERO_THRESHOLD) & (v <= ZERO_THRESHOLD))
                            | ((mt == MISSING_NAN) & np.isnan(v)))
            go_left = np.where(take_default, default_left[nd], v <= threshold[nd])
            child = np.where(go_left, left[nd], right[nd])
            node[active] = child
            done = child < 0
            if bool(done.any()):
                finished = active[done]
                total[finished] += leaf_value[-node[finished] - 1]
                active = active[~done]
    return total


def sigmoid_of(model, raw):
    import numpy as np
    return 1.0 / (1.0 + np.exp(-model["sigmoid"]
                               * np.asarray(raw, dtype=np.float64)))


def apply_isotonic(calib, p):
    """scipy's interp1d kind='linear' after sklearn's clip, reproduced exactly."""
    import numpy as np

    x = np.asarray(calib["x"], dtype=np.float64)
    y = np.asarray(calib["y"], dtype=np.float64)
    v = np.clip(np.asarray(p, dtype=np.float64),
                calib["clip_lo"], calib["clip_hi"])
    i = np.searchsorted(x, v, side="left").clip(1, x.size - 1)
    x_lo, x_hi = x[i - 1], x[i]
    y_lo, y_hi = y[i - 1], y[i]
    return (((v - x_lo) / (x_hi - x_lo)) * y_hi
            + ((x_hi - v) / (x_hi - x_lo)) * y_lo)


# ---------------------------------------------------------------------------
# the export driver
# ---------------------------------------------------------------------------


def load_meta(model_base: Path) -> dict:
    with open(model_base.with_name(model_base.name + "_meta.json"),
              "r", encoding="utf-8") as fh:
        return json.load(fh)


def export_payload(model_base: Path, approved: dict, out_dir: Path, *,
                   allow_ablation: bool = False, verbose: bool = True) -> dict:
    """Guards, then the three files. Returns the feature spec that was written."""
    import lightgbm as lgb

    stamp = check_provenance(model_base, approved, allow_ablation=allow_ablation,
                             verbose=verbose)
    booster = lgb.Booster(model_file=str(model_base.with_name(
        model_base.name + ".txt")))
    dump = booster.dump_model()
    model_payload = build_model_payload(dump)

    n_nodes = sum(len(t["feature"]) for t in model_payload["trees"])
    n_leaves = sum(len(t["leaf_value"]) for t in model_payload["trees"])
    if verbose:
        print(f"\ntrees    : {model_payload['n_trees']:,} "
              f"({n_nodes:,} split nodes, {n_leaves:,} leaves)")
        print(f"features : {model_payload['n_features']}")
        print(f"sigmoid  : {model_payload['sigmoid']}")

    calib_payload, _, _ = build_calibration_payload(
        model_base.with_name(model_base.name + "_iso.npz"))
    if verbose:
        print(f"calibrat.: {calib_payload['n_knots']:,} isotonic knots, "
              f"fitted on {calib_payload['fit_on']}, clipped to "
              f"[{calib_payload['clip_lo']:.6g}, {calib_payload['clip_hi']:.6g}]")

    out_dir = Path(out_dir)
    hashes = {
        "model.json": write_json(out_dir / "model.json", model_payload),
        "calibration.json": write_json(out_dir / "calibration.json", calib_payload),
    }
    spec = build_feature_spec(stamp, model_payload, calib_payload,
                              load_meta(model_base), hashes)
    write_json(out_dir / "feature_spec.json", spec)

    if verbose:
        print(f"\nwrote into {out_dir}:")
        for name in ("feature_spec.json", "model.json", "calibration.json"):
            size = (out_dir / name).stat().st_size
            print(f"  {name:<20} {size / 1e6:8.2f} MB")
        print(f"model_version: {spec['model_version']}")
    return spec


def read_payload(out_dir: Path):
    """Read back what was written. Verification never uses the in-memory objects."""
    out_dir = Path(out_dir)
    def load(name):
        with open(out_dir / name, "r", encoding="utf-8") as fh:
            return json.load(fh)
    spec = load("feature_spec.json")
    for name, want in spec["files"].items():
        got = sha256_file(out_dir / name)
        if got != want:
            raise Refused([f"{name} hashes to {got}, feature_spec.json records "
                           f"{want}"])
    return spec, load("model.json"), load("calibration.json")


# ---------------------------------------------------------------------------
# verification against LightGBM itself
# ---------------------------------------------------------------------------


def feature_matrix(df, spec):
    """The spec's features, in the spec's order, as float32 -- as train.py does."""
    import numpy as np

    names = [f["name"] for f in spec["features"]]
    absent = [n for n in names if n not in df.columns]
    if absent:
        raise Refused([f"the feature table has no column(s) {absent}"])
    return np.ascontiguousarray(df[names].to_numpy(dtype=np.float32)), names


def verify_against_lightgbm(model_base: Path, out_dir: Path, df, *,
                            scalar_rows: int = 500, label: str = "rows",
                            verbose: bool = True):
    """Prove the exported files reproduce booster.predict, and report the gap.

    Checks, in order:
      1. the two reference evaluators agree bit for bit on a slice
      2. raw scores match booster.predict(raw_score=True)
      3. probabilities match booster.predict()
      4. apply_isotonic matches a real sklearn IsotonicRegression rebuilt from
         the exported knots, so the interpolation formula is not taken on trust
    """
    import numpy as np
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression

    spec, model, calib = read_payload(out_dir)
    X, _ = feature_matrix(df, spec)
    booster = lgb.Booster(model_file=str(model_base.with_name(
        model_base.name + ".txt")))

    raw_ref = eval_rows_vector(model, X)
    raw_lgb = booster.predict(X, raw_score=True)
    p_ref = sigmoid_of(model, raw_ref)
    p_lgb = booster.predict(X)

    k = min(scalar_rows, X.shape[0])
    raw_scalar = eval_rows_scalar(model, [[float(v) for v in row] for row in X[:k]])
    scalar_exact = all(a == b for a, b in zip(raw_scalar, list(raw_ref[:k])))

    # An independent isotonic: sklearn's own code path, rebuilt from the knots.
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.X_thresholds_ = np.asarray(calib["x"], dtype=np.float64)
    iso.y_thresholds_ = np.asarray(calib["y"], dtype=np.float64)
    iso.X_min_, iso.X_max_ = float(calib["x"][0]), float(calib["x"][-1])
    iso.increasing_ = True
    iso._build_f(iso.X_thresholds_, iso.y_thresholds_)
    cal_sklearn = iso.predict(p_lgb)
    cal_ref = apply_isotonic(calib, p_ref)

    res = {
        "label": label,
        "n_rows": int(X.shape[0]),
        "scalar_rows": k,
        "scalar_matches_vector_exactly": bool(scalar_exact),
        "max_abs_raw": float(np.max(np.abs(raw_ref - raw_lgb))),
        "max_abs_prob": float(np.max(np.abs(p_ref - p_lgb))),
        "max_abs_calibrated": float(np.max(np.abs(cal_ref - cal_sklearn))),
        "raw_bit_identical": bool(np.array_equal(raw_ref, raw_lgb)),
        "n_rows_with_nan_feature": int(np.isnan(X).any(axis=1).sum()),
        "n_nan_cells": int(np.isnan(X).sum()),
        "p_min": float(np.min(cal_ref)),
        "p_max": float(np.max(cal_ref)),
    }
    if verbose:
        print(f"\nverification on {res['n_rows']:,} {label}")
        print(f"  rows with a missing feature : {res['n_rows_with_nan_feature']:,}"
              f"  ({res['n_nan_cells']:,} missing cells)")
        print(f"  scalar == vector on {k:,} rows : "
              f"{'bit identical' if scalar_exact else 'DIFFERS'}")
        print(f"  max |raw - booster raw|     : {res['max_abs_raw']:.3e}"
              f"  ({'bit identical' if res['raw_bit_identical'] else 'not bit identical'})")
        print(f"  max |p - booster p|         : {res['max_abs_prob']:.3e}")
        print(f"  max |isotonic - sklearn|    : {res['max_abs_calibrated']:.3e}")
        print(f"  calibrated p range          : "
              f"{res['p_min']:.6f} to {res['p_max']:.6f}")
    return res, X, raw_ref, p_ref, cal_ref


# ---------------------------------------------------------------------------
# golden fixtures from the committed 500-game sample
# ---------------------------------------------------------------------------
#
# Two files, because they pin two different things.
#
#   sample_plies.json  every ply of a deterministic slice of the sample's blitz
#                      games, with the raw per-ply inputs the TS port starts
#                      from AND the features, label and probabilities it must
#                      arrive at. This pins the feature port and the label.
#   model_cases.json   features straight in, probabilities out, for rows chosen
#                      to stress the tree walk: every row with a missing
#                      feature, the extreme row for each feature, then a seeded
#                      random fill. This pins the model runtime alone, so a
#                      failure here cannot be blamed on the feature port.


FIXTURE_INPUT_COLS = [
    "fen", "move", "mover_is_white", "mover_elo", "opp_elo",
    "tc_base", "tc_inc", "clk_before", "clk_after", "time_spent",
    "cp_before", "cp_after", "mate_before", "mate_after",
    "winpct_before", "winpct_after", "win_drop",
]


def _jsonable(v):
    import numpy as np
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        return v
    f = float(v)
    if f != f:
        return None            # a missing value is null in JSON, never 0
    return int(f) if isinstance(v, (int, np.integer)) else f


def build_fixtures(spec, X, names, pos, feat, raw, p_sigmoid, p_cal, *,
                   n_games: int, n_cases: int, seed: int):
    """Returns (sample_plies, model_cases) as JSON-ready dicts."""
    import numpy as np

    # The inputs always come from the positions table, never from the feature
    # table, even where both have the column: the fixture's job is to hand the
    # TS port the same raw per-ply data the Python pipeline started from. The
    # feature table's own copies are already in `features`.
    missing_cols = [c for c in FIXTURE_INPUT_COLS if c not in pos.columns]
    if missing_cols:
        raise Refused([f"the positions table has no column(s) {missing_cols}"])
    base = feat.drop(columns=[c for c in FIXTURE_INPUT_COLS
                              if c in feat.columns]).reset_index(drop=True)
    merged = base.merge(pos[["game_id", "ply"] + FIXTURE_INPUT_COLS],
                        on=["game_id", "ply"], how="left", validate="one_to_one")
    if len(merged) != len(feat):
        raise Refused(["the positions/features merge changed the row count"])
    # X was built row-for-row from `feat`, and a left merge keeps that order.
    if not merged.game_id.equals(feat.game_id.reset_index(drop=True)) \
            or not merged.ply.equals(feat.ply.reset_index(drop=True)):
        raise Refused(["the merge reordered the rows, so the features and the "
                       "inputs would be mismatched"])

    games = sorted(merged.game_id.unique())[:n_games]
    keep = merged.game_id.isin(games).to_numpy()
    rows = []
    for i in np.flatnonzero(keep):
        r = merged.iloc[i]
        rec = {"game_id": r.game_id, "ply": int(r.ply)}
        for c in FIXTURE_INPUT_COLS:
            rec[c] = _jsonable(r[c])
        rec["label_valid"] = bool(r.label_valid)
        rec["blunder"] = None if r.blunder is None or r.blunder != r.blunder \
            else bool(r.blunder)
        rec["features"] = [_jsonable(v) for v in X[i]]
        rec["raw"] = float(raw[i])
        rec["p_sigmoid"] = float(p_sigmoid[i])
        rec["p"] = float(p_cal[i])
        rows.append(rec)

    sample_plies = {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "what": "every ply of the first {} blitz games of data/sample_games.pgn, "
                "by sorted game_id".format(len(games)),
        "how": "src/export_web.py --fixtures. The sample was parsed and featurised "
               "only; no model was ever trained on it.",
        "model_version": spec["model_version"],
        "feature_order": names,
        "fields": {
            "features": "the model's input row, in feature_order, float32 values; "
                        "null means missing (NaN)",
            "raw": "sum of leaf values",
            "p_sigmoid": "1 / (1 + exp(-sigmoid * raw))",
            "p": "the calibrated probability the app shows",
            "blunder": "null on label-invalid rows (mate transitions)",
        },
        "n_games": len(games),
        "n_rows": len(rows),
        "rows": rows,
    }

    # --- model_cases: rows chosen to stress the tree walk --------------------
    # The three groups get fixed shares of the budget. Without a cap the first
    # group swallows all of it -- the sample has 1,926 rows with a missing
    # feature against a budget of 1,200 -- and the fixture then tests one branch
    # of NumericalDecision and nothing else.
    rng = np.random.default_rng(seed)

    def pick(pool, k, seen):
        pool = [int(i) for i in pool if int(i) not in seen]
        if len(pool) > k:
            pool = [int(i) for i in rng.choice(pool, size=k, replace=False)]
        seen.update(pool)
        return pool

    nan_quota = max(1, n_cases // 3)
    seen = set()
    nan_rows = pick(np.flatnonzero(np.isnan(X).any(axis=1)), nan_quota, seen)

    extreme = set()
    for j in range(X.shape[1]):
        col = X[:, j]
        finite = np.flatnonzero(~np.isnan(col))
        if finite.size:
            extreme.add(int(finite[np.argmin(col[finite])]))
            extreme.add(int(finite[np.argmax(col[finite])]))
    extremes = pick(sorted(extreme), n_cases // 3, seen)

    fill = pick(np.arange(X.shape[0]),
                max(0, n_cases - len(nan_rows) - len(extremes)), seen)
    chosen = sorted(nan_rows + extremes + fill)
    group_sizes = {"with a missing feature": len(nan_rows),
                   "a feature's extreme value": len(extremes),
                   "seeded random fill": len(fill)}

    model_cases = {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "what": "feature rows in, probabilities out. Pins the tree evaluator and "
                "the calibrator on their own, with no feature code involved.",
        "how": "three groups, each capped at a third of the budget so none "
               "crowds out the others: rows with a missing feature, rows holding "
               "a feature's minimum or maximum, and a random fill. Every draw "
               "uses numpy default_rng(seed={}).".format(seed),
        "model_version": spec["model_version"],
        "feature_order": names,
        "n_rows": len(chosen),
        "groups": group_sizes,
        "n_rows_with_missing": int(sum(1 for i in chosen if np.isnan(X[i]).any())),
        "rows": [{"features": [_jsonable(v) for v in X[i]],
                  "raw": float(raw[i]),
                  "p_sigmoid": float(p_sigmoid[i]),
                  "p": float(p_cal[i])} for i in chosen],
    }
    return sample_plies, model_cases


# ---------------------------------------------------------------------------
# self-test: synthetic artefacts in a temp directory, no real model needed
# ---------------------------------------------------------------------------


def _write_fake_artefact(models_dir: Path, name: str, test_rows: int,
                         body: str = "tree=0\n") -> Path:
    base = models_dir / name
    base.with_name(name + ".txt").write_text(body, encoding="utf-8")
    base.with_name(name + "_iso.npz").write_bytes(b"not really an npz, only hashed")
    meta = {"feature_set": "engine_free", "no_clock": False,
            "features": ["f%d" % i for i in range(56)], "split_col": "split",
            "results": {"B0 constant": {"n": test_rows}}}
    base.with_name(name + "_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return base


def _fake_approved(models_dir: Path, bases, roles=None) -> dict:
    roles = roles or {}
    artefacts = {}
    for base in bases:
        name = base.name
        artefacts[name] = {
            "role": roles.get(name, "shipping_candidate"),
            "note": "synthetic",
            "feature_set": "engine_free",
            "no_clock": False,
            "n_features": 56,
            "split_col": "split",
            "test_rows": read_test_rows(base.with_name(name + "_meta.json")),
            "files": {fname: sha256_file(fpath)
                      for fname, fpath in artefact_files(base).items()},
        }
    return {
        "schema_version": 1,
        "source_repo": "https://example.invalid/fake",
        "source_commit": "0" * 40,
        "source_dir": "model/models",
        "name_prefix": "full_",
        "min_test_rows": 500_000,
        "artefacts": artefacts,
    }


def self_test() -> int:
    """Plant one fault at a time and check the guards refuse for that reason.

    Everything here is synthetic and lives in a temp directory: no real
    artefact is read, so this runs in CI where models/ does not exist.
    """
    print("export guards: planted faults, synthetic artefacts only\n")
    results = []

    def case(name, fn, expect_ok, expect_substr=None):
        try:
            stamp = fn()
            ok = expect_ok
            detail = "exported" if ok else "ACCEPTED, should have refused"
        except Refused as exc:
            stamp = None
            joined = "; ".join(exc.reasons)
            if expect_ok:
                ok, detail = False, f"REFUSED, should have passed: {joined}"
            elif expect_substr and expect_substr not in joined:
                ok, detail = False, f"refused for the wrong reason: {joined}"
            else:
                ok, detail = True, f"refused: {exc.reasons[0][:72]}"
        results.append(ok)
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}\n          {detail}")
        return stamp

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        models = tmp / "models"
        elsewhere = tmp / "elsewhere"
        models.mkdir()
        elsewhere.mkdir()

        good = _write_fake_artefact(models, "full_good", 1_250_376)
        tiny = _write_fake_artefact(models, "full_sample_shaped", 6_237)
        abl = _write_fake_artefact(models, "full_ablation", 1_250_376)
        unlisted = _write_fake_artefact(models, "full_unlisted", 1_250_376)
        approved = _fake_approved(models, [good, tiny, abl],
                                  roles={"full_ablation": "ablation"})

        def run(base, **kw):
            kw.setdefault("models_dir", models)
            kw.setdefault("verbose", False)
            return check_provenance(base, approved, **kw)

        stamp = case("an approved artefact passes", lambda: run(good), True)
        if stamp is not None:
            fields_ok = (stamp["artefact"] == "full_good"
                         and stamp["test_rows"] == 1_250_376
                         and stamp["source_commit"] == approved["source_commit"]
                         and len(stamp["sha256"]) == 3
                         and "exported_at" not in stamp)
            results.append(fields_ok)
            print(f"  {'ok  ' if fields_ok else 'FAIL'}  the stamp carries name, "
                  f"hashes, rows and commit, and no timestamp")

        # guard 1, identity
        case("an artefact that is not on the list is refused",
             lambda: run(unlisted), False, "not on the approved list")

        good.with_name("full_good.txt").write_text("tree=0\nretrained\n", encoding="utf-8")
        case("a changed model file is refused",
             lambda: run(good), False, "hashes to")
        good.with_name("full_good.txt").write_text("tree=0\n", encoding="utf-8")
        case("restoring the file passes again", lambda: run(good), True)

        iso = good.with_name("full_good_iso.npz")
        keep = iso.read_bytes()
        iso.unlink()
        case("a missing calibration file is refused",
             lambda: run(good), False, "missing file")
        iso.write_bytes(keep)

        # guard 2, sanity
        case("a sample-sized artefact is refused",
             lambda: run(tiny), False, "below the floor")

        meta = good.with_name("full_good_meta.json")
        keep_meta = meta.read_text(encoding="utf-8")
        bumped = json.loads(keep_meta)
        bumped["results"]["B0 constant"]["n"] = 999_999_999
        meta.write_text(json.dumps(bumped), encoding="utf-8")
        case("a row count that disagrees with the list is refused",
             lambda: run(good), False, "sanity guard")
        meta.write_text(keep_meta, encoding="utf-8")

        # guard 3, path
        moved = elsewhere / "full_good"
        for fname, fpath in artefact_files(good).items():
            shutil.copy2(fpath, elsewhere / fname)
        case("the same approved files outside models/ are refused",
             lambda: run(moved), False, "path guard")

        renamed = models / "sneaky_good"
        for fname, fpath in artefact_files(good).items():
            shutil.copy2(fpath, models / fname.replace("full_good", "sneaky_good"))
        case("a name without the approved prefix is refused",
             lambda: run(renamed), False, "path guard")

        # role
        case("an ablation is refused by default",
             lambda: run(abl), False, "not 'shipping_candidate'")
        case("an ablation passes with --allow-ablation",
             lambda: run(abl, allow_ablation=True), True)

        # the guards are independent: each of the other two still bites when
        # the path guard is satisfied, and the path guard bites on its own.
        case("a sample-sized artefact in the right place is still refused",
             lambda: run(tiny), False, "below the floor")

    def report(name, ok):
        results.append(bool(ok))
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}")

    print()
    print("evaluator semantics: hand-built trees, answers worked out by hand")
    print()
    self_test_evaluator(report)

    print()
    build = REPO_ROOT / "build"
    made_build = build.exists()
    results.append(not made_build)
    print(f"  {'ok  ' if not made_build else 'FAIL'}  the self-test did not "
          f"create {build}")

    failed = results.count(False)
    print()
    if failed:
        print(f"FAIL: {failed} of {len(results)} checks")
        return 1
    print(f"PASS: {len(results)} checks, every planted fault refused and every "
          f"evaluator case matched")
    return 0


# ---------------------------------------------------------------------------
# the held-out sample, read-only
# ---------------------------------------------------------------------------


def load_heldout_rows(features_path: Path, splits_path: Path, names, *,
                      split: str = "test", n_rows: int = 50_000,
                      seed: int = 20260923, verbose: bool = True):
    """A seeded sample of one split's rows, read in batches, nothing written.

    The source lives outside both repos and is read-only: this opens it, keeps
    the feature columns of the chosen split, samples, and closes. Only the
    aggregate verification result ever leaves the folder -- no rows are
    committed anywhere.
    """
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    sp = pd.read_parquet(splits_path, columns=["game_id", "split"])
    wanted = set(sp.game_id[sp.split == split])
    if not wanted:
        raise Refused([f"{splits_path.name} has no games in split {split!r}"])
    if verbose:
        print(f"\nheld-out sample: {len(wanted):,} games in split {split!r} "
              f"({splits_path.name})")

    pf = pq.ParquetFile(features_path)
    chunks = []
    total = 0
    for batch in pf.iter_batches(batch_size=250_000,
                                 columns=["game_id"] + list(names)):
        df = batch.to_pandas()
        total += len(df)
        keep = df[df.game_id.isin(wanted)]
        if len(keep):
            chunks.append(keep[list(names)].to_numpy(dtype=np.float32))
    if not chunks:
        raise Refused([f"no rows of {features_path.name} are in split {split!r}"])
    X = np.concatenate(chunks, axis=0)
    if verbose:
        print(f"  {features_path.name}: {total:,} rows scanned, "
              f"{X.shape[0]:,} in the split")
    if X.shape[0] > n_rows:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(X.shape[0], size=n_rows, replace=False))
        X = np.ascontiguousarray(X[idx])
        if verbose:
            print(f"  sampled {X.shape[0]:,} rows with default_rng({seed})")
    return pd.DataFrame(X, columns=list(names))


# ---------------------------------------------------------------------------
# self-test for the evaluator: hand-built trees, no model and no data
# ---------------------------------------------------------------------------


def _tiny(feature, threshold, default_left, missing_type, leaves):
    return {"feature": [feature], "threshold": [threshold],
            "default_left": [default_left], "missing_type": [missing_type],
            "left": [-1], "right": [-2], "leaf_value": list(leaves)}


def _model_of(*trees, sigmoid=1.0):
    return {"sigmoid": sigmoid, "trees": list(trees)}


def self_test_evaluator(report) -> None:
    """Every branch of NumericalDecision, with the answer worked out by hand.

    The one that matters most is the missing_type == None node: LightGBM turns
    the NaN into 0.0 and then compares, so the default branch is NOT taken. A
    port that short-circuits every missing value to the default is wrong there,
    and this is the case that catches it.
    """
    import math

    nan = float("nan")

    def score(model, row):
        return eval_rows_scalar(model, [row])[0]

    # missing_type NaN: the default branch really is taken.
    m = _model_of(_tiny(0, 1.5, 1, MISSING_NAN, [10.0, 20.0]))
    report("NaN node, value below threshold goes left", score(m, [1.0]) == 10.0)
    report("NaN node, value above threshold goes right", score(m, [2.0]) == 20.0)
    report("NaN node, value exactly on the threshold goes left",
           score(m, [1.5]) == 10.0)
    report("NaN node, missing takes default_left", score(m, [nan]) == 10.0)
    m = _model_of(_tiny(0, 1.5, 0, MISSING_NAN, [10.0, 20.0]))
    report("NaN node with default_left=0, missing goes right",
           score(m, [nan]) == 20.0)

    # missing_type None: the NaN becomes 0.0 and is then compared.
    m = _model_of(_tiny(0, ZERO_THRESHOLD, 0, MISSING_NONE, [10.0, 20.0]))
    report("None node, missing becomes 0.0 and compares left, "
           "ignoring default_left=0", score(m, [nan]) == 10.0)
    report("None node, 0.0 compares left", score(m, [0.0]) == 10.0)
    report("None node, a positive value compares right", score(m, [1.0]) == 20.0)
    m = _model_of(_tiny(0, -5.0, 1, MISSING_NONE, [10.0, 20.0]))
    report("None node with a negative threshold, missing becomes 0.0 and goes "
           "right, ignoring default_left=1", score(m, [nan]) == 20.0)

    # missing_type Zero: a zero takes the default branch.
    m = _model_of(_tiny(0, -5.0, 1, MISSING_ZERO, [10.0, 20.0]))
    report("Zero node, 0.0 takes default_left", score(m, [0.0]) == 10.0)
    report("Zero node, missing becomes 0.0 and then takes default_left",
           score(m, [nan]) == 10.0)
    report("Zero node, a value just outside the zero band is compared",
           score(m, [1e-34]) == 20.0)
    report("Zero node, a value inside the zero band takes the default",
           score(m, [1e-36]) == 10.0)

    # accumulation and the sigmoid
    m = _model_of(_tiny(0, 0.0, 1, MISSING_NONE, [1.0, 2.0]),
                  _tiny(0, 0.0, 1, MISSING_NONE, [0.25, 0.5]))
    report("leaf values sum across trees in order", score(m, [1.0]) == 2.5)
    report("sigmoid matches 1/(1+exp(-s)) at sigmoid=1",
           abs(float(sigmoid_of(m, [2.5])[0]) - 1.0 / (1.0 + math.exp(-2.5)))
           <= 5e-16)
    m2 = _model_of(_tiny(0, 0.0, 1, MISSING_NONE, [1.0, 2.0]), sigmoid=2.0)
    report("sigmoid parameter is applied",
           abs(float(sigmoid_of(m2, [2.0])[0]) - 1.0 / (1.0 + math.exp(-4.0)))
           <= 5e-16)

    # a deeper tree, so the child encoding is exercised beyond a single split
    deep = {"feature": [0, 1], "threshold": [1.5, 2.5],
            "default_left": [1, 1], "missing_type": [MISSING_NONE] * 2,
            "left": [1, -1], "right": [-3, -2],
            "leaf_value": [10.0, 20.0, 30.0]}
    m = _model_of(deep)
    report("deep tree, left then left", score(m, [1.0, 1.0]) == 10.0)
    report("deep tree, left then right", score(m, [1.0, 9.0]) == 20.0)
    report("deep tree, right at the root", score(m, [9.0, 1.0]) == 30.0)

    # the isotonic step
    calib = {"x": [0.0, 0.5, 1.0], "y": [0.0, 0.25, 1.0],
             "clip_lo": 0.0, "clip_hi": 1.0}
    got = [float(v) for v in apply_isotonic(calib, [-1.0, 0.0, 0.25, 0.5, 0.75,
                                                    1.0, 2.0])]
    report("isotonic clips below the first knot", got[0] == 0.0)
    report("isotonic clips above the last knot", got[6] == 1.0)
    report("isotonic hits its knots exactly",
           got[1] == 0.0 and got[3] == 0.25 and got[5] == 1.0)
    report("isotonic interpolates linearly between knots",
           abs(got[2] - 0.125) <= 1e-15 and abs(got[4] - 0.625) <= 1e-15)

    # the vectorised evaluator must agree with the scalar one everywhere above
    try:
        import numpy as np
    except ImportError:                                  # pragma: no cover
        report("numpy is available for the vectorised evaluator", False)
        return
    cases = [
        (_model_of(_tiny(0, 1.5, 1, MISSING_NAN, [10.0, 20.0])),
         [[1.0], [2.0], [1.5], [nan]]),
        (_model_of(_tiny(0, ZERO_THRESHOLD, 0, MISSING_NONE, [10.0, 20.0])),
         [[nan], [0.0], [1.0]]),
        (_model_of(_tiny(0, -5.0, 1, MISSING_ZERO, [10.0, 20.0])),
         [[0.0], [nan], [1e-34], [1e-36]]),
        (_model_of(deep), [[1.0, 1.0], [1.0, 9.0], [9.0, 1.0], [nan, nan]]),
    ]
    agree = True
    for model, rows in cases:
        a = eval_rows_scalar(model, rows)
        b = eval_rows_vector(model, np.asarray(rows, dtype=np.float32))
        agree = agree and all(x == y for x, y in zip(a, list(b)))
    report("the vectorised evaluator agrees with the scalar one, bit for bit",
           agree)


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export an approved model for the web app. Runs the "
                    "provenance guards first; see the module docstring.")
    ap.add_argument("--model", help="artefact base path, e.g. models/full_free")
    ap.add_argument("--approved", help="path to docs/model-approvals.json in "
                                       "the app repo")
    ap.add_argument("--out", help="output directory for the exported payload")
    ap.add_argument("--check-only", action="store_true",
                    help="run the guards, print the provenance stamp, write "
                         "nothing")
    ap.add_argument("--allow-ablation", action="store_true",
                    help="permit an artefact whose role is 'ablation'. It is "
                         "not a shipping candidate; say why in DECISIONS.md.")
    ap.add_argument("--self-test", action="store_true",
                    help="plant faults against synthetic artefacts and check "
                         "the evaluator against hand-worked cases; no data "
                         "and no model needed")

    g = ap.add_argument_group("golden fixtures (the committed 500-game sample)")
    g.add_argument("--fixtures", help="directory for the golden fixtures")
    g.add_argument("--sample-positions", help="parse_lichess.py output for "
                                              "data/sample_games.pgn")
    g.add_argument("--sample-features", help="build_features.py output for the "
                                             "same positions")
    g.add_argument("--fixture-games", type=int, default=60,
                   help="how many games go into sample_plies.json (default 60)")
    g.add_argument("--fixture-cases", type=int, default=1200,
                   help="how many rows go into model_cases.json (default 1200)")

    v = ap.add_argument_group("held-out verification (read-only, never committed)")
    v.add_argument("--verify-features", help="a larger feature table, e.g. the "
                                             "full-month features parquet")
    v.add_argument("--verify-splits", help="the matching splits parquet")
    v.add_argument("--verify-split", default="test",
                   help="which split to sample (default test)")
    v.add_argument("--verify-rows", type=int, default=50_000,
                   help="how many rows to verify (default 50000)")
    v.add_argument("--scalar-rows", type=int, default=500,
                   help="rows the plain-Python evaluator also scores, to pin "
                        "the vectorised one (default 500)")
    ap.add_argument("--seed", type=int, default=20260923,
                    help="seed for every sample drawn here (default 20260923)")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.model or not args.approved:
        ap.error("--model and --approved are required (or use --self-test)")
    if not args.check_only and not args.out:
        ap.error("--out is required (or use --check-only)")
    if args.fixtures and not (args.sample_positions and args.sample_features):
        ap.error("--fixtures needs --sample-positions and --sample-features")
    if args.verify_features and not args.verify_splits:
        ap.error("--verify-features needs --verify-splits")

    model_base = Path(args.model)
    try:
        approved = load_approved(Path(args.approved))
        if args.check_only:
            stamp = check_provenance(model_base, approved,
                                     allow_ablation=args.allow_ablation)
            print("\nprovenance stamp:")
            print(json.dumps(stamp, indent=2))
            return 0

        out_dir = Path(args.out)
        spec = export_payload(model_base, approved, out_dir,
                              allow_ablation=args.allow_ablation)

        summary = []
        if args.sample_features:
            import pandas as pd
            feat = pd.read_parquet(args.sample_features)
            res, X, raw, p_sig, p_cal = verify_against_lightgbm(
                model_base, out_dir, feat, scalar_rows=args.scalar_rows,
                label="rows of the 500-game sample")
            summary.append(res)
            if args.fixtures:
                pos = pd.read_parquet(args.sample_positions)
                names = [f["name"] for f in spec["features"]]
                plies, cases = build_fixtures(
                    spec, X, names, pos, feat, raw, p_sig, p_cal,
                    n_games=args.fixture_games, n_cases=args.fixture_cases,
                    seed=args.seed)
                fx = Path(args.fixtures)
                write_json(fx / "sample_plies.json", plies)
                write_json(fx / "model_cases.json", cases)
                print(f"\nwrote into {fx}:")
                for name, obj in (("sample_plies.json", plies),
                                  ("model_cases.json", cases)):
                    size = (fx / name).stat().st_size
                    print(f"  {name:<20} {size / 1e6:8.2f} MB  "
                          f"{obj['n_rows']:,} rows")

        if args.verify_features:
            names = [f["name"] for f in spec["features"]]
            big = load_heldout_rows(Path(args.verify_features),
                                    Path(args.verify_splits), names,
                                    split=args.verify_split,
                                    n_rows=args.verify_rows, seed=args.seed)
            res, *_ = verify_against_lightgbm(
                model_base, out_dir, big, scalar_rows=args.scalar_rows,
                label=f"held-out rows (split {args.verify_split})")
            summary.append(res)

        if summary:
            print("\nsummary: max absolute difference against booster.predict")
            for r in summary:
                print(f"  {r['label']:<40} n={r['n_rows']:>7,}  "
                      f"raw {r['max_abs_raw']:.3e}  p {r['max_abs_prob']:.3e}  "
                      f"isotonic {r['max_abs_calibrated']:.3e}")
            worst = max(r["max_abs_prob"] for r in summary)
            if worst > 1e-9:
                print(f"\nFAIL: the worst probability difference is {worst:.3e}, "
                      f"above the 1e-9 parity tolerance.")
                return 1
            if not all(r["scalar_matches_vector_exactly"] for r in summary):
                print("\nFAIL: the plain-Python and vectorised evaluators "
                      "disagree.")
                return 1
            print("\nPASS: the exported files reproduce booster.predict within "
                  "the parity tolerance.")
    except Refused as exc:
        print("\nREFUSED. The model was not exported:", file=sys.stderr)
        for reason in exc.reasons:
            print(f"  - {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
