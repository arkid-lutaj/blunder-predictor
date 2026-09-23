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

USAGE

    python src/export_web.py --model models/full_free \\
      --approved ../docs/model-approvals.json --check-only

    python src/export_web.py --self-test     # synthetic files only, no data

The export payload itself -- model JSON, calibration, feature_spec.json --
lands in M0.2, once D11 has picked which artefact ships. What is finished here
is the gate in front of it and the provenance stamp that goes into
feature_spec.json. Nothing in this file writes to build/.
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
    print(f"PASS: {len(results)} checks, every planted fault refused")
    return 0


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
    ap.add_argument("--out", help="output directory for the exported files")
    ap.add_argument("--check-only", action="store_true",
                    help="run the guards, print the provenance stamp, write "
                         "nothing")
    ap.add_argument("--allow-ablation", action="store_true",
                    help="permit an artefact whose role is 'ablation'. It is "
                         "not a shipping candidate; say why in DECISIONS.md.")
    ap.add_argument("--self-test", action="store_true",
                    help="plant faults against synthetic artefacts, no data "
                         "needed")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.model or not args.approved:
        ap.error("--model and --approved are required (or use --self-test)")
    if not args.check_only and not args.out:
        ap.error("--out is required (or use --check-only)")

    try:
        approved = load_approved(Path(args.approved))
        stamp = check_provenance(Path(args.model), approved,
                                 allow_ablation=args.allow_ablation)
    except Refused as exc:
        print("\nREFUSED. The model was not exported:", file=sys.stderr)
        for reason in exc.reasons:
            print(f"  - {reason}", file=sys.stderr)
        return 1

    print("\nprovenance stamp:")
    print(json.dumps(stamp, indent=2))

    if args.check_only:
        return 0

    # M0.2 continues here: write feature_spec.json (with this stamp under
    # "provenance"), the model JSON, the calibration data and the golden
    # fixtures into args.out. D11 picks which artefact that is.
    print(f"\nGuards passed for {stamp['artefact']}, but the export payload is "
          f"not written yet.\nIt lands in M0.2, after D11 chooses the shipping "
          f"artefact. Use --check-only\nuntil then; nothing was written to "
          f"{args.out}.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
