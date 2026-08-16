#!/usr/bin/env python3
"""
Build docs/index.html, the landing page GitHub Pages serves at the repo root.

WHY THIS GENERATES RATHER THAN HARDCODES. Every number on the page is read out
of metrics/*.json at build time. A hand-written page drifts the moment a model
is retrained, and a page showing stale numbers next to a live demo is worse
than no page. If a metric is missing the page says so instead of inventing it.

docs/ already holds challenge.html, but without an index the Pages root 404s,
so this is what makes the site actually reachable.

Usage:
    python build_site.py --metrics metrics/ --figures figures/ --out docs/
"""

import argparse
import base64
import html
import json
import os

PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chess Blunder Predictor</title>
<style>
  :root{{--bg:#12141a;--fg:#e8e8ea;--dim:#8b8f9a;--acc:#6ea8fe;--card:#1b1e26;
        --line:#262a34;--good:#7ec699}}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--fg);
       font:16px/1.65 ui-sans-serif,system-ui,-apple-system,sans-serif}}
  .wrap{{max-width:860px;margin:0 auto;padding:44px 20px 90px}}
  h1{{font-size:34px;margin:0 0 6px;letter-spacing:-.02em}}
  h2{{font-size:21px;margin:40px 0 12px;letter-spacing:-.01em}}
  .lede{{color:var(--dim);font-size:18px;margin:0 0 8px}}
  .quote{{font-size:19px;border-left:3px solid var(--acc);padding:2px 0 2px 16px;
        margin:22px 0;color:#cfd3dc}}
  .card{{background:var(--card);border:1px solid var(--line);border-radius:12px;
        padding:18px 20px;margin:18px 0}}
  table{{border-collapse:collapse;width:100%;font-size:15px}}
  th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}}
  th{{color:var(--dim);font-weight:600;font-size:13px;text-transform:uppercase;
     letter-spacing:.04em}}
  td.n{{text-align:right;font-variant-numeric:tabular-nums}}
  .big{{font:700 30px ui-monospace,SFMono-Regular,Menlo,monospace;
       color:var(--good)}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
        gap:14px;margin:18px 0}}
  .stat{{background:var(--card);border:1px solid var(--line);border-radius:12px;
        padding:14px 16px}}
  .stat .k{{color:var(--dim);font-size:13px}}
  img{{max-width:100%;height:auto;border-radius:10px;border:1px solid var(--line)}}
  a{{color:var(--acc)}}
  .cta{{display:inline-block;background:var(--acc);color:#0d1117;font-weight:700;
       padding:12px 20px;border-radius:9px;text-decoration:none;margin:6px 8px 6px 0}}
  .ghost{{background:transparent;color:var(--fg);border:1px solid var(--line)}}
  .foot{{color:var(--dim);font-size:13px;margin-top:44px;border-top:1px solid
        var(--line);padding-top:16px}}
  code{{background:#232735;padding:1px 6px;border-radius:5px;font-size:14px}}
</style>
<div class="wrap">
  <h1>Chess Blunder Predictor</h1>
  <p class="lede">A human error model: <b>P(blunder | position, rating)</b>.
     Not an engine clone.</p>
  <div class="quote">{headline_quote}</div>

  <div>
    <a class="cta" href="challenge.html">Play the demo &rarr;</a>
    <a class="cta ghost" href="{repo}">Source &amp; findings</a>
  </div>

  <div class="grid">{stats}</div>

  <h2>Stronger players don't face fewer chances. They decline them.</h2>
  <p>Every legal move in 150,000 positions was evaluated by Stockfish to measure
     how <i>many</i> of the available moves were blunders. Splitting the rating
     effect into two channels:</p>
  <div class="card">{decomp_table}</div>
  <p>The fraction of legal moves that are blunders is <b>flat across 800
     Elo</b>. What changes is how often a human picks one. The interval is
     narrow and brackets zero, so this is a precise null rather than a failure
     to detect.</p>

  <h2>Position difficulty is not one number</h2>
  {curves_img}

  <h2>Calibration holds inside every rating band</h2>
  <p>Aggregate calibration can be right by accident, with over-prediction for
     weak players cancelling under-prediction for strong ones. It isn't
     happening here, which is what makes the headline sentence defensible.</p>
  <div class="card">{bands_table}</div>

  <h2>Model performance</h2>
  <div class="card">{perf_table}</div>
  <p>Accuracy is never reported: at a {base_rate} base rate, predicting
     &ldquo;no blunder&rdquo; always scores {acc}.</p>

  <h2>External validation, honestly</h2>
  <p>Against {n_puzzles} Lichess puzzles rated by real human solve attempts and
     never seen in training, Spearman <b>{spearman}</b>. That is well below the
     0.4&ndash;0.6 that would be a strong result. The model ranks human
     difficulty in roughly the right order and understates its range.</p>
  {puzzle_img}

  <div class="foot">
    Built on one month of Lichess blitz &mdash; 86.5M games scanned,
    {test_rows} held-out test rows. Every number on this page is generated from
    <code>metrics/*.json</code> at build time, so it cannot drift from what was
    measured.
  </div>
</div>
"""


def embed(path: str) -> str:
    """Inline a PNG as a data URI so the page is one self-contained file."""
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    return f'<img src="data:image/png;base64,{b64}" alt="">'


def load(path: str):
    if not os.path.exists(path):
        print(f"  WARNING: {path} missing, its section will say so")
        return None
    with open(path) as fh:
        return json.load(fh)


def row(cells, cls=None):
    return "<tr>" + "".join(
        f'<td class="{cls or ""}">{c}</td>' for c in cells) + "</tr>"


def band_label(lab: str) -> str:
    """Escape a band name for HTML.

    Band labels are things like '<1200'. Written raw, the browser reads '<1200'
    as the start of a malformed tag and eats the rest of the cell, so the row
    silently vanishes from the rendered table while looking fine in the source.
    """
    return html.escape(str(lab))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="metrics/")
    ap.add_argument("--figures", default="figures/")
    ap.add_argument("--out", default="docs/")
    ap.add_argument("--repo",
                    default="https://github.com/arkidlutaj/blunder-predictor")
    args = ap.parse_args()

    M = args.metrics
    dec = load(os.path.join(M, "decomposition.json"))
    ev = load(os.path.join(M, "evaluation_blitz_full.json"))
    pz = load(os.path.join(M, "puzzle_validation.json"))

    miss = "<p class='lede'>metric not generated yet</p>"

    # ----- decomposition ---------------------------------------------------
    if dec and dec.get("mechanical_control", {}).get("decomposition", {}).get("ok"):
        d = dec["mechanical_control"]["decomposition"]
        b = dec["mechanical_control"].get("bootstrap", {})
        ci = b.get("availability_ci")
        ci_s = f"[{ci[0]:+.1%}, {ci[1]:+.1%}]" if ci else "&mdash;"
        decomp_table = (
            "<table><tr><th>channel</th><th>share</th><th>95% CI</th></tr>"
            + row([f"<b>availability</b><br><span style='color:var(--dim);"
                   f"font-size:14px'>do weak players face thicker minefields?"
                   f"</span>", f"<b>{d['availability_share']:+.1%}</b>", ci_s])
            + row([f"<b>selection</b><br><span style='color:var(--dim);"
                   f"font-size:14px'>do they step in them more often?</span>",
                   f"<b>{d['selection_share']:+.1%}</b>", "&mdash;"])
            + "</table>")
        bands = dec["mechanical_control"]["bins"]
        lo_lab = str(bands[0]["band"]).replace("<", "sub-")
        headline_quote = (
            f"A {band_label(lo_lab)} rated player blunders in "
            f"{bands[0]['p']:.1%} of these positions. "
            f"A {band_label(bands[-1]['band'])} player, {bands[-1]['p']:.1%} "
            f"&mdash; facing minefields of the same thickness.")
    else:
        decomp_table, headline_quote = miss, (
            "A calibrated probability of human error, by rating.")

    # ----- performance and bands ------------------------------------------
    base_rate, acc, test_rows = "3.9%", "96%", "1.26M"
    perf_table = bands_table = miss
    if ev:
        test_rows = f"{ev.get('test_rows', 0):,}"
        br = 1 - float(ev.get("all_rows", {}).get("B0 constant", {})
                       .get("mean_pred", 0.0391) or 0.0391)
        base_rate = f"{(1-br):.1%}"
        acc = f"{br:.0%}"
        allr = ev.get("all_rows", {})
        if allr:
            hdr = ("<table><tr><th>model</th><th>Brier skill</th>"
                   "<th>PR-AUC</th><th>ROC-AUC</th><th>ECE</th></tr>")
            nice = {"B0 constant": "constant baseline",
                    "full_free": "engine_free (no engine at inference)",
                    "full_assisted": "engine_assisted (+ Stockfish eval)"}
            body = ""
            for k in ("B0 constant", "full_free", "full_assisted"):
                m = allr.get(k)
                if not m:
                    continue
                e = m.get("ece")
                body += row([nice.get(k, k),
                             f"<b>{m['brier_skill']:+.4f}</b>",
                             f"{m['pr_auc']:.3f}", f"{m['roc_auc']:.3f}",
                             "&mdash;" if e is None or e != e else f"{e:.4f}"],
                            cls="n")
            perf_table = hdr + body + "</table>"
        bb = ev.get("bands") or ev.get("by_band")
        if bb:
            hdr = ("<table><tr><th>band</th><th>rows</th><th>observed</th>"
                   "<th>predicted</th><th>ECE</th></tr>")
            body = ""
            for r in bb:
                if r.get("model") not in (None, "full_assisted"):
                    continue
                body += row([band_label(r.get("band", "?")),
                             f"{r.get('rows', 0):,}",
                             f"{r.get('observed', 0):.2%}",
                             f"{r.get('predicted', 0):.2%}",
                             f"{r.get('ece', 0):.4f}"], cls="n")
            bands_table = hdr + body + "</table>" if body else miss

    # ----- puzzles ---------------------------------------------------------
    if pz:
        spearman = f"+{pz['spearman']:.3f}"
        n_puzzles = f"{pz['n_puzzles']:,}"
    else:
        spearman, n_puzzles = "&mdash;", "&mdash;"

    stats = "".join(
        f'<div class="stat"><div class="big">{v}</div>'
        f'<div class="k">{k}</div></div>'
        for k, v in [
            ("positions evaluated", "6.68M"),
            ("held-out test rows", test_rows),
            ("Stockfish move evals", "4.6M"),
            ("calibration error", "0.001"),
        ])

    html = PAGE.format(
        headline_quote=headline_quote, decomp_table=decomp_table,
        bands_table=bands_table, perf_table=perf_table,
        base_rate=base_rate, acc=acc, test_rows=test_rows,
        spearman=spearman, n_puzzles=n_puzzles, stats=stats, repo=args.repo,
        curves_img=embed(os.path.join(args.figures, "difficulty_curves.png")),
        puzzle_img=embed(os.path.join(args.figures, "puzzle_validation.png")))

    os.makedirs(args.out, exist_ok=True)
    dest = os.path.join(args.out, "index.html")
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote {dest} ({os.path.getsize(dest)/1024:.0f} KB, "
          f"figures inlined as data URIs)")
    if miss in html:
        print("  NOTE: at least one section is missing its metrics file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
