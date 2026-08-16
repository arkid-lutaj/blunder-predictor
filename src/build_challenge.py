#!/usr/bin/env python3
"""
Build the challenge game: a static page, no server, no engine at runtime.

How it works. Every position shown to the player already had EVERY legal move
evaluated by Stockfish in eval_children.py, so the page knows the truth about
any move the player picks without running anything. It presents a handful of
candidate moves, the player picks one, and the page reveals whether that move
was a blunder.

The part that makes it more than a tactics trainer: alongside the truth, it
shows what the model predicted. "A 1500 blunders here 23% of the time." Play
ten positions and the page compares your hit rate against the model's
prediction curve and reports the rating whose predicted blunder rate matches
yours. That is a rating estimate produced by the error model, which is a much
better demo of the project than a static chart.

Requires:
    eval_children.py run with --per-move (writes *_moves.jsonl)
    an engine_free model, so no Stockfish is needed in the browser

Usage:
    python build_challenge.py --children data/children_40k.parquet \
        --moves data/children_40k_moves.jsonl \
        --features data/features_blitz.parquet \
        --model models/blitz_free --out docs/
"""

import argparse
import json
import os

import chess
import chess.svg
import lightgbm as lgb
import numpy as np
import pandas as pd

RATINGS = list(range(800, 2451, 50))


def load_iso(prefix: str):
    p = f"{prefix}_iso.npz"
    if not os.path.exists(p):
        return lambda v: v
    d = np.load(p)
    return lambda v: np.interp(v, d["x"], d["y"])


def pick_candidates(moves, board, k=5, rng=None):
    """A playable multiple choice: at least one blunder, at least one good move."""
    rng = rng or np.random.default_rng(0)
    blunders = [m for m in moves if m["blunder"]]
    good = [m for m in moves if not m["blunder"]]
    if not blunders or not good:
        return None
    n_bl = min(len(blunders), max(1, k // 2))
    n_gd = min(len(good), k - n_bl)
    chosen = (list(rng.choice(blunders, n_bl, replace=False))
              + list(rng.choice(good, n_gd, replace=False)))
    out = []
    for m in chosen:
        try:
            san = board.san(chess.Move.from_uci(m["uci"]))
        except Exception:
            continue
        out.append({"uci": m["uci"], "san": san,
                    "drop": m["drop"], "blunder": bool(m["blunder"])})
    rng.shuffle(out)
    return out if len(out) >= 3 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--children", required=True)
    ap.add_argument("--moves", default=None, help="default: <children>_moves.jsonl")
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", required=True, help="prefix, e.g. models/blitz_free")
    ap.add_argument("--out", default="docs/")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    moves_path = args.moves or (os.path.splitext(args.children)[0] + "_moves.jsonl")
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    kids = pd.read_parquet(args.children)
    kids = kids[kids.blunder_available]
    feats_df = pd.read_parquet(args.features)
    feats_df = feats_df[feats_df.label_valid].reset_index(drop=True)
    feats_df["row_id"] = feats_df.index

    # row_id is a POSITIONAL index into the label_valid rows of whatever
    # parquet eval_children.py was pointed at. Point this script at a different
    # one -- features_blitz.parquet instead of features_blitz_full.parquet, say
    # -- and every id still resolves, silently, to an unrelated position. Both
    # sides carry the FEN, so the mismatch is free to detect: check it rather
    # than trusting that the two commands were given matching arguments.
    probe = kids.head(200)
    ok = feats_df.fen.reindex(probe.row_id).to_numpy()
    bad = int((ok != probe.fen.to_numpy()).sum())
    if bad:
        raise SystemExit(
            f"FATAL: {bad}/{len(probe)} sampled row_ids disagree on FEN between\n"
            f"  --children {args.children}\n"
            f"  --features {args.features}\n"
            f"row_id is positional, so these two files must be the same parquet "
            f"eval_children.py\nwas run against. Nothing downstream is "
            f"meaningful until they match.")
    print(f"row_id/FEN consistency: {len(probe)}/{len(probe)} OK "
          f"({len(feats_df):,} label_valid rows in --features)")

    booster = lgb.Booster(model_file=f"{args.model}.txt")
    meta = json.load(open(f"{args.model}_meta.json"))
    feat_names = meta["features"]
    if "mover_elo" not in feat_names:
        print("FATAL: model has no mover_elo feature; a rating sweep is meaningless")
        return 1
    iso = load_iso(args.model)

    per_move = {}
    with open(moves_path) as fh:
        for line in fh:
            rec = json.loads(line)
            per_move[rec["row_id"]] = rec["moves"]

    pool = kids[kids.row_id.isin(per_move)].sample(
        min(len(kids), args.n * 3), random_state=args.seed)

    out_positions = []
    elo_idx = feat_names.index("mover_elo")

    for row in pool.itertuples():
        if len(out_positions) >= args.n:
            break
        board = chess.Board(row.fen)
        cands = pick_candidates(per_move[row.row_id], board, rng=rng)
        if not cands:
            continue

        frow = feats_df[feats_df.row_id == row.row_id]
        if not len(frow):
            continue
        base = frow[feat_names].to_numpy(dtype=np.float32)[0]

        grid = np.repeat(base[None, :], len(RATINGS), axis=0)
        grid[:, elo_idx] = RATINGS
        curve = iso(booster.predict(grid))

        svg = chess.svg.board(board, size=360,
                              orientation=board.turn, coordinates=True)
        out_positions.append({
            "id": int(row.row_id),
            "fen": row.fen,
            "svg": svg,
            "to_move": "White" if board.turn else "Black",
            "candidates": cands,
            "curve": [round(float(c), 5) for c in curve],
            "n_moves": int(row.n_moves),
            "frac_blunder_moves": round(float(row.frac_blunder_moves), 4),
        })

    if not out_positions:
        print("FATAL: no usable positions. Did eval_children run with --per-move?")
        return 1

    payload = {"ratings": RATINGS, "positions": out_positions,
               "model": os.path.basename(args.model),
               "feature_set": meta["feature_set"]}
    dest = os.path.join(args.out, "challenge.json")
    with open(dest, "w") as fh:
        json.dump(payload, fh)

    html_path = os.path.join(args.out, "challenge.html")
    with open(html_path, "w") as fh:
        fh.write(HTML)

    sizes = os.path.getsize(dest) / 1e6
    print(f"{len(out_positions)} positions -> {dest} ({sizes:.1f} MB)")
    print(f"wrote {html_path}")
    print(f"mean candidates/position: "
          f"{np.mean([len(p['candidates']) for p in out_positions]):.1f}")
    print("\nopen it locally with:  python -m http.server -d "
          f"{args.out} 8000   then visit http://localhost:8000/challenge.html")
    return 0


HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>Blunder Challenge</title>
<style>
  :root { --bg:#12141a; --fg:#e8e8ea; --dim:#8b8f9a; --acc:#6ea8fe;
          --bad:#f2777a; --good:#7ec699; --card:#1b1e26; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.55
         ui-sans-serif,system-ui,-apple-system,sans-serif; }
  .wrap { max-width:860px; margin:0 auto; padding:24px 18px 80px; }
  h1 { font-size:22px; margin:0 0 4px; letter-spacing:-.01em; }
  .sub { color:var(--dim); font-size:13px; margin-bottom:22px; }
  .card { background:var(--card); border:1px solid #262a34; border-radius:12px;
          padding:18px; margin-bottom:16px; }
  .board { display:flex; justify-content:center; margin-bottom:14px; }
  .board svg { max-width:100%; height:auto; border-radius:6px; }
  .moves { display:flex; flex-wrap:wrap; gap:8px; }
  button.mv { font:600 15px ui-monospace,SFMono-Regular,Menlo,monospace;
        background:#232735; color:var(--fg); border:1px solid #333949;
        border-radius:8px; padding:10px 16px; cursor:pointer; min-width:76px; }
  button.mv:hover:not(:disabled) { border-color:var(--acc); }
  button.mv:disabled { cursor:default; opacity:.95; }
  .mv.pick-bad { background:#3a2226; border-color:var(--bad); color:#ffd9da; }
  .mv.pick-good { background:#1e3227; border-color:var(--good); color:#d6f2e2; }
  .mv.reveal-bad { border-color:#5a3237; }
  .row { display:flex; justify-content:space-between; align-items:baseline;
         gap:12px; flex-wrap:wrap; }
  .num { font:600 28px ui-monospace,monospace; }
  .dim { color:var(--dim); font-size:13px; }
  input[type=range] { width:100%; accent-color:var(--acc); }
  .verdict { margin-top:14px; padding:12px 14px; border-radius:8px;
             background:#1f232d; font-size:14px; }
  .bar { height:6px; background:#262a34; border-radius:3px; overflow:hidden;
         margin-top:6px; }
  .bar > i { display:block; height:100%; background:var(--acc); }
  .ctr { text-align:center; }
  svg.curve { width:100%; height:110px; }
</style>
<div class="wrap">
  <h1>Blunder Challenge</h1>
  <div class="sub">Every legal move in these positions was evaluated by
    Stockfish, so the page knows the truth about whatever you pick. The model
    never saw the answer &mdash; it only predicts how often a player of a given
    rating blunders here.</div>

  <div class="card">
    <label class="dim">Your rating: <b id="eloLabel">1500</b></label>
    <input type="range" id="elo" min="800" max="2450" step="50" value="1500">
    <div class="row" style="margin-top:10px">
      <div><span class="num" id="score">0</span><span class="dim"> / <span id="seen">0</span> clean</span></div>
      <div class="dim ctr">model expected <b id="expected">0.0</b> blunders from you</div>
      <div class="dim">you played like <b id="implied">&mdash;</b></div>
    </div>
  </div>

  <div class="card" id="posCard">
    <div class="board" id="board"></div>
    <div class="row" style="margin-bottom:10px">
      <div class="dim"><b id="toMove">White</b> to move &mdash; pick a move</div>
      <div class="dim" id="posMeta"></div>
    </div>
    <div class="moves" id="moves"></div>
    <div class="verdict" id="verdict" style="display:none"></div>
    <div id="curveBox" style="display:none;margin-top:14px">
      <div class="dim">predicted blunder rate by rating</div>
      <svg class="curve" id="curve" viewBox="0 0 600 110" preserveAspectRatio="none"></svg>
    </div>
  </div>

  <div class="ctr"><button class="mv" id="next">Next position &rarr;</button></div>
</div>
<script>
let D=null, order=[], idx=0, seen=0, clean=0, expSum=0, answered=false;
const $=id=>document.getElementById(id);

fetch('challenge.json').then(r=>r.json()).then(d=>{
  D=d; order=d.positions.map((_,i)=>i);
  for(let i=order.length-1;i>0;i--){const j=(Math.random()*(i+1))|0;[order[i],order[j]]=[order[j],order[i]];}
  render();
});

function ratingIdx(){ return D.ratings.indexOf(+$('elo').value); }
function pCur(p){ const i=ratingIdx(); return p.curve[i<0?0:i]; }

$('elo').oninput=()=>{ $('eloLabel').textContent=$('elo').value;
  if(D&&answered) drawCurve(D.positions[order[idx]]); };
$('next').onclick=()=>{ idx=(idx+1)%order.length; render(); };

function render(){
  answered=false;
  const p=D.positions[order[idx]];
  $('board').innerHTML=p.svg;
  $('toMove').textContent=p.to_move;
  $('posMeta').textContent=p.n_moves+' legal moves';
  $('verdict').style.display='none';
  $('curveBox').style.display='none';
  const box=$('moves'); box.innerHTML='';
  p.candidates.forEach(c=>{
    const b=document.createElement('button');
    b.className='mv'; b.textContent=c.san;
    b.onclick=()=>answer(p,c,b);
    box.appendChild(b);
  });
}

function answer(p,c,btn){
  if(answered) return; answered=true;
  const pred=pCur(p);
  seen++; expSum+=pred; if(!c.blunder) clean++;
  [...$('moves').children].forEach((b,i)=>{
    b.disabled=true;
    if(p.candidates[i].blunder) b.classList.add('reveal-bad');
  });
  btn.classList.add(c.blunder?'pick-bad':'pick-good');

  $('verdict').style.display='block';
  $('verdict').innerHTML =
    (c.blunder
      ? `<b style="color:var(--bad)">Blunder.</b> ${c.san} drops ${c.drop.toFixed(1)} win%.`
      : `<b style="color:var(--good)">Fine.</b> ${c.san} costs only ${c.drop.toFixed(1)} win%.`)
    + `<br><span class="dim">The model predicted a ${$('elo').value} blunders here
       <b>${(pred*100).toFixed(1)}%</b> of the time.
       ${(p.frac_blunder_moves*100).toFixed(0)}% of all legal moves here are blunders.</span>
       <div class="bar"><i style="width:${Math.min(100,pred*100*3).toFixed(1)}%"></i></div>`;

  $('score').textContent=clean;
  $('seen').textContent=seen;
  $('expected').textContent=expSum.toFixed(1);
  $('implied').textContent=impliedRating();
  $('curveBox').style.display='block';
  drawCurve(p);
}

// Find the rating whose average predicted blunder rate, over the positions
// actually shown, matches the player's observed rate. Meaningless below ~5
// positions, so say so rather than printing a confident number.
function impliedRating(){
  if(seen<5) return `\u2014 (need ${5-seen} more)`;
  const shown=order.slice(0,idx+1).map(i=>D.positions[i]);
  const obs=(seen-clean)/seen;
  let best=0,bd=1e9;
  D.ratings.forEach((r,ri)=>{
    const m=shown.reduce((s,p)=>s+p.curve[ri],0)/shown.length;
    if(Math.abs(m-obs)<bd){bd=Math.abs(m-obs);best=r;}
  });
  return best+(bd>0.15?'?':'');
}

function drawCurve(p){
  const W=600,H=110,pad=6, n=p.curve.length, mx=Math.max(...p.curve,0.01);
  const pts=p.curve.map((v,i)=>[pad+i*(W-2*pad)/(n-1), H-pad-(v/mx)*(H-2*pad)]);
  const d=pts.map((q,i)=>(i?'L':'M')+q[0].toFixed(1)+' '+q[1].toFixed(1)).join(' ');
  const i=ratingIdx(), cx=pad+(i<0?0:i)*(W-2*pad)/(n-1);
  $('curve').innerHTML=
    `<path d="${d}" fill="none" stroke="#6ea8fe" stroke-width="2"/>`+
    `<line x1="${cx}" y1="0" x2="${cx}" y2="${H}" stroke="#8b8f9a" stroke-dasharray="3 3"/>`;
}
</script>
"""


if __name__ == "__main__":
    raise SystemExit(main())
