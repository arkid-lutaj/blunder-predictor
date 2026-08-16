#!/usr/bin/env python3
"""
Build the challenge game: a static page, no server, no engine at runtime.

How it works. Every position shown to the player already had EVERY legal move
evaluated by Stockfish in eval_children.py, so the page knows the truth about
any move the player picks without running anything. That is what lets the board
be a real board rather than a multiple choice, with no engine and no chess
library in the browser.

The part that makes it more than a tactics trainer: the model predicts how
often a player of each rating blunders in each position, and it never saw the
answers. Since it knows how hard every position is, how you do on a handful of
them is enough to estimate your rating. The page runs a posterior over the
rating grid rather than matching your hit rate to the nearest curve, so it can
show an interval that visibly narrows as you play.

POSITIONS ARE CHOSEN FOR INFORMATION, not at random. See position_info: a
position where a 1000 and a 2000 blunder equally often tells the estimate
nothing, however pretty the tactics are.

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

from build_features import check_sweep_span, rating_grid

RATINGS = list(range(800, 2451, 50))


def load_iso(prefix: str):
    p = f"{prefix}_iso.npz"
    if not os.path.exists(p):
        return lambda v: v
    d = np.load(p)
    return lambda v: np.interp(v, d["x"], d["y"])


def all_moves(moves, board):
    """Every legal move with its evaluation.

    eval_children.py --per-move already ran Stockfish over every legal child,
    so the page knows the truth about ANY move the player makes. That is what
    lets the board be free rather than a multiple choice, and it costs nothing
    extra: ~31 moves at ~40 bytes is 1.2 KB per position.
    """
    out = []
    for m in moves:
        try:
            mv = chess.Move.from_uci(m["uci"])
            san = board.san(mv)
        except Exception:
            continue
        out.append({"u": m["uci"], "s": san,
                    "w": round(float(m["win"]), 2),
                    "d": round(float(m["drop"]), 2),
                    "b": 1 if m["blunder"] else 0})
    if not out:
        return None
    # A position with no blunder available teaches nothing, and a position
    # where every move blunders is unfair. Require both kinds to exist.
    if not any(m["b"] for m in out) or all(m["b"] for m in out):
        return None
    return out


def position_info(curve: np.ndarray, lo_i: int = 4, hi_i: int = 24) -> float:
    """How well this position separates a weak player from a strong one.

    The expected log-likelihood ratio, in nats, between the two hypotheses
    "this player is 1000" and "this player is 2000", for one observation of
    whether they blundered:

        p_lo * log(p_lo/p_hi) + (1-p_lo) * log((1-p_lo)/(1-p_hi))

    That is exactly the quantity the page's rating estimate accumulates, so
    ranking on it maximises information per position asked. A position where
    both ratings blunder 5% of the time scores ~0 and is a wasted question no
    matter how pretty the tactics are.
    """
    p_lo = float(np.clip(curve[lo_i], 1e-6, 1 - 1e-6))
    p_hi = float(np.clip(curve[hi_i], 1e-6, 1 - 1e-6))
    return (p_lo * np.log(p_lo / p_hi)
            + (1 - p_lo) * np.log((1 - p_lo) / (1 - p_hi)))


def piece_sprite() -> str:
    """The twelve Cburnett piece groups from python-chess, embedded once."""
    return "".join(chess.svg.PIECES[k] for k in
                   ["K", "Q", "R", "B", "N", "P", "k", "q", "r", "b", "n", "p"])


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

    # row_id is POSITIONAL within the label_valid rows of the specific parquet
    # eval_children.py was pointed at. Point this script at a different one --
    # features_blitz.parquet instead of features_blitz_full.parquet, say -- and
    # every id still resolves, silently, to an unrelated position. Both sides
    # carry the FEN, so the mismatch is free to detect: check it rather than
    # trusting that the two commands were given matching arguments.
    probe = kids.head(200)
    got = feats_df.fen.reindex(probe.row_id).to_numpy()
    bad = int((got != probe.fen.to_numpy()).sum())
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
        min(len(kids), args.n * 20), random_state=args.seed)

    out_positions = []

    for row in pool.itertuples():
        board = chess.Board(row.fen)
        cands = all_moves(per_move[row.row_id], board)
        if not cands:
            continue

        frow = feats_df[feats_df.row_id == row.row_id]
        if not len(frow):
            continue
        base = frow[feat_names].to_numpy(dtype=np.float32)[0]

        curve = np.asarray(iso(booster.predict(
            rating_grid(base, feat_names, RATINGS))), dtype=float)

        out_positions.append({
            "id": int(row.row_id),
            "fen": row.fen,
            "white": bool(board.turn),
            "moves": cands,
            "curve": [round(float(c), 5) for c in curve],
            "n_moves": int(row.n_moves),
            "best": round(float(row.best_child_win), 2),
            "before": round(float(row.winpct_before), 2),
            "frac_blunder_moves": round(float(row.frac_blunder_moves), 4),
            "_info": position_info(curve),
        })

    if not out_positions:
        print("FATAL: no usable positions. Did eval_children run with --per-move?")
        return 1

    # Keep the most INFORMATIVE positions, not the first n that parsed.
    #
    # The page estimates your rating from how often you blunder, so a position
    # only tells it something if weak and strong players actually behave
    # differently there. Sampled at random, the median position separated a
    # 1000 from a 2000 by about 6 percentage points and a quarter of them by
    # almost nothing, which meant a 10-position run carried so little signal
    # that the estimate sat 600 points above the truth for weak players.
    # Ranking by expected log-likelihood ratio fixes that at no cost: the
    # positions are still real, still varied, just chosen to be worth asking.
    out_positions.sort(key=lambda p: -p["_info"])
    kept = out_positions[:args.n]
    info_all = np.median([p["_info"] for p in out_positions])
    info_kept = np.median([p["_info"] for p in kept])
    print(f"selected {len(kept)} of {len(out_positions)} candidates by "
          f"discriminating power")
    print(f"  median info/position {info_all:.4f} -> {info_kept:.4f} nats")
    lo = np.mean([p["curve"][0] for p in kept])
    hi = np.mean([p["curve"][-1] for p in kept])
    print(f"  mean P(blunder) {lo:.1%} at {RATINGS[0]} vs {hi:.1%} at "
          f"{RATINGS[-1]}")
    out_positions = kept
    for p in out_positions:
        del p["_info"]

    # The rating curve IS the product here -- the page's whole claim is "a 1500
    # blunders here X% of the time". A flattened sweep ships a demo that
    # understates the effect it exists to show, and it did once. Refuse to
    # write the file rather than warn.
    span = check_sweep_span([p["curve"] for p in out_positions],
                            "challenge rating sweep")
    print(f"sweep span check: median {span:.2f}x across {RATINGS[0]}-"
          f"{RATINGS[-1]} Elo")

    payload = {"ratings": RATINGS, "positions": out_positions,
               "model": os.path.basename(args.model),
               "feature_set": meta["feature_set"],
               "sprite": piece_sprite()}
    dest = os.path.join(args.out, "challenge.json")
    with open(dest, "w") as fh:
        json.dump(payload, fh)

    html_path = os.path.join(args.out, "challenge.html")
    with open(html_path, "w") as fh:
        fh.write(HTML)

    sizes = os.path.getsize(dest) / 1e6
    print(f"{len(out_positions)} positions -> {dest} ({sizes:.1f} MB)")
    print(f"wrote {html_path}")
    print(f"mean legal moves/position: "
          f"{np.mean([len(p['moves']) for p in out_positions]):.1f}")
    print("\nopen it locally with:  python -m http.server -d "
          f"{args.out} 8000   then visit http://localhost:8000/challenge.html")
    return 0


HTML = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blunder Challenge</title>
<style>
  :root{--bg:#12141a;--fg:#e8e8ea;--dim:#8b8f9a;--acc:#6ea8fe;--bad:#f2777a;
        --good:#7ec699;--warn:#e6b455;--card:#1b1e26;--line:#262a34;
        --lt:#e9e2d0;--dk:#7d8a9e;--sel:#6ea8fe}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:15px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif}
  .wrap{max-width:940px;margin:0 auto;padding:22px 16px 70px}
  h1{font-size:23px;margin:0 0 3px;letter-spacing:-.01em}
  .sub{color:var(--dim);font-size:13.5px;margin-bottom:16px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;
        padding:16px;margin-bottom:13px}
  .main{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}
  .boardwrap{display:flex;gap:9px}
  .side{flex:1;min-width:262px}
  svg.board{width:min(74vw,392px);height:auto;touch-action:none;
            border-radius:6px;user-select:none}
  .evalcol{display:flex;flex-direction:column;align-items:center;gap:5px}
  .evalbar{width:22px;height:min(74vw,392px);border-radius:5px;overflow:hidden;
           background:#39404f;position:relative;flex:none}
  .evalbar i{position:absolute;left:0;right:0;bottom:0;background:#e9edf5;
             transition:height .45s cubic-bezier(.4,0,.2,1)}
  .evalbar b{position:absolute;left:0;right:0;height:2px;background:var(--acc);
             transition:bottom .45s}
  .evalcap{font-size:10px;color:var(--dim);text-align:center;line-height:1.25;
           width:46px}
  .row{display:flex;justify-content:space-between;align-items:baseline;
       gap:10px;flex-wrap:wrap}
  .dim{color:var(--dim);font-size:13px}
  button{font:600 14px ui-sans-serif,system-ui,sans-serif;background:#232735;
         color:var(--fg);border:1px solid #333949;border-radius:8px;
         padding:9px 14px;cursor:pointer}
  button:hover:not(:disabled){border-color:var(--acc)}
  button:disabled{opacity:.4;cursor:default}
  button.primary{background:var(--acc);color:#0d1117;border-color:var(--acc)}
  .modes{display:flex;gap:8px;flex-wrap:wrap}
  .modes button.on{background:var(--acc);color:#0d1117;border-color:var(--acc)}
  .verdict{margin-top:10px;padding:11px 13px;border-radius:9px;
           background:#1f232d;font-size:14px;min-height:46px}
  .tried{display:flex;flex-wrap:wrap;gap:5px;margin-top:9px}
  .tag{font:600 12px ui-monospace,monospace;padding:3px 8px;border-radius:6px;
       background:#232735;border:1px solid #333949}
  .tag.b{border-color:var(--bad);color:#ffd9da}
  .tag.g{border-color:var(--good);color:#d6f2e2}
  .est{font:800 46px/1 ui-monospace,SFMono-Regular,Menlo,monospace;
       letter-spacing:-.02em}
  .est.pending{color:var(--dim);font-size:30px}
  .k{color:var(--dim);font-size:11px;letter-spacing:.09em;text-transform:uppercase}
  .estrow{display:flex;gap:20px;align-items:flex-end;flex-wrap:wrap}
  svg.post{width:100%;height:64px;display:block;margin-top:8px}
  .prog{height:7px;background:#232735;border-radius:99px;overflow:hidden;
        margin-top:9px}
  .prog i{display:block;height:100%;background:var(--acc);
          transition:width .35s}
  ol{margin:8px 0 12px;padding-left:20px}
  ol li{margin:3px 0;color:#c9cdd6;font-size:14px}
  .big2{font:700 21px ui-monospace,monospace}
  .pill{display:inline-block;font:600 11px ui-sans-serif;padding:2px 8px;
        border-radius:99px;background:#232735;border:1px solid #333949;
        color:var(--dim)}
  .lgl{fill:var(--sel);opacity:.32}
  .cap{fill:none;stroke:var(--sel);stroke-width:3.4;opacity:.55}
  .from{fill:var(--sel);opacity:.30}
  .hide{display:none!important}
</style>
<div class="wrap">
  <h1>Blunder Challenge</h1>
  <div class="sub">Real positions from Lichess blitz games. Play the move you
    would actually play, and find out what it cost.</div>

  <div class="card" id="howto">
    <b>What is going on here</b>
    <ol>
      <li>Every legal move in these positions was scored by Stockfish in
          advance, so the page knows what any move you play is worth. There is
          no engine running in your browser.</li>
      <li>Separately, a model was trained to predict <i>how often humans of a
          given rating blunder</i> in a position. It never saw the answers.</li>
      <li>Because the model knows how hard each position is, your hit rate on a
          handful of them is enough to estimate your rating.</li>
    </ol>
    <button id="hideHow">Got it, let me play</button>
  </div>

  <div class="card">
    <div class="k" style="margin-bottom:7px">Pick a mode</div>
    <div class="modes" id="modes">
      <button data-n="10">Quick run &middot; 10 positions</button>
      <button data-n="25">Full run &middot; 25 positions</button>
      <button data-n="0">Endless practice</button>
    </div>
    <div class="dim" id="modeNote" style="margin-top:8px">
      A run gives you a rating estimate at the end. Practice never ends and
      keeps updating the estimate as you go.</div>
  </div>

  <div class="card" id="estCard">
    <div class="estrow">
      <div>
        <div class="k">Your estimated rating</div>
        <div class="est pending" id="est">not enough data yet</div>
      </div>
      <div style="flex:1;min-width:180px">
        <div class="dim" id="estRange">Play a few positions and this fills in.</div>
        <div class="dim" id="estNote" style="margin-top:3px"></div>
      </div>
    </div>
    <svg class="post" id="post" viewBox="0 0 600 64" preserveAspectRatio="none"></svg>
    <div class="row">
      <span class="dim" id="progText">Nothing played yet</span>
      <span class="dim"><b id="score">0</b> clean of <b id="seen">0</b></span>
    </div>
    <div class="prog"><i id="progBar" style="width:0%"></i></div>
  </div>

  <div class="card main">
    <div class="boardwrap">
      <div class="evalcol">
        <div class="evalcap" id="evalTop">their<br>side</div>
        <div class="evalbar" id="evalbar"><i id="ev"></i><b id="evRef"></b></div>
        <div class="evalcap" id="evalBot">your<br>side</div>
      </div>
      <svg class="board" id="board" viewBox="0 0 360 360"></svg>
    </div>
    <div class="side">
      <div class="row">
        <div><span class="pill" id="toMove">White to move</span></div>
        <div class="dim" id="posMeta"></div>
      </div>
      <div class="verdict" id="verdict">Click one of your pieces, then where you
        want it to go.</div>
      <div class="tried" id="tried"></div>
      <div class="row" style="margin-top:12px">
        <button id="undo" disabled>Try another move</button>
        <button id="reveal">Show the best move</button>
        <button id="next" class="primary">Next position</button>
      </div>
      <div class="dim" style="margin-top:9px" id="hint">
        The bar on the left is the engine's evaluation: how much of the board it
        fills is your chance of winning. The blue line is where you started.
      </div>
    </div>
  </div>

  <div class="card hide" id="resultCard">
    <div class="k">Run complete</div>
    <div class="estrow" style="margin:6px 0 10px">
      <div><div class="est" id="finalEst">-</div></div>
      <div style="flex:1;min-width:200px">
        <div id="finalRange" class="dim"></div>
        <div id="finalBlurb" class="dim" style="margin-top:5px"></div>
      </div>
    </div>
    <div id="finalBreak" class="dim"></div>
    <div style="margin-top:12px">
      <button class="primary" id="again">Play again</button>
      <button id="keepGoing">Keep practising</button>
    </div>
  </div>
</div>
<script>
const $=i=>document.getElementById(i);
let D=null,order=[],idx=0;
let pos=null,brd=null,flip=false,sel=null,scored=false,tried=[];
let runN=10,played=[],finished=false;

fetch('challenge.json').then(r=>r.json()).then(d=>{
  D=d;
  const sp=document.createElementNS('http://www.w3.org/2000/svg','svg');
  sp.setAttribute('style','position:absolute;width:0;height:0');
  sp.innerHTML='<defs>'+d.sprite+'</defs>';
  document.body.appendChild(sp);
  order=d.positions.map((_,i)=>i);
  shuffle(order);
  setMode(10);
});
function shuffle(a){for(let i=a.length-1;i>0;i--){const j=(Math.random()*(i+1))|0;
  [a[i],a[j]]=[a[j],a[i]];}}

/* ---------- FEN ---------- */
function parseFEN(f){
  const b=Array(64).fill(null), rows=f.split(' ')[0].split('/');
  for(let r=0;r<8;r++){let c=0;
    for(const ch of rows[r]){
      if(/\d/.test(ch)) c+=+ch;
      else b[(7-r)*8+c++]=ch;
    }}
  return b;
}
const NAME={k:'king',q:'queen',r:'rook',b:'bishop',n:'knight',p:'pawn'};
const href=p=>'#'+(p===p.toUpperCase()?'white':'black')+'-'+NAME[p.toLowerCase()];

/* ---------- board ---------- */
function draw(){
  const S=45,sq=[];
  for(let i=0;i<64;i++){
    const file=i%8, rank=(i/8)|0;
    const x=(flip?7-file:file)*S, y=(flip?rank:7-rank)*S;
    const light=(file+rank)%2===1;
    sq.push(`<rect x="${x}" y="${y}" width="${S}" height="${S}" fill="${light?'var(--lt)':'var(--dk)'}"/>`);
  }
  let mk='',pc='';
  if(sel!==null){
    const [x,y]=xy(sel,S);
    mk+=`<rect class="from" x="${x}" y="${y}" width="${S}" height="${S}"/>`;
    for(const m of legalFrom(sel)){
      const t=sqIdx(m.u.slice(2,4)), [tx,ty]=xy(t,S);
      mk+= brd[t]
        ? `<circle class="cap" cx="${tx+S/2}" cy="${ty+S/2}" r="${S/2-3}"/>`
        : `<circle class="lgl" cx="${tx+S/2}" cy="${ty+S/2}" r="7"/>`;
    }
  }
  for(let i=0;i<64;i++){
    if(!brd[i]) continue;
    const [x,y]=xy(i,S);
    pc+=`<g transform="translate(${x},${y}) scale(${S/45})" data-sq="${i}"
          style="cursor:${scored?'default':'grab'}"><use href="${href(brd[i])}"/></g>`;
  }
  $('board').innerHTML=sq.join('')+mk+pc;
}
function xy(i,S){const f=i%8,r=(i/8)|0;return [(flip?7-f:f)*S,(flip?r:7-r)*S];}
function sqIdx(s){return (s.charCodeAt(1)-49)*8+(s.charCodeAt(0)-97);}
function sqName(i){return String.fromCharCode(97+i%8)+String.fromCharCode(49+((i/8)|0));}
function legalFrom(i){const n=sqName(i);return pos.moves.filter(m=>m.u.slice(0,2)===n);}

/* ---------- interaction ---------- */
function pick(i){
  if(scored||finished) return;
  if(sel===null){ if(legalFrom(i).length){sel=i;draw();} return; }
  if(i===sel){ sel=null; draw(); return; }
  const cand=legalFrom(sel).filter(m=>m.u.slice(2,4)===sqName(i));
  if(!cand.length){ sel=legalFrom(i).length?i:null; draw(); return; }
  const mv=cand.find(m=>m.u.length===5&&m.u[4]==='q')||cand[0];
  sel=null; play(mv);
}
$('board').addEventListener('pointerdown',e=>{
  const pt=$('board').createSVGPoint(); pt.x=e.clientX; pt.y=e.clientY;
  const p=pt.matrixTransform($('board').getScreenCTM().inverse());
  let f=Math.floor(p.x/45), r=7-Math.floor(p.y/45);
  if(flip){f=7-f;r=7-r;}
  if(f<0||f>7||r<0||r>7) return;
  pick(r*8+f);
  e.preventDefault();
});

/* ---------- applying a move ---------- */
function apply(u){
  const from=sqIdx(u.slice(0,2)), to=sqIdx(u.slice(2,4)), p=brd[from];
  const low=p.toLowerCase();
  if(low==='p'&&(from%8)!==(to%8)&&!brd[to]) brd[to+(p==='P'?-8:8)]=null; // en passant
  if(low==='k'&&Math.abs(to-from)===2){                                   // castling
    const rk=to>from?from+3:from-4, rt=to>from?to-1:to+1;
    brd[rt]=brd[rk]; brd[rk]=null;
  }
  brd[to]=u.length===5 ? (p===p.toUpperCase()?u[4].toUpperCase():u[4]) : p;
  brd[from]=null;
}

/* ---------- eval bar ----------
   The board flips so the side to move is always at the bottom, and the bar is
   stacked against it, so the fill grows from the bottom with the MOVER's win%.
   Every number in the data is already from the mover's point of view, so there
   is no conversion here and nowhere to invert a sign. */
function setBar(win,ref){
  $('ev').style.height=Math.max(0,Math.min(100,win)).toFixed(1)+'%';
  $('evRef').style.bottom=Math.max(0,Math.min(100,ref)).toFixed(1)+'%';
}

/* ---------- rating estimate ----------
   A posterior over the rating grid rather than a nearest-match lookup. For each
   rating r the model gives p_i(r), the chance a player of that rating blunders
   in position i, so the likelihood of what you actually did is the product of
   p_i(r) for the ones you blundered and (1 - p_i(r)) for the ones you did not.
   A flat prior over the grid turns that into a posterior, which gives both a
   best estimate and an honest interval that visibly narrows as you play. */
function posterior(){
  if(!played.length) return null;
  const n=D.ratings.length, log=new Array(n).fill(0);
  for(const rec of played){
    const c=D.positions[rec.p].curve;
    for(let k=0;k<n;k++){
      const p=Math.min(Math.max(c[k],1e-6),1-1e-6);
      log[k]+= rec.blundered ? Math.log(p) : Math.log(1-p);
    }
  }
  const mx=Math.max(...log);
  const w=log.map(v=>Math.exp(v-mx));
  const s=w.reduce((a,b)=>a+b,0);
  const post=w.map(v=>v/s);
  let best=0; post.forEach((v,k)=>{ if(v>post[best]) best=k; });
  // central 80% interval from the cumulative posterior
  let cum=0, lo=0, hi=n-1;
  for(let k=0;k<n;k++){ cum+=post[k];
    if(cum>=0.10){ lo=k; break; } }
  cum=0;
  for(let k=0;k<n;k++){ cum+=post[k];
    if(cum>=0.90){ hi=k; break; } }
  return {post, best:D.ratings[best], lo:D.ratings[lo], hi:D.ratings[hi]};
}
function drawPosterior(pt){
  const W=600,H=64,n=D.ratings.length;
  if(!pt){ $('post').innerHTML=''; return; }
  const mx=Math.max(...pt.post);
  const bars=pt.post.map((v,k)=>{
    const w=W/n, x=k*w, h=Math.max(1,(v/mx)*(H-14));
    const inside=D.ratings[k]>=pt.lo&&D.ratings[k]<=pt.hi;
    return `<rect x="${x+0.6}" y="${H-h}" width="${w-1.2}" height="${h}"
             fill="${inside?'#6ea8fe':'#39404f'}" rx="1"/>`;
  }).join('');
  const bi=D.ratings.indexOf(pt.best), bw=W/n;
  $('post').innerHTML=bars+
    `<text x="4" y="11" fill="#8b8f9a" font-size="9">${D.ratings[0]}</text>`+
    `<text x="${W-4}" y="11" fill="#8b8f9a" font-size="9" text-anchor="end">${D.ratings[n-1]}</text>`+
    `<line x1="${bi*bw+bw/2}" y1="0" x2="${bi*bw+bw/2}" y2="${H}"
      stroke="#fff" stroke-width="1.4" opacity=".8"/>`;
}
function refreshEstimate(){
  const pt=posterior();
  drawPosterior(pt);
  const done=played.length, clean=played.filter(r=>!r.blundered).length;
  $('score').textContent=clean; $('seen').textContent=done;
  if(runN){
    $('progText').textContent=`Position ${Math.min(done+ (finished?0:1),runN)} of ${runN}`;
    $('progBar').style.width=(100*done/runN).toFixed(0)+'%';
  }else{
    $('progText').textContent=`${done} played, practice mode`;
    $('progBar').style.width=(100*Math.min(done/20,1)).toFixed(0)+'%';
  }
  if(!pt||done<3){
    $('est').textContent='not enough data yet';
    $('est').classList.add('pending');
    $('estRange').textContent=`Play ${Math.max(0,3-done)} more position${3-done===1?'':'s'} for a first estimate.`;
    $('estNote').textContent='';
    return;
  }
  $('est').classList.remove('pending');
  $('est').textContent=pt.best;
  const width=pt.hi-pt.lo;
  $('estRange').innerHTML=`Probably between <b>${pt.lo}</b> and <b>${pt.hi}</b>.`;
  $('estNote').textContent = width>700
    ? 'Very rough so far. The range narrows quickly as you play more.'
    : width>350
      ? 'Getting there. A few more positions will tighten this.'
      : 'Reasonably settled now.';
}

/* ---------- a turn ---------- */
function play(mv){
  scored=true; tried.push(mv);
  apply(mv.u); draw();
  if(tried.length===1){
    played.push({p:order[idx], blundered:!!mv.b});
  }
  setBar(mv.w,pos.before);
  const best=pos.moves.reduce((a,b)=>b.w>a.w?b:a);
  const pred=curAt();
  $('verdict').innerHTML =
    (mv.b?`<b style="color:var(--bad)">That one loses a lot.</b> `
        :`<b style="color:var(--good)">Fine move.</b> `)
    + `<b>${mv.s}</b> leaves you with a ${mv.w.toFixed(0)}% chance of winning`
    + (mv.d<0.05?`, the best available.`
              :`, which is ${mv.d.toFixed(0)} points worse than <b>${best.s}</b> (${best.w.toFixed(0)}%).`)
    + `<br><span class="dim">Players around ${pt_label()} blunder here about
       <b>${(pred*100).toFixed(0)}%</b> of the time.
       ${Math.round(pos.frac_blunder_moves*100)}% of the ${pos.n_moves} legal
       moves lose 20 points or more.</span>`;
  paintTried();
  $('undo').disabled=false;
  refreshEstimate();
  if(runN && played.length>=runN) finishRun();
}
function pt_label(){
  const pt=posterior();
  return pt&&played.length>=3 ? pt.best : 'your level';
}
function curAt(){
  const pt=posterior();
  const r = pt&&played.length>=3 ? pt.best : 1500;
  const i=D.ratings.indexOf(r);
  return pos.curve[i<0?Math.floor(D.ratings.length/2):i];
}
function paintTried(){
  $('tried').innerHTML=tried.map(m=>
    `<span class="tag ${m.b?'b':'g'}">${m.s} ${m.d<0.05?'best':'-'+m.d.toFixed(0)}</span>`).join('');
}
$('undo').onclick=()=>{
  brd=parseFEN(pos.fen); scored=false; sel=null; draw();
  setBar(pos.before,pos.before);
  $('verdict').innerHTML='Try a different move. <span class="dim">Only your '
    +'first attempt counted towards the estimate.</span>';
  $('undo').disabled=true;
};
$('reveal').onclick=()=>{
  const best=pos.moves.reduce((a,b)=>b.w>a.w?b:a);
  const worst=pos.moves.reduce((a,b)=>b.w<a.w?b:a);
  $('verdict').innerHTML=`Best: <b>${best.s}</b> (${best.w.toFixed(0)}% win). `
    +`Worst: <b>${worst.s}</b> (${worst.w.toFixed(0)}%). `
    +`<span class="dim">${pos.moves.filter(m=>m.b).length} of ${pos.moves.length} `
    +`legal moves throw away 20 points or more.</span>`;
};
$('next').onclick=()=>{ if(finished) return; idx=(idx+1)%order.length; render(); };
$('hideHow').onclick=()=>$('howto').classList.add('hide');
$('again').onclick=()=>setMode(runN||10);
$('keepGoing').onclick=()=>{ runN=0; finished=false;
  $('resultCard').classList.add('hide'); markMode(); refreshEstimate();
  idx=(idx+1)%order.length; render(); };
document.querySelectorAll('#modes button').forEach(b=>{
  b.onclick=()=>setMode(+b.dataset.n);
});

function markMode(){
  document.querySelectorAll('#modes button').forEach(b=>
    b.classList.toggle('on', +b.dataset.n===runN));
}
function setMode(n){
  runN=n; played=[]; finished=false; idx=0;
  shuffle(order);
  $('resultCard').classList.add('hide');
  markMode(); render(); refreshEstimate();
}

function render(){
  pos=D.positions[order[idx]];
  brd=parseFEN(pos.fen); flip=!pos.white; sel=null; scored=false; tried=[];
  draw(); setBar(pos.before,pos.before);
  $('toMove').textContent=(pos.white?'White':'Black')+' to move - that is you';
  $('posMeta').textContent=pos.n_moves+' legal moves';
  $('verdict').textContent='Click one of your pieces, then where you want it to go.';
  $('tried').innerHTML=''; $('undo').disabled=true;
  $('evalTop').innerHTML='their<br>side';
  $('evalBot').innerHTML='your<br>side';
}

function finishRun(){
  finished=true;
  const pt=posterior();
  const clean=played.filter(r=>!r.blundered).length;
  $('finalEst').textContent=pt?pt.best:'-';
  $('finalRange').innerHTML=pt
    ? `Most likely between <b>${pt.lo}</b> and <b>${pt.hi}</b>, based on
       ${played.length} positions.` : '';
  $('finalBreak').innerHTML=`You played ${clean} clean moves out of
    ${played.length}. `+ (played.length-clean===0
      ? 'No blunders at all, so the estimate is a lower bound: play a longer run to pin it down.'
      : `You blundered ${played.length-clean} time${played.length-clean===1?'':'s'}.`);
  $('finalBlurb').textContent = pt && (pt.hi-pt.lo)>600
    ? 'That is still a wide range. A 25 position run gives a much tighter answer.'
    : 'A longer run would tighten this further.';
  $('resultCard').classList.remove('hide');
  $('resultCard').scrollIntoView({behavior:'smooth',block:'center'});
}
</script>
"""

if __name__ == "__main__":
    raise SystemExit(main())
