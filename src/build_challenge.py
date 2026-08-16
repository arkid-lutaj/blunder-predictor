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
        min(len(kids), args.n * 3), random_state=args.seed)

    out_positions = []

    for row in pool.itertuples():
        if len(out_positions) >= args.n:
            break
        board = chess.Board(row.fen)
        cands = all_moves(per_move[row.row_id], board)
        if not cands:
            continue

        frow = feats_df[feats_df.row_id == row.row_id]
        if not len(frow):
            continue
        base = frow[feat_names].to_numpy(dtype=np.float32)[0]

        curve = iso(booster.predict(rating_grid(base, feat_names, RATINGS)))

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
        })

    if not out_positions:
        print("FATAL: no usable positions. Did eval_children run with --per-move?")
        return 1

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
        --good:#7ec699;--card:#1b1e26;--lt:#e9e2d0;--dk:#7d8a9e;--sel:#6ea8fe}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:15px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif}
  .wrap{max-width:900px;margin:0 auto;padding:22px 16px 70px}
  h1{font-size:21px;margin:0 0 3px;letter-spacing:-.01em}
  .sub{color:var(--dim);font-size:13px;margin-bottom:18px}
  .card{background:var(--card);border:1px solid #262a34;border-radius:12px;
        padding:16px;margin-bottom:14px}
  .main{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}
  .boardwrap{display:flex;gap:10px}
  .side{flex:1;min-width:250px}
  svg.board{width:min(74vw,392px);height:auto;touch-action:none;
            border-radius:6px;user-select:none}
  .evalbar{width:20px;height:min(74vw,392px);border-radius:5px;overflow:hidden;
           background:#2b2f3a;position:relative;flex:none}
  .evalbar i{position:absolute;left:0;right:0;bottom:0;background:#eee;
             transition:height .45s cubic-bezier(.4,0,.2,1)}
  .evalbar b{position:absolute;left:0;right:0;height:1px;background:var(--acc);
             opacity:.85;transition:bottom .45s}
  .row{display:flex;justify-content:space-between;align-items:baseline;
       gap:10px;flex-wrap:wrap}
  .num{font:600 26px ui-monospace,SFMono-Regular,Menlo,monospace}
  .dim{color:var(--dim);font-size:13px}
  input[type=range]{width:100%;accent-color:var(--acc)}
  button{font:600 14px ui-sans-serif,system-ui,sans-serif;background:#232735;
         color:var(--fg);border:1px solid #333949;border-radius:8px;
         padding:9px 14px;cursor:pointer}
  button:hover:not(:disabled){border-color:var(--acc)}
  button:disabled{opacity:.4;cursor:default}
  .verdict{margin-top:12px;padding:11px 13px;border-radius:8px;
           background:#1f232d;font-size:14px;min-height:44px}
  .tried{display:flex;flex-wrap:wrap;gap:5px;margin-top:9px}
  .tag{font:600 12px ui-monospace,monospace;padding:3px 8px;border-radius:6px;
       background:#232735;border:1px solid #333949}
  .tag.b{border-color:var(--bad);color:#ffd9da}
  .tag.g{border-color:var(--good);color:#d6f2e2}
  svg.curve{width:100%;height:88px}
  .ctr{text-align:center;margin-top:6px}
  .lgl{fill:var(--sel);opacity:.32}
  .cap{fill:none;stroke:var(--sel);stroke-width:3.4;opacity:.55}
  .from{fill:var(--sel);opacity:.30}
</style>
<div class="wrap">
  <h1>Blunder Challenge</h1>
  <div class="sub">Every legal move in these positions was evaluated by
    Stockfish, so the page knows the truth about whatever you play. The model
    never saw the answer. It only predicts how often a player of a given rating
    blunders here. Click a piece, then its destination. Promotions auto-queen.</div>

  <div class="card">
    <label class="dim">Your rating: <b id="eloLabel">1500</b></label>
    <input type="range" id="elo" min="800" max="2450" step="50" value="1500">
    <div class="row" style="margin-top:8px">
      <div><span class="num" id="score">0</span><span class="dim"> / <span id="seen">0</span> clean</span></div>
      <div class="dim">model expected <b id="expected">0.0</b> blunders</div>
      <div class="dim">you played like <b id="implied">&mdash;</b></div>
    </div>
  </div>

  <div class="card main">
    <div class="boardwrap">
      <div class="evalbar" title="win% for the side to move, who is always at the bottom of the board. The line marks what the position was worth before you moved."><i id="ev"></i><b id="evRef"></b></div>
      <svg class="board" id="board" viewBox="0 0 360 360"></svg>
    </div>
    <div class="side">
      <div class="row">
        <div class="dim"><b id="toMove">White</b> to move</div>
        <div class="dim" id="posMeta"></div>
      </div>
      <div class="verdict" id="verdict">Pick a move.</div>
      <div class="tried" id="tried"></div>
      <div class="row" style="margin-top:12px">
        <button id="undo" disabled>&larr; Take back</button>
        <button id="reveal">Show best</button>
        <button id="next">Next &rarr;</button>
      </div>
      <div id="curveBox" style="display:none;margin-top:12px">
        <div class="dim">predicted blunder rate by rating</div>
        <svg class="curve" id="curve" viewBox="0 0 600 88" preserveAspectRatio="none"></svg>
      </div>
    </div>
  </div>
</div>
<script>
const $=i=>document.getElementById(i);
let D=null,order=[],idx=0,seen=0,clean=0,expSum=0;
let pos=null,brd=null,flip=false,sel=null,scored=false,tried=[],shownPositions=[];

fetch('challenge.json').then(r=>r.json()).then(d=>{
  D=d;
  const sp=document.createElementNS('http://www.w3.org/2000/svg','svg');
  sp.setAttribute('style','position:absolute;width:0;height:0');
  sp.innerHTML='<defs>'+d.sprite+'</defs>';
  document.body.appendChild(sp);
  order=d.positions.map((_,i)=>i);
  for(let i=order.length-1;i>0;i--){const j=(Math.random()*(i+1))|0;
    [order[i],order[j]]=[order[j],order[i]];}
  render();
});

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
  if(scored) return;
  if(sel===null){ if(legalFrom(i).length){sel=i;draw();} return; }
  if(i===sel){ sel=null; draw(); return; }
  const cand=legalFrom(sel).filter(m=>m.u.slice(2,4)===sqName(i));
  if(!cand.length){ sel=legalFrom(i).length?i:null; draw(); return; }
  // promotion: several moves share from/to. Queen unless it is the only option.
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

/* ---------- eval bar ---------- */
function setBar(win,ref){
  /* The board auto-flips so the side to move is always at the BOTTOM, and the
     bar is stacked against it, so the fill grows from the bottom with the
     MOVER's win%. That is also the perspective every number in the data uses
     (eval_children scored each child from the mover's point of view), so no
     conversion is needed and there is nowhere to invert a sign.

     An earlier version converted to White's perspective and then drew
     (100 - w) from the bottom, which made the light fill grow as the side to
     move got WORSE -- backwards under any eval-bar convention. */
  $('ev').style.height=Math.max(0,Math.min(100,win)).toFixed(1)+'%';
  $('evRef').style.bottom=Math.max(0,Math.min(100,ref)).toFixed(1)+'%';
}

/* ---------- a turn ---------- */
function play(mv){
  scored=true; tried.push(mv);
  apply(mv.u); draw();
  const pred=curAt();
  if(tried.length===1){ seen++; expSum+=pred; if(!mv.b) clean++;
    shownPositions.push(pos); }
  setBar(mv.w,pos.best);
  const best=pos.moves.reduce((a,b)=>b.w>a.w?b:a);
  $('verdict').innerHTML =
    (mv.b?`<b style="color:var(--bad)">Blunder.</b> `:`<b style="color:var(--good)">Fine.</b> `)
    + `<b>${mv.s}</b> leaves ${mv.w.toFixed(1)}% win, `
    + (mv.d<0.05?`the best there is.`:`${mv.d.toFixed(1)} points below <b>${best.s}</b> (${best.w.toFixed(1)}%).`)
    + `<br><span class="dim">The model predicted a ${$('elo').value} blunders here
       <b>${(pred*100).toFixed(1)}%</b> of the time.
       ${(pos.frac_blunder_moves*100).toFixed(0)}% of the ${pos.n_moves} legal moves lose 20%+.</span>`;
  paintTried();
  $('undo').disabled=false; $('curveBox').style.display='block'; drawCurve();
  $('score').textContent=clean; $('seen').textContent=seen;
  $('expected').textContent=expSum.toFixed(1);
  $('implied').textContent=implied();
}
function paintTried(){
  $('tried').innerHTML=tried.map(m=>
    `<span class="tag ${m.b?'b':'g'}">${m.s} ${m.d<0.05?'best':'-'+m.d.toFixed(0)}</span>`).join('');
}
$('undo').onclick=()=>{
  brd=parseFEN(pos.fen); scored=false; sel=null; draw();
  setBar(pos.before,pos.best);
  $('verdict').innerHTML='Try another move. <span class="dim">Only your first '
    +'attempt counted towards the score.</span>';
  $('undo').disabled=true;
};
$('reveal').onclick=()=>{
  const best=pos.moves.reduce((a,b)=>b.w>a.w?b:a);
  const worst=pos.moves.reduce((a,b)=>b.w<a.w?b:a);
  $('verdict').innerHTML=`Best: <b>${best.s}</b> (${best.w.toFixed(1)}%). `
    +`Worst: <b>${worst.s}</b> (${worst.w.toFixed(1)}%, -${worst.d.toFixed(1)}). `
    +`<span class="dim">${pos.moves.filter(m=>m.b).length} of ${pos.moves.length} `
    +`legal moves lose 20%+.</span>`;
};
$('next').onclick=()=>{ idx=(idx+1)%order.length; render(); };
$('elo').oninput=()=>{ $('eloLabel').textContent=$('elo').value;
  if(D){ $('implied').textContent=implied(); if(scored) drawCurve(); } };

function render(){
  pos=D.positions[order[idx]];
  brd=parseFEN(pos.fen); flip=!pos.white; sel=null; scored=false; tried=[];
  draw(); setBar(pos.before,pos.best);
  $('toMove').textContent=pos.white?'White':'Black';
  $('posMeta').textContent=pos.n_moves+' legal moves';
  $('verdict').textContent='Pick a move.';
  $('tried').innerHTML=''; $('undo').disabled=true;
  $('curveBox').style.display='none';
}

/* ---------- model readouts ---------- */
function ri(){return D.ratings.indexOf(+$('elo').value);}
function curAt(){const i=ri();return pos.curve[i<0?0:i];}
function implied(){
  if(seen<5) return `\u2014 (need ${5-seen} more)`;
  const obs=(seen-clean)/seen; let best=0,bd=1e9;
  D.ratings.forEach((r,k)=>{
    const m=shownPositions.reduce((s,p)=>s+p.curve[k],0)/shownPositions.length;
    if(Math.abs(m-obs)<bd){bd=Math.abs(m-obs);best=r;}
  });
  return best+(bd>0.15?'?':'');
}
function drawCurve(){
  const W=600,H=88,pad=6,n=pos.curve.length,mx=Math.max(...pos.curve,0.01);
  const pts=pos.curve.map((v,i)=>[pad+i*(W-2*pad)/(n-1),H-pad-(v/mx)*(H-2*pad)]);
  const d=pts.map((q,i)=>(i?'L':'M')+q[0].toFixed(1)+' '+q[1].toFixed(1)).join(' ');
  const i=ri(),cx=pad+(i<0?0:i)*(W-2*pad)/(n-1);
  $('curve').innerHTML=`<path d="${d}" fill="none" stroke="#6ea8fe" stroke-width="2"/>`
    +`<line x1="${cx}" y1="0" x2="${cx}" y2="${H}" stroke="#8b8f9a" stroke-dasharray="3 3"/>`;
}
</script>
"""

if __name__ == "__main__":
    raise SystemExit(main())
