#!/usr/bin/env python3
"""
Assign games to train / val / test with as little player overlap as possible,
without throwing data away and without letting the splits drift apart in
rating.

Why not the obvious thing. Hashing each player independently into one of K
folds and keeping only games where both players match retains 1/K of your
games. At K=10 that is 90% of the dataset destroyed. Do not do it.

What this does instead. Players are nodes, games are edges. Take connected
components and assign whole components to folds: every game is kept, and no
player can appear on both sides because a player lives in exactly one
component.

That works while the graph is sparse. Once it percolates it stops working, and
on the full month it did: one component holds 22.0% of games, so it cannot fit
in val (10%) or test (20%) and is forced into train. Worse, it is not a random
22% -- it is built from the most active players, whose median Elo is 2087
against 1589 for the dataset as a whole. Assigning it atomically dragged train's
median Elo 222 points above test's and the base rate 0.64pp apart.

So components that are too big to place atomically are CARVED. The carve is not
a dissolve: the component's PLAYER graph is partitioned directly, by growing
each fold from many random seeds until it holds its share of the games. The
giant component is nearly a tree (22,095 edges over 20,174 players, only 1,922
more edges than a spanning tree), so a balanced 3-way cut costs very few edges.
Only games whose two players land on opposite sides of the cut can leak a
player, and those are counted and reported rather than hidden.

Usage:
    python src/make_splits.py --data data/features_blitz_full.parquet \
        --out data/splits_blitz_full.parquet
"""

import argparse
import collections
import hashlib

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


class DSU:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def stable_fold(key: str, n: int) -> int:
    """Deterministic across runs, machines and Python versions (hash() is not)."""
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % n


def build_components(game_players: pd.Series) -> pd.DataFrame:
    """game_players: index game_id, value list of player names in that game."""
    dsu = DSU()
    for players in game_players:
        first = players[0]
        for other in players[1:]:
            dsu.union(first, other)
        dsu.find(first)

    comp_of_game = {g: dsu.find(ps[0]) for g, ps in game_players.items()}
    df = pd.DataFrame({
        "game_id": list(comp_of_game),
        "component": list(comp_of_game.values()),
    })
    sizes = df.groupby("component").size().rename("component_games")
    return df.join(sizes, on="component")


# ---------------------------------------------------------------------------
# Carving an oversized component
# ---------------------------------------------------------------------------


def stratified_seeds(pool: list[str], elo: dict[str, float], n: int,
                     rng: np.random.Generator) -> list[str]:
    """Pick n starting players spread evenly across the rating range.

    Players are matched against similar ratings, so a blob grown from a
    uniformly-random seed is a rating-homogeneous neighbourhood. Taking the
    seeds at even quantiles of Elo instead makes each fold's blobs span the
    whole range for free -- same seed count, same cut, same leakage, steadier
    balance. Measured over --seed 0..3 on the full month: test-train median Elo
    gap went from +5/+20/+21/+31 to +22/+15/+17/+24, i.e. the worst case
    improved and the spread halved, which is what you want from a knob you are
    going to leave alone.
    """
    if n >= len(pool):
        return list(pool)
    ranked = sorted(pool, key=lambda p: (elo.get(p, 0.0), p))
    step = len(ranked) / n
    off = rng.random()
    idx = sorted({min(len(ranked) - 1, int((i + off) * step)) for i in range(n)})
    picked = [ranked[i] for i in idx]
    rest = [p for p in ranked if p not in set(picked)]
    rng.shuffle(rest)
    return picked + rest


def grow_partition(players: list[str],
                   game_players: dict[str, tuple],
                   incidence: dict[str, list[str]],
                   elo: dict[str, float],
                   targets: dict[str, int],
                   n_seeds: int,
                   rng: np.random.Generator) -> dict[str, str]:
    """Partition PLAYERS of one component so each fold holds its game target.

    Multi-seed graph growing. Folds are grown smallest-target first from
    `n_seeds` random starting players each, breadth-first, until the fold owns
    its quota of games; whatever is left goes to the largest fold.

    The seed count is the whole trade-off. One seed grows a single contiguous
    blob, which gives the smallest possible cut but samples one neighbourhood
    of the graph -- and since players are matched against similar ratings, one
    neighbourhood is not a representative rating mix. Many seeds sample the
    component all over and keep the folds comparable, at the cost of more
    boundary. Both quantities are measured and printed, so the default is
    tunable against evidence rather than taste.

    A game counts towards a fold only when EVERY player in it is in that fold,
    which is what makes the count a count of clean, non-leaking games.
    """
    fold_of: dict[str, str] = {}
    claimed = {k: 0 for k in targets}

    def claim(u: str, f: str) -> None:
        """Assign u, then bank any game of u's that is now wholly inside f."""
        fold_of[u] = f
        for g in incidence[u]:
            if all(fold_of.get(x) == f for x in game_players[g]):
                claimed[f] += 1

    # Grow the small folds; the biggest one inherits the remainder, so we only
    # ever grow a minority of the component and the boundary stays short.
    order = sorted(targets, key=lambda k: targets[k])
    for f in order[:-1]:
        need = targets[f]
        if need <= 0:
            continue
        pool = [p for p in players if p not in fold_of]
        if not pool:
            break
        pool = stratified_seeds(pool, elo, n_seeds, rng)
        frontier = collections.deque(pool[:n_seeds])
        spare = iter(pool[n_seeds:])
        while claimed[f] < need:
            if not frontier:
                nxt = next((p for p in spare if p not in fold_of), None)
                if nxt is None:
                    break
                frontier.append(nxt)
                continue
            u = frontier.popleft()
            if u in fold_of:
                continue
            claim(u, f)
            for g in incidence[u]:
                for v in game_players[g]:
                    if v not in fold_of:
                        frontier.append(v)

    last = order[-1]
    for p in players:
        fold_of.setdefault(p, last)
    return fold_of


def carve(component_games: list[str],
          game_players: dict[str, tuple],
          elo: dict[str, float],
          targets: dict[str, int],
          n_seeds: int,
          rng: np.random.Generator) -> tuple[dict[str, str], list[str]]:
    """Split one oversized component. Returns (game -> fold, cut game ids).

    `cut` games are those whose players ended up in different folds. They are
    not assigned here: the caller places them against the global quota so they
    help balance rather than distort it.
    """
    incidence: dict[str, list[str]] = collections.defaultdict(list)
    for g in component_games:
        for p in game_players[g]:
            incidence[p].append(g)
    players = sorted(incidence)

    fold_of = grow_partition(players, game_players, incidence, elo, targets,
                             n_seeds, rng)

    assigned, cut = {}, []
    for g in component_games:
        folds = {fold_of[p] for p in game_players[g]}
        if len(folds) == 1:
            assigned[g] = folds.pop()
        else:
            cut.append(g)
    return assigned, cut


# ---------------------------------------------------------------------------
# Top-level assignment
# ---------------------------------------------------------------------------


def assign(components: pd.DataFrame,
           game_players: dict[str, tuple],
           elo: dict[str, float],
           targets: dict[str, float],
           max_component_frac: float,
           n_seeds: int,
           seed: int):
    """Assign every game to a split. Returns (series, stats dict).

    Small components are placed atomically in a deterministic random order,
    each going to whichever fold is furthest below quota. Visiting in random
    rather than size order matters: component size tracks how many games a
    player plays, which tracks rating, so largest-first quietly made train
    ~114 Elo stronger than test the first time this was written.
    """
    sizes = components.groupby("component")["component_games"].first()
    total = int(sizes.sum())
    carved_set = set(sizes[sizes > max_component_frac * total].index)
    carved_ids = sorted(carved_set)

    quota = {k: v * total for k, v in targets.items()}
    filled = {k: 0.0 for k in SPLITS}
    where: dict[str, str] = {}

    # Fill toward quotas scaled to the ATOMIC games only. Greedy max-deficit
    # placement equalises deficits in ABSOLUTE terms, so aiming it at the
    # global quota and then stopping early leaves every fold short by the same
    # number of games rather than by its share. On the full month that handed
    # 73% of val to the carved component against 10% of train, and since the
    # carved component is the high-Elo one it pushed val's median Elo 280
    # points above train's -- the same disease as before, in the other
    # direction. Scaling the quota first makes the leftover proportional.
    atomic_total = total - int(sizes[list(carved_set)].sum()) if carved_set else total
    quota_atomic = {k: v * atomic_total for k, v in targets.items()}

    for comp in sorted((c for c in sizes.index if c not in carved_set),
                       key=lambda c: hashlib.md5(str(c).encode()).hexdigest()):
        pick = max(quota_atomic, key=lambda k: quota_atomic[k] - filled[k])
        where[comp] = pick
        filled[pick] += sizes[comp]

    assignment = components["component"].map(where)
    stats = {"carved": carved_ids, "carved_games": 0, "cut_games": 0,
             "atomic_games": int(sum(filled.values()))}

    if carved_ids:
        rng = np.random.default_rng(seed)
        by_comp = components.groupby("component")["game_id"].apply(list)
        placed: dict[str, str] = {}
        all_cut: list[str] = []

        for comp in carved_ids:
            games = sorted(by_comp[comp])
            stats["carved_games"] += len(games)
            # Aim this component at the quota still OUTSTANDING, so the carve
            # repairs the atomic pass's drift instead of compounding it.
            deficit = {k: max(quota[k] - filled[k], 0.0) for k in SPLITS}
            scale = sum(deficit.values()) or 1.0
            want = {k: int(round(len(games) * deficit[k] / scale)) for k in SPLITS}
            got, cut = carve(games, game_players, elo, want, n_seeds, rng)
            for g, f in got.items():
                placed[g] = f
                filled[f] += 1
            all_cut.extend(cut)

        # Cut games last, each to whichever fold is furthest below quota. They
        # are the only games that can leak a player, so spending them on
        # balance is the best use of them.
        for g in sorted(all_cut, key=lambda x: hashlib.md5(x.encode()).hexdigest()):
            pick = max(quota, key=lambda k: quota[k] - filled[k])
            placed[g] = pick
            filled[pick] += 1

        stats["cut_games"] = len(all_cut)
        mask = components["game_id"].isin(placed)
        assignment = assignment.where(
            ~mask, components["game_id"].map(placed))

    return assignment, stats


def player_sets(merged: pd.DataFrame, col: str) -> dict[str, set]:
    out = {}
    for sp in SPLITS:
        sub = merged[merged[col] == sp]
        out[sp] = set(sub["mover"].unique())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.20)
    ap.add_argument("--max-component-frac", type=float, default=0.10,
                    help="components holding more than this fraction of games "
                         "are carved by partitioning their player graph. 1.0 "
                         "disables carving (pure component split).")
    ap.add_argument("--seeds", type=int, default=64,
                    help="graph-growing seeds per fold inside a carved "
                         "component. Higher = better rating balance, longer "
                         "cut, more straddling players.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-elo-gap", type=float, default=40.0)
    ap.add_argument("--max-base-rate-gap", type=float, default=0.15,
                    help="in percentage points")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if the balance check fails")
    args = ap.parse_args()

    targets = {"train": args.train, "val": args.val, "test": args.test}
    s = sum(targets.values())
    if abs(s - 1.0) > 1e-6:
        targets = {k: v / s for k, v in targets.items()}

    df = pd.read_parquet(args.data, columns=["game_id", "mover", "mover_elo",
                                             "label_valid", "blunder"])
    gp = df.groupby("game_id")["mover"].unique()
    game_players = {g: tuple(ps) for g, ps in gp.items()}
    game_players_list = gp.apply(list)
    n_games = len(game_players)
    n_sides = sum(len(p) for p in game_players.values())
    pg = pd.Series([p for ps in game_players.values() for p in ps]).value_counts()
    n_players = len(pg)

    print(f"{n_games:,} games | {n_players:,} distinct players | "
          f"{n_sides/n_games:.2f} sides/game | {n_sides/n_players:.2f} games/player")

    # A player with exactly one game CANNOT straddle a split, whatever the
    # split does. Multi-game players are the hard ceiling on leakage.
    multi = int((pg > 1).sum())
    print(f"players with >1 game: {multi:,} of {n_players:,} ({multi/n_players:.1%})"
          f" -- only these can straddle a split")

    player_elo = df.groupby("mover")["mover_elo"].median().to_dict()

    comps = build_components(game_players_list)
    comps["split"], stats = assign(comps, game_players, player_elo, targets,
                                   args.max_component_frac, args.seeds, args.seed)

    sizes = comps.groupby("component")["component_games"].first()
    giant = int(sizes.max())
    print(f"\ncomponents: {len(sizes):,} | singletons {int((sizes==1).sum()):,} "
          f"({(sizes==1).mean():.1%}) | largest {giant:,} games "
          f"({giant/n_games:.1%} of the dataset)")
    print("component size distribution:",
          {int(k): int(v) for k, v in sizes.value_counts().head(6).items()})

    if stats["carved"]:
        cg, cut = stats["carved_games"], stats["cut_games"]
        print(f"  CARVED {len(stats['carved'])} component(s) holding {cg:,} games "
              f"({cg/n_games:.1%}). No atomic placement of a component that size\n"
              f"  can give balanced splits, so its player graph was partitioned "
              f"directly.")
        print(f"  cut: {cut:,} of {cg:,} carved games ({cut/max(cg,1):.1%}) have "
              f"their two players on opposite\n  sides of the partition. Those "
              f"are the only games in the entire dataset that\n  can put a "
              f"player in two splits.")
    else:
        print("  no component exceeded the carve threshold; this is a pure "
              "component split.")

    # Naive alternative, reported so the choice is documented not assumed.
    for k in (5, 10):
        keep = sum(1 for ps in game_players.values()
                   if len({stable_fold(p, k) for p in ps}) == 1)
        print(f"  naive both-players-same-fold at K={k} would keep "
              f"{keep:,}/{n_games:,} games ({keep/n_games:.1%})")

    # Game-hash split, kept only as the leakage comparison.
    comps["split_gamehash"] = [
        ("train", "val", "test")[
            0 if stable_fold(g, 10) < 7 else (1 if stable_fold(g, 10) == 7 else 2)]
        for g in comps.game_id
    ]

    merged = df.merge(comps, on="game_id", how="left")

    print()
    rows = []
    for name in ("split", "split_gamehash"):
        for sp in SPLITS:
            m = merged[merged[name] == sp]
            v = m[m.label_valid]
            rows.append([name, sp, f"{m.game_id.nunique():,}", f"{len(m):,}",
                         f"{v.blunder.mean():.4f}" if len(v) else "-",
                         f"{m.mover_elo.median():.0f}"])
    print(pd.DataFrame(rows, columns=["scheme", "split", "games", "rows",
                                      "base rate", "med elo"]).to_string(index=False))

    # ---- leakage, measured both ways -------------------------------------
    players = player_sets(merged, "split")
    overlap = len(players["train"] & players["test"])
    n_test_players = max(len(players["test"]), 1)

    gh = player_sets(merged, "split_gamehash")
    leak = len(gh["train"] & gh["test"])
    gh_test_players = max(len(gh["test"]), 1)

    print()
    print(pd.DataFrame(
        [["component (this file)", f"{overlap:,}", f"{n_test_players:,}",
          f"{overlap/n_test_players:.1%}"],
         ["game-hash (naive)", f"{leak:,}", f"{gh_test_players:,}",
          f"{leak/gh_test_players:.1%}"]],
        columns=["scheme", "train&test players", "test players", "share"],
    ).to_string(index=False))

    if not stats["carved"]:
        print(f"  {'OK' if overlap == 0 else 'BROKEN'}: a pure component split "
              f"must have zero overlap.")
    else:
        share, naive_share = overlap / n_test_players, leak / gh_test_players
        kept = 1 - (share / naive_share) if naive_share else 0.0
        print(f"  Overlap is non-zero by design: carving lets the cut games "
              f"straddle.")
        print(f"  It removes {kept:.0%} of the naive split's player leakage "
              f"({naive_share:.1%} -> {share:.1%}).")
        if share > 0.5 * naive_share:
            print("  WARNING: that is more than half the naive leakage. This is "
                  "closer to a\n  game-level split than a player-disjoint one. "
                  "Say so plainly in the README\n  rather than claiming "
                  "player-disjointness.")
        else:
            print("  Most of the protection survives the carve. Describe it as "
                  "'player-disjoint\n  except within one carved component, "
                  "measured at the share above'.")

    # ---- balance ----------------------------------------------------------
    tr, te = merged[merged.split == "train"], merged[merged.split == "test"]
    d_elo = te.mover_elo.median() - tr.mover_elo.median()
    br_tr = tr[tr.label_valid].blunder.mean()
    br_te = te[te.label_valid].blunder.mean()
    d_br = (br_te - br_tr) * 100
    balanced = abs(d_elo) <= args.max_elo_gap and abs(d_br) <= args.max_base_rate_gap
    print(f"\nbalance: median Elo test - train = {d_elo:+.0f} "
          f"(tol +/-{args.max_elo_gap:g}) | base rate test - train = {d_br:+.3f}pp "
          f"(tol +/-{args.max_base_rate_gap:g}pp)")
    if balanced:
        print("  splits match on rating and base rate. OK")
    else:
        print("  WARNING: splits are not drawn from the same population. The "
              "model would\n  train on one distribution and be scored on "
              "another. Investigate before training.")

    out = comps[["game_id", "component", "component_games", "split", "split_gamehash"]]
    out.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out} ({len(out):,} games)")

    empty = [sp for sp in SPLITS if not (comps.split == sp).any()]
    if empty:
        print(f"FATAL: empty split(s): {', '.join(empty)}")
        return 1
    if not stats["carved"] and overlap:
        print("FATAL: pure component split has non-zero player overlap.")
        return 1
    if args.strict and not balanced:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
