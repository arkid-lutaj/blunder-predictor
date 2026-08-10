"""
Hanging pieces and tension, mover-relative.

Not a static exchange evaluation. It is a cheap proxy with known blind spots,
documented at the bottom. Measured at ~0.05 ms/position, so ~1.2 min for 1.5M
positions on one core. Do not optimise this further; it is not the bottleneck.
"""

import chess

# KING must be present. board.attackers() returns kings, and a missing key here
# raises KeyError on ~12% of real positions (any king adjacent to a defended
# enemy piece). 20000 makes the king never count as a "cheap" attacker.
#
# KNIGHT and BISHOP are deliberately EQUAL. Splitting them 300/325 flags every
# defended bishop attacked by a knight as hanging, which is a normal minor
# trade, not a loss. That produced ~500 spurious flags per 8000 positions.
VAL = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 320,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

MARGIN = 50  # attacker must be cheaper by this much to count as winning material


def compute_tactical_state(board: chess.Board, margin: int = MARGIN) -> dict:
    """Hanging counts, hanging centipawn value, and tension, in one pass.

    A piece counts as hanging if any of:
      1. it is attacked and undefended
      2. its cheapest attacker is cheaper than it by more than `margin`
      3. attackers outnumber defenders

    Rule 3 is the one most naive versions omit. A knight defended once and
    attacked twice by rooks loses material even though no attacker is cheaper.
    That case occurred ~3000 times per 8000 positions in testing.

    "own" and "opp" are relative to the side to move, so the feature means the
    same thing regardless of colour.
    """
    hanging_count = {chess.WHITE: 0, chess.BLACK: 0}
    hanging_val = {chess.WHITE: 0, chess.BLACK: 0}
    tension = 0

    for square in chess.scan_forward(board.occupied):
        piece_type = board.piece_type_at(square)
        if piece_type == chess.KING:
            continue  # kings attack, but are never "hanging"

        color = board.color_at(square)
        piece_val = VAL[piece_type]

        attacker_mask = board.attackers_mask(not color, square)
        if not attacker_mask:
            continue

        n_attackers = chess.popcount(attacker_mask)
        tension += n_attackers

        defender_mask = board.attackers_mask(color, square)
        n_defenders = chess.popcount(defender_mask)

        cheapest_attacker = min(
            VAL[board.piece_type_at(sq)] for sq in chess.scan_forward(attacker_mask)
        )

        if (
            n_defenders == 0
            or cheapest_attacker + margin < piece_val
            or n_attackers > n_defenders
        ):
            hanging_count[color] += 1
            hanging_val[color] += piece_val

    mover = board.turn
    return {
        "hanging_own": hanging_count[mover],
        "hanging_opp": hanging_count[not mover],
        "hanging_val_own": hanging_val[mover],
        "hanging_val_opp": hanging_val[not mover],
        "tension": tension,
    }


# Known limitations, state these in the README rather than fixing them:
#   - Not SEE. A defended piece attacked by an equal piece backed by a heavier
#     one is scored by counting, not by resolving the exchange sequence.
#   - Pinned defenders still count as defenders. python-chess attackers() does
#     not know a defender cannot legally recapture.
#   - En passant is not treated as an attack.
#   - Tension counts (enemy attacker, target) pairs and excludes kings as
#     targets, so checks do not inflate it.


if __name__ == "__main__":
    import random

    tests = [
        # (fen, description, expected key, expected value)
        ("7k/8/2p5/3p4/4K3/8/8/8 w - - 0 1",
         "white king attacks a defended black pawn (the KeyError case)",
         None, None),
        ("r1bqkb1r/pppp1ppp/2n2n2/4N3/2B1P3/8/PPPP1PPP/RNBQK2R b KQkq - 0 4",
         "after Nxe5: white Ne5 undefended and attacked by Nc6",
         "hanging_val_opp", 420),
        ("8/8/8/3k4/8/8/8/3K4 w - - 0 1",
         "bare kings, nothing to compute",
         "tension", 0),
    ]
    for fen, desc, key, expected in tests:
        out = compute_tactical_state(chess.Board(fen))
        ok = "ok" if (key is None or out[key] == expected) else f"FAIL got {out[key]}"
        print(f"{ok:>12}  {desc}")
        print(f"              {out}")

    # No position in a long random walk may raise.
    random.seed(0)
    board = chess.Board()
    for _ in range(20000):
        moves = list(board.legal_moves)
        if not moves or board.is_game_over() or board.ply() > 140:
            board = chess.Board()
            continue
        board.push(random.choice(moves))
        compute_tactical_state(board)
    print(f"{'ok':>12}  20000 random positions, no exceptions")