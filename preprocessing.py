## Converting FEN to transformer inputs
PIECE2ID = {
    "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
    "p": 7, "n": 8, "b": 9, "r": 10, "q": 11, "k": 12,
}
def castle_to_token(castle):
    if castle == "-":
        return 13
    token = 0
    if 'K' in castle:       # White can castle kingside
        token |= 1 << 0  
    if 'Q' in castle:       # White can castle queenside
        token |= 1 << 1  
    if 'k' in castle:       # Black can castle kingside
        token |= 1 << 2  
    if 'q' in castle:       # Black can castle queenside
        token |= 1 << 3  
    return token + 13

def batch_select_highest_depth_pv_scaled(batch, c_max=1000):
    selected = {"fen": [], "target": []}

    for fen, evals in zip(batch["fen"], batch["evals"]):
        if not evals:
            continue

        best_eval = max(evals, key=lambda e: e["depth"])
        if not best_eval["pvs"]:
            continue

        pv = best_eval["pvs"][0]
        cp = pv.get("cp")
        mate = pv.get("mate")

        if mate is not None:
            target = 1.0 if mate > 0 else -1.0
        elif cp is not None:
            target = max(-1.0, min(1.0, cp / c_max))        # essentially, any centipawn evaluation above a 1000 (i.e. 10 pawns worth of) is clamped to + or - 1, equating it to mate
        else:
            continue  # skip if no usable score

        selected["fen"].append(fen)
        selected["target"].append(target)

    return selected

def fen_to_token_ids(fen):
    """Convert FEN to a 66-token sequence: 64 for squares + 2 metadata tokens (one token for side-to-move, one token for castling)."""
    parts = fen.split()
    board_str, stm, castle, ep = parts[0], parts[1], parts[2], parts[3]

    seq = []
    for rank in board_str.split('/'):
        for ch in rank:
            if ch.isdigit():
                seq.extend([0] * int(ch))
            else:
                seq.append(PIECE2ID[ch])

    seq.append(castle_to_token(castle))
    seq.append(29 if stm == 'w' else 30)
    
    return seq

def fen_to_piece_maps(fen):
    parts = fen.split()
    board_str, stm = parts[0], parts[1]
    # 12 channels, with indexes: 0 = white pawn, 1 = white knight, ... 5 = white king, 6 = black pawn, ... 11 = black king
    maps = [[[0.0 for _ in range(8)] for _ in range(8)] for _ in range(12)]
    r = 0
    for rank in board_str.split('/'):
        f = 0
        for ch in rank:
            if ch.isdigit():
                f += int(ch)
            else:
                c = PIECE2ID[ch] - 1        # channels correspond to the token numbers from earlier, but 0-indexed instead
                maps[c][r][f] = 1.0
                f += 1
        r += 1
    flag = 1.0 if stm == 'w' else 0.0
    flag_map = [[[flag for _ in range(8)] for _ in range(8)]]
    
    return maps + flag_map  # final shape: 13x8x8

def add_representations(batch):
    tokens, maps = [], []
    for fen in batch["fen"]:
        tokens.append(fen_to_token_ids(fen))
        maps.append(fen_to_piece_maps(fen))  # similarly, return nested lists
    return {
        "token_representation": tokens,
        "piece_maps": maps
    }

if __name__ == '__main__':
    print(castle_to_token("KQkq"))