# scripts/02_preprocess_dataset.py
import pandas as pd
import numpy as np
import chess

PIECE_TO_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

def fen_to_tensor(fen):
    board_matrix = np.zeros((12, 8, 8), dtype=np.float32)
    board = chess.Board(fen)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - square // 8
            col = square % 8
            idx = PIECE_TO_INDEX[piece.symbol()]
            board_matrix[idx][row][col] = 1
    return board_matrix

def build_move_vocab(moves):
    unique_moves = sorted(set(moves))
    move_to_idx = {move: idx for idx, move in enumerate(unique_moves)}
    idx_to_move = {idx: move for move, idx in move_to_idx.items()}
    return move_to_idx, idx_to_move

input_path = "processed/training_data.csv"
df = pd.read_csv(input_path)

print("Building move vocabulary...")
move_to_idx, idx_to_move = build_move_vocab(df["move"])
np.save("processed/move_to_idx.npy", move_to_idx)
np.save("processed/idx_to_move.npy", idx_to_move)

print("Encoding FENs and Moves...")
X = []
y = []

for i, row in df.iterrows():
    fen_tensor = fen_to_tensor(row["fen"])
    X.append(fen_tensor)
    y.append(move_to_idx[row["move"]])

    if (i+1) % 10000 == 0:
        print(f"Processed {i+1} samples...")

X = np.stack(X)
y = np.array(y)

np.save("processed/X.npy", X)
np.save("processed/y.npy", y)
print(f"Saved tensors: X shape = {X.shape}, y shape = {y.shape}")
