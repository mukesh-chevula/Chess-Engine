import chess.pgn
import pandas as pd
from tqdm import tqdm

pgn_path = "data/db.pgn"
output_path = "processed/training_data.csv"

data = []

with open(pgn_path, encoding='utf-8') as pgn:
    game_counter = 0
    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break

        board = game.board()
        for move in game.mainline_moves():
            fen = board.fen()
            move_uci = move.uci()
            data.append((fen, move_uci))
            board.push(move)

        game_counter += 1
        if game_counter%1000==0:
            print(game_counter)
        if game_counter >= 50000 :
            break

df = pd.DataFrame(data, columns=["fen", "move"])
df.to_csv(output_path, index=False)
print(f"\nSaved dataset with {len(df)} rows to {output_path}")
