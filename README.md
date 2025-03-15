# Chess Engine — Play Against a Deep Learning-Based Chess AI

A fun and interactive chess engine where you can play move-by-move against a **deep learning-powered AI** through a **Pygame graphical interface**. This project uses a **CNN-based model** trained on thousands of real games to predict the best possible moves.

---

## Features

- Play chess against an AI model trained on real Lichess games.
- GUI-based interactive board using **Pygame** (no need to type moves manually).
- AI plays instantly in response to your move.
- Legal moves are respected, and the game progresses until checkmate or draw.
- Smart move prediction using a **Convolutional Neural Network (CNN)**.
- Clean board rendering with **PNG piece images**.
- Future features: Undo, move history, sound effects, stronger evaluation.

---

## 🧠 Tech Stack & Concepts

| Component        | Description                                 |
| ---------------- | ------------------------------------------- |
| **Python 3.12+** | Main programming language                   |
| **Pygame**       | GUI board display and event handling        |
| **PyTorch**      | Deep learning model loading and inference   |
| **python-chess** | Chess logic & board representation          |
| **CNN Model**    | Trained to predict the best move from a FEN |

---

# How to run the AI:

### Clone this GitHub repository

- Clone this git repository by ` git clone repository/link`

### Download a Chess Game Database from LiChess [Click Here to to visit download page](https://database.lichess.org)

- Convert the `.zst` file to `.pgn` file

* **Unix:** pzstd -d filename.pgn.zst (faster than unzstd)

* **Windows:** use PeaZip

### Install the required dependencies

- Run `pip insall requirements.txt` to install all required dependencies

### Run `parse_pgn.py`

- Run `parse_pgn.py` to make `training_data.csv` which will be further used in our game

### Run `preprocess_dataset.py`

- Run `preprocess_dataset.py` to make `X.npy`, `y.npy`, `idx_to_move.npy` & `move_to_idx.npy` which will be further used in our game

### Run `train_model.py`

- Run `train_model.py` to train the CNN-based move prediction model. This will generate the `move_predictor.pt` file inside the `models/ `directory.

### Play the Game using `chess_gui.py`

- Finally, run the graphical interface:
  python `chess_gui.py`
  You can now play chess against your trained AI in a fully interactive GUI!

### Note (Important)

The model (`move_predictor.pt`) and data files (`X.npy`, `y.npy`, etc.) are not pushed to the repository, so make sure to:
Parse and preprocess your own dataset (from Lichess PGNs).
Train your own model using the provided scripts.

### Future Improvements

- Add move history panel and undo functionality.
- Add audio effects for moves/check/mate.
- Add a stronger evaluation layer for deeper prediction accuracy.
- Support for black-side play or multiplayer mode.

### License

This project is open source and free to use for learning and educational purposes.
