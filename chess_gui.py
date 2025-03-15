import pygame
import chess
import torch
import torch.nn as nn
import numpy as np
import os

pygame.init()

print("Loading model and move mappings...")
model_path = "models/move_predictor.pt"
move_to_idx_path = "processed/move_to_idx.npy"
idx_to_move_path = "processed/idx_to_move.npy"

move_to_idx = np.load(move_to_idx_path, allow_pickle=True).item()
idx_to_move = np.load(idx_to_move_path, allow_pickle=True).item()
num_classes = len(move_to_idx)

class MovePredictorCNN(nn.Module):
    def __init__(self, output_classes):
        super(MovePredictorCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, output_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MovePredictorCNN(output_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ Model ready.")

WIDTH, HEIGHT = 480, 480
SQUARE_SIZE = WIDTH // 8
WHITE = (238, 238, 210)
BLACK = (118, 150, 86)

def load_piece_images():
    images = {}
    base_path = os.path.join("assets", "png")
    piece_map = {
        'P': 'pawn-w.png', 'N': 'knight-w.png', 'B': 'bishop-w.png', 'R': 'rook-w.png', 'Q': 'queen-w.png', 'K': 'king-w.png',
        'p': 'pawn-b.png', 'n': 'knight-b.png', 'b': 'bishop-b.png', 'r': 'rook-b.png', 'q': 'queen-b.png', 'k': 'king-b.png'
    }

    for piece_symbol, filename in piece_map.items():
        path = os.path.join(base_path, filename)
        image = pygame.image.load(path)
        image = pygame.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE))
        images[piece_symbol] = image
    return images

PIECE_IMAGES = load_piece_images()

piece_to_idx = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

def fen_to_tensor(fen):
    board_tensor = np.zeros((12, 8, 8), dtype=np.float32)
    board = chess.Board(fen)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            idx = piece_to_idx[piece.symbol()]
            row = 7 - (square // 8)
            col = square % 8
            board_tensor[idx, row, col] = 1.0
    return board_tensor

def predict_top_move(board):
    input_tensor = torch.tensor(fen_to_tensor(board.fen())).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        sorted_indices = torch.argsort(probs, descending=True)
        for idx in sorted_indices:
            move_uci = idx_to_move.get(idx.item())
            if move_uci:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    return move
    return None

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess Engine GUI")

def draw_board():
    for row in range(8):
        for col in range(8):
            color = WHITE if (row + col) % 2 == 0 else BLACK
            pygame.draw.rect(screen, color, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def draw_pieces(board):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)
            col = square % 8
            screen.blit(PIECE_IMAGES[piece.symbol()], (col * SQUARE_SIZE, row * SQUARE_SIZE))

board = chess.Board()
selected_square = None
running = True
font = pygame.font.SysFont(None, 36)

def show_game_over_message():
    result_text = "Game Over"
    if board.is_checkmate():
        result_text = "Checkmate! You win!" if board.turn == chess.BLACK else "Checkmate! Engine wins!"
    elif board.is_stalemate():
        result_text = "Stalemate!"
    elif board.is_insufficient_material():
        result_text = "Draw by insufficient material"
    elif board.can_claim_fifty_moves():
        result_text = "Draw by 50-move rule"
    elif board.can_claim_threefold_repetition():
        result_text = "Draw by repetition"

    text_surface = font.render(result_text, True, (255, 0, 0))
    text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(text_surface, text_rect)
    pygame.display.flip()
    pygame.time.wait(4000)  

board = chess.Board()
selected_square = None
running = True

while running:
    draw_board()
    draw_pieces(board)
    pygame.display.flip()

    if board.is_game_over():
        show_game_over_message()
        running = False
        continue

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            col = x // SQUARE_SIZE
            row = 7 - (y // SQUARE_SIZE)
            clicked_square = chess.square(col, row)

            if selected_square is None:
                piece = board.piece_at(clicked_square)
                if piece and piece.color == chess.WHITE:
                    selected_square = clicked_square
            else:
                move = chess.Move(selected_square, clicked_square)
                if move in board.legal_moves:
                    board.push(move)

                    if board.is_game_over():
                        continue  
                    engine_move = predict_top_move(board)
                    if engine_move:
                        board.push(engine_move)
                selected_square = None

pygame.quit()
