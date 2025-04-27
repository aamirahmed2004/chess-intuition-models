import tkinter as tk
from tkinter import messagebox
import chess
import random
from PIL import Image, ImageTk
import torch

from models import ChessEvalResNet, ChessEvalMultiTaskTransformer

# Size for each square (in pixels)
tile_size = 64
# Paths to 60×60 PNG assets; adjust if yours differ
paths = {
    'P':'assets/Chess_plt60.png','p':'assets/Chess_pdt60.png',
    'R':'assets/Chess_rlt60.png','r':'assets/Chess_rdt60.png',
    'N':'assets/Chess_nlt60.png','n':'assets/Chess_ndt60.png',
    'B':'assets/Chess_blt60.png','b':'assets/Chess_bdt60.png',
    'Q':'assets/Chess_qlt60.png','q':'assets/Chess_qdt60.png',
    'K':'assets/Chess_klt60.png','k':'assets/Chess_kdt60.png'
}

class ChessGUI:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessEvalMultiTaskTransformer().to(self.device)
        state_dict = torch.load('best_chess_multi_transformer.pth', map_location=self.device)
        try:
            # If state_dict is an OrderedDict, use load_state_dict
            self.model.load_state_dict(state_dict)
        except AttributeError:
            # If torch.load returned the full model, assign directly
            self.model = state_dict
        self.model.to(self.device)
        self.model.eval()

        # Initialize board and window
        self.board = chess.Board()
        self.window = tk.Tk()
        self.window.title("Simple Chess GUI")

        # Create and pack canvas
        self.canvas = tk.Canvas(
            self.window,
            width=8 * tile_size,
            height=8 * tile_size
        )
        self.canvas.pack()

        # Bind click event
        self.canvas.bind('<Button-1>', self.on_click)

        # Track selection
        self.selected_square = None

        # Preload and resize piece images
        self.piece_images = {
            sym: ImageTk.PhotoImage(
                Image.open(path).resize((tile_size, tile_size))
            )
            for sym, path in paths.items()
        }

        # Initial draw
        self.draw_board()
        self.window.mainloop()

    def draw_board(self):
        colors = ["#F0D9B5", "#B58863"]
        self.canvas.delete("all")

        for r in range(8):
            for c in range(8):
                x, y = c * tile_size, r * tile_size
                # Draw square
                self.canvas.create_rectangle(
                    x, y, x + tile_size, y + tile_size,
                    fill=colors[(r + c) % 2],
                    outline=""
                )
                # Draw piece if present
                sq = chess.square(c, 7 - r)
                piece = self.board.piece_at(sq)
                if piece:
                    img = self.piece_images[piece.symbol()]
                    self.canvas.create_image(x, y, anchor='nw', image=img)

    def on_click(self, event):
        col = event.x // tile_size
        row = event.y // tile_size
        square = chess.square(col, 7 - row)

        # Select a piece to move
        if self.selected_square is None:
            piece = self.board.piece_at(square)
            # Only allow selecting White pieces
            if piece and piece.color == chess.WHITE:
                self.selected_square = square
                self.draw_board()
                # Highlight the selected square
                x1, y1 = col * tile_size, row * tile_size
                self.canvas.create_rectangle(
                    x1, y1, x1 + tile_size, y1 + tile_size,
                    outline="red", width=3
                )
        else:
            move = chess.Move(self.selected_square, square)

            if move in self.board.legal_moves:
                self.board.push(move)
                self.selected_square = None
                self.draw_board()
                self.window.update_idletasks()
                # Check for game end
                if self.board.is_game_over():
                    messagebox.showinfo(
                        "Game Over", f"Result: {self.board.result()}"
                    )
                    return
                self.window.after(150, self.make_cpu_move)
            else:
                # Invalid: reset selection
                self.selected_square = None
                self.draw_board()

    def fen_to_tensor(self, fen: str) -> torch.Tensor:
        # Converting FEN to transformer inputs
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
        
        return torch.tensor(seq, dtype=torch.long, device=self.device)

    def evaluate_move(self, fen):
        # Returns (eval_output, win_chance)
        tensor = self.fen_to_tensor(fen).unsqueeze(0).to(self.device)  # add batch dim
        with torch.no_grad():
            eval_output, win_chance = self.model(tensor)
        return eval_output.item(), win_chance.item()

    def make_cpu_move(self):
        if self.board.is_game_over():
            return
        candidates = []
        for move in self.board.legal_moves:
            bcopy = self.board.copy()
            bcopy.push(move)
            fen = bcopy.fen()
            eval_out, win_chance = self.evaluate_move(fen)
            candidates.append((move, eval_out, win_chance))

        # Sort ascending by eval (since black tries to minimize)
        candidates.sort(key=lambda x: x[1])
        print(len(candidates))
        # Find first move with win_chance > 0.5
        for move, _, win in candidates:
            if win > 0.5:
                best_move = move
                break
        else:
            # No winning chance move found; pick best eval
            best_move = candidates[0][0]

        # Play chosen move
        self.board.push(best_move)
        self.draw_board()
        if self.board.is_game_over():
            messagebox.showinfo("Game Over", f"Result: {self.board.result()}")

if __name__ == '__main__':
    try:
        import chess
    except ImportError:
        messagebox.showerror(
            "Missing Dependency",
            "Please install python-chess: pip install python-chess"
        )
        exit(1)
    ChessGUI()
