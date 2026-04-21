import tkinter as tk
from tkinter import messagebox

BOARD_SIZE = 8
CELL_SIZE = 70
MARGIN = 20

EMPTY = 0
BLACK = 1
WHITE = 2

DIRECTIONS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


class ReversiGame:
    def __init__(self, root):
        self.root = root
        self.root.title("リバーシ")

        self.canvas_size = MARGIN * 2 + CELL_SIZE * BOARD_SIZE
        self.canvas = tk.Canvas(
            root, width=self.canvas_size, height=self.canvas_size + 80, bg="darkgreen"
        )
        self.canvas.pack()

        self.status_label = tk.Label(root, text="", font=("Meiryo", 12))
        self.status_label.pack(pady=5)

        self.restart_button = tk.Button(root, text="最初から", command=self.reset_game)
        self.restart_button.pack(pady=5)

        self.canvas.bind("<Button-1>", self.on_click)

        self.reset_game()

    def reset_game(self):
        self.board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

        mid = BOARD_SIZE // 2
        self.board[mid - 1][mid - 1] = WHITE
        self.board[mid][mid] = WHITE
        self.board[mid - 1][mid] = BLACK
        self.board[mid][mid - 1] = BLACK

        self.current_player = BLACK
        self.game_over = False
        self.draw_board()

    def draw_board(self):
        self.canvas.delete("all")

        # 盤面
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                x1 = MARGIN + col * CELL_SIZE
                y1 = MARGIN + row * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE

                self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill="green", outline="black"
                )

                if self.board[row][col] == BLACK:
                    self.canvas.create_oval(
                        x1 + 8, y1 + 8, x2 - 8, y2 - 8, fill="black"
                    )
                elif self.board[row][col] == WHITE:
                    self.canvas.create_oval(
                        x1 + 8, y1 + 8, x2 - 8, y2 - 8, fill="white"
                    )

        # 打てる場所の表示
        valid_moves = self.get_valid_moves(self.current_player)
        for row, col in valid_moves:
            cx = MARGIN + col * CELL_SIZE + CELL_SIZE // 2
            cy = MARGIN + row * CELL_SIZE + CELL_SIZE // 2
            self.canvas.create_oval(
                cx - 5, cy - 5, cx + 5, cy + 5, fill="yellow", outline=""
            )

        black_count = self.count_stones(BLACK)
        white_count = self.count_stones(WHITE)

        player_text = "黒" if self.current_player == BLACK else "白"
        self.status_label.config(
            text=f"手番: {player_text}    黒: {black_count}  白: {white_count}"
        )

    def on_click(self, event):
        if self.game_over:
            return

        col = (event.x - MARGIN) // CELL_SIZE
        row = (event.y - MARGIN) // CELL_SIZE

        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return

        if not self.is_valid_move(row, col, self.current_player):
            return

        self.place_stone(row, col, self.current_player)
        self.current_player = self.get_opponent(self.current_player)

        if not self.get_valid_moves(self.current_player):
            # 相手が打てなければパス
            opponent_name = "黒" if self.current_player == BLACK else "白"
            self.current_player = self.get_opponent(self.current_player)

            if not self.get_valid_moves(self.current_player):
                self.finish_game()
                return
            else:
                messagebox.showinfo(
                    "パス", f"{opponent_name}は置ける場所がないためパスです。"
                )

        self.draw_board()

    def is_valid_move(self, row, col, player):
        if not self.is_on_board(row, col):
            return False
        if self.board[row][col] != EMPTY:
            return False

        opponent = self.get_opponent(player)

        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            found_opponent = False

            while self.is_on_board(r, c) and self.board[r][c] == opponent:
                r += dr
                c += dc
                found_opponent = True

            if found_opponent and self.is_on_board(r, c) and self.board[r][c] == player:
                return True

        return False

    def get_valid_moves(self, player):
        moves = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.is_valid_move(row, col, player):
                    moves.append((row, col))
        return moves

    def place_stone(self, row, col, player):
        self.board[row][col] = player
        opponent = self.get_opponent(player)

        for dr, dc in DIRECTIONS:
            to_flip = []
            r, c = row + dr, col + dc

            while self.is_on_board(r, c) and self.board[r][c] == opponent:
                to_flip.append((r, c))
                r += dr
                c += dc

            if self.is_on_board(r, c) and self.board[r][c] == player:
                for fr, fc in to_flip:
                    self.board[fr][fc] = player

    def get_opponent(self, player):
        return WHITE if player == BLACK else BLACK

    def is_on_board(self, row, col):
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

    def count_stones(self, player):
        return sum(row.count(player) for row in self.board)

    def finish_game(self):
        self.game_over = True
        black_count = self.count_stones(BLACK)
        white_count = self.count_stones(WHITE)

        if black_count > white_count:
            result = "黒の勝ちです。"
        elif white_count > black_count:
            result = "白の勝ちです。"
        else:
            result = "引き分けです。"

        self.draw_board()
        messagebox.showinfo(
            "ゲーム終了",
            f"ゲーム終了\n\n黒: {black_count}\n白: {white_count}\n\n{result}",
        )


def main():
    root = tk.Tk()
    app = ReversiGame(root)
    root.mainloop()


if __name__ == "__main__":
    main()
