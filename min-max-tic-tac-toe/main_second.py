import pygame
import sys
import numpy as np

class TicTacToe:
    def __init__(self):
        pygame.init()
        self.width = 300
        self.height = 300
        self.line_width = 15
        self.win_line_width = 15
        self.board_rows = 3
        self.board_cols = 3
        self.square_size = self.width // self.board_cols
        self.circle_radius = self.square_size // 3
        self.circle_width = 15
        self.cross_width = 25
        self.space = self.square_size // 4

        self.bg_color = (28, 170, 156)
        self.line_color = (23, 145, 135)
        self.circle_color = (239, 231, 200)
        self.cross_color = (66, 66, 66)
        self.red_color = (255, 0, 0)

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Tic Tac Toe")
        self.screen.fill(self.bg_color)

        self.board = np.zeros((self.board_rows, self.board_cols), dtype=int)
        self.player = 1
        self.game_over = False

        self.draw_lines()

    def draw_lines(self):
        for col in range(1, self.board_cols):
            pygame.draw.line(self.screen, self.line_color, (col * self.square_size, 0),
                             (col * self.square_size, self.height), self.line_width)
            pygame.draw.line(self.screen, self.line_color, (0, col * self.square_size),
                             (self.width, col * self.square_size), self.line_width)

    def draw_figures(self):
        for row in range(self.board_rows):
            for col in range(self.board_cols):
                if self.board[row, col] == 1:
                    start_desc = (col * self.square_size + self.space, row * self.square_size + self.space)
                    end_desc = (col * self.square_size + self.square_size - self.space,
                                row * self.square_size + self.square_size - self.space)
                    pygame.draw.line(self.screen, self.cross_color, start_desc, end_desc, self.cross_width)
                    pygame.draw.line(self.screen, self.cross_color,
                                     (start_desc[0], end_desc[1]), (end_desc[0], start_desc[1]), self.cross_width)
                elif self.board[row, col] == 2:
                    center = (int(col * self.square_size + self.square_size / 2),
                              int(row * self.square_size + self.square_size / 2))
                    pygame.draw.circle(self.screen, self.circle_color, center, self.circle_radius, self.circle_width)

    def available_square(self, row, col):
        return self.board[row, col] == 0

    def check_win(self, player):
        for col in range(self.board_cols):
            if np.all(self.board[:, col] == player):
                return True

        for row in range(self.board_rows):
            if np.all(self.board[row, :] == player):
                return True

        if np.all(np.diag(self.board) == player):
            return True

        if np.all(np.diag(np.fliplr(self.board)) == player):
            return True

        return False

    def reset(self):
        self.screen.fill(self.bg_color)
        self.draw_lines()
        self.board = np.zeros((self.board_rows, self.board_cols), dtype=int)
        self.player = 1
        self.game_over = False

    def check_win_draw_lines(self, player):
        for col in range(self.board_cols):
            if np.all(self.board[:, col] == player):
                pygame.draw.line(
                    self.screen,
                    self.red_color,
                    (col * self.square_size + self.square_size // 2, 0),
                    (col * self.square_size + self.square_size // 2, self.height),
                    self.win_line_width,
                )

        for row in range(self.board_rows):
            if np.all(self.board[row, :] == player):
                pygame.draw.line(
                    self.screen,
                    self.red_color,
                    (0, row * self.square_size + self.square_size // 2),
                    (self.width, row * self.square_size + self.square_size // 2),
                    self.win_line_width,
                )

        if np.all(np.diag(self.board) == player):
            pygame.draw.line(
                self.screen,
                self.red_color,
                (0, 0),
                (self.width, self.height),
                self.win_line_width,
            )

        if np.all(np.diag(np.fliplr(self.board)) == player):
            pygame.draw.line(
                self.screen,
                self.red_color,
                (0, self.height),
                (self.width, 0),
                self.win_line_width,
            )

    def evaluate(self):
        if self.check_win(2):
            return 10
        elif self.check_win(1):
            return -10
        else:
            return 0

    def minimax(self, depth, is_maximizing):
        score = self.evaluate()
        if score == 10 or score == -10:
            return score
        if np.all(self.board != 0):
            return 0

        if is_maximizing:
            best_score = -np.inf
            for i in range(self.board_rows):
                for j in range(self.board_cols):
                    if self.board[i, j] == 0:
                        self.board[i, j] = 2
                        score = self.minimax(depth + 1, False)
                        self.board[i, j] = 0
                        best_score = max(score, best_score)
            return best_score
        else:
            best_score = np.inf
            for i in range(self.board_rows):
                for j in range(self.board_cols):
                    if self.board[i, j] == 0:
                        self.board[i, j] = 1
                        score = self.minimax(depth + 1, True)
                        self.board[i, j] = 0
                        best_score = min(score, best_score)
            return best_score

    def alpha_pruning(self, depth, is_maximizing, alpha=-np.inf, beta=np.inf):
        score = self.evaluate()
        if score == 10 or score == -10:
            return score
        if np.all(self.board != 0):
            return 0

        if is_maximizing:
            best_score = -np.inf
            for i in range(self.board_rows):
                for j in range(self.board_cols):
                    if self.board[i, j] == 0:
                        self.board[i, j] = 2
                        score = self.alpha_pruning(depth + 1, False, alpha, beta)
                        self.board[i, j] = 0
                        best_score = max(best_score, score)
                        alpha = max(alpha, best_score)
                        if beta <= alpha:
                            break
                if beta <= alpha:
                    break
            return best_score
        else:
            best_score = np.inf
            for i in range(self.board_rows):
                for j in range(self.board_cols):
                    if self.board[i, j] == 0:
                        self.board[i, j] = 1
                        score = self.alpha_pruning(depth + 1, True, alpha, beta)
                        self.board[i, j] = 0
                        best_score = min(best_score, score)
                        beta = min(beta, best_score)
                        if beta <= alpha:
                            break
                if beta <= alpha:
                    break
            return best_score

    def best_move(self, pruning=False):
        best_score = -np.inf
        move = (-1, -1)
        for i in range(self.board_rows):
            for j in range(self.board_cols):
                if self.board[i, j] == 0:
                    self.board[i, j] = 2
                    if pruning:
                        score = self.alpha_pruning(0, False)
                    else:
                        score = self.minimax(0, False)
                    self.board[i, j] = 0
                    if score > best_score:
                        best_score = score
                        move = (i, j)
        return move

    def main(self, pruning=False):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN and not self.game_over:
                    mouseX = event.pos[0]
                    mouseY = event.pos[1]
                    clicked_row = int(mouseY // self.square_size)
                    clicked_col = int(mouseX // self.square_size)

                    if self.available_square(clicked_row, clicked_col):
                        self.board[clicked_row, clicked_col] = 1
                        if self.check_win(1):
                            self.game_over = True
                            self.check_win_draw_lines(1)
                            continue
                        move = self.best_move(pruning)
                        if move != (-1, -1):
                            self.board[move[0], move[1]] = 2
                            if self.check_win(2):
                                self.game_over = True
                                self.check_win_draw_lines(2)
                                continue
                    else:
                        self.game_over = True
                        continue
        
                if event.type == pygame.MOUSEBUTTONDOWN and self.game_over:
                    self.reset()
            self.draw_figures()
            pygame.display.update()

        pygame.quit()

if __name__ == "__main__":
    game = TicTacToe()
    game.main(pruning=True)
