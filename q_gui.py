"""
This file contains the code for the 3D-like Tic-Tac-Toe game using Pygame.
"""
import random
import math
import torch
import pygame
import pygame.gfxdraw
from eightq_net import QueensNet

QUEEN_COLOR = (100, 200, 250)  # Bright red color
SHADOW_COLOR = (0, 0, 100, 10)  # Black color
OUTLINE_COLOR = (0, 0, 255, 255)  # Black color
SHDOW_OUTLINE_COLOR = (0, 0, 0, 250)
BLACK_CELL_COLOR = [30, 30, 30]  # Black color
WHITE_CELL_COLOR = [159, 150, 150]  # Bright white color


class GameGUI:
    """
    This class is responsible for the graphical user interface of the Tic-Tac-Toe game.
    """

    def __init__(self):
        pygame.init()
        self.game = QueensNet(size=48)
        self.board_size = 1200  # Size of the board in pixels
        self.cell_size = self.board_size // self.game.size  # Size of each cell in pixels
        self.screen = pygame.display.set_mode(
            (self.board_size, self.board_size))
        self.clock = pygame.time.Clock()
        self.board = self.game.s.detach()
        # Create a grid of indices
        self.indices = None
        self.diag_indices = None
        self.init_indices()
        self.gradients = []
        self.init_gradients()
        self.init_cell_sizes()
        pygame.display.set_caption(f'{self.game.size} Queens Problem')
        self.done_x_pos = 0
        self.directions = [(1, 0), (-1, 0), (0, 1), (0, -1)
                           ]  # right, left, down, up
        self.direction = random.choice(self.directions)
        self.pos = [0, 0]  # Initial position [x, y]
        self.speed = 1  # Speed of movement in pixels per frame
        self.last_time = pygame.time.get_ticks()
        self.selected_queen = None

    def init_cell_sizes(self):
        self.half_cell_size = self.cell_size // 2
        self.sixth_cell_size = self.cell_size // 6
        self.eight_cell_size = self.cell_size // 8

    def init_gradients(self):
        for i in range(2):
            base_color = torch.tensor(
                BLACK_CELL_COLOR, dtype=torch.float32) if i % 2 == 0 else torch.tensor(WHITE_CELL_COLOR, dtype=torch.float32)
            gradient = torch.zeros(
                (self.cell_size, self.cell_size, 3), dtype=torch.uint8)
            center = torch.tensor(self.cell_size / 2)
            for k in range(self.cell_size):
                for l in range(self.cell_size):
                    # Convert k and l to Tensor objects
                    k_tensor = torch.tensor(k, dtype=torch.float32)
                    l_tensor = torch.tensor(l, dtype=torch.float32)
                    # Calculate distance from the center
                    dist_to_center = torch.sqrt(
                        (k_tensor - center) ** 2 + (l_tensor - center) ** 2)
                    # Modify the color based on the distance to the center
                    color = base_color + (1 - dist_to_center / center) * \
                        torch.tensor([0.5, 0.5, 0.5])
                    gradient[k, l, :] = color
            self.gradients.append(
                pygame.surfarray.make_surface(gradient.cpu().numpy()))

    def init_indices(self):
        x = torch.arange(self.game.size).view(-1,
                                              1).expand(self.game.size, self.game.size)
        y = torch.arange(self.game.size).view(
            1, -1).expand(self.game.size, self.game.size)
        self.indices = torch.stack((x, y), dim=2)
        self.diag_indices = torch.zeros(
            (self.game.size, self.game.size, 2, self.game.size), dtype=torch.long)
        for x in range(self.game.size):
            for y in range(self.game.size):
                diag1_indices = torch.arange(
                    max(0, x - y), min(self.game.size, self.game.size + x - y))
                diag2_indices = torch.arange(
                    max(0, x + y - self.game.size + 1), min(x + y + 1, self.game.size))
                self.diag_indices[x, y, 0,
                                  :diag1_indices.size(0)] = diag1_indices
                self.diag_indices[x, y, 1,
                                  :diag2_indices.size(0)] = diag2_indices

    def draw_board(self, board=None):
        if board is None:
            board = self.game.s

        cell_size = self.cell_size
        eight_cell_size = self.eight_cell_size
        gradients = self.gradients
        screen = self.screen
        indices = self.indices.view(-1, 2)
        game_size = self.game.size
        board_size = self.board_size

        for i in range(game_size):
            for j in range(game_size):
                screen.blit(gradients[(i + j) % 2],
                            (j * cell_size, i * cell_size))
                rect_size = cell_size - 2 * eight_cell_size
                rect_x = j * cell_size + eight_cell_size
                rect_y = i * cell_size + eight_cell_size
                pygame.gfxdraw.box(screen, pygame.Rect(
                    rect_x, rect_y, rect_size, rect_size), SHADOW_COLOR)

        energy = self.game.energy
        if energy == 0:
            self.draw_done()
        else:
            self.draw_energy(energy, board_size, cell_size)

        if self.game.missing_queens > 0:
            self.draw_missing_queens(board, cell_size)

        for index in indices:
            x, y = index
            if board[x, y] == 1:
                self.draw_queen(x, y, QUEEN_COLOR, cell_size)

        button_surface = pygame.Surface(
            (cell_size, cell_size), pygame.SRCALPHA)
        button_surface.fill((0, 0, 0, 128))

        font = pygame.font.Font(None, 36)
        text_next = font.render("N", 1, (255, 255, 255))
        textpos = text_next.get_rect(
            centerx=cell_size // 2, centery=cell_size // 2)

        button_surface.blit(text_next, textpos)
        screen.blit(button_surface, (board_size -
                    cell_size, board_size - cell_size))

        # Create a new surface for the "Reset" text
        button_surface = pygame.Surface(
            (cell_size, cell_size), pygame.SRCALPHA)
        button_surface.fill((0, 0, 0, 128))

        text_reset = font.render("R", 1, (255, 255, 255))
        button_surface.blit(text_reset, textpos)
        screen.blit(button_surface, (board_size -
                    cell_size, board_size - 2 * cell_size))

        return energy

    def draw_missing_queens(self, board, cell_size):
        # mark red cell for every missing queen
        missing_queens = self.game.missing_queens
        for x in range(self.game.size):
            for y in range(self.game.size):
                if missing_queens == 0:
                    return
                pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(
                    y * cell_size, x * cell_size, cell_size, cell_size))
                missing_queens -= 1

    def draw_queen(self, x, y, color, cell_size):
        center = (y * cell_size + self.half_cell_size,
                  x * cell_size + self.half_cell_size)
        q_size = cell_size + 2 * self.sixth_cell_size
        offsets = range(0, q_size, self.sixth_cell_size//3)
        gradient_colors = [(max(0, color[0] - 1 * offset), max(0, color[1] -
                            1 * offset), max(0, color[2] - 1 * offset)) for offset in offsets]
        for offset in range(0, q_size, self.sixth_cell_size//3):
            highlight_radius = max(
                self.half_cell_size - 2, self.half_cell_size - offset)
            gradient_color = gradient_colors[offset //
                                             (self.sixth_cell_size // 3)]
            alpha = 255
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1] - offset, highlight_radius, (
                gradient_color[0], gradient_color[1], gradient_color[2], alpha))
            if offset < q_size - self.sixth_cell_size // 3:
                pygame.gfxdraw.filled_circle(
                    self.screen, center[0], center[1] - offset, highlight_radius, (gradient_color[0], gradient_color[1], gradient_color[2], alpha))
                # add shadow
                pygame.gfxdraw.aaellipse(self.screen, center[0], center[1] - offset + 2, highlight_radius - 2, highlight_radius - 2, (
                    SHDOW_OUTLINE_COLOR))
                # add outline
                pygame.gfxdraw.aaellipse(
                    self.screen, center[0], center[1] - offset, highlight_radius, highlight_radius, (OUTLINE_COLOR))

            # ...

            else:
                for angle in range(0, 360, 30):
                    # Calculate the coordinates of the triangle
                    angle_rad = torch.tensor(
                        angle, dtype=torch.float32) * math.pi / 180
                    x1 = center[0] + highlight_radius * torch.cos(angle_rad)
                    y1 = center[1] - offset + \
                        highlight_radius * torch.sin(angle_rad)
                    angle_rad = torch.tensor(
                        angle + 30, dtype=torch.float32) * math.pi / 180
                    x2 = center[0] + highlight_radius * torch.cos(angle_rad)
                    y2 = center[1] - offset + \
                        highlight_radius * torch.sin(angle_rad)
                    angle_rad = torch.tensor(
                        angle + 15, dtype=torch.float32) * math.pi / 180
                    x3 = center[0] + (highlight_radius +
                                      self.cell_size // 10) * torch.cos(angle_rad)
                    y3 = center[1] - offset + (highlight_radius +
                                               self.cell_size // 10) * torch.sin(angle_rad)
                    # Draw the triangle
                    pygame.gfxdraw.aatrigon(self.screen, int(x1.item()), int(y1.item()), int(x2.item()), int(
                        y2.item()), int(x3.item()), int(y3.item()), gradient_color)  # Use gradient color for the crown
                    pygame.gfxdraw.filled_trigon(self.screen, int(x1.item()), int(y1.item()), int(x2.item()), int(
                        y2.item()), int(x3.item()), int(y3.item()), gradient_color)  # Use gradient color for the crown

    def draw_energy(self, energy, board_size, cell_size):
        # Convert the energy value to a string
        energy_str = str(energy)
        # extract the number from "tensor([number])"
        energy_str = energy_str[energy_str.index(
            '(') + 1:energy_str.index(')')]
        # remove the decimal point
        energy_str = energy_str.replace('.', '')
        # add "Energy: " to the string
        energy_str = "Energy: " + energy_str

        # Create a mapping from characters to a grid of cells
        char_to_pixels = {
            '0': [[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]],
            '1': [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
            '2': [[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]],
            '3': [[1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
            '4': [[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]],
            '5': [[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
            '6': [[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1]],
            '7': [[1, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0]],
            '8': [[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]],
            '9': [[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
            ' ': [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            '-': [[0, 0, 0], [0, 0, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0]],
            'E': [[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0], [1, 1, 1]],
            'n': [[1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1]],
            'r': [[1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1]],
            'g': [[1, 1, 1], [1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1]],
            'y': [[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]],
            'b': [[1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 1, 0]],
            'w': [[1, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1]],
            'k': [[1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 1]],
            'o': [[0, 0, 0], [0, 0, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0]],
            'p': [[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 0], [1, 0, 0]],
            'c': [[0, 0, 0], [0, 0, 0], [1, 1, 1], [0, 0, 1], [0, 0, 1]],
            'm': [[1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1]],
            'a': [[0, 0, 0], [0, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
            't': [[1, 1, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
            's': [[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
            'u': [[1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]],
            'v': [[1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [0, 1, 0]],
            'l': [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 1]],
            'e': [[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0], [1, 1, 1]],
            'x': [[1, 0, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 0, 1]],
            'f': [[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0], [1, 0, 0]],
            'h': [[1, 0, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1]],
            '(': [[0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0]],
            ')': [[0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0]],
            '.': [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ',': [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0]],
            ':': [[0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ';': [[0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0]]
        }

    def draw_energy(self, energy, board_size, cell_size):
        # Convert the energy value to a string
        energy_str = str(energy)
        # extract the number from "tensor([number])"
        energy_str = energy_str[energy_str.index(
            '(') + 1:energy_str.index(')')]
        # remove the decimal point
        energy_str = energy_str.replace('.', '')
        # add "Energy: " to the string
        energy_str = "Energy:" + energy_str

        # Create a mapping from characters to a grid of cells
        char_to_pixels = {
            '0': [[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]],
            '1': [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
            '2': [[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]],
            '3': [[1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
            '4': [[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]],
            '5': [[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
            '6': [[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1]],
            '7': [[1, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0]],
            '8': [[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]],
            '9': [[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
            ' ': [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            '-': [[0, 0, 0], [0, 0, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0]],
            'E': [[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0], [1, 1, 1]],
            'n': [[1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1]],
            'r': [[1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1]],
            'g': [[1, 1, 1], [1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1]],
            'y': [[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]],
            'b': [[1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 1, 0]],
            'w': [[1, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1]],
            'k': [[1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 1]],
            'o': [[0, 0, 0], [0, 0, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0]],
            'p': [[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 0], [1, 0, 0]],
            'c': [[0, 0, 0], [0, 0, 0], [1, 1, 1], [0, 0, 1], [0, 0, 1]],
            'm': [[1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1]],
            'a': [[0, 0, 0], [0, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
            't': [[1, 1, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
            's': [[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
            'u': [[1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]],
            'v': [[1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [0, 1, 0]],
            'l': [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 1]],
            'e': [[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0], [1, 1, 1]],
            'x': [[1, 0, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 0, 1]],
            'f': [[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0], [1, 0, 0]],
            'h': [[1, 0, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 0, 1]],
            '(': [[0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0]],
            ')': [[0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0]],
            '.': [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ',': [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0]],
            ':': [[0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ';': [[0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0]]
        }
        rightest = 0
        # Draw the energy value on the board
        for i, char in enumerate(energy_str):
            pixels = char_to_pixels[char]
            for j, row in enumerate(pixels):
                for k, pixel in enumerate(row):
                    if not pixel:
                        continue
                    x = self.pos[0] + i * 4 + k
                    y = self.pos[1] + j
                    color = (0, 255, 255) if pixel else (0, 0, 0)
                    pygame.draw.rect(self.screen, color, pygame.Rect(
                        x * cell_size, y * cell_size, cell_size, cell_size))

                    rightest = max(rightest, x * cell_size)
        # update only after some time
        time = pygame.time.get_ticks()
        if time - self.last_time > 1000:
            self.last_time = time
            # Update position for next draw
            self.pos[0] += self.direction[0] * self.speed
            self.pos[1] += self.direction[1] * self.speed

            # if hit right edge change direction to left + up or down
            if rightest == board_size - self.cell_size * 2:
                self.direction = (-1, self.direction[1])
            # if hit left edge change direction to right + up or down
            elif self.pos[0] <= 0:
                self.direction = (1, self.direction[1])
            # if hit top edge change direction to down + left or right
            elif self.pos[1] <= 0:
                self.direction = (random.choice([-1, 1]), 1)
            # if hit bottom edge change direction to up + left or right
            elif self.pos[1] * self.cell_size + 50 >= board_size:
                self.direction = (random.choice([-1, 1]), -1)

    def draw_done(self):

        chars_to_pixels = {
            'D': [[1, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 0]],
            'O': [[0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0]],
            'N': [[1, 0, 0, 0, 1], [1, 1, 0, 0, 1], [1, 0, 1, 0, 1], [1, 0, 0, 1, 1], [1, 0, 0, 0, 1]],
            'E': [[1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]]
        }
        self.done_x_pos += 1
        self.done_x_pos %= self.game.size
        for i, char in enumerate("DONE"):
            pixels = chars_to_pixels[char]
            for j, row in enumerate(pixels):
                for k, pixel in enumerate(row):
                    if not pixel:
                        continue
                    x = i * 6 + k + self.done_x_pos  # Adjusted for the new character size
                    y = j + 10
                    color = (0, 255, 0) if pixel else (0, 0, 0)
                    pygame.draw.rect(self.screen, color, pygame.Rect(
                        x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))

    def handle_events(self):
        board_size = self.game.size
        cell_size = self.screen.get_width() // board_size
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                # Check if the click is on the board
                if 0 <= x <= board_size * cell_size and 0 <= y <= board_size * cell_size:
                    row, col = y // cell_size, x // cell_size
                    # If a queen is already selected, move it to the clicked cell
                    if self.game.missing_queens > 0 and self.game.s[row][col] == 0:
                        self.game.s[row, col] = 1
                        self.game.missing_queens -= 1
                        self.game.energy = self.game.get_energy()
                        break
                    # If the clicked cell contains a queen, select it
                    elif self.game.s[row][col] == 1 and self.game.missing_queens < board_size:
                        self.game.missing_queens += 1
                        self.game.s[row][col] = 0
                        self.game.energy = self.game.get_energy()
                        break

                # Check if the click is on the "Next" button
                if (board_size - 1) * cell_size <= y <= board_size * cell_size and (board_size - 1) * cell_size <= x <= board_size * cell_size:
                    for i, new_state in enumerate(self.game.next_state(self.board)):
                        self.board = new_state
                        if self.draw_board(board=new_state) == 0:
                            self.display_message("Solution found!")
                            break
                        pygame.display.flip()
                        pygame.event.pump()  # Process the Pygame event queue
                # Check if the click is on the "Reset" button
                if (board_size - 2) * cell_size <= y <= (board_size - 1) * cell_size and (board_size - 1) * cell_size <= x <= board_size * cell_size:
                    print("Reset")
                    self.game.reset()
                    self.board = self.game.s
        return True

    def display_message(self, message):
        font = pygame.font.Font(None, 36)
        text = font.render(message, 1, (255, 255, 255))
        textpos = text.get_rect(centerx=self.screen.get_width() / 2)
        self.screen.blit(text, textpos)

    def run(self):
        self.player_turn = True
        running = True
        while running:
            self.clock.tick(60)
            self.draw_board(self.board)
            pygame.display.flip()
            running = self.handle_events()
        pygame.quit()


if __name__ == "__main__":
    gui = GameGUI()

    gui.run()
