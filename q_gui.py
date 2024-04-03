"""
This file contains the code for the 3D-like Tic-Tac-Toe game using Pygame.
"""
import pygame
import pygame.gfxdraw
import numpy as np
import pyautogui
from eightq_net import QueensNet
import torch


class GameGUI:
    """
    This class is responsible for the graphical user interface of the Tic-Tac-Toe game.
    """

    def __init__(self):
        pygame.init()
        self.game = QueensNet(size=64)
        self.board_size = 1224  # Size of the board in pixels
        self.cell_size = self.board_size // self.game.size  # Size of each cell in pixels
        self.screen = pygame.display.set_mode(
            (self.board_size, self.board_size))
        self.clock = pygame.time.Clock()
        self.board = self.game.s.detach()
        # Create a grid of indices
        self.indices = None
        self.diag_indices = None
        self.init_indices()
        pygame.display.set_caption(f'{self.game.size} Queens Problem')
        self.gradients = []
        for i in range(2):
            base_color = np.array(
                [40, 40, 40]) if i % 2 == 0 else np.array([199, 190, 190])
            gradient = np.zeros(
                (self.cell_size, self.cell_size, 3), dtype=np.uint8)
            for k in range(self.cell_size):
                for l in range(self.cell_size):
                    # Calculate distance from the top-left corner
                    dist_to_corner = np.sqrt(k ** 2 + l ** 2)
                    color = base_color + dist_to_corner * \
                        np.array([0.5, 0.5, 0.5]) / self.cell_size \
                        if dist_to_corner < self.cell_size else base_color
                    gradient[k, l, :] = color
            self.gradients.append(pygame.surfarray.make_surface(gradient))

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
        half_cell_size = self.cell_size // 2
        sixth_cell_size = self.cell_size // 6
        eight_cell_size = self.cell_size // 8

        if board is None:
            board = self.game.s
        for i in range(self.game.size):
            for j in range(self.game.size):
                self.screen.blit(
                    self.gradients[(i + j) % 2], (j * self.cell_size, i * self.cell_size))
                rect_size = self.cell_size - 2 * eight_cell_size  # Size of the rectangle
                rect_x = j * self.cell_size + eight_cell_size  # X position of the rectangle
                rect_y = i * self.cell_size + eight_cell_size  # Y position of the rectangle

                pygame.gfxdraw.box(self.screen, pygame.Rect(
                    rect_x, rect_y, rect_size, rect_size), (0, 0, 100, 10))
        for index in self.indices.view(-1, 2):
            x, y = index
            center = (y * self.cell_size + half_cell_size,
                      x * self.cell_size + half_cell_size)
            if board[x, y] == 1:
                color = (150, 150, 250)  # Bright white color
                q_size = self.cell_size + 2 * sixth_cell_size
                for offset in range(0, q_size, sixth_cell_size//3):  # Increase the range of offset
                    highlight_radius = max (
                        half_cell_size - 2, half_cell_size - offset)
                    gradient_color = (
                        max(0, color[0] - 1 * offset),  # Decrease the color change rate
                        max(0, color[1] - 1 * offset),
                        max(0, color[2] - 1 * offset))
                    alpha = 255
                    pygame.gfxdraw.aacircle(self.screen, center[0], center[1] - offset, highlight_radius, (
                        gradient_color[0], gradient_color[1], gradient_color[2], alpha))
                    if offset < q_size - sixth_cell_size // 3:
                        pygame.gfxdraw.filled_circle(
                            self.screen, center[0], center[1] - offset, highlight_radius, (gradient_color[0], gradient_color[1], gradient_color[2], alpha))
                        # add shadow
                        pygame.gfxdraw.aaellipse(self.screen, center[0], center[1] - offset + 2, highlight_radius - 2, highlight_radius - 2, (
                            0, 0, 0, 250))
                        # add outline
                        pygame.gfxdraw.aaellipse(self.screen, center[0], center[1] - offset, highlight_radius, highlight_radius, (
                            0, 0, 255, 255))
                    else:
                        # make a crown
                        alpha = 255
                        for angle in range(0, 360, 30):
                            # Calculate the coordinates of the triangle
                            x1 = center[0] + highlight_radius * \
                                np.cos(np.radians(angle))
                            y1 = center[1] - offset + \
                                highlight_radius * \
                                np.sin(np.radians(angle))
                            x2 = center[0] + highlight_radius * \
                                np.cos(np.radians(angle + 30))
                            y2 = center[1] - offset + highlight_radius * \
                                np.sin(np.radians(angle + 30))
                            x3 = center[0] + (highlight_radius + self.cell_size //
                                              10) * np.cos(np.radians(angle + 15))
                            y3 = center[1] - offset + (
                                highlight_radius + self.cell_size // 10) * np.sin(np.radians(angle + 15))
                            # Draw the triangle
                            pygame.gfxdraw.aatrigon(self.screen, int(x1), int(
                                y1), int(x2), int(y2), int(x3), int(y3), gradient_color)  # Use gradient color for the crown
                            pygame.gfxdraw.filled_trigon(self.screen, int(x1), int(
                                y1), int(x2), int(y2), int(x3), int(y3), gradient_color)  # Use gradient color for the crown
        # Define a semi-transparent surface for the buttons
        button_surface = pygame.Surface(
            (self.cell_size, self.cell_size), pygame.SRCALPHA)
        # RGBA color, 128 for 50% transparency
        button_surface.fill((0, 0, 0, 128))

        # Draw the "Next" button at the bottom right corner
        font = pygame.font.Font(None, 36)
        text = font.render("Next", 1, (255, 255, 255))
        textpos = text.get_rect(centerx=self.cell_size //
                                2, centery=self.cell_size // 2)
        button_surface.blit(text, textpos)
        self.screen.blit(button_surface, (self.board_size -
                         self.cell_size, self.board_size - self.cell_size))

        # Draw the "Reset" button at the bottom right corner
        # Clear the surface for the next button
        button_surface.fill((0, 0, 0, 128))
        text = font.render("Reset", 1, (255, 255, 255))
        textpos = text.get_rect(centerx=self.cell_size //
                                2, centery=self.cell_size // 2)
        button_surface.blit(text, textpos)
        self.screen.blit(button_surface, (self.board_size -
                         self.cell_size, self.board_size - 2 * self.cell_size))
        # Load a custom font and set the size

        energy = self.game.get_energy(board)
        text = font.render(f"Energy: {energy}", 1, (0, 0, 0))
        textpos = text.get_rect(centerx=self.screen.get_width() / 2, centery=self.cell_size // 2)

        # Create a semi-transparent surface
        background = pygame.Surface((textpos.width, textpos.height), pygame.SRCALPHA)

        # Fill the surface with a color and alpha value
        background_color = (250, 150, 155,120)  # Semi-transparent red
        background.fill(background_color)

        # Draw the semi-transparent rectangle on the screen
        self.screen.blit(background, textpos)

        # Draw the text on top of the rectangle
        self.screen.blit(text, textpos)
        if energy == 0:
            # capture screen if solution found - used for GIF creation
            img = pyautogui.screenshot(
                region=(0, 30, self.board_size, self.board_size))
            img.save(f"screenshot_99998.png")
            img.save(f"screenshot_99999.png")
            return 0

    def handle_events(self):
        board_size = self.game.size
        cell_size = self.screen.get_width() // board_size
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                # Check if the click is on the "Next" button
                if (board_size - 1) * cell_size <= y <= board_size * cell_size and (board_size - 1) * cell_size <= x <= board_size * cell_size:
                    for i, new_state in enumerate(self.game.next_state(self.board)):
                        if self.game.get_energy(new_state) != self.game.get_energy(self.board):
                            # capture screen for each step - used for GIF creation
                            img = pyautogui.screenshot(
                                region=(0, 30, self.board_size, self.board_size))
                            img.save(f"screenshot_{str(i).zfill(5)}.png")
                        self.board = new_state
                        if self.draw_board(board=new_state) == 0:
                            self.display_message("Solution found!")
                            img = pyautogui.screenshot(
                                region=(0, 30, self.board_size, self.board_size))
                            img.save(f"screenshot_{str(i).zfill(5)}.png")
                            break
                        pygame.display.flip()
                        pygame.time.wait(10)  # Wait for 500 milliseconds
                        pygame.event.pump()  # Process the Pygame event queue
                    self.display_message(
                        f"Network energy: {self.game.get_energy(self.board)}")
                # Check if the click is on the "Reset" button
                elif (board_size - 2) * cell_size <= y <= (board_size - 1) * cell_size and (board_size - 1) * cell_size <= x <= board_size * cell_size:
                    self.game.reset()
                    self.board = self.game.s
                else:
                    self.display_message("Invalid move!")
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
