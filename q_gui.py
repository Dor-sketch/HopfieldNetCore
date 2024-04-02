"""
This file contains the code for the 3D-like Tic-Tac-Toe game using Pygame.
"""
import pygame
import pygame.gfxdraw
import numpy as np
import pyautogui
from eightq_net import QueensNet

class GameGUI:
    """
    This class is responsible for the graphical user interface of the Tic-Tac-Toe game.
    """

    def __init__(self):
        pygame.init()
        self.game = QueensNet(size=8)
        self.board_size = 1200  # Size of the board in pixels
        self.cell_size = self.board_size // self.game.size  # Size of each cell in pixels
        self.screen = pygame.display.set_mode(
            (self.board_size, self.board_size))
        self.clock = pygame.time.Clock()
        self.board = self.game.s
        pygame.display.set_caption(f'{self.game.size} Queens Problem')
        self.gradients = []
        for i in range(2):
            base_color = np.array(
                [0, 250, 0]) if i % 2 == 0 else np.array([115, 115, 116])
            gradient = np.zeros(
                (self.cell_size, self.cell_size, 3), dtype=np.uint8)
            for k in range(self.cell_size):
                for l in range(self.cell_size):
                    # Calculate distance from the top-left corner
                    dist_to_corner = np.sqrt(k ** 2 + l ** 2)
                    color = base_color + dist_to_corner * \
                        (255 - base_color) / (np.sqrt(2) * self.cell_size)
                    gradient[k, l, :] = color
            self.gradients.append(pygame.surfarray.make_surface(gradient))

    def draw_board(self, board=None):
        if board is None:
            board = self.game.s
        for i in range(self.game.size):
            for j in range(self.game.size):
                # Use the pre-calculated gradient
                self.screen.blit(
                    self.gradients[(i + j) % 2], (j * self.cell_size, i * self.cell_size))
                # # Add a 3D effect to the squares
                pygame.gfxdraw.box(self.screen, pygame.Rect(
                    j * self.cell_size + 5, i * self.cell_size + 5, self.cell_size - 10, self.cell_size - 10), (0, 0, 250, 50))
        for x in range(self.game.size):
            for y in range(self.game.size):
                center = (y * self.cell_size + self.cell_size // 2,
                          x * self.cell_size + self.cell_size // 2)
                if board[x, y] == 1:
                    # Check if this queen threatens any other
                    threatens = any(
                        board[i, y] == 1 for i in range(self.game.size) if i != x
                    ) or any(
                        board[x, j] == 1 for j in range(self.game.size) if j != y
                    ) or any(
                        board[i, i + y - x] == 1 for i in range(max(0, x - y), min(self.game.size, self.game.size + x - y)) if i != x
                    ) or any(
                        board[i, i + x + y] == 1 for i in range(max(0, -x - y), min(self.game.size, self.game.size - x - y)) if i != x
                    )
                    # Choose the color based on whether this queen threatens any other
                    color = (155, 0, 0) if threatens else (0, 155, 0)
                    color = (0, 0, 155) if threatens else (0, 0, 155)
                    q_size = self.cell_size // 2
                    for offset in range(0, q_size, q_size // 6):
                        highlight_radius = q_size - offset
                        gradient_color = (
                            color[0] + (255 - color[0]) * offset // (q_size),
                            color[1] + (255 - color[1]) * offset // (q_size),
                            color[2]
                        )
                        alpha = 255 - 128 + 128 * offset // (q_size)
                        pygame.gfxdraw.aacircle(self.screen, center[0], center[1] - offset, highlight_radius, (
                            gradient_color[0], gradient_color[1], gradient_color[2], alpha))
                        if offset < q_size - self.cell_size // 6:
                            pygame.gfxdraw.filled_circle(
                                self.screen, center[0], center[1] - offset, highlight_radius, (gradient_color[0], gradient_color[1], gradient_color[2], alpha))
                        else:
                            # make a crown
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
                                    y1), int(x2), int(y2), int(x3), int(y3), color)
                                pygame.gfxdraw.filled_trigon(self.screen, int(x1), int(
                                    y1), int(x2), int(y2), int(x3), int(y3), color)

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

        # Draw the energy on top of the board
        font = pygame.font.Font(None, 36)
        energy = self.game.get_energy(board)
        text = font.render(f"Energy: {energy}", 1, (0, 0, 0))
        textpos = text.get_rect(
            centerx=self.screen.get_width() / 2, centery=self.cell_size // 2)
        self.screen.blit(text, textpos)
        if energy == 0:
            # capture screen if solution found - used for GIF creation
            # img = pyautogui.screenshot(
            #     region=(0, 30, self.board_size, self.board_size))
            # img.save(f"screenshot_99998.png")
            # img.save(f"screenshot_99999.png")
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
                        # if self.game.get_energy(new_state) != self.game.get_energy(self.board):
                        #     # capture screen for each step - used for GIF creation
                        #     img = pyautogui.screenshot(
                        #         region=(0, 30, self.board_size, self.board_size))
                        #     img.save(f"screenshot_{str(i).zfill(5)}.png")
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
