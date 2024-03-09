"""
This is the main file to run the game.
"""

from gui import GUI

NUMBER_OF_NEURONS = 9


if __name__ == "__main__":
    gui = GUI(NUMBER_OF_NEURONS)
    gui.run()
