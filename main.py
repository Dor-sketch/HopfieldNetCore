"""
This is the main file to run the game.
"""

from tsp_examp import TSPNet
from gui import GUI

NUMBER_OF_NEURONS = 9


class Main:
    """
    use the gui to run the tsp neural network
    """
    def __init__(self):
        self.gui = GUI(TSPNet, NUMBER_OF_NEURONS)
        self.tsp_net = TSPNet()

    def run(self):
        self.gui.run()


if __name__ == "__main__":
    main = Main()
    main.run()
