"""
This is the main file to run the game.
"""

from tsp_net import TSPNet
from gui import GUI
from tsp_map import Map, CITY_SET_A


class Main:
    """
    use the gui to run the tsp neural network
    """
    def __init__(self):
        road_map = Map(CITY_SET_A)
        self.gui = GUI(len(road_map), TSPNet)

    def run(self):
        self.gui.run()


if __name__ == "__main__":
    main = Main()
    main.run()
