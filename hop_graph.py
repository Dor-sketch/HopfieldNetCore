"""
This module contains the class HopGraph, which is used to display the energy landscape, the weights of the network and the overlap between the current state and the stored patterns.
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt


class HopGraph:
    def __init__(self, hopfield):
        self.hopfield = hopfield
        self.N = hopfield.N

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def plot_three_d(self):
        """
        Display the 3D plot of the energy landscape
        """
        # open new figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        n = self.N
        states = list(itertools.product([-1, 1], repeat=n))

        posibble_states_num = len(states)
        x = np.linspace(0, posibble_states_num, posibble_states_num)
        y = np.linspace(0, posibble_states_num, posibble_states_num)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros((posibble_states_num, posibble_states_num))

        # convert the list of states to a numpy array
        states = np.array(states)
        # compute the energy for each possible state
        for state in states:
            energy = self.hopfield.getEnergy(state)
            Z[state[0] + 1, state[1] + 1] = energy

        # plot the surface
        ax.plot_surface(X, Y, Z, cmap="viridis")
        plt.show()

    def weights(self):
        """
        Display the weights of the network
        """
        # open new figure
        fig, ax = plt.subplots()
        ax.imshow(self.hopfield.weights, cmap="viridis")
        ax.set_title("Weights of the Hopfield Network")
        plt.show()

    def get_overlap(self, stored_pattern):
        """
        Display the overlap between the current state and the stored patterns
        """
        # open dialog with the stored patterns
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 4)
        fig.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_title(
            "Overlap with Stored Patterns",
            fontsize=20,
            color="blue",
            fontweight="bold",
            fontstyle="italic",
            fontfamily="serif",
        )
        ax.set_xlabel("Stored Pattern")
        ax.set_ylabel("Overlap")

        ax.bar(
            range(len(stored_pattern)),
            [
                self.hopfield.overlap_value(np.array(pattern))
                for pattern in stored_pattern
            ],
            color="lightblue" if len(stored_pattern) > 1 else "blue",
        )
        plt.show()

    def view_stored(self, stored):
        """
        Display the stored patterns a new figure with small network graphs
        """
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)
        ax.set_title(
            "Stored Patterns",
            fontsize=20,
            color="lightblue",
            fontweight="bold",
            fontstyle="italic",
            fontfamily="serif",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        print(f'in view_stored: {stored}')
        for i, pattern in enumerate(stored):
            ax = fig.add_subplot(len(stored), 1, i + 1)
            ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"Pattern {i + 1}")
            colors = ["red" if x < 0 else "green" for x in pattern]
            ax.bar(range(len(pattern)), pattern, color=colors)

        plt.tight_layout()
        plt.show()
