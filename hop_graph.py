"""
This module contains the class HopGraph, which is used to display the energy landscape, the weights of the network and the overlap between the current state and the stored patterns.
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from hop_proof import subscript
from scipy.interpolate import griddata


class HopGraph:
    def __init__(self, hopfield):
        self.hopfield = hopfield
        self.N = hopfield.N

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def plot_two_d(self, stored_pattern):
        """
        Display the 3D plot of the energy landscape but in 2D
        This means 3d plot with only 2 neurons

        """
        # open new figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        n = self.N
        states = list(itertools.product([-1, 1], repeat=n))
        states = np.array(states[:4])

        posibble_states_num = len(states)
        energy = np.zeros(posibble_states_num)

        for i, state in enumerate(states):
            energy[i] = self.hopfield.getEnergy(np.array(state))

        # cut only the first 4 states

        x = np.arange(-1, 2, 2)
        y = np.arange(-1, 2, 2)
        x, y = np.meshgrid(x, y)
        z = energy.reshape((x.shape[0], x.shape[1]))

        ax.plot_surface(x, y, z, cmap="viridis")
        ax.set_xlabel(f"s{subscript[2]}")  # 2 = 1 its zero indexed
        ax.set_ylabel(f"s{subscript[3]}")

        # only ticks at 0 and 1 note that 0 is -1 and 1 is 1
        ax.set_xticks([-1, 1])
        ax.set_yticks([-1, 1])
        ax.set_zticks([min(energy), max(energy)])

        # add red dots for the possible states
        for i, state in enumerate(states):
            ax.scatter(state[0], state[1], energy[i], color="red")
        ax.set_zlabel("E(s)")
        ax.set_title("Energy Landscape")

        plt.show()

    def plot_three_d(self, stored_pattern):

        # Number of neurons
        n = self.N

        # Generate all possible states
        states = np.array(
            [
                [1 if x == "1" else -1 for x in format(i, "0" + str(n) + "b")]
                for i in range(2**n)
            ]
        )
        # Calculate the energy for each state
        energies = np.array([self.hopfield.getEnergy(state) for state in states])
        is_stored = np.array(
            [
                any(np.array_equal(stored, state) for stored in stored_pattern)
                for state in states
            ]
        )
        is_false_mem = np.array(
            [
                any(np.array_equal(-stored, state) for stored in stored_pattern)
                for state in states
            ]
        )
        # mark false mem
        original_states = states.copy()
        states = states.astype(float)  # Convert states to float
        print(is_stored)

        states[:, :2] += np.random.normal(scale=0.5, size=states[:, :2].shape)

        # Create a grid of points
        # Create a grid of points
        min_idices = np.min(states[:, :2])
        max_idices = np.max(states[:, :2])
        grid_x, grid_y = np.mgrid[min_idices:max_idices:50j, min_idices:max_idices:50j]
        print(states[:, :2])
        print(energies)
        print((grid_x, grid_y))

        # Interpolate the energy for each point on the grid
        grid_z = griddata(states[:, :2], energies, (grid_x, grid_y), method="cubic")
        grid_z = np.nan_to_num(grid_z, nan=-1.0)

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot the energy as a continuous surface

        # Normalize to [0,1]
        norm = plt.Normalize(grid_z.min(), grid_z.max())
        colors = cm.viridis(norm(grid_z))
        rcount, ccount, _ = colors.shape

        ax.plot_surface(
            grid_x,
            grid_y,
            grid_z,
            rcount=rcount,
            ccount=ccount,
            facecolors=colors,
            shade=False,
        )
        ax.set_facecolor((0, 0, 0, 0))

        # Plot the stored patterns
        for i, state in enumerate(states):
            if is_stored[i]:
                if np.NaN == energies[i]:
                    energies[i] = 0
                ax.text(
                    state[0],
                    state[1],
                    energies[i],
                    f"s = {original_states[i]}\nE = {energies[i]:.2f}",
                    color="green",
                )
            if is_false_mem[i]:
                if np.NaN == energies[i]:
                    energies[i] = 0
                ax.text(
                    state[0],
                    state[1],
                    energies[i],
                    f"s = {original_states[i]}\nE = {energies[i]:.2f}",
                    color="red",
                )

        ax.set_zlabel("Energy")
        ax.set_xlabel(f"neuron 0")
        ax.set_ylabel(f"nueron 1")
        ax.set_title("Energy Landscape")
        min_val = np.min(states)
        max_val = np.max(states)

        ax.set_xticks([min_val, max_val])
        ax.set_yticks([min_val, max_val])
        ax.set_xticklabels(["off", "on"])
        ax.set_yticklabels(["off", "on"])

        # Show the plot
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
        print(f"in view_stored: {stored}")
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
