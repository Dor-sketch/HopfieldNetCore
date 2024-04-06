"""
This module contains the implementation of a Hopfield network that solves the 8-queens problem.
"""
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from sklearn.manifold import TSNE
import torch
from joblib import Parallel, delayed
from scipy.interpolate import griddata

# comenting out the import of Hopfield class
# from hopfield import Hopfield


def Kronecker_delta(i, j):
    return 1 if i == j else 0


ROW_PENALTY = 100
COL_PENALTY = 100
DIAG_PENALTY = 100
WHITE_Q = '\u2655'
BLACK_Q = '\u265B'


class QueensNet:
    def __init__(self, size=8):
        self.size = size  # Number of queens
        self.N = size ** 2
        self.s = self.get_random_state()
        # self.neurons = self.s.flatten()
        self.J = self.get_synaptic_matrix()
        self.external_iterations = 0
        self.energy = self.get_energy()
        self.missing_queens = 0
        self.n = size  # Number of queens
        # Number of actions is n*n (for an n*n chessboard)
        self.nA = size * size

    def reset(self):
        # print("Resetting the network")
        indices = self.get_random_indices()
        self.s = torch.zeros((self.size, self.size))
        self.s[torch.arange(self.size), indices] = 1
        self.external_iterations = 0
        self.energy = self.get_energy()
        self.missing_queens = 0

    def get_random_state(self):
        """Initialize a random state with exactly one queen per row."""
        state = torch.zeros((self.size, self.size))
        indices = self.get_random_indices()
        self.place_queen(state, indices)
        return state

    def place_queen(self, s, random_idx):
        # reset the state
        s.zero_()
        s[torch.arange(self.size), random_idx] = 1

    def get_random_indices(self):
        """Generate random indices for the state."""
        return torch.randperm(self.size)

    def get_synaptic_matrix(self):
        """Construct a synaptic matrix that penalizes queens threatening each other."""
        indices = torch.arange(self.size)
        i, j, k, l = torch.meshgrid(indices, indices, indices, indices)

        row_penalty = -ROW_PENALTY * (i == k) * (j != l)
        col_penalty = -COL_PENALTY * (i != k) * (j == l)
        diag_penalty = -DIAG_PENALTY * (abs(i - k) == abs(j - l))

        J = row_penalty + col_penalty + diag_penalty

        # Set diagonal elements to 0
        J[(i == k) & (j == l)] = 0

        return J

    def get_energy(self, s=None):
        """Calculate the energy of a state."""
        if s is None:
            s = self.s
        energy = torch.sum(self.J * torch.tensordot(s, s, dims=0))
        return -energy / 2  # Because each interaction is counted twice

    def print_queens(self, s=None):
        """
        Display the queens on the board graphically
        """
        if s is None:
            s = self.s
        for i in range(self.size):
            for j in range(self.size):
                if s[i, j] == 1:
                    print(BLACK_Q, end=" ")
                else:
                    print(".", end=" ")
            print()
        print()

    def get_queens_string(self, s=None):
        """
        Display the queens on the board graphically
        """
        if s is None:
            s = self.s
        queens = ""
        for i in range(self.size):
            for j in range(self.size):
                if s[i, j] == 1:
                    queens += BLACK_Q + "  "
                else:
                    queens += ".  "
            queens += "\n"
        return queens.rstrip()  # Remove trailing whitespace

    def next_state(self, s=None, T=2.0):
        """
        Calculate the next state of the network
        """
        start_energy = self.energy
        if start_energy ==   0:
            # print(f'Solution found in {self.external_iterations} ext iterations')
            return self.s
        s = self.s.clone()  # Create a copy of the state to avoid modifying the original state
        iterations = self.size ** 2 * 10
        #  pre generate random idxs for each iteration
        idxs = torch.randperm(self.size).repeat(self.size)

        # Simulated annealing
        for it in range(min(iterations, len(idxs))):
            # Decrease T gradually
            T = T * 0.99

            # Select a row at random
            if it < len(idxs):
                i = idxs[it]
            else:
                print("Index out of bounds: ", it)
                print("self.size: ", self.size)
                break
            current_col = torch.argmax(s[i])

            # Check if there are any rows or columns without queens
            empty_cols = (s.sum(dim=0) == 0).nonzero(as_tuple=True)[0]
            if empty_cols.nelement() > 0:
                # Move the queen to an empty column
                new_col = empty_cols[torch.randint(
                    0, empty_cols.nelement(), (1,)).item()]
            else:
                # Try moving the queen to a random column
                new_col = torch.randint(0, self.size, (1,)).item()

            s[i, current_col].zero_()  # In-place zero
            s[i, new_col].fill_(1)  # In-place fill
            new_energy = self.get_energy(s)

            # If the new state has lower energy, accept it
            # Otherwise, accept it with a probability that decreases with the energy difference and the temperature
            if new_energy < start_energy or torch.rand(1).item() < torch.exp((start_energy - new_energy) / T):
                start_energy = new_energy
                self.energy = new_energy
            else:
                # Move the queen back to the original column
                s[i, new_col].zero_()  # In-place zero
                s[i, current_col].fill_(1)  # In-place fill

            if start_energy == 0:
                yield s
                break
            yield s

        self.external_iterations += 1
        # self.neurons.copy_(s.flatten())  # In-place copy
        self.s.copy_(s)  # In-place copy
        # self.print_queens(s)
        return s

    def print_synaptic_matrix(self, J):
        with open('synaptic_matrix.txt', 'w') as f:
            for i in range(self.size):
                for j in range(self.size):
                    for k in range(self.size):
                        for l in range(self.size):
                            f.write(
                                f"J({i},{j}),({k},{l}) = {J[i, j, k, l]}\n")

    def update_state(self, X, i, s):
        """
        Update the state of the network
        """
        field = torch.sum(self.J[X, i] * s)
        s[X, i] = 1 if (field > 0) else 0

    def calculate_energy(self):
        """Calculate the energy of the current state."""
        energy = torch.sum(
            self.J * self.s.unsqueeze(-1).unsqueeze(-1) * self.s)
        return energy / 2  # Because each interaction is counted twice





    def print_queens_energy_3d(self, s=None, sample_size = 10000):
        if s is None:
            s = self.s
        x_size = int(np.sqrt(sample_size))

        # Generate all possible states for a single queen using numpy
        sampled_states = [self.get_random_indices() for _ in range(x_size * x_size)]

        # Convert the list of states to a numpy array
        states_array = np.array(sampled_states)

        # Use t-SNE to reduce the dimensionality to 2
        tsne = TSNE(n_components=2, perplexity=50, learning_rate=100, n_iter=2000, random_state=42)
        states_2d = tsne.fit_transform(states_array)

        # Now, states_2d is a 2D representation of your states
        grid_x, grid_y = states_2d[:, 0], states_2d[:, 1]

        # Reshape grid_x and grid_y to match the shape of grid_z
        grid_x = grid_x.reshape((x_size, x_size))
        grid_y = grid_y.reshape((x_size, x_size))
        grid_z = np.zeros((x_size, x_size))

        # List to store states with zero energy
        self.zero_energy_states = []

        # Calculate the energy for the sampled states in parallel
        energies = Parallel(n_jobs=-1)(delayed(lambda state: self.get_energy(self.place_queen(s, state)))(state) for state in sampled_states)
        for i, energy in enumerate(energies):
            # Store the energy in grid_z
            grid_z[i // x_size, i % x_size] = energy

            # If the energy is zero, store the state
            if energy == 0:
                self.zero_energy_states.append(sampled_states[i])

        # Create a grid of points where you want to estimate z-values
        grid_x_new, grid_y_new = np.mgrid[grid_x.min():grid_x.max():100j, grid_y.min():grid_y.max():100j]

        # Interpolate z-values for the new grid points
        grid_z_new = griddata((grid_x.flatten(), grid_y.flatten()), grid_z.flatten(), (grid_x_new, grid_y_new), method='cubic')


        fig = plt.figure(figsize=(18, 12))  # Adjust the size as needed
        plt.axis('off')

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        fig.patch.set_facecolor('black')
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor('black')

        norm = Normalize(grid_z.min(), grid_z.max())
        colors = cm.magma(norm(grid_z.flatten()))
        colors_new = cm.magma(norm(grid_z_new.flatten()))
        colors[grid_z.flatten() == 0] = [1, 0, 0, 1]  # RGBA for red
        ax.plot_surface(grid_x_new, grid_y_new, grid_z_new, facecolors=colors_new.reshape(grid_x_new.shape + (4,)), shade=False)


        scatter = ax.scatter(grid_x.flatten(), grid_y.flatten(), grid_z.flatten(
        ), c=colors, s=10, picker=5)  # s is the size of the points

        def on_pick(event):
            ind = event.ind[0]  # Get the first index from the event
            print(
                f"Clicked on ({grid_x.flatten()[ind]}, {grid_y.flatten()[ind]})")
            print(f"State: {sampled_states[ind]}")
            print(f"Energy: {grid_z.flatten()[ind]}")
            self.place_queen(self.s, sampled_states[ind])

            # Remove previous annotations
            for txt in ax.texts:
                txt.set_visible(False)

            # Display the board state in a text box
            anchored_text = AnchoredText(
                self.get_queens_string(),
                loc='upper right',
                prop=dict(
                    backgroundcolor='black',
                    color='white',
                    size=15,
                    weight='bold'
                )
            )
            ax.add_artist(anchored_text)
            plt.draw()

        fig.canvas.mpl_connect('pick_event', on_pick)
        plt.show()


# q = QueensNet()
# q.print_queens_energy_3d()
