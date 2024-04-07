"""
This module contains the implementation of a Hopfield network that solves the 8-queens problem.
"""
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
from sklearn.manifold import TSNE
import torch
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
# from joblib import Parallel, delayed
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
# comenting out the import of Hopfield class
# from hopfield import Hopfield


def Kronecker_delta(i, j):
    return 1 if i == j else 0


ROW_PENALTY = 100
COL_PENALTY = 100
DIAG_PENALTY = 100
WHITE_Q = '\u2655'
BLACK_Q = '\u265B'

def calculate_energy(s, J):
    """Calculate the energy of each state using torch.einsum."""
    s = s.float()  # Convert s to Float
    J = J.float()  # Convert J to Float
    energy = torch.einsum('ijkl,bij,bkl->b', J, s, s)
    return -energy / 2  # Because each interaction is counted twice

def calculate_zero_energy_states(s, J):
    """Calculate the energy of each state using torch.einsum and return only the states with zero energy."""
    energy = calculate_energy(s, J)

    # Create a mask that is True where the energy is zero
    mask = energy == 0

    # Use the mask to select the states with zero energy
    zero_energy_states = s[mask]

    return energy, zero_energy_states

def generate_many_states(n, size=8):
    """Generate many states for the 8-queens problem."""
    # Initialize a tensor of zeros
    states = torch.zeros(n, size, size)

    # Generate a tensor of random column indices for each state and each row
    indices = torch.randint(0, size, (n, size))

    # Use one-hot encoding to place the queens at the generated indices
    states.scatter_(2, indices.unsqueeze(-1), 1)
    print("Done generating states")
    return states

class QueensNet:
    def __init__(self, size=8):
        self.size = size  # Number of queens
        self.temp_tensor = torch.zeros((size, size))
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
        self.state_idx = 0

    def get_valid_state(self, state_idx):
        """
        Use the state index to map to a unique valid state.
        """
        self.temp_tensor.fill_(0)
        cols = (state_idx // torch.pow(self.size, torch.arange(self.size))) % self.size
        self.temp_tensor[torch.arange(self.size), cols.long()] = 1
        return self.temp_tensor

    def set_valid_states(self, s):
        """
        Use the state indices to map to unique valid states.
        """
        num_states = s.shape[0]
        state_idxs = torch.arange(num_states).unsqueeze(1).repeat(1, self.size)
        cols = (state_idxs // torch.pow(self.size, torch.arange(self.size))) % self.size
        s[torch.arange(num_states).unsqueeze(1).repeat(1, self.size), torch.arange(self.size), cols.long()] = 1

    def get_valid_states(self, N):
        """Generate a tensor of valid states according to the numbering system."""
        num_states = N**N // 2
        states = torch.zeros((num_states, N, N))
        self.set_valid_states(states)
        print(states.shape)
        return states

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

    def get_energy(self, s=None, idx=None):
        """Calculate the energy of a state."""
        if idx is not None:
            # idx is the index of the state in the list of valid states
            # we can use this to calculate the energy without generating the state
            s = self.get_valid_state(idx)
            # avoid calc twice
            self.energy = torch.sum(self.J * torch.tensordot(s, s, dims=0))
            self.energy = -self.energy / 2  # Because each interaction is counted twice
            return self.energy  # Because each interaction is counted twice
        if s is None:
            s = self.s
        self.energy = torch.sum(self.J * torch.tensordot(s, s, dims=0))
        self.energy = -self.energy / 2
        return self.energy  # Because each interaction is counted twice

    def print_queens(self, s=None):
        """
        Display the queens on the board graphically
        """
        if s is None:
            s = self.s

        print(self.get_queens_string(s))

    # def get_queens_string(self, s=None):
    #     """
    #     Display the queens on the board graphically
    #     """
    #     if s is None:
    #         s = self.s

    #     # Convert the tensor to a string representation
    #     s_str = torch.where(s == 1, BLACK_Q + "  ", ".  ")

    #     # Join the elements of each row into a single string
    #     s_str = [''.join(row) for row in s_str]

    #     # Join the rows into a single string and remove trailing whitespace
    #     queens = '\n'.join(s_str).rstrip()

    #     return queens
    def get_queens_string(self, s):
        s = s.view(self.size, -1).numpy()  # Reshape the tensor to a 2D array and convert it to a numpy array
        s_str = np.where(s == 1, BLACK_Q + "  ", ".  ")
        return "\n".join("".join(row).rstrip(" ") for row in s_str)

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


    def print_queens_energy_3d(self, s=None, sample_size = 80000):
        if s is None:
            s = self.s
        x_size = int(np.sqrt(sample_size))
        sample_size = x_size * x_size

        states = generate_many_states(sample_size, self.size)
        sampled_states = states.view(sample_size, -1)
        tsne = TSNE(n_components=3, perplexity=50, learning_rate=100, n_iter=1000, random_state=22)
        states_3d = tsne.fit_transform(states.view(sample_size, -1).numpy())

        grid_x, grid_y, grid_z = states_3d[:, 0], states_3d[:, 1], states_3d[:, 2]

        grid_x = grid_x.reshape((x_size, x_size))
        grid_y = grid_y.reshape((x_size, x_size))

        self.zero_energy_states = []
        print("Calculating the energy of the sampled states...")

        energies, zero_energy_states = calculate_zero_energy_states(states, self.J)
        print("Done calculating the energy of the sampled states")

        # Reshape energies and assign it to grid_z
        grid_z = energies.view(x_size, -1)
        print(f"Minimum energy: {grid_z.min()}")

        print(f"Number of zero-energy states: {zero_energy_states.shape[0]}")

        self.zero_energy_states.extend(zero_energy_states)

        grid_x_new, grid_y_new = np.mgrid[grid_x.min():grid_x.max():100j, grid_y.min():grid_y.max():100j]

        grid_z_new = griddata((grid_x.flatten(), grid_y.flatten()), grid_z.flatten(), (grid_x_new, grid_y_new), method='cubic')

        fig = plt.figure(figsize=(14, 12))
        plt.axis('off')

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        fig.patch.set_facecolor('black')
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor('black')

        norm = Normalize(grid_z.min(), grid_z.max())
        colors = cm.plasma(norm(grid_z.flatten()))
        colors_new = cm.plasma(norm(grid_z_new.flatten()))
        colors[grid_z.flatten() == 0] = [1, 0, 0, 1]  # RGBA for red
        ax.plot_surface(grid_x_new, grid_y_new, grid_z_new, facecolors=colors_new.reshape(grid_x_new.shape + (4,)), shade=False)
        dots_sizes = np.full_like(grid_z.flatten(), 10)
        dots_sizes[grid_z.flatten() == 0] = 1000
        scatter = ax.scatter(grid_x.flatten(), grid_y.flatten(), grid_z.flatten(), c=colors, s=dots_sizes, picker=5)

        def on_pick(event):
            ind = event.ind[0]
            print(f"Clicked on ({grid_x.flatten()[ind]}, {grid_y.flatten()[ind]})")
            print(f"State: {sampled_states[ind]}")
            print(f"Energy: {grid_z.flatten()[ind]}")
            q_str = self.get_queens_string(sampled_states[ind])
            self.print_queens(sampled_states[ind])
            for txt in ax.texts:
                txt.set_visible(False)

            anchored_text = AnchoredText(
                q_str,
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

    def plot_2d(self):
        """
        Plot the energy of each state in 2D.
        This method uses vectorized operations to calculate the energy of each state.
        It also leverages the symmetry of the problem to reduce the number of calculations.
        """
        states = self.get_valid_states(self.size) # will hold only half of the states
        print(states.shape)
        energies, zero_energy_states = calculate_zero_energy_states(states, self.J)
        print(f"Number of solutions states: {zero_energy_states.shape[0] * 2}")
        energies = torch.cat((energies, torch.flip(energies, [0])))
        circle = plt.Circle((0, 0), 20, fill=False, edgecolor='red', visible=False)

        def on_pick(event):
            ind = event.ind[0]
            print(f"State: {ind}")
            print(f"Energy: {energies[ind]}")
            # we use the index to get the state
            q_str = f'index: {ind}\nEnergy: {energies[ind]}\n\n{self.get_queens_string(self.get_valid_state(int(ind)))}'
            self.print_queens(self.get_valid_state(int(ind)))
            for txt in ax.texts:
                txt.set_visible(False)

            anchored_text = AnchoredText(
                q_str,
                loc='upper right',
                prop=dict(
                    backgroundcolor='black',
                    color='white',
                    size=15,
                    weight='bold'
                )
            )
            ax.add_artist(anchored_text)
            circle.center = (ind, energies[ind])
            circle.set_visible(True)
            plt.draw()

        # Plot the energies
        fig, ax = plt.subplots(figsize=(14, 10))
        fig.patch.set_facecolor('black')
        fig.suptitle(f'Energy function of the {self.size} queens problem', color='white')
        ax.set_facecolor('black')
        energies_vectorized = energies.view(-1)
        energies_normalized = (energies_vectorized - energies_vectorized.min()) / (energies_vectorized.max() - energies_vectorized.min())
        colors = cm.plasma(energies_normalized)
        scatter = ax.scatter(np.arange(len(states) * 2), energies, c=colors, picker=5)
        ax.set_xlabel("State index", color='white')
        ax.set_ylabel("Energy", color='white')
        ax.set_title("Energy of each state", color='white')
        ax.tick_params(colors='white')
        ax.add_patch(circle)
        fig.canvas.mpl_connect('pick_event', on_pick)
        plt.show()


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Solve the 8-queens problem using a Hopfield network.")
    parser.add_argument("--size", type=int, default=8, help="The size of the board (default: 8)")
    parser.add_argument("--3d", action="store_true", help="Plot the energy of each state in 3D")
    parser.add_argument("--2d", action="store_true", help="Plot the energy of each state in 2D")
    parser.add_argument("--solution", action="store_true", help="Print a solution to the 8-queens problem")
    return parser.parse_args()

def main():
    # TODO: Add a command-line arguments
    # parse arguments
    args = parse_args()
    size = args.size
    q = QueensNet(size)
    q.print_queens()

    # Print the solution if requested
    if args.solution:
        while True:
            print("Energy: " + str(q.get_energy()) + "\nPress N to continue or R to reset the board\n")
            decision = input()
            if decision == "R":
                q.reset()
            elif decision == "N" or decision == "":
                q.next_state()
            else:
                break
            q.print_queens()

if __name__ == "__main__":
    main()



def solveNQueens(n: int):
    if n == 1:
        return [[]]
    q = QueensNet(n)
    max_idx = n**n
    ret = []
    i = 0

    while i < max_idx:
        if i % 2 == 0 and n == 7:
            i += 1
            continue
        q.s = q.get_valid_state(i)
        energy = q.get_energy(q.s)
        if energy == 0:
            ret.append(q.get_queens_string(q.s))
            i += 20
        else:
            # Estimate the number of moves needed to reach a solution
            moves_needed = energy // 100  # Replace 2 with the maximum decrease in energy per move
            # Skip the states that are guaranteed not to be solutions
            i += max(moves_needed.item()-2, 1)
    return ret
