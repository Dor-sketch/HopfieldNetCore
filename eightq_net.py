"""
This module contains the implementation of a Hopfield network that solves the 8-queens problem.
"""

import torch
# comenting out the import of Hopfield class
# from hopfield import Hopfield

def Kronecker_delta(i, j):
    return 1 if i == j else 0


ROW_PENALTY = 100
COL_PENALTY = 100
DIAG_PENALTY = 100


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
        state[torch.arange(self.size), indices] = 1
        return state

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
                    print("Q", end=" ")
                else:
                    print(".", end=" ")
            print()
        print()
    def next_state(self, s=None, T=2.0):
        """
        Calculate the next state of the network
        """
        start_energy = self.energy
        if start_energy == 0:
            # print(f'Solution found in {self.external_iterations} ext iterations')
            return self.s
        s = self.s.clone()  # Create a copy of the state to avoid modifying the original state
        iterations = self.size ** 2  * 10
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
                new_col = empty_cols[torch.randint(0, empty_cols.nelement(), (1,)).item()]
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
        s[X, i] = 1 if (field> 0) else 0

    def calculate_energy(self):
        """Calculate the energy of the current state."""
        energy = torch.sum(self.J * self.s.unsqueeze(-1).unsqueeze(-1) * self.s)
        return energy / 2  # Because each interaction is counted twice