import numpy as np
from hopfield import Hopfield

def Kronecker_delta(i, j):
    return 1 if i == j else 0


ROW_PENALTY = 100
COL_PENALTY = 100
DIAG_PENALTY = 100


class QueensNet(Hopfield):
    def __init__(self, size=8):
        self.size = size  # Number of queens
        self.N = size ** 2  # Number of neurons, +1 for the bias neuron
        self.s = self.get_random_state()
        self.neurons = self.s.flatten()
        self.J = self.get_synaptic_matrix()
        self.external_iterations = 0

    def reset(self):
        self.s = self.get_random_state()
        self.neurons = self.s.flatten()
        self.external_iterations = 0

    def get_random_state(self):
        """Initialize a random state with exactly one queen per row."""
        state = np.zeros((self.size, self.size))
        for i in range(self.size):
            state[i, np.random.randint(self.size)] = 1
        return state

    def get_synaptic_matrix(self):
        """Construct a synaptic matrix that penalizes queens threatening each other."""
        J = np.zeros((self.size, self.size, self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    for l in range(self.size):
                        if i != k or j != l:  # Skip the same queen
                            J[i, j, k, l] = (
                                -ROW_PENALTY *
                                Kronecker_delta(i, k) *
                                (1 - Kronecker_delta(j, l))
                                - COL_PENALTY *
                                (1 - Kronecker_delta(i, k)) *
                                Kronecker_delta(j, l)
                                - DIAG_PENALTY *
                                Kronecker_delta(abs(i - k), abs(j - l))
                            )
        return J

    def get_energy(self, s=None):
        """Calculate the energy of a state."""
        if s is None:
            s = self.s
        energy = np.sum(self.J * np.tensordot(s, s, axes=0))
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


    def next_state(self, s=None, T=1.0):
        """
        Calculate the next state of the network
        """
        start_energy = self.get_energy()
        if start_energy == 0:
            print(f'Solution found in {self.external_iterations} ext iterations')
            return self.s
        s = self.s.copy()  # Create a copy of the state to avoid modifying the original state
        iterations = self.size ** 2 * 5

        for it in range(iterations):
            # Select a row at random
            i = np.random.randint(self.size)
            current_col = np.argmax(s[i])

            # Try moving the queen to a random column
            new_col = np.random.randint(self.size)
            s[i, current_col] = 0
            s[i, new_col] = 1
            new_energy = self.get_energy(s)

            # If the new state has lower energy, accept it
            # Otherwise, accept it with a probability that decreases with the energy difference and the temperature
            if new_energy < start_energy or np.random.rand() < np.exp((start_energy - new_energy) / T):
                start_energy = new_energy
            else:
                # Move the queen back to the original column
                s[i, new_col] = 0
                s[i, current_col] = 1

            if start_energy == 0:  # Optimal energy for 8 queens
                print(
                    f'Solution found in {self.external_iterations} iterations')
                yield s.copy()
                break
            yield s.copy()


        self.external_iterations += 1
        self.neurons = s.flatten()
        self.s = s
        self.print_queens(s)
        return s


    def update_state(self, X, i, s):
        """
        Update the state of the network
        """
        field = 0
        for Y in range(self.size):
            for j in range(self.size):
                field += self.J[X, i, Y, j] * s[Y, j]
        s[X, i] = 1 if (field> 0) else 0

    def print_synaptic_matrix(self, J):
        with open('synaptic_matrix.txt', 'w') as f:
            for i in range(self.size):
                for j in range(self.size):
                    for k in range(self.size):
                        for l in range(self.size):
                            f.write(
                                f"J({i},{j}),({k},{l}) = {J[i, j, k, l]}\n")
                        # f.write(f'J({i},{j}),bias = {J[i, j, i, self.size]}\n')

    def calculate_energy(self):
        """Calculate the energy of the current state."""
        energy = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.s[i, j] == 1:
                    for k in range(self.size):
                        for l in range(self.size):
                            if self.s[k, l] == 1:
                                energy += self.J[i, j, k, l]
        return energy / 2  # Because each interaction is counted twice

    def solve(self, iterations=1000):
        for _ in range(iterations):
            self.update_state()
            if self.calculate_energy() == -56:  # Optimal energy for 8 queens
                break
        return self.s
