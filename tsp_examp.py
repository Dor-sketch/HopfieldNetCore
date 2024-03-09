
from hopfield import Hopfield
import numpy as np

TSP_TOTAL_N = 9  # for 3 cities X, Y, Z
TSP_N = 3  # to iterate over the cities in a given trip
N_WITH_BIAS = TSP_N + 1

# represent the distance between cities
# from X    Y    Z
#     (1)  (2)  (3)  to X
#     (1)  (2)  (3)  to Y
#     (1)  (2)  (3)  to Z

"""
X -> X = 0
X -> Y = 1
X -> Z = 10

Y -> X = 1
Y -> Y = 0
Y -> Z = 30

Z -> X = 10
Z -> Y = 30
Z -> Z = 0
"""

import numpy as np

DIST = np.array([[0, 1, 10],  # Distances from X to X, Y, Z
                 [1, 0, 30],  # Distances from Y to X, Y, Z
                 [10, 30, 0]])  # Distances from Z to X, Y, Z

DIST = { # Distances from X to X, Y, Z
    'X': {'X': 0, 'Y': 30, 'Z': 10},
    'Y': {'X': 1, 'Y': 30, 'Z': 10},
    'Z': {'X': 10, 'Y': 10, 'Z': 0}
}

DIST2 = np.array([[0, 1, 1],  # Distances from X to X, Y, Z
                 [1, 0, 1],  # Distances from Y to X, Y, Z
                 [1, 1, 0]])  # Distances from Z to X, Y, Z

INITIAL_STATE = np.array([[1, 1, 0],
                          [0, 0, 1],
                          [0, 0, 0]])

INITIAL_STATE2 = np.array([[1, 0, 0],
                          [1, 1, 0],
                          [0, 0, 1]])

INITIAL_STATE_OPTIMAL = np.array([[0, 1, 0],
                                  [0, 0, 1],
                                  [1, 0, 0]])

INITIAL_STATE_OPTIMAL2 = np.array([[0, 1, 0],
                                   [1, 0, 0],
                                   [0, 0, 1]])

CITIES = ['X', 'Y', 'Z']
DAYS = ['1', '2', '3']

# Note that the state is a 3x3 matrix where each row represents a city and each column represents a step in the trip
# we wil use i and j to iterate over the rows and X and Y to iterate over the columns
# Each synapse is represented by a 4D matrix J[X][i][Y][j]
# where X and Y are the cities (this side ->) and i and j are the steps in the trip (downwards)

# valid state is a state representing a valid path through the cities

#      I    II   III
#           2        to X
#      1             to Y
#                 3  to Z



class TSPEnergy:
    def __init__(self, dist=DIST):
        self.dist = dist

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def energy_a(self, s):
        """
        Calculate the energy of a state using the constraint a
        add to the energy penalty for visiting the same city in different days
        """
        energy = 0
        print(s)
        for X in range(TSP_N-1):
            for i in range(TSP_N-1):
                for j in range(TSP_N-1):
                    if j != i:
                        print(f'{{x: {X}, i: {i}, j: {j}}}')
                        energy += s[X][i] * s[X][j]
        return 0.5 * energy


    def energy_b(self, s):
        """
        Calculate the energy of a state using the constraint b
        """
        energy = 0
        for i in range(TSP_N):
            for X in range(TSP_N):
                for Y in range(TSP_N):
                    if Y != X:
                        energy += s[X][i] * s[Y][i]
        return 0.5 * energy


    def energy_c(self, s):
        """
        Calculate the energy of a state using the constraint c
        """
        return 0.5 * (np.sum(s) - TSP_N) ** 2


    def energy_d(self, s):
        """
        Calculate the energy of a state
        """
        energy = 0
        for X, city in enumerate(CITIES):
            for Y, city2 in enumerate(CITIES):
                if Y != X:
                    for i, day in enumerate(DAYS):
                        energy += DIST[city][city2] * s[X][i] * (
                            s[Y][i - 1 if i > 0 else TSP_N - 1] + s[Y][i + 1 if i < TSP_N - 1 else 0])

        return energy * 0.5


    def get_energy_with_constraints(self, s=None, dist=DIST) -> float:
        """
        Calculate the energy of a state
        """

        energy = A * self.energy_a(s) \
            + B * self.energy_b(s) \
            + C * self.energy_c(s) \
            + D * self.energy_d(s)
        return energy

def Kronecker_delta(i, j):
    return 1 if i == j else 0

# BETA = 0.03
# # weights - can be modified
# A = np.random.uniform(0, BETA)
# B = np.random.uniform(0, 1)
# C = np.random.uniform(1-BETA, 1)
# D = np.random.uniform(0, BETA) + 1/TSP_N

A = 100
B = 100
C = 100
D = 100


class TSPNet(Hopfield):
    def __init__(self, N=TSP_N, dist=DIST):
        super().__init__(N)
        self.dist = dist
        self.J = self.get_synaptic_matrix_with_constraints(INITIAL_STATE)
        self.s = INITIAL_STATE
        self.neurons = INITIAL_STATE.flatten()

    def get_route(self, s=None):
        if s is None:
            s = self.s
        route = []
        for i, day in enumerate(DAYS):
            for X, city in enumerate(CITIES):
                if s[X][i] == 1:
                    route.append(city)
        return route

    def get_energy(self, s=None, dist=DIST) -> float:
        energy = 0
        if s is None:
            s = self.s
        else:
            # change to 3x3 matrix if s is a 1D array
            s = s.reshape((TSP_N, TSP_N))
            print(s)

        with TSPEnergy(dist) as tsp_energy:
            energy = tsp_energy.get_energy_with_constraints(s, dist)
        return energy

    def reset(self):
        self.J = self.get_synaptic_matrix_with_constraints(INITIAL_STATE)
        self.s = np.random.choice([0, 1], size=(TSP_N, TSP_N))
        self.neurons = self.s.flatten()

    def next_state_single(self, s=None, dist=DIST):
        """
        choose single neuron to update
        """
        if s is None:
            s = self.s
        scores = np.zeros((TSP_N, TSP_N))
        self.print_synaptic_matrix()
        for X, city in enumerate(CITIES):
            for i, day in enumerate(DAYS):
                field = 0
                for Y, city2 in enumerate(CITIES):
                    for j, day2 in enumerate(DAYS):
                        field += self.J[X][i][Y][j] * s[Y][j]
                        print(f'{self.J[X][i][Y][j]}*{s[Y][j]}', end=' ')
                    print()
                print(f' bias = {self.J[X][i][i][TSP_N]}*{s[X][i]}')
                    # add the bias neurons
                print(f'(J{city}{day},bias + {field}) = {field + self.J[X][i][i][TSP_N]}')
                print()
                scores[X][i] = self.J[X][i][i][TSP_N] + field
        chosen = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
        print(f'Chosen: {chosen}')
        self.s[chosen] = 1 if scores[chosen] >= 0 else 0
        self.neurons = self.s.flatten()

    def next_state(self, s=None, dist=DIST):
        if s is None:
            s = self.s
        new_s = s.copy()
        self.print_synaptic_matrix()
        for X, city in enumerate(CITIES):
            for i, day in enumerate(DAYS):
                field = 0
                for Y, city2 in enumerate(CITIES):
                    for j, day2 in enumerate(DAYS):
                        field += self.J[X][i][Y][j] * s[Y][j]
                        print(f'{self.J[X][i][Y][j]}*{s[Y][j]}', end=' ')
                    print()
                print(f' bias = {self.J[X][i][i][TSP_N]}*{s[X][i]}')
                    # add the bias neurons
                print(f'(J{city}{day},bias + {field}) = {field + self.J[X][i][i][TSP_N]}')
                print()
                s[X][i] = 1 if self.J[X][i][i][TSP_N] + field >= 0 else 0
        self.neurons = s.flatten()
        self.s = s
        return s


    def get_synaptic_matrix_with_constraints(self, dist=DIST) -> np.ndarray:
        """
        Calculate the synaptic matrix based on the custom Energy function
        with constraints designed for the TSP.
        """
        J = np.zeros((TSP_N, TSP_N, TSP_N, TSP_N+1)) # +1 for the bias
        for X, city in enumerate(CITIES):
            for i, day in enumerate(DAYS):
                for Y, city2 in enumerate(CITIES):
                    for j, day2 in enumerate(DAYS):
                        J[X][i][Y][j] =  \
                            - A * Kronecker_delta(X, Y) * (1 - Kronecker_delta(i, j)) \
                            - B * Kronecker_delta(i, j) * (1 - Kronecker_delta(X, Y)) \
                            - C \
                            - D * DIST[city][city2] * (Kronecker_delta(i-1, j) + Kronecker_delta(i+1, j))
                        print(f'J{city}{day},{city2}{day2}: {J[X][i][Y][j]}')
                    # Add the bias synapse to every neuron in the next layer
                    J[X][i][Y][TSP_N] = TSP_N * C
        return J


    def get_energy_with_constraints_and_weights(self, s=None, dist=DIST) -> float:
        """
        Calculate the energy of a state
        """
        energy = 0
        J = self.J
        if s is None:
            s = self.s

        for X in range(TSP_N):
            for i in range(TSP_N):
                for Y in range(TSP_N):
                    for j in range(TSP_N):
                        energy += J[X][i][Y][j] * s[X][i] * s[Y][j]

        # add the bias neurons
        for X in range(TSP_N):
            for i in range(TSP_N):
                energy += J[X][i][TSP_N-1][TSP_N-1] * s[X][i]

        return -energy * 0.5


    def print_synaptic_matrix(self) -> None:
        print(self.s)
        for X, city in enumerate(CITIES):
            for i, day in enumerate(DAYS):
                for Y, city2 in enumerate(CITIES):
                    for j, day2 in enumerate(DAYS):
                        print(f'J{city}{day},{city2}{day2}: {self.J[X][i][Y][j]:4}', end=' ')
                    # print bias
                    print(f'J{city}{day},bias: {self.J[X][i][i][TSP_N]}')
                print('')


    # def get_synaptic_matrix(self, dist=DIST) -> np.ndarray:
    #     """
    #     Wrong implementation - no weights for the constraints
    #     Calculate the synaptic matrix of a state
    #     """
    #     J = np.zeros((TSP_N, TSP_N, TSP_N, TSP_N))
    #     for X in range(TSP_N):
    #         for Y in range(TSP_N):
    #             if X != Y:
    #                 for i in range(TSP_N):
    #                     for j in range(TSP_N):
    #                         if abs(i - j) == 1:
    #                             J[X][i][Y][j] = -dist[X][Y]
    #     return J


    def get_energy_using_weight(self, s, dist=DIST, only_dot_product=False) -> float:
        """
        Calculate the energy of a state
        """
        energy = 0
        J = self.get_synaptic_matrix()

        for X in range(TSP_N):
            for i in range(TSP_N):
                for Y in range(TSP_N):
                    for j in range(TSP_N):
                        energy += J[X][i][Y][j] * s[X][i] * s[Y][j]
        if only_dot_product:
            return energy
        return -energy * 0.5


    def update_state(self, X, i, s=None, J=None) -> int:
        """
        Update the state s by flipping the value of the neuron (i, j)
        """
        field = 0
        if J is None:
            J = self.J
        if s is None:
            s = self.s
        for Y in range(self.N):
            for j in range(self.N):
                print(s)
                print(X, i, Y, j)
                print(J[X][i][Y][j])
                field += J[X][i][Y][j] * s[Y][j]
        s[X][i] = 1 if field > 0 else 0
        return s[X][i]