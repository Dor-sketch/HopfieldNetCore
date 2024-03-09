

import numpy as np

TSP_TOTAL_N = 9 # for 3 cities X, Y, Z
TSP_N = 3  # to iterate over the cities in a given trip
# represent the distance between cities
# from X    Y    Z
#     (1)  (2)  (3)  to X
#     (1)  (2)  (3)  to Y
#     (1)  (2)  (3)  to Z

DIST = np.array([[0 , 30, 10],
                [30, 0, 1],
                [10, 1, 0]])

INITIAL_STATE = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]])

# Note that the state is a 3x3 matrix where each row represents a city and each column represents a step in the trip
# we wil use i and j to iterate over the rows and X and Y to iterate over the columns
# Each synapse is represented by a 4D matrix J[X][i][Y][j]
# where X and Y are the cities (this side ->) and i and j are the steps in the trip (downwards)

# valid state is a state representing a valid path through the cities

#      I    II   III
#           2        to X
#      1             to Y
#                 3  to Z


def energy_d(s):
    """
    Calculate the energy of a state
    """
    energy = 0
    print(s)
    for X in range(TSP_N):
        for Y in range(TSP_N):
            if X != Y:
                for i in range(TSP_N - 1):
                    print(f'sXi = s{X}{i}: {s[X][i]}')
                    print(f'sY(i+1) = s{Y}{i+1}: {s[Y][i+1]}')
                    print(f'sY(i-1) = s{Y}{i-1}: {s[Y][i-1]}')
                    print(f'DIST[X][Y]: {DIST[X][Y]}')
                    print()
                    energy += s[X][i] * (s[Y][i + 1] + s[Y][i - 1]) * DIST[X][Y]
                    # adding the distance between cities iff:
                    # - the city X is visited in the state (s[X][i] == 1)
                    # - the city Y is visited either before or after the city X (s[Y][i - 1] == 1 or s[Y][i + 1] == 1)
    return energy * 0.5


def get_synaptic_matrix(s, dist=DIST) -> np.ndarray:
    """
    Calculate the synaptic matrix of a state
    """
    J = np.zeros((TSP_N, TSP_N, TSP_N, TSP_N))
    for X in range(TSP_N):
        for Y in range(TSP_N):
            if X != Y:
                for i in range(TSP_N):
                    for j in range(TSP_N):
                        if abs(i - j) == 1:
                            J[X][i][Y][j] = -dist[X][Y]
    return J

def get_energy_using_weight(s, dist=DIST, only_dot_product=False) -> float:
    """
    Calculate the energy of a state
    """
    energy = 0
    J = get_synaptic_matrix(s, dist)

    for X in range(TSP_N):
        for i in range(TSP_N):
            for Y in range(TSP_N):
                for j in range(TSP_N):
                    energy += J[X][i][Y][j] * s[X][i] * s[Y][j]
    if only_dot_product:
        return energy
    return -energy * 0.5

def update_state(s, X, i, J=None) -> int:
    """
    Update the state s by flipping the value of the neuron (i, j)
    """
    field = 0
    if J is None:
        J = get_synaptic_matrix(s)
    for Y in range(TSP_N):
        for j in range(TSP_N):
            field += J[X][i][Y][j] * s[Y][j]
    s[X][i] = 1 if field > 0 else 0
    return s[X][i]



# s = INITIAL_STATE.copy()
# print(get_energy_using_weight(INITIAL_STATE))
# for X in range(TSP_N):
#     for i in range(TSP_N):
#         update_state(s, X, i)
# print(get_energy_using_weight(s))
# print(s)



def energy_a(s):
    """
    Calculate the energy of a state using the constraint a
    """
    energy = 0
    for X in range(TSP_N):
        for i in range(TSP_N):
            for j in range(TSP_N):
                energy += s[X][i] * s[X][j]
    return 0.5 * energy

def energy_b(s):
    """
    Calculate the energy of a state using the constraint b
    """
    energy = 0
    for i in range(TSP_N):
        for X in range(TSP_N):
            for Y in range(TSP_N):
                energy += s[X][i] * s[Y][i]
    return 0.5 * energy

def energy_c(s):
    """
    Calculate the energy of a state using the constraint c
    """
    return 0.5 * (np.sum(s) - TSP_N) ** 2




def Kronecker_delta(i, j):
    return 1 if i == j else 0

# weights - can be modified
A = 1
B = 1
C = 1
D = 1

def get_synaptic_matrix_with_constraints(s, dist=DIST) -> np.ndarray:
    """
    Calculate the synaptic matrix of a state
    """
    N_WITH_BIAS = TSP_N
    J = np.zeros((N_WITH_BIAS, N_WITH_BIAS, N_WITH_BIAS, N_WITH_BIAS))
    for X in range(TSP_N):
        for i in range(TSP_N):
            for Y in range(TSP_N):
                for j in range(TSP_N):
                    J[X][i][Y][j] = \
                        - A * Kronecker_delta(X, Y) * (1 - Kronecker_delta(i, j)) \
                        - B* Kronecker_delta(i, j) * (1 - Kronecker_delta(X, Y)) \
                        - C \
                        - D * DIST[X][Y] * (
                            Kronecker_delta(i-1, j) + Kronecker_delta(i+1, j))

    for X in range(TSP_N):
        for i in range(TSP_N):
            J[X][i][TSP_N-1][TSP_N-1] = 2 * TSP_N * C
            J[TSP_N-1][TSP_N-1][X][i] = 2 * TSP_N * C

    return J


print(get_synaptic_matrix_with_constraints(INITIAL_STATE))

def get_energy_with_constraints(s, dist=DIST) -> float:
    """
    Calculate the energy of a state
    """
    print(f'energy_a: {energy_a(s)}')
    print(f'energy_b: {energy_b(s)}')
    print(f'energy_c: {energy_c(s)}')
    print(f'energy_d: {energy_d(s)}')

    energy = A *  energy_a(s) \
            + B * energy_b(s) \
            + C * energy_c(s) \
            + D * energy_d(s)
    return energy

def get_energy_with_constraints_and_weights(s, dist=DIST) -> float:
    """
    Calculate the energy of a state
    """
    energy = 0
    J = get_synaptic_matrix_with_constraints(s, dist)

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

J = get_synaptic_matrix_with_constraints(INITIAL_STATE)

for X in range(TSP_N):
    for i in range(TSP_N):
        for Y in range(TSP_N):
            for j in range(TSP_N):
                print(f'J[{X}][{i}][{Y}][{j}]: {J[X][i][Y][j]}', end=' ')
            print()


print(get_energy_with_constraints(INITIAL_STATE))
print(get_energy_with_constraints_and_weights(INITIAL_STATE))

for X in range(TSP_N):
    for i in range(TSP_N):
        update_state(INITIAL_STATE, X, i, J)

def next_state(s, dist=DIST):
    J = get_synaptic_matrix_with_constraints(s, dist)
    new_s = s.copy()
    for X in range(TSP_N):
        for i in range(TSP_N):
            update_state(new_s, X, i, J)
    print(new_s)
    return s

for i in range(10):
    state = next_state(INITIAL_STATE)
    print(get_energy_with_constraints_and_weights(state))
    
print(next_state)