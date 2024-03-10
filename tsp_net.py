"""
TSP using Hopfield network

Note that the state is a 3x3 matrix where each row represents a city and each
column represents a step in the trip.
we wil use i and j to iterate over the rows and X and Y to iterate over the columns
Each synapse is represented by a 4D matrix J[X][i][Y][j]
where X and Y are the self.road_map (this side ->) and i and j are the steps in the trip (downwards)
valid state is a state representing a valid path through the self.road_map

     I    II   III
          2        to X
     1             to Y
                3  to Z

"""

import numpy as np
from hopfield import Hopfield
from tsp_map import Map
from tsp_energy import TSPEnergy
from tsp_weights import A, B, C, D


def Kronecker_delta(i, j):
    return 1 if i == j else 0


class TSPNet(Hopfield):
    def __init__(self, road_map: Map):
        self.road_map = Map()
        print(self.road_map.city_set)
        self.N = len(self.road_map.city_set)
        self.days = [str(i) for i in range(1, self.N + 1)]
        self.s = self.get_random_state()
        self.J = self.get_synaptic_matrix_with_constraints()
        self.neurons = self.s.flatten()

    def get_random_state(self):
        return np.random.choice([0, 1], size=(self.N, self.N))

    def get_route(self, s=None):
        if s is None:
            s = self.s
        route = []
        for i, day in enumerate(self.days):
            for X, city in enumerate(self.road_map):
                if s[X][i] == 1:
                    route.append(city)
        return route

    def is_stable(self, s=None):
        """
        for the TSP, a state is stable if it represents a valid path through the self.road_map
        """
        # check if the state is a valid path through the self.road_map
        # meaning each row has only one 1 - not more and not less
        if s is None:
            s = self.s
        for X in range(self.N):
            if sum(s[X]) != 1:
                return False
        # check if each column has only one 1 - not more and not less
        for i in range(self.N):
            if sum(s[:, i]) != 1:
                return False
        return True

    def getLocalField(self, X, i=None, s=None, J=None):
        """
        Returns the local field of neuron i.
        hᵢ = Σjᵢⱼ•sⱼ(t)
        """
        local_field = 0
        if J is None:
            J = self.J
        if s is None:
            s = self.s
        # change to x and y from X based on colls
        if i is None:
            X, i = divmod(X - 1, 3)

        for Y in range(2):
            for j in range(2):
                local_field += J[X][i][Y][j] * s[Y][j]
        return local_field

    def get_energy(self, s=None, map=None) -> float:
        energy = 0
        if s is None:
            s = self.s
        else:
            # change to suitable matrix if s is a 1D array
            print(s)
            s = s.reshape((self.N, self.N))

        with TSPEnergy(self.road_map) as tsp_energy:
            energy = tsp_energy.get_energy_with_constraints(s)
        return energy

    def reset(self):
        # synapses don't change
        self.s = np.random.choice([0, 1], size=(self.N, self.N))
        self.neurons = self.s.flatten()

    def next_state(self, s=None, map=None):
        """
        Calculate the next state of the network
        """
        self.road_map.take_snapshot(self.get_route(self.s))

        iterations = 5 * self.N * self.N * self.N
        history = []
        for _ in range(iterations):
            X, i = np.random.randint(self.N, size=2)
            before = self.s[X][i]
            self.update_state(X, i, self.s)
            print(
                f"X={X}, i={i}, before={before} => after={self.s[X][i]}, energy={self.get_energy(self.s)}"
            )
            if self.get_energy(s) < 1:
                break
            history.append(self.get_energy(self.s))
            if len(history) > 2 and history[-1] == history[-2]:
                break

        self.neurons = self.s.flatten()
        self.road_map.take_snapshot(self.get_route(self.s))
        return self.s

    def next_state_all(self, s=None, map=None):
        """
        Calculate the next state of the network
        """
        if s is None:
            s = self.s
        new_s = s.copy()
        self.print_synaptic_matrix()
        for X, city in enumerate(self.road_map):
            for i, day in enumerate(self.days):
                field = 0
                for Y, city2 in enumerate(self.road_map.city_set.copy()):
                    for j, day2 in enumerate(self.days):
                        field += self.J[X][i][Y][j] * s[Y][j]
                        print(f"{self.J[X][i][Y][j]}*{s[Y][j]}", end=" ")
                    print()
                print(f" bias = {self.J[X][i][i][self.N]}*{s[X][i]}")
                print(
                    f"(J{city}{day},bias + {field}) = {field + self.J[X][i][i][self.N]}\n"
                )
                new_s[X][i] = 1 if (self.J[X][i][i][self.N] + field > 0) else 0
        self.neurons = new_s.flatten()
        self.s = new_s
        print(f"END: {self.get_energy_with_constraints_and_weights(new_s)}")
        return s

    def get_synaptic_matrix_with_constraints(self) -> np.ndarray:
        """
        Calculate the synaptic matrix based on the custom Energy function
        with constraints designed for the TSP.
        """
        J = np.zeros((self.N, self.N, self.N, self.N + 1))  # +1 for the bias
        map_copy = self.road_map.city_set.copy()
        for X, city in enumerate(self.road_map):
            for i, day in enumerate(self.days):
                for Y, city2 in enumerate(map_copy):
                    for j, day2 in enumerate(self.days):
                        print(f"J{city}{day},{city2}{day2}")
                        J[X][i][Y][j] = (
                            -A * Kronecker_delta(X, Y) *
                            (1 - Kronecker_delta(i, j))
                            - B * Kronecker_delta(i, j) *
                            (1 - Kronecker_delta(X, Y))
                            - C
                            - D
                            * self.road_map[city][city2]
                            * (Kronecker_delta(i - 1, j) + Kronecker_delta(i + 1, j))
                        )
                        print(f"J{city}{day},{city2}{day2}: {J[X][i][Y][j]}")
                    # Add the bias synapse to every neuron in the next layer
                    J[X][i][Y][self.N] = 2 * self.N * C

        # flatten the synaptic matrix for the gui
        # Suppose 'weights' is your 4D numpy array
        self.weights = np.random.rand(100, 100)
        for X in range(self.N):
            for i in range(self.N):
                for Y in range(self.N):
                    for j in range(self.N):
                        self.weights[X * self.N + i][Y *
                                                     self.N + j] = J[X][i][Y][j]
                    # # add bias
                    # self.weights[X * self.N + i][self.N * self.N] = J[X][i][i][self.N]
        print(f"weigh shape: {self.weights.shape}")
        self.print_synaptic_matrix(J)
        return J

    def get_energy_with_constraints_and_weights(self, s=None, map=None) -> float:
        """
        This method is used to test weather the energy function is implemented correctly
        in the synaptic matrix
        """
        energy = 0
        if s is None:
            s = self.s
        else:
            # change to 3x3 matrix if s is a 1D array
            s = s.reshape((self.N, self.N))
            print(s)

        with TSPEnergy(self.road_map) as tsp_energy:
            energy = tsp_energy.get_energy_with_constraints(s)
        return energy

    def print_synaptic_matrix(self, J=None):
        """
        Print the synaptic matrix
        """
        if J is None:
            J = self.J
        for X, city in enumerate(self.road_map):
            for i, day in enumerate(self.days):
                for Y, city2 in enumerate(self.road_map.city_set.copy()):
                    for j, day2 in enumerate(self.days):
                        print(
                            f"J{city}{day},{city2}{day2}: {J[X][i][Y][j]:4}", end=" ")
                    # print bias
                    print(f"J{city}{day},bias: {J[X][i][i][self.N]}")
                print("")

    def get_energy_using_weight(self, s, map=None, only_dot_product=False) -> float:
        """
        Calculate the energy of a state
        """
        energy = 0
        J = self.J

        for X in range(self.N):
            for i in range(self.N):
                for Y in range(self.N):
                    for j in range(self.N):
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
                field += J[X][i][Y][j] * s[Y][j]
        s[X][i] = 1 if field + J[X][i][i][self.N] > 0 else 0
        return s[X][i]
