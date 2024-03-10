"""
This module contains the energy function for the TSP problem
"""

import math
from tsp_map import Map
from tsp_weights import A, B, C, D


class TSPEnergy:
    def __init__(self, road_map: Map):
        self.road_map = Map()
        self.N = len(self.road_map)
        self.days = [str(i) for i in range(1, self.N + 1)]

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
        for X in range(self.N - 1):  # -1 to avoid bias
            for i in range(self.N - 1):
                for j in range(self.N - 1):
                    if j != i:
                        energy += s[X][i] * s[X][j]
        return 0.5 * energy

    def energy_b(self, s):
        """
        Calculate the energy of a state using the constraint b
        """
        energy = 0
        for i in range(self.N):
            for X in range(self.N):
                for Y in range(self.N):
                    if Y != X:
                        energy += s[X][i] * s[Y][i]
        return 0.5 * energy

    def energy_c(self, s):
        """
        Calculate the energy of a state using the constraint c
        """
        energy = 0
        for X, city in enumerate(self.road_map):
            for i, day in enumerate(self.days):
                energy += s[X][i]
        energy = (energy - self.N) ** 2
        return (0.5 * energy)

    def energy_d(self, s):
        """
        Calculate the energy of a state
        """
        energy = 0
        cities = self.road_map.city_set.copy()
        for X, city in enumerate(self.road_map):
            for Y, city2 in enumerate(cities):
                if Y != X:
                    for i, day in enumerate(self.days):
                        if i < self.N - 1:
                            energy += self.road_map[city][city2] * s[X][i] * (
                                s[Y][i - 1 if i > 0 else self.N - 1]
                                + s[Y][i + 1 if i < self.N - 1 else 0])
        return energy * 0.5

    def get_energy_with_constraints(self, s=None) -> float:
        """
        Calculate the energy of a state
        """
        energy = A * self.energy_a(s) \
            + B * self.energy_b(s) \
            + C * self.energy_c(s) \
            + D * self.energy_d(s)
        return energy
