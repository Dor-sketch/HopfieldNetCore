"""
This module implements the Hopfield network.
Note the use of J as the weights matrix and sᵢ as the state of the neurons i.
"""

import numpy as np
from hop_proof import subscript, print_eq


def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


def binary_activation(x):
    if x > 0:
        return 1
    else:
        return 0


# Weights matrix representing the connections between neurons
# W[i][j] is the weight of the connection between neuron i and neuron j
DEFAULT_WEIGHTS = np.array(
    [
        [0, 0.25, -0.25, -0.25],
        [0.25, 0, -0.25, -0.25],
        [-0.25, -0.25, 0, 0.25],
        [-0.25, -0.25, 0.25, 0],
    ]
)

DEFAULT_STATES = np.array([1, -1, -1, -1])


class Hopfield:
    """
    This class implements a Hopfield network.
    """

    def __init__(self, N=4, J=DEFAULT_WEIGHTS, next_state_func=sign):
        """
        Create a new Hopfield network with N neurons.
        neurons can be in state 1 or -1.
        weights are random matrix of size N x N. where Jᵢⱼ == Jᵢⱼ and Jᵢᵢ == 0

        N: number of neurons
        """
        self.next_state_func = next_state_func
        self.N = N
        self.neurons = np.random.choice([1, -1], N)  # Random initial state
        self.weights = np.zeros((N, N))
        if J is not None and N == DEFAULT_STATES.shape[0]:
            self.weights = J.copy()
            self.neurons = DEFAULT_STATES
        else:
            self.weights = np.random.rand(N, N)
            # set to 0 the diagonal of the weights matrix
            np.fill_diagonal(self.weights, 0)
            # make mat symmetric
            self.weights = (self.weights + self.weights.T) / 2
        self.is_stable = False
        self.t = 0

    def stable(self):
        """
        Returns true if the network has converged to a stable state.
        """
        return self.is_stable

    def reset(self):
        """
        Reset the network to its initial state.
        """
        self.neurons = np.random.choice([1, -1], self.N)
        self.is_stable = False
        self.t = 0

    @ print_eq
    def overlap_value(self, stored_pattern):
        """
        Returns the ovarlap between a network state in time t and a stored pattern.

        m = 1/N Σ sᵢ * sᵢ

        Note the result is between -1 and 1.
        """
        return np.dot(self.neurons, stored_pattern) / self.N

    def has_converged(self, new_state):
        """
        Returns true if the network has converged to a stable state.
        No change of the sign of the neurons in the next state.

        sᵢ(t) • sᵢ(t+1) > 0 for all i
        """
        for i in range(self.neurons.shape[0]):
            if self.neurons[i] * new_state[i] < 0:
                return False
        return True

    def store_patterns(self, patterns):
        """
        Store patterns by updating synabpses vals.

        Jij = sum 1/N si * sj if i != j or 0 if in diognal
        """
        self.weights = np.zeros((self.N, self.N))
        for pattern in patterns:
            self.neurons = pattern
            for i in range(len(self.neurons)):
                for j in range(len(self.neurons)):
                    if (i != j):
                        self.weights[i][j] += \
                            (1/self.N) * self.neurons[i]*self.neurons[j]

    def next_state(self):
        """
        Compute the next state of the network.
        Using usynchoronous update (one neuron at a time)
        """
        print(self.neurons)
        new_state = np.zeros(self.neurons.shape)
        for i in range(self.neurons.shape[0]):
            # a state will be updated (s(t+1) = -s) iff sᵢ(t)•sgn(hᵢ(t)) < 0
            # where hᵢ(t) = Σj{ᵢⱼ}•sⱼ(t)
            # its equivalent to sᵢ(t)•hᵢ(t) < 0
            new_state[i] = self.next_state_func(
                np.dot(self.weights[i], self.neurons))

        if self.has_converged(new_state):
            self.is_stable = True
        self.neurons = new_state
        self.t += 1

    def getSynapticScore(self, i, j, network_state=None):
        """
        Returns the synaptic score between neuron i and neuron j.
        e(i, j) = -(1/2) * (Jᵢⱼ * sᵢ * sⱼ)

        Note that if sᵢ and sⱼ have the same sign, the synaptic score is negative.
        If sᵢ and sⱼ have different signs, the synaptic score is positive.

        This means that the energy of the network is lower when neighboring neurons have the same sign.

        It helps to proof that the energy is converging to a local minimum:
        - The network has a finite number of possible states
        - If a noiron is updated during the computation of the next state,
                the energy of the network will decrease (see the proof concept in hop_proof.py)
        """
        if network_state is None:
            network_state = self.neurons
        # Ensure i and j are within the valid range of indices
        if i < len(self.weights) and j < len(self.weights[i]):
            weight = self.weights[i][j]
        else:
            print(f"Invalid index: i={i}, j={j}")
            return
        return -(1 / 2) * weight * network_state[i] * network_state[j]

    def getLocalField(self, i):
        """
        Returns the local field of neuron i.
        hᵢ = Σjᵢⱼ•sⱼ(t)
        """
        local_field = 0
        eq_str = f'h{subscript[i+1]} = '
        for j in range(len(self.weights[i])):
            if i != j:  # Exclude the neuron itself
                local_field += self.weights[i][j] * self.neurons[j]
                eq_str += f"({self.weights[i][j] * self.neurons[j]}) + "
        # print(eq_str[:-2])
        return local_field

    def get_energy(self, network_state=None):
        """
        Returns the energy of the network.
        E = ΣΣeᵢⱼ

        The enregy of the network is a function of the network state.
        Note the synapse does not change the energy of the network because the weights are constant
        during the computation of the next state.
        The nuber of possible energys is 2^N where N is the number of neurons.
        """
        energy = 0

        if network_state is None:
            network_state = self.neurons
        print(network_state)
        for i in range(network_state.shape[0]):
            for j in range(network_state.shape[0]):
                energy += self.getSynapticScore(i, j, network_state)
        print(f"Energy: {energy}")
        return energy

    def getCitiesEnergy(self, S, dist, X1):
        """
        Returns the energy of the network for the TSP problem.

        S: the state of the network
        dist: the distance matrix between cities
        X1: the index of the neuron to update
        """

    def setCustomEnergy(self, custom_weights=None):
        """
        Set a custom energy function for the network by setting the weights matrix.
        """
        # example fot the TSP problem

        # use only distance between cities
        # let dix{XY} be the distance between city X and city Y
        # let sX be the state of the city X
        # let s{Yi-1} and s{Yi+1} be the states of the cities Y before and after the city X

        # E_D = 1/2 ΣΣΣdix{XY} * sX * (s{Yi-1} + s{Yi+1})

        # now add constraints
        # E_A = 1/2 ΣΣΣsxi * sxj // only 1 city per line - to avoid the same city twice in the same line
        # E_B = 1/2 ΣΣΣsxi * sYi // only 1 city per column - to avoid the same city twice in the same column
        # E_C = 1/2 (ΣΣsxi-N)^2 - N^2 // N cities in the tour - n set neurons to 1 ...
        # hard - well add neurons to the network to deal with the constraints (BIAS neurons))
