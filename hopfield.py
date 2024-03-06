"""
This module implements the Hopfield network.
Note the use of J as the weights matrix and sᵢ as the state of the neurons i.
"""

import numpy as np
from hop_proof import subscript

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


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


# decorator to print equations
def print_eq(func):
    def wrapper(self, stored_pattern):
        print("Calculating dot product:")
        print(self.neurons)
        stored_pattern = np.array(stored_pattern).flatten()  # ensure stored_pattern is a flat array
        print(stored_pattern)
        dot_product = 0
        for neuron, pattern in zip(self.neurons, stored_pattern):
            product = neuron * pattern
            print(f"{neuron} * {pattern} = {product}")
            dot_product += product
        result = dot_product / self.N
        print(f"Dot product of neurons and stored pattern: {dot_product}")
        print(f"m = {dot_product} / {self.N} = {result}")
        return func(self, stored_pattern)
    return wrapper






class Hopfield:
    """
    This class implements a Hopfield network.
    """

    def __init__(self, N=4, J=DEFAULT_WEIGHTS):
        """
        Create a new Hopfield network with N neurons.
        neurons can be in state 1 or -1.
        weights are random matrix of size N x N. where Jᵢⱼ == Jᵢⱼ and Jᵢᵢ == 0

        N: number of neurons
        """
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
            new_state[i] = sign(np.dot(self.weights[i], self.neurons))

        if self.has_converged(new_state):
            self.is_stable = True
        self.neurons = new_state
        self.t += 1

    def getSynapticScore(self, i, j):
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
        return -(1 / 2) * self.weights[i][j] * self.neurons[i] * self.neurons[j]


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

    def getEnergy(self):
        """
        Returns the energy of the network.
        E = ΣΣeᵢⱼ

        The enregy of the network is a function of the network state.
        Note the synapse does not change the energy of the network because the weights are constant
        during the computation of the next state.
        The nuber of possible energys is 2^N where N is the number of neurons.
        """
        energy = 0
        for i in range(self.neurons.shape[0]):
            for j in range(self.neurons.shape[0]):
                energy += self.getSynapticScore(i, j)
        return energy
