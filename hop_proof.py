"""
A very verbose information about the next state of the network and basic
theory of the Hopfield network energy.
"""

import io
import sys
import numpy as np

EXAMPLE = np.array([1, -1, -1])

EXAMPLE_WEIGHTS = np.array(
    [
        [0, 0.25, -0.25, -0.25],
        [0.25, 0, -0.25, -0.25],
        [-0.25, -0.25, 0, 0.25],
        [-0.25, -0.25, 0.25, 0],
    ]
)

subscript = {}
for i in range(300):
    subscript[i + 1] = chr(8320 + i)


def part1(neurons=EXAMPLE, weights=EXAMPLE_WEIGHTS):
    """
    This function is a verbose explanation of the energy of the Hopfield network.

    E = -1/2 * ΣᵢΣⱼ Jᵢⱼ * sᵢ * sⱼ
    """
    print()
    for i in range(neurons.shape[0]):
        if i != 0:
            print("   + ", end="")
        else:
            print("E = {", end="")
        for j in range(neurons.shape[0]):
            print(
                f"e{subscript[i+1]}{subscript[j+1]} = -(1/2)•(J{subscript[i+1]}{subscript[j+1]}•s{subscript[i+1]}•s{subscript[j+1]})",
                end=" + " if j != neurons.shape[0] - 1 else "",
            )
        if i == neurons.shape[0] - 1:
            print("}")
        else:
            print()


def part2(neurons=EXAMPLE, weights=EXAMPLE_WEIGHTS):
    """
    This part only arranges the energy equation in a more readable way.
    """
    print("\nWe can extract -1/2")
    print(" -1/2 * {", end="")
    for i in range(neurons.shape[0]):
        if i != 0:
            print("     ", end="")
            print(" + ", end="")
        for j in range(neurons.shape[0]):
            print(
                f"j{subscript[i+1]}{subscript[j+1]}•s{subscript[i+1]}•s{subscript[j+1]}",
                end=" + " if j != neurons.shape[0] - 1 else "\n",
            )
    print("}")
    print("The diagonal is 0 so we can write the energy as:")
    print(" -1/2 *   {", end="")
    for i in range(neurons.shape[0]):
        if i != 0:
            print("     ", end="")
            print(" + ", end="")
        for j in range(neurons.shape[0]):
            if i == j:
                print("    0    ", end=" + ")
            else:
                print(
                    f"j{subscript[i+1]}{subscript[j+1]}•s{subscript[i+1]}•s{subscript[j+1]}",
                    end=" + " if j != neurons.shape[0] - 1 else "\n",
                )
    print("}")


def part3(neurons=EXAMPLE, weights=EXAMPLE_WEIGHTS):
    """
    In this part we substitute the new energy from the energy equation.
    """
    print(
        "Note that if sᵢ updated, we can subtract the energy of the network to get the energy difference"
    )
    print("E(t+1) - E(t) = -1/2 * (hᵢ(t+1) - hᵢ(t))")
    print(
        "lets say noiron 2 is the only one updated. Well mark s(t+1) as s' and s(t) as s"
    )
    print("{")
    for i in range(neurons.shape[0]):
        if i != 0:
            print("     + ", end="")
        else:
            print("       ", end="")
        for j in range(neurons.shape[0]):
            if i == j:
                print("                       ", end="   ")
            else:
                si = "s" if i != 1 else "s'"
                sj = "s" if j != 1 else "s'"
                print(
                    f"j{subscript[i+1]}{subscript[j+1]}{si}{subscript[i+1]}{sj}{subscript[j+1]} - ",
                    end="",
                )
                print(
                    f"j{subscript[i+1]}{subscript[j+1]}s{subscript[i+1]}s{subscript[j+1]}",
                    end=" + " if j != neurons.shape[0] - 1 else "\n",
                )
        print()
    print("}")
    print(
        "If its the only neuron updated, all the other terms that do not contain s2 will be 0: so E(t+1) - E(t) ="
    )
    print("-1/2 * {")
    for i in range(neurons.shape[0]):
        if i != 0:
            print("     + ", end="")
        else:
            print("       ", end="")
        for j in range(neurons.shape[0]):
            if i == j or i != 1 and j != 1:
                print("                        ", end="   ")
            else:
                si = "s" if i != 1 else "s'"
                sj = "s" if j != 1 else "s'"
                print(
                    f"j{subscript[i+1]}{subscript[j+1]}{si}{subscript[i+1]}{sj}{subscript[j+1]} - ",
                    end="",
                )
                print(
                    f"j{subscript[i+1]}{subscript[j+1]}s{subscript[i+1]}s{subscript[j+1]}",
                    end=" + " if j != neurons.shape[0] - 1 else "\n",
                )
        print()
    print("}")
    print("-1/2 * {")
    for i in range(neurons.shape[0]):
        print("    ", end="")
        if i != 0:
            print(" + ", end="")
        else:
            print("   ", end="")
        for j in range(neurons.shape[0]):
            if i == j or i != 1 and j != 1:
                print("                        ", end="   ")
            else:
                si = "s" if i != 1 else "s'"
                sj = "s" if j != 1 else "s'"
                print(
                    f"j{subscript[i+1]}{subscript[j+1]}{si}{subscript[i+1]}{sj}{subscript[j+1]} - ",
                    end="",
                )
                print(
                    f"j{subscript[i+1]}{subscript[j+1]}s{subscript[i+1]}s{subscript[j+1]}",
                    end=" + " if j != neurons.shape[0] - 1 else "\n",
                )
        print()
    print("}")


def proof_concept(neurons=EXAMPLE, weights=EXAMPLE_WEIGHTS):
    """
    This function is a verbose explanation of the energy of the Hopfield network.
    It explains one of the basic ideas behind the convergence of the network and
    the energy of the network: why the energy of the network is converging to a local minimum.

    This assume symmetric weights, and unsynchronized update of the neurons.
    """
    neurons = EXAMPLE
    weights = EXAMPLE_WEIGHTS
    # Create a StringIO object and redirect stdout to it
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    part1(neurons, weights)
    part2(neurons, weights)
    part3(neurons, weights)

    print("We can extract the weights:")
    print("-1/2 * {")
    for i in range(neurons.shape[0]):
        print("    ", end="")
        if i != 0:
            print(" + ", end="")
        else:
            print("   ", end="")
        for j in range(neurons.shape[0]):
            if i == j or i != 1 and j != 1:
                print("                   ", end="   ")
            else:
                si = "s" if i != 1 else "s'"
                sj = "s" if j != 1 else "s'"
                print(
                    f"j{subscript[i+1]}{subscript[j+1]}({si}{subscript[i+1]}{sj}{subscript[j+1]} - s{subscript[i+1]}s{subscript[j+1]})",
                    end=" + " if j != neurons.shape[0] - 1 else "\n",
                )
        print()
    print("}")
    print("-1/2 * {")
    for i in range(neurons.shape[0]):
        print("    ", end="")
        if i != 0:
            print(" + ", end="")
        else:
            print("   ", end="")
        for j in range(neurons.shape[0]):
            if i == j or i != 1 and j != 1:
                print("                   ", end="   ")
            else:
                si = "s" if i != 1 else "s'"
                sj = "s" if j != 1 else "s'"
                print(
                    f"j{subscript[i+1]}{subscript[j+1]}s{subscript[j+1]}(s'{subscript[2]} - s{subscript[2]})",
                    end=" + " if j != neurons.shape[0] - 1 else "\n",
                )
        print()
    print("}")
    print("since the matrix is symmetric, we can write the energy difference as:")
    print("{")
    for i in range(neurons.shape[0]):
        print("    ", end="")
        for j in range(neurons.shape[0]):
            if i == j or i != 1 and j != 1 or i > j:
                print("                   ", end="   ")
            else:
                si = "s" if i != 1 else "s'"
                sj = "s" if j != 1 else "s'"
                print(
                    f"j{subscript[i+1]}{subscript[j+1]}s{subscript[j+1]}(s'{subscript[2]} - s{subscript[2]})",
                    end=" + " if j != neurons.shape[0] - 1 else "\n",
                )
        print()
    print("}")
    print("We can extract the weights and the local field of the neuron 2:")
    str_weights = ""
    for i in range(neurons.shape[0]):
        for j in range(neurons.shape[0]):
            if i < j and i == 1 or (j == 1 and i == 0):
                str_weights += f"j{subscript[i+1]}{subscript[j+1]}s{subscript[j+1]} + "
    str_weights = str_weights[:-2]
    print(f"[h{subscript[2]}=({str_weights})]*(s'{subscript[2]} - s{subscript[2]})\n\n")
    for i in range(neurons.shape[0]):
        print("| ", end="")
        for j in range(neurons.shape[0]):
            print(f"e{subscript[i+1]}{subscript[j+1]} = {weights[i][j]}", end=" | ")
        print()
    energy = np.zeros((neurons.shape[0], neurons.shape[0]))
    for i in range(neurons.shape[0]):
        for j in range(neurons.shape[0]):
            energy[i][j] = weights[i][j]
    print(energy)

    # Reset stdout to its original value
    sys.stdout = old_stdout

    # Get the string value from the StringIO object
    output = buffer.getvalue()

    return output


def generate_equation(neurons, new_state, weights, t):
    """
    Generate the equation for the next state.
    sum_{j=1}^{N} Jᵢⱼ * sⱼ(t)
    (sometimes markes with hᵢ(t) = \sum_{j=1}^{N} J_{ᵢⱼ} * sⱼ(t))
    """
    neurons_str = ", ".join(
        [f"s{subscript[i+1]}({t}) = {neurons[i]}" for i in range(len(neurons))]
    )
    cur_str = f"Current state: S({t})   = {{ {neurons_str} }}\n"
    equation = ""
    updated_neurons_str = ", ".join(
        [f"s{subscript[i+1]}({t+1}) = {new_state[i]}" for i in range(len(new_state))]
    )
    signs_str = ", ".join(
        [
            f"s{subscript[i+1]}({t+1}) = sgn(Σ J{subscript[i+1]}{subscript[ord('J')]}•s{subscript[ord('J')]}({t}))"
            for i in range(len(neurons))
        ]
    )
    equation += f"Updating States... S({t}+1) = {{ {signs_str} }}\n"
    neurons_str = ""
    for i in range(neurons.shape[0]):
        neurons_str += (
            f"    s{subscript[i+1]}({t+1}) = "
            + f"{' + '.join([f'{weights[i][j]}•{neurons[j]}' for j in range(neurons.shape[0])])}"
        )
        weights_neurons_sum = sum(
            [weights[i][j] * neurons[j] for j in range(neurons.shape[0])]
        )
        neurons_str += f" = sgn({weights_neurons_sum}) = {new_state[i]}\n"
    equation += neurons_str
    equation += cur_str + f"Updated state: S({t}+1) = {{ {updated_neurons_str} }}"
    print(equation)
