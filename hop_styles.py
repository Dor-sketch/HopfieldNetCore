"""
This module contains the class HopStyles, which is used to style the graph of the Hopfield network.
"""

ACTIVE_COLOR = "yellow"
IDLE_COLOR = "darkblue"


class HopStyles:
    def __init__(self, hopfield):
        self.hopfield = hopfield
        self.N = hopfield.N

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def get_edges_style(self):
        """
        strong synaptics are in stronger green, weak synaptics are in lighter green / grey
        """
        colors = []
        widths = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                weight = self.hopfield.weights[i][j]
                # make many shades of green
                if weight > 0:
                    colors.append((weight, 0, weight, weight))
                else:
                    colors.append((0.5, 0.5, 0.5, abs(weight)))
                widths.append(abs(weight) * 10)

        # normalize widths, if very big graph make the edges smaller
        max_width = max(widths)
        if max_width != 0:
            widths = [width / max_width for width in widths]
        if len(widths) >= 100:
            widths = [width / 40 for width in widths]
            widths = [width if width > 0.1 else 0.1 for width in widths]
        return colors, widths

    def get_nodes_colors(self, neurons=None):
        """
        Return the color of the nodes
        """
        if neurons is None:
            neurons = self.hopfield.neurons
        return [
            ACTIVE_COLOR if neuron == 1 else IDLE_COLOR for neuron in list(neurons)
        ]

    def get_nodes_sizes(self):
        """
        Return the size of the nones based on ther energy
        """
        nodes_sizes = []
        for i in range(self.N):
            energy = self.hopfield.getLocalField(i)
            nodes_sizes.append(abs(energy) * 100)
        # normalize the sizes
        max_size = max(nodes_sizes)
        if max_size != 0:
            nodes_sizes = [size / max_size * 1000 for size in nodes_sizes]
        else:
            nodes_sizes = [100 for _ in nodes_sizes]
        return nodes_sizes
