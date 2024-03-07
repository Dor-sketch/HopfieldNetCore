"""
This module contains the GUI class that is used to visualize the Hopfield network
"""

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
import networkx as nx
from hopfield import Hopfield
from hop_proof import proof_concept, generate_equation


ACTIVE_COLOR = "yellow"
IDLE_COLOR = "darkblue"
BUTTONS_COLOR = "lightblue"


class Memory:
    """
    This class is used to store the state of the neurons and the weights of the Hopfield network
    """

    def __init__(self, neurons, weights):
        self.neurons = neurons
        self.weights = weights


class GUI:
    """
    This class implements the GUI for the Hopfield network.
    """

    def __init__(self, N):
        self.hopfield = Hopfield(N)
        self.N = N
        self.fig, self.ax = plt.subplots()
        self.graph = nx.Graph()
        self.init_graph(N)
        self.bnext = None
        self.breset = None
        self.setup_buttons()
        self.patterns = []
        self.stored = []
        # make window bigger
        self.fig.set_size_inches(14, 8)

    def setup_buttons(self):
        """
        Add buttons to the plot. for next state and reset
        """
        buttons = [
            {"position": [0.81, 0.05, 0.1, 0.07],
                "label": "Next", "callback": self.next},
            {"position": [0.7,  0.05, 0.1, 0.07],
                "label": "Reset", "callback": self.reset},
            {"position": [0.59, 0.05, 0.1, 0.07],
                "label": "Weights", "callback": self.weights},
            {"position": [0.48, 0.05, 0.1, 0.07],
                "label": "Theory", "callback": self.theory},
            {"position": [0.37, 0.05, 0.1, 0.07],
                "label": "Update Eq", "callback": self.weights_eq},
            {"position": [0.26, 0.05, 0.1, 0.07],
                "label": "Nothing to Store", "callback": self.store},
            {"position": [0.15, 0.05, 0.1, 0.07],
                "label": "Add", "callback": self.add},
            {"position": [0.04, 0.05, 0.1, 0.07],
                "label": "Overlap", "callback": self.get_overlap},
            {"position": [0.04, 0.15, 0.1, 0.07],
                "label": "Energy", "callback": self.energy},
            {"position": [0.04, 0.25, 0.1, 0.07],
                "label": "View Stored", "callback": self.view_stored},
            {"position": [0.04, 0.35, 0.1, 0.07],
                "label": "Make Gif", "callback": self.make_gif},
        ]

        for button in buttons:
            ax = plt.axes(button["position"])
            b = Button(ax, button["label"],
                       color=BUTTONS_COLOR, hovercolor='0.975')
            b.on_clicked(button["callback"])
            if button["label"] == "Add":
                self.badd = b
            if button["label"] == "Nothing to Store":
                self.bstore = b
            if button["label"] == "Next":
                self.bnext = b
            if button["label"] == "Reset":
                self.breset = b
            if button["label"] == "Weights":
                self.bweights = b
            if button["label"] == "Theory":
                self.btheory = b
            if button["label"] == "Update Eq":
                self.bupdate = b
            if button["label"] == "Overlap":
                self.boverlap = b
            if button["label"] == "Energy":
                self.benergy = b
            if button["label"] == "View Stored":
                self.bview = b
            if button["label"] == "Make Gif":
                self.bgif = b
            b.label.set_fontstyle("italic")
            b.label.set_fontfamily("serif")

    def view_stored(self, event):
        """
        Display the stored patterns a new figure with small network graphs
        """
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)
        fig.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_title("Stored Patterns", fontsize=20, color="lightblue",
                     fontweight="bold", fontstyle="italic", fontfamily="serif")
        ax.set_xlabel("Stored Pattern", color="lightblue")
        ax.set_ylabel("Neuron State", color="lightblue")
        ax.set_xticks([])
        ax.set_yticks([])


        for i, pattern in enumerate(self.stored):
            ax = fig.add_subplot(self.N, 1, i + 1)
            ax.set_title(f"Pattern {i + 1}")
            ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("Neuron")
            ax.set_ylabel("State")
            ax.bar(range(len(pattern)), pattern, color="lightblue")

        plt.tight_layout()
        plt.show()

    def get_overlap(self, event):
        """
        Display the overlap between the current state and the stored patterns
        """
        # open dialog with the stored patterns
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 4)
        fig.set_facecolor("black")
        ax.set_facecolor("black")
        ax.set_title("Overlap with Stored Patterns", fontsize=20, color="lightblue",
                     fontweight="bold", fontstyle="italic", fontfamily="serif")
        ax.set_xlabel("Stored Pattern")
        ax.set_ylabel("Overlap")

        ax.bar(range(len(self.stored)), [self.hopfield.overlap_value(np.array(pattern))
                                         for pattern in self.stored], color="lightblue")
        plt.show()

    def add(self, event):
        self.patterns.append(self.hopfield.neurons.copy())
        self.badd.label.set_text(f"Add More")
        self.bstore.label.set_text("Store {}".format(len(self.patterns)))
        self.badd.label.set_color("blue")
        self.bstore.label.set_color("green")
        self.draw_graph()
        plt.draw()

    def on_click(self, event):
        # Check if a node was clicked
        clicked_node = None
        # check valid click
        if event.xdata is None or event.ydata is None:
            return
        for node in self.graph.nodes:
            if (
                event.xdata - 0.03 < self.pos[node][0] < event.xdata + 0.03
                and event.ydata - 0.03 < self.pos[node][1] < event.ydata + 0.03
            ):
                clicked_node = node
                break
        if clicked_node is not None:
            # Change the state of the clicked node
            self.hopfield.neurons[clicked_node] *= -1
            self.draw_graph()
            plt.draw()

    def store(self, event):
        self.hopfield.store_patterns(self.patterns)
        self.bstore.label.set_text("Nothing to store")
        self.badd.label.set_text("Add")
        self.bstore.label.set_color("black")
        self.badd.label.set_color("black")
        for pattern in self.patterns:
            if not any(np.array_equal(pattern, stored) for stored in self.stored):
                self.stored.append(pattern)
                component_pattern = -1 * pattern
                if not any(np.array_equal(component_pattern, stored) for stored in self.stored):
                    self.stored.append(component_pattern)

        self.patterns = self.patterns[:0]
        self.draw_graph()
        plt.draw()

    def theory(self, event):
        # open new figure like old math text books make it scrollable with blackboard style
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 14)
        fig.set_facecolor("black")
        ax.set_facecolor("black")
        ax.set_title("Why Converging? (WIP)", fontsize=20, color="lightblue",
                     fontweight="bold", fontstyle="italic", fontfamily="serif")
        p = proof_concept()
        # add the equation to the figure like whiteboard
        ax.text(0.5, 0.5, p, va="center", fontsize=10,
                color="white", fontstyle="italic", fontfamily="serif", ha="center")
        ax.axis("off")
        plt.show()

    def weights_eq(self, event):
        old_state = self.hopfield.neurons
        self.hopfield.next_state()
        generate_equation(old_state,
                          self.hopfield.neurons, self.hopfield.weights, self.hopfield.t - 1)
        self.draw_graph()
        plt.draw()

    def energy(self, event):
        """
        Display the energy of the current state
        """
        energy = self.hopfield.getEnergy()
        print(f"Energy: {energy}")

    def weights(self, event):
        """
        Display the weights of the network
        """
        # open new figure
        fig, ax = plt.subplots()
        ax.imshow(self.hopfield.weights, cmap="viridis")
        ax.set_title("Weights of the Hopfield Network")
        plt.show()

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
            widths = [width / max_width * 10 for width in widths]
        if len(widths) > 100:
            widths = [width / 10 for width in widths]
            widths = [width if width > 0.1 else 0.1 for width in widths]
        return colors, widths

    def get_nodes_colors(self, neurons=None):
        """
        Return the color of the nodes
        """
        if neurons is None:
            neurons = self.hopfield.neurons
        print(f'in get_nodes_colors: {neurons}')
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

    def init_graph(self, N):
        """
        Draw the graph with the current state of the neurons
        """

        # Create nodes
        for i in range(N):
            self.graph.add_node(i)

        # Create edges
        for i in range(N):
            for j in range(i + 1, N):
                weight = self.hopfield.weights[i][j]
                self.graph.add_edge(
                    i, j, weight=weight, alpha=0.5, width=weight * 10
                )

        self.pos = nx.spring_layout(self.graph, seed=42, iterations=100)
        self.draw_graph()

    def draw_graph(self):
        """
        update the graph with the current state of the neurons
        """
        self.ax.clear()
        self.update_labels()
        edges_colors, edge_widths = self.get_edges_style()
        node_colors = self.get_nodes_colors()
        node_sizes = self.get_nodes_sizes()

        nx.draw_networkx(  # Draw the graph
            self.graph,
            self.pos,
            node_color=node_colors,
            with_labels=True,
            ax=self.ax,
            width=edge_widths,
            # node_size=node_sizes,
            node_size=1000,
            edge_color=edges_colors,
        )

    def update_labels(self):
        self.ax.set_title("My Hopfield Network", fontsize=20, color="darkblue",
                          fontweight="bold", fontstyle="italic", fontfamily="serif")
        energy = self.hopfield.getEnergy()
        self.ax.text(1, 1, f"Energy: {energy:.2f}", ha="center", va="center",
                     fontsize=12, color="black", fontweight="bold", fontstyle="italic")

    def next(self, event):
        self.hopfield.next_state()
        if self.hopfield.is_stable:
            print("Converged")
            self.bnext.label.set_text("Converged")
            self.bnext.label.set_color("green")
        else:
            self.draw_graph()
            plt.draw()

    def reset(self, event):
        # remove the "converged" label
        self.setup_buttons()
        self.hopfield = Hopfield(self.N)
        self.draw_graph()
        plt.draw()  # Use plt.draw() instead of plt.show() to update the current figure

    def run(self):
        # Draw the initial graph
        self.draw_graph()
        # Connect the click event to the handler
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()

    def make_gif(self, event):
        """
        Create a gif of the network state
        """
        images = []

        for i in range(20):
            # update the GUI and redraw the graph
            self.reset(None)
            self.draw_graph()
            plt.draw()
            plt.pause(0.1)  # pause a bit for the plot to update

            # save the current figure to an image file
            self.fig.savefig(f"{i}.png")

            # load the image file
            img = Image.open(f"{i}.png")
            # make it transparent
            img = img.convert("RGBA")
            datas = img.getdata()
            newData = []
            for item in datas:
                if item[0] == 255 and item[1] == 255 and item[2] == 255:
                    newData.append((255, 255, 255, 0))
                else:
                    newData.append(item)
            img.putdata(newData)
            images.append(img)

        images[0].save('movie.gif', save_all=True,
                       append_images=images[1:], optimize=False, duration=100, loop=0)


if __name__ == "__main__":
    gui = GUI(10)
    gui.run()
