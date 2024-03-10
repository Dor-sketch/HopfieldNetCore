"""
This module contains the GUI class that is used to visualize the Hopfield network
"""

import random # for random colors
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
import networkx as nx
from hop_proof import proof_concept, generate_equation
from hop_graph import HopGraph
from hop_styles import HopStyles
from hop_storage import HopStorage

BUTTONS_COLOR = "lightblue"


class GUI:
    """
    This class implements the GUI for the Hopfield network.
    """

    def __init__(self, N, hopfield):
        self.hopfield = hopfield(N)
        self.from_node = None
        self.N = N*N
        self.fig, self.ax = plt.subplots()
        self.graph = nx.Graph()
        self.init_graph()
        self.bnext = None
        self.breset = None
        self.setup_buttons()
        self.patterns = []
        self.storage = HopStorage()
        # make window bigger
        self.fig.set_size_inches(14, 8)

    def setup_buttons(self):
        """
        Add buttons to the plot. for next state and reset
        """
        buttons = [
            {
                "position": [0.81, 0.05, 0.1, 0.07],
                "label": "Next",
                "callback": self.next,
            },
            {
                "position": [0.7, 0.05, 0.1, 0.07],
                "label": "Reset",
                "callback": self.reset,
            },
            {
                "position": [0.59, 0.05, 0.1, 0.07],
                "label": "Weights",
                "callback": self.weights,
            },
            {
                "position": [0.48, 0.05, 0.1, 0.07],
                "label": "Theory",
                "callback": self.theory,
            },
            {
                "position": [0.37, 0.05, 0.1, 0.07],
                "label": "Update Eq",
                "callback": self.weights_eq,
            },
            {
                "position": [0.26, 0.05, 0.1, 0.07],
                "label": "Nothing to Store",
                "callback": self.store,
            },
            {"position": [0.15, 0.05, 0.1, 0.07], "label": "Add", "callback": self.add},
            {
                "position": [0.04, 0.05, 0.1, 0.07],
                "label": "Overlap",
                "callback": self.get_overlap,
            },
            {
                "position": [0.04, 0.15, 0.1, 0.07],
                "label": "Energy",
                "callback": self.energy,
            },
            {
                "position": [0.04, 0.25, 0.1, 0.07],
                "label": "View Stored",
                "callback": self.view_stored,
            },
            {
                "position": [0.04, 0.35, 0.1, 0.07],
                "label": "Make Gif",
                "callback": self.make_gif,
            },
            {
                "position": [0.04, 0.45, 0.1, 0.07],
                "label": "3D",
                "callback": self.plot_three_d,
            },
            {
                "position": [0.04, 0.55, 0.1, 0.07],
                "label": "Plot TSP (WIP)",
                "callback": self.plot_tsp,
            },
        ]

        for button in buttons:
            ax = plt.axes(button["position"])
            b = Button(ax, button["label"], color=BUTTONS_COLOR, hovercolor="0.975")
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
            if button["label"] == "3D":
                self.b3d = b
            if button["label"] == "Plot TSP (WIP)":
                self.btsp = b
            b.label.set_fontstyle("italic")
            b.label.set_fontfamily("serif")

    def plot_tsp(self, event):
        """
        Display the TSP solution (a route) in a new figure
        """
        # Assuming city_coords is a dictionary mapping city names to their coordinates
        self.hopfield.road_map.plot_route(self.hopfield.get_route())


    def plot_three_d(self, event):
        with HopGraph(self.hopfield) as h:
            h.plot_three_d(self.storage.get_stored())

    def view_stored(self, event):
        """
        Display the stored patterns a new figure with small network graphs
        """
        with HopGraph(self.hopfield) as h:
            h.view_stored(self.storage.get_stored())

    def get_overlap(self, event):
        with HopGraph(self.hopfield) as h:
            h.get_overlap(self.storage.get_stored())

    def add(self, event):
        self.storage.add(self.hopfield.neurons.copy())
        self.badd.label.set_text(f"Add More")
        self.bstore.label.set_text("Store {}".format(len(self.storage.added)))
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
            # check if right click
            if event.button == 3:
                # mark as from and get other node to remove edge
                print(f"Right click on node {clicked_node}")
                if self.from_node is None:
                    self.from_node = clicked_node
                    # change color of the node

                else:
                    # print the values of the nodes synaptics on the clicked node
                    # print it on weights button
                    self.bweights.label.set_text(
                        f"W[{self.from_node}][{clicked_node}]: {self.hopfield.weights[self.from_node][clicked_node]:.2f}"
                    )
                    self.bweights.label.set_color("red")
                    self.bweights.label.set_fontsize(10)
                    self.bweights.label.set_fontstyle("italic")
                    self.from_node = None
            else:
            # Change the state of the clicked node
                self.hopfield.neurons[clicked_node] *= -1
            self.draw_graph()
            plt.draw()

    def store(self, event):
        self.hopfield.store_patterns(self.storage.added)
        self.bstore.label.set_text("Nothing to store")
        self.badd.label.set_text("Add")
        self.bstore.label.set_color("black")
        self.badd.label.set_color("black")
        self.storage.store()
        self.draw_graph()
        plt.draw()

    def theory(self, event):
        # open new figure like old math text books make it scrollable with blackboard style
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 14)
        fig.set_facecolor("black")
        ax.set_facecolor("black")
        ax.set_title(
            "Why Converging? (WIP)",
            fontsize=20,
            color="lightblue",
            fontweight="bold",
            fontstyle="italic",
            fontfamily="serif",
        )
        p = proof_concept()
        # add the equation to the figure like whiteboard
        ax.text(
            0.5,
            0.5,
            p,
            va="center",
            fontsize=10,
            color="white",
            fontstyle="italic",
            fontfamily="serif",
            ha="center",
        )
        ax.axis("off")
        plt.show()

    def weights_eq(self, event):
        old_state = self.hopfield.neurons
        self.hopfield.next_state()
        generate_equation(
            old_state, self.hopfield.neurons, self.hopfield.weights, self.hopfield.t - 1
        )
        self.draw_graph()
        plt.draw()

    def energy(self, event):
        """
        Display the energy of the current state
        """
        energy = self.hopfield.get_energy()
        print(f"Energy: {energy}")

    def weights(self, event):
        """
        Display the weights of the network
        """
        # open new figure
        with HopGraph(self.hopfield) as h:
            h.weights()

    def init_graph(self):
        """
        Draw the graph with the current state of the neurons
        """

        N = self.N

        # Create nodes
        for i in range(N):
            self.graph.add_node(i)

        print(f'self.hopfield.weights: {self.hopfield.weights}')
        # Create edges
        for i in range(N):
            for j in range(i + 1, N):
                weight = self.hopfield.weights[i][j]
                self.graph.add_edge(i, j, weight=weight, alpha=0.5, width=weight * 10)


        self.pos = nx.spring_layout(self.graph, seed=42, iterations=100)
        self.draw_graph()

    def draw_graph(self):
        """
        update the graph with the current state of the neurons
        """
        self.ax.clear()
        self.update_labels()
        with HopStyles(self.hopfield) as h:
            node_colors = h.get_nodes_colors()
            node_sizes = h.get_nodes_sizes()
            edges_colors, edge_widths = h.get_edges_style()

        if self.from_node is not None:
            node_colors[self.from_node] = "red"
            edges_colors = ["red" if edge[0] == self.from_node else "black" for edge in self.graph.edges]

        edges_colors = [(random.uniform(0.1, 0.2), 0,
                         random.uniform(0.1, 0.2), 0.1)] * len(self.graph.edges)  # Initialize with default color

        for i, edge in enumerate(self.graph.edges):
            if self.hopfield.neurons[edge[0]] == 1:
                # get random violet color
                color = (random.uniform(0.4, 0.6), 0,
                         random.uniform(0.4, 0.6), 0.5)
                edges_colors[i] = color

        nx.draw_networkx(  # Draw the graph
            self.graph,
            self.pos,
            node_color=node_colors,
            with_labels=True,
            ax=self.ax,
            width=edge_widths,
            # node_size=node_sizes,
            node_size= 2000 / self.N,
            edge_color=edges_colors,
        )

    def update_labels(self):
        self.ax.set_title(
            "My Hopfield Network",
            fontsize=20,
            color="darkblue",
            fontweight="bold",
            fontstyle="italic",
            fontfamily="serif",
        )
        energy = self.hopfield.get_energy()
        self.ax.text(
            0.5,
            1,
            f"Energy: {energy:.5f}",

            fontsize=12,
            color="black",
            fontweight="bold",
            fontstyle="italic",
        )

    def next(self, event):
        self.hopfield.next_state()
        if self.hopfield.is_stable():
            print("Converged")
            self.hopfield.road_map.plot_route(self.hopfield.get_route())
            self.bnext.label.set_text("Converged")
            self.bnext.label.set_color("green")
        else:
            self.draw_graph()
            plt.draw()

    def reset(self, event):
        # remove the "converged" label
        self.setup_buttons()
        self.storage.resert()
        self.hopfield.reset()
        self.draw_graph()
        plt.draw()  # Use plt.draw() instead of plt.show() to update the current figure

    def run(self):
        # Draw the initial graph
        self.draw_graph()
        # Connect the click event to the handler
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
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

        images[0].save(
            "movie.gif",
            save_all=True,
            append_images=images[1:],
            optimize=False,
            duration=100,
            loop=0,
        )
