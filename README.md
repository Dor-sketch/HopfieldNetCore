# Hopfield Network Visualizer 🧠

This tool is a graphical user interface (GUI) designed to visualize and interact with a Hopfield network, a type of recurrent artificial neural network used as a content-addressable memory system. The GUI provides a dynamic and interactive way to explore the network's behavior, including its state transitions, energy landscape, and pattern storage and retrieval capabilities.

This is an ongoing project, and the GUI is being developed to support educational purposes, allowing users to understand the network's dynamics and properties through visual feedback and interactive controls. Custom implementations of the Hopfield network's functionalities are used to provide a comprehensive and intuitive learning experience. For example, see the TSP (Travelling Salesman Problem) file for a Hopfield network implementation for the TSP.


<p align="center">
  <img src="images/test/quality_animation_white.gif" alt="Network Animation" width="600">
</p>

## Key Concepts 🗝️

- **Weights Matrix (J)**: Represents the connections between neurons. Jᵢⱼ denotes the weight between neurons i and j.
- **Neuron States (sᵢ)**: The state of neuron i, which can be either 1 or -1.
- **Energy**: Reflects the current state's stability. Lower energy indicates a more stable or converged state.
- **Overlap Value**: Measures the similarity between the network's current state and a stored pattern, aiding in pattern recognition.

## Features

- **Interactive Network Visualization**: Visualize the network's state, adjust it in real time, and observe how it evolves.
- **Pattern Management**: Add, store, and view patterns within the network to understand associative memory functionalities.
- **Dynamic Controls**: Utilize interactive buttons to manipulate the network's state, analyze its properties, and explore theoretical concepts.
- **Educational Insights**: Access detailed explanations and mathematical equations that underpin the network's operations, enhancing understanding of neural network dynamics.
- **Advanced Visualizations**: Explore the network in 3D and create GIFs to visualize the network's state changes over time.

<p align="center">
  <img src="images/energy_func_visu.png" alt="Energy Function Visualization" width="400">

  <img src="images/land.png" alt="Landscape Visualization" width="400">

  <img src="images/land2.png" alt="Landscape Visualization" width="400">

  <img src="images/v.png" alt="3D Visualization" width="400">
</p>

## How to Use 🛠️

1. **Initialization**: Launch the GUI to start with a Hopfield network of a specified size.
2. **Interaction**: Use the GUI buttons to interact with the network, applying operations like updating states, resetting, and storing patterns.
3. **Analysis**: Observe the network's behavior through visual feedback, understanding the impact of your interactions on its state and properties.

| Weights Before Training | Weights After Training |
|-------------------------|------------------------|
| ![alt text](images/weights_before.png) | ![alt text](images/weights_after.png) |

## Installation 📦

Ensure you have Python and necessary libraries (Matplotlib, NetworkX, PIL) installed. Clone the repository and run the script to launch the GUI.

## Contributing 🤝

Contributions are welcome! If you have suggestions for improvements or new features, please feel free to submit pull requests or open issues.

## License 📄

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.