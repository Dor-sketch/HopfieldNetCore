"""
A class to represent a map of cities
"""
import math
import matplotlib.pyplot as plt
import numpy as np

CITY_SET_A = {'A': (0.25, 0.16),
              'B': (0.85, 0.35),
              'C': (0.65, 0.24),
              'D': (0.7, 0.5),
              'E': (0.15, 0.22),
              'F': (0.25, 0.78),
              'G': (0.4, 0.45),
              'H': (0.9, 0.65),
              'I': (0.55, 0.9),
              'J': (0.6, 0.28)}


class Map:
    """
    A class to represent a map of cities
    """

    def __init__(self, city_set=None):
        if city_set is None:
            city_set = CITY_SET_A
        self.city_set = city_set
        self.cities = list(self.city_set.keys())
        self.index = 0
        self.isValid = False
        self.count = 0

    def __getitem__(self, city_a):
        # Return a dictionary that maps other cities to their distances from city_a
        return {city_b: self._distance(self.city_set[city_a], self.city_set[city_b]) for city_b in self.city_set}

    def _distance(self, coords_a, coords_b):
        # Calculate the Euclidean distance between two cities
        return math.sqrt((coords_a[0] - coords_b[0]) ** 2 + (coords_a[1] - coords_b[1]) ** 2)

    def plot_route(self, route):
        # Get the coordinates of the cities in the route
        route_coords = [self.city_set[city] for city in route]
        # Unpack the city coordinates
        x, y = zip(*route_coords)
        # Calculate the total route distance
        total_distance = sum(self._distance(
            route_coords[i], route_coords[i + 1]) for i in range(len(route_coords) - 1))

        # Create a new figure if it doesn't exist, otherwise clear the existing one
        fig, ax = plt.subplots()
        is_valid = True
        # Plot the cities
        ax.plot(x, y, 'o-', color='skyblue', linewidth=2, markersize=10)
        # Plot the first city
        ax.plot(*route_coords[0], 'ro')  # First city in red
        # Set the title with the total route distance
        ax.set_title(f'Route ({total_distance:.2f} Light Years)')
        # check for missing cities or cities that are not in the city set
        if len(route) != len(self.city_set) or not all(city in self.city_set for city in route):
            is_valid = False
            city_list = list(self.city_set.keys())
            missing_cities = [city for city in city_list if city not in route]
            ax.text(0.5, 0.5, f'Missing Cities: {missing_cities}', transform=ax.transAxes, fontsize=12,
                    verticalalignment='center', horizontalalignment='center', color='red')
            for city in missing_cities:
                ax.text(*self.city_set[city], city, color='red', fontsize=12)

        repeated_cities = [city for city in route if route.count(city) > 1]
        if repeated_cities:
            is_valid = False
            # mark the repeated cities with the number of times they are repeated
            for city in repeated_cities:
                ax.text(*self.city_set[city],
                        route.count(city), color='red', fontsize=12)

        if is_valid:
            self.isValid = True
            # congrats message
            ax.text(0.5, 0.5, 'Congratulations! You have a valid route :)', transform=ax.transAxes, fontsize=12,
                    verticalalignment='center', horizontalalignment='center', color='green')
        # Draw the plot and pause for a short while to allow the window to update
        plt.draw()
        plt.pause(0.1)
        plt.show()

    def __iter__(self):
        return self

    def is_valid(self):
        return self.isValid

    def __next__(self):
        if self.index >= len(self.cities):
            self.index = 0
            raise StopIteration
        result = self.cities[self.index]
        self.index += 1
        return result

    def __str__(self):
        return str(self.city_set)

    def __len__(self):
        return len(self.cities)

    def take_snapshot(self, route):
        # Get the coordinates of the cities in the route
        route_coords = [self.city_set[city] for city in route]
        # Unpack the city coordinates
        x, y = zip(*route_coords)
        # Calculate the total route distance
        total_distance = sum(self._distance(
            route_coords[i], route_coords[i + 1]) for i in range(len(route_coords) - 1))

        # Create a new figure if it doesn't exist, otherwise clear the existing one
        fig, ax = plt.subplots()
        plt.style.use('dark_background')  # Use dark background
        ax.grid(True, linestyle='-', color='0.75')  # Add grid
        is_valid = True

        # Plot the cities with gradient color and increased size
        colors = plt.cm.viridis(np.linspace(0, 1, len(route_coords)))
        for i in range(len(route_coords) - 1):
            ax.plot(*zip(route_coords[i], route_coords[i + 1]),
                    'o-', color=colors[i], linewidth=2, markersize=12)

        # Plot the first city
        ax.plot(*route_coords[0], 'ro', markersize=14)  # First city in red

        # Set the title with the total route distance
        ax.set_title(f'Route ({total_distance:.2f} Light Years)',
                     fontname='Comic Sans MS', fontsize=14)

        # Check for missing cities or cities that are not in the city set
        if len(route) != len(self.city_set) or not all(city in self.city_set for city in route):
            is_valid = False
            city_list = list(self.city_set.keys())
            missing_cities = [city for city in city_list if city not in route]
            for city in missing_cities:
                ax.text(*self.city_set[city], city, color='red',
                        fontsize=12, fontname='Comic Sans MS')

        repeated_cities = [city for city in route if route.count(city) > 1]
        if repeated_cities:
            is_valid = False
            # Mark the repeated cities with the number of times they are repeated
            for city in repeated_cities:
                ax.text(*self.city_set[city], route.count(city),
                        color='red', fontsize=12, fontname='Comic Sans MS')

        if is_valid:
            self.isValid = True
            # Congrats message
            ax.text(0.5, 0.5, 'Congratulations! You have a valid route :)', transform=ax.transAxes, fontsize=12,
                    verticalalignment='center', horizontalalignment='center', color='green', fontname='Comic Sans MS')

        # Save the figure instead of showing it
        plt.savefig('route' + str(self.count).zfill(3) + '.png')
        self.count += 1
