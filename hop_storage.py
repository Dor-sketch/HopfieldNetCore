"""
This class is used to store patterns and to add new patterns to the storage.
"""

import numpy as np

# decorator to print the class inner state
def print_state(func):
    def wrapper(self, *args, **kwargs):
        print(f"Stored patterns: {self.stored}")
        print(f"Added patterns: {self.added}")
        return func(self, *args, **kwargs)
    return wrapper

class HopStorage:
    def __init__(self):
        self.stored = []
        self.added = []

    @print_state
    def add(self, pattern):
        self.added.append(pattern.copy())


    @print_state
    def store(self):
        for pattern in self.added:
            if not any(np.array_equal(pattern, stored) for stored in self.stored):
                self.stored.append(pattern.copy())
        self.added = self.added[:0]


    def resert(self):
        self.stored = []
        self.added = []

    @print_state
    def get_stored(self):
        stored_and_spurious = self.stored.copy()  # create a copy of self.stored
        # add spurious patterns
        # spurious patterns are the stored patterns with some neurons flipped
        # and also combinations of stored patterns resulting in spurious patterns
        # TODO: implement spurious patterns generation. Is needed?
        return stored_and_spurious
