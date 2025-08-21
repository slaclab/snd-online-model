import numpy as np


class TestInterface:
    """
    A mock interface for testing purposes.
    It simulates the behavior of an interface that provides input variables.
    """
    def get_input_variables(self, input_variables: list) -> dict:
        """
        Simulate getting input variables by returning a dictionary with random values
        within the specified ranges of the input variables.
        """
        return {
            var.name: np.random.uniform(var.value_range[0], var.value_range[1])
            for var in input_variables
        }
