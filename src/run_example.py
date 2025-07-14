import logging
import time
import collections
import numpy as np
from model.snd_model import SNDModel

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MultiLineDict(collections.UserDict):
    def __str__(self):
        return "\n" + "\n".join(f"{k} = {v}" for k, v in self.data.items())


def run_example():
    """
    Run an example of the SNDModel.
    """
    # Import the model from LUME-Model config file
    snd_model = SNDModel("model/snd_model.yml")
    input_variables = snd_model.input_variables

    # Create a dummy input dictionary
    # from each variable in input_variables, randomly generate a value from the variable's range
    random_input = {
        var.name: np.random.uniform(var.value_range[0], var.value_range[1]) for var in input_variables
    }
    logger.debug("Input values: %s", MultiLineDict(random_input))

    # Evaluate the model with the random input
    output = snd_model.evaluate(random_input)
    logger.debug("Output values: %s", MultiLineDict(output))


if __name__ == "__main__":
    while True:
        try:
            run_example()
            time.sleep(5)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Exiting.")
            exit(0)
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            logger.info("Retrying in 5 seconds...")
            time.sleep(5)
