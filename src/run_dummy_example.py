import logging
import time
import collections
from model.snd_model import SNDModel
from interface.test_interface import TestInterface

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MultiLineDict(collections.UserDict):
    def __str__(self):
        return "\n" + "\n".join(f"{k} = {v}" for k, v in self.data.items())


def setup():
    """
    Run an example of the SNDModel.
    """
    # Import the model from LUME-Model config file
    # and instantiate the SNDModel using default configuration
    snd_model = SNDModel("model/snd_model.yml")

    # Instantiate an interface for I/O operations
    interface = TestInterface()

    return snd_model, interface, snd_model.input_variables

def run_iteration(snd_model, interface, input_vars):
    """
     Run a single iteration of the example, evaluating the SNDModel with input from the interface.
    """
    # Get the input variable from the interface
    input_dict = interface.get_input_variables(input_vars)
    logger.debug("Input values: %s", MultiLineDict(input_dict))

    # Check if energy has changed too much from the default value
    energy_change_threshold = 100  # eV
    if abs(input_dict['energy'] - snd_model.input_variables[0].default_value) > energy_change_threshold:
        logger.info("Energy has changed significantly, reinstantiating model.")
        logger.info(f"Default energy: {snd_model.input_variables[0].default_value}.")
        logger.info(f"Input energy: {input_dict['energy']}.")
        # TODO: Do the same for delay. For now, we use the default value.
        snd_model.initialize_model(input_dict['energy'], snd_model.input_variables[1].default_value)
        # Set new default energy
        snd_model.input_variables[0].default_value = input_dict['energy']
        logger.info(f"New default energy is {snd_model.input_variables[0].default_value}.")

        # Update default value for each motor /each input based on new energy
        # Only needed for dummy example to sample from a reasonable range
        for name, motor in snd_model.snd.motor_dict.items():
            snd_model.input_variables[snd_model.input_names.index(name)].default_value = motor.wm()
            snd_model.input_variables[snd_model.input_names.index(name)].value_range = [motor.wm()-0.0001, motor.wm()+0.0001]

        # Get new input variables from the interface, skipping the first two (energy and delay)
        input_dict = interface.get_input_variables(input_vars[2:])
        logger.debug("Updated input values: %s", MultiLineDict(input_dict))

    # Evaluate the model with the random input
    output = snd_model.evaluate(input_dict)
    logger.debug("Output values: %s", MultiLineDict(output))


if __name__ == "__main__":
    snd_model, interface, input_pvs = setup()
    logger.info("Starting SNDModel example with the test interface...")
    while True:
        try:
            run_iteration(snd_model, interface, input_pvs)
            time.sleep(5)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Exiting.")
            exit(0)
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            logger.info("Retrying in 5 seconds...")
            time.sleep(5)
