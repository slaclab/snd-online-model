import logging
import time
import collections
import numpy as np
from model.snd_model import SNDModel
from interface.epics_interface import EPICSInterface

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
    # Get PV names from the mapping
    input_pvs = [snd_model.pv_map[n] for n in snd_model.input_names]

    # Instantiate an interface for I/O operations
    interface = EPICSInterface()

    return snd_model, interface, input_pvs

def run_iteration(snd_model, interface, input_pvs):
    """
     Run a single iteration of the example, evaluating the SNDModel with input from the interface.
    """
    # Get the input variable from the interface
    input_dict = interface.get_input_variables(input_pvs)
    input_dict = {snd_model.input_names[i]: input_dict[pv] for i, pv in enumerate(input_pvs)}
    logger.debug("Input values: %s", MultiLineDict(input_dict))

    # Check if energy has changed too much from the default value
    energy_change_threshold = 100  # eV
    if abs(input_dict['energy'] - snd_model.input_variables[0].default_value) > energy_change_threshold:
        logger.info("Energy has changed significantly, reinstantiating model.")
        logger.info(f"Default energy: {snd_model.input_variables[0].default_value}.")
        logger.info(f"Input energy: {input_dict['energy']}.")
        # TODO: Do the same for delay. For now use the input value for delay
        snd_model.initialize_model(input_dict['energy'], input_dict['delay'])
        # Set new default energy and delay
        snd_model.input_variables[0].default_value = input_dict['energy']
        snd_model.input_variables[1].default_value = input_dict['delay']
        logger.info(f"New default energy is {snd_model.input_variables[0].default_value}.")
        logger.info(f"New delay is {snd_model.input_variables[1].default_value}.")

        # Update default value for each motor /each input based on new energy
        # this is needed here for validation of the input range (will only throw a warning)
        for name, motor in snd_model.snd.motor_dict.items():
            snd_model.input_variables[snd_model.input_names.index(name)].default_value = motor.wm()
            snd_model.input_variables[snd_model.input_names.index(name)].value_range = [motor.wm()-0.0001, motor.wm()+0.0001]

    # Evaluate the model with the random input
    output = snd_model.evaluate(input_dict)
    logger.debug("Output values: %s", MultiLineDict(output))


if __name__ == "__main__":
    snd_model, interface, input_pvs = setup()
    logger.info("Starting SNDModel example with EPICS interface...")
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
