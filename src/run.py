import argparse
import logging
import time
import collections
import numpy as np
import mlflow
from mlflow_run import MLflowRun
from model.snd_model import SNDModel


logging.basicConfig(
    handlers=[logging.FileHandler("snd_run.log"), logging.StreamHandler()],
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


class MultiLineDict(collections.UserDict):
    def __str__(self):
        return "\n" + "\n".join(f"{k} = {v}" for k, v in self.data.items())


def get_interface(interface_name, pvname_list=None):
    if interface_name == "test":
        from interface.test_interface import TestInterface

        return TestInterface()
    elif interface_name == "epics":
        from interface.epics_interface import EPICSInterface

        return EPICSInterface(pvname_list)
    else:
        raise ValueError(f"Unknown interface: {interface_name}")


def get_input_vars(snd_model, interface_name):
    if interface_name == "test":
        # Use model input variable objects
        return snd_model.input_variables
    elif interface_name == "epics":
        # Use PV names from the model's pv_map
        return [snd_model.pv_map["name"][n] for n in snd_model.input_names]
    else:
        raise ValueError(f"Unknown interface: {interface_name}")


def pv_mapping():
    """Loads  the PV mapping from json file."""
    import json

    with open("model/pv_mapping.json", "r") as file:
        pv_mapping = json.load(file)
    return pv_mapping


def run_iteration(snd_model, interface, input_vars, interface_name):
    """
    Run a single iteration of the SNDModel evaluation using the specified interface.

    Parameters
    ----------
    snd_model : SNDModel
        The SNDModel instance to evaluate.
    interface : object
        The interface instance (TestInterface or EPICSInterface) for input retrieval.
    input_vars : list
        List of input variable names or PVs, depending on the interface.
    interface_name : str
        The name of the interface to use ('test' or 'epics').

    Returns
    -------
    None
    """
    # Get the input variable from the interface
    input_dict = interface.get_input_variables(input_vars)
    if interface_name == "epics":
        # Map PVs back to model input names
        logger.debug(f"Raw input values from EPICS: {MultiLineDict(input_dict)}")
        posixseconds = int(max(d["posixseconds"] for d in input_dict.values()))
        input_dict = {
            snd_model.input_names[i]: input_dict[pv]["value"]
            for i, pv in enumerate(input_vars)
        }
    logger.debug("Input values: %s", MultiLineDict(input_dict))

    # Check if t1_tth or delay have changed too much from the default value
    # if so, we need to reinitialize the model and obtain new defaults and ranges
    delay_change_threshold = 0.1  # ps
    delay_idx = snd_model.input_names.index("delay")
    t1_tth_idx = snd_model.input_names.index("t1_tth")
    theta_change_threshold = 3.49e-06 # radians
    t1_tth_input_sim_units = np.deg2rad(input_dict["t1_tth"]) if interface_name == "epics" else input_dict["t1_tth"]

    if (
        abs(t1_tth_input_sim_units - snd_model.input_variables[t1_tth_idx].default_value)
        > theta_change_threshold
        or abs(input_dict["delay"] - snd_model.input_variables[delay_idx].default_value)
        > delay_change_threshold
    ):
        logger.info("t1_tth or delay has changed significantly, reinstantiating model.")
        logger.info(
            f"Old t1_tth: {snd_model.input_variables[t1_tth_idx].default_value}."
        )
        logger.info(
            f"New t1_tth: {t1_tth_input_sim_units}."
        )
        logger.info(
            f"Old delay: {snd_model.input_variables[delay_idx].default_value}."
        )
        logger.info(
            f"New delay: {input_dict['delay']}."
        )
        snd_model.initialize_model(
            two_theta=t1_tth_input_sim_units, delay=input_dict["delay"]
        )

        # Set new default energy and delay
        snd_model.input_variables[t1_tth_idx].default_value = t1_tth_input_sim_units
        snd_model.input_variables[delay_idx].default_value = input_dict["delay"]

        # Update default value for each motor/each input based on new energy
        # this is needed here for validation of the input range (will only throw a warning)
        logger.info("Updating default values for all motors based on new t1_tth/delay.")
        # Disable validation for all inputs temporarily, since these are not scaled to simulation units yet
        snd_model.input_validation_config = {
            k: "none" for k in snd_model.input_names
        }
        for name, motor in snd_model.snd.motor_dict.items():
            snd_model.input_variables[
                snd_model.input_names.index(name)
            ].default_value = motor.wm()
            snd_model.input_variables[snd_model.input_names.index(name)].value_range = [
                motor.wm() - 0.0001,
                motor.wm() + 0.0001,
            ]

        if interface_name == "test":
            # Get new input variables from the interface, skipping t1_tth and delay
            input_dict = interface.get_input_variables(
                [x for x in input_vars if x.name not in ["t1_tth", "delay"]]
            )
            input_dict["t1_tth"] = t1_tth_input_sim_units # add back default t1_tth
            logger.debug("Updated input values: %s", MultiLineDict(input_dict))

    # Evaluate the model with the input
    snd_model.input_validation_config = {k: "warn" for k in snd_model.input_names}
    if interface_name == "epics":
        # Transform input from PV units to simulation units
        input_dict = snd_model.input_transform(input_dict)
        logger.debug("Transformed input values: %s", MultiLineDict(input_dict))

    # Evaluate the model
    output = snd_model.evaluate(input_dict)

    # Log input after transformation and output
    # one line to log at same timestamp
    mlflow.log_metrics(
        input_dict | output,
        timestamp=(posixseconds * 1000 if interface_name == "epics" else None),
    )
    logger.debug("Output values: %s", MultiLineDict(output))


def main():
    """
    Main entry point for running the SNDModel application with CLI interface selection.

    Parses command-line arguments to select the interface, initializes the model and interface,
    and runs the evaluation loop.

    You can run the script with:
        python run.py --interface test
        python run.py --interface epics

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description="Run SNDModel with selected interface."
    )
    parser.add_argument(
        "--interface",
        "-i",
        choices=["test", "epics"],
        required=True,
        help="Interface to use",
    )
    args = parser.parse_args()
    logger.info("Running with interface: %s", args.interface)
    snd_model = SNDModel("model/snd_model.yml")
    snd_model.pv_map = pv_mapping()
    input_vars = get_input_vars(snd_model, args.interface)
    interface = get_interface(
        args.interface, input_vars if args.interface == "epics" else None
    )

    with MLflowRun() as run:
        while True:
            try:
                run_iteration(snd_model, interface, input_vars, args.interface)
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received. Exiting.")
                exit(0)
            except Exception as e:
                logger.error(f"An error occurred: {e}")
                logger.info("Retrying in 5 seconds...")
                time.sleep(5)


if __name__ == "__main__":
    main()
