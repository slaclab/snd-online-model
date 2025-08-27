import numpy as np
import logging
from lume_model.base import LUMEBaseModel
from lcls_beamline_toolbox.models.split_and_delay_motion import SND
from pydantic import ConfigDict

logger = logging.getLogger(__name__)

inputs = np.zeros(22)


class SNDModel(LUMEBaseModel):
    model_config = ConfigDict(extra="allow")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        t1_tth_idx = self.input_names.index("t1_tth")
        delay_idx = self.input_names.index("delay")
        self.snd = self.initialize_model(
            two_theta=self.input_variables[t1_tth_idx].default_value,
            delay=self.input_variables[delay_idx].default_value,
        )
        self.pv_map = None

    def initialize_model(self, two_theta=37.674, delay=0):
        """
        Initialize the model with default values for energy and delay.

        Parameters
        ----------
        energy : float, optional
            Photon energy value to initialize the model (default is 10000).
        delay : float, optional
            Delay value to initialize the model (default is 0).

        Returns
        -------
        SND
            The initialized SND model instance.
        """
        self.snd = SND(two_theta=np.deg2rad(two_theta), delay=delay)
        return self.snd

    def input_transform(self, input_dict):
        """
        Transform input dictionary values from PV units to simulation units.

        Parameters
        ----------
        input_dict : dict
            Dictionary of input variable names and their values in PV units.

        Returns
        -------
        dict
            Dictionary of input variable names and their values in simulation units.
        """
        return {
            name: value * self.pv_map["unit_conversion"][name]
            for name, value in input_dict.items()
        }

    def output_transform(self, output_dict):
        """
        Transform output dictionary values from simulation units to PV units.

        Parameters
        ----------
        output_dict : dict
            Dictionary of output variable names and their values in simulation units.

        Returns
        -------
        dict
            Dictionary of output variable names and their values in PV units.
        """
        # TODO: Not sure if any transformation is needed here. Remove if not needed.
        return output_dict

    def _evaluate(self, input_dict, transform=True):
        """
        Evaluate the SND model with the given input dictionary.

        Parameters
        ----------
        input_dict : dict
            Dictionary of input variable names and their values.
        transform : bool, optional
            Whether to transform input values from PV units to simulation units (default is True).

        Returns
        -------
        dict
            Dictionary of output variable names and their evaluated values.
        """
        # The following serves as a starting point where the PVs can be used for defining
        # motor positions. However, this is currently incompatible with tight input ranges.
        for name, motor in self.snd.motor_dict.items():
            motor.mv(input_dict[name])

        self.snd.propagate_delay()
        self.snd.propagate_cc()
        output_dict = {
            "t1_dh_sum": self.snd.get_t1_dh_sum(),
            "dd_sum": self.snd.get_dd_sum(),
            "t4_dh_sum": self.snd.get_t4_dh_sum(),
            "do_sum": self.snd.get_do_sum(),
            "dd_cx": self.snd.get_dd_cx(),
            "dd_cy": self.snd.get_dd_cy(),
            "do_cx": self.snd.get_do_cx(),
            "do_cy": self.snd.get_do_cy(),
            "IP_sum": self.snd.get_IP_sum(),
            "IP_cx": self.snd.get_IP_cx(),
            "IP_cy": self.snd.get_IP_cy(),
        }

        return output_dict
