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
        energy_idx = self.input_names.index("energy")
        delay_idx = self.input_names.index("delay")
        self.snd = self.initialize_model(
            energy=self.input_variables[energy_idx].default_value,
            delay=self.input_variables[delay_idx].default_value,
        )
        self.pv_map = None

    def initialize_model(self, energy=10000, delay=0):
        """Initialize the model with default values for energy and delay."""
        # the following line places motors to "default positions" based on the photon energy
        # and delay. Those positions can be compared to the current PV readbacks.
        self.snd = SND(energy, delay)
        return self.snd

    def input_transform(self, input_dict):
        """Transform input dictionary values from PV units to sim units."""
        return {
            name: value * self.pv_map["unit_conversion"][name]
            for name, value in input_dict.items()
        }

    def output_transform(self, output_dict):
        """Transform output dictionary values from sim units to PV units."""
        # TODO: Not sure if any transformation is needed here. Remove if not needed.
        return output_dict

    def _evaluate(self, input_dict, transform=True):
        if transform:
            input_dict = self.input_transform(input_dict)

        # The following serves as a starting point where the PVs can be used for defining
        # motor positions. However, this is currently incompatible with tight input ranges.
        for name, motor in self.snd.motor_dict.items():
            default_pos = motor.wm()
            pv_pos = input_dict[name]
            offset = pv_pos - default_pos  # TODO: what is the offset for?
            motor.mv(pv_pos)

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

        if transform:
            output_dict = self.output_transform(output_dict)

        return output_dict
