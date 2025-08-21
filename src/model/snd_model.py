import numpy as np
import logging
import json
from lume_model.base import LUMEBaseModel
from lume_model.variables import ScalarVariable
from lcls_beamline_toolbox.models.split_and_delay_motion import SND
from pydantic import ConfigDict

logger = logging.getLogger(__name__)

inputs = np.zeros(22)

class SNDModel(LUMEBaseModel):
    model_config = ConfigDict(extra="allow")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.snd = self.initialize_model(
            energy=self.input_variables[0].default_value,
            delay=self.input_variables[1].default_value
        )

    def initialize_model(self, energy=10000, delay=0):
        """Initialize the model with default values for energy and delay."""
        # the following line places motors to "default positions" based on the photon energy
        # and delay. Those positions can be compared to the current PV readbacks.
        self.snd = SND(energy, delay)
        return self.snd

    def pv_mapping(self):
        """Loads  the PV mapping from json file."""
        with open('pv_mapping.json', 'r') as file:
            pv_mapping = json.load(file)
        return pv_mapping

    def _evaluate(self, input_dict):
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
            't1_dh_sum': self.snd.get_t1_dh_sum(),
            'dd_sum': self.snd.get_dd_sum(),
            't4_dh_sum': self.snd.get_t4_dh_sum(),
            'do_sum': self.snd.get_do_sum(),
            'dd_cx': self.snd.get_dd_cx(),
            'dd_cy': self.snd.get_dd_cy(),
            'do_cx': self.snd.get_do_cx(),
            'do_cy': self.snd.get_do_cy(),
            'IP_sum': self.snd.get_IP_sum(),
            'IP_cx': self.snd.get_IP_cx(),
            'IP_cy': self.snd.get_IP_cy()
        }
        return output_dict
