import numpy as np
from lume_model.base import LUMEBaseModel
from lume_model.variables import ScalarVariable
from lcls_beamline_toolbox.models.split_and_delay_motion import SND

inputs = np.zeros(22)

class SNDModel(LUMEBaseModel):
    def _evaluate(self, input_dict):
        snd = SND(input_dict["energy"], delay=input_dict["delay"])
        del input_dict["energy"]
        del input_dict["delay"]
        for i, motor in enumerate(snd.motor_list):
            motor.mv(motor.wm() + list(input_dict.values())[i])
        snd.propagate_delay()
        snd.propagate_cc()
        output_dict = {
            't1_dh_sum': snd.get_t1_dh_sum(),
            'dd_sum': snd.get_dd_sum(),
            't4_dh_sum': snd.get_t4_dh_sum(),
            'do_sum': snd.get_do_sum(),
            'dd_cx': snd.get_dd_cx(),
            'dd_cy': snd.get_dd_cy(),
            'do_cx': snd.get_do_cx(),
            'do_cy': snd.get_do_cy(),
            'IP_sum': snd.get_IP_sum(),
            'IP_cx': snd.get_IP_cx(),
            'IP_cy': snd.get_IP_cy()
        }
        return output_dict
