import os
from epics import caget


class EPICSInterface:
    """Interface for interacting with EPICS Process Variables (PVs)."""

    def __init__(self):
        """Check environment variables."""
        if "EPICS_CA_ADDR_LIST" not in os.environ:
            raise EnvironmentError(
                "EPICS_CA_ADDR_LIST environment variable is not set."
            )
        if "EPICS_CA_AUTO_ADDR_LIST" not in os.environ:
            raise EnvironmentError(
                "EPICS_CA_AUTO_ADDR_LIST environment variable is not set."
            )

    def get_value(self, pv_name):
        """
        Get the value of a Process Variable (PV) from EPICS.

        Parameters
        ----------
        pv_name : str
            The name of the PV to retrieve.

        Returns
        -------
        value
            The value of the PV.
        """
        return caget(pv_name)

    def get_input_variables(self, input_pvs: list) -> dict:
        """
        Retrieves the input variables from EPICS.

        Parameters
        ----------
        input_pvs : list of str
            A list of input variable names to retrieve.

        Returns
        -------
        dict
            A dictionary with PV names as keys and their values as values.
        """
        return {pv: self.get_value(pv) for pv in input_pvs}
