# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
from MDAnalysis.analysis.base import AnalysisBase

from WatAnalysis.preprocess import make_selection

# Boltzmann constant [J/K]
B_CONST = 1.38065040000000e-23
# [a.u.] = [Bohr] -> [Angstrom]
AU_2_ANGSTROM = 5.29177208590000e-01
ANGSTROM_2_M = 1e-10
# [Bohr] -> [m]
AU_2_M = AU_2_ANGSTROM * ANGSTROM_2_M
# [a.u.] = hbar / EHartree -> [s]
AU_2_S = 2.41888432650478e-17
# Atomic mass unit [kg]
AMU = 1.66053878200000e-27
PS_2_S = 1e-12


class SelectedTemperature(AnalysisBase):
    """
    TBC

    CP2K velocity: https://manual.cp2k.org/trunk/CP2K_INPUT/MOTION/PRINT/VELOCITIES.html
    """

    def __init__(
        self, ag, u_vels=None, zero_p=False, v_format="xyz", unit="au", verbose=False
    ):
        self.ag = ag
        trajectory = ag.universe.trajectory
        super().__init__(trajectory, verbose=verbose)
        self.n_frames = len(trajectory)
        if u_vels is not None:
            self.vels = u_vels.trajectory
        else:
            self.vels = None
        self.zero_p = zero_p
        self.v_fmt = v_format
        self.unit = unit

    def _prepare(self):
        self.temperature = np.zeros((self.n_frames), dtype=np.float64)

    def _single_frame(self):
        # u.trajectory[ii].positions
        if self.vels is None:
            ts_vels = self._ts.velocities
        else:
            ts_vels = self.vels[self._frame_index].positions
        ts_vels2 = np.sum(ts_vels * ts_vels, axis=-1)
        ts_vels2 = ts_vels2[self.ag.indices]
        ts_masses = self.ag.masses
        if self.zero_p:
            ts_dgf = 3 * len(self.ag) - 3
        else:
            ts_dgf = 3 * len(self.ag)
        self.temperature[self._frame_index] = np.sum(ts_vels2 * ts_masses) / ts_dgf

    def _conclude(self):
        if self.unit == "au":
            prefactor = AMU * (AU_2_M / AU_2_S) ** 2 / B_CONST
            # print("au prefactor: ", prefactor)
        elif self.unit == "metal":
            prefactor = AMU * (ANGSTROM_2_M / PS_2_S) ** 2 / B_CONST
        else:
            raise AttributeError("Unsupported unit %s" % self.unit)
        self.temperature *= prefactor
        return self.temperature


class InterfaceTemperature(SelectedTemperature):
    def __init__(
        self,
        universe,
        u_vels=None,
        zero_p=False,
        v_format="xyz",
        unit="au",
        verbose=False,
        **kwargs,
    ):
        select = make_selection(**kwargs)
        # print("selection: ", select)
        super().__init__(
            universe.select_atoms(select, updating=True),
            u_vels,
            zero_p,
            v_format,
            unit,
            verbose,
        )
