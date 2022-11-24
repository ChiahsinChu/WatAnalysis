from ase import io
import numpy as np
from MDAnalysis.analysis.base import AnalysisBase

from WatAnalysis.preprocess import make_selection

# Boltzmann constant [J/K]
B_CONST = 1.38065040000000E-23
# [a.u.] = [Bohr] -> [Angstrom]
AU_2_ANGSTROM = 5.29177208590000E-01
ANGSTROM_2_M = 1E-10
# [Bohr] -> [m]
AU_2_M = AU_2_ANGSTROM * ANGSTROM_2_M
# [a.u.] = hbar / EHartree -> [s]
AU_2_S = 2.41888432650478E-17
# Atomic mass unit [kg]
AMU = 1.66053878200000E-27



class SelectedTemperature(AnalysisBase):
    """
    TBC
    
    CP2K velocity: https://manual.cp2k.org/trunk/CP2K_INPUT/MOTION/PRINT/VELOCITIES.html
    """

    def __init__(self,
                 ag,
                 velocities,
                 zero_p=False,
                 v_format="cp2k",
                 verbose=False):
        self.ag = ag
        trajectory = ag.universe.trajectory
        super().__init__(trajectory, verbose=verbose)
        self.n_frames = len(trajectory)
        self.v2 = np.sum(velocities * velocities, axis=-1)
        self.zero_p = zero_p
        self.v_fmt = v_format

    def _prepare(self):
        self.temperature = np.zeros((self.n_frames), dtype=np.int32)

    def _single_frame(self):
        ts_sel_ids = self.ag.indices
        ts_vels = self.v2[self._frame_index][ts_sel_ids]
        ts_masses = self.ag.masses
        if self.zero_p:
            ts_dgf = 3 * len(self.ag) - 3
        else:
            ts_dgf = 3 * len(self.ag)
        self.temperature = np.sum(ts_vels * ts_masses) / ts_dgf

    def _conclude(self):
        if self.v_fmt == "cp2k":
            self.temperature = self.temperature * AMU * (AU_2_M / AU_2_S)**2 / B_CONST
        return self.temperature


class InterfaceTemperature(SelectedTemperature):

    def __init__(self, universe, velocities, verbose=False, **kwargs):
        select = make_selection(**kwargs)
        # print("selection: ", select)
        super().__init__(universe.select_atoms(select, updating=True),
                         velocities, verbose)
