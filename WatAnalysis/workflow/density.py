# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np

from WatAnalysis import utils, waterstructure
from WatAnalysis.workflow.base import (
    DataRequirement,
    PlanarInterfaceAnalysisBase,
    SingleAnalysis,
)


class DensityAnalysis(SingleAnalysis):
    """
    General class to calculate 1D particle density profile.

    Parameters
    ----------
    selection : str
        Atom selection string used by MDAnalysis.core.universe.Universe.select_atoms(selection)
    mol_mass : float
        Molecular mass of the atoms in g/mol
    label : str
        Label to identify the intermediate results in the analysis object
    d_bin : float
        Bin width for histogram calculations in Angstroms.
        (Default: 0.1)
    """

    def __init__(
        self,
        selection: str,
        mol_mass: float,
        label: str,
        d_bin: float = 0.1,
    ) -> None:
        super().__init__()
        self.selection = selection
        self.mol_mass = mol_mass
        self.label = label

        assert d_bin > 0, "Bin width must be greater than 0."
        self.d_bin = d_bin

        self.data_requirements = {
            f"coord_1d_{self.label}": DataRequirement(
                f"coord_1d_{self.label}",
                atomic=True,
                dim=1,
                selection=self.selection,
            ),
        }

        self.ag = None
        self.r_wrapped = None

    def _prepare(self, analyser: PlanarInterfaceAnalysisBase):
        self.ag = analyser.universe.select_atoms(self.selection)

    def _single_frame(self, analyser: PlanarInterfaceAnalysisBase):
        update_flag = analyser.data_requirements[f"coord_1d_{self.label}"].update_flag
        if not update_flag:
            # copy the coordinates to the intermediate array
            np.copyto(
                getattr(analyser, f"coord_1d_{self.label}")[analyser._frame_index],
                self.ag.positions[:, analyser.axis, np.newaxis],
            )
            # set the flag to True
            analyser.data_requirements[f"coord_1d_{self.label}"].set_update_flag(True)

    def _conclude(self, analyser: PlanarInterfaceAnalysisBase):
        box_length = analyser.universe.dimensions[analyser.axis]
        # within one cell length positive of z1.
        r_unwrapped = getattr(analyser, f"coord_1d_{self.label}")
        self.r_wrapped = utils.mic_1d(
            r_unwrapped - analyser.r_surf_ref[:, np.newaxis, np.newaxis],
            box_length=box_length,
            ref=box_length / 2,
        )
        bins, density = waterstructure.calc_density_profile(
            (analyser.r_surf_lo.mean(), analyser.r_surf_hi.mean()),
            self.r_wrapped.flatten(),
            cross_area=analyser.cross_area,
            n_frames=analyser.n_frames,
            dz=self.d_bin,
            sym=False,
            mol_mass=self.mol_mass,
        )
        density_sym = (density[::-1] + density) / 2

        self.results.bins = bins
        self.results.density = density
        self.results.density_sym = density_sym
