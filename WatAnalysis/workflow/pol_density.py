# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
from MDAnalysis.exceptions import NoDataError
from scipy import integrate

from WatAnalysis import utils
from WatAnalysis.workflow.base import (
    DataRequirement,
    PlanarInterfaceAnalysisBase,
    SingleAnalysis,
)


class PolarisationDensityAnalysis(SingleAnalysis):
    """
    General class to calculate 1D polarisation density profile.

    Parameters
    ----------
    selection : str
        Atom selection string used by MDAnalysis.core.universe.Universe.select_atoms(selection)
    label : str
        Label to identify the intermediate results in the analysis object
    d_bin : float
        Bin width for histogram calculations in Angstroms.
        (Default: 0.1)
    """

    def __init__(
        self,
        selection: str,
        label: str,
        d_bin: float = 0.1,
    ) -> None:
        super().__init__()
        self.selection = selection
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

        if not hasattr(self.ag, "charges"):
            raise NoDataError("No charges defined given atomgroup.")

        if not np.allclose(self.ag.total_charge(), 0.0, atol=1e-5):
            raise Warning("Total charge of the atomgroup is not zero.")

    def _single_frame(self, analyser: PlanarInterfaceAnalysisBase):
        update_flag = analyser.data_requirements[
            f"update_coord_1d_{self.label}"
        ].update_flag
        if not update_flag:
            # copy the coordinates to the intermediate array
            np.copyto(
                getattr(analyser, f"coord_1d_{self.label}")[analyser._frame_index],
                self.ag.positions[:, analyser.axis, np.newaxis],
            )
            # set the flag to True
            analyser.data_requirements[f"update_coord_1d_{self.label}"].set_update_flag(
                True
            )

    def _conclude(self, analyser: PlanarInterfaceAnalysisBase):
        box_length = analyser.universe.dimensions[analyser.axis]
        # within one cell length positive of z1.
        r_unwrapped = getattr(analyser, f"coord_1d_{self.label}").squeeze()
        self.r_wrapped = utils.mic_1d(
            r_unwrapped - analyser.r_surf_ref[:, np.newaxis, np.newaxis],
            box_length=box_length,
            ref=box_length / 2,
        )

        bin_edges = np.arange(0.0, analyser.r_surf_hi.mean() + self.d_bin, self.d_bin)
        bins = utils.bin_edges_to_grid(bin_edges)
        bin_volumes = np.diff(bin_edges) * analyser.cross_area

        # charge density [e/A^3]
        q_density, bin_edges = np.histogram(
            self.r_wrapped.flatten(),
            bins=bin_edges,
            weights=np.repeat(self.ag.charges, analyser.n_frames),
        )
        q_density /= bin_volumes * analyser.n_frames

        pol_density = -integrate.cumulative_trapezoid(q_density, bins, initial=0)
        density_sym = (pol_density[::-1] + pol_density) / 2

        self.results.bins = bins
        self.results.density = pol_density
        self.results.density_sym = density_sym
