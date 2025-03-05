# SPDX-License-Identifier: LGPL-3.0-or-later
import freud
import numpy as np
from MDAnalysis.analysis.distances import capped_distance
from scipy import stats

from WatAnalysis import utils
from WatAnalysis.workflow.base import (
    DataRequirement,
    PlanarInterfaceAnalysisBase,
    SingleAnalysis,
)


class Q6OrderParameter(SingleAnalysis):
    """
    Calculate atomic Q6 order parameter.
    Ref: Steinhardt, P. J., Nelson, D. R., & Ronchetti, M. Physical Review B, 1983, 28(2), 784-805.

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
        cutoff: float = 3.5,
    ) -> None:
        super().__init__()
        self.selection = selection
        self.label = label
        self.cutoff = cutoff

        assert d_bin > 0, "Bin width must be greater than 0."
        self.d_bin = d_bin

        self.q6_calculator = freud.order.Steinhardt(l=6)

        self.data_requirements = {
            f"c6_{self.label}": DataRequirement(
                f"c6_{self.label}",
                atomic=True,
                dim=1,
                selection=self.selection,
            ),
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
        update_flag = analyser.data_requirements[f"c6_{self.label}"].update_flag
        if not update_flag:
            # Compute neighbor list
            nlist = (
                freud.AABBQuery.from_system(analyser._ts, 3)
                .query(self.ag.positions, {"r_max": self.cutoff})
                .toNeighborList()
            )
            # Compute Q6 order parameter
            self.q6_calculator.compute(system=analyser._ts, neighbors=nlist)
            # copy Q6 to the intermediate array
            np.copyto(
                getattr(analyser, f"c6_{self.label}")[analyser._frame_index],
                self.q6_calculator.particle_order[:, np.newaxis],
            )
            # set the flag to True
            analyser.data_requirements[f"c6_{self.label}"].set_update_flag(True)

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

        bin_edges = np.arange(0.0, analyser.r_surf_hi.mean() + self.d_bin, self.d_bin)
        bins = utils.bin_edges_to_grid(bin_edges)

        c6 = getattr(analyser, f"c6_{self.label}")
        bin_means, bin_edges, _binnumber = stats.binned_statistic(
            self.r_wrapped.flatten(), c6.flatten(), bins=bin_edges
        )

        self.results.bins = bins
        self.results.c6 = bin_means


class LocalStructureIndex(SingleAnalysis):
    """
    Calculate local structure index.
    Ref: Shiratani, E., & Sasai, M. The Journal of Chemical Physics, 1998, 108(8), 3264-3276.

    Parameters
    ----------
    selection : str
        Atom selection string used by MDAnalysis.core.universe.Universe.select_atoms(selection)
    label : str
        Label to identify the intermediate results in the analysis object
    d_bin : float
        Bin width for histogram calculations in Angstroms.
        (Default: 0.1)
    cutoff : float
        Maximum distance for neighbor search in Angstroms.
        (Default: 3.7)
    """

    def __init__(
        self,
        selection: str,
        label: str,
        d_bin: float = 0.1,
        cutoff: float = 3.7,
    ) -> None:
        super().__init__()
        self.selection = selection
        self.label = label
        self.cutoff = cutoff

        assert d_bin > 0, "Bin width must be greater than 0."
        self.d_bin = d_bin

        self.data_requirements = {
            f"lsi_{self.label}": DataRequirement(
                f"lsi_{self.label}",
                atomic=True,
                dim=1,
                selection=self.selection,
            ),
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
        update_flag = analyser.data_requirements[f"lsi_{self.label}"].update_flag
        if not update_flag:
            pairs, distances = capped_distance(
                self.ag,
                self.ag,
                max_cutoff=self.cutoff,
                min_cutoff=0.1,
                box=analyser._ts.dimensions,
            )
            for ii in range(self.ag):
                mask = pairs[:, 0] == ii
                ds_i = distances[mask]
                # Skip if not enough neighbors
                if len(ds_i) < 2:
                    continue
                # Compute LSI
                delta_d = ds_i - np.mean(ds_i)
                getattr(analyser, f"lsi_{self.label}")[analyser._frame_index, ii, 0] = (
                    np.sum(np.exp(-((delta_d / 0.1) ** 2)))
                )
            # set the flag to True
            analyser.data_requirements[f"lsi_{self.label}"].set_update_flag(True)

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

        bin_edges = np.arange(0.0, analyser.r_surf_hi.mean() + self.d_bin, self.d_bin)
        bins = utils.bin_edges_to_grid(bin_edges)

        lsi = getattr(analyser, f"lsi_{self.label}")
        bin_means, bin_edges, _binnumber = stats.binned_statistic(
            self.r_wrapped.flatten(), lsi.flatten(), bins=bin_edges
        )

        self.results.bins = bins
        self.results.lsi = bin_means
