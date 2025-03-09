# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import List, Union

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


class SteinhardtOrderParameter(SingleAnalysis):
    """
    Calculate atomic Steinhardt order parameter.
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
    cutoff : float
        Maximum distance for neighbor search in Angstroms.
        (Default: 3.5)
    l : int or List[int]
        Spherical harmonics order for the Steinhardt order parameter.
        (Default: 6)
    kwargs : dict of keyword arguments
        Additional keyword arguments for freud.order.Steinhardt.
    """

    def __init__(
        self,
        selection: str,
        label: str,
        d_bin: float = 0.1,
        cutoff: float = 3.5,
        l: Union[int, List[int]] = 6,
        **kwargs,
    ) -> None:
        super().__init__()
        self.selection = selection
        self.label = label
        self.cutoff = cutoff
        if isinstance(l, int):
            l = [l]
        self.n_l = len(l)
        
        assert d_bin > 0, "Bin width must be greater than 0."
        self.d_bin = d_bin

        self.calculator = freud.order.Steinhardt(l=l, **kwargs)

        self.data_requirements = {
            f"Steinhardt_{self.label}": DataRequirement(
                f"Steinhardt_{self.label}",
                atomic=True,
                dim=len(l),
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
        update_flag = analyser.data_requirements[f"Steinhardt_{self.label}"].update_flag
        if not update_flag:
            # ts_box = analyser._ts.dimensions
            # box = freud.box.Box.from_matrix(geometry.cellpar_to_cell(ts_box))
            # box = freud.Box.from_box_lengths_and_angles(*ts_box[:3], *np.deg2rad(ts_box[3:]), dimensions=3)
            # Compute neighbor list
            nlist = (
                freud.AABBQuery.from_system(analyser._ts, 3)
                .query(self.ag.positions, {"r_max": self.cutoff})
                .toNeighborList()
            )
            # Compute Steinhardt order parameter
            self.calculator.compute(system=analyser._ts, neighbors=nlist)
            ql = self.calculator.ql[self.ag.indices].reshape([self.ag.n_atoms, -1])
            # copy Steinhardt order parameter to the intermediate array
            np.copyto(
                getattr(analyser, f"Steinhardt_{self.label}")[analyser._frame_index],
                ql,
            )
            # set the flag to True
            analyser.data_requirements[f"Steinhardt_{self.label}"].set_update_flag(True)

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

        # nf x n_atoms x n_order_params
        order_params = getattr(analyser, f"Steinhardt_{self.label}")
        # calculate hist for every order_param
        all_bin_means = []
        for ii in range(self.n_l):
            order_param = order_params[:, :, ii]
            mask = ~np.isnan(order_param[:, :, np.newaxis])
            bin_means, bin_edges, _binnumber = stats.binned_statistic(
                self.r_wrapped[mask].flatten(), order_param[mask].flatten(), bins=bin_edges
            )
            all_bin_means.append(bin_means)
            
        self.results.bins = bins
        self.results.order_params = np.array(all_bin_means)
        self.results.l = self.calculator.l
        
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
            ts_lsi = calc_atomic_lsi(
                positions=self.ag.positions,
                box=analyser._ts.dimensions,
                cutoff=self.cutoff,
            )
            # copy LSI to the intermediate array
            np.copyto(
                getattr(analyser, f"lsi_{self.label}")[analyser._frame_index],
                ts_lsi[:, np.newaxis],
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
        mask = lsi > 0
        bin_means, bin_edges, _binnumber = stats.binned_statistic(
            self.r_wrapped[mask], lsi[mask], bins=bin_edges
        )

        self.results.bins = bins
        self.results.lsi = bin_means


def calc_atomic_lsi(positions, box, cutoff=3.7):
    pairs, distances = capped_distance(
        positions,
        positions,
        max_cutoff=cutoff + 1.0,
        min_cutoff=0.1,
        box=box,
        return_distances=True,
    )
    n_atoms = positions.shape[0]
    lsi = np.zeros((n_atoms))
    for ii in range(n_atoms):
        mask = pairs[:, 0] == ii
        ds_i = distances[mask]

        n_list = np.count_nonzero(ds_i <= cutoff) + 1
        # Make sure there are neighbors beyond the cutoff
        # Skip if not enough neighbors
        if n_list > len(ds_i) or n_list < 3:
            continue

        sorted_indices = np.argsort(ds_i)
        sorted_distances = ds_i[sorted_indices][:n_list]
        delta_d = np.diff(sorted_distances)
        # Compute LSI
        lsi[ii] = np.mean((delta_d - np.mean(delta_d)) ** 2)
    return lsi
