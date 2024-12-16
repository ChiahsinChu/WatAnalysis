# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import List, Union

import numpy as np
from ase.geometry import cellpar_to_cell
from MDAnalysis.analysis.base import AnalysisBase

# from MDAnalysis.analysis.waterdynamics import AngularDistribution
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.core.universe import Universe
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.lib.distances import capped_distance, minimize_vectors
from toolbox.utils.utils import calc_water_density

from WatAnalysis.preprocess import make_selection, make_selection_two


class WaterStructure(AnalysisBase):
    def __init__(
        self,
        universe: Universe,
        axis: int = 2,
        verbose: bool = False,
        surf_ids: Union[List, np.ndarray] = None,
        oxygen_sel: str = "name O",
        hydrogen_sel: str = "name H",
        min_vector: bool = True,
        symm: bool = True,
    ):
        self.universe = universe
        trajectory = self.universe.trajectory
        super().__init__(trajectory, verbose=verbose)
        self.n_frames = len(trajectory)

        self.axis = axis
        self.ave_axis = np.delete(np.arange(3), self.axis)
        self.surf_ids = surf_ids
        self.oxygen_ag = self.universe.select_atoms(oxygen_sel)
        self.hydrogen_ag = self.universe.select_atoms(hydrogen_sel)
        self.min_vector = min_vector
        self.symm = symm

    def _prepare(self):
        # placeholder for water z
        self.z_water = np.zeros((self.n_frames, self.oxygen_ag.n_atoms))
        self.geo_dipole_water = np.zeros((self.n_frames, self.oxygen_ag.n_atoms))
        self.z_hi = 0.0
        self.z_lo = 0.0
        self.cross_area = 0.0

    def _single_frame(self):
        # dimension
        ts_cellpar = self._ts.dimensions
        ts_cell = cellpar_to_cell(ts_cellpar)
        ts_area = np.linalg.norm(
            np.cross(ts_cell[self.ave_axis[0]], ts_cell[self.ave_axis[1]])
        )
        self.cross_area += ts_area

        coords = self._ts.positions
        coords_oxygen = self.oxygen_ag.positions
        coords_hydrogen = self.hydrogen_ag.positions.reshape(-1, 2, 3)

        # surface position (refs)
        z1 = np.mean(coords[self.surf_ids[0], self.axis])
        z2 = np.mean(coords[self.surf_ids[1], self.axis])
        self.z_lo += np.min([z1, z2])
        self.z_hi += np.max([z1, z2])
        # print(z1, z2)

        # water density
        np.copyto(self.z_water[self._frame_index], coords_oxygen[:, self.axis])
        # cos theta
        if self.min_vector:
            bond_vec_1 = minimize_vectors(
                coords_hydrogen[:, 0, :] - coords_oxygen, box=ts_cellpar
            )
            bond_vec_2 = minimize_vectors(
                coords_hydrogen[:, 1, :] - coords_oxygen, box=ts_cellpar
            )
            bond_vectors = bond_vec_1 + bond_vec_2
        else:
            bond_vectors = coords_hydrogen.mean(axis=1) - coords_oxygen
        cos_theta = (bond_vectors[:, self.axis]) / np.linalg.norm(bond_vectors, axis=-1)
        np.copyto(self.geo_dipole_water[self._frame_index], cos_theta)

    def _conclude(self):
        # ave surface position
        self.z_lo /= self.n_frames
        self.z_hi /= self.n_frames
        self.cross_area /= self.n_frames
        self.z_water_flatten = self.z_water.flatten()
        self.geo_dipole_water_flatten = self.geo_dipole_water.flatten()

        # water density
        out = np.histogram(
            self.z_water_flatten,
            bins=int((self.z_hi - self.z_lo) / 0.1),
            range=(self.z_lo, self.z_hi),
        )
        n_water = out[0] / self.n_frames
        grid_volume = np.diff(out[1]) * self.cross_area
        grid = out[1][:-1] + np.diff(out[1]) / 2
        rho = calc_water_density(n_water, grid_volume)
        x = grid - self.z_lo
        if self.symm:
            y = (np.flip(rho) + rho) / 2
        else:
            y = rho
        self.results["rho_water"] = [x, y]

        # water orientation
        out = np.histogram(
            self.z_water_flatten,
            bins=int((self.z_hi - self.z_lo) / 0.1),
            range=(self.z_lo, self.z_hi),
            weights=self.geo_dipole_water_flatten,
        )
        grid = out[1][:-1] + np.diff(out[1]) / 2
        x = grid - self.z_lo
        y = out[0] / self.n_frames / self.cross_area
        if self.symm:
            y = (y - np.flip(y)) / 2
        self.results["geo_dipole_water"] = [x, y]

    def calc_sel_water(self, interval):
        mask = (self.z_water > (self.z_lo + interval[0])) & (
            self.z_water <= (self.z_lo + interval[1])
        )
        n_water = np.count_nonzero(mask, axis=1)
        theta_lo = (
            np.arccos(self.geo_dipole_water_flatten[mask.flatten()]) / np.pi * 180
        )

        mask = (self.z_water < (self.z_hi - interval[0])) & (
            self.z_water >= (self.z_hi - interval[1])
        )
        theta_hi = (
            np.arccos(-self.geo_dipole_water_flatten[mask.flatten()]) / np.pi * 180
        )
        n_water += np.count_nonzero(mask, axis=1)

        theta = np.concatenate([theta_lo, theta_hi])
        out = np.histogram(theta, bins=90, range=(0.0, 180.0), density=True)
        grid = out[1][:-1] + np.diff(out[1]) / 2
        return n_water, [grid, out[0]]


class WatCoverage(AnalysisBase):
    def __init__(self, universe, verbose=False, **kwargs):
        select = make_selection(**kwargs)
        # print("selection: ", select)
        self.universe = universe
        trajectory = universe.trajectory
        super().__init__(trajectory, verbose=verbose)
        self.n_frames = len(trajectory)
        self.ag = universe.select_atoms(select, updating=True)

    def _prepare(self):
        # placeholder for water z
        self.n_water = np.zeros((self.n_frames), dtype=np.int32)

    def _single_frame(self):
        self.n_water[self._frame_index] = len(self.ag)

    def _conclude(self):
        return self.n_water


class AngularDistribution(AnalysisBase):
    def __init__(
        self, universe, nbins=50, axis="z", updating=True, verbose=False, **kwargs
    ):
        trajectory = universe.trajectory
        super().__init__(trajectory, verbose=verbose)

        select = make_selection_two(**kwargs)
        # print("selection: ", select)
        self.universe = universe
        self.updating = updating
        self.ags = self._make_selections(select)
        self.nbins = nbins
        self.axis = axis

    def _prepare(self):
        self.ts_cosOH = []
        self.ts_cosHH = []
        self.ts_cosD = []

    def _single_frame(self):
        # BUG: be careful! for OHH only yet!
        axis_dict = {"x": 0, "y": 1, "z": 2}
        axis = axis_dict[self.axis]

        cosOH, cosHH, cosD = self._getCosTheta(self.ags[0], axis)
        self.ts_cosOH.extend(cosOH.tolist())
        self.ts_cosHH.extend(cosHH.tolist())
        self.ts_cosD.extend(cosD.tolist())
        cosOH, cosHH, cosD = self._getCosTheta(self.ags[1], axis)
        self.ts_cosOH.extend((-cosOH).tolist())
        self.ts_cosHH.extend((-cosHH).tolist())
        self.ts_cosD.extend((-cosD).tolist())

    def _conclude(self):
        self.results = {}

        thetaOH = np.arccos(self.ts_cosOH) / np.pi * 180
        thetaHH = np.arccos(self.ts_cosHH) / np.pi * 180
        thetaD = np.arccos(self.ts_cosD) / np.pi * 180

        cos_hist_interval = np.linspace(-1.0, 1.0, self.nbins)
        theta_hist_interval = np.linspace(0.0, 180.0, self.nbins)

        hist_cosOH = np.histogram(self.ts_cosOH, cos_hist_interval, density=True)
        hist_cosHH = np.histogram(self.ts_cosHH, cos_hist_interval, density=True)
        hist_cosD = np.histogram(self.ts_cosD, cos_hist_interval, density=True)
        hist_OH = np.histogram(thetaOH, theta_hist_interval, density=True)
        hist_HH = np.histogram(thetaHH, theta_hist_interval, density=True)
        hist_D = np.histogram(thetaD, theta_hist_interval, density=True)

        for label in ["cosOH", "cosHH", "cosD", "OH", "HH", "D"]:
            output = locals()["hist_%s" % label]
            self.results[label] = np.transpose(
                np.concatenate(
                    ([output[1][:-1] + (output[1][1] - output[1][0]) / 2], [output[0]])
                )
            )

        # self.results['cosOH'] =
        # self.results['cosHH'] = np.transpose(
        #     np.concatenate(
        #         ([output[1][1][:-1] + (output[1][1][1] - output[1][0][0]) / 2],
        #          [output[1][0]])))
        # self.results['cosD'] = np.transpose(
        #     np.concatenate(
        #         ([output[2][1][:-1] + (output[2][1][1] - output[2][1][0]) / 2],
        #          [output[2][0]])))
        # self.results['OH'] = np.transpose(
        #     np.concatenate(
        #         ([output[3][1][:-1] + (output[3][1][1] - output[3][1][0]) / 2],
        #          [output[3][0]])))
        # self.results['HH'] = np.transpose(
        #     np.concatenate(
        #         ([output[4][1][:-1] + (output[4][1][1] - output[4][1][0]) / 2],
        #          [output[4][0]])))
        # self.results['D'] = np.transpose(
        #     np.concatenate(
        #         ([output[5][1][:-1] + (output[5][1][1] - output[5][1][0]) / 2],
        #          [output[5][0]])))

    def _getCosTheta(self, ag, axis):
        ts_positions = ag.positions
        ts_p_O = ts_positions[::3]
        ts_p_H1 = ts_positions[1::3]
        ts_p_H2 = ts_positions[2::3]

        vec_OH_0 = minimize_vectors(vectors=ts_p_H1 - ts_p_O, box=self._ts.dimensions)
        vec_OH_1 = minimize_vectors(vectors=ts_p_H2 - ts_p_O, box=self._ts.dimensions)
        cosOH = vec_OH_0[:, axis] / np.linalg.norm(vec_OH_0, axis=-1)
        # self.ts_cosOH.extend(cosOH.tolist())
        cosOH = np.append(cosOH, vec_OH_1[:, axis] / np.linalg.norm(vec_OH_1, axis=-1))
        # self.ts_cosOH.extend(cosOH.tolist())

        vec_HH = ts_p_H1 - ts_p_H2
        cosHH = vec_HH[:, axis] / np.linalg.norm(vec_HH, axis=-1)
        # self.ts_cosHH.extend(cosHH.tolist())

        vec_D = vec_OH_0 + vec_OH_1
        cosD = vec_D[:, axis] / np.linalg.norm(vec_D, axis=-1)
        # self.ts_cosD.extend(cosD.tolist())

        return cosOH, cosHH, cosD

    def _make_selections(self, l_selection_str):
        selection = []
        for sel in l_selection_str:
            sel_ag = self.universe.select_atoms(sel, updating=self.updating)
            # TODO: check why it does not work
            sel_ag.unwrap()
            selection.append(sel_ag)
        return selection


# class AD(AngularDistribution):

#     def __init__(self,
#                  universe,
#                  nbins=50,
#                  nproc=1,
#                  axis="z",
#                  updating=True,
#                  **kwargs):
#         select = make_selection_two(**kwargs)
#         # print("selection: ", select)
#         super().__init__(universe, select, nbins, nproc, axis)
#         self.updating = updating

#     def _getHistogram(self, universe, selection, bins, axis):
#         """
#         This function gets a normalized histogram of the cos(theta) values. It
#         return a list of list.
#         """
#         a_lo = self._getCosTheta(universe, selection[::2], axis)
#         a_hi = self._getCosTheta(universe, selection[1::2], axis)
#         # print(np.shape(a_lo))
#         # print(np.shape(a_hi))

#         cosThetaOH = np.concatenate([np.array(a_lo[0]), -np.array(a_hi[0])])
#         cosThetaHH = np.concatenate([np.array(a_lo[1]), -np.array(a_hi[1])])
#         cosThetadip = np.concatenate([np.array(a_lo[2]), -np.array(a_hi[2])])
#         ThetaOH = np.arccos(cosThetaOH) / np.pi * 180
#         ThetaHH = np.arccos(cosThetaHH) / np.pi * 180
#         Thetadip = np.arccos(cosThetadip) / np.pi * 180

#         coshistInterval = np.linspace(-1., 1., bins)
#         anglehistInterval = np.linspace(0., 180., bins)

#         histcosThetaOH = np.histogram(cosThetaOH,
#                                       coshistInterval,
#                                       density=True)
#         histcosThetaHH = np.histogram(cosThetaHH,
#                                       coshistInterval,
#                                       density=True)
#         histcosThetadip = np.histogram(cosThetadip,
#                                        coshistInterval,
#                                        density=True)
#         histThetaOH = np.histogram(ThetaOH, anglehistInterval, density=True)
#         histThetaHH = np.histogram(ThetaHH, anglehistInterval, density=True)
#         histThetadip = np.histogram(Thetadip, anglehistInterval, density=True)

#         return (histcosThetaOH, histcosThetaHH, histcosThetadip, histThetaOH,
#                 histThetaHH, histThetadip)

#     def run(self, **kwargs):
#         """Function to evaluate the angular distribution of cos(theta)"""

#         selection = self._selection_serial(self.universe, self.selection_str)

#         self.graph = {}
#         output = self._getHistogram(self.universe, selection, self.bins,
#                                     self.axis)
#         # this is to format the exit of the file
#         # maybe this output could be improved
#         self.graph['cosOH'] = np.transpose(
#             np.concatenate(
#                 ([output[0][1][:-1] + (output[0][1][1] - output[0][1][0]) / 2],
#                  [output[0][0]])))
#         self.graph['cosHH'] = np.transpose(
#             np.concatenate(
#                 ([output[1][1][:-1] + (output[1][1][1] - output[1][0][0]) / 2],
#                  [output[1][0]])))
#         self.graph['cosD'] = np.transpose(
#             np.concatenate(
#                 ([output[2][1][:-1] + (output[2][1][1] - output[2][1][0]) / 2],
#                  [output[2][0]])))
#         self.graph['OH'] = np.transpose(
#             np.concatenate(
#                 ([output[3][1][:-1] + (output[3][1][1] - output[3][1][0]) / 2],
#                  [output[3][0]])))
#         self.graph['HH'] = np.transpose(
#             np.concatenate(
#                 ([output[4][1][:-1] + (output[4][1][1] - output[4][1][0]) / 2],
#                  [output[4][0]])))
#         self.graph['D'] = np.transpose(
#             np.concatenate(
#                 ([output[5][1][:-1] + (output[5][1][1] - output[5][1][0]) / 2],
#                  [output[5][0]])))

#         # listcosOH = [list(output[0][1]), list(output[0][0])]
#         # listcosHH = [list(output[1][1]), list(output[1][0])]
#         # listcosdip = [list(output[2][1]), list(output[2][0])]
#         # listOH = [list(output[3][1]), list(output[3][0])]
#         # listHH = [list(output[4][1]), list(output[4][0])]
#         # listdip = [list(output[5][1]), list(output[5][0])]

#         # self.graph.append(self._hist2column(listcosOH))
#         # self.graph.append(self._hist2column(listcosHH))
#         # self.graph.append(self._hist2column(listcosdip))
#         # self.graph.append(self._hist2column(listOH))
#         # self.graph.append(self._hist2column(listHH))
#         # self.graph.append(self._hist2column(listdip))

#     def _selection_serial(self, universe, l_selection_str):
#         selection = []
#         for ts in ProgressBar(universe.trajectory,
#                               verbose=True,
#                               total=universe.trajectory.n_frames):
#             tmp_ag = universe.select_atoms(l_selection_str[0],
#                                            updating=self.updating)
#             tmp_ag.unwrap()
#             selection.append(tmp_ag)
#             tmp_ag = universe.select_atoms(l_selection_str[1],
#                                            updating=self.updating)
#             tmp_ag.unwrap()
#             selection.append(tmp_ag)
#         return selection


class HBA(HydrogenBondAnalysis):
    def __init__(
        self,
        universe,
        donors_sel=None,
        hydrogens_sel=None,
        acceptors_sel=None,
        between=None,
        d_h_cutoff=1.2,
        d_a_cutoff=3,
        d_h_a_angle_cutoff=150,
        update_acceptors=False,
        update_donors=False,
    ):
        self.update_acceptors = update_acceptors
        self.update_donors = update_donors
        update_selection = update_donors | update_acceptors
        super().__init__(
            universe,
            donors_sel,
            hydrogens_sel,
            acceptors_sel,
            between,
            d_h_cutoff,
            d_a_cutoff,
            d_h_a_angle_cutoff,
            update_selection,
        )

    def _prepare(self):
        self.results.hbonds = [[], [], [], [], [], []]

        # Set atom selections if they have not been provided
        if not self.acceptors_sel:
            self.acceptors_sel = self.guess_acceptors()
        if not self.hydrogens_sel:
            self.hydrogens_sel = self.guess_hydrogens()

        # Select atom groups
        self._acceptors = self.u.select_atoms(
            self.acceptors_sel, updating=self.update_acceptors
        )
        self._donors, self._hydrogens = self._get_dh_pairs()

    def _get_dh_pairs(self):
        """Finds donor-hydrogen pairs.

        Returns
        -------
        donors, hydrogens: AtomGroup, AtomGroup
            AtomGroups corresponding to all donors and all hydrogens. AtomGroups are ordered such that, if zipped, will
            produce a list of donor-hydrogen pairs.
        """

        # If donors_sel is not provided, use topology to find d-h pairs
        if not self.donors_sel:
            # We're using u._topology.bonds rather than u.bonds as it is a million times faster to access.
            # This is because u.bonds also calculates properties of each bond (e.g bond length).
            # See https://github.com/MDAnalysis/mdanalysis/issues/2396#issuecomment-596251787
            if not (
                hasattr(self.u._topology, "bonds")
                and len(self.u._topology.bonds.values) != 0
            ):
                raise NoDataError(
                    "Cannot assign donor-hydrogen pairs via topology as no bond information is present. "
                    "Please either: load a topology file with bond information; use the guess_bonds() "
                    "topology guesser; or set HydrogenBondAnalysis.donors_sel so that a distance cutoff "
                    "can be used."
                )

            hydrogens = self.u.select_atoms(self.hydrogens_sel)
            donors = (
                sum(h.bonded_atoms[0] for h in hydrogens)
                if hydrogens
                else AtomGroup([], self.u)
            )

        # Otherwise, use d_h_cutoff as a cutoff distance
        else:
            hydrogens = self.u.select_atoms(self.hydrogens_sel)
            donors = self.u.select_atoms(self.donors_sel, updating=self.update_donors)
            donors_indices, hydrogen_indices = capped_distance(
                donors.positions,
                hydrogens.positions,
                max_cutoff=self.d_h_cutoff,
                box=self.u.dimensions,
                return_distances=False,
            ).T

            donors = donors[donors_indices]
            hydrogens = hydrogens[hydrogen_indices]

        return donors, hydrogens
