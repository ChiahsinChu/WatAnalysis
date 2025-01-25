# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import List, Union

import numpy as np
from ase.cell import Cell
from MDAnalysis.analysis.base import AnalysisBase

# from MDAnalysis.analysis.waterdynamics import AngularDistribution
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.core.universe import Universe
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.lib.distances import capped_distance, minimize_vectors

from . import utils
from .preprocess import make_selection, make_selection_two


class WaterStructure(AnalysisBase):
    """Analysis class for studying the structure and properties of water molecules
    in a given molecular dynamics simulation.
    Parameters
    ----------
    universe : Universe
        The MDAnalysis Universe object containing the simulation data.
    surf_ids : Union[List, np.ndarray], optional
        List or array of surface atom indices.
    **kwargs : dict
        Additional keyword arguments for customization:
        - verbose : bool, optional
            If True, enables verbose output.
        - axis : int, optional
            The axis along which the analysis is performed (default is 2).
        - oxygen_sel : str, optional
            Selection string for oxygen atoms (default is "name O").
        - hydrogen_sel : str, optional
            Selection string for hydrogen atoms (default is "name H").
        - min_vector : bool, optional
            If True, uses minimum image convention for vectors (default is True).
        - dz : float, optional
            Bin width for histogram calculations (default is 0.1).
        - oh_cutoff : float, optional
            Cutoff distance for identifying water molecules (default is 1.3).
    Methods
    -------
    run()
        Run the analysis and computes the results.
    calc_sel_water(interval, n_bins=90)
        Calculates properties of water in a selected region relative to the surfaces."""

    def __init__(
        self, universe: Universe, surf_ids: Union[List, np.ndarray] = None, **kwargs
    ):
        self.universe = universe
        trajectory = self.universe.trajectory
        super().__init__(trajectory, verbose=kwargs.get("verbose", False))
        self.n_frames = len(trajectory)

        self.axis = kwargs.get("axis", 2)
        self.ave_axis = np.delete(np.arange(3), self.axis)
        self.surf_ids = surf_ids
        self.oxygen_ag = self.universe.select_atoms(kwargs.get("oxygen_sel", "name O"))
        self.hydrogen_ag = self.universe.select_atoms(
            kwargs.get("hydrogen_sel", "name H")
        )
        self.min_vector = kwargs.get("min_vector", True)
        self.dz = kwargs.get("dz", 0.1)

        self.water_dict = utils.identify_water_molecules(
            self.hydrogen_ag.positions,
            self.oxygen_ag.positions,
            self.universe.dimensions,
            oh_cutoff=kwargs.get("oh_cutoff", 1.3),
        )

        self.z_water = None
        self.geo_dipole_water = None
        self.z1 = None
        self.z2 = None
        self.cross_area = None

    def _prepare(self):
        # placeholder for water z
        self.z_water = np.zeros((self.n_frames, self.oxygen_ag.n_atoms))
        self.geo_dipole_water = np.zeros((self.n_frames, self.oxygen_ag.n_atoms))
        self.z1 = np.zeros(self.n_frames)
        self.z2 = np.zeros(self.n_frames)
        self.cross_area = 0.0

    def _single_frame(self):
        # dimension
        ts_box = self._ts.dimensions
        ts_area = Cell.new(ts_box).area(self.axis)
        self.cross_area += ts_area

        coords = self._ts.positions
        coords_oxygen = self.oxygen_ag.positions
        coords_hydrogen = self.hydrogen_ag.positions

        # surface position (absolute)
        surf1_z = coords[self.surf_ids[0], self.axis]
        surf2_z = coords[self.surf_ids[1], self.axis]
        box_length = ts_box[2]
        self.z1[self._frame_index] = utils.mic_1d(
            surf1_z, box_length, ref=surf1_z[0]
        ).mean()
        self.z2[self._frame_index] = utils.mic_1d(
            surf2_z, box_length, ref=surf2_z[0]
        ).mean()

        # water density
        np.copyto(self.z_water[self._frame_index], coords_oxygen[:, self.axis])

        # Dipoles
        dipole = utils.get_dipoles(
            coords_hydrogen,
            coords_oxygen,
            self.water_dict,
            box=ts_box,
            mic=self.min_vector,
        )
        cos_theta = (dipole[:, self.axis]) / np.linalg.norm(dipole, axis=-1)
        np.copyto(self.geo_dipole_water[self._frame_index], cos_theta)

    def _conclude(self):
        # ave surface area
        self.cross_area /= self.n_frames

        # Refer everything to the left surface (z1), then move everything
        # to be within one cell length positive of z1
        box_length = self.universe.dimensions[2]
        z1 = np.zeros(self.z1.shape)
        z2 = utils.mic_1d(
            self.z2 - self.z1,
            box_length=box_length,
            ref=box_length / 2,
        )
        z_water = utils.mic_1d(
            self.z_water - self.z1[:, np.newaxis],
            box_length=box_length,
            ref=box_length / 2,
        )

        # Update attributes
        self.z1 = z1
        self.z2 = z2
        self.z_water = z_water

        self.results.rho_water = self.density_profile()
        self.results.geo_dipole_water = self.orientation_profile()

    def density_profile(self, only_valid_dipoles=False):
        """
        Calculate density profile using histogram
        """
        z1_mean = np.mean(self.z1)
        z2_mean = np.mean(self.z2)

        # Check valid water molecules (O with 2 H)
        # In this way, the density corresponds to the
        # orientation profile
        if only_valid_dipoles:
            valid = ~np.isnan(self.geo_dipole_water.flatten())
        else:
            valid = np.ones(self.z_water.flatten().size, dtype=bool)

        # Make histogram
        counts, bin_edges = np.histogram(
            self.z_water.flatten()[valid],
            bins=int((z2_mean - z1_mean) / self.dz),
            range=(z1_mean, z2_mean),
        )

        # Spatial coordinates
        z = utils.bin_edges_to_grid(bin_edges)

        # Density values
        n_water = counts / self.n_frames
        grid_volume = np.diff(bin_edges) * self.cross_area
        rho = utils.water_density(n_water, grid_volume)
        return z, rho

    def orientation_profile(self):
        """
        Calculate orientation profile using histogram
        """
        z1_mean = np.mean(self.z1)
        z2_mean = np.mean(self.z2)

        # Check valid water molecules (O with 2 H)
        valid = ~np.isnan(self.geo_dipole_water.flatten())

        counts, bin_edges = np.histogram(
            self.z_water.flatten()[valid],
            bins=int((z2_mean - z1_mean) / self.dz),
            range=(z1_mean, z2_mean),
            weights=self.geo_dipole_water.flatten()[valid],
        )

        z = utils.bin_edges_to_grid(bin_edges)
        y = counts / self.n_frames
        return z, y

    def costheta_profile(self):
        """
        Dipole angle profile
        """
        z, rho = self.density_profile(only_valid_dipoles=True)
        _, rho_cos_theta = self.orientation_profile()
        return z, rho_cos_theta / rho

    def calc_sel_water(self, interval, n_bins=90):
        """
        Calculate properties of water in a selected region relative to the surfaces
        """
        valid = ~np.isnan(self.geo_dipole_water)
        mask = (
            (self.z_water > (self.z1[:, np.newaxis] + interval[0]))
            & (self.z_water <= (self.z1[:, np.newaxis] + interval[1]))
            & valid
        )

        n_water = np.count_nonzero(mask, axis=1)
        theta_lo = np.arccos(self.geo_dipole_water[mask].flatten()) / np.pi * 180

        mask = (
            (self.z_water < (self.z2[:, np.newaxis] - interval[0]))
            & (self.z_water >= (self.z2[:, np.newaxis] - interval[1]))
            & valid
        )
        theta_hi = np.arccos(-self.geo_dipole_water[mask].flatten()) / np.pi * 180
        n_water += np.count_nonzero(mask, axis=1)

        theta = np.concatenate([theta_lo, theta_hi])
        out = np.histogram(
            theta,
            bins=n_bins,
            range=(0.0, 180.0),
            density=True,
        )
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
