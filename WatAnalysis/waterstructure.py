# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import Dict, List, Tuple, Union

import numpy as np
from ase.cell import Cell
from ase.geometry import cellpar_to_cell
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.core.universe import Universe
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.lib.distances import capped_distance, minimize_vectors, distance_array

from . import utils
from .preprocess import make_selection, make_selection_two


def calc_water_dipoles(
    h_positions: np.ndarray,
    o_positions: np.ndarray,
    water_dict: Dict[int, List[int]],
    box: np.ndarray,
    mic: bool = True,
) -> np.ndarray:
    """
    Calculate dipole moments for water molecules.

    Parameters
    ----------
    h_positions : np.ndarray
        Positions of hydrogen atoms.
    o_positions : np.ndarray
        Positions of oxygen atoms.
    water_dict : Dict[int, List[int]]
        Dictionary mapping oxygen atom indices to two bonded hydrogen atom indices.
    box : np.ndarray
        Simulation cell defining periodic boundaries.

    Returns
    -------
    np.ndarray
        Array of dipole vectors for each oxygen atom. Entries are NaN for non-water oxygen atoms.
    """
    o_indices = np.array([k for k in water_dict.keys()])
    h1_indices = np.array([v[0] for v in water_dict.values()])
    h2_indices = np.array([v[1] for v in water_dict.values()])

    oh1_vectors = h_positions[h1_indices] - o_positions
    oh2_vectors = h_positions[h2_indices] - o_positions

    if mic:
        oh1_vectors = minimize_vectors(oh1_vectors, box)
        oh2_vectors = minimize_vectors(oh2_vectors, box)

    dipoles = np.ones(o_positions.shape) * np.nan
    dipoles[o_indices, :] = oh1_vectors + oh2_vectors
    return dipoles


class WaterStructure(AnalysisBase):
    """
    Analysis class for studying the structure and properties of water molecules
    in a given molecular dynamics simulation.

    Parameters
    ----------
    universe : Universe
        The MDAnalysis Universe object containing the simulation data.
    surf_ids : Union[List, np.ndarray], optional
        List or array of surface atom indices of the form [surf_1, surf_2]
        where surf_1 and surf_2 are arrays containing the indices corresponding
        to the left surface and right surface, respectively.
    **kwargs : dict
        Additional keyword arguments for customization:
        - verbose : bool, optional.
            If True, enables verbose output.
        - axis : int, optional.
            The axis along which the analysis is performed (default is 2).
            x=0, y=1, z=2.
        - oxygen_sel : str, optional.
            Selection string for oxygen atoms (default is "name O").
        - hydrogen_sel : str, optional.
            Selection string for hydrogen atoms (default is "name H").
        - min_vector : bool, optional.
            If True, uses minimum image convention for vectors (default is True).
            Can be disabled for unwrapped trajectories to compute faster.
        - dz : float, optional.
            Bin width for histogram calculations in Angstroms. Must be positive.
            (Default is 0.1.)
        - oh_cutoff : float, optional.
            Cutoff distance for identifying water molecules in Angstroms (default is 1.3).
        - ignore_warnings : bool, optional.
            If True, ignore warnings about non-water species (default is False).

    Methods
    -------
    run()
        Run the analysis and computes the results.
    density_profile()
        Compute the density profile (rho).
    costheta_profile()
        Compute the mean dipole angle profile (<cos theta>).
    orientation_profile(self):
        Compute the orientation profile (rho * <cos theta>).
    calc_sel_water(interval, n_bins=90)
        Calculates properties of water in a selected region relative to the surfaces.

    Outputs
    -------
    Water structure data are returned in a :class:`Dict` and can be accessed
    via :attr:`WaterStructure.results`::, including:
        - rho_water : (2, n_bins) np.ndarray
            The density profile of water molecules.
        - geo_dipole_water : (2, n_bins) np.ndarray
            The orientation profile of water molecules.
    """

    def __init__(
        self,
        universe: Universe,
        surf_ids: Union[List, np.ndarray] = None,
        **kwargs,
    ):
        self.universe = universe
        trajectory = self.universe.trajectory
        super().__init__(trajectory, verbose=kwargs.get("verbose", False))
        self.n_frames = len(trajectory)

        self.axis = kwargs.get("axis", 2)
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
            ignore_warnings=kwargs.get("ignore_warnings", False),
        )

        self.z_water = None
        self.cos_theta_water = None
        self.z1 = None
        self.z2 = None
        self.cross_area = None

    def _prepare(self):
        # Initialize empty arrays
        self.z_water = np.zeros((self.n_frames, self.oxygen_ag.n_atoms))
        self.cos_theta_water = np.zeros((self.n_frames, self.oxygen_ag.n_atoms))
        self.z1 = np.zeros(self.n_frames)
        self.z2 = np.zeros(self.n_frames)
        self.cross_area = 0.0

    def _single_frame(self):
        ts_box = self._ts.dimensions
        ts_area = Cell.new(ts_box).area(self.axis)
        self.cross_area += ts_area

        coords = self._ts.positions
        coords_oxygen = self.oxygen_ag.positions
        coords_hydrogen = self.hydrogen_ag.positions

        # Absolute surface positions
        surf1_z = coords[self.surf_ids[0], self.axis]
        surf2_z = coords[self.surf_ids[1], self.axis]
        box_length = ts_box[self.axis]
        # Use MIC in case part of the surface crosses the cell boundaries
        self.z1[self._frame_index] = utils.mic_1d(
            surf1_z, box_length, ref=surf1_z[0]
        ).mean()
        self.z2[self._frame_index] = utils.mic_1d(
            surf2_z, box_length, ref=surf2_z[0]
        ).mean()

        # Save oxygen locations for water density analysis
        np.copyto(self.z_water[self._frame_index], coords_oxygen[:, self.axis])

        # Calculate dipoles and project on self.axis
        dipole = calc_water_dipoles(
            coords_hydrogen,
            coords_oxygen,
            self.water_dict,
            box=ts_box,
            mic=self.min_vector,
        )
        cos_theta = (dipole[:, self.axis]) / np.linalg.norm(dipole, axis=-1)
        np.copyto(self.cos_theta_water[self._frame_index], cos_theta)

    def _conclude(self):
        # Average surface area
        self.cross_area /= self.n_frames

        box_length = self.universe.dimensions[self.axis]
        # Step 1: Set z1 as the reference point (zero)
        z1 = np.zeros(self.z1.shape)
        # Step 2: Calculate z2 positions relative to z1 using minimum image convention
        # By setting ref=box_length/2, all positions are within one cell length positive of z1
        z2 = utils.mic_1d(
            self.z2 - self.z1,
            box_length=box_length,
            ref=box_length / 2,
        )
        # Step 3: Calculate z_water relative to z_1, ensuring that all water positions are
        # within one cell length positive of z1.
        z_water = utils.mic_1d(
            self.z_water - self.z1[:, np.newaxis],
            box_length=box_length,
            ref=box_length / 2,
        )

        # Update attributes to the final relative coordinates
        self.z1 = z1
        self.z2 = z2
        self.z_water = z_water

        self.results.rho_water = self.calc_density_profile()
        self.results.geo_dipole_water = self.calc_orientation_profile()

    def calc_density_profile(
        self, only_valid_dipoles: bool = False, sym: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the density profile of water molecules using a histogram.

        Parameters
        ----------
        only_valid_dipoles : bool, optional
            If True, only consider valid water molecules (O with 2 H) for the
            density calculation. Default is False.
        sym : bool, optional
            If True, the density profile is symmetrized about the center of the
            simulation box. Default is False.

        Returns
        -------
        z : ndarray
            The spatial coordinates corresponding to the bin centers.
        rho : ndarray
            The density values of water molecules.
        """
        z1_mean = np.mean(self.z1)
        z2_mean = np.mean(self.z2)

        # Check valid water molecules (O with 2 H)
        # In this way, the density rho corresponds to the density in the
        # orientation profile rho * <cos theta>
        if only_valid_dipoles:
            valid = ~np.isnan(self.cos_theta_water.flatten())
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
        rho = utils.calc_water_density(n_water, grid_volume)
        if sym:
            rho = (rho[::-1] + rho) / 2
        return z, rho

    def calc_orientation_profile(
        self, sym: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the orientation profile of water molecules.
        This method computes the orientation profile of water molecules
        by calculating the mean positions along the z-axis and creating
        a histogram of the dipole orientations.

        Parameters
        ----------
        sym : bool, optional
            If True, the density profile is symmetrized about the center of the
            simulation box. Default is False.

        Returns
        -------
        z : ndarray
            The grid points along the z-axis.
        rho_cos_theta : ndarray
            The orientation profile of water molecules.
        """
        z1_mean = np.mean(self.z1)
        z2_mean = np.mean(self.z2)

        # Check valid water molecules (O with 2 H)
        valid = ~np.isnan(self.cos_theta_water.flatten())

        counts, bin_edges = np.histogram(
            self.z_water.flatten()[valid],
            bins=int((z2_mean - z1_mean) / self.dz),
            range=(z1_mean, z2_mean),
            weights=self.cos_theta_water.flatten()[valid],
        )

        z = utils.bin_edges_to_grid(bin_edges)
        n_water = counts / self.n_frames
        grid_volume = np.diff(bin_edges) * self.cross_area
        rho_cos_theta = utils.calc_water_density(n_water, grid_volume)

        if sym:
            rho_cos_theta = (rho_cos_theta - rho_cos_theta[::-1]) / 2
        return z, rho_cos_theta

    def calc_costheta_profile(self, sym: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the average cosine theta profile.

        Parameters
        ----------
        sym : bool, optional
            If True, the density profile is symmetrized about the center of the
            simulation box. Default is False.

        Returns
        -------
        z : ndarray
            The grid points along the z-axis.
        avg_cos_theta : ndarray
            The cosine theta profile.
        """
        z1_mean = np.mean(self.z1)
        z2_mean = np.mean(self.z2)
        z_coords = self.z_water.flatten()
        cos_theta = self.cos_theta_water.flatten()

        bin_edges = np.linspace(
            z1_mean, z2_mean, int((z2_mean - z1_mean) / self.dz) + 1
        )
        z = utils.bin_edges_to_grid(bin_edges)

        # Digitize z-coordinates to find which bin each value belongs to
        bin_indices = np.digitize(z_coords, bin_edges)

        # Compute average cos(theta) in each bin
        avg_cos_theta = np.array(
            [
                (
                    cos_theta[bin_indices == i].mean()
                    if np.any(bin_indices == i)
                    else np.nan
                )
                for i in range(1, len(bin_edges))
            ]
        )

        if sym:
            avg_cos_theta = (avg_cos_theta - avg_cos_theta[::-1]) / 2

        return z, avg_cos_theta

    def calc_sel_water(
        self, interval: Tuple[float, float], n_bins: int = 90
    ) -> Tuple[float, list[np.ndarray]]:
        """
        Calculate properties of water in a selected region relative to the surfaces.

        Parameters
        ----------
        interval : tuple of float
            The interval range to select the region of interest.
        n_bins : int, optional
            The number of bins for the histogram (default is 90).

        Returns
        -------
        n_water : float
            The number of water molecules in the selected region.
        histogram : list[ndarray]
            A list [x, y] containing the grid of angles (x) and the
            probability density of the angular distribution (y).
        """
        valid = ~np.isnan(self.cos_theta_water)
        mask = (
            (self.z_water > (self.z1[:, np.newaxis] + interval[0]))
            & (self.z_water <= (self.z1[:, np.newaxis] + interval[1]))
            & valid
        )

        n_water = np.count_nonzero(mask, axis=1)
        lower_surface_angles = (
            np.arccos(self.cos_theta_water[mask].flatten()) / np.pi * 180
        )

        mask = (
            (self.z_water < (self.z2[:, np.newaxis] - interval[0]))
            & (self.z_water >= (self.z2[:, np.newaxis] - interval[1]))
            & valid
        )
        upper_surface_angles = (
            np.arccos(-self.cos_theta_water[mask].flatten()) / np.pi * 180
        )
        n_water += np.count_nonzero(mask, axis=1)

        combined_angles = np.concatenate([lower_surface_angles, upper_surface_angles])
        angle_distribution, bin_edges = np.histogram(
            combined_angles,
            bins=n_bins,
            range=(0.0, 180.0),
            density=True,
        )
        grid = utils.bin_edges_to_grid(bin_edges)
        return n_water, [grid, angle_distribution]


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


class DeprecatedWaterStructure(AnalysisBase):
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
        rho = utils.calc_water_density(n_water, grid_volume)
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


class AlkaliStructure(AnalysisBase):
    def __init__(
        self,
        universe: Universe,
        surf_ids: Union[List, np.ndarray] = None,
        **kwargs,
    ):
        self.universe = universe
        trajectory = self.universe.trajectory
        super().__init__(trajectory, verbose=kwargs.get("verbose", False))
        self.n_frames = len(trajectory)

        self.axis = kwargs.get("axis", 2)
        self.surf_ids = surf_ids
        self.oxygen_ag = self.universe.select_atoms(kwargs.get("oxygen_sel", "name O"))
        self.hydrogen_ag = self.universe.select_atoms(
            kwargs.get("hydrogen_sel", "name H")
        )
        self.min_vector = kwargs.get("min_vector", True)
        self.dz = kwargs.get("dz", 0.1)

        self.z_oxygen = None
        self.cos_theta_OH = None
        self.cns = None

        self.z1 = None
        self.z2 = None
        self.cross_area = None

    def _prepare(self):
        # Initialize empty arrays
        self.z_oxygen = np.zeros((self.n_frames, self.oxygen_ag.n_atoms))
        self.cos_theta_OH = np.zeros((self.n_frames, self.oxygen_ag.n_atoms))
        self.cns = np.zeros((self.n_frames, self.oxygen_ag.n_atoms), dtype=int)

        self.z1 = np.zeros(self.n_frames)
        self.z2 = np.zeros(self.n_frames)
        self.cross_area = 0.0

    def _single_frame(self):
        ts_box = self._ts.dimensions
        ts_area = Cell.new(ts_box).area(self.axis)
        self.cross_area += ts_area

        coords = self._ts.positions

        # Absolute surface positions
        surf1_z = coords[self.surf_ids[0], self.axis]
        surf2_z = coords[self.surf_ids[1], self.axis]
        box_length = ts_box[self.axis]
        # Use MIC in case part of the surface crosses the cell boundaries
        self.z1[self._frame_index] = utils.mic_1d(
            surf1_z, box_length, ref=surf1_z[0]
        ).mean()
        self.z2[self._frame_index] = utils.mic_1d(
            surf2_z, box_length, ref=surf2_z[0]
        ).mean()

        coords_oxygen = self.oxygen_ag.positions
        coords_hydrogen = self.hydrogen_ag.positions

        all_distances = np.zeros((coords_hydrogen.shape[0], coords_oxygen.shape[0]))
        distance_array(coords_hydrogen, coords_oxygen, result=all_distances, box=ts_box)
        # H to O mapping
        H_to_O_mapping = np.argmin(all_distances, axis=1)
        oxygen_ids, cns = np.unique(H_to_O_mapping, return_counts=True)

        np.copyto(
            self.z_oxygen[self._frame_index],
            np.pad(
                coords_oxygen[oxygen_ids, self.axis],
                (0, len(coords_oxygen) - len(oxygen_ids)),
                mode="constant",
                constant_values=np.nan,
            ),
        )
        np.copyto(
            self.cns[self._frame_index],
            np.pad(
                cns,
                (0, len(coords_oxygen) - len(oxygen_ids)),
                mode="constant",
                constant_values=-1,
            ),
        )

        OH_vectors = minimize_vectors(
            coords_hydrogen - coords_oxygen[H_to_O_mapping], box=ts_box
        )
        dipoles = np.zeros((len(oxygen_ids), 3))
        for count, ii in enumerate(oxygen_ids):
            dipoles[count] = OH_vectors[np.where(H_to_O_mapping == ii)[0]].mean(axis=0)
        cos_theta = (dipoles[:, self.axis]) / np.linalg.norm(dipoles, axis=-1)
        np.copyto(
            self.cos_theta_OH[self._frame_index],
            np.pad(
                cos_theta,
                (0, len(coords_oxygen) - len(oxygen_ids)),
                mode="constant",
                constant_values=np.nan,
            ),
        )

    def _conclude(self):
        # Average surface area
        self.cross_area /= self.n_frames

        box_length = self.universe.dimensions[self.axis]
        # Step 1: Set z1 as the reference point (zero)
        z1 = np.zeros(self.z1.shape)
        # Step 2: Calculate z2 positions relative to z1 using minimum image convention
        # By setting ref=box_length/2, all positions are within one cell length positive of z1
        z2 = utils.mic_1d(
            self.z2 - self.z1,
            box_length=box_length,
            ref=box_length / 2,
        )
        # Step 3: Calculate z_water relative to z_1, ensuring that all water positions are
        # within one cell length positive of z1.
        z_oxygen = utils.mic_1d(
            self.z_oxygen - self.z1[:, np.newaxis],
            box_length=box_length,
            ref=box_length / 2,
        )

        # Update attributes to the final relative coordinates
        self.z1 = z1
        self.z2 = z2
        self.z_oxygen = z_oxygen

        self.results.rho_water = self.calc_density_profile(cn=2)
        self.results.geo_dipole_water = self.calc_orientation_profile(cn=2)
        self.results.rho_OH = self.calc_density_profile(cn=1)
        self.results.geo_dipole_OH = self.calc_orientation_profile(cn=1)

    def calc_density_profile(
        self, cn: int = 2, sym: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        mol_mass = 15.999 + cn * 1.008

        z1_mean = np.mean(self.z1)
        z2_mean = np.mean(self.z2)

        mask = self.cns == cn
        counts, bin_edges = np.histogram(
            self.z_oxygen[mask],
            bins=int((z2_mean - z1_mean) / self.dz),
            range=(z1_mean, z2_mean),
        )

        # Spatial coordinates
        z = utils.bin_edges_to_grid(bin_edges)

        # Density values
        n = counts / self.n_frames
        grid_volume = np.diff(bin_edges) * self.cross_area
        rho = utils.calc_density(n, grid_volume, mol_mass)
        if sym:
            rho = (rho[::-1] + rho) / 2
        return z, rho

    def calc_orientation_profile(
        self, cn: int = 2, sym: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        mol_mass = 15.999 + cn * 1.008

        z1_mean = np.mean(self.z1)
        z2_mean = np.mean(self.z2)

        mask = self.cns == cn
        counts, bin_edges = np.histogram(
            self.z_oxygen[mask],
            bins=int((z2_mean - z1_mean) / self.dz),
            range=(z1_mean, z2_mean),
            weights=self.cos_theta_OH[mask],
        )

        z = utils.bin_edges_to_grid(bin_edges)
        n = counts / self.n_frames
        grid_volume = np.diff(bin_edges) * self.cross_area
        rho_cos_theta = utils.calc_density(n, grid_volume, mol_mass)

        if sym:
            rho_cos_theta = (rho_cos_theta - rho_cos_theta[::-1]) / 2
        return z, rho_cos_theta

    def calc_sel_species(
        self, interval: Tuple[float, float], n_bins: int = 90, cn: int = 2
    ) -> Tuple[float, list[np.ndarray]]:
        """
        Calculate properties of water in a selected region relative to the surfaces.

        Parameters
        ----------
        interval : tuple of float
            The interval range to select the region of interest.
        n_bins : int, optional
            The number of bins for the histogram (default is 90).

        Returns
        -------
        n_water : float
            The number of water molecules in the selected region.
        histogram : list[ndarray]
            A list [x, y] containing the grid of angles (x) and the
            probability density of the angular distribution (y).
        """
        cn_mask = self.cns == cn

        mask = (
            (self.z_oxygen > (self.z1[:, np.newaxis] + interval[0]))
            & (self.z_oxygen <= (self.z1[:, np.newaxis] + interval[1]))
            & cn_mask
        )

        n_water = np.count_nonzero(mask, axis=1)
        lower_surface_angles = np.arccos(self.cos_theta_OH[mask]) / np.pi * 180

        mask = (
            (self.z_oxygen < (self.z2[:, np.newaxis] - interval[0]))
            & (self.z_oxygen >= (self.z2[:, np.newaxis] - interval[1]))
            & cn_mask
        )
        upper_surface_angles = np.arccos(-self.cos_theta_OH[mask]) / np.pi * 180
        n_water += np.count_nonzero(mask, axis=1)

        combined_angles = np.concatenate([lower_surface_angles, upper_surface_angles])
        angle_distribution, bin_edges = np.histogram(
            combined_angles,
            bins=n_bins,
            range=(0.0, 180.0),
            density=True,
        )
        grid = utils.bin_edges_to_grid(bin_edges)
        return n_water, [grid, angle_distribution]
