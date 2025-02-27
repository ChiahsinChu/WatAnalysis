"""
Class for analysis of atomistic simulations of confined water. Collects
all relevant data from a trajectory, and the user can then choose which 
analyses to perform.
"""

from typing import List, Tuple, Union, Optional

import numpy as np
from ase.cell import Cell
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.core.universe import Universe

from . import utils
from . import waterstructure
from . import waterdynamics


class WaterAnalysis(AnalysisBase):
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
    run(start=None, stop=None, step=None, verbose=False)
        Collect data from the trajectory.
    density_profile(only_valid_dipoles=False, sym=False, dz=None)
        Compute the time-averaged water density profile.
    orientation_profile(sym=False, dz=None)
        Compute the time-averaged water orientation profile.
    costheta_profile(sym=False, dz=None)
        Compute the time-averaged <cos theta> profile.
    count_in_region(interval, only_valid_dipoles=False)
        Count the number of water molecules in a selected region.
    angular_distribution(interval, n_bins=90)
        Calculate the distribution of dipole angles for water in a selected region.
    dipole_autocorrelation(max_tau, delta_tau, interval, step=1)
        Calculate the water dipole autocorrelation.
    survival_probability(max_tau, delta_tau, interval, step=1)
        Calculate the water survival probability.
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

        self.results.z_water = None
        self.results.cos_theta = None
        self.results.z1 = None
        self.results.z2 = None
        self.results.cross_area = None
        self.results.dipoles = None

    def _prepare(self):
        # Initialize empty arrays
        self.results.z_water = np.zeros((self.n_frames, self.oxygen_ag.n_atoms))
        self.results.cos_theta = np.zeros((self.n_frames, self.oxygen_ag.n_atoms))
        self.results.z1 = np.zeros(self.n_frames)
        self.results.z2 = np.zeros(self.n_frames)
        self.results.dipoles = np.zeros((self.n_frames, self.oxygen_ag.n_atoms, 3))
        self.results.cross_area = 0.0

    def _single_frame(self):
        ts_box = self._ts.dimensions
        ts_area = Cell.new(ts_box).area(self.axis)
        self.results.cross_area += ts_area

        coords = self._ts.positions
        coords_oxygen = self.oxygen_ag.positions
        coords_hydrogen = self.hydrogen_ag.positions

        # Absolute surface positions
        surf1_z = coords[self.surf_ids[0], self.axis]
        surf2_z = coords[self.surf_ids[1], self.axis]
        box_length = ts_box[self.axis]
        # Use MIC in case part of the surface crosses the cell boundaries
        self.results.z1[self._frame_index] = utils.mic_1d(
            surf1_z, box_length, ref=surf1_z[0]
        ).mean()
        self.results.z2[self._frame_index] = utils.mic_1d(
            surf2_z, box_length, ref=surf2_z[0]
        ).mean()

        # Save oxygen locations for water density analysis
        np.copyto(self.results.z_water[self._frame_index], coords_oxygen[:, self.axis])

        # Calculate dipoles and project on self.axis
        dipole = waterstructure.calc_water_dipoles(
            coords_hydrogen,
            coords_oxygen,
            self.water_dict,
            box=ts_box,
            mic=self.min_vector,
        )
        dipole /= np.linalg.norm(dipole, axis=-1, keepdims=True)
        np.copyto(self.results.dipoles[self._frame_index], dipole)
        cos_theta = dipole[:, self.axis]
        np.copyto(self.results.cos_theta[self._frame_index], cos_theta)

    def _conclude(self):
        # Average surface area
        self.results.cross_area /= self.n_frames

        box_length = self.universe.dimensions[self.axis]
        # Step 1: Set z1 as the reference point (zero)
        z1 = np.zeros(self.results.z1.shape)
        # Step 2: Calculate z2 positions relative to z1 using minimum image convention
        # By setting ref=box_length/2, all positions are within one cell length positive of z1
        z2 = utils.mic_1d(
            self.results.z2 - self.results.z1,
            box_length=box_length,
            ref=box_length / 2,
        )
        # Step 3: Calculate z_water relative to z_1, ensuring that all water positions are
        # within one cell length positive of z1.
        z_water = utils.mic_1d(
            self.results.z_water - self.results.z1[:, np.newaxis],
            box_length=box_length,
            ref=box_length / 2,
        )

        # Update attributes to the final relative coordinates
        self.results.z1 = z1
        self.results.z2 = z2
        self.results.z_water = z_water

    def density_profile(
        self,
        only_valid_dipoles: bool = False,
        sym: bool = False,
        dz: Optional[float] = None,
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
        dz : float, optional
            Bin width for histogram calculations in Angstroms. Must be positive.
            If not provided, defaults to self.dz (which is 0.1 A by default).

        Returns
        -------
        z : ndarray
            The spatial coordinates corresponding to the bin centers.
        rho : ndarray
            The density values of water molecules.
        """
        # Check valid water molecules (O with 2 H)
        # In this way, the density rho corresponds to the density in the
        # orientation profile rho * <cos theta>
        if only_valid_dipoles:
            valid = ~np.isnan(self.results.cos_theta.flatten())
        else:
            valid = np.ones(self.results.z_water.flatten().shape, dtype=bool)

        if dz is None:
            dz = self.dz

        return waterstructure.calc_density_profile(
            (self.results.z1.mean(), self.results.z2.mean()),
            self.results.z_water.flatten()[valid],
            cross_area=self.results.cross_area,
            n_frames=self.n_frames,
            dz=dz,
            sym=sym,
        )

    def orientation_profile(
        self, sym: bool = False, dz: Optional[float] = None
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
        dz : float, optional
            Bin width for histogram calculations in Angstroms. Must be positive.
            If not provided, defaults to self.dz (which is 0.1 A by default).

        Returns
        -------
        z : ndarray
            The grid points along the z-axis.
        rho_cos_theta : ndarray
            The orientation profile of water molecules.
        """
        if dz is None:
            dz = self.dz

        return waterstructure.calc_orientation_profile(
            (self.results.z1.mean(), self.results.z2.mean()),
            self.results.z_water,
            self.results.cos_theta,
            self.results.cross_area,
            dz,
            sym=sym,
        )

    def costheta_profile(
        self, sym: bool = False, dz: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the average cosine theta profile.

        Parameters
        ----------
        sym : bool, optional
            If True, the density profile is symmetrized about the center of the
            simulation box. Default is False.
        dz : float, optional
            Bin width for histogram calculations in Angstroms. Must be positive.
            If not provided, defaults to self.dz (which is 0.1 A by default).

        Returns
        -------
        z : ndarray
            The grid points along the z-axis.
        avg_cos_theta : ndarray
            The cosine theta profile.
        """
        if dz is None:
            dz = self.dz

        return waterstructure.calc_costheta_profile(
            (self.results.z1.mean(), self.results.z2.mean()),
            self.results.z_water,
            self.results.cos_theta,
            dz,
            sym=sym,
        )

    def count_in_region(
        self, interval: Tuple[float, float], only_valid_dipoles: bool = False
    ):
        """
        Count the number of water molecules in a specified region.

        Parameters
        ----------
        interval : Tuple[float, float]
            The interval (z1, z2) defining the region, relative to the surfaces,
            in which to count water molecules.
        only_valid_dipoles : bool, optional
            If True, only consider valid dipoles (non-NaN values in self.results.cos_theta).
            Default is False.

        Returns
        -------
        np.ndarray
            The number of water molecules in the specified region over time for all analyzed
            frames.
        """

        if only_valid_dipoles:
            valid = ~np.isnan(self.results.cos_theta)
        else:
            valid = np.ones(self.results.z_water.shape, dtype=bool)

        mask_lo, mask_hi = utils.get_region_masks(
            self.results.z_water, self.results.z1, self.results.z2, interval
        )

        return np.count_nonzero(mask_lo & valid, axis=1) + np.count_nonzero(
            mask_hi & valid, axis=1
        )

    def angular_distribution(
        self, interval: Tuple[float, float], n_bins: int = 90
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate distribution of dipole angles for water in a selected
        region relative to the surfaces.

        Parameters
        ----------
        interval : Tuple of float
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
        valid = ~np.isnan(self.results.cos_theta)
        mask_lo, mask_hi = utils.get_region_masks(
            self.results.z_water, self.results.z1, self.results.z2, interval
        )

        return waterstructure.calc_angular_distribution(
            mask_lo & valid,
            mask_hi & valid,
            self.results.cos_theta,
            n_bins=n_bins,
        )

    def dipole_autocorrelation(
        self,
        max_tau: int,
        delta_tau: int,
        interval: Tuple[float, float],
        step: int = 1,
    ):
        """
        Calculate the autocorrelation function for water molecule dipole vectors.

        Parameters
        ----------
        max_tau : int
            Maximum lag time to calculate the autocorrelation function C(tau) for
        delta_tau : int
            Time interval between lag times (points on the C(tau) vs. tau curve)
        interval : Tuple[float, float]
            The region of interest defined by a tuple of two floats representing
            the lower and upper bounds relative to the surface, in Angstrom.
        step : int
            Step size for time origins. If equal to max_tau, there is no overlap between
            time windows considered in the calculation (so more uncorrelated).

        Returns
        -------
        tau : numpy.ndarray
            Array of lag times
        acf : numpy.ndarray
            Normalized dipole autocorrelation function values for each lag time
        """
        valid = ~np.isnan(self.results.cos_theta)
        mask_lo, mask_hi = utils.get_region_masks(
            self.results.z_water, self.results.z1, self.results.z2, interval
        )
        return waterdynamics.calc_vector_autocorrelation(
            max_tau=max_tau,
            delta_tau=delta_tau,
            step=step,
            vectors=self.results.dipoles,
            mask=(mask_lo | mask_hi) & valid,
        )

    def survival_probability(
        self,
        max_tau: int,
        delta_tau: int,
        interval: Tuple[float, float],
        step: int = 1,
    ) -> np.ndarray:
        """
        Calculate the water survival probability.

        This method calculates the probability that water molecules will remain within a
        specified region over a given time interval.

        Parameters
        ----------
        max_tau : int
            The maximum time delay for which the survival probability is calculated.
        delta_tau : int
            The time delay interval for calculating the survival probability (spacing
            between points on the survival probability vs. tau curve).
        interval : Tuple[float, float]
            The region of interest defined by a tuple of two floats representing
            the lower and upper bounds relative to the surface, in Angstrom.
        step : int, optional
            The step size between time origins that are taken into account.
            By increasing the step the analysis can be sped up at a loss of statistics.
            If equal to max_tau, there is no overlap between time windows considered in the
            calculation (so more uncorrelated). Defaults to 1.

        Returns
        -------
        tau : numpy.ndarray
            Array of lag times
        acf : numpy.ndarray
            Survival probability values for each lag time
        """
        mask_lo, mask_hi = utils.get_region_masks(
            self.results.z_water, self.results.z1, self.results.z2, interval
        )
        return waterdynamics.calc_survival_probability(
            max_tau=max_tau,
            delta_tau=delta_tau,
            step=step,
            mask=mask_lo | mask_hi,
        )
