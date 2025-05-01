# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Functionality for computing vibrational spectra from molecular dynamics
trajectories of water at interfaces
"""

from typing import List, Optional, Union

import numpy as np
from MDAnalysis import Universe
from MDAnalysis.analysis.base import AnalysisBase, Results

from scipy import signal
from scipy.constants import speed_of_light

from . import utils
from .multitrajbase import MultiTrajsAnalysisBase
from .waterdynamics import calc_vector_autocorrelation


def calc_full_vacf(velocities: np.ndarray) -> np.ndarray:
    """
    Calculate the full velocity autocorrelation function (VACF).

    Parameters
    ----------
    velocities : np.ndarray
        The velocities of the atoms in the system.

    Returns
    -------
    full_vacf: np.ndarray
        The full normalised VACF including both positive and negative lags.
    """
    full_vacf_x = [
        signal.correlate(velocities[:, ii, 0], velocities[:, ii, 0])
        for ii in range(velocities.shape[1])
    ]
    full_vacf_y = [
        signal.correlate(velocities[:, ii, 1], velocities[:, ii, 1])
        for ii in range(velocities.shape[1])
    ]
    full_vacf_z = [
        signal.correlate(velocities[:, ii, 2], velocities[:, ii, 2])
        for ii in range(velocities.shape[1])
    ]
    full_vacf = (
        np.mean(full_vacf_x, axis=0)
        + np.mean(full_vacf_y, axis=0)
        + np.mean(full_vacf_z, axis=0)
    )
    del full_vacf_x, full_vacf_y, full_vacf_z
    # Normalize ACF
    full_vacf = full_vacf / full_vacf.max()
    return full_vacf


def calc_power_spectrum(vacf: np.ndarray, ts: float, full: bool):
    """
    Calculate the power spectrum.

    Parameters
    ----------
    vacf : np.ndarray
        The normalised VACF.
    ts : float
        The time step of the simulation in picoseconds.
    full : bool
        Whether the provided VACF includes both positive and negative lags, or
        only positive lags. If False, symmetrizes the provided VACF around x=0
        to obtain the full VACF.

    Returns
    -------
    wave_numbers: np.ndarray
        The positive frequencies of the power spectrum in unit of 1 / cm.
    power_spectrum: np.ndarray
        The power spectrum of the VACF corresponding to the wavenumbers.
    """
    if full:
        full_vacf = vacf
    else:
        full_vacf = np.concatenate([vacf[::-1][:-1], vacf])
    power_spectrum = np.abs(np.fft.fft(full_vacf))
    freqs = np.fft.fftfreq(full_vacf.size, ts)
    wave_numbers = freqs * 1e12 / speed_of_light / 100
    return (
        wave_numbers[: wave_numbers.size // 2],
        power_spectrum[: wave_numbers.size // 2],
    )


class InterfaceVACFDeprecated(MultiTrajsAnalysisBase):
    def __init__(
        self,
        universe_pos: Universe,
        universe_vel: Universe,
        surf_ids: Union[List, np.ndarray] = None,
        max_tau: int = None,
        d_tau: int = None,
        interval: Optional[List[float]] = None,
        **kwargs,
    ):
        self.universe_pos = universe_pos
        self.universe_vel = universe_vel

        self.surf_ids = surf_ids

        assert d_tau > 0
        assert max_tau >= d_tau
        self.max_tau = max_tau
        self.d_tau = d_tau
        self.tau_list = np.arange(0, max_tau + d_tau, d_tau, dtype=int)

        if interval is not None:
            assert len(interval) == 2
            assert interval[1] > interval[0]
        self.interval = interval
        self.axis = kwargs.pop("axis", 2)
        self.oxygen_ag = self.universe_pos.select_atoms(
            kwargs.pop("oxygen_sel", "name O")
        )
        sel_kw = kwargs.pop("hydrogen_sel", "name H")
        self.hydrogen_vel_ag = self.universe_vel.select_atoms(sel_kw)
        self.water_dict = utils.identify_water_molecules(
            self.universe_pos.select_atoms(sel_kw).positions,
            self.oxygen_ag.positions,
            self.universe_pos.dimensions,
            oh_cutoff=kwargs.pop("oh_cutoff", 1.3),
            ignore_warnings=kwargs.pop("ignore_warnings", False),
        )

        super().__init__([universe_pos.trajectory, universe_vel.trajectory], **kwargs)

        self._vacf = None
        self._oxygen_mask = None
        self._hydrogen_velocities = None

    def _prepare(self):
        self._vacf = np.full([len(self.tau_list), self.n_frames], np.nan, np.float64)

        self._oxygen_mask = np.zeros(
            [self.max_tau + 1, len(self.oxygen_ag)], dtype=bool
        )
        self._hydrogen_velocities = np.zeros(
            [self.max_tau + 1, len(self.hydrogen_vel_ag), 3]
        )

    def _single_frame(self):
        # start_idx
        start_idx = self._frame_index % (self.max_tau + 1)

        ts_pos = self._all_ts[0]

        ts_box = ts_pos.dimensions
        coords = ts_pos.positions
        coords_oxygen = self.oxygen_ag.positions
        velocities_hydrogen = self.hydrogen_vel_ag.positions

        # Absolute surface positions
        surf1_z = coords[self.surf_ids[0], self.axis]
        surf2_z = coords[self.surf_ids[1], self.axis]
        box_length = ts_box[self.axis]
        # Use MIC in case part of the surface crosses the cell boundaries
        z1 = utils.mic_1d(surf1_z, box_length, ref=surf1_z[0]).mean()
        z2 = utils.mic_1d(surf2_z, box_length, ref=surf2_z[0]).mean()

        z_hi = utils.mic_1d(
            z2 - z1,
            box_length=box_length,
            ref=box_length / 2,
        )
        z_oxygen = utils.mic_1d(
            coords_oxygen[:, self.axis] - z1,
            box_length=box_length,
            ref=box_length / 2,
        )

        if self.interval is not None:
            mask_lo = (z_oxygen > self.interval[0]) & (z_oxygen <= self.interval[1])
            mask_hi = ((z_hi - z_oxygen) > self.interval[0]) & (
                (z_hi - z_oxygen) <= self.interval[1]
            )
            # mask to select water in this frame
            mask = mask_lo | mask_hi
        else:
            mask = np.ones(len(z_oxygen), dtype=bool)

        np.copyto(self._oxygen_mask[start_idx], mask)
        np.copyto(self._hydrogen_velocities[start_idx], velocities_hydrogen)

        for ii, tau in enumerate(self.tau_list):
            if self._frame_index - tau >= 0:
                end_idx = (self._frame_index - tau) % (self.max_tau + 1)
                old_mask = self._oxygen_mask[end_idx]
                # mask for oxygen, the hydrogens attached to which are used to calculate VACF
                mask_vacf = mask & old_mask
                sel_hydrogen_ids = []
                for ii_oxygen in np.where(mask_vacf)[0]:
                    sel_hydrogen_ids.append(self.water_dict[ii_oxygen])
                sel_hydrogen_ids = np.concatenate(sel_hydrogen_ids)
                self._vacf[ii, self._frame_index] = np.mean(
                    np.sum(
                        velocities_hydrogen[sel_hydrogen_ids]
                        * self._hydrogen_velocities[end_idx][sel_hydrogen_ids],
                        axis=-1,
                    )
                )

    def _conclude(self):
        self.results.vacf = [np.mean(vacf[~np.isnan(vacf)]) for vacf in self._vacf]

        # vacf = [np.mean(vacf[~np.isnan(vacf)]) for vacf in self._vacf]
        # self.results.vacf = np.array(vacf) / vacf[0]
        # self.results.full_vacf = np.concatenate(
        #     [self.results.vacf[::-1][:-1], self.results.vacf]
        # )


class InterfaceVACF(MultiTrajsAnalysisBase):
    """
    InterfaceVACF class for calculating the velocity autocorrelation function (VACF)
    of hydrogen atoms in a specified interval relative to a surface.

    Parameters
    ----------
    universe_pos : Universe
        The MDAnalysis Universe object containing the position trajectory.
    universe_vel : Universe
        The MDAnalysis Universe object containing the velocity trajectory.
    surf_ids : Union[List, np.ndarray], optional
        List or array of surface atom indices. Default is None.
    interval : Optional[List[float]], optional
        List of two floats specifying the interval relative to the surface
        within which to calculate the VACF. Default is None.
    **kwargs : dict
        Additional keyword arguments for customization.
        - axis : int, optional.
            The axis along which to calculate the VACF. Default is 2.
        - oxygen_sel : str, optional.
            Selection string for oxygen atoms. Default is "name O".
        - hydrogen_sel : str, optional.
            Selection string for hydrogen atoms. Default is "name H".
        - oh_cutoff : float, optional.
            Cutoff distance for identifying water molecules. Default is 1.3.
        - ignore_warnings : bool, optional.
            Whether to ignore warnings during water molecule identification. Default is False.

    Methods
    -------
    run(start=None, stop=None, step=None, verbose=False)
        Collect data for the provided trajectory.
    calc_vacf(max_tau, delta_tau=1, correlation_step=1)
        Calculate the VACF from the collected data. Returns the time delay axis and calculated VACF.

    Outputs
    -------
    The VACF and the time delay axis can be calculated and accessed from the calc_vacf method.
    """

    def __init__(
        self,
        universe_pos: Universe,
        universe_vel: Universe,
        surf_ids: Union[List, np.ndarray] = None,
        interval: Optional[List[float]] = None,
        **kwargs,
    ):
        self.universe_pos = universe_pos
        self.universe_vel = universe_vel

        self.surf_ids = surf_ids
        self.n_frames = len(universe_pos.trajectory)

        if interval is not None:
            assert len(interval) == 2
            assert interval[1] > interval[0]
        self.interval = interval

        # Get kwargs
        self.axis = kwargs.pop("axis", 2)
        self.oxygen_ag = self.universe_pos.select_atoms(
            kwargs.pop("oxygen_sel", "name O")
        )
        hydrogen_sel = kwargs.pop("hydrogen_sel", "name H")
        self.hydrogen_vel_ag = self.universe_vel.select_atoms(hydrogen_sel)

        # Guess water molecule topology
        self.water_dict = utils.identify_water_molecules(
            self.universe_pos.select_atoms(hydrogen_sel).positions,
            self.oxygen_ag.positions,
            self.universe_pos.dimensions,
            oh_cutoff=kwargs.pop("oh_cutoff", 1.3),
            ignore_warnings=kwargs.pop("ignore_warnings", False),
        )

        super().__init__([universe_pos.trajectory, universe_vel.trajectory], **kwargs)

        self._vacf = None
        self._oxygen_mask = None
        self._hydrogen_velocities = None

    def _prepare(self):
        self._oxygen_mask = np.zeros([self.n_frames, len(self.oxygen_ag)], dtype=bool)
        self._hydrogen_velocities = np.zeros(
            [self.n_frames, len(self.hydrogen_vel_ag), 3], dtype=np.float32
        )

    def _single_frame(self):
        start_idx = self._frame_index  # % (self.max_tau + 1)

        # Get positions from the position trajectory
        ts_pos = self._all_ts[0]
        ts_box = ts_pos.dimensions
        coords = ts_pos.positions
        coords_oxygen = self.oxygen_ag.positions

        # Get velocities from the velocity trajectory
        velocities_hydrogen = self.hydrogen_vel_ag.positions

        # Absolute surface positions
        surf1_z = coords[self.surf_ids[0], self.axis]
        surf2_z = coords[self.surf_ids[1], self.axis]
        box_length = ts_box[self.axis]

        # Use MIC in case part of the surface crosses the cell boundaries
        z1 = utils.mic_1d(surf1_z, box_length, ref=surf1_z[0]).mean()
        z2 = utils.mic_1d(surf2_z, box_length, ref=surf2_z[0]).mean()

        # Define all coordinates with respect to surface z1, wrap to first unit cell
        z_hi = utils.mic_1d(
            z2 - z1,
            box_length=box_length,
            ref=box_length / 2,
        )
        z_oxygen = utils.mic_1d(
            coords_oxygen[:, self.axis] - z1,
            box_length=box_length,
            ref=box_length / 2,
        )

        # Create mask for oxygen atoms within the interval; used to select hydrogens later
        if self.interval is not None:
            mask_lo = (z_oxygen > self.interval[0]) & (z_oxygen <= self.interval[1])
            mask_hi = ((z_hi - z_oxygen) > self.interval[0]) & (
                (z_hi - z_oxygen) <= self.interval[1]
            )
            mask = mask_lo | mask_hi
        else:
            mask = np.ones(len(z_oxygen), dtype=bool)

        np.copyto(self._oxygen_mask[start_idx], mask)
        np.copyto(self._hydrogen_velocities[start_idx], velocities_hydrogen)

    def calc_vacf(
        self,
        max_tau: int,
        delta_tau: int = 1,
        correlation_step: int = 1,
    ):
        """
        Calculate the velocity autocorrelation function (VACF) for hydrogen atoms in
        the interval given by self.interval.

        Parameters
        ----------
        max_tau : int
            The maximum time lag (tau) for which to calculate the VACF.
        delta_tau : int, default=1
            The interval between the points on the VACF vs. tau curve. Should be at least 1.
        correlation_step : int, default=1
            The step size for the correlation calculation. Choosing a larger correlation
            step size speeds up calculation at the cost of less statistics. A larger correlation
            step size also reduces correlation between VACF values computed from different
            time origins.

        Returns
        -------
        tau : ndarray
            Array of time lags (tau) for which the VACF is calculated.
        vacf : ndarray
            Array of VACF values corresponding to each time lag.
        """
        mask = np.zeros(self._hydrogen_velocities.shape[:2], dtype=bool)

        for i, mask_ts in enumerate(mask):
            oxygen_ids = np.nonzero(self._oxygen_mask[i])[0]
            sel_hydrogen_ids = []
            for ii in oxygen_ids:
                sel_hydrogen_ids.append(self.water_dict[ii])
            sel_hydrogen_ids = np.concatenate(sel_hydrogen_ids)
            mask_ts[sel_hydrogen_ids] = True

        tau, vacf = calc_vector_autocorrelation(
            max_tau=max_tau,
            delta_tau=delta_tau,
            step=correlation_step,
            vectors=self._hydrogen_velocities,
            mask=mask,
        )
        return tau, vacf

class InterfaceVACF2(AnalysisBase):
    def __init__(
        self,
        universe: Universe,
        surf_ids: Union[List, np.ndarray] = None,
        interval: Optional[List[float]] = None,
        **kwargs,
    ):
        self.universe = universe
        super().__init__(universe.trajectory, verbose=kwargs.get("verbose", False))
        
        self.surf_ids = surf_ids
        self.n_frames = universe.trajectory.n_frames

        if interval is not None:
            assert len(interval) == 2
            assert interval[1] > interval[0]
        self.interval = interval

        # Get kwargs
        self.axis = kwargs.pop("axis", 2)
        self.oxygen_ag = self.universe.select_atoms(
            kwargs.pop("oxygen_sel", "name O")
        )
        self.hydrogen_ag = self.universe.select_atoms(
            kwargs.pop("hydrogen_sel", "name H")
        )
        
        # Guess water molecule topology
        print(self.hydrogen_ag.positions.shape, self.oxygen_ag.positions.shape)
        self.water_dict = utils.identify_water_molecules(
            self.hydrogen_ag.positions,
            self.oxygen_ag.positions,
            self.universe.dimensions,
            oh_cutoff=kwargs.pop("oh_cutoff", 1.3),
            ignore_warnings=kwargs.pop("ignore_warnings", False),
        )

        self._vacf = None
        self._oxygen_mask = None
        self._hydrogen_velocities = None

    def _prepare(self):
        self._oxygen_mask = np.zeros([self.n_frames, len(self.oxygen_ag)], dtype=bool)
        self._hydrogen_velocities = np.zeros(
            [self.n_frames, len(self.hydrogen_ag), 3], dtype=np.float32
        )

    def _single_frame(self):
        start_idx = self._frame_index  # % (self.max_tau + 1)

        # Get positions from the position trajectory
        ts_box = self._ts.dimensions
        coords = self._ts.positions
        coords_oxygen = self.oxygen_ag.positions

        # Get velocities from the velocity trajectory
        velocities_hydrogen = self.hydrogen_ag.velocities

        # Absolute surface positions
        surf1_z = coords[self.surf_ids[0], self.axis]
        surf2_z = coords[self.surf_ids[1], self.axis]
        box_length = ts_box[self.axis]

        # Use MIC in case part of the surface crosses the cell boundaries
        z1 = utils.mic_1d(surf1_z, box_length, ref=surf1_z[0]).mean()
        z2 = utils.mic_1d(surf2_z, box_length, ref=surf2_z[0]).mean()

        # Define all coordinates with respect to surface z1, wrap to first unit cell
        z_hi = utils.mic_1d(
            z2 - z1,
            box_length=box_length,
            ref=box_length / 2,
        )
        z_oxygen = utils.mic_1d(
            coords_oxygen[:, self.axis] - z1,
            box_length=box_length,
            ref=box_length / 2,
        )

        # Create mask for oxygen atoms within the interval; used to select hydrogens later
        if self.interval is not None:
            mask_lo = (z_oxygen > self.interval[0]) & (z_oxygen <= self.interval[1])
            mask_hi = ((z_hi - z_oxygen) > self.interval[0]) & (
                (z_hi - z_oxygen) <= self.interval[1]
            )
            mask = mask_lo | mask_hi
        else:
            mask = np.ones(len(z_oxygen), dtype=bool)

        np.copyto(self._oxygen_mask[start_idx], mask)
        np.copyto(self._hydrogen_velocities[start_idx], velocities_hydrogen)

    def calc_vacf(
        self,
        max_tau: int,
        delta_tau: int = 1,
        correlation_step: int = 1,
    ):
        """
        Calculate the velocity autocorrelation function (VACF) for hydrogen atoms in
        the interval given by self.interval.

        Parameters
        ----------
        max_tau : int
            The maximum time lag (tau) for which to calculate the VACF.
        delta_tau : int, default=1
            The interval between the points on the VACF vs. tau curve. Should be at least 1.
        correlation_step : int, default=1
            The step size for the correlation calculation. Choosing a larger correlation
            step size speeds up calculation at the cost of less statistics. A larger correlation
            step size also reduces correlation between VACF values computed from different
            time origins.

        Returns
        -------
        tau : ndarray
            Array of time lags (tau) for which the VACF is calculated.
        vacf : ndarray
            Array of VACF values corresponding to each time lag.
        """
        mask = np.zeros(self._hydrogen_velocities.shape[:2], dtype=bool)

        for i, mask_ts in enumerate(mask):
            oxygen_ids = np.nonzero(self._oxygen_mask[i])[0]
            sel_hydrogen_ids = []
            for ii in oxygen_ids:
                sel_hydrogen_ids.append(self.water_dict[ii])
            sel_hydrogen_ids = np.concatenate(sel_hydrogen_ids)
            mask_ts[sel_hydrogen_ids] = True

        tau, vacf = calc_vector_autocorrelation(
            max_tau=max_tau,
            delta_tau=delta_tau,
            step=correlation_step,
            vectors=self._hydrogen_velocities,
            mask=mask,
        )
        return tau, vacf