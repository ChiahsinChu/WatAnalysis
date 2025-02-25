# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import List, Union, Optional

import numpy as np
from scipy import signal

from MDAnalysis import Universe

from .multitrajbase import MultiTrajsAnalysisBase
from . import utils


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
    full_vacf_x = [signal.correlate(velocities[:, ii, 0], velocities[:, ii, 0]) for ii in range(velocities.shape[1])]
    full_vacf_y = [signal.correlate(velocities[:, ii, 1], velocities[:, ii, 1]) for ii in range(velocities.shape[1])]
    full_vacf_z = [signal.correlate(velocities[:, ii, 2], velocities[:, ii, 2]) for ii in range(velocities.shape[1])]
    full_vacf = np.mean(full_vacf_x, axis=0) + np.mean(full_vacf_y, axis=0) + np.mean(full_vacf_z, axis=0)
    del full_vacf_x, full_vacf_y, full_vacf_z
    # Normalize ACF
    full_vacf = full_vacf / full_vacf.max()
    return full_vacf


def calc_power_spectrum(full_vacf, ts):
    """
    Calculate the power spectrum.

    Parameters
    ----------
    full_vacf : np.ndarray
        The full normalised VACF including both positive and negative lags.
    ts : float
        The time step of the simulation.

    Returns
    -------
    freqs: np.ndarray
        The frequencies of the power spectrum in unit of 1 / time unit.
    power_spectrum: np.ndarray
        The power spectrum of the VACF.
    """
    power_spectrum = np.abs(np.fft.fft(full_vacf))
    freqs = np.fft.fftfreq(full_vacf.size, ts)
    return freqs, power_spectrum


class InterfaceVelocityACF(MultiTrajsAnalysisBase):
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

        self._oxygen_mask = np.zeros([self.max_tau + 1, len(self.oxygen_ag)], dtype=bool)
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
                    np.sum(velocities_hydrogen[sel_hydrogen_ids]
                    * self._hydrogen_velocities[end_idx][sel_hydrogen_ids], axis=-1)
                )

    def _conclude(self):
        self.results.vacf = [np.mean(vacf[~np.isnan(vacf)]) for vacf in self._vacf]
