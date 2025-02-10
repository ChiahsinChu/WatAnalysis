import numpy as np
from scipy import signal


def calc_full_vacf(velocities: np.ndarray) -> np.ndarray:
    """
    Calculate the full velocity autocorrelation function (VACF).

    Parameters
    ----------
    velocities: np.ndarray
        The velocities of the atoms in the system.

    Returns
    -------
    full_vacf: np.ndarray
        The full normalised VACF including both positive and negative lags.
    """
    full_vacf_x = signal.correlate(velocities[:, :, 0], velocities[:, :, 0])
    full_vacf_y = signal.correlate(velocities[:, :, 1], velocities[:, :, 1])
    full_vacf_z = signal.correlate(velocities[:, :, 2], velocities[:, :, 2])
    full_vacf = full_vacf_x + full_vacf_y + full_vacf_z
    del full_vacf_x, full_vacf_y, full_vacf_z
    full_vacf = full_vacf.mean(axis=1)
    # Normalize ACF
    full_vacf = full_vacf / full_vacf.max()
    return full_vacf


def calc_power_spectrum(full_vacf, ts):
    """
    Calculate the power spectrum.

    Parameters
    ----------
    full_vacf: np.ndarray
        The full normalised VACF including both positive and negative lags.
    ts: float
        The time step of the simulation.

    Returns
    -------
    freqs: np.ndarray
        The frequencies of the power spectrum in unit of 1 / time unit of ts.
    power_spectrum: np.ndarray
        The power spectrum of the VACF.
    """
    power_spectrum = np.abs(np.fft.fft(full_vacf))
    freqs = np.fft.fftfreq(full_vacf.size, ts)
    return freqs, power_spectrum

