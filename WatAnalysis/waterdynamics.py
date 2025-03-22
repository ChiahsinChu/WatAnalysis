# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Functionality for computing dynamical quantities from molecular dynamics
trajectories of water at interfaces
"""

from typing import Optional

import numpy as np


def calc_vector_correlation(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    max_tau: int,
    delta_tau: int,
    step: int = 1,
    mask: Optional[np.ndarray] = None,
    normalize: bool = True,
    modifier_func: Optional[callable] = None,
):
    """
    Calculate correlation functions for atomic vectorial quantity over time.

    Parameters
    ----------
    vector_a : numpy.ndarray
        Array of vectors with shape (num_timesteps, num_particles, n_dimensions)
    vector_b : numpy.ndarray
        Array of vectors with shape (num_timesteps, num_particles, n_dimensions)
    max_tau : int
        Maximum lag time to calculate ACF for
    delta_tau : int
        Time interval between lag times (points on the C(tau) vs. tau curve)
    step : int
        Step size for time origins. If equal to max_tau, there is no overlap between
        time windows considered in the calculation (so more uncorrelated).
    mask : numpy.ndarray
        Boolean mask array indicating which particles to include, shape
        (num_timesteps, num_particles)
    normalize : bool
        Whether to normalize the ACF by the zero-lag value
    modifier_func : callable
        Function to apply to the dot products before averaging
        Useful when calculating water reorientation correlation functions

    Returns
    -------
    tau : numpy.ndarray
        Array of lag times
    acf : numpy.ndarray
        Normalized autocorrelation function values for each lag time
    """
    tau = np.arange(start=0, stop=max_tau, step=delta_tau)
    acf = np.zeros(tau.shape)
    if mask is None:
        mask = np.ones(vector_a.shape[:2], dtype=bool)
    mask = np.expand_dims(mask, axis=2)

    # Calculate ACF for each lag time
    for i, t in enumerate(tau):
        n_selected_vectors = None
        if t == 0:
            # For t=0, just calculate the dot product with itself
            dot_products = np.sum(
                vector_a * vector_b * mask, axis=2
            )  # Shape: (num_timesteps, num_molecules)
            n_selected_vectors = np.count_nonzero(mask)
        else:
            # For t > 0, calculate the dot products between shifted arrays
            _vectors_0 = vector_a[:-t:step] * mask[:-t:step]  # dipole(t=0)
            _vectors_t = vector_b[t::step] * mask[t::step]  # dipole(t=tau)
            dot_products = np.sum(
                _vectors_0 * _vectors_t, axis=2
            )  # Shape: ((num_timesteps - t)//step, num_molecules)
            n_selected_vectors = np.count_nonzero(mask[:-t:step] * mask[t::step])
        if modifier_func is not None:
            zero_mask = np.isclose(dot_products, 0)
            dot_products = modifier_func(dot_products)
            dot_products[zero_mask] = 0.0
        # Average over molecules and time origins
        acf[i] = np.sum(dot_products) / n_selected_vectors

    if normalize:
        # Normalize the ACF
        acf /= acf[0]  # Normalize by the zero-lag value
    return tau, acf


def calc_vector_autocorrelation(
    max_tau: int,
    delta_tau: int,
    step: int,
    vectors: np.ndarray,
    mask: Optional[np.ndarray] = None,
    normalize: bool = True,
    modifier_func: Optional[callable] = None,
):
    """
    Calculate the autocorrelation functions for atomic vectorial quantity over time.

    Parameters
    ----------
    max_tau : int
        Maximum lag time to calculate ACF for
    delta_tau : int
        Time interval between lag times (points on the C(tau) vs. tau curve)
    step : int
        Step size for time origins. If equal to max_tau, there is no overlap between
        time windows considered in the calculation (so more uncorrelated).
    vectors : numpy.ndarray
        Array of vectors with shape (num_timesteps, num_particles, n_dimensions)
    mask : numpy.ndarray
        Boolean mask array indicating which particles to include, shape
        (num_timesteps, num_particles)

    Returns
    -------
    tau : numpy.ndarray
        Array of lag times
    acf : numpy.ndarray
        Normalized autocorrelation function values for each lag time
    """
    return calc_vector_correlation(
        vector_a=vectors,
        vector_b=vectors,
        max_tau=max_tau,
        delta_tau=delta_tau,
        step=step,
        mask=mask,
        normalize=normalize,
        modifier_func=modifier_func,
    )


def calc_survival_probability(
    max_tau: int,
    delta_tau: int,
    step: int,
    mask: np.ndarray,
):
    """
    Calculate the probability that particles remain within a specified region
    over a given time interval.

    Parameters
    ----------
    max_tau : int
        The maximum time delay for which the survival probability is calculated.
    delta_tau : int
        The time delay interval for calculating the survival probability (spacing
        between points on the survival probability vs. tau curve).
    step : int
        The step size between time origins that are taken into account.
        By increasing the step the analysis can be sped up at a loss of statistics.
        If equal to max_tau, there is no overlap between time windows considered in the
        calculation (so more uncorrelated). Defaults to 1.
    mask : numpy.ndarray
        Boolean mask array indicating which molecules are in the region of interest for
        all time steps, shape (num_timesteps, num_molecules)

    Returns
    -------
    tau : numpy.ndarray
        Array of lag times
    acf : numpy.ndarray
        Survival probability values for each lag time
    """
    tau_range = np.arange(start=0, stop=max_tau, step=delta_tau)
    acf = np.zeros(tau_range.shape)

    # Calculate continuous ACF for each lag time
    for i, tau in enumerate(tau_range):
        if tau > 0:
            # N(t), shape: (num_timesteps - tau, )
            n_t = np.sum(mask, axis=1)[:-tau:step]

            # shape: ((num_timesteps - tau)//step, num_molecules)
            intersection = np.ones(mask[:-tau:step].shape)
            for k in range(tau):
                intersection *= mask[k : -tau + k : step]
            intersection *= mask[tau::step]

            # N(t,tau), shape: (num_timesteps - tau, )
            n_t_tau = np.sum(intersection, axis=1)

            acf[i] = np.mean(n_t_tau / n_t)
        else:
            acf[i] = 1

    # Normalize the ACF
    acf /= acf[0]  # Normalize by the zero-lag value
    return tau_range, acf
