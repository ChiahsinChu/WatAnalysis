# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import Dict, List

import numpy as np
from MDAnalysis.lib.distances import distance_array
from scipy import constants


def get_cum_ave(data):
    """
    Calculate the cumulative average of a data array.

    Parameters
    ----------
    data : np.ndarray
        Array of data points.

    Returns
    -------
    np.ndarray
        Array of cumulative averages.
    """
    cum_sum = data.cumsum()
    cum_ave = cum_sum / (np.arange(len(data)) + 1)
    return cum_ave


def bin_edges_to_grid(bin_edges: np.ndarray):
    """
    Convert bin edges to grid points at bin centers.

    Parameters
    ----------
    bin_edges : np.ndarray
        Array of bin edges.

    Returns
    -------
    np.ndarray
        Array of grid points at bin centers.
    """
    return bin_edges[:-1] + np.diff(bin_edges) / 2


def calc_density(n, v, mol_mass: float):
    """
    calculate density (g/cm^3) from the number of particles

    Parameters
    ----------
    n : int or array
        number of particles
    v : float or array
        volume
    mol_mass : float
        mole mass in g/mol
    """
    rho = (n / constants.Avogadro * mol_mass) / (
        v * (constants.angstrom / constants.centi) ** 3
    )
    return rho


def calc_water_density(n, v):
    """
    Calculate the density of water from the number of particles and volume.

    Parameters
    ----------
    n : int or array
        number of particles
    v : float or array
        volume

    Returns
    -------
    float or array
        Density of water in g/cm^3
    """
    return calc_density(n, v, 18.015)


def identify_water_molecules(
    h_positions: np.ndarray,
    o_positions: np.ndarray,
    box: np.ndarray,
    oh_cutoff: float,
    ignore_warnings: bool = False,
) -> Dict[int, List[int]]:
    """
    Identify water molecules based on proximity of hydrogen and oxygen atoms.

    Parameters
    ----------
    h_positions : np.ndarray
        Positions of hydrogen atoms.
    o_positions : np.ndarray
        Positions of oxygen atoms.
    box : np.ndarray
        Simulation cell defining periodic boundaries.
    oh_cutoff : float
        Maximum O-H distance to consider as a bond.
    ignore_warnings : bool
        If True, ignore warnings about non-water species

    Returns
    -------
    Dict[int, List[int]]
        Dictionary mapping oxygen atom indices to lists of two bonded hydrogen atom indices.
    """
    water_dict = {}

    all_distances = np.zeros((o_positions.shape[0], h_positions.shape[0]))
    distance_array(o_positions, h_positions, result=all_distances, box=box)
    saved_h_ids = []
    for ii, ds in enumerate(all_distances):
        mask = ds < oh_cutoff
        if np.sum(mask) != 2:
            if not ignore_warnings:
                raise Warning(
                    f"Oxygen atom {ii} has {np.sum(mask)} hydrogen atoms within {oh_cutoff} Å."
                )
            continue
        water_dict[ii] = np.where(mask)[0].tolist()
        saved_h_ids.append(water_dict[ii])
    saved_h_ids = np.concatenate(saved_h_ids)
    assert np.unique(saved_h_ids).shape[0] == saved_h_ids.shape[0]
    return water_dict


def mic_1d(x: np.ndarray, box_length: float, ref: float = 0.0) -> np.ndarray:
    """
    Apply the minimum image convention to a 1D coordinate in a periodic cell.

    Parameters
    ----------
    x : np.ndarray
        Coordinates to be wrapped.
    box_length : float
        Length of the periodic cell.
    ref : float
        Reference coordinate around which the coordinates are wrapped.

    Returns
    -------
    np.ndarray
        Wrapped coordinates within the first principal cell centered around
        the reference coordinate.
    """
    _x = x - ref
    _x = _x - np.round(_x / box_length) * box_length
    return _x + ref


def exponential_moving_average(data, alpha=0.1):
    """Exponential moving average"""
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema
