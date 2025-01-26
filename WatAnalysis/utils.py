# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import List, Dict
import numpy as np
from scipy import constants
from MDAnalysis.lib.distances import minimize_vectors, capped_distance


def get_cum_ave(data):
    cum_sum = data.cumsum()
    cum_ave = cum_sum / (np.arange(len(data)) + 1)
    return cum_ave


def density(n, v, mol_mass: float):
    """
    Calculate density in g/cm³ from the number of particles.

    Parameters
    ----------
    n : int or array-like
        Number of particles.
    v : float or array-like
        Volume in Å³.
    mol_mass : float
        Molar mass in g/mol.

    Returns
    -------
    float or array-like
        Density in g/cm³.
    """
    rho = (n / constants.Avogadro * mol_mass) / (
        v * (constants.angstrom / constants.centi) ** 3
    )
    return rho


def water_density(n, v):
    """
    Calculate the density of water in g/cm³.

    Parameters
    ----------
    n : int or array-like
        Number of water molecules.
    v : float or array-like
        Volume in Å³.

    Returns
    -------
    float or array-like
        Density in g/cm³.
    """
    return density(n, v, 18.015)


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


def identify_water_molecules(
    h_positions: np.ndarray,
    o_positions: np.ndarray,
    box: np.ndarray,
    oh_cutoff: float,
) -> Dict[int, List[int]]:
    """
    Identify water molecules based on proximity of hydrogen and oxygen atoms.

    Parameters
    ----------
    h_positions : np.ndarray
        Positions of hydrogen atoms.
    o_positions : np.ndarray
        Positions of oxygen atoms.
    box: np.ndarray
        Simulation cell defining periodic boundaries.
    oh_cutoff : float
        Maximum O-H distance to consider as a bond.

    Returns
    -------
    Dict[int, List[int]]
        Dictionary mapping oxygen atom indices to lists of two bonded hydrogen atom indices.
    """
    water_dict = {i: [] for i in range(o_positions.shape[0])}

    for h_idx, hpos in enumerate(h_positions):
        pairs, distances = capped_distance(
            hpos,
            o_positions,
            max_cutoff=oh_cutoff,
            box=box,
            return_distances=True,
        )

        if len(pairs) > 0:
            closest_o_idx = pairs[np.argmin(distances)][1]
            water_dict[closest_o_idx].append(h_idx)

    water_dict = {key: value for key, value in water_dict.items() if len(value) == 2}
    return water_dict


def get_dipoles(
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
    _temp = x - ref
    _temp = _temp - np.round(_temp / box_length) * box_length
    return _temp + ref
