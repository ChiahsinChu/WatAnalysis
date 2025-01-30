# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import List, Dict
import numpy as np
from MDAnalysis.lib.distances import capped_distance


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
