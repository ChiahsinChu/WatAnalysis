from multiprocessing.pool import Pool
from functools import partial

from dataclasses import dataclass
from typing import List, Tuple, NamedTuple, Union

import json
import os
import io

import MDAnalysis as mda
from MDAnalysis.lib.distances import minimize_vectors

import numpy as np

from ai2_kit.algorithm.proton_transfer import AlgorithmParameter


class AnalysisResult(NamedTuple):
    indicator_position: Tuple[float, float, float]
    transfers: List[Tuple[int, int, int]]


@dataclass
class DepletionSystemInfo:  # Information about the system
    initial_acceptor: int
    u: mda.Universe
    donor_elements: List[str]


class DepletionSystem(object):
    def __init__(self, sys_info: DepletionSystemInfo, parameter: AlgorithmParameter):
        self.u = sys_info.u
        self.cell = sys_info.u.dimensions
        self.donor_elements = sys_info.donor_elements
        self.r_a = parameter.r_a
        self.r_h = parameter.r_h
        self.g_threshold = parameter.g_threshold
        self.max_depth = parameter.max_depth
        self.rho_0 = parameter.rho_0
        self.rho_max = parameter.rho_max

    def frame_analysis(
        self,
        prev_acceptor: int,
        donor_query: str,
        time: int,
    ):
        self.u.trajectory[time]
        depleted_site = prev_acceptor
        transfers = []
        list_of_paths = [[prev_acceptor]]  # Start from the depleted site
        list_of_weights = [[1]]
        for depth in range(self.max_depth):
            for j, path in enumerate(list_of_paths):
                found = False
                if depth == len(path) - 1:
                    # Find potential DONORS near the depleted acceptor
                    donors = self.u.select_atoms(
                        f"(around {self.r_a} index {path[-1]}) and ({donor_query})"
                    )
                    for donor in donors.ix:
                        # Check if protons are ABSENT around the donor
                        protons = self.u.select_atoms(
                            f"(around {self.r_h} index {donor}) and (name H)"
                        )
                        g, proton = self.calculate_g(donor, path[-1], protons.ix)
                        if (g >= self.g_threshold) and (donor not in path):
                            found = True
                            list_of_weights.append(
                                list_of_weights[j] + [g * list_of_weights[j][-1]]
                            )
                            list_of_paths.append(path + [donor])
                            if proton > 0 and all(w >= 0.9 for w in list_of_weights[j]):
                                depleted_site = donor
                                transfers.append((int(donor), int(proton), int(depth)))
                if found:
                    list_of_paths.pop(j)
                    list_of_weights.pop(j)
        indicator_position = self.calculate_position(list_of_paths, list_of_weights)
        result = AnalysisResult(
            indicator_position=tuple(indicator_position[0]), transfers=transfers
        )
        return depleted_site, result

    def calculate_g(self, donor: int, acceptor: int, protons: list):
        donor_pos = self.u.atoms[donor].position
        acceptor_pos = self.u.atoms[acceptor].position
        g_value = 0
        proton_index = -1
        for i, proton in enumerate(protons):
            proton_pos = self.u.atoms[proton].position
            r_da = minimize_vectors(acceptor_pos - donor_pos, self.cell)
            r_dh = minimize_vectors(proton_pos - donor_pos, self.cell)
            z1 = np.dot(r_dh, r_da)
            z2 = np.dot(r_da, r_da)
            z = z1 / z2
            p = (self.rho_max - z) / (self.rho_max - self.rho_0)
            if p >= 1:
                g = 0
            elif p <= 0:
                g = 1
                proton_index = protons[i]
            else:
                g = -6 * (p**5) + 15 * (p**4) - 10 * (p**3) + 1
            g_value = g_value + g
        return g_value, proton_index

    def calculate_position(self, paths: list, weights: list):
        positions_all = []
        nodes_all = []
        weights_all = []
        for i, path in enumerate(paths):
            for j, node in enumerate(path):
                if node not in nodes_all:
                    acceptor_pos = self.u.atoms[path[0]].position
                    if j == 0:
                        positions_all.append(acceptor_pos)
                    else:
                        donor_pos = self.u.atoms[node].position
                        min_vector = minimize_vectors(
                            donor_pos - acceptor_pos, self.cell
                        )
                        real_donor_pos = min_vector + acceptor_pos
                        positions_all.append(real_donor_pos)
                    nodes_all.append(node)
                    weights_all.append(weights[i][j])
                else:
                    index = nodes_all.index(node)
                    weights_all[index] = max(weights[i][j], weights_all[index])
        p = np.array(positions_all).reshape(-1, 3)
        w = np.array(weights_all).reshape(1, -1)
        z = w @ p
        pos_ind = z / w.sum()
        return pos_ind

    def analysis(self, initial_depleted_site: int, out_dir: str):
        depleted_site = initial_depleted_site
        donor_query = " or ".join([f"(name {el})" for el in self.donor_elements])
        rand_file = io.FileIO(
            os.path.join(out_dir, f"{initial_depleted_site}.jsonl"), "w"
        )
        writer = io.BufferedWriter(rand_file)
        line = (tuple(self.u.atoms[initial_depleted_site].position.astype(float)), [])
        writer.write((json.dumps(line) + "\n").encode("utf-8"))
        for time in range(self.u.trajectory.n_frames - 1):
            depleted_site, result = self.frame_analysis(
                depleted_site, donor_query, time + 1
            )
            line = (result.indicator_position, result.transfers)
            writer.write((json.dumps(line) + "\n").encode("utf-8"))
        writer.flush()


def proton_depletion_detection(
    universe: mda.Universe,
    out_dir: str,
    donor_elements: List[str],
    initial_acceptor: Union[List[int], np.ndarray],
    core_num: int = 4,
    r_a: float = 3.5,
    r_h: float = 1.3,
    rho_0: float = 1 / 2.2,
    rho_max: float = 0.5,
    max_depth: int = 3,
    g_threshold: float = 0.0001,
):
    os.makedirs(out_dir, exist_ok=True)

    sys_info = DepletionSystemInfo(
        initial_acceptor=-1, u=universe, donor_elements=donor_elements
    )

    parameter = AlgorithmParameter(
        r_a=r_a,
        r_h=r_h,
        rho_0=rho_0,
        rho_max=rho_max,
        max_depth=max_depth,
        g_threshold=g_threshold,
    )

    system = DepletionSystem(
        sys_info,
        parameter,
    )

    with Pool(processes=core_num) as pool:
        pool.map(partial(system.analysis, out_dir=out_dir), initial_acceptor)


def calculate_g(
    position_donor: np.ndarray,
    position_acceptor: np.ndarray,
    position_protons: np.ndarray,
    box: np.ndarray,
    rho_0: float,
    rho_max: float,
):
    g_value = 0
    proton_index = -1
    for ii, position_proton in enumerate(position_protons):
        r_da = minimize_vectors(position_acceptor - position_donor, box=box)
        r_dh = minimize_vectors(position_proton - position_donor, box=box)
        z1 = np.dot(r_dh, r_da)
        z2 = np.dot(r_da, r_da)
        z = z1 / z2
        p = (rho_max - z) / (rho_max - rho_0)
        if p >= 1:
            g = 0
        elif p <= 0:
            g = 1
            proton_index = ii
        else:
            g = -6 * (p**5) + 15 * (p**4) - 10 * (p**3) + 1
        g_value = g_value + g
    return g_value, proton_index
