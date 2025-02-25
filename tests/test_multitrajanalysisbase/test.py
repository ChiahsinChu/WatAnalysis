# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import List

import numpy as np
from ase import io
from MDAnalysis import Universe

from WatAnalysis.multitrajbase import MultiTrajsAnalysisBase


class ToyMTClass(MultiTrajsAnalysisBase):
    def __init__(self, universes: List[Universe], verbose=False, **kwargs):
        super().__init__([u.trajectory for u in universes], verbose, **kwargs)
        self.universes = universes

    def _prepare(self):
        self.n_atoms = len(self.universes[0].atoms)
        self.all_coordinates = [
            np.zeros([self.n_frames, self.n_atoms, 3])
            for ii in range(len(self._trajectories))
        ]

    def _single_frame(self):
        for ii, ts in enumerate(self._all_ts):
            np.copyto(self.all_coordinates[ii][self._frame_index], ts.positions)

    def _conclude(self):
        for ii, coordinates in enumerate(self.all_coordinates):
            np.save("coordinates.%03d.npy" % ii, coordinates)


class TestMTClass(unittest.TestCase):
    def setUp(self) -> None:
        self.topo = "coord.xyz"
        self.pos_traj = "pos_traj.xyz"
        self.vel_traj = "vel_traj.xyz"

        u_pos = Universe(self.topo, self.pos_traj, topology_format="XYZ", format="XYZ")
        u_vel = Universe(self.topo, self.vel_traj, topology_format="XYZ", format="XYZ")

        self.obj = ToyMTClass(universes=[u_pos, u_vel], verbose=False)
        self.obj.run()

    def test(self):
        test_coordinates = np.load("coordinates.000.npy")
        for atoms, test_coordinate in zip(io.iread(self.pos_traj), test_coordinates):
            ref_coordinate = atoms.get_positions()
            diff = test_coordinate - ref_coordinate
            self.assertTrue(np.max(np.abs(diff)) < 1e-8)

        test_coordinates = np.load("coordinates.001.npy")
        for atoms, test_coordinate in zip(io.iread(self.vel_traj), test_coordinates):
            ref_coordinate = atoms.get_positions()
            diff = test_coordinate - ref_coordinate
            self.assertTrue(np.max(np.abs(diff)) < 1e-8)


if __name__ == "__main__":
    unittest.main()
