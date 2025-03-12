# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import MDAnalysis as mda
import numpy as np
from ase import io
from MDAnalysis import transformations as trans
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import (
    HydrogenBondAnalysis as RefHydrogenBondAnalysis,
)

from WatAnalysis.workflow import HydrogenBondAnalysis, PlanarInterfaceAnalysisBase


class TestHydrogenBondAnalysis(unittest.TestCase):
    def setUp(self):
        topo = "../data/pt_100_OH/coord.xyz"
        traj = "../data/pt_100_OH/traj.xyz"

        u = mda.Universe(topo, traj, topology_format="XYZ", format="XYZ")
        atoms = io.read(topo)
        transform = trans.boxdimensions.set_dimensions(atoms.cell.cellpar())
        u.trajectory.add_transformations(transform)

        metal_ids = np.where(atoms.symbols == "Pt")[0]
        surf_ids = [metal_ids[:16], metal_ids[-16:]]

        self.test_task = PlanarInterfaceAnalysisBase(
            universe=u,
            surf_ids=surf_ids,
            verbose=False,
            workflow=[
                HydrogenBondAnalysis(
                    HB_cutoff={"DA": 3.5, "DHA": 150.0},
                ),
            ],
        )
        self.ref_task = RefHydrogenBondAnalysis(
            universe=u,
            donors_sel="name O",
            hydrogens_sel="name H",
            acceptors_sel="name O",
            d_h_cutoff=1.3,
            d_a_cutoff=3.5,
        )

        self.test_task.run()
        self.ref_task.run()

    def test_consisitency(self):
        # number of HBs
        np.testing.assert_equal(
            self.test_task.workflow[0].results.hbonds.shape[0],
            self.ref_task.hbonds.shape[0],
        )
        # frame index
        np.testing.assert_allclose(
            self.ref_task.hbonds[:, 0], self.test_task.workflow[0].results.hbonds[:, 0]
        )
        # (sorted) D-A distance
        np.testing.assert_allclose(
            np.sort(self.ref_task.hbonds[:, 4]),
            np.sort(self.test_task.workflow[0].results.hbonds[:, 3]),
        )
        # (sorted) D-H-A angle
        np.testing.assert_allclose(
            np.sort(self.ref_task.hbonds[:, 5]),
            np.sort(self.test_task.workflow[0].results.hbonds[:, 4]),
        )
