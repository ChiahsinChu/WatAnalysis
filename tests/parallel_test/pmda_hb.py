# SPDX-License-Identifier: LGPL-3.0-or-later
import time

import MDAnalysis as mda
from MDAnalysis import transformations as trans
from pmda.hbond_analysis import HydrogenBondAnalysis as HBA


def main():
    # load Universe
    # dt in ps
    u = mda.Universe(
        "../input_data/interface.psf", "../input_data/trajectory.xyz", dt=0.025
    )
    dim = [16.869, 16.869, 41.478, 90, 90, 120]
    transform = trans.boxdimensions.set_dimensions(dim)
    u.trajectory.add_transformations(transform)
    u.transfer_to_memory()

    hbonds = HBA(
        universe=u,
        donors_sel=None,
        hydrogens_sel="name H",
        acceptors_sel="name O",
        d_a_cutoff=3.0,
        d_h_a_angle_cutoff=150,
        update_selections=False,
    )
    hbonds.run()
    return hbonds


if __name__ == "__main__":
    start = time.process_time()
    hbonds = main()
    print(hbonds.hbonds)
    fmt = "\nWork Completed! Used Time: {:.3f} seconds"
    print(fmt.format(time.process_time() - start))
