# SPDX-License-Identifier: LGPL-3.0-or-later
import time

import MDAnalysis as mda

# from pmda.hbond_analysis import HydrogenBondAnalysis as HBA
from cresc.analysis import HBondAnalysis
from cresc.parallel import parallel_exec
from MDAnalysis import transformations as trans


def main():
    # load Universe
    # dt in ps
    u = mda.Universe(
        "../input_data/interface.psf", "../input_data/trajectory.xyz", dt=0.025
    )
    dim = [16.869, 16.869, 41.478, 90, 90, 120]
    transform = trans.boxdimensions.set_dimensions(dim)
    u.trajectory.add_transformations(transform)

    hba_para = HBondAnalysis(
        universe=u,
        donors_sel=None,
        hydrogens_sel="name H",
        acceptors_sel="name O",
        d_a_cutoff=3.0,
        d_h_a_angle_cutoff=150,
        update_selections=False,
    )

    parallel_exec(hba_para.run, 0, 10000, 1, 4)

    hbonds = hba_para.results["hbonds"]
    return hbonds


if __name__ == "__main__":
    start = time.process_time()
    hbonds = main()
    print(hbonds)
    fmt = "\nWork Completed! Used Time: {:.3f} seconds"
    print(fmt.format(time.process_time() - start))
