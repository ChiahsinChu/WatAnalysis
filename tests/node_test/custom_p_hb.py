# SPDX-License-Identifier: LGPL-3.0-or-later
import time

# modules for parallel computing
import dask
import dask.multiprocessing

# modules for analysis
import MDAnalysis as mda
from MDAnalysis import transformations as trans
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis

dask.config.set(scheduler="processes")

# n_jobs = cpu_count()
n_jobs = 4


@dask.delayed
def run_block(blockslice, ana_base):
    """
    ana_base: object of AnalysisBase Class/Child Class
    """
    # universe.transfer_to_memory(start=blockslice.start, stop=blockslice.stop)
    ana_base.run(start=blockslice.start, stop=blockslice.stop, verbose=True)
    return ana_base


def main():
    # load trajectory as Universe
    ## dt in ps
    u = mda.Universe(
        "../input_data/interface.psf", "../input_data/trajectory.xyz", dt=0.025
    )
    dim = [16.869, 16.869, 41.478, 90, 90, 120]
    transform = trans.boxdimensions.set_dimensions(dim)
    u.trajectory.add_transformations(transform)
    # u.transfer_to_memory()

    # init analysis method class
    hbonds = HydrogenBondAnalysis(
        universe=u,
        donors_sel=None,
        hydrogens_sel="name H",
        acceptors_sel="name O",
        d_a_cutoff=3.0,
        d_h_a_angle_cutoff=150,
        update_selections=False,
    )

    # generate block
    n_frames = u.trajectory.n_frames
    n_blocks = n_jobs  #  it can be any realistic value (0 < n_blocks <= n_jobs)
    n_frames_per_block = n_frames // n_blocks
    blocks = [
        range(i * n_frames_per_block, (i + 1) * n_frames_per_block)
        for i in range(n_blocks - 1)
    ]
    blocks.append(range((n_blocks - 1) * n_frames_per_block, n_frames))
    print(n_jobs)
    print(blocks)

    # generate jobs
    jobs = []
    for bs in blocks:
        jobs.append(run_block(bs, hbonds))
    jobs = dask.delayed(jobs)

    # compute
    results = jobs.compute()
    return results


if __name__ == "__main__":
    start = time.process_time()
    results = main()
    for tmp_ana_base in results:
        print(tmp_ana_base.results.hbonds.shape)
        print(tmp_ana_base.results.hbonds[0])
    fmt = "\nWork Completed! Used Time: {:.3f} seconds"
    print(fmt.format(time.process_time() - start))
