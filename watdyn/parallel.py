import time, logging
# modules for parallel computing
import dask
import dask.multiprocessing
from multiprocessing import cpu_count

logger = logging.getLogger(__name__)

dask.config.set(scheduler='processes')
n_jobs = cpu_count()


@dask.delayed
def run_block(blockslice, ana_base):
    """
    Args:
        blockslice: TBC
        ana_base: MDA AnalysisBase Class/Child Class
    Return:
        ana_base with results of block
    """
    #universe.transfer_to_memory(start=blockslice.start, stop=blockslice.stop)
    ana_base.run(start=blockslice.start, stop=blockslice.stop, verbose=True)
    return ana_base


def parallel_run(universe, ana_base):
    tot_start = time.process_time()
    print("TASK START at ", tot_start)
    print(tot_start)

    # generate block
    start = time.process_time()
    print("START: Generate block")
    print(start)

    n_frames = universe.trajectory.n_frames
    n_blocks = n_jobs  #  it can be any realistic value (0 < n_blocks <= n_jobs)
    n_frames_per_block = n_frames // n_blocks
    blocks = [
        range(i * n_frames_per_block, (i + 1) * n_frames_per_block)
        for i in range(n_blocks - 1)
    ]
    blocks.append(range((n_blocks - 1) * n_frames_per_block, n_frames))
    print(blocks)

    print("END: Generate block")
    print(time.process_time())
    fmt = "\nUsed Time: {:.3f} seconds"
    print(fmt.format(time.process_time() - start))

    # compute
    start = time.process_time()
    print("START: Compute")
    print(start)

    jobs = []
    for bs in blocks:
        jobs.append(run_block(bs, ana_base))
    jobs = dask.delayed(jobs)
    results = jobs.compute()

    print("END: Compute")
    print(time.process_time())
    fmt = "\nUsed Time: {:.3f} seconds"
    print(fmt.format(time.process_time() - start))

    print("TASK END at ", time.process_time())
    fmt = "\nTotal Used Time: {:.3f} seconds"
    print(fmt.format(time.process_time() - tot_start))

    return results
