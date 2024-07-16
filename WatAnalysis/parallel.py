# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
import time
from multiprocessing import cpu_count

# modules for parallel computing
import dask
import dask.multiprocessing

logger = logging.getLogger(__name__)

dask.config.set(scheduler="processes")
n_jobs = cpu_count()


def parallel_run(universe, ana_base):
    """
    based on MDA user guide
    """
    tot_start = time.time()
    print("TASK START at ", tot_start)
    print(tot_start)

    # generate block
    start = time.time()
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
    print(time.time())
    fmt = "\nUsed Time: {:.3f} seconds"
    print(fmt.format(time.time() - start))

    # compute
    start = time.time()
    print("START: Compute")
    print(start)

    jobs = []
    for bs in blocks:
        jobs.append(run_block(bs, ana_base))
    jobs = dask.delayed(jobs)
    results = jobs.compute()

    print("END: Compute")
    print(time.time())
    fmt = "\nUsed Time: {:.3f} seconds"
    print(fmt.format(time.time() - start))

    print("TASK END at ", time.time())
    fmt = "\nTotal Used Time: {:.3f} seconds"
    print(fmt.format(time.time() - tot_start))

    return results


@dask.delayed
def run_block(blockslice, ana_base):
    """
    Args:
        blockslice: TBC
        ana_base: MDA AnalysisBase Class/Child Class
    Return:
        ana_base with results of block
    """
    # universe.transfer_to_memory(start=blockslice.start, stop=blockslice.stop)
    ana_base.run(start=blockslice.start, stop=blockslice.stop, verbose=True)
    return ana_base


# >>>>>>>>>>>>>>> Ke Xiong Code >>>>>>>>>>>>>>> #
def parallel_exec(singlefunc, start, stop, step, n_proc, *args, **kwargs):
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    region = range(start, stop, step)
    blocks = slice_split(region, n_proc)

    if singlefunc.__class__.__name__ == "function":
        run_para = partial(
            _parallel_function_formap, singlefunc=singlefunc, *args, **kwargs
        )

    elif singlefunc.__class__.__name__ == "method":
        # type of method we should tell
        if singlefunc.__self__.__class__.__base__.__base__.__name__ == "AnalysisBase":
            # more variables should be added.

            # it seems not work
            singlefunc.__self__.para = True
            singlefunc.__self__._para_region = region

        run_para = partial(
            _parallel_function_formap, singlefunc=singlefunc, *args, **kwargs
        )

    # co future
    with ProcessPoolExecutor(n_proc) as pool_executor:
        raw_result = list(pool_executor.map(run_para, blocks))
    raw_result.append([start, stop, step])

    # raw_result process
    result = para_raw_data_process(singlefunc, raw_result)
    if not result:
        print("no result process match, raw result will be collected.")
        result = raw_result

    try:
        if singlefunc.__self__.__class__.__base__.__base__.__name__ == "AnalysisBase":
            singlefunc.__self__.para = None
    except:
        pass
    # singlefunc.__self__.ag.universe.trajectory._reopen()

    return result


def slice_split(src, num):
    len_src = len(src)
    div = len_src // num
    mod = len_src % num
    sdx = 0
    splist = []
    for p_len in [1] * mod + [0] * (num - mod):
        edx = p_len + div + sdx
        splist.append(src[sdx:edx])
        sdx = edx
    return splist


def _parallel_function_formap(block, singlefunc, *args, **kwargs):
    start = block.start
    stop = block.stop
    step = block.step

    return singlefunc(start=start, stop=stop, step=step, *args, **kwargs)


def para_raw_data_process(func, rawdata, *args):
    result = []
    if func.__class__.__name__ == "method":
        if func.__name__ == "trans2ase":
            result = []
            for item in rawdata:
                result += item
        elif func.__name__ == "get_memory_slice":
            from MDAnalysis.coordinates.memory import MemoryReader

            coords = []
            for item in rawdata:
                single_coords = item.get_array()
                coords.append(single_coords)
            coords = np.concatenate(coords)
            memory_slice = MemoryReader(coords)
            result = memory_slice
        elif func.__name__ == "get_distance":
            result = np.concatenate(rawdata)
        ##suit for analysis method.
        elif func.__self__.__class__.__base__.__base__.__name__ == "AnalysisBase":
            result = func.__self__._parallel_conclude(rawdata)
        elif func.__self__.__class__.__name__ == "OffsetGenerator":
            result = rawdata

    elif func.__class__.__name__ == "function":
        if func.__name__ == "offset_para_func":
            import numpy as np

            offsets = np.concatenate(rawdata)
            offsets = np.unique(offsets)
            n_frames = len(offsets)
            result = (n_frames, offsets)
        if func.__name__ == "trans2ase":
            result = []
            for item in rawdata:
                result += item
    return result


# <<<<<<<<<<<<<<< Ke Xiong Code <<<<<<<<<<<<<<< #
