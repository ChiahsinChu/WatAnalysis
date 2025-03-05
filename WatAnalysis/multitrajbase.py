# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import List

import numpy as np
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.coordinates.base import ReaderBase
from MDAnalysis.lib.log import ProgressBar

logger = logging.getLogger(__name__)


# todo: add check for all objects in trajectories (at least make sure they have identical length)
class MultiTrajsAnalysisBase(AnalysisBase):
    def __init__(self, trajectories: List[ReaderBase], verbose=False, **kwargs):
        super().__init__(trajectories[0], verbose=verbose, **kwargs)
        self._trajectories = trajectories

        self.start = None
        self.stop = None
        self.step = None

        self.n_frames = None
        self.frames = None
        self.times = None

        self._sliced_trajectories = None
        self._frame_index = None
        self._all_ts = None

    def _setup_frames(
        self,
        trajectories,
        start=None,
        stop=None,
        step=None,
        frames=None,
    ):
        self._trajectories = trajectories
        if frames is not None:
            if not all(opt is None for opt in [start, stop, step]):
                raise ValueError("start/stop/step cannot be combined with " "frames")
            slicer = frames
        else:
            start, stop, step = self._trajectories[0].check_slice_indices(
                start, stop, step
            )
            slicer = slice(start, stop, step)
        self.start = start
        self.stop = stop
        self.step = step

        self._sliced_trajectories = [trajectory[slicer] for trajectory in trajectories]

        self.n_frames = len(self._sliced_trajectories[0])
        self.frames = np.zeros(self.n_frames, dtype=int)
        self.times = np.zeros(self.n_frames)

    def run(
        self,
        start=None,
        stop=None,
        step=None,
        frames=None,
        verbose=None,
        *,
        progressbar_kwargs={},
    ):
        logger.info("Choosing frames to analyze")
        # if verbose unchanged, use class default
        verbose = getattr(self, "_verbose", False) if verbose is None else verbose

        self._setup_frames(
            self._trajectories,
            start=start,
            stop=stop,
            step=step,
            frames=frames,
        )
        logger.info("Starting preparation")
        self._prepare()
        logger.info("Starting analysis loop over %d trajectory frames", self.n_frames)

        iter_objects = [
            ProgressBar(_sliced_trajectory, verbose=verbose, **progressbar_kwargs)
            for _sliced_trajectory in self._sliced_trajectories
        ]

        for i, all_ts in enumerate(zip(*iter_objects)):
            self._frame_index = i
            self._all_ts = all_ts
            self.frames[i] = all_ts[0].frame
            self.times[i] = all_ts[0].time
            self._single_frame()

        logger.info("Finishing up")
        self._conclude()
        return self
