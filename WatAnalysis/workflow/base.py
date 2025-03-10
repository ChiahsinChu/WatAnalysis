# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import numpy as np
from ase.cell import Cell
from MDAnalysis.analysis.base import AnalysisBase, Results
from MDAnalysis.core.universe import Universe

from WatAnalysis import utils


class PlanarInterfaceAnalysisBase(AnalysisBase):
    """
    Analysis class for studying the structure and properties of water molecules
    in a given molecular dynamics simulation.

    Parameters
    ----------
    universe : Universe
        The MDAnalysis Universe object containing the simulation data.
    surf_ids : Union[List, np.ndarray], optional
        List or array of surface atom indices of the form [surf_1, surf_2]
        where surf_1 and surf_2 are arrays containing the indices corresponding
        to the left surface and right surface, respectively.
    axis : int
        The axis perpendicular to the surfaces (default is 2).
        x=0, y=1, z=2.
    workflow : List[SingleAnalysis]
        List of analysis tasks to be performed in order.
    **kwargs : dict
        Additional keyword arguments for customization:
        - verbose : bool, optional.
            If True, enables verbose output.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from MDAnalysis import Universe
    >>> from WatAnalysis.workflow import PlanarInterfaceAnalysisBase, DensityAnalysis
    >>>
    >>> # set up universe and surface indices...
    >>>
    >>> obj = PlanarInterfaceAnalysisBase(
    ...     universe=universe,
    ...     surf_ids=surf_ids,
    ...     workflow=[
    ...         DensityAnalysis(selection="name O", mol_mass=15.999, label="oxygen"),
    ...         DensityAnalysis(selection="name H", mol_mass=1.008, label="hydrogen"),
    ...     ],
    ... )
    >>> obj.run()
    >>>
    >>> plt.plot(obj.workflow[0].results.bins, obj.workflow[0].results.density)
    >>> plt.show()
    """

    def __init__(
        self,
        universe: Universe,
        surf_ids: Union[List, np.ndarray],
        axis: int = 2,
        workflow: list = [],
        **kwargs,
    ):
        self.universe = universe
        trajectory = self.universe.trajectory
        super().__init__(trajectory, verbose=kwargs.get("verbose", False))
        self.n_frames = len(trajectory)
        self.surf_ids = surf_ids
        self.axis = axis
        self.workflow = workflow

        self.cross_area = None
        self.r_surf_lo = None
        self.r_surf_hi = None
        self.r_surf_ref = None

        self.data_requirements: Dict[str, DataRequirement] = {}
        for task in self.workflow:
            self.data_requirements.update(task.data_requirements)

    def _prepare(self):
        # Initialize empty arrays
        self.cross_area = 0.0
        self.r_surf_lo = np.zeros(self.n_frames)
        self.r_surf_hi = np.zeros(self.n_frames)

        for k, v in self.data_requirements.items():
            if v.atomic:
                ag = self.universe.select_atoms(v.selection)
                a = np.full((self.n_frames, ag.n_atoms, v.dim), v.value)
            else:
                a = np.full((self.n_frames, v.dim), v.value)
            setattr(self, k, a)

        for task in self.workflow:
            task._prepare(self)

    def _single_frame(self):
        ts_box = self._ts.dimensions
        ts_area = Cell.new(ts_box).area(self.axis)
        self.cross_area += ts_area

        coords = self._ts.positions

        # Absolute surface positions
        r_surf_lo = coords[self.surf_ids[0], self.axis]
        r_surf_hi = coords[self.surf_ids[1], self.axis]
        box_length = ts_box[self.axis]
        # Use MIC in case part of the surface crosses the cell boundaries
        self.r_surf_lo[self._frame_index] = utils.mic_1d(
            r_surf_lo, box_length, ref=r_surf_lo[0]
        ).mean()
        self.r_surf_hi[self._frame_index] = utils.mic_1d(
            r_surf_hi, box_length, ref=r_surf_hi[0]
        ).mean()

        # Update flags for the intermediate arrays
        for v in self.data_requirements.values():
            v.set_update_flag(False)

        for task in self.workflow:
            task._single_frame(self)

    def _conclude(self):
        # Average surface area
        self.cross_area /= self.n_frames

        box_length = self.universe.dimensions[self.axis]
        # Step 1: Set r_surf_lo as the reference point (zero)
        r_surf_lo = np.zeros(self.r_surf_lo.shape)
        # Step 2: Calculate z2 positions relative to z1 using minimum image convention
        # By setting ref=box_length/2, all positions are within one cell length positive of z1
        r_surf_hi = utils.mic_1d(
            self.r_surf_hi - self.r_surf_lo,
            box_length=box_length,
            ref=box_length / 2,
        )
        # Update attributes to the final relative coordinates
        self.r_surf_ref = self.r_surf_lo.copy()
        self.r_surf_lo = r_surf_lo
        self.r_surf_hi = r_surf_hi

        for task in self.workflow:
            task._conclude(self)


class SingleAnalysis(ABC):
    def __init__(
        self,
    ) -> None:
        self.results = Results()
        self.data_requirements: Dict[str, DataRequirement] = {}

    @abstractmethod
    def _prepare(self, analyser: PlanarInterfaceAnalysisBase):
        pass

    @abstractmethod
    def _single_frame(self, analyser: PlanarInterfaceAnalysisBase):
        pass

    @abstractmethod
    def _conclude(self, analyser: PlanarInterfaceAnalysisBase):
        pass

    @staticmethod
    def _setattr(analyser: PlanarInterfaceAnalysisBase, k: str, v: Any):
        if not hasattr(analyser, k):
            setattr(analyser, k, v)


class DataRequirement:
    """
    Class for intermediate data requirements for analysis tasks.
    """

    def __init__(
        self,
        var_name: str,
        atomic: bool,
        dim: int,
        value: Any = 0.0,
        selection: str = "all",
    ) -> None:
        self.var_name = var_name
        self.atomic = atomic
        self.dim = dim
        self.value = value
        self.selection = selection

        # flag for updating the intermediate arrays
        self.update_flag = False

    def set_update_flag(self, flag: bool):
        self.update_flag = flag
