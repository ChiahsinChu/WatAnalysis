from MDAnalysis.analysis.waterdynamics import WaterOrientationalRelaxation
from MDAnalysis.analysis.waterdynamics import AngularDistribution
from MDAnalysis.analysis.waterdynamics import MeanSquareDisplacement
from MDAnalysis.analysis.waterdynamics import SurvivalProbability

from MDAnalysis.lib.log import ProgressBar

from WatAnalysis.preprocess import make_selection


class WOR(WaterOrientationalRelaxation):

    def __init__(self, universe, t0, tf, dtmax, nproc=1, **kwargs):
        """
        sel_region, surf_ids, c_ag, select_all, bonded
        """
        select = make_selection(**kwargs)
        print("selection: ", select)
        super().__init__(universe, select, t0, tf, dtmax, nproc)


class AD(AngularDistribution):

    def __init__(self,
                 universe,
                 bins=40,
                 nproc=1,
                 axis="z",
                 updating=True,
                 **kwargs):
        select = make_selection(**kwargs)
        print("selection: ", select)
        super().__init__(universe, select, bins, nproc, axis)
        # TODO: check if updating works
        self.updating = updating

    def _selection_serial(self, universe, selection_str):
        selection = []
        for ts in ProgressBar(universe.trajectory,
                              verbose=True,
                              total=universe.trajectory.n_frames):
            selection.append(
                universe.select_atoms(selection_str, updating=self.updating))
        return selection


class MSD(MeanSquareDisplacement):

    def __init__(self, universe, t0, tf, dtmax, nproc=1, **kwargs):
        select = make_selection(**kwargs)
        print("selection: ", select)
        super().__init__(universe, select, t0, tf, dtmax, nproc)


class SP(SurvivalProbability):

    def __init__(self, universe, select, verbose=False, **kwargs):
        select = make_selection(**kwargs)
        print("selection: ", select)
        super().__init__(universe, select, verbose)