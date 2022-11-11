import numpy as np

from MDAnalysis.analysis.waterdynamics import WaterOrientationalRelaxation
from MDAnalysis.analysis.waterdynamics import AngularDistribution
from MDAnalysis.analysis.waterdynamics import MeanSquareDisplacement
from MDAnalysis.analysis.waterdynamics import SurvivalProbability

from MDAnalysis.lib.log import ProgressBar

from WatAnalysis.preprocess import make_selection, make_selection_two


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
        select = make_selection_two(**kwargs)
        # print("selection: ", select)
        super().__init__(universe, select, bins, nproc, axis)
        # TODO: check if updating works
        self.updating = updating

    def _getHistogram(self, universe, selection, bins, axis):
        """
        This function gets a normalized histogram of the cos(theta) values. It
        return a list of list.
        """
        a_lo = self._getCosTheta(universe, selection[::2], axis)
        a_hi = self._getCosTheta(universe, selection[1::2], axis)
        # print(np.shape(a_lo))
        # print(np.shape(a_hi))

        cosThetaOH = np.concatenate([np.array(a_lo[0]), -np.array(a_hi[0])])
        cosThetaHH = np.concatenate([np.array(a_lo[1]), -np.array(a_hi[1])])
        cosThetadip = np.concatenate([np.array(a_lo[2]), -np.array(a_hi[2])])
        ThetaOH = np.arccos(cosThetaOH) / np.pi * 180
        ThetaHH = np.arccos(cosThetaHH) / np.pi * 180
        Thetadip = np.arccos(cosThetadip) / np.pi * 180
        histInterval = bins
        histcosThetaOH = np.histogram(cosThetaOH, histInterval, density=True)
        histcosThetaHH = np.histogram(cosThetaHH, histInterval, density=True)
        histcosThetadip = np.histogram(cosThetadip, histInterval, density=True)
        histThetaOH = np.histogram(ThetaOH, histInterval, density=True)
        histThetaHH = np.histogram(ThetaHH, histInterval, density=True)
        histThetadip = np.histogram(Thetadip, histInterval, density=True)

        return (histcosThetaOH, histcosThetaHH, histcosThetadip, histThetaOH,
                histThetaHH, histThetadip)

    def run(self, **kwargs):
        """Function to evaluate the angular distribution of cos(theta)"""

        selection = self._selection_serial(self.universe, self.selection_str)

        self.graph = []
        output = self._getHistogram(self.universe, selection, self.bins,
                                    self.axis)
        # this is to format the exit of the file
        # maybe this output could be improved
        listcosOH = [list(output[0][1]), list(output[0][0])]
        listcosHH = [list(output[1][1]), list(output[1][0])]
        listcosdip = [list(output[2][1]), list(output[2][0])]
        listOH = [list(output[3][1]), list(output[3][0])]
        listHH = [list(output[4][1]), list(output[4][0])]
        listdip = [list(output[5][1]), list(output[5][0])]

        self.graph.append(self._hist2column(listcosOH))
        self.graph.append(self._hist2column(listcosHH))
        self.graph.append(self._hist2column(listcosdip))
        self.graph.append(self._hist2column(listOH))
        self.graph.append(self._hist2column(listHH))
        self.graph.append(self._hist2column(listdip))

    def _selection_serial(self, universe, l_selection_str):
        selection = []
        for ts in ProgressBar(universe.trajectory,
                              verbose=True,
                              total=universe.trajectory.n_frames):
            selection.append(
                universe.select_atoms(l_selection_str[0],
                                      updating=self.updating))
            selection.append(
                universe.select_atoms(l_selection_str[1],
                                      updating=self.updating))
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