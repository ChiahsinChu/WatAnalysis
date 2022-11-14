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

    def __init__(self, universe, t0, tf, dtmax, nproc=1, perp="z", **kwargs):
        select = make_selection(**kwargs)
        print("selection: ", select)
        super().__init__(universe, select, t0, tf, dtmax, nproc)
        # TODO: exception capture
        perp_dict = {'x': 0, 'y': 1, 'z': 2}
        self.perp = perp_dict[perp]

    def _getOneDeltaPoint(self, universe, repInd, i, t0, dt):
        val_para = 0
        val_perp = 0

        n = 0
        for j in range(len(repInd[i]) // 3):
            begj = 3 * j
            universe.trajectory[t0]
            # Plus zero is to avoid 0to be equal to 0tp
            Ot0 = repInd[i][begj].position + 0

            universe.trajectory[t0 + dt]
            # Plus zero is to avoid 0to be equal to 0tp
            Otp = repInd[i][begj].position + 0

            # position oxygen
            OVector = Ot0 - Otp
            # here it is the difference with
            # waterdynamics.WaterOrientationalRelaxation
            val_perp += np.square(OVector[self.perp])
            val_para += (np.dot(OVector, OVector) -
                         np.square(OVector[self.perp]))
            # valO += np.dot(OVector, OVector)
            n += 1

        # if no water molecules remain in selection, there is nothing to get
        # the mean, so n = 0.
        return val_perp / n if n > 0 else 0, val_para / n if n > 0 else 0

    def _getMeanOnePoint(self, universe, selection1, dt,
                         totalFrames):
        """
        This function gets one point of the plot C_vec vs t. It's uses the
        _getOneDeltaPoint() function to calculate the average.

        """
        repInd = self._repeatedIndex(selection1, dt, totalFrames)
        sumsdt = 0
        n = 0.0
        sumDeltaO_perp = 0.0
        sumDeltaO_para = 0.0
        # valOList_perp = []
        # valOList_para = []

        for j in range(totalFrames // dt - 1):
            a_perp, a_para = self._getOneDeltaPoint(universe, repInd, j,
                                                    sumsdt, dt)
            sumDeltaO_perp += a_perp
            sumDeltaO_para += a_para
            # valOList_perp.append(a_perp)
            # valOList_para.append(a_para)
            sumsdt += dt
            n += 1

        # if no water molecules remain in selection, there is nothing to get
        # the mean, so n = 0.
        return sumDeltaO_perp / n if n > 0 else 0, sumDeltaO_para / n if n > 0 else 0

    def run(self, **kwargs):
        """Analyze trajectory and produce timeseries"""

        # All the selection to an array, this way is faster than selecting
        # later.
        if self.nproc == 1:
            selection_out = self._selection_serial(self.universe,
                                                   self.selection)
        else:
            # parallel not yet implemented
            # selection = selection_parallel(universe, selection_str, nproc)
            selection_out = self._selection_serial(self.universe,
                                                   self.selection)
        self.timeseries_perp = []
        self.timeseries_para = []
        for dt in list(range(1, self.dtmax + 1)):
            output_perp, output_para = self._getMeanOnePoint(
                self.universe, selection_out, self.selection, dt, self.tf)
            self.timeseries_perp.append(output_perp)
            self.timeseries_para.append(output_para)
        self.timeseries = np.array(self.timeseries_para) + np.array(
            self.timeseries_perp)


class SP(SurvivalProbability):

    def __init__(self, universe, select, verbose=False, **kwargs):
        select = make_selection(**kwargs)
        print("selection: ", select)
        super().__init__(universe, select, verbose)