import numpy as np
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.waterdynamics import (MeanSquareDisplacement,
                                               SurvivalProbability,
                                               WaterOrientationalRelaxation)

from WatAnalysis.preprocess import make_selection


class WOR(WaterOrientationalRelaxation):

    def __init__(self, universe, t0, tf, dtmax, nproc=1, **kwargs):
        """
        sel_region, surf_ids, c_ag, select_all, bonded
        """
        select = make_selection(**kwargs)
        print("selection: ", select)
        super().__init__(universe, select, t0, tf, dtmax, nproc)


# class MSD(AnalysisBase):
#     def __init__(self, universe, t0, tf, dtmax, perp="z", verbose=False, **kwargs):
#         self.universe = universe
#         trajectory = universe.trajectory
#         super().__init__(trajectory, verbose=verbose)

#         select = make_selection(**kwargs)
#         # print("selection: ", select)
#         self.n_frames = len(trajectory)
#         self.ag = universe.select_atoms(select, updating=True)

#         self.selection = select
#         self.t0 = t0
#         self.tf = tf
#         self.dtmax = dtmax
#         # TODO: exception capture
#         perp_dict = {'x': 0, 'y': 1, 'z': 2}
#         self.perp = perp_dict[perp]

#     def _prepare(self):
#         self.timeseries = []
    
#     def _single_frame(self):
#         for dt in list(range(1, self.dtmax + 1)):
#             repInd = self._repeatedIndex(selection1, dt, totalFrames)
#             sumsdt = 0
#             n = 0.0
#             sumDeltaO = 0.0
#             valOList = []

#             for j in range(totalFrames // dt - 1):
#                 a = self._getOneDeltaPoint(universe, repInd, j, sumsdt, dt)
#                 sumDeltaO += a
#                 valOList.append(a)
#                 sumsdt += dt
#                 n += 1

#             # return sumDeltaO / n if n > 0 else 0

#             output = self._getMeanOnePoint(self.universe, selection_out,
#                                            self.selection, dt, self.tf)
#             self.timeseries.append(output)

#     def _conclude(self):
#         pass

#     def _repeatedIndex(self, selection, dt, totalFrames):
#         """
#         Indicate the comparation between all the t+dt.
#         The results is a list of list with all the repeated index per frame
#         (or time).

#         - Ex: dt=1, so compare frames (1,2),(2,3),(3,4)...
#         - Ex: dt=2, so compare frames (1,3),(3,5),(5,7)...
#         - Ex: dt=3, so compare frames (1,4),(4,7),(7,10)...
#         """
#         rep = []
#         for i in range(int(round((totalFrames - 1) / float(dt)))):
#             if (dt * i + dt < totalFrames):
#                 rep.append(
#                     self._sameMolecTandDT(selection, dt * i, (dt * i) + dt))
#         return rep

#     def _getOneDeltaPoint(self, universe, repInd, i, t0, dt):
#         """
#         Gives one point to calculate the mean and gets one point of the plot
#         C_vect vs t.

#         - Ex: t0=1 and dt=1 so calculate the t0-dt=1-2 interval.
#         - Ex: t0=5 and dt=3 so calcultate the t0-dt=5-8 interval

#         i = come from getMeanOnePoint (named j) (int)
#         """
#         valO = 0
#         n = 0
#         for j in range(len(repInd[i]) // 3):
#             begj = 3 * j
#             universe.trajectory[t0]
#             # Plus zero is to avoid 0to be equal to 0tp
#             Ot0 = repInd[i][begj].position + 0

#             universe.trajectory[t0 + dt]
#             # Plus zero is to avoid 0to be equal to 0tp
#             Otp = repInd[i][begj].position + 0

#             # position oxygen
#             OVector = Ot0 - Otp
#             # here it is the difference with
#             # waterdynamics.WaterOrientationalRelaxation
#             valO += np.dot(OVector, OVector)
#             n += 1

#         # if no water molecules remain in selection, there is nothing to get
#         # the mean, so n = 0.
#         return valO / n if n > 0 else 0

#     def _getMeanOnePoint(self, universe, selection1, selection_str, dt,
#                          totalFrames):
#         """
#         This function gets one point of the plot C_vec vs t. It's uses the
#         _getOneDeltaPoint() function to calculate the average.

#         """
#         pass

#     def _sameMolecTandDT(self, selection, t0d, tf):
#         """
#         Compare the molecules in the t0d selection and the t0d+dt selection and
#         select only the particles that are repeated in both frame. This is to
#         consider only the molecules that remains in the selection after the dt
#         time has elapsed. The result is a list with the indexs of the atoms.
#         """
#         a = set(selection[t0d])
#         b = set(selection[tf])
#         sort = sorted(list(a.intersection(b)))
#         return sort




# TODO: fix the slice analysis
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

    def _getMeanOnePoint(self, universe, selection1, dt, totalFrames):
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
                self.universe, selection_out, dt, self.tf)
            self.timeseries_perp.append(output_perp)
            self.timeseries_para.append(output_para)
        self.timeseries = np.array(self.timeseries_para) + np.array(
            self.timeseries_perp)


class SP(SurvivalProbability):

    def __init__(self, universe, verbose=False, **kwargs):
        select = make_selection(**kwargs)
        # print("selection: ", select)
        super().__init__(universe, select, verbose)
