# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Functionality for computing dynamical quantities from molecular dynamics
trajectories of water at interfaces
"""

import numpy as np
from MDAnalysis.analysis.msd import EinsteinMSD

from WatAnalysis.preprocess import make_selection

from .waterdynamics import SurvivalProbability, WaterOrientationalRelaxation


def calc_vector_autocorrelation(
    max_tau: int,
    delta_tau: int,
    step: int,
    vectors: np.ndarray,
    mask: np.ndarray,
):
    """
    Calculate the autocorrelation function for a vector quantity over time.

    Parameters
    ----------
    max_tau : int
        Maximum lag time to calculate ACF for
    delta_tau : int
        Time interval between lag times (points on the C(tau) vs. tau curve)
    step : int
        Step size for time origins. If equal to max_tau, there is no overlap between
        time windows considered in the calculation (so more uncorrelated).
    vectors : numpy.ndarray
        Array of vectors with shape (num_timesteps, num_particles, 3)
    mask : numpy.ndarray
        Boolean mask array indicating which particles to include, shape
        (num_timesteps, num_particles)

    Returns
    -------
    tau : numpy.ndarray
        Array of lag times
    acf : numpy.ndarray
        Normalized autocorrelation function values for each lag time
    """
    tau = np.arange(start=0, stop=max_tau, step=delta_tau)
    acf = np.zeros(tau.shape)
    mask = np.expand_dims(mask, axis=2)

    # Calculate ACF for each lag time
    for i, t in enumerate(tau):
        n_selected_vectors = None
        if t == 0:
            # For t=0, just calculate the dot product with itself
            dot_products = np.sum(
                vectors * vectors * mask, axis=2
            )  # Shape: (num_timesteps, num_molecules)
            n_selected_vectors = np.count_nonzero(mask)
        else:
            # For t > 0, calculate the dot products between shifted arrays
            _vectors_0 = vectors[:-t:step] * mask[:-t:step]  # dipole(t=0)
            _vectors_t = vectors[t::step] * mask[t::step]  # dipole(t=tau)
            dot_products = np.sum(
                _vectors_0 * _vectors_t, axis=2
            )  # Shape: ((num_timesteps - t)//step, num_molecules)
            n_selected_vectors = np.count_nonzero(mask[:-t:step] * mask[t::step])

        # Average over molecules and time origins
        acf[i] = np.sum(dot_products) / n_selected_vectors

    # Normalize the ACF
    acf /= acf[0]  # Normalize by the zero-lag value
    return tau, acf


def calc_survival_probability(
    max_tau: int,
    delta_tau: int,
    step: int,
    mask: np.ndarray,
):
    """
    Calculate the probability that particles remain within a specified region
    over a given time interval.

    Parameters
    ----------
    max_tau : int
        The maximum time delay for which the survival probability is calculated.
    delta_tau : int
        The time delay interval for calculating the survival probability (spacing
        between points on the survival probability vs. tau curve).
    step : int
        The step size between time origins that are taken into account.
        By increasing the step the analysis can be sped up at a loss of statistics.
        If equal to max_tau, there is no overlap between time windows considered in the
        calculation (so more uncorrelated). Defaults to 1.
    mask : numpy.ndarray
        Boolean mask array indicating which molecules are in the region of interest for
        all time steps, shape (num_timesteps, num_molecules)

    Returns
    -------
    tau : numpy.ndarray
        Array of lag times
    acf : numpy.ndarray
        Survival probability values for each lag time
    """
    tau_range = np.arange(start=0, stop=max_tau, step=delta_tau)
    acf = np.zeros(tau_range.shape)

    # Calculate continuous ACF for each lag time
    for i, tau in enumerate(tau_range):
        if tau > 0:
            # N(t), shape: (num_timesteps - tau, )
            n_t = np.sum(mask, axis=1)[:-tau:step]

            # shape: ((num_timesteps - tau)//step, num_molecules)
            intersection = np.ones(mask[:-tau:step].shape)
            for k in range(tau):
                intersection *= mask[k : -tau + k : step]
            intersection *= mask[tau::step]

            # N(t,tau), shape: (num_timesteps - tau, )
            n_t_tau = np.sum(intersection, axis=1)

            acf[i] = np.mean(n_t_tau / n_t)
        else:
            acf[i] = 1

    # Normalize the ACF
    acf /= acf[0]  # Normalize by the zero-lag value
    return tau_range, acf


class WOR(WaterOrientationalRelaxation):
    def __init__(
        self,
        universe,
        t0,
        tf,
        dtmax,
        nproc=1,
        **kwargs,
    ):
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


# # TODO: fix the slice analysis
# class MSD(MeanSquareDisplacement):
#     def __init__(self, universe, t0, tf, dtmax, nproc=1, axis=2, **kwargs):
#         select = make_selection(**kwargs)
#         print("selection: ", select)
#         super().__init__(universe, select, t0, tf, dtmax, nproc)
#         # TODO: exception capture
#         self.axis = axis

#     def _getOneDeltaPoint(self, universe, repInd, i, t0, dt):
#         val_para = 0
#         val_perp = 0

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
#             val_perp += np.square(OVector[self.axis])
#             val_para += np.dot(OVector, OVector) - np.square(OVector[self.axis])
#             # valO += np.dot(OVector, OVector)
#             n += 1

#         # if no water molecules remain in selection, there is nothing to get
#         # the mean, so n = 0.
#         return val_perp / n if n > 0 else 0, val_para / n if n > 0 else 0

#     def _getMeanOnePoint(self, universe, selection1, dt, totalFrames):
#         """
#         This function gets one point of the plot C_vec vs t. It's uses the
#         _getOneDeltaPoint() function to calculate the average.

#         """
#         repInd = self._repeatedIndex(selection1, dt, totalFrames)
#         sumsdt = 0
#         n = 0.0
#         sumDeltaO_perp = 0.0
#         sumDeltaO_para = 0.0
#         # valOList_perp = []
#         # valOList_para = []

#         for j in range(totalFrames // dt - 1):
#             a_perp, a_para = self._getOneDeltaPoint(universe, repInd, j, sumsdt, dt)
#             sumDeltaO_perp += a_perp
#             sumDeltaO_para += a_para
#             # valOList_perp.append(a_perp)
#             # valOList_para.append(a_para)
#             sumsdt += dt
#             n += 1

#         # if no water molecules remain in selection, there is nothing to get
#         # the mean, so n = 0.
#         return sumDeltaO_perp / n if n > 0 else 0, sumDeltaO_para / n if n > 0 else 0

#     def run(self):
#         super().run()

#     def run(self, **kwargs):
#         """Analyze trajectory and produce timeseries"""

#         # All the selection to an array, this way is faster than selecting
#         # later.
#         if self.nproc == 1:
#             selection_out = self._selection_serial(self.universe, self.selection)
#         else:
#             # parallel not yet implemented
#             # selection = selection_parallel(universe, selection_str, nproc)
#             selection_out = self._selection_serial(self.universe, self.selection)

#         self.timeseries_perp = []
#         self.timeseries_para = []
#         for dt in list(range(1, self.dtmax + 1)):
#             output_perp, output_para = self._getMeanOnePoint(
#                 self.universe, selection_out, dt, self.tf
#             )
#             self.timeseries_perp.append(output_perp)
#             self.timeseries_para.append(output_para)
#         self.timeseries = np.array(self.timeseries_para) + np.array(
#             self.timeseries_perp
#         )


class SP(SurvivalProbability):
    def __init__(
        self,
        universe,
        verbose=False,
        **kwargs,
    ):
        select = make_selection(**kwargs)
        # print("selection: ", select)
        super().__init__(universe, select, verbose)


class MSD(EinsteinMSD):
    def __init__(
        self,
        universe,
        msd_type="z",
        fft=True,
        verbose=False,
        **kwargs,
    ):
        select = make_selection(**kwargs)
        # print("selection: ", select)
        super().__init__(universe, select, msd_type, fft, verbose=verbose)
