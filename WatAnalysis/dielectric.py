import numpy as np

from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.analysis.dielectric import DielectricConstant
from MDAnalysis.units import constants, convert

from WatAnalysis.preprocess import make_selection, make_selection_two


class InverseDielectricConstant(AnalysisBase):

    def __init__(self,
                 universe,
                 bins,
                 axis="z",
                 temperature=330,
                 make_whole=True,
                 verbose=False,
                 **kwargs) -> None:
        super().__init__(universe.trajectory, verbose)

        self.universe = universe
        self.atoms = universe.atoms
        self.bins = bins
        self.nbins = len(bins) - 1
        axis_dict = {"x": 0, "y": 1, "z": 2}
        self.axis = axis_dict[axis]
        self.temperature = temperature
        self.make_whole = make_whole
        self.kwargs = kwargs

    def _prepare(self):
        if not hasattr(self.atoms, "charges"):
            raise NoDataError("No charges defined given atomgroup.")

        if not np.allclose(
                self.atoms.total_charge(compound='fragments'), 0.0, atol=1E-5):
            raise NotImplementedError("Analysis for non-neutral systems or"
                                      " systems with free charges are not"
                                      " available.")

        self.results.m = np.zeros((self.nbins))
        self.results.mM = np.zeros((self.nbins))
        self.results.M = 0
        self.results.M2 = 0
        self.volume = 0

        self.ags = []
        for ii in range(self.nbins):
            sel_region = [self.bins[ii], self.bins[ii + 1]]
            select = make_selection(sel_region=sel_region, **self.kwargs)
            self.ags.append(self.universe.select_atoms(select, updating=True))

    def _single_frame(self):
        ave_axis = np.delete(np.arange(3), self.axis)
        volume = self._ts.volume
        self.volume += volume

        if self.make_whole:
            self.atoms.unwrap()

        M = np.dot(self.atoms.charges,
                   self.atoms.positions)[self.axis] / volume
        self.results.M += M
        self.results.M2 += M * M

        for ii in range(self.nbins):
            if self.make_whole:
                self.ags[ii].unwrap()
            bin_volume = self._ts.dimensions[
                ave_axis[0]] * self._ts.dimensions[ave_axis[1]] * (
                    self.bins[ii + 1] - self.bins[ii]) * 2
            m = np.dot(self.ags[ii].charges,
                       self.ags[ii].positions)[self.axis] / bin_volume
            self.results.m[ii] += m
            self.results.mM[ii] += m * M

    def _conclude(self):
        self.results.m /= self.n_frames
        self.results.mM /= self.n_frames
        self.results.M /= self.n_frames
        self.results.M2 /= self.n_frames
        self.volume /= self.n_frames

        x_fluct = self.results.mM - self.results.m * self.results.M
        M_fluct = self.results.M2 - self.results.M * self.results.M
        const = convert(
            constants["Boltzman_constant"], "kJ/mol",
            "eV") * self.temperature * constants["electric_constant"]
        self.results.inveps = 1 - x_fluct / (const + M_fluct / self.volume)


class DeprecatedDC(DielectricConstant):

    def __init__(self,
                 universe,
                 bins,
                 temperature=330,
                 make_whole=True,
                 verbose=False,
                 **kwargs) -> None:
        self.universe = universe
        self.bins = bins
        self.nbins = len(bins) - 1
        self.kwargs = kwargs
        super().__init__(universe.atoms,
                         temperature,
                         make_whole,
                         verbose=verbose)

    def _prepare(self):
        super()._prepare()

        # reset
        self.volume = np.zeros(self.nbins)
        self.results.M = np.zeros((self.nbins, 3))
        self.results.M2 = np.zeros((self.nbins, 3))
        self.results.fluct = np.zeros((self.nbins, 3))
        self.results.eps = np.zeros((self.nbins, 3))
        self.results.eps_mean = np.zeros(self.nbins)

        self.ags = []
        for ii in range(self.nbins):
            sel_region = [self.bins[ii], self.bins[ii + 1]]
            select = make_selection_two(sel_region=sel_region, **self.kwargs)
            self.ags.append(
                self.universe.select_atoms(select[0], updating=True))
            self.ags.append(
                self.universe.select_atoms(select[1], updating=True))

    def _single_frame(self):

        for ii in range(self.nbins):
            # lower surface
            if self.make_whole:
                self.ags[2 * ii].unwrap()

            # volume of each bin rather than the whole system
            # self.volume += self.atomgroup.universe.trajectory.ts.volume
            self.volume[ii] += self._ts.dimensions[0] * self._ts.dimensions[
                1] * (self.bins[ii + 1] - self.bins[ii])

            M = np.dot(self.ags[2 * ii].charges, self.ags[2 * ii].positions)
            self.results.M[ii] += M
            self.results.M2[ii] += M * M

            # upper surface
            if self.make_whole:
                self.ags[2 * ii + 1].unwrap()

            # volume of each bin rather than the whole system
            # self.volume += self.atomgroup.universe.trajectory.ts.volume
            self.volume[ii] += self._ts.dimensions[0] * self._ts.dimensions[
                1] * (self.bins[ii + 1] - self.bins[ii])

            M = np.dot(self.ags[2 * ii + 1].charges,
                       self.ags[2 * ii + 1].positions) * np.array(
                           [1., 1., -1.])
            self.results.M[ii] += M
            self.results.M2[ii] += M * M

    def _conclude(self):
        self.results.M /= self.n_frames
        self.results.M2 /= self.n_frames
        self.volume /= self.n_frames

        self.results.fluct = self.results.M2 - self.results.M * self.results.M

        self.results.eps = self.results.fluct / (
            convert(constants["Boltzman_constant"], "kJ/mol", "eV") *
            self.temperature * np.reshape(self.volume, (self.nbins, 1)) *
            constants["electric_constant"])

        self.results.eps_mean = self.results.eps.mean(axis=-1)

        self.results.eps += 1
        self.results.eps_mean += 1
