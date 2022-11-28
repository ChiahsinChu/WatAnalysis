import numpy as np
from MDAnalysis.analysis.dielectric import DielectricConstant
from MDAnalysis.units import constants, convert
from MDAnalysis.exceptions import NoDataError

from WatAnalysis.preprocess import make_selection


class DC(DielectricConstant):

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
            select = make_selection(sel_region=sel_region, **self.kwargs)
            self.ags.append(self.universe.select_atoms(select))

    def _single_frame(self):
        for ii in range(self.nbins):
            if self.make_whole:
                self.ags[ii].unwrap()

            # volume of each bin rather than the whole system
            # self.volume += self.atomgroup.universe.trajectory.ts.volume
            self.volume[
                ii] += self._ts.dimensions[0] * self._ts.dimensions[1] * (self.bins[ii+1] - self.bins[ii])

            M = np.dot(self.ags[ii].charges, self.ags[ii].positions)
            self.results.M[ii] += M
            self.results.M2[ii] += M * M

    def _conclude(self):
        self.results.M /= self.n_frames
        self.results.M2 /= self.n_frames
        self.volume /= self.n_frames

        self.results.fluct = self.results.M2 - self.results.M * self.results.M

        self.results.eps = self.results.fluct / (
            convert(constants["Boltzman_constant"], "kJ/mol", "eV") *
            self.temperature * np.reshape(self.volume, (self.nbins, 1)) * constants["electric_constant"])

        self.results.eps_mean = self.results.eps.mean(axis=-1)

        self.results.eps += 1
        self.results.eps_mean += 1
