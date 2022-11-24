import numpy as np
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis

from MDAnalysis.lib.distances import capped_distance
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.core.groups import AtomGroup

from WatAnalysis.preprocess import make_selection


class WatCoverage(AnalysisBase):

    def __init__(self, universe, verbose=False, **kwargs):
        select = make_selection(**kwargs)
        # print("selection: ", select)
        self.universe = universe
        trajectory = universe.trajectory
        super().__init__(trajectory, verbose=verbose)
        self.n_frames = len(trajectory)
        self.ag = universe.select_atoms(select, updating=True)

    def _prepare(self):
        # placeholder for water z
        self.n_water = np.zeros((self.n_frames), dtype=np.int32)

    def _single_frame(self):
        self.n_water[self._frame_index] = len(self.ag)

    def _conclude(self):
        return self.n_water


class HBA(HydrogenBondAnalysis):

    def __init__(self,
                 universe,
                 donors_sel=None,
                 hydrogens_sel=None,
                 acceptors_sel=None,
                 between=None,
                 d_h_cutoff=1.2,
                 d_a_cutoff=3,
                 d_h_a_angle_cutoff=150,
                 update_acceptors=False,
                 update_donors=False):
        self.update_acceptors = update_acceptors
        self.update_donors = update_donors
        update_selection = (update_donors | update_acceptors)
        super().__init__(universe, donors_sel, hydrogens_sel, acceptors_sel,
                         between, d_h_cutoff, d_a_cutoff, d_h_a_angle_cutoff,
                         update_selection)

    def _prepare(self):
        self.results.hbonds = [[], [], [], [], [], []]

        # Set atom selections if they have not been provided
        if not self.acceptors_sel:
            self.acceptors_sel = self.guess_acceptors()
        if not self.hydrogens_sel:
            self.hydrogens_sel = self.guess_hydrogens()

        # Select atom groups
        self._acceptors = self.u.select_atoms(self.acceptors_sel,
                                              updating=self.update_acceptors)
        self._donors, self._hydrogens = self._get_dh_pairs()

    def _get_dh_pairs(self):
        """Finds donor-hydrogen pairs.

        Returns
        -------
        donors, hydrogens: AtomGroup, AtomGroup
            AtomGroups corresponding to all donors and all hydrogens. AtomGroups are ordered such that, if zipped, will
            produce a list of donor-hydrogen pairs.
        """

        # If donors_sel is not provided, use topology to find d-h pairs
        if not self.donors_sel:
            # We're using u._topology.bonds rather than u.bonds as it is a million times faster to access.
            # This is because u.bonds also calculates properties of each bond (e.g bond length).
            # See https://github.com/MDAnalysis/mdanalysis/issues/2396#issuecomment-596251787
            if not (hasattr(self.u._topology, 'bonds')
                    and len(self.u._topology.bonds.values) != 0):
                raise NoDataError(
                    'Cannot assign donor-hydrogen pairs via topology as no bond information is present. '
                    'Please either: load a topology file with bond information; use the guess_bonds() '
                    'topology guesser; or set HydrogenBondAnalysis.donors_sel so that a distance cutoff '
                    'can be used.')

            hydrogens = self.u.select_atoms(self.hydrogens_sel)
            donors = sum(h.bonded_atoms[0] for h in hydrogens) if hydrogens \
                else AtomGroup([], self.u)

        # Otherwise, use d_h_cutoff as a cutoff distance
        else:
            hydrogens = self.u.select_atoms(self.hydrogens_sel)
            donors = self.u.select_atoms(self.donors_sel,
                                         updating=self.update_donors)
            donors_indices, hydrogen_indices = capped_distance(
                donors.positions,
                hydrogens.positions,
                max_cutoff=self.d_h_cutoff,
                box=self.u.dimensions,
                return_distances=False).T

            donors = donors[donors_indices]
            hydrogens = hydrogens[hydrogen_indices]

        return donors, hydrogens
