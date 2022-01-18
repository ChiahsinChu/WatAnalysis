from signal import raise_signal
from tkinter.messagebox import NO
import numpy as np
from MDAnalysis.lib.distances import capped_distance, calc_angles
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis


class PartialHBAnalysis(HydrogenBondAnalysis):

    def __init__(self,
                 universe,
                 regions=None,
                 surf_ids=None,
                 donors_sel=None,
                 hydrogens_sel=None,
                 acceptors_sel=None,
                 between=None,
                 d_h_cutoff=1.2,
                 d_a_cutoff=3.0,
                 d_h_a_angle_cutoff=150,
                 update_selections=True):

        super(PartialHBAnalysis,
              self).__init__(universe, donors_sel, hydrogens_sel,
                             acceptors_sel, between, d_h_cutoff, d_a_cutoff,
                             d_h_a_angle_cutoff, update_selections)

        self.regions = regions
        if regions is not None:
            self.regions = regions[np.argsort(regions[:, 0])]

        self.surf_ids = surf_ids
        if surf_ids is not None:
            # TODO: universe should be wrapped before passed
            self.surf_ids = np.array(surf_ids, dtype=np.int32)

        # trajectory value initial
        self.ag = universe.atoms
        self.ag_natoms = len(self.ag)

        #parallel value initial
        self.para = None
        self._para_region = None

    def _parallel_init(self, *args, **kwargs):

        start = self._para_region.start
        stop = self._para_region.stop
        step = self._para_region.step
        self._setup_frames(self._trajectory, start, stop, step)
        self._prepare()

    def _single_frame(self):
        box = self._ts.dimensions
        # Update donor-hydrogen pairs if necessary
        if self.update_selections:
            self._donors, self._hydrogens = self._get_dh_pairs()
        donors = self._donors
        hydrogens = self._hydrogens

        if self.regions is not None:
            if self.surf_ids is None:
                raise AttributeError(
                    'surf_ids is required when regions is not None')
            ts_surf_zlo = self._ts.positions.T[2][self.surf_ids[0]]
            ts_surf_zhi = self._ts.positions.T[2][self.surf_ids[1]]
            z_surf = [np.mean(ts_surf_zlo), np.mean(ts_surf_zhi)]
            print(z_surf)
            mask = self._get_mask(z_surf, donors.positions)
            donors = donors[mask]
            hydrogens = hydrogens[mask]

        # find D and A within cutoff distance of one another
        # min_cutoff = 1.0 as an atom cannot form a hydrogen bond with itself
        d_a_indices, d_a_distances = capped_distance(
            donors.positions,
            self._acceptors.positions,
            max_cutoff=self.d_a_cutoff,
            min_cutoff=1.0,
            box=box,
            return_distances=True,
        )

        # Remove D-A pairs more than d_a_cutoff away from one another
        tmp_donors = donors[d_a_indices.T[0]]
        tmp_hydrogens = hydrogens[d_a_indices.T[0]]
        tmp_acceptors = self._acceptors[d_a_indices.T[1]]

        # Remove donor-acceptor pairs between pairs of AtomGroups we are not
        # interested in
        if self.between_ags is not None:
            tmp_donors, tmp_hydrogens, tmp_acceptors = \
                self._filter_atoms(tmp_donors, tmp_hydrogens, tmp_acceptors)

        # Find D-H-A angles greater than d_h_a_angle_cutoff
        d_h_a_angles = np.rad2deg(
            calc_angles(tmp_donors.positions,
                        tmp_hydrogens.positions,
                        tmp_acceptors.positions,
                        box=box))
        hbond_indices = np.where(d_h_a_angles > self.d_h_a_angle)[0]

        # Retrieve atoms, distances and angles of hydrogen bonds
        hbond_donors = tmp_donors[hbond_indices]
        hbond_hydrogens = tmp_hydrogens[hbond_indices]
        hbond_acceptors = tmp_acceptors[hbond_indices]
        hbond_distances = d_a_distances[hbond_indices]
        hbond_angles = d_h_a_angles[hbond_indices]

        # Store data on hydrogen bonds found at this frame
        self.results.hbonds[0].extend(
            np.full_like(hbond_donors, self._ts.frame))
        self.results.hbonds[1].extend(hbond_donors.indices)
        self.results.hbonds[2].extend(hbond_hydrogens.indices)
        self.results.hbonds[3].extend(hbond_acceptors.indices)
        self.results.hbonds[4].extend(hbond_distances)
        self.results.hbonds[5].extend(hbond_angles)

    def _get_mask(self, z_surf, positions):
        abs_regions = self._get_abs_regions(z_surf)
        z_coords = positions[:, 2]

    def _get_abs_regions(self, z_surf):
        pass

    def run(self, start=None, stop=None, step=None, verbose=None):

        #self._trajectory._reopen()
        if verbose == True:
            print(" ", end='')
        super().run(start, stop, step,
                    verbose)  ### will be problem for conclude operation

        if self.para:
            block_result = self._para_block_result()
            if block_result == None:
                raise ValueError(
                    "in parallel, block result has not been defined or no data output!"
                )
            #logger.info("block_anal finished.")
            return block_result

    def _para_block_result(self, ):

        # data need to be transformed, which is usually relative to values in prepare() method.

        return [self.results["hbonds"]]

    def _parallel_conclude(self, rawdata):

        total_array = np.empty(shape=(0, rawdata[0][0].shape[1]))
        for single_data in rawdata:
            total_array = np.concatenate([total_array, single_data[0]], axis=0)

        self.results["hbonds"] = total_array

        return "FINISH PARA CONCLUDE"
