from curses import raw
import numpy as np
from MDAnalysis.lib.distances import capped_distance, calc_angles
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis


class PartialHBAnalysis(HydrogenBondAnalysis):

    def __init__(self,
                 universe,
                 region=None,
                 surf_ids=None,
                 donors_sel=None,
                 hydrogens_sel=None,
                 acceptors_sel=None,
                 between=None,
                 d_h_cutoff=1.2,
                 d_a_cutoff=3.0,
                 angle_cutoff_type="d_h_a",
                 angle_cutoff=150,
                 update_selections=True):
        """
        Parameters
        ----------
        region : (2, )-shape List (optional)
            region in which the HB is analyzed
            z positions w.r.t surface
            if None, all HB is considered
        surf_ids : (2, n_surf)-shape List (optional)
            atomic indices of surface atoms
            expected to be not None if region is not None
            
        other parameters are identical with those in the parent class
        """
        super(PartialHBAnalysis,
              self).__init__(universe, donors_sel, hydrogens_sel,
                             acceptors_sel, between, d_h_cutoff, d_a_cutoff,
                             angle_cutoff, update_selections)

        self.surf_ids = surf_ids
        if surf_ids is not None:
            # TODO: universe should be wrapped before passed
            self.surf_ids = np.array(surf_ids, dtype=np.int32)

        self.region = region
        if region is not None:
            self.region = np.array(self.region)
            if self.region.shape != (2, ):
                raise AttributeError(
                    'region is expected to be a (2, )-shape array')
            if self.surf_ids is None:
                raise AttributeError(
                    'surf_ids is expected to be not None when regions is not None'
                )
        self.angle_cutoff_type = angle_cutoff_type
        if self.angle_cutoff_type != "d_h_a" and self.angle_cutoff_type != "h_d_a":
            raise AttributeError('Unknown angle cutoff type!')
        self.angle_cutoff = angle_cutoff

        # trajectory value initial
        self.ag = universe.atoms
        self.ag_natoms = len(self.ag)

        #parallel value initial
        self.para = None
        self._para_region = None

    def _single_frame(self):
        box = self._ts.dimensions

        # Update donor-hydrogen pairs if necessary
        if self.update_selections:
            self._donors, self._hydrogens = self._get_dh_pairs()
        donors = self._donors
        hydrogens = self._hydrogens

        if self.region is not None:
            ts_surf_zlo = self._ts.positions.T[2][self.surf_ids[0]]
            ts_surf_zhi = self._ts.positions.T[2][self.surf_ids[1]]
            z_surf = [np.mean(ts_surf_zlo), np.mean(ts_surf_zhi)]
            print(z_surf)

            # Mask to select atoms at specified regions
            mask = self._get_mask(z_surf, self._donors.positions)
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

        # Find angles and compared with the angle cutoff
        if self.angle_cutoff_type == "d_h_a":
            d_h_a_angles = np.rad2deg(
                calc_angles(tmp_donors.positions,
                            tmp_hydrogens.positions,
                            tmp_acceptors.positions,
                            box=box))
            hbond_indices = np.where(d_h_a_angles > self.angle_cutoff)[0]
        elif self.angle_cutoff_type == "h_d_a":
            h_d_a_angles = np.rad2deg(
                calc_angles(tmp_hydrogens.positions,
                            tmp_donors.positions,
                            tmp_acceptors.positions,
                            box=box))
            hbond_indices = np.where(h_d_a_angles < self.angle_cutoff)[0]

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
        z_coords = positions[:, 2]
        abs_region = self._get_abs_region(self.region, z_surf)
        print(abs_region)
        mask_0 = z_coords >= abs_region[0][0]
        mask_1 = z_coords < abs_region[0][1]
        mask_2 = z_coords >= abs_region[1][0]
        mask_3 = z_coords < abs_region[1][1]
        mask = (mask_0 & mask_1) | (mask_2 & mask_3)
        return mask

    def _get_abs_region(self, region, z_surf):
        region_0 = z_surf[0] + region
        region_1 = z_surf[1] - region
        tmp = [region_1[1], region_1[0]]
        region_1 = np.array(tmp)
        abs_region = [region_0, region_1]
        return np.array(abs_region)

    def _parallel_init(self, *args, **kwargs):

        start = self._para_region.start
        stop = self._para_region.stop
        step = self._para_region.step
        self._setup_frames(self._trajectory, start, stop, step)
        self._prepare()

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

        return [self.results.hbonds]

    def _parallel_conclude(self, rawdata):
        # set attributes for further analysis
        method_attr = rawdata[-1]
        del rawdata[-1]
        self.start = method_attr[0]
        self.stop = method_attr[1]
        self.step = method_attr[2]
        self.frames = np.arange(self.start, self.stop, self.step)

        total_array = np.empty(shape=(0, rawdata[0][0].shape[1]))
        for single_data in rawdata:
            total_array = np.concatenate([total_array, single_data[0]], axis=0)

        self.results["hbonds"] = total_array

        return "FINISH PARA CONCLUDE"
