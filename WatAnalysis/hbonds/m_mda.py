# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis
from MDAnalysis.lib.distances import calc_angles, capped_distance


class PartialHBAnalysis(HydrogenBondAnalysis):
    def __init__(
        self,
        universe,
        hb_region=None,
        surf_ids=None,
        donors_sel=None,
        hydrogens_sel=None,
        acceptors_sel=None,
        between=None,
        d_h_cutoff=1.2,
        d_a_cutoff=3.0,
        angle_cutoff_type="d_h_a",
        angle_cutoff=None,
        update_selections=True,
        update_masks=False,
    ):
        """
        Parameters
        ----------
        hb_region : (2, )-shape List (None)
            region (z positions w.r.t surface) in which the HB is analyzed
            if None, all HB is considered
        surf_ids : (2, n_surf)-shape List (None)
            atomic indices of surface atoms
            expected to be not None if region is not None
        angle_cutoff_type : d_h_a or h_d_a
            type for setting angle cutoff
            d_h_a: donor-hydrogen-acceptor
            h_d_a: hydrogen-donor-acceptor
        update_masks : bool (False)
            if calculate the masks for donors & acceptors individually
            False if donors & acceptors are exactly the same
            and then share their mask
        Other parameters are identical with their counterparts in the parent class.
        """

        # set new cutoff attributes
        self.angle_cutoff_type = angle_cutoff_type
        if self.angle_cutoff_type == "d_h_a":
            if angle_cutoff is None:
                self.angle_cutoff = 150
            else:
                self.angle_cutoff = angle_cutoff
        elif self.angle_cutoff_type == "h_d_a":
            if angle_cutoff is None:
                self.angle_cutoff = 30
            else:
                self.angle_cutoff = angle_cutoff
        else:
            raise AttributeError("Unknown angle cutoff type!")
        self.update_masks = update_masks

        super(PartialHBAnalysis, self).__init__(
            universe,
            donors_sel,
            hydrogens_sel,
            acceptors_sel,
            between,
            d_h_cutoff,
            d_a_cutoff,
            self.angle_cutoff,
            update_selections,
        )

        self.surf_ids = surf_ids
        if surf_ids is not None:
            # TODO: universe should be wrapped before passed
            self.surf_ids = np.array(surf_ids, dtype=np.int32)

        self.hb_region = hb_region
        if hb_region is not None:
            self.hb_region = np.array(self.hb_region)
            if self.hb_region.shape != (2,):
                raise AttributeError("hb_region is expected to be a (2, )-shape array")
            if self.surf_ids is None:
                raise AttributeError(
                    "surf_ids is expected to be not None when hb_region is not None"
                )

        # trajectory value initial
        self.ag = universe.atoms
        self.ag_natoms = len(self.ag)

        # parallel value initial
        self.para = None
        self._para_region = None

    def _prepare(self):
        """
        Add z coords of donors/hydrogens/acceptors w.r.t. the
        nearest surface into results

        self.results.hbonds:
        frame_idx, donor_idx, hydrogen_idx, acceptor_idx, HB distance,
        HB angle, donor_zcoord, hydrogen_zcoord, acceptor_zcoord
        """
        super(PartialHBAnalysis, self)._prepare()
        self.results.hbonds = [[], [], [], [], [], [], [], [], []]
        self.zs_surf = np.zeros((self.n_frames, 2), dtype=np.float32)

    def _single_frame(self):
        box = self._ts.dimensions

        # Update donor-hydrogen pairs if necessary
        if self.update_selections:
            self._donors, self._hydrogens = self._get_dh_pairs()
        donors = self._donors
        hydrogens = self._hydrogens
        acceptors = self._acceptors

        # get selected AtomGroup donors, hydrogens, acceptors
        if self.hb_region is not None:
            ts_surf_zlo = self._ts.positions.T[2][self.surf_ids[0]]
            ts_surf_zhi = self._ts.positions.T[2][self.surf_ids[1]]
            z_surf = [np.mean(ts_surf_zlo), np.mean(ts_surf_zhi)]
            np.copyto(self.zs_surf[self._frame_index], z_surf)
            # print(z_surf)

            # Mask to select atoms at specified regions
            mask = self._get_mask(z_surf, self._donors.positions)
            donors = self._donors[mask]
            hydrogens = self._hydrogens[mask]
            left_donors = self._donors[mask == False]
            left_hydrogens = self._hydrogens[mask == False]
            if self.update_masks == False:
                mask = mask[::2]
            else:
                mask = self._get_mask(z_surf, self._acceptors.positions)
            acceptors = self._acceptors[mask]

        # find D (selected) and A (all) within cutoff distance of one another
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
            tmp_donors, tmp_hydrogens, tmp_acceptors = self._filter_atoms(
                tmp_donors, tmp_hydrogens, tmp_acceptors
            )

        # Find angles and compared with the angle cutoff
        if self.angle_cutoff_type == "d_h_a":
            cutoff_angles = np.rad2deg(
                calc_angles(
                    tmp_donors.positions,
                    tmp_hydrogens.positions,
                    tmp_acceptors.positions,
                    box=box,
                )
            )
            hbond_indices = np.where(cutoff_angles >= self.angle_cutoff)[0]
        elif self.angle_cutoff_type == "h_d_a":
            cutoff_angles = np.rad2deg(
                calc_angles(
                    tmp_hydrogens.positions,
                    tmp_donors.positions,
                    tmp_acceptors.positions,
                    box=box,
                )
            )
            hbond_indices = np.where(cutoff_angles <= self.angle_cutoff)[0]

        # Retrieve atoms, distances and angles of hydrogen bonds
        hbond_donors = tmp_donors[hbond_indices]
        hbond_hydrogens = tmp_hydrogens[hbond_indices]
        hbond_acceptors = tmp_acceptors[hbond_indices]
        hbond_distances = d_a_distances[hbond_indices]
        hbond_angles = cutoff_angles[hbond_indices]
        if self.hb_region is not None:
            donor_zcoords = self._get_rel_pos(hbond_donors.positions[:, 2], z_surf)
            hydrogen_zcoords = self._get_rel_pos(
                hbond_hydrogens.positions[:, 2], z_surf
            )
            acceptor_zcoords = self._get_rel_pos(
                hbond_acceptors.positions[:, 2], z_surf
            )

        # Store data on hydrogen bonds found at this frame
        self.results.hbonds[0].extend(np.full_like(hbond_donors, self._ts.frame))
        self.results.hbonds[1].extend(hbond_donors.indices)
        self.results.hbonds[2].extend(hbond_hydrogens.indices)
        self.results.hbonds[3].extend(hbond_acceptors.indices)
        self.results.hbonds[4].extend(hbond_distances)
        self.results.hbonds[5].extend(hbond_angles)

        if self.hb_region is None:
            placeholder = np.zeros_like(hbond_angles).tolist()
            self.results.hbonds[6].extend(placeholder)
            self.results.hbonds[7].extend(placeholder)
            self.results.hbonds[8].extend(placeholder)
        else:
            self.results.hbonds[6].extend(donor_zcoords)
            self.results.hbonds[7].extend(hydrogen_zcoords)
            self.results.hbonds[8].extend(acceptor_zcoords)
            # consider D(not selected)-A(selected) pair
            d_a_indices, d_a_distances = capped_distance(
                left_donors.positions,
                acceptors.positions,
                max_cutoff=self.d_a_cutoff,
                min_cutoff=1.0,
                box=box,
                return_distances=True,
            )

            # Remove D-A pairs more than d_a_cutoff away from one another
            tmp_donors = left_donors[d_a_indices.T[0]]
            tmp_hydrogens = left_hydrogens[d_a_indices.T[0]]
            tmp_acceptors = acceptors[d_a_indices.T[1]]

            # Remove donor-acceptor pairs between pairs of AtomGroups we are not
            # interested in
            if self.between_ags is not None:
                tmp_donors, tmp_hydrogens, tmp_acceptors = self._filter_atoms(
                    tmp_donors, tmp_hydrogens, tmp_acceptors
                )

            # Find angles and compared with the angle cutoff
            if self.angle_cutoff_type == "d_h_a":
                cutoff_angles = np.rad2deg(
                    calc_angles(
                        tmp_donors.positions,
                        tmp_hydrogens.positions,
                        tmp_acceptors.positions,
                        box=box,
                    )
                )
                hbond_indices = np.where(cutoff_angles > self.angle_cutoff)[0]
            elif self.angle_cutoff_type == "h_d_a":
                cutoff_angles = np.rad2deg(
                    calc_angles(
                        tmp_hydrogens.positions,
                        tmp_donors.positions,
                        tmp_acceptors.positions,
                        box=box,
                    )
                )
                hbond_indices = np.where(cutoff_angles < self.angle_cutoff)[0]

            # Retrieve atoms, distances and angles of hydrogen bonds
            hbond_donors = tmp_donors[hbond_indices]
            hbond_hydrogens = tmp_hydrogens[hbond_indices]
            hbond_acceptors = tmp_acceptors[hbond_indices]
            hbond_distances = d_a_distances[hbond_indices]
            hbond_angles = cutoff_angles[hbond_indices]
            donor_zcoords = self._get_rel_pos(hbond_donors.positions[:, 2], z_surf)
            hydrogen_zcoords = self._get_rel_pos(
                hbond_hydrogens.positions[:, 2], z_surf
            )
            acceptor_zcoords = self._get_rel_pos(
                hbond_acceptors.positions[:, 2], z_surf
            )

            # Store data on hydrogen bonds found at this frame
            self.results.hbonds[0].extend(np.full_like(hbond_donors, self._ts.frame))
            self.results.hbonds[1].extend(hbond_donors.indices)
            self.results.hbonds[2].extend(hbond_hydrogens.indices)
            self.results.hbonds[3].extend(hbond_acceptors.indices)
            self.results.hbonds[4].extend(hbond_distances)
            self.results.hbonds[5].extend(hbond_angles)
            self.results.hbonds[6].extend(donor_zcoords)
            self.results.hbonds[7].extend(hydrogen_zcoords)
            self.results.hbonds[8].extend(acceptor_zcoords)

    def _get_mask(self, z_surf, positions):
        """
        TBC
        """
        z_coords = positions[:, 2]
        abs_region = self._get_abs_region(self.hb_region, z_surf)
        # print(abs_region)
        mask_0 = z_coords >= abs_region[0][0]
        mask_1 = z_coords < abs_region[0][1]
        mask_2 = z_coords >= abs_region[1][0]
        mask_3 = z_coords < abs_region[1][1]
        mask = (mask_0 & mask_1) | (mask_2 & mask_3)
        return mask

    def _get_abs_region(self, hb_region, z_surf):
        """
        TBC
        """
        region_0 = z_surf[0] + hb_region
        region_1 = z_surf[1] - hb_region
        tmp = [region_1[1], region_1[0]]
        region_1 = np.array(tmp)
        abs_region = [region_0, region_1]
        return np.array(abs_region)

    def _get_rel_pos(self, positions, z_surf):
        """
        TBC
        """
        # z positions w.r.t. lower surface
        rel_pos_0 = positions - z_surf[0]
        # z positions w.r.t. upper surface
        rel_pos_1 = z_surf[1] - positions
        # take the shorter distance
        rel_pos = np.concatenate(([rel_pos_0], [rel_pos_1]), axis=0)
        # pos for lower, neg for upper
        mask = -2 * np.argmin(rel_pos, axis=0) + 1
        rel_pos = np.min(rel_pos, axis=0) * mask
        return rel_pos

    def _parallel_init(self, *args, **kwargs):
        start = self._para_region.start
        stop = self._para_region.stop
        step = self._para_region.step
        self._setup_frames(self._trajectory, start, stop, step)
        self._prepare()

    def run(self, start=None, stop=None, step=None, verbose=None):
        # self._trajectory._reopen()
        if verbose == True:
            print(" ", end="")
        super().run(
            start, stop, step, verbose
        )  ### will be problem for conclude operation

        if self.para:
            block_result = self._para_block_result()
            if block_result == None:
                raise ValueError(
                    "in parallel, block result has not been defined or no data output!"
                )
            # logger.info("block_anal finished.")
            return block_result

    def _para_block_result(
        self,
    ):
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
