# SPDX-License-Identifier: LGPL-3.0-or-later
"""
This module include some Child Classes of pmda
analysis method
"""

import numpy as np
from MDAnalysis.lib.distances import calc_angles, capped_distance
from pmda.hbond_analysis import HydrogenBondAnalysis


class ParallelHBAnalysis(HydrogenBondAnalysis):
    """
    Introduction:
        Child Class of HydrogenBondAnalysis in pmda.hbond_analysis
        Parallel analysis of hydrogen bonds **at metal/water interfaces**

    Modifications:
        Parent Class [HydrogenBondAnalysis] returns the hydrogen bonds for
        the whole systems. However, more often than not, we concern only the
        hydrogen bonds within a given region (e.g. the interface region).

    """

    def __init__(
        self,
        universe,
        surf_ids=None,
        regions=None,
        donors_sel=None,
        hydrogens_sel=None,
        acceptors_sel=None,
        d_h_cutoff=1.2,
        d_a_cutoff=3.0,
        d_h_a_angle_cutoff=150,
        update_selections=True,
    ):
        """
        Parameters
        ----------
        regions : (n, 2)-shape array
            distance in z direction w.r.t. surface
            If region is not None, consider only the **donors**
            in the region; if region is None, consider all donors.
        """
        super(ParallelHBAnalysis, self).__init__(
            universe,
            donors_sel=donors_sel,
            hydrogens_sel=hydrogens_sel,
            acceptors_sel=acceptors_sel,
            d_h_cutoff=d_h_cutoff,
            d_a_cutoff=d_a_cutoff,
            d_h_a_angle_cutoff=d_h_a_angle_cutoff,
            update_selections=update_selections,
        )
        self.surf_ids = surf_ids
        if self.surf_ids is not None:
            self.surf_ids = np.array(self.surf_ids, dtype=np.int32)
        self.regions = np.array(regions)
        # TODO: sort self.region

    def _prepare(self):
        super(ParallelHBAnalysis, self)._prepare()

    def _single_frame(self, ts, atomgroups):
        u = atomgroups[0].universe
        box = ts.dimensions

        # Update donor-hydrogen pairs if necessary
        if self.update_selections:
            acceptors = u.select_atoms(self.acceptors_sel)
            donors, hydrogens = self._get_dh_pairs(u)
        else:
            acceptors = u.atoms[self._acceptors_ids]
            donors = u.atoms[self._donors_ids]
            hydrogens = u.atoms[self._hydrogens_ids]

        # ! >>>>> new code >>>>> !
        if self.regions is not None:
            donors_ids = donors.ids
            # get position of upper/lower surface
            if self.surf_ids is None:
                raise AttributeError(
                    "surf_ids should be specified when regions is not None"
                )
            surf_lo = u.atoms[self.surf_ids[0]]
            surf_hi = u.atoms[self.surf_ids[1]]
            z_surf = [
                np.mean(surf_lo.positions[:, 2]),
                np.mean(surf_hi.positions[:, 2]),
            ]
            # print(z_surf)
            sel_command = self._get_sel_command(z_surf, donors_ids)
            # print(sel_command)
            u_donors = donors.universe.select_atoms(sel_command, updating=True)
            donors = u_donors.atoms

            # print(donors.ids)
        # ! <<<<< new code <<<<< !

        # find D and A within cutoff distance of one another
        # min_cutoff = 1.0 as an atom cannot form a hydrogen bond with itself
        d_a_indices, d_a_distances = capped_distance(
            donors.positions,
            acceptors.positions,
            max_cutoff=self.d_a_cutoff,
            min_cutoff=1.0,
            box=box,
            return_distances=True,
        )

        # Remove D-A pairs more than d_a_cutoff away from one another
        tmp_donors = donors[d_a_indices.T[0]]
        tmp_hydrogens = hydrogens[d_a_indices.T[0]]
        tmp_acceptors = acceptors[d_a_indices.T[1]]

        # Find D-H-A angles greater than d_h_a_angle_cutoff
        d_h_a_angles = np.rad2deg(
            calc_angles(
                tmp_donors.positions,
                tmp_hydrogens.positions,
                tmp_acceptors.positions,
                box=box,
            )
        )
        hbond_indices = np.where(d_h_a_angles > self.d_h_a_angle)[0]

        # Retrieve atoms, distances and angles of hydrogen bonds
        hbond_donors = tmp_donors[hbond_indices]
        hbond_hydrogens = tmp_hydrogens[hbond_indices]
        hbond_acceptors = tmp_acceptors[hbond_indices]
        hbond_distances = d_a_distances[hbond_indices]
        hbond_angles = d_h_a_angles[hbond_indices]

        # Store data on hydrogen bonds found at this frame
        hbonds = [[], [], [], [], [], []]
        hbonds[0].extend(np.full_like(hbond_donors, ts.frame))
        hbonds[1].extend(hbond_donors.ids)
        hbonds[2].extend(hbond_hydrogens.ids)
        hbonds[3].extend(hbond_acceptors.ids)
        hbonds[4].extend(hbond_distances)
        hbonds[5].extend(hbond_angles)
        return np.asarray(hbonds).T

    def _conclude(self):
        super(ParallelHBAnalysis, self)._conclude()

    def _get_sel_command(self, z_surf, donors_ids):
        abs_regions = self._get_abs_regions(z_surf)
        # print(abs_regions)
        sel_command = ""
        for region in abs_regions:
            sel_command = (
                sel_command
                + "(prop z >="
                + str(region[0])
                + " and prop z < "
                + str(region[1])
                + ")"
                + " or "
            )
        sel_command = "(" + sel_command[:-4] + ") and ("
        for idx in donors_ids:
            sel_command = sel_command + "index " + str(idx) + " or "
        sel_command = sel_command[:-4] + ")"
        return sel_command

    def _get_abs_regions(self, z_surf):
        abs_lo_regions = self.regions.copy() + z_surf[0]
        abs_hi_regions = z_surf[1] - self.regions.copy()
        tmp = abs_hi_regions.copy()[:, 0]
        abs_hi_regions[:, 0] = abs_hi_regions[:, 1]
        abs_hi_regions[:, 1] = tmp
        abs_regions = np.concatenate((abs_lo_regions, abs_hi_regions), axis=0)
        return abs_regions
