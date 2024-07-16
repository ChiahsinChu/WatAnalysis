# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
from watdyn.basic import ECAnalysis


def ParallelHBAnalysis(args):
    """
    See pmda
    """
    pass


# deprecated
class HBLifeTime(ECAnalysis):
    """
    Inherit from MDAnalysis.analysis.base.AnalysisBase Class
    Calculate average HB life time
    DOI: 10.1021/acs.jpcc.6b07504
    """

    def __init__(self, atomgroup, cell, O_idx, H_idx, verbose=True):
        ECAnalysis.__init__(self, atomgroup=atomgroup, verbose=verbose)

        self.cell = cell
        self.O_idx = O_idx
        self.H_idx = H_idx
        if len(O_idx) * 2 != len(H_idx):
            raise AttributeError("Only support pure water box!")

    def _prepare(self):
        """
        Get necessary parameters for further analysis

        all_water_z:
        surf_z_ave_each:
        cell_volume:
        xy_area:
        """

        # get cell parameters
        from ase import Atoms

        ase_atoms = Atoms()
        ase_atoms.set_cell(self.cell)
        self.cell = ase_atoms.get_cell()
        self.cellpar = ase_atoms.cell.cellpar()

        # placeholder for characteristic function between i and j
        self.all_water_h = np.zeros(
            (self.n_frames, len(self.O_idx), len(self.O_idx)), dtype=np.float32
        )

    def _single_frame(self):
        """
        Get data in a single frame

        Args:
            ts_water_p: coord of water
        """

        ts_water_p = self._ts.positions.T[self.O_idx]

    def _conclude(self):
        self.surf_zlo_ave = self.surf_z_ave_each[:, 0].mean(axis=0)
        self.surf_zhi_ave = self.surf_z_ave_each[:, 1].mean(axis=0)

        # cross-correlation function
        self.all_water_h = np.transpose(self.all_water_h)
        self.all_water_g = np.transpose(self.all_water_g)
        for wat_idx in range(len(self.water_idx)):
            wat_h = self.all_water_h[wat_idx]
            wat_g = self.all_water_g[wat_idx]
            output = stattools.ccf(wat_h, wat_g)
            np.copyto(self.all_water_ccf[wat_idx], output)
        self.water_ccf = np.mean(self.all_water_ccf, axis=0)
        self.water_ccf_min = np.min(self.all_water_ccf, axis=0)
        self.water_ccf_max = np.max(self.all_water_ccf, axis=0)
        self.results = (self.water_ccf, self.water_ccf_min, self.water_ccf_max)

    def _get_surf_idx(self, slab_idx, surf_natoms):
        """
        Get the indexes of atoms of surface

        Args:
            slab_idx: index list for slab atoms
            surf_natoms: num of atoms exposed on the surface
        Returns:
            >>> Consider electrolyte-center model is used! <<<
            upper_slab_idx: idx of atoms of upper surface
            lower_slab_idx: idx of atoms of lower surface
        """
        from ase import Atoms

        slab = Atoms("H" + str(len(slab_idx)))
        slab.set_cell(self.cell)
        slab.set_pbc(True)
        p = self.ag.universe.trajectory[0].positions[slab_idx]
        slab.set_positions(p)
        slab.center(about=slab[0].position)
        slab.wrap()
        slab.center()

        z_coords = np.array([slab.positions[:, 2]])
        slab_idx = np.atleast_2d(slab_idx)
        data = np.concatenate((slab_idx, z_coords), axis=0)
        data = np.transpose(data)

        # sort from small z to large z
        data = data[data[:, 1].argsort()]
        upper_slab_idx = data[:surf_natoms][:, 0]
        upper_slab_idx.sort()
        upper_slab_idx = np.array(upper_slab_idx, dtype=int)
        lower_slab_idx = data[-surf_natoms:][:, 0]
        lower_slab_idx.sort()
        lower_slab_idx = np.array(lower_slab_idx, dtype=int)
        surf_idx = np.array([lower_slab_idx, upper_slab_idx])
        self.surf_idx = surf_idx

    def _get_surf_region(self, _z_lo, _z_hi):
        """
        Return:
            numpy array of (z_lo, z_hi)
            z_lo: average z for lower surface
            z_hi: average z for upper surface
        """
        # fold all _z_lo about _z_lo.min()
        # _z_lo = self.ag.universe.trajectory[0].positions[surf_idx[0],2]
        tmp = _z_lo.min()
        for z in _z_lo:
            z = z + np.floor((tmp - z) / self.cellpar[2]) * self.cellpar[2]
        z_lo = np.mean(_z_lo)

        # _z_hi = self.ag.universe.trajectory[0].positions[surf_idx[1],2]
        tmp = _z_hi.max()
        for z in _z_hi:
            z = z + np.floor((tmp - z) / self.cellpar[2]) * self.cellpar[2]
        z_hi = np.mean(_z_hi)

        if z_hi < z_lo:
            z_hi = z_hi + self.cellpar[2]
        surf_region = np.array([z_lo, z_hi], dtype=float)
        return surf_region

    def _ref_water(self, water_z, surf_zlo):
        for z in water_z:
            if z < surf_zlo:
                z = z + self.cellpar[2]
        water_z = water_z - surf_zlo
        return water_z
