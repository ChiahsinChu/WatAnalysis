import numpy as np
from ase import Atoms, Atom
from statsmodels.tsa import stattools
import logging

logger = logging.getLogger(__name__)

from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.transformations import wrap

# TODO: define soft interface
# TODO: water position relative to soft interface
# TODO: time-correlation function (Fig. 2b)
# TODO: instanton time (water desorption) (Fig. 2c)
"""

from MDAnalysis.transformations import translate

def get_com_pos(ag):

    elements = ag.elements.copy()
    xyz = ag.positions.copy()
    sel_element = (elements=='Sn')
    ti_com_z = xyz[sel_element].mean(axis=0)[-1]

    return ti_com_z

uni.dimensions = cell
# translate and wrap
ag = uni.atoms
metal_com_z = get_com_pos(ag)
workflow = [translate([0,0, -metal_com_z]), wrap(ag)]
uni.trajectory.add_transformations(*workflow)
"""


class ECAnalysis(AnalysisBase):

    def __init__(self, atomgroup, verbose=True):
        trajectory = atomgroup.universe.trajectory
        super(ECAnalysis, self).__init__(trajectory, verbose=verbose)

        # trajectory value initial
        self.ag = atomgroup
        self.ag_natoms = len(self.ag)
        self.n_frames = len(trajectory)

        #parallel value initial
        self.para = None
        self._para_region = None
        #MORE NEED TO CORRECT

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

    def _para_block_result(self):
        # data need to be transformed, which is usually relative to values in prepare() method.

        return None

    def _parallel_conclude(self, rawdata):

        raise NotImplementedError("ECAnalysis is father class")


class WatAdsCCF(ECAnalysis):
    """
    Inherit from MDAnalysis.analysis.base.AnalysisBase Class
    Calculate dynamic indicator functions of ALL water molecules
    """

    def __init__(self,
                 atomgroup,
                 cell,
                 slab_idx,
                 surf_natoms,
                 water_idx,
                 boundary=[2.7, 4.5],
                 verbose=True):
        ECAnalysis.__init__(self, atomgroup=atomgroup, verbose=verbose)

        self.cell = cell
        self.slab_idx = slab_idx
        self.surf_natoms = surf_natoms
        self.surf_natoms = int(self.surf_natoms)
        self.water_idx = water_idx
        self.boundary = boundary

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

        # wrap
        # TODO: add wrap cell
        """
        for timestep in self.ag.universe.trajectory: 
            timestep.dimensions = self.cellpar.tolist()
        for ts in self.ag.universe.trajectory:
            self.ag.wrap(compound='atoms')
        """

        # get surface index
        self._get_surf_idx(self.slab_idx, self.surf_natoms)

        # placeholder for surface ave z positions
        self.surf_z_ave_each = np.zeros((self.n_frames, 2), dtype=np.float32)
        # placeholder for water h
        self.all_water_h = np.zeros((self.n_frames, len(self.water_idx)),
                                    dtype=np.float32)
        # placeholder for water g
        self.all_water_g = np.zeros((self.n_frames, len(self.water_idx)),
                                    dtype=np.float32)
        # placeholder for CCF
        self.all_water_ccf = np.zeros((len(self.water_idx), self.n_frames),
                                      dtype=np.float32)

    def _single_frame(self):
        """
        Get data in a single frame

        Args:
            ts_surf_zlo: list of atomic idx at bottom surface
            ts_surf_zhi: list of atomic idx at top surface 
            ts_surf_region: [ave z of bottom surface, ave z of top surface]
            ts_water_z: z of water refered to bottom surface
        """
        ts_surf_zlo = self._ts.positions.T[2][self.surf_idx[0]]
        ts_surf_zhi = self._ts.positions.T[2][self.surf_idx[1]]
        ts_surf_region = self._get_surf_region(ts_surf_zlo, ts_surf_zhi)
        np.copyto(self.surf_z_ave_each[self._frame_index], ts_surf_region)

        ts_water_z = self._ts.positions.T[2][self.water_idx]
        # metal-water-metal in z direction
        ts_water_z = self._ref_water(ts_water_z, ts_surf_region[0])
        for idx, z in enumerate(ts_water_z):
            d_lo = np.abs(z - ts_surf_zlo)
            d_hi = np.abs(z - ts_surf_zhi)
            d_to_surf = np.min([d_lo, d_hi])
            if d_to_surf < self.boundary[0]:
                # adsorption state
                self.all_water_h[self._frame_index][idx] = 1.0
            elif d_to_surf > self.boundary[1]:
                # bulk water
                self.all_water_g[self._frame_index][idx] = 1.0

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
        #_z_lo = self.ag.universe.trajectory[0].positions[surf_idx[0],2]
        tmp = _z_lo.min()
        for z in _z_lo:
            z = z + np.floor((tmp - z) / self.cellpar[2]) * self.cellpar[2]
        z_lo = np.mean(_z_lo)

        #_z_hi = self.ag.universe.trajectory[0].positions[surf_idx[1],2]
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


class WaterZDensityAnal(ECAnalysis):
    """
    Inherit from MDAnalysis.analysis.base.AnalysisBase Class
    Calculate water density along z direction
    """

    def __init__(self,
                 atomgroup,
                 cell,
                 dz,
                 slab_idx,
                 surf_natoms,
                 water_idx,
                 verbose=True):
        """
        atomgroup: MDA Universe.atoms object
        """
        ECAnalysis.__init__(self, atomgroup=atomgroup, verbose=verbose)
        #super(WaterZDensityAnal, self).__init__(atomgroup, verbose=verbose)

        self.cell = cell
        self.dz = dz
        self.slab_idx = slab_idx
        self.surf_natoms = surf_natoms
        self.surf_natoms = int(self.surf_natoms)
        self.water_idx = water_idx

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
        a = self.cell[0]
        b = self.cell[1]
        c = self.cell[2]
        self.cell_volume = np.dot(np.cross(a, b), c)
        self.xy_area = np.cross(a, b)[2]

        # wrap
        # TODO: add wrap cell
        """
        for timestep in self.ag.universe.trajectory: 
            timestep.dimensions = self.cellpar.tolist()
        for ts in self.ag.universe.trajectory:
            self.ag.wrap(compound='atoms')
        """

        # get surface index
        self._get_surf_idx(self.slab_idx, self.surf_natoms)

        # placeholder for water z
        self.all_water_z = np.zeros((self.n_frames, len(self.water_idx)),
                                    dtype=np.float32)
        # placeholder for surface ave z positions
        self.surf_z_ave_each = np.zeros((self.n_frames, 2), dtype=np.float32)

    def _single_frame(self):
        """
        Get data in a single frame

        Args:
            ts_surf_zlo: list of atomic idx at bottom surface
            ts_surf_zhi: list of atomic idx at top surface 
            ts_surf_region: [ave z of bottom surface, ave z of top surface]
            ts_water_z: z of water refered to bottom surface
        """
        ts_surf_zlo = self._ts.positions.T[2][self.surf_idx[0]]
        ts_surf_zhi = self._ts.positions.T[2][self.surf_idx[1]]
        ts_surf_region = self._get_surf_region(ts_surf_zlo, ts_surf_zhi)
        np.copyto(self.surf_z_ave_each[self._frame_index], ts_surf_region)

        ts_O_coord = self._ts.positions[self.water_idx]
        ts_water_z = self._ref_water(ts_O_coord.T[2], ts_surf_region[0])
        np.copyto(self.all_water_z[self._frame_index], ts_water_z)

    def _conclude(self):

        #self.ag.universe._trajectory._reopen()

        self.surf_zlo_ave = self.surf_z_ave_each[:, 0].mean(axis=0)
        self.surf_zhi_ave = self.surf_z_ave_each[:, 1].mean(axis=0)
        self.surf_space = self.surf_zhi_ave - self.surf_zlo_ave

        self.density, z = np.histogram(self.all_water_z,
                                       bins=np.arange(0, self.surf_space,
                                                      self.dz))
        self.final_z = z[:-1] + self.dz / 2
        self.final_density = (self.density / NA * 18.015) / (
            self.xy_area * self.dz * ANG_TO_CM**3) / self.n_frames

        self.results = (self.final_z, self.final_density)

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
        #_z_lo = self.ag.universe.trajectory[0].positions[surf_idx[0],2]
        tmp = _z_lo.min()
        for z in _z_lo:
            z = z + np.floor((tmp - z) / self.cellpar[2]) * self.cellpar[2]
        z_lo = np.mean(_z_lo)

        #_z_hi = self.ag.universe.trajectory[0].positions[surf_idx[1],2]
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

    def _para_block_result(self):

        #data responding to _parallel_conclude rawdata list
        return (list(self.all_water_z), list(self.surf_z_ave_each))

    def _parallel_conclude(self, rawdata, *args, **kwargs):
        # for map results process, might be static
        # without it, the parallel_exec will be interrupt.

        self._parallel_init()
        self.all_water_z = np.concatenate(
            [block_result[0] for block_result in rawdata])
        self.surf_z_ave_each = np.concatenate(
            [block_result[1] for block_result in rawdata])
        self._conclude()
        return "FINISH PARA CONCLUDE"

    def interval_optimize(self, new_dz):
        dz = new_dz

        density, z = np.histogram(self.all_water_z,
                                  bins=np.arange(0, self.surf_space, dz))
        final_z = z[:-1] + dz / 2
        final_density = density / (self.xy_area *
                                   dz) / self.n_frames / self.bulk_density

        return final_z, final_density


class WaterZOriAnal(WaterZDensityAnal):
    """
    Inherit from MDAnalysis.analysis.base.AnalysisBase Class
    Calculate water orientational dipole along z direction
    """

    def __init__(self,
                 atomgroup,
                 cell,
                 dz,
                 slab_idx,
                 surf_natoms,
                 water_idx,
                 H_idx,
                 verbose=True,
                 rearrange=False):
        """
        atomgroup: MDA Universe.atoms object
        """
        WaterZDensityAnal.__init__(self,
                                   atomgroup=atomgroup,
                                   cell=cell,
                                   dz=dz,
                                   slab_idx=slab_idx,
                                   surf_natoms=surf_natoms,
                                   water_idx=water_idx,
                                   verbose=verbose)

        self.H_idx = H_idx
        self.rearrange = rearrange

    def _prepare(self):
        super(WaterZOriAnal, self)._prepare()

        # placeholder for water dipole
        self.all_water_ori = np.zeros((self.n_frames, len(self.water_idx)),
                                      dtype=np.float32)

    def _single_frame(self):
        super(WaterZOriAnal, self)._single_frame()

        ts_O_coord = self._ts.positions[self.water_idx]
        ts_H_coord = self._ts.positions[self.H_idx]
        ts_water_ori = self._get_water_ori(ts_O_coord, ts_H_coord)
        np.copyto(self.all_water_ori[self._frame_index], ts_water_ori)

    def _conclude(self):
        super(WaterZOriAnal, self)._conclude()

        all_water_z = np.reshape(self.all_water_z, [1, -1])
        all_water_ori = np.reshape(self.all_water_ori, [1, -1])
        input_data = np.concatenate(([all_water_z, all_water_ori]), axis=0)
        input_data = np.transpose(input_data)
        output = common.get_1d_distribution(input_data,
                                            bins=np.arange(
                                                0, self.surf_space, self.dz))
        self.final_ori = output[1].copy()
        for i in range(int(len(self.final_ori) / 2), len(self.final_ori)):
            self.final_ori[i] = -self.final_ori[i]
        self.final_ori = self.final_ori * self.final_density
        self.results = (self.final_z, self.final_density, self.final_ori)

    def _get_water_ori(self, O_coord, H_coord):
        """
        TBC
        """
        if self.rearrange == False:
            if 2 * len(O_coord) != len(H_coord):
                raise AttributeError("Mismatched number of O and H.")
            water_ori = np.zeros((len(O_coord)), dtype=np.float32)
            for idx, item in enumerate(O_coord):
                p_oxygen = item.astype(np.float32)
                p_hydrogen_0 = H_coord[idx * 2]
                p_hydrogen_1 = H_coord[idx * 2 + 1]
                vec_OH_0 = np.empty((1, 3))
                calc_bonds_vector(p_hydrogen_0,
                                  p_oxygen,
                                  box=self.cellpar,
                                  result=vec_OH_0)
                vec_OH_1 = np.empty((1, 3))
                calc_bonds_vector(p_hydrogen_1,
                                  p_oxygen,
                                  box=self.cellpar,
                                  result=vec_OH_1)
                vec_OH = np.reshape((vec_OH_0 + vec_OH_1), [-1])
                water_ori[idx] = vec_OH[2] / np.linalg.norm(vec_OH)
            return water_ori
        else:
            pass

    def _find_center(self, atom_idx, center_idx_list):
        """
        TBC
        """
        D, D_len = self.get_distances(atom_idx, center_idx_list, mic=True)
        center_idx = center_idx_list[np.argmin(D_len)]
        bond_vec = -D[np.argmin(D_len)]
        return center_idx, bond_vec

    def get_pair_dict(self, mode="full"):
        """
        mode:
            mode to find water molecules
            *** simple/simple_mic/rearrange/rearrange_mic are only ***
            *** available for the model without proton hopping!    ***

            "simple": O-H-H, mic=False when calculate d_{O-H}
            "simple_mic": O-H-H, mic=True when calculate d_{O-H} 
            "rearrange": O... and H..., mic=False when calculate d_{O-H} 
            "rearrange_mic": O... and H..., mic=True when calculate d_{O-H} 
            "full": find water with voronoi polyhedra (available for the systems
            with proton hopping)
        """
        if mode == "simple":
            O_idx = self.water_idx
            pair_dict = init_pair_dict(O_idx)
            for idx in O_idx:
                pair_dict[idx]["idx"].append(idx + 1)
                bond_vec, D_len = self.get_distance(idx, idx + 1)
                pair_dict[idx]["vec"].append(bond_vec)
                pair_dict[idx]["idx"].append(idx + 2)
                bond_vec, D_len = self.get_distance(idx, idx + 2)
                pair_dict[idx]["vec"].append(bond_vec)
            return pair_dict
        elif mode == "simple_mic":
            cell_pbc = self.get_pbc()
            self.set_pbc([True, True, False])
            O_idx = self.water_idx
            pair_dict = init_pair_dict(O_idx)
            for idx in O_idx:
                pair_dict[idx]["idx"].append(idx + 1)
                bond_vec, D_len = self.get_distance(idx, idx + 1, mic=True)
                pair_dict[idx]["vec"].append(bond_vec)
                pair_dict[idx]["idx"].append(idx + 2)
                bond_vec, D_len = self.get_distance(idx, idx + 2, mic=True)
                pair_dict[idx]["vec"].append(bond_vec)
            self.set_pbc(cell_pbc)
            return pair_dict
        elif mode == "rearrange":
            H_idx = common.get_elem_idxs(self, "H")
            O_idx = common.get_elem_idxs(self, "O")
            pair_dict = init_pair_dict(O_idx)
            for idx in O_idx:
                pair_dict[idx]["idx"].append(H_idx[2 * idx])
                bond_vec, D_len = self.get_distance(idx, H_idx[2 * idx])
                pair_dict[idx]["vec"].append(bond_vec)
                pair_dict[idx]["idx"].append(H_idx[2 * idx + 1])
                bond_vec, D_len = self.get_distance(idx, H_idx[2 * idx + 1])
                pair_dict[idx]["vec"].append(bond_vec)
            return pair_dict
        elif mode == "rearrange_mic":
            cell_pbc = self.get_pbc()
            self.set_pbc([True, True, False])
            H_idx = common.get_elem_idxs(self, "H")
            O_idx = common.get_elem_idxs(self, "O")
            pair_dict = init_pair_dict(O_idx)
            for idx in O_idx:
                pair_dict[idx]["idx"].append(H_idx[2 * idx])
                bond_vec, D_len = self.get_distance(idx,
                                                    H_idx[2 * idx],
                                                    mic=True)
                pair_dict[idx]["vec"].append(bond_vec)
                pair_dict[idx]["idx"].append(H_idx[2 * idx + 1])
                bond_vec, D_len = self.get_distance(idx,
                                                    H_idx[2 * idx + 1],
                                                    mic=True)
                pair_dict[idx]["vec"].append(bond_vec)
            self.set_pbc(cell_pbc)
            return pair_dict
        elif mode == "full":
            cell_pbc = self.get_pbc()
            self.set_pbc([True, True, False])
            H_idx = common.get_elem_idxs(self, "H")
            O_idx = common.get_elem_idxs(self, "O")
            pair_dict = init_pair_dict(O_idx)
            for idx in H_idx:
                center_idx, bond_vec = self.find_center(idx, O_idx)
                pair_dict[center_idx]["idx"].append(idx)
                pair_dict[center_idx]["vec"].append(bond_vec)
            self.set_pbc(cell_pbc)
            return pair_dict
        else:
            raise AttributeError(
                "Supported mode: simple/simple_mic/rearrange/rearrange_mic/full."
            )

    def get_wrapped_waterbox(self, mode="full"):
        """
        TBC
        Args:
            mode: mode in self.get_pair_dict(mode)
        """
        atoms = self.copy()
        waterbox = Atoms()
        pair_dict = self.get_pair_dict(mode=mode)
        atoms.wrap()
        cos_data = []

        if mode == "full":
            for idx in self.water_idx:
                if len(pair_dict[idx]["idx"]) == 2:
                    # water molecule
                    # O-H-H
                    atom = Atom(symbol="O", position=atoms[idx].position)
                    waterbox.extend(atom)
                    p = atoms[idx].position + np.array(
                        pair_dict[idx]["vec"][0])
                    atom = Atom(symbol="H", position=p)
                    waterbox.extend(atom)
                    p = atoms[idx].position + np.array(
                        pair_dict[idx]["vec"][1])
                    atom = Atom(symbol="H", position=p)
                    waterbox.extend(atom)
                    dip_vec = np.array(pair_dict[idx]["vec"][0]) + np.array(
                        pair_dict[idx]["vec"][1])
                    cos_data.append([
                        atoms[idx].position[2],
                        dip_vec[2] / np.linalg.norm(dip_vec)
                    ])
                else:
                    # other than water
                    continue
            if len(waterbox) == 3 * len(self.water_idx):
                self.waterbox = waterbox
                return np.array(cos_data, dtype=float)
            else:
                self.waterbox = None
                return None
        else:
            for k, v in pair_dict.items():
                atom = Atom(symbol="O", position=atoms[k].position)
                waterbox.extend(atom)
                p = atoms[k].position + np.array(v["vec"][0])
                atom = Atom(symbol="H", position=p)
                waterbox.extend(atom)
                p = atoms[k].position + np.array(v["vec"][1])
                atom = Atom(symbol="H", position=p)
                waterbox.extend(atom)
                dip_vec = np.array(v["vec"][0]) + np.array(v["vec"][1])
                cos_data.append([
                    atoms[k].position[2], dip_vec[2] / np.linalg.norm(dip_vec)
                ])
            self.waterbox = waterbox
            return np.array(cos_data, dtype=float)

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
        #_z_lo = self.ag.universe.trajectory[0].positions[surf_idx[0],2]
        tmp = _z_lo.min()
        for z in _z_lo:
            z = z + np.floor((tmp - z) / self.cellpar[2]) * self.cellpar[2]
        z_lo = np.mean(_z_lo)

        #_z_hi = self.ag.universe.trajectory[0].positions[surf_idx[1],2]
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


class MetalPotDrop(ECAnalysis):
    """
    Inherit from MDAnalysis.analysis.base.AnalysisBase Class
    Calculate potential drop along z direction
    """

    def __init__(self, atomgroup, cell, model, atype, verbose=True):
        """
        atomgroup: MDA Universe.atoms object
        """
        ECAnalysis.__init__(self, atomgroup=atomgroup, verbose=verbose)

        self.cell = cell
        self.model = DeepPotDrop(model)
        self.atype = atype

    def _prepare(self):
        """
        TBC
        """
        # get cell parameters
        from ase import Atoms

        ase_atoms = Atoms()
        ase_atoms.set_cell(self.cell)
        self.cell = ase_atoms.get_cell()

        # placeholder of output
        self.all_potdrop = np.zeros(self.n_frames, dtype=np.float32)

    def _single_frame(self):
        """
        Get potential drop in a single frame

        Args:
            ts_coord: coord of snapshot
            ts_potdrop: potential drop predicted by DP model 
        """
        ts_coord = self._ts.positions
        ts_coord = np.reshape(ts_coord, -1)
        ts_potdrop = self.model.eval(ts_coord, self.cell, self.atype)
        ts_potdrop = np.reshape(ts_potdrop, -1)
        self.all_potdrop[self._frame_index] = ts_potdrop[0]
        print("i =", self._frame_index)

    def _conclude(self):
        self.results = self.all_potdrop
