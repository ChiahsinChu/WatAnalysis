import os
from scipy import stats
import numpy as np
from MDAnalysis.transformations import translate, wrap
from MDAnalysis.analysis.base import AnalysisBase

from mdadist.distances import calc_bonds_vector, distance_array

NA = 6.02214076E+23
ANG_TO_CM = 1E-08


def get_com_pos(universe, elem_type):
    elem_ag = universe.select_atoms("name " + elem_type)
    p = elem_ag.positions.copy()
    ti_com_z = p.mean(axis=0)[-1]
    return ti_com_z


def center(universe, elem_type, cell=None):
    """
    TBC
    """
    # check/set dimensions of universe
    if universe.dimensions is None:
        if cell is None:
            raise AttributeError('Cell parameter required!')
        else:
            universe.dimensions = cell
    # translate and wrap
    metal_com_z = get_com_pos(universe, elem_type)
    workflow = [translate([0, 0, -metal_com_z]), wrap(universe.atoms)]
    universe.trajectory.add_transformations(*workflow)
    #return universe

class Density1DAnalysis(AnalysisBase):
    """
    Parameters
    ----------
    universe: AtomGroup Object
        tbc
    water_sel: List or Array
        [a, b, c, alpha, beta, gamma]
    dim: int (2)
        
    delta: float (0.1)
        tbc
    """
    def __init__(self,
                 universe,
                 water_sel='name O',
                 dim=2, 
                 delta=0.1,
                 mass=18.015):
        super().__init__(universe.trajectory)
        self._cell = universe.dimensions
        self._dim = dim
        self._delta = delta
        self._mass = mass
        self._nwat = len(universe.select_atoms(water_sel))
        self._O_ids = universe.select_atoms(water_sel).indices

        # check cell
        if self._cell is None:
            raise AttributeError('Cell parameters should be set.')

        #parallel value initial
        self.para = None
        self._para_region = None


    def _prepare(self):
        # cross area along the selected dimension
        dims = self._cell[: 3]
        dims = np.delete(dims, self._dim)
        self.cross_area = dims[0] * dims[1] * np.sin(self._cell[self._dim + 3] / 180 * np.pi)
        # placeholder
        self.all_coords = np.zeros((self.n_frames, self._nwat),
                                    dtype=np.float32)


    def _single_frame(self):
        ts_coord = self._ts.positions[self._O_ids].T[self._dim]
        np.copyto(self.all_coords[self._frame_index], ts_coord)


    def _conclude(self):
        bins = np.arange(0, self._cell[self._dim], self._delta)
        grids, density = self._get_density(self.all_coords, bins)
        self.results = {}
        self.results['grids'] = grids
        self.results['density'] = density

    def _get_density(self, coords, bins):
        density, grids = np.histogram(coords, bins=bins)
        grids = grids[:-1] + self._delta / 2
        density = (density / NA * self._mass) / (
            self.cross_area * self._delta * ANG_TO_CM**3) / self.n_frames
        return grids, density
        


    def _parallel_init(self, *args, **kwargs):
    
        start = self._para_region.start
        stop = self._para_region.stop
        step = self._para_region.step
        self._setup_frames(self._trajectory, start, stop, step)
        self._prepare()


    def run(self, start=None, stop=None, step=None, verbose=None):
        if verbose == True:
            print(" ", end='')
        super().run(start, stop, step, verbose)

        if self.para:
            block_result = self._para_block_result()
            if block_result == None:
                raise ValueError(
                    "in parallel, block result has not been defined or no data output!"
                )
            return block_result


    def to_file(self, output_file):
        output = np.concatenate(([self.results['grids']], [self.results['density']]), axis=0)
        output = np.transpose(output)
        if os.path.splitext(output_file)[-1][1:] == "npy":
            np.save(output_file, output)
        else:
            np.savetxt(output_file, output)


    def _para_block_result(self, ):
        return self.results


    def _parallel_conclude(self, rawdata):
        method_attr = rawdata[-1]
        del rawdata[-1]
        self.start = method_attr[0]
        self.stop = method_attr[1]
        self.step = method_attr[2]
        self.frames = np.arange(self.start, self.stop, self.step)

        self.results = {}
        density = []
        for single_data in rawdata:
            density.append(single_data['density'])
        density = np.mean(density, axis=0)
        self.results['grids'] = single_data['grids']
        self.results['density']
        return "FINISH PARA CONCLUDE"


class InterfaceWatDensity(Density1DAnalysis):
    """
    TBC
    """
    def __init__(self, universe, water_sel='name O', dim=2, delta=0.1, **kwargs):
        super().__init__(universe, water_sel, dim, delta, mass=18.015)
        
        self._surf_ids = kwargs.get('surf_ids', None)
        if self._surf_ids is None:
            # if no surf ids provided
            slab_sel = kwargs.get('slab_sel', None)
            self._surf_natoms = kwargs.get('surf_natoms', None)
            if slab_sel is not None and self._surf_natoms is not None:
                self._slab_ids = universe.select_atoms(slab_sel).indices
            else:
                raise AttributeError('slab_sel and surf_natoms should be provided in the absence of surf_ids')


    def _prepare(self):
        super()._prepare()
        # placeholder for surface coords
        self.surf_coords = np.zeros((self.n_frames, 2), dtype=np.float32)
        #print(self.surf_ids)
    

    def _single_frame(self):
        # save surface coords
        ts_surf_lo = self._ts.positions[self.surf_ids[0]].T[self._dim]
        ts_surf_hi = self._ts.positions[self.surf_ids[1]].T[self._dim]
        ts_surf_region = self._get_surf_region(ts_surf_lo, ts_surf_hi)
        np.copyto(self.surf_coords[self._frame_index], ts_surf_region)

        # save water coords w.r.t. lower surfaces
        _ts_coord = self._ts.positions[self._O_ids].T[self._dim]
        ts_coord = self._ref_water(_ts_coord, ts_surf_region[0])
        np.copyto(self.all_coords[self._frame_index], ts_coord)


    def _conclude(self):
        surf_lo_ave = self.surf_coords[:, 0].mean(axis=0)
        surf_hi_ave = self.surf_coords[:, 1].mean(axis=0)
        self._surf_space = surf_hi_ave - surf_lo_ave
        bins = np.arange(0, (self._surf_space + self._delta), self._delta)
        
        grids, density = self._get_density(self.all_coords, bins)
        self.results = {}
        self.results['grids'] = grids[:len(grids) // 2]
        self.results['density'] = (density + density[np.arange(len(density)-1, -1, -1)])[:len(grids) // 2] / 2


    def _parallel_conclude(self, rawdata):
        method_attr = rawdata[-1]
        del rawdata[-1]
        self.start = method_attr[0]
        self.stop = method_attr[1]
        self.step = method_attr[2]
        self.frames = np.arange(self.start, self.stop, self.step)

        n_grids = 0
        for single_data in rawdata:
            if n_grids == 0 or len(single_data['grids']) < n_grids:
                n_grids = len(single_data['grids'])
        
        _density = []
        _grids = []
        for single_data in rawdata:
            _density.append(single_data['density'][: n_grids])
            _grids.append(single_data['grids'][: n_grids])
        self.results['grids'] = np.mean(_grids, axis=0) 
        self.results['density'] = np.mean(_density, axis=0)

        return "FINISH PARA CONCLUDE"

    
    @property
    def surf_ids(self):
        """
        Get the indices of surface atoms
        """
        if self._surf_ids is None:
            from ase import Atoms
            slab = Atoms("H" + str(len(self._slab_ids)))
            slab.set_cell(self._cell)
            slab.set_pbc(True)
            pos = self._trajectory[0].positions[self._slab_ids]
            slab.set_positions(pos)
            slab.center(about=slab[0].position)
            slab.wrap()
            slab.center()

            coords = np.array([slab.positions[:, self._dim]])
            slab_ids = np.atleast_2d(self._slab_ids)
            data = np.concatenate((slab_ids, coords), axis=0)
            data = np.transpose(data)

            # sort from small coord to large coord
            data = data[data[:, 1].argsort()]
            upper_slab_ids = data[:self._surf_natoms][:, 0]
            upper_slab_ids.sort()
            lower_slab_ids = data[-self._surf_natoms:][:, 0]
            lower_slab_ids.sort()
            self._surf_ids = np.array([lower_slab_ids, upper_slab_ids], dtype=int)
        return self._surf_ids


    def _get_surf_region(self, _coord_lo, _coord_hi):
        """
        Return:
            numpy array of (coord_lo, coord_hi)
            coord_lo: average coord for lower surface
            coord_hi: average coord for upper surface
        """
        # fold all _coord_lo about _coord_lo.min()
        tmp = _coord_lo.min()
        for coord in _coord_lo:
            coord = coord + np.floor((tmp - coord) / self._cell[self._dim]) * self._cell[self._dim]
        coord_lo = np.mean(_coord_lo)
        # fold all _coord_hi about _coord_hi.max()
        tmp = _coord_hi.max()
        for coord in _coord_hi:
            coord = coord + np.floor((tmp - coord) / self._cell[self._dim]) * self._cell[self._dim]
        coord_hi = np.mean(_coord_hi)

        if coord_hi < coord_lo:
            coord_hi = coord_hi + self._cell[self._dim]

        return np.array([coord_lo, coord_hi], dtype=float)


    def _ref_water(self, water_coords, surf_lo):
        """
        water coord w.r.t. lower surfaces
        """
        for coord in water_coords:
            while coord < surf_lo:
                coord = coord + self._cell[self._dim]
        water_coords = water_coords - surf_lo
        return water_coords


class InterfaceWatOri(InterfaceWatDensity):
    """
    TBC
    """
    def __init__(self, universe, O_sel='name O', H_sel='name H', dim=2, delta=0.1, update_pairs=False, **kwargs):
        super().__init__(universe, O_sel, dim, delta, surf_ids=kwargs.get('surf_ids', None), slab_sel=kwargs.get('slab_sel', None), surf_natoms=kwargs.get('surf_natoms', None))
        self._H_ids = universe.select_atoms(H_sel).indices
        if len(self._H_ids) != self._nwat * 2:
            raise AttributeError('Only pure water has been supported yet.')
        self._update_pairs = update_pairs
        self.pairs = kwargs.get('pairs', None)
        if self.pairs is None:
            self._get_pairs(self._trajectory[0].positions[self._O_ids], self._trajectory[0].positions[self._H_ids])


    def _prepare(self):
        super()._prepare()
        self.all_oris = np.zeros((self.n_frames, self._nwat),
                                    dtype=np.float32)


    def _single_frame(self):
        super()._single_frame()
        ts_O_coord = self._ts.positions[self._O_ids]
        ts_H_coord = self._ts.positions[self._H_ids]
        if self._update_pairs:
            self._get_pairs(ts_O_coord, ts_H_coord)
        ts_water_oris = self._get_water_ori(ts_O_coord, ts_H_coord)
        np.copyto(self.all_oris[self._frame_index], ts_water_oris)


    def _conclude(self):
        super()._conclude()
        all_coords = np.reshape(self.all_coords, -1)
        all_oris = np.reshape(self.all_oris, -1)
        bins = np.arange(0, (self._surf_space + self._delta), self._delta)
        water_cos, bin_edges, binnumber = stats.binned_statistic(x=all_coords, value=all_oris, bins=bins)
        water_cos = (water_cos - water_cos[np.arange(len(water_cos)-1, -1, -1)])[:len(water_cos) // 2] / 2
        self.results['ori_dipole'] = water_cos * self.results['density']

    def _parallel_conclude(self, rawdata):
        method_attr = rawdata[-1]
        del rawdata[-1]
        self.start = method_attr[0]
        self.stop = method_attr[1]
        self.step = method_attr[2]
        self.frames = np.arange(self.start, self.stop, self.step)

        n_grids = 0
        for single_data in rawdata:
            if n_grids == 0 or len(single_data['grids']) < n_grids:
                n_grids = len(single_data['grids'])
        
        _density = []
        _grids = []
        _ori_dipole = []
        for single_data in rawdata:
            _density.append(single_data['density'][: n_grids])
            _grids.append(single_data['grids'][: n_grids])
            _ori_dipole.append(single_data['ori_dipole'][: n_grids])
        self.results['grids'] = np.mean(_grids, axis=0) 
        self.results['density'] = np.mean(_density, axis=0)
        self.results['ori_dipole'] = np.mean(_ori_dipole, axis=0)

        return "FINISH PARA CONCLUDE"


    def _get_water_ori(self, O_coord, H_coord):
        """
        TBC
        """
        water_oris = np.zeros((self._nwat, 3), dtype=np.float32)
        for O_id, H_ids in enumerate(self.pairs):
            tmp_vec = np.empty((2, 3))
            calc_bonds_vector(O_coord[O_id],
                              H_coord[H_ids],
                              box=self._cell,
                              result=tmp_vec)
            tmp_vec = np.mean(tmp_vec, axis=0)
            np.copyto(water_oris[O_id], tmp_vec[self._dim] / np.linalg.norm(tmp_vec))
        return water_oris

    def _get_pairs(self, O_coords, H_coords):
        all_distances = np.empty((self._nwat, self._nwat * 2), dtype=float)
        distance_array(O_coords, H_coords, box=self._cell, result=all_distances)
        #self.pairs = self._H_ids[np.argsort(all_distances, axis=-1)[:, :2]]
        self.pairs = np.argsort(all_distances, axis=-1)[:, :2]
