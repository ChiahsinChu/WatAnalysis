from tkinter.messagebox import NO
import numpy as np
from MDAnalysis.transformations import translate, wrap
from MDAnalysis.analysis.base import AnalysisBase


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
        self._atomgroup = universe.select_atoms(water_sel)
        self._cell = universe.dimensions
        self._dim = dim
        self._delta = delta
        self._mass = mass
        self._natoms = len(self._atomgroup)
        self._mask = self._atomgroup.indices

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
        self.all_coords = np.zeros((self.n_frames, self._natoms),
                                    dtype=np.float32)


    def _single_frame(self):
        ts_coord = self._ts.positions[self._mask].T[self._dim]
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
        _ts_coord = self._ts.positions[self._mask].T[self._dim]
        ts_coord = self._ref_water(_ts_coord, ts_surf_region[0])
        np.copyto(self.all_coords[self._frame_index], ts_coord)


    def _conclude(self):
        surf_lo_ave = self.surf_coords[:, 0].mean(axis=0)
        surf_hi_ave = self.surf_coords[:, 1].mean(axis=0)
        bins = np.arange(0, (surf_hi_ave - surf_lo_ave + self._delta), self._delta)
        
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



        