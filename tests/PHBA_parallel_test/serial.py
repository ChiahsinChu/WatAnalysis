import numpy as np
import time
import MDAnalysis as mda
from MDAnalysis import transformations as trans

from WatAnalysis.hbonds.m_mda import PartialHBAnalysis

# load trajectory
u = mda.Universe("../input_data/interface.psf", "../input_data/trajectory.xyz")
dim = [16.869, 16.869, 41.478, 90, 90, 120]
transform = trans.boxdimensions.set_dimensions(dim)
u.trajectory.add_transformations(transform)

start = time.time()
hbonds = PartialHBAnalysis(universe=u,
                           hb_region=[0, 4.5],
                           surf_ids=[np.arange(871, 906),
                                     np.arange(691, 726)],
                           donors_sel=None,
                           hydrogens_sel="name H",
                           acceptors_sel="name O",
                           d_a_cutoff=3.0,
                           angle_cutoff_type='d_h_a',
                           angle_cutoff=150,
                           update_selections=False)
hbonds.run()
print(hbonds.results.hbonds.shape)
fmt = "\nWork Completed! Used Time: {:.3f} seconds"
print(fmt.format(time.time() - start))