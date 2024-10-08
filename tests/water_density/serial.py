# SPDX-License-Identifier: LGPL-3.0-or-later
import time

import MDAnalysis as mda
import numpy as np
from MDAnalysis import transformations as trans

from WatAnalysis.base import InterfaceWatDensity

# load trajectory
u = mda.Universe("../input_data/interface.psf", "../input_data/trajectory.xyz")
dim = [16.869, 16.869, 41.478, 90, 90, 120]
transform = trans.boxdimensions.set_dimensions(dim)
u.trajectory.add_transformations(transform)

start = time.time()
density = InterfaceWatDensity(universe=u, slab_sel="name Pt", surf_natoms=36)
density.run()
fmt = "\nWork Completed! Used Time: {:.3f} seconds"
print(fmt.format(time.time() - start))

output = np.concatenate(
    ([density.results["grids"]], [density.results["density"]]), axis=0
)
np.save("./serial.npy", output)
