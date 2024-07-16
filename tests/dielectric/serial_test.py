# SPDX-License-Identifier: LGPL-3.0-or-later
import time

import MDAnalysis as mda
import numpy as np
from ase import io
from MDAnalysis import transformations as trans
from zjxpack.postprocess.metal import ECMetal

from WatAnalysis.dielectric import InverseDielectricConstant as IDC

dim = [11.246, 11.246, 35.94, 90, 90, 90]

atoms = io.read("/data/jxzhu/2022_leiden/02.nnp_validation/input/coord.xyz")
atoms.set_cell(dim)
atoms = ECMetal(atoms, metal_type="Pt", surf_atom_num=16)
surf_ids = atoms.get_surf_idx()
print(surf_ids)

# load trajectory
u = mda.Universe(
    "/data/jxzhu/2022_leiden/02.nnp_validation/input/interface.psf",
    "/data/jxzhu/2022_leiden/02.nnp_validation/input/aimd.xyz",
)
transform = trans.boxdimensions.set_dimensions(dim)
u.trajectory.add_transformations(transform)

start = time.time()
task = IDC(
    universe=u,
    bins=np.arange(10),
    axis="z",
    temperature=330,
    make_whole=False,
    serial=False,
    verbose=True,
    surf_ids=surf_ids,
    c_ag="name O",
    select_all=True,
)
task.run()
parallel = task.results.inveps
print(parallel)
print(task.results)
end = time.time()
print("Used time (parallel): ", end - start)

# start = time.time()
# task = IDC(
#     universe=u,
#     bins=np.arange(10),
#     axis="z",
#     temperature=330,
#     make_whole=False,
#     serial=True,
#     verbose=True,
#     surf_ids=surf_ids,
#     c_ag="name O",
#     select_all=True,
# )
# task.run()
# serial = task.results.inveps
# end = time.time()
# print("Used time (serial): ", end - start)

# if False in np.abs(parallel - serial) <= 1e-8:
#     print("Test Failed!")
# else:
#     print("Test Passed!")
