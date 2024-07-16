# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
from ase import io

data = np.load("./hbonds.npy")[:76]
sel_ids = np.unique(data[:, 1:3])
atoms = io.read("../input_data/coord.xyz")
for atom in atoms:
    if atom.index in sel_ids:
        if atom.symbol == "O":
            if atom.position[2] > 20:
                atom.symbol = "S"
            else:
                atom.symbol = "Se"
        else:
            if atom.position[2] > 20:
                atom.symbol = "Li"
            else:
                atom.symbol = "Na"
print(atoms)
io.write("./coord.xyz", atoms)
