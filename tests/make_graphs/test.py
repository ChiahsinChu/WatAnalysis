# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np

from WatAnalysis.hbonds import postprocess

data = np.load("../PHBA/hbonds.npy")
postprocess.get_graphs(data, "./output")
