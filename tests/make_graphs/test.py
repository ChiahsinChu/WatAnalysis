import numpy as np

from WatAnalysis.hbonds import postprocess


data = np.load("../PHBA/hbonds.npy")
postprocess.get_graphs(data, "./output")
