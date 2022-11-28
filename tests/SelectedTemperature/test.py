import numpy as np
import time
import MDAnalysis as mda
from MDAnalysis import transformations as trans

import matplotlib.pyplot as plt

from WatAnalysis.temp import SelectedTemperature as ST

# load trajectory
u = mda.Universe("data/interface.psf", "data/trajectory.xyz")
dim = [11.246, 11.246, 35.94, 90, 90, 90]
# transform = trans.boxdimensions.set_dimensions(dim)
# u.trajectory.add_transformations(transform)
u_vels = mda.Universe("data/velocity.xyz")
ag = u.select_atoms("name O or name H")
job = ST(ag, u_vels)
job.run()

# plot 
dt = 0.5 / 1000 * 10
t = np.arange(len(temp)) * dt

dft_temp = np.loadtxt("data/dft_temp.txt")

plt.plot(t, temp, color="blue", alpha=0.5, label="code")
plt.plot(t, dft_temp[:, 0], color="red", alpha=0.5, label="DFT")

plt.xlim(t.min(), t.max())
plt.xlabel("Time [ps]")
plt.ylabel("Temperature [K]")
plt.legend(loc='center',
           bbox_to_anchor=(0.5, -0.2),
           ncol=2,
           borderaxespad=0)

plt.savefig("water_temp.png", bbox_inches="tight")
plt.show()