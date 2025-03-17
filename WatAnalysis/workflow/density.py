# SPDX-License-Identifier: LGPL-3.0-or-later
from WatAnalysis import waterstructure
from WatAnalysis.workflow.base import (
    OneDimCoordSingleAnalysis,
    PlanarInterfaceAnalysisBase,
)


class DensityAnalysis(OneDimCoordSingleAnalysis):
    """
    General class to calculate 1D particle density profile.

    Parameters
    ----------
    selection : str
        Atom selection string used by MDAnalysis.core.universe.Universe.select_atoms(selection)
    mol_mass : float
        Molecular mass of the atoms in g/mol
    label : str
        Label to identify the intermediate results in the analysis object
    d_bin : float
        Bin width for histogram calculations in Angstroms.
        (Default: 0.1)
    """

    def __init__(
        self,
        selection: str,
        mol_mass: float,
        label: str,
        d_bin: float = 0.1,
    ) -> None:
        super().__init__(selection=selection, label=label)
        self.mol_mass = mol_mass

        assert d_bin > 0, "Bin width must be greater than 0."
        self.d_bin = d_bin

    def _conclude(self, analyser: PlanarInterfaceAnalysisBase):
        super()._conclude(analyser)

        bins, density = waterstructure.calc_density_profile(
            (analyser.r_surf_lo.mean(), analyser.r_surf_hi.mean()),
            self.r_wrapped.flatten(),
            cross_area=analyser.cross_area,
            n_frames=analyser.n_frames,
            dz=self.d_bin,
            sym=False,
            mol_mass=self.mol_mass,
        )
        density_sym = (density[::-1] + density) / 2

        self.results.bins = bins
        self.results.density = density
        self.results.density_sym = density_sym
