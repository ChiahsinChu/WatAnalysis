# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np

from WatAnalysis import utils, waterdynamics
from WatAnalysis.workflow.base import (
    DataRequirement,
    PlanarInterfaceAnalysisBase,
    SingleAnalysis,
)


class FluxCorrelationFunction(SingleAnalysis):
    """
    Calculate the flux correlation function for a given selection of atoms.
    Ref: Limmer, D. T., et al. J. Phys. Chem. C 2015, 119 (42), 24016-24024.


    Parameters
    ----------
    selection : str
        Atom selection string used by MDAnalysis.core.universe.Universe.select_atoms(selection)
    label : str
        Label to identify the intermediate results in the analysis object
    """

    def __init__(
        self,
        selection: str,
        label: str,
        cutoff: float = 2.7,
        **kwargs,
    ) -> None:
        super().__init__()
        self.selection = selection
        self.label = label
        self.cutoff = cutoff
        self.acf_kwargs = kwargs

        self.data_requirements = {
            f"ad_indicator_{self.label}": DataRequirement(
                f"ad_indicator_{self.label}",
                atomic=True,
                dim=1,
                selection=self.selection,
            ),
        }

        self.ag = None

    def _prepare(self, analyser: PlanarInterfaceAnalysisBase):
        self.ag = analyser.universe.select_atoms(self.selection)

    def _single_frame(self, analyser: PlanarInterfaceAnalysisBase):
        update_flag = analyser.data_requirements[
            f"ad_indicator_{self.label}"
        ].update_flag
        if not update_flag:
            ts_box = analyser._ts.dimensions
            box_length = ts_box[analyser.axis]

            # calculate mask based on self.cutoff
            ts_r_surf_lo = analyser.r_surf_lo[analyser._frame_index]
            ts_r_surf_hi = utils.mic_1d(
                analyser.r_surf_hi[analyser._frame_index] - ts_r_surf_lo,
                box_length,
                ref=box_length / 2,
            )
            ts_wrapped_r = utils.mic_1d(
                self.ag.positions[:, analyser.axis] - ts_r_surf_lo,
                box_length,
                ref=box_length / 2,
            )
            # z distance between the oxygen atom and the surface
            ts_wrapped_r = np.min([ts_wrapped_r, ts_r_surf_hi - ts_wrapped_r], axis=0)
            mask = ts_wrapped_r < self.cutoff
            # set adsorbed indicator to 1 if the atom is within the cutoff distance
            getattr(analyser, f"ad_indicator_{self.label}")[
                analyser._frame_index, mask, 0
            ] = 1.0

            # set the flag to True
            analyser.data_requirements[f"ad_indicator_{self.label}"].set_update_flag(
                True
            )

    def _conclude(self, analyser: PlanarInterfaceAnalysisBase):
        self.acf_kwargs["normalize"] = False
        ad_indicator = getattr(analyser, f"ad_indicator_{self.label}")
        tau, flux_cf = waterdynamics.calc_vector_autocorrelation(
            vectors=ad_indicator, **self.acf_kwargs
        )
        self.results.tau = tau
        self.results.cf = flux_cf / np.mean(ad_indicator)
