# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import Optional, Tuple

import numpy as np

from WatAnalysis import utils, waterdynamics
from WatAnalysis.workflow.base import (
    OneDimCoordSingleAnalysis,
    PlanarInterfaceAnalysisBase,
)
from WatAnalysis.workflow.dipole import DipoleBaseSingleAnalysis


class FluxCorrelationFunction(OneDimCoordSingleAnalysis):
    """
    Calculate the flux correlation function for a given selection of atoms.
    Ref: Limmer, D. T., et al. J. Phys. Chem. C 2015, 119 (42), 24016-24024.


    Parameters
    ----------
    selection : str
        Atom selection string used by MDAnalysis.core.universe.Universe.select_atoms(selection)
    label : str
        Label to identify the intermediate results in the analysis object
    interval_i : Tuple[Optional[float], Optional[float]],
        interval for initial state
    interval_f : Tuple[Optional[float], Optional[float]],
        interval for final state
    """

    def __init__(
        self,
        selection: str,
        label: str,
        interval_i: Tuple[Optional[float], Optional[float]],
        interval_f: Tuple[Optional[float], Optional[float]],
        exclude_number: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(selection=selection, label=label)

        self.interval_i = interval_i
        self.interval_f = interval_f
        self.exclude_number = exclude_number
        self.acf_kwargs = kwargs

    def _conclude(self, analyser: PlanarInterfaceAnalysisBase):
        super()._conclude(analyser)

        mask_lo, mask_hi = utils.get_region_masks(
            self.r_wrapped.squeeze(),
            analyser.r_surf_lo,
            analyser.r_surf_hi,
            self.interval_i,
        )
        mask_i = mask_lo | mask_hi
        # convert bool to float
        indicator_i = mask_i.astype(float)[:, :, np.newaxis]

        mask_lo, mask_hi = utils.get_region_masks(
            self.r_wrapped.squeeze(),
            analyser.r_surf_lo,
            analyser.r_surf_hi,
            self.interval_f,
        )
        mask_f = mask_lo | mask_hi
        # convert bool to float
        indicator_f = mask_f.astype(float)[:, :, np.newaxis]

        tau, cf = waterdynamics.calc_vector_correlation(
            vector_a=indicator_i,
            vector_b=indicator_f,
            **self.acf_kwargs,
        )
        self.results.tau = tau
        self.results.cf = cf / (
            np.mean(indicator_i) - self.exclude_number / self.ag.n_atoms
        )


class SurvivalProbability(OneDimCoordSingleAnalysis):
    """
    Calculate the flux correlation function for a given selection of atoms.
    Ref: Limmer, D. T., et al. J. Phys. Chem. C 2015, 119 (42), 24016-24024.


    Parameters
    ----------
    selection : str
        Atom selection string used by MDAnalysis.core.universe.Universe.select_atoms(selection)
    label : str
        Label to identify the intermediate results in the analysis object
    interval : Tuple[Optional[float], Optional[float]],
        The interval for selection of atoms
    exclude_number : int
        Number of atoms to exclude from the calculation
        Useful when there are some atoms need to be excluded from the calculation
    """

    def __init__(
        self,
        selection: str,
        label: str,
        interval: Tuple[Optional[float], Optional[float]],
        exclude_number: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(selection=selection, label=label)

        self.interval = interval
        self.exclude_number = exclude_number
        self.acf_kwargs = kwargs

    def _conclude(self, analyser: PlanarInterfaceAnalysisBase):
        super()._conclude(analyser)

        mask_lo, mask_hi = utils.get_region_masks(
            self.r_wrapped.squeeze(),
            analyser.r_surf_lo,
            analyser.r_surf_hi,
            self.interval,
        )
        mask = mask_lo | mask_hi
        # convert bool to float
        ad_indicator = mask.astype(float)[:, :, np.newaxis]
        self.acf_kwargs["normalize"] = False
        tau, cf = waterdynamics.calc_vector_autocorrelation(
            vectors=ad_indicator, **self.acf_kwargs
        )

        self.results.tau = tau
        self.results.cf = (cf - self.exclude_number / self.ag.n_atoms) / (
            np.mean(ad_indicator) - self.exclude_number / self.ag.n_atoms
        )


class WaterReorientation(DipoleBaseSingleAnalysis):
    def __init__(
        self,
        selection_oxygen: str,
        selection_hydrogen: str,
        label: str,
        interval: Tuple[Optional[float], Optional[float]],
        **kwargs,
    ) -> None:
        super().__init__(
            selection_oxygen=selection_oxygen,
            selection_hydrogen=selection_hydrogen,
            label=label,
            interval=interval,
        )

        self.acf_kwargs = kwargs

    def _conclude(self, analyser: PlanarInterfaceAnalysisBase):
        super()._conclude(analyser)

        mask_lo, mask_hi = utils.get_region_masks(
            self.r_wrapped.squeeze(),
            analyser.r_surf_lo,
            analyser.r_surf_hi,
            self.interval,
        )
        mask_cn = getattr(analyser, f"cn_{self.label}") == 2
        mask = (mask_lo | mask_hi) & mask_cn.squeeze()

        tau, cf = waterdynamics.calc_vector_autocorrelation(
            vectors=getattr(analyser, f"dipole_{self.label}"),
            mask=mask,
            modifier_func=self.lg2,
            **self.acf_kwargs,
        )

        self.results.tau = tau
        self.results.cf = cf

    @staticmethod
    def lg2(x):
        """Second Legendre polynomial"""
        return (3 * x * x - 1) / 2
