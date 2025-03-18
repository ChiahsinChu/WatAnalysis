# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import Optional, Tuple

import numpy as np
from MDAnalysis.lib.distances import distance_array, minimize_vectors

from WatAnalysis import utils, waterdynamics
from WatAnalysis.workflow.base import (
    DataRequirement,
    OneDimCoordSingleAnalysis,
    PlanarInterfaceAnalysisBase,
)


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


class WaterReorientation(OneDimCoordSingleAnalysis):
    def __init__(
        self,
        selection_oxygen: str,
        selection_hydrogen: str,
        label: str,
        interval: Tuple[Optional[float], Optional[float]],
        **kwargs,
    ) -> None:
        super().__init__(selection=selection_oxygen, label=label)

        self.interval = interval
        self.acf_kwargs = kwargs

        self.selection_hydrogen = selection_hydrogen
        self.data_requirements.update(
            {
                f"dipole_{self.label}": DataRequirement(
                    f"dipole_{self.label}",
                    atomic=True,
                    dim=3,
                    selection=self.selection,
                ),
                f"cn_{self.label}": DataRequirement(
                    f"cn_{self.label}",
                    atomic=True,
                    dim=1,
                    selection=self.selection,
                    dtype=np.int32,
                ),
            }
        )

        self.ag_hydrogen = None

    def _prepare(self, analyser: PlanarInterfaceAnalysisBase):
        super()._prepare(analyser)
        self.ag_hydrogen = analyser.universe.select_atoms(self.selection_hydrogen)

    def _single_frame(self, analyser: PlanarInterfaceAnalysisBase):
        super()._single_frame(analyser)

        update_flag = (
            analyser.data_requirements[f"cn_{self.label}"].update_flag
            and analyser.data_requirements[f"dipole_{self.label}"].update_flag
        )
        if update_flag:
            return

        ts_box = analyser._ts.dimensions
        coords_oxygen = self.ag.positions
        coords_hydrogen = self.ag_hydrogen.positions

        all_distances = np.empty(
            (self.ag_hydrogen.n_atoms, self.ag.n_atoms), dtype=np.float64
        )
        distance_array(
            coords_hydrogen,
            coords_oxygen,
            result=all_distances,
            box=ts_box,
        )
        # H to O mapping
        H_to_O_mapping = np.argmin(all_distances, axis=1)
        out = np.unique(H_to_O_mapping, return_counts=True)
        cns = np.zeros(self.ag.n_atoms, dtype=np.int32)
        oxygen_ids = out[0]
        cns[oxygen_ids] = out[1]
        # copy the coordinates to the intermediate array
        np.copyto(
            getattr(analyser, f"cn_{self.label}")[analyser._frame_index],
            cns[:, np.newaxis],
        )
        # set the flag to True
        analyser.data_requirements[f"cn_{self.label}"].set_update_flag(True)

        OH_vectors = minimize_vectors(
            coords_hydrogen - coords_oxygen[H_to_O_mapping], box=ts_box
        )
        dipoles = np.zeros((self.ag.n_atoms, 3))
        for ii in range(self.ag.n_atoms):
            # tmp_vectors = OH_vectors[np.where(H_to_O_mapping == ii)[0]]
            # print(tmp_vectors, np.linalg.norm(tmp_vectors, axis=-1))
            tmp_ids = np.where(H_to_O_mapping == ii)[0]
            if len(tmp_ids) > 0:
                dipoles[ii] = OH_vectors[tmp_ids].mean(axis=0)

        # cos_theta = (dipoles[:, analyser.axis]) / np.linalg.norm(dipoles, axis=-1)
        np.copyto(
            getattr(analyser, f"dipole_{self.label}")[analyser._frame_index],
            dipoles,
        )
        # set the flag to True
        analyser.data_requirements[f"dipole_{self.label}"].set_update_flag(True)

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
            **self.acf_kwargs,
        )

        self.results.tau = tau
        self.results.cf = cf
