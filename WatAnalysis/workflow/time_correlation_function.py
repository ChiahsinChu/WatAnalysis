# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import Optional, Tuple

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
    cutoff_ad : float
        Cutoff distance in Angstroms to define the adsorbed state
    cutoff_des : float
        Cutoff distance in Angstroms to define the desorbed state
    """

    def __init__(
        self,
        selection: str,
        label: str,
        cutoff_ad: float,
        cutoff_des: float,
        exclude_number: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.selection = selection
        self.label = label
        assert cutoff_des >= cutoff_ad, "cutoff_des must be no less than cutoff_ad"
        self.cutoff_ad = cutoff_ad
        self.cutoff_des = cutoff_des
        self.exclude_number = exclude_number
        self.acf_kwargs = kwargs

        self.data_requirements = {
            f"ad_indicator_{self.label}": DataRequirement(
                f"ad_indicator_{self.label}",
                atomic=True,
                dim=1,
                selection=self.selection,
            ),
            f"des_indicator_{self.label}": DataRequirement(
                f"des_indicator_{self.label}",
                atomic=True,
                dim=1,
                selection=self.selection,
            ),
        }

        self.ag = None

    def _prepare(self, analyser: PlanarInterfaceAnalysisBase):
        self.ag = analyser.universe.select_atoms(self.selection)

    def _single_frame(self, analyser: PlanarInterfaceAnalysisBase):
        ts_box = analyser._ts.dimensions
        box_length = ts_box[analyser.axis]
        ts_wrapped_r = None

        update_flag = analyser.data_requirements[
            f"ad_indicator_{self.label}"
        ].update_flag
        if not update_flag:
            if ts_wrapped_r is None:
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
                ts_wrapped_r = np.min(
                    [ts_wrapped_r, ts_r_surf_hi - ts_wrapped_r], axis=0
                )

            # adsorbed mask
            mask = ts_wrapped_r < self.cutoff_ad
            # set adsorbed indicator to 1 if the atom is within the cutoff distance
            getattr(analyser, f"ad_indicator_{self.label}")[
                analyser._frame_index, mask, 0
            ] = 1.0
            # set the flag to True
            analyser.data_requirements[f"ad_indicator_{self.label}"].set_update_flag(
                True
            )

        update_flag = analyser.data_requirements[
            f"des_indicator_{self.label}"
        ].update_flag
        if not update_flag:
            if ts_wrapped_r is None:
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
                ts_wrapped_r = np.min(
                    [ts_wrapped_r, ts_r_surf_hi - ts_wrapped_r], axis=0
                )

            # desorbed mask
            mask = ts_wrapped_r > self.cutoff_des
            getattr(analyser, f"des_indicator_{self.label}")[
                analyser._frame_index, mask, 0
            ] = 1.0
            # set the flag to True
            analyser.data_requirements[f"des_indicator_{self.label}"].set_update_flag(
                True
            )

    def _conclude(self, analyser: PlanarInterfaceAnalysisBase):
        self.acf_kwargs["normalize"] = False
        ad_indicator = getattr(analyser, f"ad_indicator_{self.label}")
        des_indicator = getattr(analyser, f"des_indicator_{self.label}")
        tau, cf = waterdynamics.calc_vector_correlation(
            vector_a=ad_indicator,
            vector_b=des_indicator,
            **self.acf_kwargs,
        )
        self.results.tau = tau
        self.results.cf = cf / (
            np.mean(ad_indicator) - self.exclude_number / self.ag.n_atoms
        )


class SurvivalProbability(SingleAnalysis):
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
        super().__init__()
        self.selection = selection
        self.label = label
        self.interval = interval
        self.exclude_number = exclude_number
        self.acf_kwargs = kwargs

        self.data_requirements = {
            f"coord_1d_{self.label}": DataRequirement(
                f"coord_1d_{self.label}",
                atomic=True,
                dim=1,
                selection=self.selection,
            ),
        }

        self.ag = None

    def _prepare(self, analyser: PlanarInterfaceAnalysisBase):
        self.ag = analyser.universe.select_atoms(self.selection)

    def _single_frame(self, analyser: PlanarInterfaceAnalysisBase):
        update_flag = analyser.data_requirements[f"coord_1d_{self.label}"].update_flag
        if not update_flag:
            # copy the coordinates to the intermediate array
            np.copyto(
                getattr(analyser, f"coord_1d_{self.label}")[analyser._frame_index],
                self.ag.positions[:, analyser.axis, np.newaxis],
            )
            # set the flag to True
            analyser.data_requirements[f"coord_1d_{self.label}"].set_update_flag(True)

    def _conclude(self, analyser: PlanarInterfaceAnalysisBase):
        box_length = analyser.universe.dimensions[analyser.axis]
        # within one cell length positive of z1.
        r_unwrapped = getattr(analyser, f"coord_1d_{self.label}")
        self.r_wrapped = utils.mic_1d(
            r_unwrapped - analyser.r_surf_ref[:, np.newaxis, np.newaxis],
            box_length=box_length,
            ref=box_length / 2,
        ).squeeze()
        mask_lo, mask_hi = utils.get_region_masks(
            self.r_wrapped, analyser.r_surf_lo, analyser.r_surf_hi, self.interval
        )
        mask = mask_lo | mask_hi
        # convert bool to float
        ad_indicator = mask.astype(float)
        self.acf_kwargs["normalize"] = False
        tau, cf = waterdynamics.calc_vector_autocorrelation(
            vectors=ad_indicator, **self.acf_kwargs
        )

        self.results.tau = tau
        self.results.cf = (cf - self.exclude_number / self.ag.n_atoms) / (
            np.mean(ad_indicator) - self.exclude_number / self.ag.n_atoms
        )
