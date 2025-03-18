# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import Optional, Tuple

import numpy as np
from MDAnalysis.lib.distances import distance_array, minimize_vectors

from WatAnalysis import utils
from WatAnalysis.workflow.base import (
    DataRequirement,
    OneDimCoordSingleAnalysis,
    PlanarInterfaceAnalysisBase,
)


class DipoleBaseSingleAnalysis(OneDimCoordSingleAnalysis):
    def __init__(
        self,
        selection_oxygen: str,
        selection_hydrogen: str,
        label: str,
        interval: Tuple[Optional[float], Optional[float]],
    ) -> None:
        super().__init__(selection=selection_oxygen, label=label)

        self.interval = interval

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

        np.copyto(
            getattr(analyser, f"dipole_{self.label}")[analyser._frame_index],
            dipoles,
        )
        # set the flag to True
        analyser.data_requirements[f"dipole_{self.label}"].set_update_flag(True)


class AngularDistribution(DipoleBaseSingleAnalysis):
    def __init__(
        self,
        selection_oxygen: str,
        selection_hydrogen: str,
        label: str,
        interval: Tuple[Optional[float], Optional[float]],
        cn: int = 2,
        d_bin: float = 5.0,
    ) -> None:
        super().__init__(
            selection_oxygen=selection_oxygen,
            selection_hydrogen=selection_hydrogen,
            label=label,
            interval=interval,
        )

        self.cn = cn

        assert d_bin > 0, "Bin width must be greater than 0."
        self.d_bin = d_bin

    def _conclude(self, analyser: PlanarInterfaceAnalysisBase):
        super()._conclude(analyser)

        mask_lo, mask_hi = utils.get_region_masks(
            self.r_wrapped.squeeze(),
            analyser.r_surf_lo,
            analyser.r_surf_hi,
            self.interval,
        )
        mask_cn = getattr(analyser, f"cn_{self.label}") == self.cn

        bin_edges = np.arange(0, 180 + self.d_bin, self.d_bin)
        bins = utils.bin_edges_to_grid(bin_edges)

        # nf x nat x 3
        dipole_vectors = getattr(analyser, f"dipole_{self.label}")

        # water at the lower surface
        mask = mask_lo & mask_cn.squeeze()
        sel_vectors = dipole_vectors[mask].reshape(-1, 3)
        cos_theta = (sel_vectors[:, analyser.axis]) / np.linalg.norm(
            sel_vectors, axis=-1
        )
        angles_lo = np.deg2rad(np.arccos(cos_theta))
        # water at the upper surface
        mask = mask_hi & mask_cn.squeeze()
        sel_vectors = dipole_vectors[mask].reshape(-1, 3)
        cos_theta = -(sel_vectors[:, analyser.axis]) / np.linalg.norm(
            sel_vectors, axis=-1
        )
        angles_hi = np.deg2rad(np.arccos(cos_theta))

        angles = np.concatenate([angles_lo, angles_hi])
        density, _ = np.histogram(angles, bins=bin_edges, density=True)

        self.results.bins = bins
        self.results.density = density
