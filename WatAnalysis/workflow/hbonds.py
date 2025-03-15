# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import Dict

import freud
import numpy as np
from ase import geometry
from MDAnalysis.lib.distances import (
    calc_angles,
    capped_distance,
    distance_array,
    minimize_vectors,
)

from WatAnalysis import utils
from WatAnalysis.workflow.base import (
    DataRequirement,
    PlanarInterfaceAnalysisBase,
    SingleAnalysis,
)


class HydrogenBondAnalysis(SingleAnalysis):
    def __init__(
        self,
        oxygen_sel: str = "name O",
        hydrogens_sel: str = "name H",
        label: str = "oxygen",
        OH_cutoff: float = 1.3,
        HB_cutoff: Dict[str, float] = None,
        max_HB: int = 4,
    ) -> None:
        super().__init__()

        self.oxygen_sel = oxygen_sel
        self.hydrogens_sel = hydrogens_sel

        self.label = label
        self.OH_cutoff = OH_cutoff
        # self.HB_cutoff = HB_cutoff
        self.DA_cutoff = HB_cutoff.get("DA", 3.5)
        self.HDA_cutoff = HB_cutoff.get("HDA", None)
        self.DHA_cutoff = HB_cutoff.get("DHA", None)
        assert (
            self.HDA_cutoff is not None or self.DHA_cutoff is not None
        ), "Either HDA_cutoff or DHA_cutoff must be provided."
        self.max_HB = max_HB

        self.data_requirements = {
            f"donor_z_{self.label}": DataRequirement(
                f"donor_z_{self.label}",
                atomic=True,
                dim=max_HB,
                selection=self.oxygen_sel,
                value=np.NaN,
            ),
            f"acceptor_z_{self.label}": DataRequirement(
                f"acceptor_z_{self.label}",
                atomic=True,
                dim=max_HB,
                selection=self.oxygen_sel,
                value=np.NaN,
            ),
            f"donor_cn_{self.label}": DataRequirement(
                f"donor_cn_{self.label}",
                atomic=True,
                dim=max_HB,
                selection=self.oxygen_sel,
                value=0,
                dtype=int,
            ),
            f"acceptor_cn_{self.label}": DataRequirement(
                f"acceptor_cn_{self.label}",
                atomic=True,
                dim=max_HB,
                selection=self.oxygen_sel,
                value=0,
                dtype=int,
            ),
            f"da_distance_{self.label}": DataRequirement(
                f"da_distance_{self.label}",
                atomic=True,
                dim=max_HB,
                selection=self.oxygen_sel,
                value=np.NaN,
            ),
            f"hbond_angle_{self.label}": DataRequirement(
                f"hbond_angle_{self.label}",
                atomic=True,
                dim=max_HB,
                selection=self.oxygen_sel,
                value=np.NaN,
            ),
        }

        self.ag_oxygen = None
        self.ag_hydrogen = None

    def _prepare(self, analyser: PlanarInterfaceAnalysisBase):
        self.ag_oxygen = analyser.universe.select_atoms(self.oxygen_sel)
        self.ag_hydrogen = analyser.universe.select_atoms(self.hydrogens_sel)

    def _single_frame(self, analyser: PlanarInterfaceAnalysisBase):
        ts_box = analyser._ts.dimensions

        # return [O_id, H_id] for each O-H pair
        ts_OH_pairs = capped_distance(
            self.ag_oxygen.positions,
            self.ag_hydrogen.positions,
            max_cutoff=self.OH_cutoff,
            box=ts_box,
            return_distances=False,
        )
        # find A-D pairs based on O-O distance
        _donors = self.ag_oxygen[ts_OH_pairs[:, 0]]
        _hydrogens = self.ag_hydrogen[ts_OH_pairs[:, 1]]
        ts_OO_pairs, DA_distances = capped_distance(
            _donors.positions,
            self.ag_oxygen.positions,
            max_cutoff=self.DA_cutoff,
            min_cutoff=1.0,
            box=ts_box,
            return_distances=True,
        )
        tmp_donors = _donors[ts_OO_pairs[:, 0]]
        tmp_hydrogens = _hydrogens[ts_OO_pairs[:, 0]]
        tmp_acceptors = self.ag_oxygen[ts_OO_pairs[:, 1]]
        # check the angle threshold is H-O...O or O-H...O
        # Find angles and compared with the angle cutoff
        if self.HDA_cutoff is not None:
            cutoff_angles = np.rad2deg(
                calc_angles(
                    tmp_hydrogens.positions,
                    tmp_donors.positions,
                    tmp_acceptors.positions,
                    box=ts_box,
                )
            )
            hbond_indices = np.where(cutoff_angles <= self.HDA_cutoff)[0]
        elif self.DHA_cutoff is not None:
            cutoff_angles = np.rad2deg(
                calc_angles(
                    tmp_donors.positions,
                    tmp_hydrogens.positions,
                    tmp_acceptors.positions,
                    box=ts_box,
                )
            )
            hbond_indices = np.where(cutoff_angles >= self.DHA_cutoff)[0]
        else:
            raise ValueError("Either HDA_cutoff or DHA_cutoff must be provided.")

        # Retrieve atoms, distances and angles of hydrogen bonds
        hbond_donors = tmp_donors[hbond_indices]
        hbond_acceptors = tmp_acceptors[hbond_indices]

        np.copyto(
            getattr(analyser, f"donor_z_{self.label}")[analyser._frame_index],
            np.pad(
                hbond_donors.positions[:, analyser.axis],
                (0, self.max_HB * self.ag_oxygen.n_atoms - hbond_donors.n_atoms),
                mode="constant",
                constant_values=np.nan,
            ).reshape(self.ag_oxygen.n_atoms, self.max_HB),
        )
        np.copyto(
            getattr(analyser, f"acceptor_z_{self.label}")[analyser._frame_index],
            np.pad(
                hbond_acceptors.positions[:, analyser.axis],
                (0, self.max_HB * self.ag_oxygen.n_atoms - hbond_acceptors.n_atoms),
                mode="constant",
                constant_values=np.nan,
            ).reshape(self.ag_oxygen.n_atoms, self.max_HB),
        )
        np.copyto(
            getattr(analyser, f"da_distance_{self.label}")[analyser._frame_index],
            np.pad(
                DA_distances[hbond_indices],
                (0, self.max_HB * self.ag_oxygen.n_atoms - hbond_donors.n_atoms),
                mode="constant",
                constant_values=np.nan,
            ).reshape(self.ag_oxygen.n_atoms, self.max_HB),
        )
        np.copyto(
            getattr(analyser, f"hbond_angle_{self.label}")[analyser._frame_index],
            np.pad(
                cutoff_angles[hbond_indices],
                (0, self.max_HB * self.ag_oxygen.n_atoms - hbond_donors.n_atoms),
                mode="constant",
                constant_values=np.nan,
            ).reshape(self.ag_oxygen.n_atoms, self.max_HB),
        )
        # calculate CN for each oxygen atom

        ts_cn = np.zeros(self.ag_oxygen.n_atoms, dtype=int)
        oxygen_ids, _cn = np.unique(ts_OH_pairs[:, 0], return_counts=True)
        ts_cn[oxygen_ids] = _cn
        full_cn = np.zeros(analyser.universe.atoms.n_atoms, dtype=int)
        full_cn[self.ag_oxygen.indices] = ts_cn
        # print(full_cn)
        np.copyto(
            getattr(analyser, f"donor_cn_{self.label}")[analyser._frame_index],
            np.pad(
                full_cn[hbond_donors.indices],
                (0, self.max_HB * self.ag_oxygen.n_atoms - hbond_donors.n_atoms),
                mode="constant",
                constant_values=0,
            ).reshape(self.ag_oxygen.n_atoms, self.max_HB),
        )
        np.copyto(
            getattr(analyser, f"acceptor_cn_{self.label}")[analyser._frame_index],
            np.pad(
                full_cn[hbond_acceptors.indices],
                (0, self.max_HB * self.ag_oxygen.n_atoms - hbond_acceptors.n_atoms),
                mode="constant",
                constant_values=0,
            ).reshape(self.ag_oxygen.n_atoms, self.max_HB),
        )

        # set the flag to True
        analyser.data_requirements[f"donor_z_{self.label}"].set_update_flag(True)
        analyser.data_requirements[f"acceptor_z_{self.label}"].set_update_flag(True)
        analyser.data_requirements[f"da_distance_{self.label}"].set_update_flag(True)
        analyser.data_requirements[f"hbond_angle_{self.label}"].set_update_flag(True)
        analyser.data_requirements[f"donor_cn_{self.label}"].set_update_flag(True)
        analyser.data_requirements[f"acceptor_cn_{self.label}"].set_update_flag(True)

    def _conclude(self, analyser: PlanarInterfaceAnalysisBase):
        box_length = analyser.universe.dimensions[analyser.axis]
        # within one cell length positive of z1.
        donor_z_unwrapped = getattr(analyser, f"donor_z_{self.label}")
        donor_z_wrapped = utils.mic_1d(
            donor_z_unwrapped - analyser.r_surf_ref[:, np.newaxis, np.newaxis],
            box_length=box_length,
            ref=box_length / 2,
        )
        acceptor_z_unwrapped = getattr(analyser, f"acceptor_z_{self.label}")
        acceptor_z_wrapped = utils.mic_1d(
            acceptor_z_unwrapped - analyser.r_surf_ref[:, np.newaxis, np.newaxis],
            box_length=box_length,
            ref=box_length / 2,
        )
        frame_index = np.tile(
            np.arange(analyser.n_frames).reshape(-1, 1, 1),
            (1, self.ag_oxygen.n_atoms, self.max_HB),
        )
        mask = ~np.isnan(donor_z_wrapped)
        # output is [frame_index, z_donor, z_acceptor, distance, angle, CN_donor, CN_acceptor]
        self.results.hbonds = np.stack(
            [
                frame_index[mask],
                donor_z_wrapped[mask],
                acceptor_z_wrapped[mask],
                getattr(analyser, f"da_distance_{self.label}")[mask],
                getattr(analyser, f"hbond_angle_{self.label}")[mask],
                getattr(analyser, f"donor_cn_{self.label}")[mask],
                getattr(analyser, f"acceptor_cn_{self.label}")[mask],
            ],
            axis=-1,
        )


class RadialCorrelationFunction(SingleAnalysis):
    def __init__(
        self,
        oxygen_sel: str = "name O",
        hydrogens_sel: str = "name H",
        label: str = "oxygen",
        d_bin: float = 0.1,
        cutoff: float = 4.5,
    ) -> None:
        super().__init__()
        self.oxygen_sel = oxygen_sel
        self.hydrogens_sel = hydrogens_sel
        self.label = label

        assert d_bin > 0, "Bin width must be greater than 0."
        self.d_bin = d_bin
        self.cutoff = cutoff

        self.ag_oxygen = None
        self.ag_hydrogen = None
        self.calculator = None
        self.corr_func = None

    def _prepare(self, analyser: PlanarInterfaceAnalysisBase):
        self.ag_oxygen = analyser.universe.select_atoms(self.oxygen_sel)
        self.ag_hydrogen = analyser.universe.select_atoms(self.hydrogens_sel)

        cell_vectors = geometry.cellpar_to_cell(analyser.universe.dimensions)
        cell_vectors = np.delete(cell_vectors, analyser.axis, axis=0)
        r_max = utils.calc_pbc_r_max(*cell_vectors)
        bins = int(r_max / self.d_bin)
        self.calculator = freud.density.CorrelationFunction(
            bins=bins, r_max=r_max / 1.01
        )
        self.corr_func = np.zeros((analyser.n_frames, 3, bins))

    def _single_frame(self, analyser: PlanarInterfaceAnalysisBase):
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
            self.ag_oxygen.positions[:, analyser.axis] - ts_r_surf_lo,
            box_length,
            ref=box_length / 2,
        )
        # z distance between the oxygen atom and the surface
        ts_wrapped_r = np.min([ts_wrapped_r, ts_r_surf_hi - ts_wrapped_r], axis=0)
        mask = ts_wrapped_r < self.cutoff

        ts_selected_oxygen = self.ag_oxygen[mask]
        # print(ts_selected_oxygen.positions[:, analyser.axis])

        # Compute neighbor list of selected oxygen atoms
        nlist = (
            freud.AABBQuery.from_system(analyser._ts, 3)
            .query(ts_selected_oxygen.positions, {"r_max": 5.0})
            .toNeighborList()
        )

        ds = distance_array(
            ts_selected_oxygen.positions,
            self.ag_oxygen.positions,
            box=analyser._ts.dimensions,
        )
        min_vectors = minimize_vectors(
            self.ag_oxygen.positions[np.argsort(ds, axis=1)[:, 1]]
            - ts_selected_oxygen.positions,
            box=analyser._ts.dimensions,
        )
        # print(min_vectors)
        # calculate the angles
        angles = np.arccos(min_vectors[:, 0] / np.linalg.norm(min_vectors, axis=1))
        self.calculator.compute(
            system=analyser._ts,
            neighbors=nlist,
            values=analyser._ts.positions[:, analyser.axis],
            query_points=ts_selected_oxygen.positions,
            query_values=angles,
        )
        np.copyto(self.corr_func[analyser._frame_index, 0], self.calculator.correlation)

        ds = distance_array(
            ts_selected_oxygen.positions,
            self.ag_hydrogen.positions,
            box=analyser._ts.dimensions,
        )
        min_vectors = minimize_vectors(
            self.ag_hydrogen.positions[np.argsort(ds, axis=1)[:, 0]]
            - ts_selected_oxygen.positions,
            box=analyser._ts.dimensions,
        )
        angles = np.arccos(min_vectors[:, 0] / np.linalg.norm(min_vectors, axis=1))
        self.calculator.compute(
            system=analyser._ts,
            neighbors=nlist,
            values=analyser._ts.positions[:, analyser.axis],
            query_points=ts_selected_oxygen.positions,
            query_values=angles,
        )
        np.copyto(self.corr_func[analyser._frame_index, 1], self.calculator.correlation)

        min_vectors = minimize_vectors(
            self.ag_hydrogen.positions[np.argsort(ds, axis=1)[:, 1]]
            - ts_selected_oxygen.positions,
            box=analyser._ts.dimensions,
        )
        angles = np.arccos(min_vectors[:, 0] / np.linalg.norm(min_vectors, axis=1))
        self.calculator.compute(
            system=analyser._ts,
            neighbors=nlist,
            values=analyser._ts.positions[:, analyser.axis],
            query_points=ts_selected_oxygen.positions,
            query_values=angles,
        )
        np.copyto(self.corr_func[analyser._frame_index, 2], self.calculator.correlation)

    def _conclude(self, analyser: PlanarInterfaceAnalysisBase):
        self.results.bins = self.calculator.bin_centers
        corr_func = np.mean(self.corr_func, axis=0)
        # normalised corr_func to the first bin
        self.results.corr_func = corr_func / corr_func[:, 0][:, np.newaxis]
