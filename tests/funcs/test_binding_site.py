"""
Tests for binding site detection, pose collection, scoring and surface area.
Uses the committed trajectories in tests/data (real_li fixture in conftest.py): the synthetic
trajectories place every atom at random in each frame, so they contain no binding sites.
"""

import numpy as np
import pandas as pd
import pytest
import mdtraj as md
from pylipid.func import (get_node_list, collect_bound_poses, vectorize_poses,
                           calculate_scores, write_bound_poses)
from pylipid.func.binding_site import surface_area_chunk, MARTINI_CG_RADII


@pytest.fixture(scope="module")
def li(real_li):
    return real_li


@pytest.fixture(scope="module")
def atom_indices(li):
    """Protein atom indices per residue and lipid atom indices, built as in LipidInteraction.analyze_bound_poses."""
    protein_atom_indices = [[atom.index for atom in residue.atoms] for residue in li._protein_ref.top.residues]
    lipid_atom_indices = [li._protein_ref.n_atoms + atom_idx for atom_idx in np.arange(li._lipid_ref.n_atoms)]
    return protein_atom_indices, lipid_atom_indices


@pytest.fixture(scope="module")
def pose_traj(li):
    """Bound-pose trajectories for every detected binding site, keyed by binding site id."""
    assert li._node_list, "no binding sites detected in the real data"
    binding_site_map = {bs_id: nodes for bs_id, nodes in enumerate(li._node_list)}
    pose_traj, pose_info = collect_bound_poses(
        binding_site_map,
        li._contact_residues_low,
        li.trajfile_list,
        li.topfile_list,
        li.lipid,
        li._protein_ref,
        li._lipid_ref,
        stride=li.stride,
        nprot=li.nprot,
    )
    return pose_traj


def _protein_frames(li, n_frames):
    """The first frames of the first trajectory, protein atoms only."""
    traj = md.load(li.trajfile_list[0], top=li.topfile_list[0])[:n_frames]
    return traj.atom_slice([atom.index for atom in traj.top.atoms if atom.residue.is_protein])


class TestBindingSites:

    def test_get_node_list(self, li):
        node_list, modularity = get_node_list(np.nan_to_num(li.interaction_corrcoef), threshold=4)
        assert isinstance(node_list, list)
        assert len(node_list) > 0
        assert all(len(nodes) >= 4 for nodes in node_list)

    def test_sites_match_reference(self, li, real_traj_dir):
        """Binding sites agree with Interactions_CHOL.csv (from 10x longer versions of these runs).

        With the committed data, PyLipID 1.6.0 finds 10 sites, each overlapping a different
        reference site with Jaccard 0.73-0.95; the thresholds leave margin for small changes.
        """
        ref = pd.read_csv(real_traj_dir["reference_csv"])
        ref_sites = [set(ref.loc[ref["Binding Site ID"] == b, "Residue ID"])
                     for b in sorted(ref["Binding Site ID"].unique()) if b >= 0]
        assert len(li._node_list) >= 8
        matched = set()
        for nodes in li._node_list:
            jaccard = [len(site & set(nodes)) / len(site | set(nodes)) for site in ref_sites]
            best = int(np.argmax(jaccard))
            assert jaccard[best] >= 0.6, "site {} overlaps no reference site".format(sorted(nodes))
            matched.add(best)
        assert len(matched) >= 8

    def test_collect_binding_poses(self, li, pose_traj):
        assert isinstance(pose_traj, dict)
        assert set(pose_traj) <= set(range(len(li._node_list)))
        assert all(isinstance(traj, md.Trajectory) for traj in pose_traj.values())
        assert sum(traj.n_frames for traj in pose_traj.values()) > 0


class TestBindingPoses:

    def test_vectorize_poses(self, li, pose_traj, atom_indices):
        protein_atom_indices, lipid_atom_indices = atom_indices
        for bs_id, traj in pose_traj.items():
            nodes = li._node_list[bs_id]
            dist_matrix = vectorize_poses(traj, nodes, protein_atom_indices, lipid_atom_indices)
            assert dist_matrix.shape == (len(lipid_atom_indices), traj.n_frames, len(nodes))
            assert np.all(np.isfinite(dist_matrix)) and np.all(dist_matrix >= 0)

    def test_calculate_scores(self, li, pose_traj, atom_indices):
        protein_atom_indices, lipid_atom_indices = atom_indices
        scored = 0
        for bs_id, traj in pose_traj.items():
            if traj.n_frames < 5:
                continue
            dist_matrix = vectorize_poses(traj, li._node_list[bs_id], protein_atom_indices, lipid_atom_indices)
            scores = calculate_scores(dist_matrix)
            assert len(scores) == traj.n_frames
            scores_weighted = calculate_scores(dist_matrix, score_weights={0: 10})
            assert len(scores_weighted) == traj.n_frames
            scored += 1
        assert scored > 0, "no binding site had enough poses to score"

    def test_write_binding_poses(self, li, pose_traj, atom_indices, tmp_path):
        protein_atom_indices, lipid_atom_indices = atom_indices
        bs_id, traj = max(pose_traj.items(), key=lambda item: item[1].n_frames)
        dist_matrix = vectorize_poses(traj, li._node_list[bs_id], protein_atom_indices, lipid_atom_indices)
        scores = calculate_scores(dist_matrix)
        pose_indices = np.argsort(scores)[::-1][:min(3, traj.n_frames)]
        write_bound_poses(traj, pose_indices, str(tmp_path), pose_prefix="BSid{}_top".format(bs_id),
                          pose_format="gro")
        assert len(list(tmp_path.glob("BSid{}_top*.gro".format(bs_id)))) == len(pose_indices)

    def test_compute_surface_area(self, li):
        serial = li.compute_surface_area(plot_data=False, num_cpus=1)
        assert serial.shape[0] == sum(md.load(f, top=t).n_frames for f, t in zip(li.trajfile_list, li.topfile_list))
        # frames are calculated independently, so splitting them between workers gives identical results
        assert serial.equals(li.compute_surface_area(plot_data=False, num_cpus=3))

    def test_surface_area_radii_applied(self, li):
        # radii are keyed by bead name (mdtraj would match them against element symbols and ignore them)
        protein_traj = _protein_frames(li, 3)
        site_atoms = [[atom.index for residue_id in li._node_list[0]
                       for atom in protein_traj.top.residue(residue_id).atoms]]
        default = surface_area_chunk(protein_traj, site_atoms, MARTINI_CG_RADII)
        larger = surface_area_chunk(protein_traj, site_atoms, {**MARTINI_CG_RADII, "BB": 0.35})
        assert default.shape == (3, 1)
        assert not np.array_equal(default, larger)

    def test_surface_area_frames_independent(self, li):
        protein_traj = _protein_frames(li, 6)
        site_atoms = [[atom.index for residue_id in nodes for atom in protein_traj.top.residue(residue_id).atoms]
                      for nodes in li._node_list[:2]]
        whole = surface_area_chunk(protein_traj, site_atoms, MARTINI_CG_RADII)
        chunks = np.concatenate([surface_area_chunk(protein_traj[i:i + 2], site_atoms, MARTINI_CG_RADII)
                                 for i in (0, 2, 4)])
        np.testing.assert_array_equal(whole, chunks)
