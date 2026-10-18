"""
Tests for pose clustering (DBSCAN and KMeans).
Uses the committed trajectories in tests/data (real_li fixture in conftest.py), which contain binding sites.
"""

import pytest
import numpy as np
from sklearn.decomposition import PCA
from pylipid.func import collect_bound_poses, vectorize_poses, write_bound_poses
from pylipid.func import cluster_DBSCAN, cluster_KMeans


@pytest.fixture(scope="module")
def site_vectors(real_li):
    """For each binding site with at least 5 bound poses: (bs_id, pose trajectory, PCA-reduced pose vectors)."""
    li = real_li
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
    protein_atom_indices = [[atom.index for atom in residue.atoms] for residue in li._protein_ref.top.residues]
    lipid_atom_indices = [li._protein_ref.n_atoms + atom_idx for atom_idx in np.arange(li._lipid_ref.n_atoms)]
    sites = []
    for bs_id, traj in pose_traj.items():
        if traj.n_frames < 5:
            continue
        dist_matrix = vectorize_poses(traj, li._node_list[bs_id], protein_atom_indices, lipid_atom_indices)
        lipid_dist_per_pose = [dist_matrix[:, pose_id, :].ravel() for pose_id in np.arange(dist_matrix.shape[1])]
        transformed = PCA(n_components=min(2, len(lipid_dist_per_pose) - 1)).fit_transform(lipid_dist_per_pose)
        sites.append((bs_id, traj, transformed))
    assert sites, "no binding site had enough poses to cluster"
    return sites


class TestCluster:

    def test_cluster_KMeans_few_samples(self):
        # fewer samples than clusters falls back to DBSCAN but must still return labels only
        labels = cluster_KMeans(np.array([[0.0, 0.0], [0.1, 0.1]]), n_clusters=3)
        assert len(labels) == 2

    def test_cluster_DBSCAN(self, site_vectors, tmp_path):
        for bs_id, traj, transformed in site_vectors:
            cluster_labels, core_sample_indices = cluster_DBSCAN(transformed, eps=None, min_samples=None,
                                                                 metric="euclidean")
            assert len(cluster_labels) == traj.n_frames
            assert len(core_sample_indices) == len([label for label in np.unique(cluster_labels) if label != -1])
            cluster_id_set = [label for label in np.unique(cluster_labels) if label != -1]
            if cluster_id_set:
                selected_pose_id = [np.random.choice(np.where(cluster_labels == cid)[0], 1)[0]
                                    for cid in cluster_id_set]
                write_bound_poses(traj, selected_pose_id, str(tmp_path),
                                  pose_prefix="BSid{}_cluster_DBSCAN".format(bs_id), pose_format="gro")

    def test_cluster_KMeans(self, site_vectors, tmp_path):
        for bs_id, traj, transformed in site_vectors:
            n_clusters = min(3, traj.n_frames)
            cluster_labels = cluster_KMeans(transformed, n_clusters=n_clusters)
            assert len(cluster_labels) == traj.n_frames
            assert len(np.unique(cluster_labels)) <= n_clusters
            selected_pose_id = [np.random.choice(np.where(cluster_labels == cid)[0], 1)[0]
                                for cid in np.unique(cluster_labels)]
            write_bound_poses(traj, selected_pose_id, str(tmp_path),
                              pose_prefix="BSid{}_cluster_KMeans".format(bs_id), pose_format="gro")
