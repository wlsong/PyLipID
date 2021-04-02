import unittest
import os
import shutil
import numpy as np
from sklearn.decomposition import PCA
from pylipid.func import collect_bound_poses, vectorize_poses, write_bound_poses
from pylipid.func import cluster_DBSCAN, cluster_KMeans
from pylipid import LipidInteraction

class TestCluster(unittest.TestCase):

    def setUp(self):
        trajfile_list = ["../data/run1/protein_lipids.xtc", "../data/run2/protein_lipids.xtc"]
        topfile_list = ["../data/run1/protein_lipids.gro", "../data/run2/protein_lipids.gro"]
        lipid = "CHOL"
        cutoffs = [0.55, 0.8]
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(file_dir, "binding_site")
        self.li = LipidInteraction(trajfile_list, topfile_list, cutoffs=cutoffs, lipid=lipid,
                                   nprot=1, save_dir=self.save_dir)
        self.li.collect_residue_contacts(write_log=False, print_log=True)
        node_list = self.li.compute_binding_nodes(print_data=False)
        binding_site_map = {bs_id: nodes for bs_id, nodes in enumerate(node_list)}
        contact_residue_dict = self.li.contact_residues_low
        self.pose_pool = collect_bound_poses(binding_site_map, contact_residue_dict, self.li.trajfile_list[0],
                                             self.li.topfile_list[0], self.li.lipid, self.li.stride, self.li.nprot)

        def test_cluster_DBSCAN(self):
            for bs_id, nodes in enumerate(self.li._node_list):
                dist_matrix, pose_traj = vectorize_poses(self.pose_pool[bs_id], nodes, self.li._protein_ref,
                                                         self.li._lipid_ref)
                lipid_dist_per_pose = [dist_matrix[:, pose_id, :].ravel()
                                       for pose_id in np.arange(dist_matrix.shape[1])]
                transformed_data = PCA(n_components=0.95).fit_transform(lipid_dist_per_pose)
                cluster_labels = cluster_DBSCAN(transformed_data, eps=None, min_samples=None,
                                                metric="euclidean")
                self.assertEqual(len(cluster_labels), len(lipid_dist_per_pose))
                cluster_id_set = [label for label in np.unique(cluster_labels) if label != -1]
                selected_pose_id = [np.random.choice(np.where(cluster_labels == cluster_id)[0], 1)[0]
                                    for cluster_id in cluster_id_set]
                write_bound_poses(pose_traj, selected_pose_id, self.save_dir,
                                  pose_prefix="BSid{}_cluster_DBSCAN".format(bs_id), pose_format="gro")

        def test_cluster_KMeans(self):
            for bs_id, nodes in enumerate(self.li._node_list):
                dist_matrix, pose_traj = vectorize_poses(self.pose_pool[bs_id], nodes, self.li._protein_ref,
                                                         self.li._lipid_ref)
                lipid_dist_per_pose = [dist_matrix[:, pose_id, :].ravel()
                                       for pose_id in np.arange(dist_matrix.shape[1])]
                transformed_data = PCA(n_components=0.95).fit_transform(lipid_dist_per_pose)
                cluster_labels = cluster_KMeans(transformed_data, n_clusters=5)
                self.assertEqual(len(cluster_labels), len(lipid_dist_per_pose))
                cluster_id_set = [label for label in np.unique(cluster_labels) if label != -1]
                selected_pose_id = [np.random.choice(np.where(cluster_labels == cluster_id)[0], 1)[0]
                                    for cluster_id in cluster_id_set]
                write_bound_poses(pose_traj, selected_pose_id, self.save_dir,
                                  pose_prefix="BSid{}_cluster_KMeans".format(bs_id), pose_format="gro")

        def tearDown(self):
            shutil.rmtree(self.save_dir)

    if __name__ == "__main__":
        unittest.main()
