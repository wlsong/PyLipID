import unittest
import os
import shutil
import numpy as np
from pylipid.funcs import get_node_list, collect_bound_poses, vectorize_poses, calculate_scores, write_bound_poses
from pylipid.funcs import calculate_site_surface_area
from pylipid import LipidInteraction


class TestBindingSites(unittest.TestCase):

    def setUp(self):
        trajfile_list = ["../data/run1/protein_lipids.xtc", "../data/run2/protein_lipids.xtc"]
        topfile_list = ["../data/run1/protein_lipids.gro", "../data/run2/protein_lipids.gro"]
        lipid = "CHOL"
        cutoffs = [0.55, 0.8]
        file_dir = os.path.dirname( os.path.abspath(__file__) )
        self.save_dir = os.path.join(file_dir, "test_binding_site")
        self.li = LipidInteraction(trajfile_list, topfile_list, cutoffs=cutoffs, lipid=lipid,
                                   nprot=1, save_dir=self.save_dir)
        self.li.collect_residue_contacts(write_log=False, print_log=True)

    def test_get_node_list(self):
        corrcoef = self.li.interaction_corrcoef
        node_list = get_node_list(corrcoef)
        self.assertIsInstance(node_list, list)

    def test_collect_binding_poses(self):
        binding_site_map = {bs_id: nodes for bs_id, nodes in enumerate(self.li._node_list)}
        contact_list = self.li.contact_residues_low
        pose_pool = collect_bound_poses(binding_site_map, contact_list, self.li.trajfile_list[0],
                                        self.li.topfile_list[0], self.li.lipid, self.li.stride, self.li.nprot)
        self.assertIsInstance(pose_pool, dict)

    def tearDown(self):
        shutil.rmtree(self.save_dir)


class TestBindingPoses(unittest.TestCase):

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

    def test_vectorize_poses(self):
        for bs_id, nodes in enumerate(self.li._node_list):
            dist_matrix, pose_traj = vectorize_poses(self.pose_pool[bs_id], nodes, self.li._protein_ref, self.li._lipid_ref)
            self.assertEqual(dist_matrix.shape[0], self.li._lipid_ref.n_atoms)
            self.assertEqual(dist_matrix.shape[1], len(self.pose_pool[bs_id]))
            self.assertEqual(dist_matrix.shape[2], len(self.li._node_list[bs_id]))
        return

    def test_calculate_scores(self):
        for bs_id, nodes in enumerate(self.li._node_list):
            dist_matrix, pose_traj = vectorize_poses(self.pose_pool[bs_id], nodes, self.li._protein_ref, self.li._lipid_ref)
            scores = calculate_scores(dist_matrix)
            self.assertEqual(len(scores), pose_traj.n_frames)
            scores = calculate_scores(dist_matrix, score_weights={"RHO": 10})
            self.assertEqual(len(scores), pose_traj.n_frames)


    def test_write_binding_poses(self):
        for bs_id, nodes in enumerate(self.li._node_list):
            dist_matrix, pose_traj = vectorize_poses(self.pose_pool[bs_id], nodes, self.li._protein_ref, self.li._lipid_ref)
            scores = calculate_scores(dist_matrix)
            num_of_poses = min(5, pose_traj.n_frames)
            pose_indices = np.argsort(scores)[::-1][:num_of_poses]
            write_bound_poses(pose_traj, pose_indices, self.save_dir, pose_prefix="BSid{}_top".format(bs_id),
                              pose_format="gro")


    def test_calculate_site_surface_area(self):
        binding_site_map = {bs_id: nodes for bs_id, nodes in enumerate(self.li._node_list)}
        radii_book = {"BB": 0.36, "SC1": 0.33, "SC2": 0.33, "SC3": 0.33}
        surface_area= calculate_site_surface_area(binding_site_map, radii_book, self.li.trajfile_list,
                                                  self.li.topfile_list, self.li.nprot, self.li.timeunit,
                                                  self.li.stride, self.li.dt_traj)

    def tearDown(self):
        shutil.rmtree(self.save_dir)


if __name__ == "__main__":
    unittest.main()
