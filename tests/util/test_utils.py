import os
import unittest
import numpy as np
import shutil
import mdtraj as md
from pylipid.utils import write_pymol_script
from pylipid.utils import check_dir
from pylipid.utils import rmsd
from pylipid.utils import sparse_corrcoef
from pylipid.utils import get_traj_info

class TestScript(unittest.TestCase):

    def setUp(self):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(file_dir, "test_util")
        check_dir(self.save_dir)

    def test_write_pymol_script(self):
        write_pymol_script(os.path.join(self.save_dir, "show_bs_info.py"),
                           "../data/receptor.pdb", "../data/Interactions_CHOL.csv", "CHOL", 10)

    def test_rmsd(self):
        matrix_a = np.random.random(size=(100, 5))
        matrix_b = np.random.random(size=(100, 5))
        value = rmsd(matrix_a, matrix_b)
        self.assertIsInstance(value, float)

    def test_sparse_corrcoef(self):
        A = np.random.normal(size=(4, 500))
        corrcoefs = sparse_corrcoef(A)
        self.assertEqual(len(corrcoefs), len(A))

    def test_get_traj_info(self):
        trajfile = "../data/run1/protein_lipids.xtc"
        topfile = "../data/run1/protein_lipids.gro"
        traj = md.load(trajfile, top=topfile)
        traj_info, protein_ref, lipid_ref = get_traj_info(traj, "CHOL")
        self.assertIsInstance(protein_ref, md.Trajectory)
        self.assertIsInstance(lipid_ref, md.Trajectory)

    def tearDown(self):
        shutil.rmtree(self.save_dir)


if __name__ == "__main__":
    unittest.main()
