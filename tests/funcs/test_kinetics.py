import unittest
import os
import shutil
import numpy as np
from pylipid.funcs import cal_koff, cal_survival_func
from pylipid import LipidInteraction

class TestKinetics(unittest.TestCase):

    def setUp(self):
        trajfile_list = ["../data/run1/protein_lipids.xtc", "../data/run2/protein_lipids.xtc"]
        topfile_list = ["../data/run1/protein_lipids.gro", "../data/run2/protein_lipids.gro"]
        lipid = "CHOL"
        cutoffs = [0.55, 1.0]
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(file_dir, "test_kinetics")
        self.li = LipidInteraction(trajfile_list, topfile_list, cutoffs=cutoffs, lipid=lipid,
                                   nprot=1, save_dir=self.save_dir)
        self.li.collect_residue_contacts()
        self.t_total = np.max(self.li._T_total)
        self.timestep = np.min(self.li._timesteps)

    def test_cal_survivla_function(self):
        delta_t_list = np.arange(0, self.t_total, self.timestep)
        survival_func = cal_survival_func(np.concatenate(self.li.durations[25]), self.t_total, delta_t_list)
        self.assertIsInstance(survival_func, dict)

    def test_cal_koff(self):
        koff, restime, properties = cal_koff(np.concatenate(self.li.durations[25]), self.t_total,
                                             self.timestep, nbootstrap=20, initial_guess=[1,1,1,1])
        print(koff)
        print(restime)
        print(properties)
        self.assertIsInstance(properties, dict)

    def tearDown(self):
        shutil.rmtree(self.li.save_dir)


if __name__ == "__main__":
    unittest.main()
