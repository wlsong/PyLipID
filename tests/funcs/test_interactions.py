import unittest
import numpy as np
from pylipid.funcs import cal_contact_residues, Duration, cal_interaction_frequency

class TestCutoff(unittest.TestCase):

    def test_cal_contact_residues(self):
        dr0 = [0.9, 0.95, 1.2, 1.1, 1.0, 0.9]
        dr1 = [0.95, 0.9, 0.95, 1.1, 1.2, 1.1]
        dr2 = [0.90, 0.90, 0.85, 0.95, 1.0, 1.1]
        dist_matrix = np.array([dr0, dr1, dr2])
        contact_list, frame_id_set, residue_id_set = cal_contact_residues(dist_matrix, 1.0)
        self.assertEqual(contact_list, [[0, 1, 2], [0, 1, 2], [1, 2], [2], [0, 2], [0]])
        self.assertListEqual(list(frame_id_set), [0, 1, 4, 5, 0, 1, 2, 0, 1, 2, 3, 4])
        self.assertListEqual(list(residue_id_set), [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2])

    def test_Duration(self):
        contact_low = [  np.array([0,1,2]),    np.array([2,3]),    np.array([2,3]),  np.array([1,3]),  np.array([]),   np.array([])]
        contact_high = [np.array([0,1,2,3]), np.array([0,1,2,3]), np.array([1,2,3]), np.array([1,3]), np.array([1,3]), np.array([])]
        durations = Duration(contact_low, contact_high, 2).cal_durations()
        self.assertEqual(durations, [4, 6, 8, 10])

    def test_cal_interaction_frequency(self):
        contact_list = [[0], [0,1], [1,2], [0,2], [], [], [1], [0], [], []]
        occupancy, lipidcount = cal_interaction_frequency(contact_list)
        self.assertEqual(occupancy, 60)
        self.assertEqual(lipidcount, 1.5)


if __name__ == "__main__":
    unittest.main()
