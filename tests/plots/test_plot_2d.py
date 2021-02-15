import os
import unittest
import numpy as np
import shutil
from pylipid.utils import check_dir
from pylipid.plots import plot_corrcoef


class TestPlot2d(unittest.TestCase):

    def setUp(self):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(file_dir, "test_plot1d")
        check_dir(self.save_dir)

    def test_plot_corrcoef(self):
        corrcoef = np.random.normal(size=(250,250))
        residue_index = np.arange(250) + 134
        plot_corrcoef(corrcoef, residue_index, cmap="coolwarm",
                      fn=os.path.join(self.save_dir, "corrcoef_plot_1.pdf"), title="This is a test")

        # gap in sequences
        residue_index = np.concatenate([np.arange(250) + 134, np.arange(500, 899), np.arange(100, 400)])
        corrcoef = np.random.normal(size=(len(residue_index), len(residue_index)))
        plot_corrcoef(corrcoef, residue_index, cmap="coolwarm",
                      fn=os.path.join(self.save_dir, "corrcoef_plot_2.pdf"), title="This is a test")

    def tearDown(self):
        shutil.rmtree(self.save_dir)

if __name__ == "__main__":
    unittest.main()

