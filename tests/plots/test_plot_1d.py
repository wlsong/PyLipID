import os
import unittest
import numpy as np
import shutil
import pandas as pd
from pylipid.utils import check_dir
from pylipid.plots import plot_residue_data, plot_residue_data_logos
from pylipid.plots import plot_binding_site_data, plot_surface_area

class TestPlot1d(unittest.TestCase):

    def setUp(self):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(file_dir, "test_plot1d")
        check_dir(self.save_dir)

    def test_plot_residue_data(self):
        letters_base = [chr(num) for num in np.arange(65, 65 + 26)]
        # basic
        residue_index = np.arange(120) + 34
        interactions = np.random.random(size=120) * 150
        plot_residue_data(residue_index, interactions, gap=20,
                          ylabel=None, fn=os.path.join(self.save_dir, "plot_residue_data_1.pdf"),
                          title="This is a test")
        logos = np.random.choice(letters_base, size=120)
        plot_residue_data_logos(residue_index, logos, interactions, gap=100, letter_map=None,
                                color_scheme="chemistry", ylabel="interactions",
                                fn=os.path.join(self.save_dir, "plot_residue_data_logo_1.pdf"))

        # gap in-between sequence
        residue_index = np.concatenate([np.arange(120), np.arange(138, 183), np.arange(355, 382)])
        interactions = np.random.random(size=(120+(183-138)+(312-255)))
        plot_residue_data(residue_index, interactions, gap=50,
                          ylabel=None, fn=os.path.join(self.save_dir, "plot_residue_data_2.pdf"),
                          title="This is a test")
        logos = np.random.choice(letters_base, size=len(interactions))
        plot_residue_data_logos(residue_index, logos, interactions, gap=100,
                          ylabel=None, fn=os.path.join(self.save_dir, "plot_residue_data_logo_2.pdf"))

        # two chains in sequences.
        residue_index = np.concatenate([np.arange(150)+24, np.arange(10, 120)])
        interactions = np.random.random(size=(150+(120-10)))
        plot_residue_data(residue_index, interactions, gap=20,
                          ylabel=None, fn=os.path.join(self.save_dir, "plot_residue_data_3.pdf"),
                          title="This is a test")
        logos = np.random.choice(letters_base, size=len(interactions))
        plot_residue_data_logos(residue_index, logos, interactions, gap=100,
                                ylabel=None, fn=os.path.join(self.save_dir, "plot_residue_data_logo_3.pdf"))

        # check letter mapping
        letter_map = {"ABC": "A", "BCD": "B", "CEF": "C", "DEF": "D", "EFG": "E", "FGH": "F"}
        three_letter_seq = np.random.choice(list(letter_map.keys()), size=23)
        residue_index = np.arange(23) + 52
        interactions = np.random.random(size=23)
        plot_residue_data_logos(residue_index, three_letter_seq, interactions, gap=100,
                                ylabel=None, fn=os.path.join(self.save_dir, "plot_residue_data_logo_4.pdf"),
                                letter_map=letter_map)

    def test_plot_binding_site_data(self):
        toy_dataset = {}
        for bs_id, length in zip(np.arange(12), [40,80,120,200]*3):
            toy_dataset[f"Binding Site {bs_id}"] = np.random.normal(loc=0, size=length)
        data_processed = pd.DataFrame(
            dict([(bs_label, pd.Series(data)) for bs_label, data in toy_dataset.items()])
        )
        plot_binding_site_data(data_processed, os.path.join(self.save_dir, "binding_site_data.pdf"),
                               title="Binding Site data", ylabel="RMSD (nm)")

    def test_plot_surface_area(self):
        full_set = []
        for dummy in range(4):
            toy_dataset = {f"Binding Site {bs_id}": np.random.normal(loc=0, size=20)
                           for bs_id in np.arange(12)}
            toy_dataset["Time"] = np.arange(20) * 0.1
            full_set.append(pd.DataFrame(toy_dataset))
        full_set_dataframe = pd.concat(full_set, keys=[(0,0), (0,1), (1,0), (1,1)])
        plot_surface_area(full_set_dataframe, os.path.join(self.save_dir, "surface_area_data.pdf"), timeunit=None)

    def tearDown(self):
        shutil.rmtree(self.save_dir)


if __name__ == "__main__":
    unittest.main()

