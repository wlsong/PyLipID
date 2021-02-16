import unittest
import os
import shutil
from pylipid.api import LipidInteraction
from pylipid.utils import check_dir

class TestLipidInteraction(unittest.TestCase):

    def test_pylipid(self):
        trajfile_list = ["../data/run1/protein_lipids.xtc", "../data/run2/protein_lipids.xtc"]
        topfile_list = ["../data/run1/protein_lipids.gro", "../data/run2/protein_lipids.gro"]
        lipid = "CHOL"
        cutoffs = [0.55, 0.8]
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = check_dir(os.path.join(file_dir, "test_pylipid"))
        li = LipidInteraction(trajfile_list, topfile_list, cutoffs=cutoffs, lipid=lipid,
                                   nprot=1, save_dir=self.save_dir)
        li.collect_residue_contacts(write_log=True, print_log=True)

        li.compute_residue_koff(print_data=False, plot_data=True)
        li.compute_residue_koff(10)
        li.compute_residue_koff([2,4,5,10])
        li.compute_residue_koff(20, print_data=False, plot_data=False)

        li.compute_binding_nodes(threshold=4)
        li.compute_binding_nodes(threshold=2, print_data=False)

        li.compute_site_koff()
        li.compute_site_koff(binding_site_id=[1,2,3], print_data=True, plot_data=True, sort_residue="Residence Time")
        li.compute_site_koff(binding_site_id=1, print_data=False, plot_data=False, sort_residue="Residence Time")
        li.compute_site_koff(binding_site_id=1, print_data=True, plot_data=True, sort_residue="Duration")

        li.analyze_bound_poses()
        li.analyze_bound_poses(binding_site_id=[1,2,3])
        li.analyze_bound_poses(binding_site_id=[1,2,3], n_clusters=2)

        li.compute_surface_area()
        li.compute_surface_area(binding_site_id=[1,2,3])
        li.compute_surface_area(binding_site_id=[1, 2, 3], radii={"BB": 0.30, "SC1": 0.2})

        li.write_site_info()
        li.write_site_info(sort_residue="Duration")

        li.save_data(item="Dataset")
        li.save_data(item="Duration")

        li.save_coordinate(item="Residence Time")
        li.save_coordinate(item="Duration")

        li.save_pymol_script(pdb_file="../data/receptor.pdb")

        li.plot(item="Duration")
        li.plot(item="Residence Time")

        li.plot_logo(item="Residence Time")
        li.plot_logo(item="Lipid Count")

    def tearDown(self):
        shutil.rmtree(self.save_dir)

if __name__ == "__main__":
    unittest.main()





