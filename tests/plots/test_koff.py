import os
import unittest
import numpy as np
import shutil
from pylipid.plots import plot_koff
from pylipid.funcs import cal_koff
from pylipid.utils import check_dir

class TestPlot(unittest.TestCase):

    def setUp(self):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(file_dir, "test_plot")
        check_dir(self.save_dir)

    def test_koff(self):
        t_total = 150
        timestep = 1
        durations = np.random.normal(loc=50, scale=15, size=400)
        koff, restime, properties = cal_koff(durations, t_total, timestep, nbootstrap=10,
                                             initial_guess=[1., 1., 1., 1.], cap=True)

        plot_koff(durations, properties["delta_t_list"], properties["survival_rates"],
                  properties["n_fitted"], survival_rates_bootstraps=properties["survival_rates_boot_set"],
                  fig_fn=os.path.join(self.save_dir, "test_koff_plot.pdf"), title="test koff",
                  timeunit="ns", t_total=t_total, text=None)

        # set the text printed on the right
        tu = "ns"
        text = "{:18s} = {:.3f} {:2s}$^{{-1}} $\n".format("$k_{{off1}}$", properties["ks"][0], tu)
        text += "{:18s} = {:.3f} {:2s}$^{{-1}} $\n".format("$k_{{off2}}$", properties["ks"][1], tu)
        text += "{:14s} = {:.4f}\n".format("$R^2$", properties["r_squared"])
        ks_boot_avg = np.mean(properties["ks_boot_set"], axis=0)
        cv_avg = 100 * np.std(properties["ks_boot_set"], axis=0) / np.mean(properties["ks_boot_set"], axis=0)
        text += "{:18s} = {:.3f} {:2s}$^{{-1}}$ ({:3.1f}%)\n".format("$k_{{off1, boot}}$",
                                                                     ks_boot_avg[0], tu, cv_avg[0])
        text += "{:18s} = {:.3f} {:2s}$^{{-1}}$ ({:3.1f}%)\n".format("$k_{{off2, boot}}$",
                                                                     ks_boot_avg[1], tu, cv_avg[1])
        text += "{:18s} = {:.3f} {:2s}".format("$Res. Time$", properties["res_time"], tu)

        plot_koff(durations, properties["delta_t_list"], properties["survival_rates"],
                  properties["n_fitted"], survival_rates_bootstraps=properties["survival_rates_boot_set"],
                  fig_fn=os.path.join(self.save_dir, "test_koff_plot_withText.pdf"), title="test koff",
                  timeunit="ns", t_total=t_total, text=text)


    def tearDown(self):
        shutil.rmtree(self.save_dir)

if __name__ == "__main__":
    unittest.main()

