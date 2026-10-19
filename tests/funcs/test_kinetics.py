"""
Tests for koff and survival function calculations.
Uses the synthetic_traj_dir session fixture from conftest.py — no real data files needed.
"""

import pytest
import numpy as np
import shutil
from pylipid.func import cal_koff, cal_survival_func
from pylipid.api import LipidInteraction


@pytest.fixture(scope="module")
def li(synthetic_traj_dir, tmp_path_factory):
    save_dir = str(tmp_path_factory.mktemp("kinetics_output"))
    instance = LipidInteraction(
        synthetic_traj_dir["trajfile_list"],
        cutoffs=synthetic_traj_dir["cutoffs"],
        topfile_list=synthetic_traj_dir["topfile_list"],
        lipid=synthetic_traj_dir["lipid"],
        nprot=1,
        save_dir=save_dir,
    )
    instance.collect_residue_contacts()
    instance.compute_residue_duration()
    yield instance
    shutil.rmtree(save_dir, ignore_errors=True)


class TestKinetics:

    def test_cal_survival_function(self, li):
        # Use residue 0 — guaranteed to exist in synthetic data
        t_total = np.max(li._T_total)
        timestep = np.min(li._timesteps)
        durations = np.concatenate(li._duration[0])
        delta_t_list = np.arange(0, t_total, timestep)
        survival_func = cal_survival_func(durations, t_total, delta_t_list)
        assert isinstance(survival_func, dict)

    def test_cal_koff(self, li):
        t_total = np.max(li._T_total)
        timestep = np.min(li._timesteps)
        durations = np.concatenate(li._duration[0])
        koff, restime, properties = cal_koff(durations, t_total, timestep,
                                             nbootstrap=5,
                                             initial_guess=[1, 1, 1, 1])
        assert isinstance(properties, dict)
