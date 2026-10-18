"""
End-to-end test of the LipidInteraction API on the committed trajectories (tests/data).

Restores the original upstream test_pylipid workflow, which was accidentally replaced by a
copy of conftest.py, and checks the main outputs.
"""

import os
import numpy as np
import pandas as pd
import pytest
from pylipid.api import LipidInteraction


@pytest.fixture(scope="module")
def li(real_traj_dir, tmp_path_factory):
    np.random.seed(0)
    instance = LipidInteraction(
        real_traj_dir["trajfile_list"],
        cutoffs=real_traj_dir["cutoffs"],
        topfile_list=real_traj_dir["topfile_list"],
        lipid=real_traj_dir["lipid"],
        nprot=1,
        save_dir=str(tmp_path_factory.mktemp("api_output")),
    )
    instance.collect_residue_contacts()
    return instance


def test_pylipid_workflow(li, real_traj_dir):
    n_residues = len(li.residue_list)

    occupancies = li.compute_residue_occupancy()
    assert len(occupancies) == n_residues
    li.compute_residue_occupancy(10)

    durations = li.compute_residue_duration()
    assert len(durations) == n_residues
    li.compute_residue_duration(10)
    li.compute_residue_duration([2, 3, 4])

    koffs, restimes = li.compute_residue_koff(plot_data=False)
    assert len(koffs) == n_residues and len(restimes) == n_residues
    li.compute_residue_koff(10)
    li.compute_residue_koff([2, 4, 5, 10])

    node_list = li.compute_binding_nodes(threshold=4)
    assert len(node_list) > 0, "no binding sites found in the real data"
    li.compute_binding_nodes(threshold=2, print_data=False)
    li.compute_binding_nodes(threshold=4, print_data=False)

    site_koffs, site_restimes = li.compute_site_koff(plot_data=False)
    assert len(site_koffs) == len(li._node_list)
    li.compute_site_koff(binding_site_id=[1, 2, 3])
    li.compute_site_koff(binding_site_id=1)

    li.compute_site_duration()
    li.compute_site_duration(1)
    li.compute_site_duration([0, 1])

    li.analyze_bound_poses()
    li.analyze_bound_poses(binding_site_id=[1, 2, 3])
    li.analyze_bound_poses(binding_site_id=[1, 2, 3], n_clusters=2)

    li.compute_surface_area()
    li.compute_surface_area(binding_site_id=[1, 2, 3])
    li.compute_surface_area(binding_site_id=[1, 2, 3], radii={"BB": 0.30, "SC1": 0.2})

    li.write_site_info()
    li.write_site_info(sort_residue="Duration")

    li.show_stats_per_traj()

    li.save_data(item="Dataset")
    li.save_data(item="Duration")

    li.save_coordinate(item="Residence Time")
    li.save_coordinate(item="Duration")

    li.save_pymol_script(pdb_file=real_traj_dir["pdb_file"])

    li.plot(item="Duration")
    li.plot(item="Residence Time")

    li.plot_logo(item="Residence Time")
    li.plot_logo(item="Lipid Count")

    dataset = li.dataset
    assert len(dataset) == n_residues
    for column in ["Residue", "Occupancy", "Duration", "Residence Time", "Binding Site ID"]:
        assert column in dataset.columns
    assert dataset["Occupancy"].between(0, 100).all()
    saved_files = [f for _, _, files in os.walk(li.save_dir) for f in files]
    assert any(f.endswith(".csv") for f in saved_files)
    assert any(f.endswith(".pml") or f.endswith(".py") for f in saved_files)


def test_parallel_contacts_match_serial(real_traj_dir, tmp_path, monkeypatch):
    """Worker processes reading coordinates from shared memory give exactly the serial results."""
    from pylipid.api import api

    def contacts(num_cpus):
        li = LipidInteraction(real_traj_dir["trajfile_list"][:1], cutoffs=real_traj_dir["cutoffs"],
                              topfile_list=real_traj_dir["topfile_list"][:1], lipid=real_traj_dir["lipid"],
                              nprot=1, save_dir=str(tmp_path / "n{}".format(num_cpus)))
        li.collect_residue_contacts(num_cpus=num_cpus)
        return li

    serial = contacts(1)
    # the test system is below the size at which workers are used: force them
    monkeypatch.setattr(api, "_MIN_PARALLEL_DISTANCES", 0)
    parallel = contacts(2)
    assert serial._contact_residues_low == parallel._contact_residues_low
    assert serial._contact_residues_high == parallel._contact_residues_high
    np.testing.assert_array_equal(serial.interaction_corrcoef, parallel.interaction_corrcoef)


def test_seeded_bootstrap_is_reproducible(real_li):
    """np.random.seed in the calling process fixes the bootstrapped koffs, whatever the number of workers."""
    residues = list(range(40))

    def bootstrap(num_cpus):
        np.random.seed(1)
        real_li.compute_residue_koff(residue_id=residues, num_cpus=num_cpus, plot_data=False)
        return real_li.dataset["Koff Bootstrap avg"].to_numpy(copy=True)[residues]

    first = bootstrap(2)
    np.testing.assert_array_equal(first, bootstrap(2))
    np.testing.assert_array_equal(first, bootstrap(3))
