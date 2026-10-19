"""
Tests for utility functions: write_pymol_script, rmsd, sparse_corrcoef, get_traj_info.
Uses the synthetic_traj_dir session fixture from conftest.py for trajectory-dependent tests.
write_pymol_script and get_traj_info are tested with synthetic data — no real files needed.
"""

import os
import unittest
import numpy as np
import shutil
import pytest
import mdtraj as md
from pylipid.util import write_pymol_script, check_dir, rmsd, sparse_corrcoef, get_traj_info


class TestRmsdAndCorrcoef(unittest.TestCase):
    """Pure-function tests — no trajectory data needed."""

    def test_rmsd(self):
        matrix_a = np.random.random(size=(100, 5))
        matrix_b = np.random.random(size=(100, 5))
        value = rmsd(matrix_a, matrix_b)
        self.assertIsInstance(value, float)

    def test_sparse_corrcoef_ndarray(self):
        """sparse_corrcoef should accept a plain ndarray as well as a sparse matrix."""
        A = np.random.normal(size=(4, 500))
        corrcoefs = sparse_corrcoef(A)
        self.assertEqual(len(corrcoefs), len(A))


# ---------------------------------------------------------------------------
# Trajectory-dependent tests — use pytest fixtures via a thin wrapper
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("synthetic_traj_dir")
class TestTrajUtils:
    """Tests that require a trajectory. Uses synthetic_traj_dir fixture."""

    def test_get_traj_info(self, synthetic_traj_dir, tmp_path):
        trajfile = synthetic_traj_dir["trajfile_list"][0]
        topfile  = synthetic_traj_dir["topfile_list"][0]
        traj = md.load(trajfile, top=topfile)
        traj_info, protein_ref, lipid_ref = get_traj_info(traj, synthetic_traj_dir["lipid"])
        assert isinstance(protein_ref, md.Trajectory)
        assert isinstance(lipid_ref, md.Trajectory)

    def test_write_pymol_script(self, synthetic_traj_dir, tmp_path):
        # write_pymol_script needs a PDB and a CSV file — generate minimal ones
        pdb_file = synthetic_traj_dir["pdb_file"]

        # Minimal Dataset CSV that write_pymol_script can read
        csv_file = str(tmp_path / "Interactions_CHOL.csv")
        with open(csv_file, "w") as f:
            f.write("Residue,Residue ID,Binding Site ID,Residence Time\n")
            for i in range(5):
                f.write(f"{i}ALA,{i},0,{float(i)}\n")

        script_fn = str(tmp_path / "show_bs_info.py")
        write_pymol_script(script_fn, pdb_file, csv_file, "CHOL", 1)
        assert os.path.exists(script_fn)
