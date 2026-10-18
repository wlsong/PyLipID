"""
Pytest fixtures for PyLipID tests.

Generates small synthetic MDTraj trajectories in a temporary directory so that
CI can run without any committed trajectory data files.

Topology:
  - 30 protein residues (ALA/GLY alternating, one chain)
  - 20 CHOL lipid molecules (3 atoms each: C1, C2, O1)
  - Cubic box, 8 nm side
  - 2 trajectory files, 100 frames each
"""

import os
import numpy as np
import pytest
import mdtraj as md
from pylipid.api import LipidInteraction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AA_NAMES = ["ALA", "GLY", "VAL", "LEU", "ILE", "PRO", "PHE", "TRP",
             "MET", "SER"]
_N_PROTEIN_RESIDUES = 30
_N_LIPID_MOLECULES  = 20
_N_FRAMES           = 100
_BOX_NM             = 8.0          # cubic box side length in nm
_N_RUNS             = 2            # number of trajectory files


def _build_topology():
    """Build a minimal MDTraj topology: protein + CHOL lipids."""
    top = md.Topology()

    # --- protein chain ---
    pchain = top.add_chain()
    for i in range(_N_PROTEIN_RESIDUES):
        resname = _AA_NAMES[i % len(_AA_NAMES)]
        res = top.add_residue(resname, pchain)
        top.add_atom("CA",  md.element.carbon,   res)
        top.add_atom("CB",  md.element.carbon,   res)
        top.add_atom("N",   md.element.nitrogen, res)
        top.add_atom("C",   md.element.carbon,   res)
        top.add_atom("O",   md.element.oxygen,   res)

    # --- lipid chain ---
    lchain = top.add_chain()
    for _ in range(_N_LIPID_MOLECULES):
        res = top.add_residue("CHOL", lchain)
        top.add_atom("C1",  md.element.carbon, res)
        top.add_atom("C2",  md.element.carbon, res)
        top.add_atom("O1",  md.element.oxygen, res)

    return top


def _build_trajectory(top, rng, n_frames=_N_FRAMES):
    """
    Build a synthetic trajectory.

    Protein residues are placed in a cluster near the box centre.
    Lipid molecules are placed nearby so that some will fall within
    the contact cutoffs and generate non-trivial interaction data.
    """
    n_atoms = top.n_atoms
    xyz = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)

    atoms_per_prot_res = 5   # CA CB N C O
    atoms_per_lipid    = 3   # C1 C2 O1
    n_prot_atoms = _N_PROTEIN_RESIDUES * atoms_per_prot_res

    centre = _BOX_NM / 2.0

    for frame in range(n_frames):
        # protein — small fluctuations around centre
        prot_base = rng.uniform(centre - 1.0, centre + 1.0,
                                size=(n_prot_atoms, 3)).astype(np.float32)
        prot_noise = rng.normal(0, 0.05, size=prot_base.shape).astype(np.float32)
        xyz[frame, :n_prot_atoms] = prot_base + prot_noise

        # lipids — placed close enough to protein to generate contacts
        for lip_idx in range(_N_LIPID_MOLECULES):
            start = n_prot_atoms + lip_idx * atoms_per_lipid
            # Alternate between near (within cutoff) and far positions
            if lip_idx < _N_LIPID_MOLECULES // 2:
                offset = rng.uniform(0.3, 0.6, size=(atoms_per_lipid, 3))
            else:
                offset = rng.uniform(1.5, 3.0, size=(atoms_per_lipid, 3))
            sign = rng.choice([-1, 1], size=(atoms_per_lipid, 3))
            xyz[frame, start:start + atoms_per_lipid] = (
                centre + sign * offset.astype(np.float32)
            )

    unitcell_lengths = np.full((n_frames, 3), _BOX_NM, dtype=np.float32)
    unitcell_angles  = np.full((n_frames, 3), 90.0,    dtype=np.float32)

    return md.Trajectory(xyz, top,
                         unitcell_lengths=unitcell_lengths,
                         unitcell_angles=unitcell_angles)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_traj_dir(tmp_path_factory):
    """
    Session-scoped fixture: builds synthetic trajectory files once per test
    session and returns a dict with file paths and metadata.

    Returns
    -------
    dict with keys:
        trajfile_list  : list of .xtc paths
        topfile_list   : list of .gro paths
        pdb_file       : path to receptor .pdb
        lipid          : lipid residue name (str)
        cutoffs        : [lower, upper] cutoffs in nm
    """
    rng = np.random.default_rng(42)
    top = _build_topology()
    base = tmp_path_factory.mktemp("traj_data")

    trajfile_list = []
    topfile_list  = []

    for run_idx in range(_N_RUNS):
        run_dir = base / "run{}".format(run_idx + 1)
        run_dir.mkdir()

        traj = _build_trajectory(top, rng)

        xtc_path = str(run_dir / "protein_lipids.xtc")
        gro_path = str(run_dir / "protein_lipids.gro")

        traj.save_xtc(xtc_path)
        traj[0].save_gro(gro_path)

        trajfile_list.append(xtc_path)
        topfile_list.append(gro_path)

    # receptor PDB (protein atoms only)
    prot_atom_indices = top.select("protein")
    prot_traj = traj[0].atom_slice(prot_atom_indices)
    pdb_path = str(base / "receptor.pdb")
    prot_traj.save_pdb(pdb_path)

    return {
        "trajfile_list": trajfile_list,
        "topfile_list":  topfile_list,
        "pdb_file":      pdb_path,
        "lipid":         "CHOL",
        "cutoffs":       [0.55, 0.8],
    }


# ---------------------------------------------------------------------------
# Real data (tests/data)
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@pytest.fixture(scope="session")
def real_traj_dir():
    """
    The committed coarse-grained trajectories in tests/data: two 251-frame runs
    of a 302-residue protein in a membrane with CHOL and six other lipids.

    Interactions_CHOL.csv holds PyLipID results for the full-length (2501-frame)
    versions of these runs, so values differ, but its binding sites should be
    recovered.
    """
    runs = [os.path.join(_DATA_DIR, "run{}".format(i)) for i in (1, 2)]
    return {
        "trajfile_list": [os.path.join(r, "protein_lipids.xtc") for r in runs],
        "topfile_list":  [os.path.join(r, "protein_lipids.gro") for r in runs],
        "pdb_file":      os.path.join(_DATA_DIR, "receptor.pdb"),
        "reference_csv": os.path.join(_DATA_DIR, "Interactions_CHOL.csv"),
        "lipid":         "CHOL",
        "cutoffs":       [0.55, 0.8],
    }


@pytest.fixture(scope="session")
def real_li(real_traj_dir, tmp_path_factory):
    """
    LipidInteraction on the real trajectories with residue contacts, residue
    kinetics and binding sites computed once per test session.
    """
    np.random.seed(0)
    li = LipidInteraction(
        real_traj_dir["trajfile_list"],
        cutoffs=real_traj_dir["cutoffs"],
        topfile_list=real_traj_dir["topfile_list"],
        lipid=real_traj_dir["lipid"],
        nprot=1,
        save_dir=str(tmp_path_factory.mktemp("real_output")),
    )
    li.collect_residue_contacts()
    li.compute_residue_koff(plot_data=False)
    li.compute_binding_nodes(threshold=4, print_data=False)
    return li
