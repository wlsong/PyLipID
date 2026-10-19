PyLipID - Analysis of Protein–Lipid Interactions from Molecular Dynamics
========================================================================

.. image:: https://github.com/pstansfeld/PyLipID/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/pstansfeld/PyLipID/actions/workflows/ci.yml
.. image:: https://img.shields.io/pypi/v/PyLipID
   :target: https://pypi.org/project/pylipid/
.. image:: https://img.shields.io/pypi/pyversions/PyLipID
   :target: https://pypi.org/project/pylipid/

.. image:: docs/static/pylipid_logo_smallsize.png
   :align: center

PyLipID is a Python package for analysing lipid–protein interactions from
Molecular Dynamics (MD) simulations. It supports coarse-grained and all-atom
simulations, handles formats supported by MDTraj, and works in scripts or
Jupyter notebooks.

Documentation: https://pylipid.readthedocs.io

Features
========

- Dual-cutoff contact detection to improve coarse-grained accuracy.
- Binding site detection using NetworkX Louvain community analysis.
- Kinetic parameters: koff and residence times per residue and site.
- Interaction metrics including occupancy, count, duration.
- Representative bound pose extraction and clustering.
- Binding site surface area (SASA) calculations.
- Publication-ready plots and PyMOL scripts.

Installation
============

From PyPI::

    pip install pylipid

From source::

    git clone https://github.com/pstansfeld/PyLipID
    cd PyLipID
    pip install .

Editable/development install::

    pip install -e .

Requires Python >= 3.11 (tested on 3.11–3.14).

Quick Start
===========

.. code-block:: python

    from pylipid.api import LipidInteraction

    trajfile_list = ["run1/protein_lipids.xtc", "run2/protein_lipids.xtc"]
    topfile_list  = ["run1/protein_lipids.gro", "run2/protein_lipids.gro"]

    li = LipidInteraction(
        trajfile_list,
        topfile_list=topfile_list,
        cutoffs=[0.55, 0.8],
        lipid="CHOL",
        nprot=1,
    )

    li.collect_residue_contacts()
    li.compute_residue_duration()
    li.compute_residue_koff()
    li.compute_binding_nodes(threshold=4)
    li.compute_site_koff()

    li.plot("Residence Time")
    li.save_data("Dataset")
    li.save_pymol_script(pdb_file="receptor.pdb")

PyLipID runs its calculations in parallel worker processes. When running it as a script, put the
code under ``if __name__ == "__main__":`` (required on macOS and Windows, where the script would otherwise
fail or hang); Jupyter notebooks need no change.

What's New in v1.6.0
====================

Changes to Results
------------------
- Binding site surface areas (compute_surface_area) now use the MARTINI bead radii PyLipID has always
  documented (BB 0.26 nm, SC1-SC4 0.23 nm) and any ``radii`` given. Before, mdtraj matched these against
  element symbols, so they were ignored and the radii of boron and sulfur (0.19/0.18 nm) were used. Each
  frame is also calculated separately, working around an mdtraj issue that adds part of the previous
  frame's area to each frame in a multi-frame call. Site surface areas are typically 5-20% smaller than
  in earlier versions.
- With ``np.random.seed()`` set in the calling script, bootstrapped koffs and the poses chosen to
  represent pose clusters are now reproducible (worker processes previously ignored the seed). Note that
  with scipy 1.17 (the last release for Python 3.11) curve_fit does not always return the same fit for
  the same data, so koff values can still vary slightly between runs there; scipy >= 1.18 (Python >= 3.12)
  is repeatable.

Bug Fixes
---------
- analyze_bound_poses no longer fails when many bound poses are identical (DBSCAN eps of 0), and a
  ``min_samples`` passed to it is now used instead of being overwritten.
- Corrected binding-site detection off-by-one error.
- Fixed calculate_scores returning None.
- Fixed sparse_corrcoef matrix type issues.
- Fixed DataFrame column/index mismatches in analyze_bound_poses.
- Fixed missing return in compute_residue_lipidcount.
- Fixed a crash when analyze_bound_poses was run more than once with pandas 3.
- cluster_KMeans now always returns labels (it returned a tuple when there were fewer poses than clusters).
- plot, plot_logo, save_coordinate and write_site_info calculate missing residue data on demand instead of raising KeyError.
- Fixed write_PDB errors for .gro files lacking chain IDs.
- Fixed figure overwriting in plot_koff.
- Updated deprecated Matplotlib calls.
- Fixed LogNorm crash on zero-containing matrices.
- Replaced np.float128 with np.longdouble.
- Corrected output directory typo.

Performance
-----------
- collect_residue_contacts computes distances in large batches instead of once per lipid, and shares
  the batches between worker processes that read the coordinates from shared memory (``num_cpus``;
  based on Alex Moger's parallel contacts in `PR #1 <https://github.com/pstansfeld/PyLipID/pull/1>`_).
  Results are identical to the serial calculation; on a
  2 x 2,000-frame system contacts take 8 s instead of 40 s.
- Bound-pose scoring evaluates the kernel density estimate in numpy (~20x faster; same scores to
  within floating-point rounding).
- Surface areas are shared between worker processes by chunks of frames rather than by trajectory.
- Progress bars for parallel steps now follow the calculation.
- Survival-function calculation for koff is vectorised (identical results, ~20% faster koff).
- Nested progress bars implemented.

Dependencies
------------
- Removed python-louvain; using NetworkX built-in Louvain.
- Removed statsmodels and p_tqdm (no longer used).
- Minimum versions added.
- Dropped Python < 3.11; supports Python 3.11–3.14, numpy 2, pandas 3 and current mdtraj.

Packaging & CI
--------------
- Migrated to pyproject.toml.
- CI moved from Travis CI to GitHub Actions.

Citation
========

.. image:: https://img.shields.io/badge/DOI-10.1021/acs.jctc.1c00708-blue
   :target: https://doi.org/10.1021/acs.jctc.1c00708

::

    @article{song_pylipid_2022,
        author  = {Song, Wanling and Corey, Robin A. and Ansell, T. Bertie and
                   Cassidy, C. Keith and Horrell, Michael R. and Duncan, Anna L.
                   and Stansfeld, Phillip J. and Sansom, Mark S.P.},
        title   = {PyLipID: A Python Package for Analysis of Protein-Lipid
                   Interactions from Molecular Dynamics Simulations},
        journal = {J. Chem. Theory Comput.},
        volume  = {18},
        number  = {2},
        pages   = {1188--1201},
        year    = {2022},
        doi     = {10.1021/acs.jctc.1c00708},
        url     = {https://doi.org/10.1021/acs.jctc.1c00708},
    }

License
=======

MIT License. See LICENSE.txt.
