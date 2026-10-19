
==========================================================
PyLipID - A Python Package For Lipid Interactions Analysis
==========================================================

PyLipID is a python package for analyzing lipid interactions with membrane proteins from
Molecular Dynamics Simulations. PyLipID has the following main features, please check out
the tutorials for examples and the documentations for the API functionalities:

    * Detection of binding sites via calculating community structures in the interactions networks.
    * Calculation of lipid koff and residence time for interaction with binding sites and residues.
    * Analysis of lipid interactions with binding sites and residues using a couple of metrics.
    * Generation of representative bound poses for binding sites.
    * Analysis of bound poses for binding sites via automated clustering scheme.
    * Adoption of a dual-cutoff scheme to overcome the 'rattling in cage' effect of coarse-grained simulations.
    * Generation of manuscript-ready figures for analysis.


Citation |DOI for Citing PyLipID|
=================================

If you use PyLipID in scientific research, please cite the following paper: ::

	@article{song_pylipid_2022,
		author = {Song, Wanling and Corey, Robin A. and Ansell, T. Bertie and
		            Cassidy, C. Keith and Horrell, Michael R. and Duncan, Anna L.
		            and Stansfeld, Phillip J. and Sansom, Mark S.P.},
		title = {PyLipID: A Python Package for Analysis of Protein-Lipid Interactions from Molecular Dynamics Simulations},
		journal = {J. Chem. Theory Comput.},
		volume = {18},
		number = {2},
		pages = {1188--1201},
		year = {2022},
		doi = {10.1021/acs.jctc.1c00708},
		url = {https://doi.org/10.1021/acs.jctc.1c00708},
	}

.. |DOI for Citing PyLipID| image:: https://img.shields.io/badge/DOI-10.1021/acs.jctc.1c00708-blue
   :target: https://doi.org/10.1021/acs.jctc.1c00708



Installation
============

PyLipID can be installed with `pip <https://pip.pypa.io>`_

.. code-block:: bash

  $ pip install pylipid

Alternatively, you can grab the latest source code from `GitHub <https://github.com/pstansfeld/PyLipID.git>`_:

.. code-block:: bash

  $ git clone https://github.com/pstansfeld/PyLipID.git
  $ cd PyLipID
  $ pip install .


Usage
=====

The :doc:`tutorial` is the place to go to learn how to use the PyLipID. The :doc:`api/index`
provides API-level documentation.

A no-brainer demo script is available at :doc:`demo` to run PyLipID with all the analysis.


License
=======

PyLipID is made available under the MIT License. For more details,
see `LICENSE.txt <https://github.com/pstansfeld/PyLipID/blob/master/LICENSE.txt>`_.


Table of Contents
=================

.. toctree::
   :maxdepth: 2

   INSTALL
   api/index
   tutorial
   gallery
   demo



