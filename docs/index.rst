
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


Citation |DOI for Citing PyEMMA|
================================

If you use PyLipID in scientific research, please cite the following paper: ::

	@article{song_pylipid_2021,
		author = {Song, Wanling. and Corey, Robin A. and Ansell, T. Bertie. and
		            Cassidy, C. Keith. and Horrell, Michael R. and Duncan, Anna L.
		            and Stansfeld, Phillip J. and Sansom, Mark S.P.},
		title = {PyLipID: A Python package for analysis of protein-lipid interactions from MD simulations},
		journal = {BioRxiv},
		year = {2021},
		url = {https://doi.org/10.1101/2021.07.14.452312},
		doi = {10.1101/2021.07.14.452312},
		urldate = {2021-07-14},
		month = jul,
	}

.. |DOI for Citing PyEMMA| image:: https://img.shields.io/badge/DOI-10.1101/2021.07.14.452312-blue
   :target: https://doi.org/10.1101/2021.07.14.452312



Installation
============

PyLipID can be installed with `pip <https://pip.pypa.io>`_

.. code-block:: bash

  $ pip install pylipid

Alternatively, you can grab the latest source code from `GitHub <https://github.com/wlsong/PyLipID.git>`_:

.. code-block:: bash

  $ git clone git://github.com/wlsong/PyLipID.git
  $ python setup.py install


Usage
=====

The :doc:`tutorial` is the place to go to learn how to use the PyLipID. The :doc:`api/index`
provides API-level documentation.

A no-brainer demo script is available at :doc:`demo` to run PyLipID with all the analysis.


License
=======

PyLipID is made available under the MIT License. For more details,
see `LICENSE.txt <https://github.com/wlsong/PyLipID/blob/master/LICENSE.txt>`_.


Table of Contents
=================

.. toctree::
   :maxdepth: 2

   INSTALL
   api/index
   tutorial
   gallery
   demo



