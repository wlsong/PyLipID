
==========================================================
PyLipID - A Python Package For Lipid Interactions Analysis
==========================================================

.. image:: https://travis-ci.com/wlsong/PyLipID.svg?branch=master
   :target: https://travis-ci.com/github/wlsong/PyLipID
.. image:: https://img.shields.io/pypi/v/PyLipID
   :target: https://pypi.org/project/pylipid/

.. image:: docs/static/pylipid_logo_smallsize.png
    :align: center


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

PyLipID can be used from Jupyter (former IPython, recommended), or by writing Python scripts.
The documentaion and tutorials can be found at `pylipid.readthedocs.io <https://pylipid.readthedocs.io>`_.

Installation
============

We recommend installing PyLipID using the package installer `pip`:

``pip install pylipid``

Alternatively, PyLipID can be installed from the source code. The package is available for
download on Github via:

``git clone https://github.com/wlsong/PyLipID``

Once the source code is downloaded, enter the source code directory and install the package as follow:

``python setup.py install``


Citation |DOI for Citing PyEMMA|
================================

If you use PyLipID in scientific research, please cite the following paper: ::

	@article{song_pylipid_2022,
		author = {Song, Wanling. and Corey, Robin A. and Ansell, T. Bertie. and
		            Cassidy, C. Keith. and Horrell, Michael R. and Duncan, Anna L.
		            and Stansfeld, Phillip J. and Sansom, Mark S.P.},
		title = {PyLipID: A Python package for analysis of protein-lipid interactions from MD simulations},
		journal = {J. Chem. Theory Comput},
		year = {2022},
		url = {https://doi.org/10.1021/acs.jctc.1c00708},
		doi = {10.1021/acs.jctc.1c00708},
		urldate = {2022-02-18},
	}

.. |DOI for Citing PyEMMA| image:: https://img.shields.io/badge/DOI-10.1021/acs.jctc.1c00708-blue
   :target: https://doi.org/10.1021/acs.jctc.1c00708
