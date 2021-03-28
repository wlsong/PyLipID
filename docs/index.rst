
==========================================================
PyLipID - A Python Package For Lipid Interactions Analysis
==========================================================

PyLipID is a python package for analyzing lipid interactions with membrane proteins from
Molecular Dynamics Simulations. PyLipID has the following main features, please check out
the tutorials for examples and the documentations for the API functionalities:

    * Analysis of lipid interactions with protein residues using a couple of metrics.
    * Detection of binding sites via calculating community structures in the interactions networks.
    * Analysis of lipid interactions with binding sites using a couple of metrics.
    * Calculation of lipid koff and residence time for protein residues and binding sites.
    * Generation of representative bound poses for binding sites.
    * Analysis of bound poses for binding sites via automated clustering scheme.
    * Adoption of a dual-cutoff scheme to overcome the 'cage-rattling' effect of coarse-grained simulations.
    * Generation of manuscript-ready figures for analysis.


Citation
=========

If you use PyLipID in scientific software, please cite the following paper:

PLACEHOLDER


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

A no-brainer script is available at :doc:`mics` to run PyLipID with all the analysis.


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
   mics



