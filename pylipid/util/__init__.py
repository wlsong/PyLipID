##############################################################################
# PyLipID: A python module for analysing protein-lipid interactions
#
# Author: Wanling Song
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
##############################################################################


r"""
util module
=============
The ``util`` module contains other assisting functions:

.. currentmodule:: pylipid.util

.. autosummary::
   :toctree: generated/

   check_dir
   write_PDB
   write_pymol_script
   sparse_corrcoef
   rmsd
   get_traj_info

"""


from .directory import check_dir
from .coordinate import write_PDB
from .pymol_script import write_pymol_script
from .corrcoef import sparse_corrcoef
from .rmsd import rmsd
from .trajectory import get_traj_info