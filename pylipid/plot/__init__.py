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
plot module
============
The ``plot`` module provides functions for aiding in the scientific analysis:

.. currentmodule:: pylipid.plot

.. autosummary::
   :toctree: generated/

   plot_koff
   plot_residue_data
   plot_residue_data_logo
   surface_area
   plot_surface_area
   plot_binding_site_data
   plot_corrcoef

"""


from .koff import plot_koff
from .plot1d import plot_residue_data
from .plot1d import plot_residue_data_logos
from .plot1d import plot_surface_area
from .plot1d import plot_binding_site_data
from .plot2d import plot_corrcoef

