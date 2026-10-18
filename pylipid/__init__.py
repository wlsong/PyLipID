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

# --- BEGIN: macOS-safe bootstrap -------------------------------------------
# PyLipID's worker pools use the 'spawn' start method explicitly (multiprocessing.get_context("spawn")), so the
# global start method is left unchanged for the user's own code.

# Use a non-interactive backend so importing PyLipID never loads macOS AppKit
try:
    import matplotlib
    matplotlib.use("Agg", force=True)
except Exception:
    pass
# --- END: macOS-safe bootstrap ---------------------------------------------

from ._version import __version__
from .api import LipidInteraction
