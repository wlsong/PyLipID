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

"""This module contains functions for 2D plot.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

__all__ = ["plot_corrcoef"]


def plot_corrcoef(corrcoef, residue_index, cmap="Reds", vmin=None, vmax=None,
                  fn=None, title=None, fig_close=False):
    """Plot correlation coefficient matrix.

    Parameters
    ----------
    corrcoef : array_like
        A scalar 2D array of correlation coefficient matrix.
    residue_index : array_list, optional, default=None
        A 1D array of residue index.
    cmap : str or `matplotlib.colors.Colormap`, optional, default="coolwarm"
        A Colormap instance or matplotlib register colormap name. The
        colormap maps *corrcoef* to colors.
    fn : str, optional, default=None
        Figure name. By default the figure is saved as "Figure_corrcoef.pdf" as the current
        working directory.
    title : str, optional, default=None

    """
    plt.rcParams["font.size"] = 10
    plt.rcParams["font.weight"] = "bold"

    if fn is None:
        fn = os.path.join(os.getcwd(), "Figure_Correlation_Matrix.pdf")

    if len(corrcoef) <= 20:
        fig, ax = plt.subplots(1, 1, figsize=(2.5, 1.8))
        majorlocator = 5
        minorlocator = 1
    elif 20 < len(corrcoef) <= 50:
        fig, ax = plt.subplots(1, 1, figsize=(2.9, 2.0))
        majorlocator = 10
        minorlocator = 1
    elif 50 < len(corrcoef) <= 500:
        fig, ax = plt.subplots(1, 1, figsize=(4.9, 3.5))
        majorlocator = 50
        minorlocator = 10
    elif 500 <= len(corrcoef) < 1000:
        fig, ax = plt.subplots(1, 1, figsize=(5.9, 4.5))
        majorlocator = 100
        minorlocator = 10
    elif 1000 <= len(corrcoef) < 2000:
        fig, ax = plt.subplots(1, 1, figsize=(7.9, 6.5))
        majorlocator = 200
        minorlocator = 20
    elif len(corrcoef) >= 2000:
        fig, ax = plt.subplots(1, 1, figsize=(8.9, 7.5))
        majorlocator = 500
        minorlocator = 100

    # sort index, check duplicates in residue index.
    majorticks = []
    minorticks = []
    ticklabels = []
    breaks = []
    for idx, resi in enumerate(residue_index):
        if not resi%majorlocator:
            majorticks.append(idx-0.5)
            ticklabels.append(resi)
        elif not resi%minorlocator:
            minorticks.append(idx-0.5)
        if idx > 0 and resi - residue_index[idx-1] != 1:
            breaks.append(idx-0.5)

    x = y = np.arange(len(residue_index)+1, dtype=float)
    x -= 0.5
    y -= 0.5

    corrcoef = np.nan_to_num(corrcoef)
    if vmax is None:
        vmax = np.percentile(np.unique(np.ravel(corrcoef)), 99)
    if vmin is None:
        vmin = np.percentile(np.unique(np.ravel(corrcoef)), 1)
    pcm = ax.pcolormesh(x, y, corrcoef, cmap=cmap, 
                        norm=colors.LogNorm(vmax=vmax, vmin=vmin))
    fig.colorbar(pcm, ax=ax)
    # set ticks
    ax.set_xticks(majorticks)
    ax.set_xticks(minorticks, minor=True)
    ax.xaxis.set_ticklabels(ticklabels, fontsize=10, weight="bold")
    ax.set_yticks(majorticks)
    ax.set_yticks(minorticks, minor=True)
    ax.yaxis.set_ticklabels(ticklabels, fontsize=10, weight="bold")
    ax.tick_params(which='both', direction='out')

    ax.set_xlabel("Residue Index", fontsize=10, weight="bold")
    ax.set_ylabel("Residue Index", fontsize=10, weight="bold")

    if len(breaks) > 0:
        for break_line in breaks:
            ax.axhline(break_line, linewidth=0.8, color="black", linestyle="--")
            ax.axvline(break_line, linewidth=0.8, color="black", linestyle="--")

    if title is not None:
        ax.set_title(title, fontsize=8, weight="bold")

    plt.tight_layout()
    fig.savefig(fn, dpi=200)
    if fig_close:
        plt.close()

    return

