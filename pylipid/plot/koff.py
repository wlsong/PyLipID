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

"""This module contains functions for plotting koff.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

__all__ = ["plot_koff"]


def plot_koff(durations, delta_t_list, survival_rates, n_fitted,
              survival_rates_bootstraps=None, fig_fn=None, title=None,
              timeunit=None, text=None, t_total=None, fig_close=True):
    """Plot the koff figure.

    The koff figure contains two axes. The left axis plot the sorted
    interaction durations, and the right one plot normalised survival rates.

    Parameters
    ----------
    durations : array_like
            Collected interaction durations.
    delta_t_list : array_like
            List of :math:`\Delta t` at which the survival rates are calculated.
    survival_rates : array_like
            Survival rates calculated at delta_t_list.
    n_fitted: array_like
            The values of fitted bi-exponential at delta_t_list.
    survival_rates_bootstraps : list of array, optional, default=None
            List of bootstrapped survival rates.
    fig_fn : str, optional, default=None
            Name of the koff figure. by default the figure will be saved as "koff.png"
            in the current working directory.
    title : str, optional, default=None
            Figure title. Default is None.
    t_total : scalar, optional, default=None
            Duration of simulation trajectories. The xlim of both axes will set to t_total if
            a value is given, otherwise xlim will be determined by matplotlib.
    timeunit : {"ns", "us", None}, optional, default=None
            Time unit of the given durations. Default is None.
    text : str, optional, default=None
            Text printed next to the koff figure. The default is None.


    """
    # plot settings
    if timeunit is None:
        xlabel = "Duration (timeunit)"
    elif timeunit == "ns":
        xlabel = "Duration (ns)"
    elif timeunit == "us":
        xlabel = r"Duration ($\mu s$)"

    if text is None:
        fig = plt.figure(1, figsize=(5.5, 3.5))
        left, width = 0.13, 0.33
        bottom, height = 0.17, 0.68
        left_h = left + width + 0.05
        rect_scatter = [left, bottom, width, height]
        rect_histy = [left_h, bottom, width, height]
        axScatter = fig.add_axes(rect_scatter)
        axHisty = fig.add_axes(rect_histy)
    else:
        fig = plt.figure(1, figsize=(8.2, 3.5))
        left, width = 0.0975, 0.23
        bottom, height = 0.17, 0.68
        left_h = left + width + 0.0375
        rect_scatter = [left, bottom, width, height]
        rect_histy = [left_h, bottom, width, height]
        axScatter = fig.add_axes(rect_scatter)
        axHisty = fig.add_axes(rect_histy)

    # plot original data
    x = np.sort(durations)
    y = np.arange(len(x)) + 1
    axScatter.scatter(x[::-1], y, label="Contacts", s=10, c="#176BA0")
    axScatter.set_xlim(0, x[-1] * 1.1)
    axScatter.legend(loc="upper right", prop={"size": 10, "weight": "bold"}, frameon=False, handletextpad=0.1)
    axScatter.set_ylabel("Sorted Index", fontsize=10, weight="bold")
    axScatter.set_xlabel(xlabel, fontsize=10, weight="bold")
    # plot survival function
    axHisty.scatter(delta_t_list, survival_rates, zorder=8, s=10, label="Survival func.", c="#7a5195")
    axHisty.yaxis.set_label_position("right")
    axHisty.yaxis.tick_right()
    axHisty.set_xlabel(r"$\Delta$t", fontsize=10, weight="bold")
    axHisty.set_ylabel("Probability", fontsize=10, weight="bold")
    axHisty.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axHisty.set_ylim(-0.1, 1.1)
    # plot the fitted curve
    axHisty.plot(delta_t_list, n_fitted, 'r--', linewidth=3, zorder=10, label="Fitted biexpo.")
    # plot bootstrapped survival functions
    if survival_rates_bootstraps is not None:
        for boot_idx, survival_rates_boot in enumerate(np.atleast_2d(survival_rates_bootstraps)):
            if boot_idx == 0:
                axHisty.plot(delta_t_list, survival_rates_boot, color="gray", alpha=0.5,
                             label="Bootstrapping", linewidth=3)
            else:
                axHisty.plot(delta_t_list, survival_rates_boot, color="gray", alpha=0.5, linewidth=3)

    axHisty.legend(loc="upper right", prop={"size": 8, "weight": "bold"}, frameon=False)
    # set xlim
    if t_total is not None:
        axScatter.set_xlim(0, t_total)
        axHisty.set_xlim(0, t_total)

    if title is not None:
        fig.text(0.13, 0.89, title, fontdict={"size":12, "weight": "bold"})

    # set ticklabel fonts
    for ax in [axHisty, axScatter]:
        for label in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
            plt.setp(label, fontsize=10, weight="bold")

    # print text on the right
    axHisty.text(1.4, 1.0, text, verticalalignment='top', horizontalalignment='left', transform=axHisty.transAxes,
                 fontdict={"size": 8, "weight": "normal"}, linespacing=2)

    if fig_fn is None:
        fig_fn = os.path.join(os.getcwd(), "koff.pdf")
    fig.savefig(fig_fn, dpi=300)
    if fig_close:
        plt.close()

    return

