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

"""This module contains functions that plot interactions as a function of residue index.
"""

import os
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import logomaker

__all__ = ["plot_residue_data", "plot_residue_data_logos",
           "plot_binding_site_data", "plot_surface_area", "AxisIndex"]


def plot_residue_data(residue_index, interactions, gap=200, ylabel=None,
                      fn=None, title=None, fig_close=False):
    """Plot interactions as a function of residue index
    
    Parameters
    ----------
    residue_index : list
            Residue indices in an ascending order. If a residue index is smaller than its preceding one,
            the plotting function will consider it as the start of a new chain and will plot the following
            in a new figure. A gap of less than 50 missing residues will be marked as gray areas in the figure.
    interactions : list
            Plotting values correspond to residue_index.
    gap : int, optional, default=200
            The number of missing residues in residue_index that initiate a new figure. The gap between two adjacent
            index in residue_index that is smaller than the provided value will be considered as missing residues
            and will be marked as gray areas in the figure, whereas a gap that is larger than the provided value
            will start a new figure and plot the following data in that new figure. This can help to make figures
            more compressed. The default gap is 200.
    ylabel : str, optional, default=None
            y axis label. Default is "Interactions".
    fn : str, optional, default=None
            Figure name. By default the figure is saved as "Figure_interactions.pdf" as the current
            working directory.
    title : str, optional, default=None
            Figure title.

    """
    bar_color = "#176BA0"
    if ylabel is None:
        ylabel = "Interactions"

    if fn is None:
        fn = os.path.join(os.getcwd(), "Figure_interactions.pdf")

    # check for chain breaks
    gray_areas = defaultdict(list)  # show grey area to indicate missing residues
    chain_starts = [0]  # plot in separate figures if the gap between two adjacent residues is larger than 50
    for idx in np.arange(1, len(residue_index)):
        if residue_index[idx] - residue_index[idx - 1] < 0:
            chain_starts.append(idx)
        elif residue_index[idx] - residue_index[idx - 1] > gap:
            chain_starts.append(idx)
        elif 1 < residue_index[idx] - residue_index[idx - 1] <= gap:
            gray_areas[chain_starts[-1]].append([residue_index[idx - 1] + 1, residue_index[idx] - 1])
    chain_starts.append(len(residue_index))

    # plot
    for chain_idx in np.arange(len(chain_starts[:-1])):
        df = interactions[chain_starts[chain_idx]:chain_starts[chain_idx + 1]]
        resi_selected = residue_index[chain_starts[chain_idx]:chain_starts[chain_idx + 1]]
        if 0 < len(df) <= 20:
            fig, ax = plt.subplots(1, 1, figsize=(2.8, 1.5))
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
        elif 20 < len(df) <= 50:
            fig, ax = plt.subplots(1, 1, figsize=(3.2, 1.5))
            ax.xaxis.set_major_locator(MultipleLocator(10))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
        elif 50 < len(df) <= 300:
            fig, ax = plt.subplots(1, 1, figsize=(3.8, 1.8))
            ax.xaxis.set_major_locator(MultipleLocator(50))
            ax.xaxis.set_minor_locator(MultipleLocator(10))
        elif 300 < len(df) <= 1000:
            fig, ax = plt.subplots(1, 1, figsize=(4.5, 1.8))
            ax.xaxis.set_major_locator(MultipleLocator(100))
            ax.xaxis.set_minor_locator(MultipleLocator(10))
        elif 1000 < len(df) <= 2000:
            fig, ax = plt.subplots(1, 1, figsize=(6.0, 1.8))
            ax.xaxis.set_major_locator(MultipleLocator(200))
            ax.xaxis.set_minor_locator(MultipleLocator(50))
        elif len(df) > 2000:
            fig, ax = plt.subplots(1, 1, figsize=(7.5, 1.8))
            ax.xaxis.set_major_locator(MultipleLocator(500))
            ax.xaxis.set_minor_locator(MultipleLocator(100))
        ax.bar(resi_selected, df, 1.0, linewidth=0, color=bar_color)
        # plot missing residue area
        if chain_starts[chain_idx] in gray_areas.keys():
            for gray_area in gray_areas[chain_starts[chain_idx]]:
                ax.axvspan(gray_area[0], gray_area[1], facecolor="#c0c0c0", alpha=0.3)
        # axis setting
        ax.set_ylim(0, df.max() * 1.05)
        ax.set_xlim(resi_selected.min() - 1, resi_selected.max() + 1)
        ax.set_ylabel(ylabel, fontsize=8, weight="bold")
        ax.set_xlabel("Residue Index", fontsize=8, weight="bold")
        for label in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
            plt.setp(label, fontsize=8, weight="bold")
        if title is not None:
            ax.set_title(title, fontsize=8, weight="bold")
        plt.tight_layout()
        if len(chain_starts) == 2:
            fig.savefig(fn, dpi=300)
        else:
            name, ext = os.path.splitext(fn)
            fig.savefig("{}_{}{}".format(name, chain_idx, ext), dpi=300)
        if fig_close:
            plt.close()

    return


def plot_residue_data_logos(residue_index, logos, interactions, gap=1000, letter_map=None,
                            color_scheme="chemistry", ylabel=None, title=None, fn=None, fig_close=False):
    """Plot interactions using `logomaker.Logo
    <https://logomaker.readthedocs.io/en/latest/implementation.html#logo-class>`_.

    Parameters
    -----------
    residue_index : list
            Residue indices in an ascending order. If a residue index is smaller than its preceding one,
            the plotting function will consider it as the start of a new chain and will plot the following
            in a new figure.
    logos : list of str
            Single letter logos in the corresponding order as residue_index. The height of logos in the figure
            will be determined by values given in interactions. Three-letter name of the 20 common amino acids
            are accepted and will be converted to their corresponding single-letter names in this function by
            the default. Other mappings can be defined via letter_map.
    interactions : list
            Plotting values in the corresponding order as residue_index.
    gap : int, optional, default=1000
            The number of missing residues in residue_index that starts a new figure. A gap between two adjacent
            index in residue_index that is smaller than the provided value will be considered as missing residues
            and will be marked as gray areas in the figure, whereas a gap that is larger than the provided value
            will start a new figure and plot the following data in that new figure. This can help to make figures
            more compressed. The gap needs to be greater than 1000. The default is 1000.
    letter_map : dict, optional, default=None
            A dictionary that maps provided names to single-letter logos in the form of
            {"provided name": "single_letter logo"}.
    color_scheme : str, optional, default="chemistry"
            The color scheme used by logomaker.Logo(). See
            `Color Schemes <https://logomaker.readthedocs.io/en/latest/examples.html#color-schemes>`_ for accepted values.
            Default is "chemistry".
    ylabel : str, optional, default=None
            y axis label. Default is "Interactions".
    fn : str, optional, default=None
            Figure name. By default the figure is saved as "Figure_interactions_logo.pdf" as the current
            working directory.
    fig_close : bool, optional, default=False

    """
    # single-letter dictionary
    single_letter = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
                     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
                     'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
    if letter_map is not None:
        single_letter.update(letter_map)

    logos_checked = []
    for name in logos:
        if len(name) == 1:
            logos_checked.append(name)
        else:
            logos_checked.append(single_letter[name])
    if ylabel is None:
        ylabel = "Interactions"
    if fn is None:
        fn = os.path.join(os.getcwd(), "Figure_interactions_logo.pdf")

    length = 100
    # check for chain breaks, gray_areas and axis breaks
    axis_obj = AxisIndex(residue_index, logos_checked, interactions, length, gap)
    axis_obj.sort()
    # plot
    for page_idx in axis_obj.breaks.keys():
        n_rows = len(axis_obj.breaks[page_idx])
        fig, axes = plt.subplots(n_rows, 1, figsize=(4.5, 1.3 * n_rows), sharey=True)
        plt.subplots_adjust(hspace=0.5, left=0.2)
        ymax = []
        for ax_idx, ax in enumerate(np.atleast_1d(axes)):
            resi_selected = [item[0] for item in axis_obj.breaks[page_idx][ax_idx]]
            logos_selected = [item[1] for item in axis_obj.breaks[page_idx][ax_idx]]
            interaction_selected = [item[2] for item in axis_obj.breaks[page_idx][ax_idx]]
            ymax.append(np.max(interaction_selected))
            if np.sum(interaction_selected) > 0:
                df = pd.DataFrame({"Resid": resi_selected, "Resn": logos_selected, "Data": interaction_selected})
                matrix = df.pivot(index="Resid", columns='Resn', values="Data").fillna(0)
                logomaker.Logo(matrix, color_scheme=color_scheme, ax=ax)
            if ax_idx == (n_rows - 1):
                ax.set_xlabel("Residue Index", fontsize=8, weight="bold")
            ax.xaxis.set_major_locator(MultipleLocator(20))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.set_xlim(resi_selected[0] - 0.5, resi_selected[-1] + 0.5)
            ax.set_ylabel(ylabel, fontsize=8, weight="bold", va="center")
            for label in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
                plt.setp(label, fontsize=8, weight="bold")
        np.atleast_1d(axes)[-1].set_ylim(0, np.max(ymax) * 1.05)
        # plot missing areas
        if page_idx in axis_obj.gray_areas.keys():
            for item in axis_obj.gray_areas[page_idx]:
                np.atleast_1d(axes)[item[0]].axvspan(item[1], item[2], facecolor="#c0c0c0", alpha=0.3)
        if title is not None:
            np.atleast_1d(axes)[0].set_title(title, fontsize=10, weight="bold")
        plt.tight_layout()
        if len(axis_obj.breaks.keys()) == 1:
            fig.savefig(fn, dpi=300)
        else:
            name, ext = os.path.splitext(fn)
            fig.savefig("{}_{}{}".format(name, page_idx, ext), dpi=300)
        if fig_close:
            plt.close()

    return


def plot_binding_site_data(data, fig_fn, ylabel=None, title=None, fig_close=False):
    """Plot surface area in a matplotlib violin plot.

    Parameters
    ----------
    data : padnas.DataFrame
    fig_fn : str
    ylabel : str, optional, default=None
    title : str, optional, default=None
    fig_close : bool

    """
    from itertools import cycle as _cycle

    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value

    if ylabel is None:
        ylabel = ""
    if title is None:
        title = ""

    color_set = _cycle(plt.get_cmap("tab10").colors)
    plt.rcParams["font.size"] = 10
    plt.rcParams["font.weight"] = "bold"

    BS_names = [col for col in data.columns]
    BS_id_set = [int(name.split()[-1]) for name in BS_names]
    BS_id_set.sort()
    data_processed = [np.sort(data["Binding Site {}".format(bs_id)].dropna().tolist())
                      for bs_id in BS_id_set]
    colors = [next(color_set) for dummy in BS_id_set]
    fig, ax = plt.subplots(1, 1, figsize=(len(BS_id_set)*0.6, 2.8))
    plt.subplots_adjust(bottom=0.20, top=0.83)
    ax.set_title(title, fontsize=10, weight="bold")
    parts = ax.violinplot(data_processed, showmeans=False, showmedians=False, showextrema=False)
    for pc_idx, pc in enumerate(parts['bodies']):
        pc.set_facecolors(colors[pc_idx])
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    # deal with the situation in which the columns in data have different lengths.
    quartile1, medians, quartile3 = np.array([np.percentile(d, [25, 50, 75]) for d in data_processed]).T
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(data_processed, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    ax.scatter(inds, medians, marker='o', color='white', s=3, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(BS_id_set) + 1))
    ax.set_xticklabels(BS_id_set, fontsize=10, weight="bold")
    ax.set_xlim(0.25, len(BS_id_set) + 0.75)
    ax.set_xlabel('Binding Site', fontsize=10, weight="bold")
    ax.set_ylabel(ylabel, fontsize=10, weight="bold")
    plt.tight_layout()
    fig.savefig(fig_fn, dpi=200)
    if fig_close:
        plt.close()
    return


def plot_surface_area(surface_area, fig_fn, timeunit=None, fig_close=False):
    """Plot surface area as a function of time.

    Parameters
    ----------
    surface_area : pandas.DataFrame
    save_dir : str
    timeunit : str or None, optional, default=None

    See also
    ---------
    pylipid.func.calculate_surface_area
        The function that generates surface_area data.
    pylipid.plot.plot_surface_area_stats
        The function that plot surface_area in a matplotlib violin plot.

    """
    from itertools import cycle as _cycle

    color_set = _cycle(plt.get_cmap("tab10").colors)
    plt.rcParams["font.size"] = 10
    plt.rcParams["font.weight"] = "normal"

    if timeunit is None:
        timeunit = ""
    elif timeunit == "ns":
        timeunit = " (ns)"
    elif timeunit == "us":
        timeunit = r" ($\mu$s)"

    row_set = list(set([ind[:2] for ind in surface_area.index]))
    row_set.sort()
    col_set = [col for col in surface_area.columns if col != "Time"]
    colors = [next(color_set) for dummy in col_set]
    fig, axes = plt.subplots(len(row_set), len(col_set), figsize=(len(col_set)*2.4, len(row_set)*1.6),
                             sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.2, hspace=0.16)
    if len(col_set) == 1:
        axes = np.atleast_1d(axes)[:, np.newaxis]
    else:
        axes = np.atleast_2d(axes)
    for row_idx, row in enumerate(row_set):
        df = surface_area.loc[row]
        for col_idx, bs_name in enumerate(col_set):
            axes[row_idx, col_idx].plot(df["Time"], df[bs_name], color=colors[col_idx],
                                        label="traj {} prot {}".format(row[0], row[1]))
            if row_idx == len(row_set)-1:
                axes[row_idx, col_idx].set_xlabel("Time{}".format(timeunit), fontsize=10)
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(r"Area (nm$^2$)", fontsize=10)
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(bs_name, fontsize=10)
            axes[row_idx, col_idx].legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(fig_fn, dpi=200)
    if fig_close:
        plt.close()

    return


class AxisIndex:
    """Build axes for logo figure."""

    def __init__(self, residue_index, logos, interactions, length, gap):
        self.page_idx = 0
        self.length = length
        self.gap = gap
        self.residue_index = residue_index
        self.logos = logos
        self.interactions = interactions
        self.axis_start = (residue_index[0] // length) * length
        self.breaks = defaultdict(list)
        self.breaks[self.page_idx].append([])
        self.gray_areas = defaultdict(list)

    def fill_missing(self, start_value, end_value):
        for xloci in np.arange(start_value, end_value + 1):
            self.breaks[self.page_idx][-1].append((xloci, "A", 0))
        self.gray_areas[self.page_idx].append((len(self.breaks[self.page_idx]) - 1, start_value, end_value))

    def new_axis(self, pointer):
        self.breaks[self.page_idx].append([])
        self.axis_start = self.residue_index[pointer]
        self.breaks[self.page_idx][-1].append(
            (self.residue_index[pointer], self.logos[pointer], self.interactions[pointer]))

    def new_page(self, pointer):
        if len(self.breaks[self.page_idx][-1]) < self.length:
            self.fill_missing(self.axis_start + len(self.breaks[self.page_idx][-1]), self.axis_start + self.length - 1)
        self.page_idx += 1
        self.breaks[self.page_idx].append([])
        self.axis_start = (self.residue_index[pointer] // self.length) * self.length
        if self.axis_start != self.residue_index[pointer]:
            self.fill_missing(self.axis_start, self.residue_index[pointer] - 1)
        self.breaks[self.page_idx][-1].append(
            (self.residue_index[pointer], self.logos[pointer], self.interactions[pointer]))

    def new_gap(self, pointer):
        gray_start = self.residue_index[pointer - 1] + 1
        for xloci in np.arange(self.residue_index[pointer - 1] + 1, self.residue_index[pointer]):
            if xloci - self.axis_start < self.length:
                self.breaks[self.page_idx][-1].append((xloci, "A", 0))
            else:
                self.gray_areas[self.page_idx].append((len(self.breaks[self.page_idx]) - 1, gray_start, xloci - 1))
                self.breaks[self.page_idx].append([])
                self.breaks[self.page_idx][-1].append((xloci, "A", 0))
                self.axis_start = xloci
                gray_start = xloci
        self.gray_areas[self.page_idx].append(
            (len(self.breaks[self.page_idx]) - 1, gray_start, self.residue_index[pointer] - 1))
        self.breaks[self.page_idx][-1].append(
            (self.residue_index[pointer], self.logos[pointer], self.interactions[pointer]))

    def sort(self):
        end = False
        if self.axis_start != self.residue_index[0]:
            self.fill_missing(self.axis_start, self.residue_index[0] - 1)
        self.breaks[self.page_idx][-1].append((self.residue_index[0], self.logos[0], self.interactions[0]))
        pointer = 1
        while not end:
            if self.residue_index[pointer] - self.residue_index[pointer - 1] == 1 and self.residue_index[
                pointer] - self.axis_start < self.length:
                self.breaks[self.page_idx][-1].append(
                    (self.residue_index[pointer], self.logos[pointer], self.interactions[pointer]))
                pointer += 1
            elif self.residue_index[pointer] - self.residue_index[pointer - 1] == 1 and self.residue_index[
                pointer] - self.axis_start >= self.length:
                self.new_axis(pointer)
                pointer += 1
            elif self.residue_index[pointer] - self.residue_index[pointer - 1] < 0:
                self.new_page(pointer)
                pointer += 1
            elif 1 < self.residue_index[pointer] - self.residue_index[pointer - 1] <= self.gap:
                self.new_gap(pointer)
                pointer += 1
            elif self.residue_index[pointer] - self.residue_index[pointer - 1] > self.gap:
                self.new_page(pointer)
                pointer += 1
            if pointer == len(self.residue_index):
                end = True
        if len(self.breaks[self.page_idx][-1]) < self.length:
            self.fill_missing(self.axis_start + len(self.breaks[self.page_idx][-1]), self.axis_start + self.length - 1)
