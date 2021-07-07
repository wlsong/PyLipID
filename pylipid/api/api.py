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

from collections import defaultdict
from itertools import product
from functools import partial
import pickle
import os
import re
import warnings
import mdtraj as md
import numpy as np
np.seterr(all='ignore')
from scipy.sparse import coo_matrix
import pandas as pd
from tqdm import trange, tqdm
from p_tqdm import p_map
from ..func import cal_contact_residues
from ..func import Duration
from ..func import cal_lipidcount, cal_occupancy
from ..func import get_node_list
from ..func import collect_bound_poses
from ..func import analyze_pose_wrapper, calculate_koff_wrapper, calculate_surface_area_wrapper
from ..plot import plot_surface_area, plot_binding_site_data
from ..plot import plot_residue_data, plot_corrcoef, plot_residue_data_logo
from ..util import check_dir, write_PDB, write_pymol_script, sparse_corrcoef, get_traj_info


class LipidInteraction:
    def __init__(self, trajfile_list, cutoffs=[0.475, 0.7], lipid="CHOL", topfile_list=None, lipid_atoms=None,
                 nprot=1, resi_offset=0, save_dir=None, timeunit="us", stride=1, dt_traj=None):

        """The main class that handles calculation and controls workflow.

        ``LipidInteraction`` reads trajectory information via `mdtraj.load()`, so it supports most of the trajectory
        formats. ``LipidInteraction`` calculates lipid interactions with both protein residues and the calculated
        binding sites, and provides a couple of assisting functions to plot data and present data in various forms.

        The methods of ``LipidInteraction`` can be divided into three groups based on their roles: one for calculation
        of interaction with protein residues, one for binding site and the last that contains assisting functions for
        plotting and generating data. Each of the first two groups has a core function to collect/calculate the required
        data for the rest of the functions in that group, i.e. ``collect_residue_contacts`` that builds lipid index for
        residues as a function of time for residue analysis; and ``compute_binding_sites`` that calculates the binding
        sites using the interaction network of the residues. The rest of the methods in each group are independent of
        each other.

        ``LipidInteraction`` also has an attribute, named ``dataset``, which stores the calculation interaction data in
        a `pandas.DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_ object
        and updates automatically after calculation. It records interaction data for protein residues by
        rows, including, for each residue, the interaction residence times, averaged durations, occupancy and lipid
        count etc., and binding site IDs and the various interaction data of the belonging binding site.

        For the computing-demanding functions of, i.e. ``compute_residue_koff``, ``compute_site_koff``,
        ``analyze_bound_poses``, and ``compute_surface_area``, PyLipID uses the python multiprocessing library
        to speed up the calculation. Users can specify the number of CPUs these functions can use, otherwise all the
        CPUs in the system will be used by default.

        Parameters
        ----------
        trajfile_list : str or a list of str
            Trajectory filename(s). Read by mdtraj.load() to obtain trajectory information.

        cutoffs : list of two scalar or a scalar, default=[0.475, 0.7]
            Cutoff value(s) for defining contacts. When a list of two scalar are provided, the dual-cutoff scheme
            will be used. A contact in the dual-cutoff scheme starts when a lipid gets closer than the lower cutoff,
            and ends when this lipid moves farther than the upper cutoff. The duration between the two time points is
            the duration of this contact.

        lipid : str, default="CHOL"
            Lipid name in topology.

        topfile_list : str or a list of str, default=None
            Topology filename(s). Most trajectory formats do not contain topology information. Provide either
            the path to a RCSB PDB file, a trajectory, or a topology for each trajectory in `trajfile_list`
            for the topology information. See `mdtraj.load() <https://mdtraj.org>`_. for more information.

        lipid_atoms : list of str, default=None
            Lipid atom names. Only interactions of the provided atoms will be considered for the calculation of contacts.
            If None, all atoms of the lipid molecule will be used.

        nprot : int, default=1
            Number of protein copies in the system. If the system has N copies of the protein, 'nprot=N' will report
            averaged values from the N copies, but 'nprot=1' will report interaction values for each copy.

        resi_offset : int, default=0
            Shift residue index in the reported results from what is shown in the topology. Can be useful for
            MARTINI force field.

        save_dir : str, default=None
            The root directory to store the generated data. By default, a directory Interaction_{lipid} will be created
            in the current working directory, under which all the generated data are stored.

        timeunit : {"us", "ns"}, default="us"
            The time unit used for reporting results. "us" is micro-second and "ns" is nanosecond.

        stride : int, default=1
            Only read every stride-th frame. The same stride in mdtraj.load().

        dt_traj : float, default=None
            Timestep of trajectories. It is required when trajectories do not have timestep information. Not needed for
            trajectory formats of e.g. xtc, trr etc. If None, timestep information will take from trajectories.

        """
        self._trajfile_list = np.atleast_1d(trajfile_list)
        if len(np.atleast_1d(topfile_list)) == len(self._trajfile_list):
            self._topfile_list = np.atleast_1d(topfile_list)
        elif len(self._trajfile_list) > 1 and len(np.atleast_1d(topfile_list)) == 1:
            self._topfile_list = [topfile_list for dummy in self._trajfile_list]
        else:
            raise ValueError(
                "topfile_list should either have the same length as trajfile_list or have one valid file name.")

        if len(np.atleast_1d(cutoffs)) == 1:
            self._cutoffs = np.array([np.atleast_1d(cutoffs)[0] for dummy in range(2)])
        elif len(np.atleast_1d(cutoffs)) == 2:
            self._cutoffs = np.sort(np.array(cutoffs, dtype=float))
        else:
            raise ValueError("cutoffs should be either a scalar or a list of two scalars.")

        self._dt_traj = dt_traj
        self._lipid = lipid
        self._lipid_atoms = lipid_atoms
        self._nprot = int(nprot)
        self._timeunit = timeunit
        self._stride = int(stride)
        self._resi_offset = resi_offset
        self.dataset = pd.DataFrame()
        self._save_dir = check_dir(os.getcwd(), "Interaction_{}".format(self._lipid)) if save_dir is None \
            else check_dir(save_dir, "Interaction_{}".format(self._lipid))
        return

    #############################################
    #     attributes
    #############################################

    def dataset(self):
        """Summary of lipid interaction stored in a pandas.DataFrame() object."""
        return self.dataset

    @property
    def residue_list(self):
        """A list of Residue names."""
        return self._residue_list

    @property
    def node_list(self):
        """A list of binding site residue indices. """
        return self._node_list

    @property
    def lipid(self):
        """Lipid residue name."""
        return self._lipid

    @property
    def lipid_atoms(self):
        """Lipid atom names"""
        return self._lipid_atoms

    @property
    def cutoffs(self):
        """Cutoffs used for calculating contacts. """
        return self._cutoffs

    @property
    def nprot(self):
        """Number of protein copies in system. """
        return self._nprot

    @property
    def stride(self):
        """Stride"""
        return self._stride

    @property
    def trajfile_list(self):
        """Trajectory filenames """
        return self._trajfile_list

    @property
    def topfile_list(self):
        """Topology filenames"""
        return self._topfile_list

    @property
    def dt_traj(self):
        """Trajectory timestep"""
        return self._dt_traj

    @property
    def resi_offset(self):
        """Residue index offset"""
        return self._resi_offset

    @property
    def save_dir(self):
        """Root directory for the generated data."""
        return self._save_dir

    @property
    def timeunit(self):
        """Time unit used for reporting results. """
        return self._timeunit

    def koff(self, residue_id=None, residue_name=None):
        """Residue koff"""
        if residue_id is not None and residue_name is not None:
            assert self.dataset[self.dataset["Residue ID"] == residue_id]["Residue"] == residue_name, \
                "residue_id and residue_name are pointing to different residues!"
            return self._koff[residue_id]
        elif residue_id is not None:
            return self._koff[residue_id]
        elif residue_name is not None:
            return self._koff[self._residue_map[residue_name]]

    def res_time(self, residue_id=None, residue_name=None):
        """Residue residence time"""
        if residue_id is not None and residue_name is not None:
            assert self.dataset[self.dataset["Residue ID"] == residue_id]["Residue"] == residue_name, \
                "residue_id and residue_name are pointing to different residues!"
            return self._res_time[residue_id]
        elif residue_id is not None:
            return self._res_time[residue_id]
        elif residue_name is not None:
            return self._res_time[self._residue_map[residue_name]]

    def koff_bs(self, bs_id):
        """Binding site koff"""
        return self._koff_BS[bs_id]

    def res_time_bs(self, bs_id):
        """Binding site residence time"""
        return self._res_time_BS[bs_id]

    def residue(self, residue_id=None, residue_name=None, print_data=True):
        """Obtain the lipid interaction information for a residue

        Use either residue_id or residue_name to indicate the residue identity.
        Return the interaction information in a pandas.DataFrame object.

        Parameters
        ----------
        residue_id : int or list of int, default=None
            The residue ID that is used by PyLipID for identifying residues. The ID starts from 0, i.e. the ID
            of N-th residue is (N-1). If None, all residues are selected.
        residue_name : str or list of str, default=None
            The residue name as stored in PyLipID dataset. The residue name is in the format of resi+resn

        Returns
        -------
        df : pandas.DataFrame
            A pandas.DataFrame of interaction information of the residue.

        """
        if residue_id is not None and residue_name is not None:
            assert self.dataset[self.dataset["Residue ID"] == residue_id]["Residue"] == residue_name, \
                "residue_id and residue_name are pointing to different residues!"
            df = self.dataset[self.dataset["Residue ID"] == residue_id]
        elif residue_id is not None:
            df = self.dataset[self.dataset["Residue ID"] == residue_id]
        elif residue_name is not None:
            df = self.dataset[self.dataset["Residue"] == residue_name]
        if print_data:
            print(df)
        return df

    def binding_site(self, binding_site_id, print_data=True, sort_residue="Residence Time"):
        """Obtain the lipid interaction information for a binding site.

        Use binding site ID to access the information. Return the lipid interaction information of the
        binding site in a pandas.DataFrame object. If print_data is True, the binding site info will be
        formatted and print out.

        """
        df = self.dataset[self.dataset["Binding Site ID"] == binding_site_id].sort_values(by="Residence Time")
        if print_data:
            text = self._format_BS_print_info(binding_site_id, self._node_list[binding_site_id], sort_residue)
            print(text)
        return df

    ########################################
    #     interaction calculation
    ########################################
    def collect_residue_contacts(self):
        r"""Create contacting lipid index for residues.

        This function creates contacting lipid index for residues that are used for the rest of calculation in PyLipID.
        The design of contacting lipid index is to assist the calculation of contacts using a dual-cutoff scheme, which
        considers a lipid as being in contact when the lipid moves closer than the lower cutoff and as being dissociated
        when the lipid moves farther than the upper cutoff.

        The lipid indices created by this method are stored in the private class variables of
        ``_contact_residue_high`` and ``_contact_residue_low`` for each of the cutoffs. These indices are python
        dictionary objects with residue indices as their keys. For each residue, the lipid index stores the residue index
        of contacting lipid molecules from each trajectory frame in a list.

        The lipid index of the lower cutoff, i.e. ``_contact_residue_low`` is used to calculate lipid occupancy and lipid
        count.

        The Pearson correlation matrix of lipid interactions for protein residues is also calculated in this function and
        stored in the class variable of ``interaction_corrcoef``.

        The class attribute :meth:`~LipidInteraction.dataset` which stores the summary of lipid interaction as a
        pandas.DataFrame object, is initialized in this method.

        """
        self._protein_ref = None
        self._lipid_ref = None
        self._T_total = []
        self._timesteps = []
        self._protein_residue_id = []
        # initialise data for interaction matrix
        col = []
        row = []
        data = []
        ncol_start = 0
        # calculate interactions from trajectories
        for traj_idx in trange(len(self._trajfile_list), desc="COLLECT INTERACTIONS FROM TRAJECTORIES",
                               total=len(self._trajfile_list)):
            traj = md.load(self._trajfile_list[traj_idx], top=self._topfile_list[traj_idx], stride=self._stride)
            traj_info, self._protein_ref, self._lipid_ref = get_traj_info(traj, self._lipid,
                                                                          lipid_atoms=self._lipid_atoms,
                                                                          resi_offset=self._resi_offset,
                                                                          nprot=self._nprot,
                                                                          protein_ref=self._protein_ref,
                                                                          lipid_ref=self._lipid_ref)
            if self._dt_traj is None:
                timestep = traj.timestep / 1000000.0 if self._timeunit == "us" else traj.timestep / 1000.0
            else:
                timestep = float(self._dt_traj * self._stride)
            self._T_total.append((traj.n_frames - 1) * timestep)
            self._timesteps.append(timestep)
            if len(self._protein_residue_id) == 0:
                self._protein_residue_id = traj_info["protein_residue_id"]
                self._residue_list = traj_info["residue_list"]
                self._nresi_per_protein = len(self._residue_list)
                self._duration = dict()
                self._occupancy = dict()
                self._lipid_count = dict()
                self._contact_residues_high = {residue_id: [] for residue_id in self._protein_residue_id}
                self._contact_residues_low = {residue_id: [] for residue_id in self._protein_residue_id}
                self._koff = np.zeros(self._nresi_per_protein)
                self._koff_boot = np.zeros(self._nresi_per_protein)
                self._r_squared = np.zeros(self._nresi_per_protein)
                self._r_squared_boot = np.zeros(self._nresi_per_protein)
                self._res_time = np.zeros(self._nresi_per_protein)
                self._residue_map = {residue_name: residue_id
                                     for residue_id, residue_name in zip(self._protein_residue_id, self._residue_list)}
            else:
                assert len(self._protein_residue_id) == len(traj_info["protein_residue_id"]), \
                    "Trajectory {} contains {} residues whereas trajectory {} contains {} residues".format(
                        traj_idx, len(traj_info["protein_residue_id"]), traj_idx - 1, len(self._protein_residue_id))
            ncol_per_protein = len(traj_info["lipid_residue_atomid_list"]) * traj.n_frames
            for protein_idx in np.arange(self._nprot, dtype=int):

                for residue_id, residue_atom_indices in enumerate(
                        traj_info["protein_residue_atomid_list"][protein_idx]):
                    # calculate interaction per residue
                    dist_matrix = np.array([np.min(
                        md.compute_distances(traj, np.array(list(product(residue_atom_indices, lipid_atom_indices))),
                                             periodic=True, opt=True),
                        axis=1) for lipid_atom_indices in traj_info["lipid_residue_atomid_list"]])
                    contact_low, frame_id_set_low, lipid_id_set_low = cal_contact_residues(dist_matrix, self._cutoffs[0])
                    contact_high, _, _ = cal_contact_residues(dist_matrix, self._cutoffs[1])
                    self._contact_residues_high[residue_id].append(contact_high)
                    self._contact_residues_low[residue_id].append(contact_low)
                    # update coordinates for coo_matrix
                    col.append([ncol_start + ncol_per_protein * protein_idx + lipid_id * traj.n_frames +
                                frame_id for frame_id, lipid_id in zip(frame_id_set_low, lipid_id_set_low)])
                    row.append([residue_id for dummy in np.arange(len(frame_id_set_low), dtype=int)])
                    data.append(dist_matrix[lipid_id_set_low, frame_id_set_low])

            ncol_start += ncol_per_protein * self._nprot
        # calculate correlation coefficient matrix
        row = np.concatenate(row)
        col = np.concatenate(col)
        data = np.concatenate(data)
        contact_info = coo_matrix((data, (row, col)), shape=(self._nresi_per_protein, ncol_start))
        self.interaction_corrcoef = sparse_corrcoef(contact_info)
        self.dataset = pd.DataFrame({"Residue": [residue for residue in self._residue_list],
                                     "Residue ID": self._protein_residue_id})
        return

    def compute_residue_duration(self, residue_id=None):
        r"""Calculate lipid contact durations for residues

        PyLipID calculates lipid contacts using a dual-cutoff scheme. In this scheme, a continuous contact starts when
        a molecule moves closer than the lower distance cutoff and ends when the molecule moves out of the upper cutoff.
        The duration between these two time points is the duration of the contact.

        PyLipID implements this dual-cutoff tactic by creating a lipid index for the lower and upper
        cutoff respectively, which records the lipid molecules within that distance cutoff at each trajectory frame
        for residues. Such lipid indices are created by the method :meth:`~LipidInteraction.collect_residue_contacts`,
        and are stored in the private class variables of ``_contact_residue_high`` and ``_contact_residue_low`` for
        each of the cutoffs.

        For calculation of contact durations, a lipid molecule that appears in the lipid index of the lower cutoff is
        searched in the subsequent frames of the upper lipid index for that residue and the search then stops if this
        molecule disappears from the upper cutoff index. This lipid molecule is labeled as 'checked' in the searched
        frames in both lipid indices, and the duration of this contact is calculated from the number of frames in which
        this lipid molecule appears in the lipid indices. This calculation iterates until all lipid molecules in the
        lower lipid index are labeled as 'checked'.

        This function returns a list of contact durations or lists of contact durations if multiple residue IDs are
        provided.

        Parameters
        ----------
        residue_id : int or list of int, default=None
            The residue ID, or residue index, that is used by PyLipID for identifying residues. The ID starts from 0,
            i.e. the ID of N-th residue is (N-1). If None, all residues are selected.

        Returns
        -------
        durations : list
            A list of contact durations or lists of contact durations if multiple residue IDs are provided.

        See Also
        --------
        pylipid.api.LipidInteraction.collect_residue_contacts
            Create the lipid index.
        pylipid.api.LipidInteraction.compute_site_duration
            Calculate durations of contacts with binding sites.
        pylipid.func.Duration
            Calculate contact durations from lipid index.

        """
        self._check_calculation("Residue", self.collect_residue_contacts)
        if residue_id is None:
            selected_residue_id = self._protein_residue_id
        else:
            selected_residue_id = np.atleast_1d(residue_id)
        for residue_id in tqdm(selected_residue_id, desc="CALCULATE DURATION PER RESIDUE"):
            self._duration[residue_id] = [
                        Duration(self._contact_residues_low[residue_id][(traj_idx*self._nprot)+protein_idx],
                                 self._contact_residues_high[residue_id][(traj_idx*self._nprot)+protein_idx],
                                 self._timesteps[traj_idx]).cal_durations()
                        for traj_idx in np.arange(len(self.trajfile_list))
                        for protein_idx in np.arange(self._nprot, dtype=int)]       
        self.dataset["Duration"] = [np.mean(np.concatenate(self._duration[residue_id]))
                                    if len(self._duration[residue_id]) > 0 else 0
                                    for residue_id in self._protein_residue_id]
        self.dataset["Duration std"] = [np.std(np.concatenate(self._duration[residue_id]))
                                        if len(self._duration[residue_id]) > 0 else 0
                                        for residue_id in self._protein_residue_id]

        if len(selected_residue_id) == 1:
            return self._duration[residue_id]
        else:
            return [self._duration[residue_id] for residue_id in selected_residue_id]

    def compute_residue_occupancy(self, residue_id=None):
        """Calculate the percentage of frames in which the specified residue formed lipid contacts for residues.

        The lipid occupancy is calculated using the lower cutoff, and calculated as the percentage of frames in which
        the specified lipid species formed contact with residues within the lower distance cutoff.

        The returned occupancy list contains data from all protein copies and all trajectories.

        Parameters
        ----------
        residue_id : int or list of int, default=None
            The residue ID, or residue index, that is used by PyLipID for identifying residues. The ID starts from 0,
            i.e. the ID of N-th residue is (N-1). If None, all residues are selected.

        Returns
        -------
        occupancies : list
            A list of lipid occupancies, of length of n_trajs x n_proteins, or lists of lipid occupancies if multiple
            residue IDs are provided.

        See Also
        --------
        pylipid.api.LipidInteraction.collect_residue_contacts
            Create the lipid index.
        pylipid.api.LipidInteraction.compute_site_occupancy
            Calculate binding site occupancy
        pylipid.func.cal_occupancy
            Calculate the percentage of frames in which a contact is formed.

        """
        self._check_calculation("Duration", self.compute_residue_duration)
        if residue_id is None:
            selected_residue_id = self._protein_residue_id
        else:
            selected_residue_id = np.atleast_1d(residue_id)
        for residue_id in tqdm(selected_residue_id, desc="CALCULATE OCCUPANCY"):
            self._occupancy[residue_id] = [cal_occupancy(self._contact_residues_low[residue_id][(traj_idx*self._nprot)+protein_idx])
                                          for traj_idx in np.arange(len(self.trajfile_list))
                                          for protein_idx in np.arange(self._nprot, dtype=int)]
        self.dataset["Occupancy"] = [np.mean(self._occupancy[residue_id])
                                     if len(self._occupancy[residue_id]) > 0 else 0
                                     for residue_id in self._protein_residue_id]
        self.dataset["Occupancy std"] = [np.std(self._occupancy[residue_id])
                                         if len(self._occupancy[residue_id]) > 0 else 0
                                         for residue_id in self._protein_residue_id]

        if len(selected_residue_id) == 1:
            return self._occupancy[residue_id]
        else:
            return [self._occupancy[residue_id] for residue_id in selected_residue_id]

    def compute_residue_lipidcount(self, residue_id=None):
        """Calculate the average number of contacting lipids for residues.

        This method calculates the number of specified lipid within the lower distance cutoff to a residue. The
        reported value is averaged from the trajectory frames in which interaction between the specified lipid and the
        residue is formed. Thus the returned values report the average number of surrounding lipid molecules when
        the lipids are bound.

        The returned lipid count list contains data from each of the protein copies and each of the trajectories.

        Parameters
        ----------
        residue_id : int or list of int, default=None
            The residue ID, or residue index, that is used by PyLipID for identifying residues. The ID starts from 0,
            i.e. the ID of N-th residue is (N-1). If None, all residues are selected.

        Returns
        -------
        lipidcounts : list
            A list of lipid counts, of length of n_trajs x n_proteins, or lists of lipid counts if multiple
            residue IDs are provided.

        See Also
        --------
        pylipid.api.LipidInteraction.collect_residue_contacts
            Create the lipid index.
        pylipid.api.LipidInteraction.compute_site_lipidcount
            Calculate binding site lipid count.
        pylipid.func.cal_lipidcount
            Calculate the average number of contacting molecules.

        """
        self._check_calculation("Residue", self.collect_residue_contacts)
        if residue_id is None:
            selected_residue_id = self._protein_residue_id
        else:
            selected_residue_id = np.atleast_1d(residue_id)
        for residue_id in tqdm(selected_residue_id, desc="CALCULATE RESIDUE LIPIDCOUNT"):
            self._lipid_count[residue_id] = [cal_lipidcount(self._contact_residues_low[residue_id][(traj_idx*self._nprot)+protein_idx])
                                           for traj_idx in np.arange(len(self.trajfile_list))
                                           for protein_idx in np.arange(self._nprot, dtype=int)]
        self.dataset["Lipid Count"] = [np.mean(self._lipid_count[residue_id])
                                       if len(self._lipid_count[residue_id]) > 0 else 0
                                       for residue_id in self._protein_residue_id]
        self.dataset["Lipid Count std"] = [np.std(self._lipid_count[residue_id])
                                           if len(self._lipid_count[residue_id]) > 0 else 0
                                           for residue_id in self._protein_residue_id]
        if len(selected_residue_id) == 1:
            self._lipid_count[residue_id]
        else:
            return [self._lipid_count[residue_id] for residue_id in selected_residue_id]

    def compute_residue_koff(self, residue_id=None, nbootstrap=10, initial_guess=[1., 1., 1., 1.],
                             save_dir=None, plot_data=True, fig_close=True, fig_format="pdf", num_cpus=None):
        r"""Calculate interaction koff and residence time for residues.

        The koff is calculated from a survival time correlation function which describes the relaxation of the bound
        lipids [1]_. Often the interactions between lipid and protein surface are be divided into prolonged interactions and
        quick diffusive contacts. Thus PyLipID fits the normalised survival function to a bi-exponential curve which
        describes the long and short decay periods.

        The survival time correlation function σ(t) is calculated as follow

        .. math::

            \sigma(t) = \frac{1}{N_{j}} \frac{1}{T-t} \sum_{j=1}^{N_{j}} \sum_{v=0}^{T-t}\tilde{n}_{j}(v, v+t)

        where T is the length of the simulation trajectory, :math:`N_{j}` is the total number of lipid contacts and
        :math:`\sum_{v=0}^{T-t} \tilde{n}_{j}(v, v+t)` is a binary function that takes the value 1 if the contact of
        lipid j lasts from time ν to time v+t and 0 otherwise. The values of :math:`\sigma(t)` are calculated for every
        value of t from 0 to T ns, for each time step of the trajectories, and normalized by dividing by :math:`\sigma(t)`,
        so that the survival time-correlation function has value 1 at t = 0.

        The normalized survival function is then fitted to a biexponential to model the long and short decays of
        lipid relaxation:

        .. math::
            \sigma(t) \sim A e^{-k_{1} t}+B e^{-k_{2} t}\left(k_{1} \leq k_{2}\right)

        PyLipID takes :math:`k_{1}` as the the dissociation :math:`k_{off}`, and calculates the residence time as
        :math:`\tau=1 / k_{off}`. PyLipID raises a warning for the impact on the accuracy of :math:`k_{off}`
        calculation if trajectories are of different lengths when multiple trajectories are provided. PyLipID measures
        the :math:`r^{2}` of the biexponential fitting to the survival function to show the quality of the
        :math:`k_{off}` estimation. In addition, PyLipID bootstraps the contact durations and measures the
        :math:`k_{off}` of the bootstrapped data, to report how well lipid contacts are sampled from simulations. The
        lipid contact sampling, the curve-fitting and the bootstrap results can be conveniently checked via the
        :math:`k_{off}` plot.

        The calculation of koff for residues can be time-consuming, thus PyLipID uses python multiprocessing to
        parallize the calculation. The number of CPUs used for multiprocessing can be specificed, otherwise all the
        available CPUs will be used by default.

        Parameters
        ----------
        residue_id : int or list of int, default=None
            The residue ID, or residue index, that is used by PyLipID for identifying residues. The ID starts from 0,
            i.e. the ID of N-th residue is (N-1). If None, all residues are selected.

        nbootstrap : int, default=10
            Number of bootstrap on the interaction durations. For each bootstrap, samples of the size of the original
            dataset are drawn from the collected durations with replacement. :math:`k_{koff}` and :math:`r^{2}` are
            calculated for each bootstrap.

        initial_guess : array_like, default=None
            The initial guess for the curve-fitting of the biexponential curve. Used by scipy.optimize.curve_fit.

        save_dir : str, default=None
            The the directory for saving the koff figures of residues if plot_data is True. By default, the koff figures
            are saved in the directory of Reisidue_koffs_{lipid} under the root directory defined when ``LipidInteraction``
            was initiated.

        plot_data : bool, default=True
            If True, plot the koff figures fir residues.

        fig_close : bool, default=True
            Use matplotlib.pyplot.close() to close the koff figures. Can save memory if many figures are open and plotted.

        fig_format : str, default="pdf"
            The format of koff figures. Support formats that are supported by matplotlib.pyplot.savefig().

        num_cpus : int or None, default=None
            Number of CPUs used for multiprocessing. If None, all the available CPUs will be used.

        Returns
        ---------
        koff : scalar or list of scalar
            The calculated koffs for selected residues.

        restime : scalar or list of scalar
            The calculated residence times for selected residues.

        See Also
        ---------
        pylipid.api.LipidInteraction.collect_residue_contacts
            Create the lipid index.
        pylipid.api.LipidInteraction.compute_site_koff
            Calculate binding site koffs and residence times.
        pylipid.func.cal_koff
            Calculate residence time and koff.
        pylipid.func.cal_survival_func
            Compute the normalised survival function.

        References
        -----------
        .. [1] García, Angel E.Stiller, Lewis. Computation of the mean residence time of water in the hydration shells
               of biomolecules. 1993. Journal of Computational Chemistry.

        """
        self._check_calculation("Residue", self.compute_residue_koff)

        if plot_data:
            koff_dir = check_dir(save_dir, "Reisidue_koffs_{}".format(self._lipid)) if save_dir is not None \
                else check_dir(self._save_dir, "Residue_koffs_{}".format(self._lipid))
        if len(set(self._residue_list)) != len(self._residue_list):
            residue_name_set = ["{}_ResidueID{}".format(residue, residue_id) for residue, residue_id in
                                zip(self._residue_list, self._protein_residue_id)]
        else:
            residue_name_set = self._residue_list
        if residue_id is not None:
            selected_residue_id = np.atleast_1d(residue_id)
        else:
            selected_residue_id = self._protein_residue_id
        residues_missing_durations = [residue_id for residue_id in selected_residue_id
                                      if len(self._duration[residue_id]) == 0]
        if len(residues_missing_durations) > 0:
            self.compute_residue_duration(residue_id=residues_missing_durations)
        t_total = np.max(self._T_total)
        same_length = np.all(np.array(self._T_total) == t_total)
        if not same_length:
            warnings.warn(
                "Trajectories have different lengths. This will impair the accuracy of koff calculation!")
        timestep = np.min(self._timesteps)
        same_timestep = np.all(np.array(self._timesteps) == timestep)
        if not same_timestep:
            warnings.warn(
                "Trajectories have different timesteps. This will impair the accuracy of koff calculation!")
        if plot_data:
            fn_set = [os.path.join(koff_dir, "{}.{}".format(residue_name_set[residue_id], fig_format))
                      for residue_id in selected_residue_id]
        else:
            fn_set = [False for dummy in selected_residue_id]
                
        returned_values = p_map(partial(calculate_koff_wrapper, t_total=t_total, timestep=timestep, nbootstrap=nbootstrap,
                                        initial_guess=initial_guess, plot_data=plot_data, timeunit=self._timeunit,
                                        fig_close=fig_close),
                                [np.concatenate(self._duration[residue_id]) for residue_id in selected_residue_id],
                                [residue_name_set[residue_id] for residue_id in selected_residue_id],
                                fn_set, num_cpus=num_cpus, desc="CALCULATE KOFF FOR RESIDUES")

        for residue_id, returned_value in zip(selected_residue_id, returned_values):
            self._koff[residue_id] = returned_value[0]
            self._res_time[residue_id] = returned_value[1]
            self._r_squared[residue_id] = returned_value[2]
            self._koff_boot[residue_id] = returned_value[3]
            self._r_squared_boot[residue_id] = returned_value[4]
        # update dataset
        self.dataset["Koff"] = self._koff
        self.dataset["Residence Time"] = self._res_time
        self.dataset["R Squared"] = self._r_squared
        self.dataset["Koff Bootstrap avg"] = self._koff_boot
        self.dataset["R Squared Bootstrap avg"] = self._r_squared_boot

        if len(selected_residue_id) == 1:
            return self._koff[selected_residue_id[0]], self._res_time[selected_residue_id[0]]
        else:
            return [self._koff[residue_id] for residue_id in selected_residue_id], \
                   [self._res_time[residue_id] for residue_id in selected_residue_id]

    def compute_binding_nodes(self, threshold=4, print_data=True):
        r"""Calculate binding sites.

        Binding sites are defined based on a community analysis of protein residue-interaction networks that are created
        from the lipid interaction correlation matrix. Given the definition of a lipid binding site, namely a
        cluster of residues that bind to the same lipid molecule at the same time, PyLipID creates a distance vector
        for each residue that records the distances to all lipid molecules as a function of time, and calculate the
        Pearson correlation matrix of protein residues for binding the same lipid molecules. This correlation matrix is
        calculated by :meth:`~LipidInteraction.collect_residue_contacts()` and stored in the class variable
        ``interaction_corrcoef``.

        The protein residue interaction network is constructed based on the Pearson correlation matrix.
        In this network, the nodes are the protein residues and the weights are the Pearson correlation
        coefficients of pairs of residues. The interaction network is then decomposed into sub-units or communities,
        which are groups of nodes that are more densely connected internally than with the rest of the network.

        For the calculation of communities, the Louvain algorithm [1]_ is used to find high modularity network partitions.
        Modularity, which measures the quality of network partiions, is defined as [2]_

        .. math::
            Q=\frac{1}{2 m} \sum_{i, j}\left[A_{i j}-\frac{k_{i} k_{j}}{2 m}\right] \delta\left(c_{i}, c_{j}\right)

        where :math:`A_{i j}` is the weight of the edge between node i and node j; :math:`k_{i}` is the sum of weights
        of the nodes attached to the node i, i.e. the degree of the node; :math:`c_{i}` is the community to which node i
        assigned; :math:`\delta\left(c_{i}, c_{j}\right)` is 1 if i=j and 0 otherwise; and
        :math:`m=\frac{1}{2} \sum_{i j} A_{i j}` is the number of edges. In the modularity optimization, the Louvain
        algorithm orders the nodes in the network, and then, one by one, removes and inserts each node in a different
        community c_i until no significant increase in modularity. After modularity optimization, all the nodes that
        belong to the same community are merged into a single node, of which the edge weights are the sum of the weights
        of the comprising nodes. This optimization-aggregation loop is iterated until all nodes are collapsed into one.

        By default, this method returns binding sites of at least 4 residues. This filtering step is particularly helpful
        for analysis on smaller amount of trajectory frames, in which false correlation is more likely to happen among
        2 or 3 residues.

        Parameters
        ----------
        threshold : int, default=4
            The minimum size of binding sites. Only binding sites with more residues than the threshold will be returned.

        print : bool, default=True
            If True, print a summary of binding site information.

        Returns
        -------
        node_list: list
            Binding site node list, i.e. a list of binding sites which contains sets of binding site residue indices
        modularity : float or None
            The modularity of network partition. It measure the quality of network partition. The value is between 1 and
            -1. The bigger the modularity, the better the partition.

        See Also
        --------
        pylipid.func.get_node_list
            Calculates community structures in interaction network.

        References
        ----------
        .. [1] Blondel, V. D.; Guillaume, J.-L.; Lambiotte, R.; Lefebvre, E., Fast unfolding of communities in large
               networks. Journal of Statistical Mechanics: Theory and Experiment 2008, 2008 (10), P10008

        .. [2] Newman, M. E. J., Analysis of weighted networks. Physical Review E 2004, 70 (5), 056131.

        """
        self._check_calculation("Residue", self.compute_residue_koff)
        corrcoef_raw = np.nan_to_num(self.interaction_corrcoef)
        corrcoef = np.copy(corrcoef_raw)
        node_list, modularity = get_node_list(corrcoef, threshold=threshold)
        self._node_list = node_list
        self._network_modularity = modularity
        if len(self._node_list) == 0:
            print("*"*30)
            print(" No binding site detected!!")
            print("*"*30)
        else:
            residue_BS_identifiers = np.ones(self._nresi_per_protein, dtype=int) * -1
            for bs_id, nodes in enumerate(self._node_list):
                residue_BS_identifiers[nodes] = int(bs_id)
            # update dataset
            self.dataset["Binding Site ID"] = residue_BS_identifiers
            # initialise variable for binding site interactions
            self._duration_BS = dict()
            self._occupancy_BS = dict()
            self._lipid_count_BS = dict()
            self._koff_BS = np.zeros(len(self._node_list))
            self._koff_BS_boot = np.zeros(len(self._node_list))
            self._res_time_BS = np.zeros(len(self._node_list))
            self._r_squared_BS = np.zeros(len(self._node_list))
            self._r_squared_BS_boot = np.zeros(len(self._node_list))
            if print_data:
                print(f"Network modularity: {modularity:.3f}")
                for bs_id, nodes in enumerate(self._node_list):
                    print("#" * 25)
                    print(f"Binding Site ID: {bs_id}")
                    print("{:>10s} -- {:<12s}".format("Residue", "Residue ID"))
                    for node in nodes:
                        print("{:>10s} -- {:<12d}".format(self._residue_list[node], self._protein_residue_id[node]))
                    print("#" * 25)
        return node_list, modularity

    def compute_site_duration(self, binding_site_id=None):
        """Calculate interaction durations for binding sites.

        PyLipID calculates lipid contacts using a dual-cutoff scheme. In this scheme, a continuous contact starts when
        a molecule moves closer than the lower distance cutoff and ends when the molecule moves out of the upper cutoff.
        The duration between these two time points is the duration of the contact.

        PyLipID implements this dual-cutoff tactic by creating a lipid index for the lower and upper
        cutoff respectively, which records the lipid molecules within that distance cutoff at each trajectory frame
        for residues. Such lipid indices are created by the method :meth:`~LipidInteraction.collect_residue_contacts`,
        and are stored in the private class variables of ``_contact_residue_high`` and ``_contact_residue_low`` for
        each of the cutoffs.

        For calculating contacts for binding sites, the interacting lipid molecules with binding site residues are
        merged with duplicates removed to form the lipid indices for the upper cutoff and lower cutoff respectively.
        Similar to the calculation of residues, a contact duration of a binding sites are calculated as the duration
        between the time point of a lipid molecule appearing in the lipid index of the lower cutoff and of this molecule
        disappeared from the upper cutoff index.

        This function returns a list of contact durations or lists of contact durations if multiple binding site IDs are
        provided.

        Parameters
        ----------
        binding_site_id : int or list of int, default=None
            The binding site ID used in PyLipID. This ID is the index in the binding site node list that is
            calculated by the method ``compute_binding_nodes``. The ID of the N-th binding site is (N-1). If None,
            the contact duration of all binding sites are calculated.

        Returns
        -------
        durations_BS : list
            A list of contact durations or lists of contact durations if multiple binding site IDs are provided.

        See Also
        ---------
        pylipid.api.LipidInteraction.collect_residue_contacts
            Create the lipid index.
        pylipid.api.LipidInteraction.compute_residue_duration
            Calculate residue contact durations.
        pylipid.func.Duration
            Calculate contact durations from lipid index.

        """
        self._check_calculation("Binding Site ID", self.compute_binding_nodes, print_data=False)
        selected_bs_id = np.atleast_1d(binding_site_id) if binding_site_id is not None \
            else np.arange(len(self._node_list), dtype=int)
        for bs_id in tqdm(selected_bs_id, desc="CALCULATE DURATION PER BINDING SITE"):
            nodes = self._node_list[bs_id]
            durations_BS = []
            for traj_idx in np.arange(len(self._trajfile_list), dtype=int):
                for protein_idx in np.arange(self._nprot, dtype=int):
                    list_to_take = traj_idx * self._nprot + protein_idx
                    n_frames = len(self._contact_residues_low[nodes[0]][list_to_take])
                    contact_BS_low = [np.unique(np.concatenate(
                        [self._contact_residues_low[node][list_to_take][frame_idx] for node in nodes]))
                        for frame_idx in np.arange(n_frames)]
                    contact_BS_high = [np.unique(np.concatenate(
                        [self._contact_residues_high[node][list_to_take][frame_idx] for node in nodes]))
                        for frame_idx in np.arange(n_frames)]
                    durations_BS.append(
                        Duration(contact_BS_low, contact_BS_high, self._timesteps[traj_idx]).cal_durations())
            self._duration_BS[bs_id] = durations_BS
        # update dataset
        durations_BS_per_residue = np.zeros(self._nresi_per_protein)
        for bs_id, nodes in enumerate(self._node_list):
            durations_BS_per_residue[nodes] = np.mean(np.concatenate(self._duration_BS[bs_id])) \
                if len(self._duration_BS[bs_id]) > 0 else 0
        self.dataset["Binding Site Duration"] = durations_BS_per_residue

        if len(selected_bs_id) == 1:
            return self._duration_BS[bs_id]
        else:
            return [self._duration_BS[bs_id] for bs_id in selected_bs_id]

    def compute_site_occupancy(self, binding_site_id=None):
        """Calculate the percentage of frames in which the specified lipid contacts are formed for binding sites.

        Similar to calculation on residues, the lipid occupancy is calculated using the lower cutoff, and calculated as
        the percentage of frames in which the specified lipid species formed contact with the binding site within the
        lower distance cutoff. The lipid index for a binding site is generated from merging, with duplicates removed,

        The returned list of occupancies contains data from all protein copies and all trajectories.

        Parameters
        ----------
        binding_site_id : int or list of int, default=None
            The binding site ID used in PyLipID. This ID is the index in the binding site node list that is
            calculated by the method ``compute_binding_nodes``. The ID of the N-th binding site is (N-1). If None,
            the contact duration of all binding sites are calculated.

        Returns
        -------
        occupancy_BS : list
            A list of lipid occupancies or lists of lipid occupancies if multiple binding site IDs are provided.

        See Also
        ---------
        pylipid.api.LipidInteraction.collect_residue_contacts
            Create the lipid index.
        pylipid.api.LipidInteraction.compute_residue_occupancy
            Calculate lipid occupancy for residues.
        pylipid.func.cal_occupancy
            Calculate the percentage of frames in which a contact is formed.

        """
        self._check_calculation("Binding Site ID", self.compute_binding_nodes, print_data=False)
        selected_bs_id = np.atleast_1d(binding_site_id) if binding_site_id is not None \
            else np.arange(len(self._node_list), dtype=int)
        for bs_id in tqdm(selected_bs_id, desc="CALCULATE OCCUPANCY PER BINDING SITE"):
            nodes = self._node_list[bs_id]
            occupancy_BS = []
            for traj_idx in np.arange(len(self._trajfile_list), dtype=int):
                for protein_idx in np.arange(self._nprot, dtype=int):
                    list_to_take = traj_idx * self._nprot + protein_idx
                    n_frames = len(self._contact_residues_low[nodes[0]][list_to_take])
                    contact_BS_low = [np.unique(np.concatenate(
                        [self._contact_residues_low[node][list_to_take][frame_idx] for node in nodes]))
                        for frame_idx in np.arange(n_frames)]
                    occupancy_BS.append(cal_occupancy(contact_BS_low))
            self._occupancy_BS[bs_id] = occupancy_BS
        # update dataset
        occupancy_BS_per_residue = np.zeros(self._nresi_per_protein)
        for bs_id, nodes in enumerate(self._node_list):
            occupancy_BS_per_residue[nodes] = np.mean(self._occupancy_BS[bs_id]) \
                if len(self._occupancy_BS[bs_id]) > 0 else 0
        self.dataset["Binding Site Occupancy"] = occupancy_BS_per_residue

        if len(selected_bs_id) == 1:
            return self._occupancy_BS[bs_id]
        else:
            return [self._occupancy_BS[bs_id] for bs_id in selected_bs_id]

    def compute_site_lipidcount(self, binding_site_id=None):
        """Calculate the average number of contacting lipids for binding site.

        This method calculates the number of specified lipid within the lower distance cutoff to a binding site. The
        reported value is averaged from the trajectory frames in which interaction between the specified lipid and the
        binding site is formed. Thus the returned values report the average number of surrounding lipid molecules when
        the lipids are bound.

        The returned lipid count list contains data from each of the protein copies and each of the trajectories.

        Parameters
        ----------
        binding_site_id : int or list of int, default=None
            The binding site ID used in PyLipID. This ID is the index in the binding site node list that is
            calculated by the method ``compute_binding_nodes``. The ID of the N-th binding site is (N-1). If None,
            the contact duration of all binding sites are calculated.

        Returns
        -------
        lipidcount_BS : list
            A list of lipid counts or lists of lipid counts if multiple binding site IDs are provided.

        See Also
        ---------
        pylipid.api.LipidInteraction.collect_residue_contacts
            Create the lipid index.
        pylipid.api.LipidInteraction.compute_residue_lipidcount
            Calculate lipid count for residues.
        pylipid.func.cal_lipidcount
            Calculate the average number of contacting molecules.

        """
        self._check_calculation("Binding Site ID", self.compute_binding_nodes, print_data=False)
        selected_bs_id = np.atleast_1d(binding_site_id) if binding_site_id is not None \
            else np.arange(len(self._node_list), dtype=int)
        for bs_id in tqdm(selected_bs_id, desc="CALCULATE LIPIDCOUNT PER BINDING SITE"):
            nodes = self._node_list[bs_id]
            lipidcount_BS = []
            for traj_idx in np.arange(len(self._trajfile_list), dtype=int):
                for protein_idx in np.arange(self._nprot, dtype=int):
                    list_to_take = traj_idx * self._nprot + protein_idx
                    n_frames = len(self._contact_residues_low[nodes[0]][list_to_take])
                    contact_BS_low = [np.unique(np.concatenate(
                        [self._contact_residues_low[node][list_to_take][frame_idx] for node in nodes]))
                        for frame_idx in np.arange(n_frames)]
                    lipidcount_BS.append(cal_lipidcount(contact_BS_low))
            self._lipid_count_BS[bs_id] = lipidcount_BS
        # update dataset
        lipidcount_BS_per_residue = np.zeros(self._nresi_per_protein)
        for bs_id, nodes in enumerate(self._node_list):
            lipidcount_BS_per_residue[nodes] = np.mean(self._lipid_count_BS[bs_id]) \
                if len(self._lipid_count_BS[bs_id]) > 0 else 0
        self.dataset["Binding Site Lipid Count"] = lipidcount_BS_per_residue

        if len(selected_bs_id) == 1:
            return self._lipid_count_BS[bs_id]
        else:
            return [self._lipid_count_BS[bs_id] for bs_id in selected_bs_id]

    def compute_site_koff(self, binding_site_id=None, nbootstrap=10, initial_guess=[1., 1., 1., 1.],
                          save_dir=None, plot_data=True, fig_close=True, fig_format="pdf", num_cpus=None):
        r"""Calculate interactions koff and residence time for binding sites.

        The koff is calculated from a survival time correlation function which describes the relaxation of the bound
        lipids [1]_. Often the interactions between lipid and protein surface are be divided into prolonged interactions and
        quick diffusive contacts. Thus PyLipID fits the normalised survival function to a bi-exponential curve which
        describes the long and short decay periods.

        The survival time correlation function σ(t) is calculated as follow

        .. math::

            \sigma(t) = \frac{1}{N_{j}} \frac{1}{T-t} \sum_{j=1}^{N_{j}} \sum_{v=0}^{T-t}\tilde{n}_{j}(v, v+t)

        where T is the length of the simulation trajectory, :math:`N_{j}` is the total number of lipid contacts and
        :math:`\sum_{v=0}^{T-t} \tilde{n}_{j}(v, v+t)` is a binary function that takes the value 1 if the contact of
        lipid j lasts from time ν to time v+t and 0 otherwise. The values of :math:`\sigma(t)` are calculated for every
        value of t from 0 to T ns, for each time step of the trajectories, and normalized by dividing by :math:`\sigma(t)`,
        so that the survival time-correlation function has value 1 at t = 0.

        The normalized survival function is then fitted to a biexponential to model the long and short decays of
        lipid relaxation:

        .. math::
            \sigma(t) \sim A e^{-k_{1} t}+B e^{-k_{2} t}\left(k_{1} \leq k_{2}\right)

        PyLipID takes :math:`k_{1}` as the the dissociation :math:`k_{off}`, and calculates the residence time as
        :math:`\tau=1 / k_{o f f}`. PyLipID raises a warning for the impact on the accuracy of :math:`k_{off}`
        calculation if trajectories are of different lengths when multiple trajectories are provided. PyLipID measures
        the :math:`r^{2}` of the biexponential fitting to the survival function to show the quality of the
        :math:`k_{off}` estimation. In addition, PyLipID bootstraps the contact durations and measures the
        :math:`k_{off}` of the bootstrapped data, to report how well lipid contacts are sampled from simulations. The
        lipid contact sampling, the curve-fitting and the bootstrap results can be conveniently checked via the
        :math:`k_{off}` plot.

        The durations of lipid contact with binding sites are calculated using
        :meth:`~LipidInteraction.compute_site_duration`. See its page for the definition of lipid contact
        with binding site.

        The calculation of koff for binding sites can be time-consuming, thus PyLipID uses python multiprocessing to
        parallize the calculation. The number of CPUs used for multiprocessing can be specificed, otherwise all the
        available CPUs will be used by default.

        Parameters
        ----------
        binding_site_id : int or list of int, default=None
            The binding site ID used in PyLipID. This ID is the index in the binding site node list that is
            calculated by the method ``compute_binding_nodes``. The ID of the N-th binding site is (N-1). If None,
            the contact duration of all binding sites are calculated.

        nbootstrap : int, default=10
            Number of bootstrap on the interaction durations. For each bootstrap, samples of the size of the original
            dataset are drawn from the collected durations with replacement. :math:`k_{koff}` and :math:`r^{2}` are
            calculated for each bootstrap.

        initial_guess : array_like, default=None
            The initial guess for the curve-fitting of the biexponential curve. Used by scipy.optimize.curve_fit.

        save_dir : str, default=None
            The the directory for saving the koff figures of residues if plot_data is True. By default, the koff figures
            are saved in the directory of Binding_Sites_koffs_{lipid} under the root directory defined when ``LipidInteraction``
            was initiated.

        plot_data : bool, default=True
            If True, plot the koff figures fir residues.

        fig_close : bool, default=True
            Use matplotlib.pyplot.close() to close the koff figures. Can save memory if many figures are open and plotted.

        fig_format : str, default="pdf"
            The format of koff figures. Support formats that are supported by matplotlib.pyplot.savefig().

        num_cpus : int or None, default=None
            Number of CPUs used for multiprocessing. If None, all the available CPUs will be used.

        Returns
        ---------
        koff : scalar or list of scalar
            The calculated koffs for selected binding sites.

        restime : scalar or list of scalar
            The calculated residence times for selected binding sites.

        See Also
        ---------
        pylipid.api.LipidInteraction.collect_residue_contacts
            Create the lipid index.
        pylipid.api.LipidInteraction.compute_residue_koff
            Calculate koffs and residence times for residues.
        pylipid.func.cal_koff
            Calculate residence time and koff.
        pylipid.func.cal_survival_func
            Compute the normalised survival function.

        References
        -----------
        .. [1] García, Angel E.Stiller, Lewis. Computation of the mean residence time of water in the hydration shells
               of biomolecules. 1993. Journal of Computational Chemistry.

        """
        self._check_calculation("Binding Site Duration", self.compute_site_duration)
        if plot_data:
            BS_dir = check_dir(save_dir, "Binding_Sites_koffs_{}".format(self._lipid)) if save_dir is not None \
                else check_dir(self._save_dir, "Binding_Sites_koffs_{}".format(self._lipid))

        selected_bs_id = np.atleast_1d(binding_site_id) if binding_site_id is not None \
            else np.arange(len(self._node_list), dtype=int)
        binding_sites_missing_durations = [bs_id for bs_id in selected_bs_id
                                           if len(self._duration_BS[bs_id]) == 0]
        if len(binding_sites_missing_durations) > 0:
            self.compute_site_duration(binding_site_id=binding_sites_missing_durations)
        t_total = np.max(self._T_total)
        same_length = np.all(np.array(self._T_total) == t_total)
        if not same_length:
            warnings.warn(
                "Trajectories have different lengths. This will impair the accuracy of koff calculation!")
        timestep = np.min(self._timesteps)
        same_timestep = np.all(np.array(self._timesteps) == timestep)
        if not same_timestep:
            warnings.warn(
                "Trajectories have different timesteps. This will impair the accuracy of koff calculation!")
        if plot_data:
            fn_set = [os.path.join(BS_dir, f"BS_id{bs_id}.{fig_format}") for bs_id in selected_bs_id]
        else:
            fn_set = [False for dummy in selected_bs_id]
        returned_values = p_map(partial(calculate_koff_wrapper, t_total=t_total, timestep=timestep, nbootstrap=nbootstrap,
                                        initial_guess=initial_guess, plot_data=plot_data, timeunit=self._timeunit,
                                        fig_close=fig_close),
                                [np.concatenate(self._duration_BS[bs_id]) for bs_id in selected_bs_id],
                                [f"Binding Site {bs_id}" for bs_id in selected_bs_id],
                                fn_set, num_cpus=num_cpus, desc="CALCULATE KOFF FOR BINDING SITES")
        for bs_id, returned_value in zip(selected_bs_id, returned_values):
            self._koff_BS[bs_id] = returned_value[0]
            self._res_time_BS[bs_id] = returned_value[1]
            self._r_squared_BS[bs_id] = returned_value[2]
            self._koff_BS_boot[bs_id] = returned_value[3]
            self._r_squared_BS_boot[bs_id] = returned_value[4]
        # update dataset
        for data, column_name in zip(
                [self._koff_BS, self._koff_BS_boot, self._res_time_BS, self._r_squared_BS, self._r_squared_BS_boot],
                ["Binding Site Koff", "Binding Site Koff Bootstrap avg", "Binding Site Residence Time",
                 "Binding Site R Squared", "Binding Site R Squared Bootstrap avg"]):
            data_per_residue = np.zeros(self._nresi_per_protein)
            for bs_id, nodes in enumerate(self._node_list):
                data_per_residue[nodes] = data[bs_id]
            self.dataset[column_name] = data_per_residue
        if len(selected_bs_id) == 1:
            return self._koff_BS[bs_id], self._res_time_BS[bs_id]
        else:
            return [self._koff_BS[bs_id] for bs_id in selected_bs_id], \
                   [self._res_time_BS[bs_id] for bs_id in selected_bs_id]

    def analyze_bound_poses(self, binding_site_id=None, n_top_poses=3, n_clusters="auto", pose_format="gro",
                            score_weights=None, kde_bw=0.15, pca_component=0.90, plot_rmsd=True, save_dir=None,
                            eps=None, min_samples=None, metric="euclidean",
                            fig_close=False, fig_format="pdf", num_cpus=None):
        r"""Analyze bound poses for binding sites.

        This function can find representative bound poses, cluster the bound poses and calculate pose RMSD for
        binding sites.

        If ``n_top_poses`` is an integer larger than 0, this method will find the representative bound poses for the specified
        binding sites. To do so, it evaluates all the bound poses in a binding site using a density-based scoring function
        and ranks the poses using based on the scores. The scoring function is defined as:

        .. math::
            \text { score }=\sum_{i} W_{i} \cdot \hat{f}_{i, H}(D)

        where :math:`W_{i}` is the weight given to atom i of the lipid molecule, H is the bandwidth and
        :math:`\hat{f}_{i, H}(D)` is a multivariate kernel density etimation of the position of atom i in the specified
        binding site. :math:`\hat{f}_{i, H}(D)` is calculated from all the bound lipid poses in that binding site.
        The position of atom i is a `p`-variant vector, :math:`\left[D_{i 1}, D_{i 2}, \ldots, D_{i p}\right]` where
        :math:`D_{i p}` is the minimum distance to the residue `p` of the binding site. The multivariant kernel density
        is estimated by `KDEMultivariate 
        <https://www.statsmodels.org/devel/generated/statsmodels.nonparametric.kernel_density.KDEMultivariate.html>`_ 
        provided by Statsmodels. Higher weights can be given to e.g. the headgroup atoms of phospholipids, to generate
        better defined binding poses, but all lipid atoms are weighted equally by default. The use of relative positions
        of lipid atoms in their binding site makes the analysis independent of the conformational changes in the rest
        part of the protein.

        If ``n_clusters`` is given an integer larger than 0, this method will cluster the lipid bound poses in the specified
        binding site using `KMeans <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_
        provided by scikit. The KMeans cluster separates the samples into `n` clusters of equal variances, via minimizing
        the `inertia`, which is defined as:

        .. math::
            \sum_{i=0}^{n} \min _{u_{i} \in C}\left(\left\|x_{i}-u_{i}\right\|^{2}\right)

        where :math:`u_{i}` is the `centroid`  of cluster i. KMeans scales well with large dataset but performs poorly
        with clusters of varying sizes and density, which are often the case for lipid poses in a binding site.

        If ``n_clusters`` is set to `auto`, this method will cluster the bound poses using a density-based cluster
        `DBSCAN <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`_ provided by scikit.
        DBSCAN finds clusters of core samples of high density. A sample point is a core sample if at least `min_samples`
        points are within distance :math:`\varepsilon` of it. A cluster is defined as a set of sample points that are
        mutually density-connected and density-reachable, i.e. there is a path
        :math:`\left\langle p_{1}, p_{2}, \ldots, p_{n}\right\rangle` where each :math:`p_{i+1}` is within distance
        :math:`\varepsilon` of :math:`p_{i}` for any two p in the two. The values of `min_samples` and :math:`\varepsilon`
        determine the performance of this cluster. If None, `min_samples` takes the value of 2 * ndim. If
        :math:`\varepsilon` is None, it is set as the value at the knee of the k-distance plot.

        For writing out the cluster poses, this method will randomly select one pose from each cluster in the case of
        using KMeans or one from the core samples of each cluster when DBSCAN is used, and writes the selected lipid
        pose with the protein conformation to which it binds using MDTraj, in the provided pose format.

        The root mean square deviation (RMSD) of a lipid bound pose in a binding site is calculated from the relative
        position of the pose in the binding site compared to the average position of the bound poses. Thus, the pose
        RMSD is defined as:

        .. math::
            RMSD=\sqrt{\frac{\sum_{i=0}^{N} \sum_{j=0}^{M}\left(D_{i j}-\bar{D}_{j}\right)^{2}}{N}}

        where :math:`D_{i j}` is the distance of atom `i` to the residue `j` in the binding site; :math:`\bar{D}_{j}` is the
        average distance of atom `i` from all bound poses in the binding site to residue `j`; `N` is the number of atoms in
        the lipid molecule and `M` is the number of residues in the binding site.

        Parameters
        ----------
        binding_site_id : int or list of int, default=None
            The binding site ID used in PyLipID. This ID is the index in the binding site node list that is
            calculated by the method ``compute_binding_nodes``. The ID of the N-th binding site is (N-1). If None,
            the contact duration of all binding sites are calculated.

        n_top_poses : int, default=3
            Number of representative bound poses written out for the selected binding site.

        n_clusters : int or 'auto'
            Number of clusters to form for bound poses of the selected binding site.  If ``n_clusters`` is set to 'auto'`, the
            density-based clusterer DBSCAN will be used. If ``n_clusters``  is given a non-zero integer, KMeans is used.

        pose_format : str, default="gro"
            The coordinate format the representative poses and clsutered poses are saved with. Support the formats
            that are included in MDtraj.save().

        score_weights : dict or None, default=None
            The weights given to atoms in the scoring function for finding the representative bound poses. It should in
            the format of a Python dictionary {atom name: weight}. The atom name should be consisten with the topology.
            By default, all atoms in the lipid molecule are weighted equally.

        kde_bw : scalar, default=0.15
            The bandwidth for the Gaussian kernel. Used in the density estimation of the lipid atom coordinates in the binding
            site. Used by the function
            `KDEMultivariate <https://www.statsmodels.org/devel/generated/statsmodels.nonparametric.kernel_density.KDEMultivariate.html>`_ .

        pca_component : int, float or ‘mle’, default=0.90
            The PCA used to decrease the dimensions of lipid atom coordinates. The coordinate of a lipid atom in
            the binding site is expressed as a distance vector of the minimum distances to the residues in that binding site,
            i.e. :math:`[D_{i 1}, D_{i 2}, .., D_{i p}]`. This can be in high dimensions. Hence, PCA is used on this distance
            vector prior to calculation of the density. This PCA is carried out by
            `PCA <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_ in sci-kit.

        plot_rmsd : bool, default=True
            Plot the binding site RMSDs in a violinplot.

        eps : float or None, default=None
            The minimum neighbour distance used by
            `DBSCAN <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`_
            if None, the value will determined from as the elbow point of the sorted minimum neighbour distance 
            of all the data points. 

        min_samples : int or None, default=None
            The minimum number of samples to be considered as core samples used by 
            `DBSCAN <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`_ .
            If None, the value will be automatically determined based on the size of data. 

        metric : str, default='euclidean'
            The metric used to calculated neighbour distance used by 
            `DBSCAN <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`_ .
            Default is the Euclidean distance. 

        fig_close : bool, default=False
            This parameter control whether to close the plotted figures using plt.close(). It can save memory if 
            many figures are generated. 

        fig_format : str, default="pdf"
            Figure format. Support formats included in matplotlib.pyplot.

        num_cpus : int or None default=None
            The number of CPUs used for the tasks of ranking the poses and clustering poses. Python multiprocessing deployed by
            `p_tqdm <https://github.com/swansonk14/p_tqdm>`_ is used to speed up these calculations. 

        save_dir : str or None, default=None
            The root directory for saving the pose analysis results. If None, the root directory set at the initiation of
            ``LipidInteraction`` will be used. The representative poses and clustered poses will be stored in the directory
            of Bound_Poses_{lipid} under the root directory.

        Returns
        -------
        pose_pool : dict
            Coordinates of the bound poses for the selected binding sites stored in a python dictionary
            {binding_site_id: pose coordinates}. The poses coordinates include lipid coordinates and those of the receptor
            at the time the pose was bound. The pose coordinates are stored in a
            `mdtraj.Trajectory <https://mdtraj.org/1.9.4/api/generated/mdtraj.Trajectory.html>`_ object.

        rmsd_data : pandas.DataFrame
            Bound poses RMSDs are stored by columns with column name of binding site id.

        See Also
        --------
        pylipid.func.collect_bound_poses
            Collect bound pose coordinates from trajectories.
        pylipid.func.vectorize_poses
            Convert bound poses into distance vectors.
        pylipid.func.calculate_scores
            Score the bound poses based on the probability density function of the position of lipid atoms
        pylipid.func.analyze_pose_wrapper
            A wrapper function that ranks poses, clusters poses and calculates pose RMSD

        """
        self._check_calculation("Binding Site ID", self.compute_binding_nodes, print_data=False)

        pose_dir = check_dir(save_dir, "Bound_Poses_{}".format(self._lipid)) if save_dir is not None \
            else check_dir(self._save_dir, "Bound_Poses_{}".format(self._lipid))
        if "Binding Site RMSD" in self.dataset.columns:
            pose_rmsd_per_residue = self.dataset.columns["Binding Site Pose RMSD"]
        else:
            pose_rmsd_per_residue = np.zeros(self._nresi_per_protein)
        if binding_site_id is not None:
            selected_bs_id = np.atleast_1d(binding_site_id)
        else:
            selected_bs_id = np.arange(len(self._node_list), dtype=int)

        # store bound lipid poses
        selected_bs_map = {bs_id: self._node_list[bs_id] for bs_id in selected_bs_id}
        pose_traj, pose_info = collect_bound_poses(selected_bs_map, self._contact_residues_low, self._trajfile_list,
                                                   self._topfile_list, self._lipid, self._protein_ref, self._lipid_ref,
                                                   stride=self._stride, nprot=self._nprot)
        protein_atom_indices = [[atom.index for atom in residue.atoms]
                                for residue in self._protein_ref.top.residues]
        lipid_atom_indices = [self._protein_ref.n_atoms + atom_idx
                              for atom_idx in np.arange(self._lipid_ref.n_atoms)]
        atom_weights = {atom_idx: 1 for atom_idx in np.arange(self._lipid_ref.n_atoms)}
        if score_weights is not None:
            translate = {atom_idx: score_weights[self._lipid_ref.top.atom(atom_idx).name]
                         for atom_idx in np.arange(self._lipid_ref.n_atoms)
                         if self._lipid_ref.top.atom(atom_idx).name in score_weights.keys()}
            atom_weights.update(translate)

        if n_top_poses > 0:
            # multiprocessing wrapped under p_tqdm
            rmsd_set = p_map(partial(analyze_pose_wrapper, protein_atom_indices=protein_atom_indices,
                                     lipid_atom_indices=lipid_atom_indices, n_top_poses=n_top_poses,
                                     pose_dir=pose_dir, atom_weights=atom_weights, kde_bw=kde_bw,
                                     pca_component=pca_component, pose_format=pose_format, n_clusters=n_clusters,
                                     eps=eps, min_samples=min_samples, metric=metric,
                                     trajfile_list=self._trajfile_list),
                             selected_bs_id, [pose_traj[bs_id] for bs_id in selected_bs_id],
                             [self._node_list[bs_id] for bs_id in selected_bs_id],
                             [pose_info[bs_id] for bs_id in selected_bs_id], num_cpus=num_cpus, desc="ANALYZE BOUND POSES")
            RMSD_set = {}
            for bs_id, rmsd in zip(selected_bs_id, rmsd_set):
                RMSD_set["Binding Site {}".format(bs_id)] = rmsd
                pose_rmsd_per_residue[self._node_list[bs_id]] = np.mean(RMSD_set["Binding Site {}".format(bs_id)])
        # update dataset
        self.dataset["Binding Site Pose RMSD"] = pose_rmsd_per_residue
        pose_rmsd_data = pd.DataFrame(
            dict([(bs_label, pd.Series(rmsd_set)) for bs_label, rmsd_set in RMSD_set.items()]))
        # plot RMSD
        if plot_rmsd and n_top_poses > 0:
            plot_binding_site_data(pose_rmsd_data, os.path.join(pose_dir, f"Pose_RMSD_violinplot.{fig_format}"),
                                   title="{}".format(self._lipid), ylabel="RMSD (nm)", fig_close=fig_close)
        return pose_traj, pose_rmsd_data

    def compute_surface_area(self, binding_site_id=None, radii=None, plot_data=True, save_dir=None,
                             fig_close=False, fig_format="pdf", num_cpus=None):
        """Calculate binding site surface areas.

        The accessible surface area is calculated using the Shrake and Rupley algorithm [1]_. The basic idea of this
        algorithm is to generate a mesh of points representing the surface of each atom and then count the number of
        points that are not within the radius of any other atoms. The surface area can be derived from this number of
        exposed points.

        This method utilizes the shrake_rupley function of MDTraj for calculation of the surface area. In implementation,
        this method scripts the protein coordinates out of the simulation system and obtains the accessible surface area
        of a binding site by summing those of its comprising residues

        Atom radius is required for calculation of surface areas. MDtraj defines the radii for common atoms (see
        `here <https://github.com/mdtraj/mdtraj/blob/master/mdtraj/geometry/sasa.py#L56>`_). The radius of the BB bead
        in MARTINI2 is defined as 0.26 nm, the SC1/SC2/SC3 are defined as 0.23 nm in this method. Use the param ``radii``
        to define or change of definition of atom radius.

        Parameters
        -----------
        binding_site_id : int or list of int, default=None
            The binding site ID used in PyLipID. This ID is the index in the binding site node list that is
            calculated by the method ``compute_binding_nodes``. The ID of the N-th binding site is (N-1). If None,
            the contact duration of all binding sites are calculated.

        radii : dict or None, default=None
            The atom radii in the python dictionary format of {atom name: radius}

        plot_data : bool, default=True
            Plot surface area data for the selected binding sites in a violinplot and in a time series plot.

        save_dir : str or None, default=None
            The directory for saving the surface area plot. If None, it will save in Bound_Poses_{lipid} under the root
            directory defined at the initiation of ``LipidInteraction``.

        fig_close : bool, default=False
            This parameter control whether to close the plotted figures using plt.close(). It can save memory if
            many figures are generated.

        fig_format : str, default="pdf"
            Figure format. Support formats included in matplotlib.pyplot.

        num_cpus : int or None default=None
            The number of CPUs used for calculating the surface areas. Python multiprocessing deployed by
            `p_tqdm <https://github.com/swansonk14/p_tqdm>`_ is used to speed up these calculations.

        Returns
        -------
        surface_area : pandas.DataFrame
            Binding site surface areas as a function of time for the selected binding sites. The surface area values are
            stored by columns with the column name of binding site id and the time information is stored in the column
            named "Time".

        See Also
        ---------
        pylipid.func.calculate_surface_area_wrapper
            A wrapper function for calculating binding site surface area from a trajectory.
        pylipid.plot.plot_surface_area
            Plot binding site surface area as a function of time.
        pylipid.plot.plot_binding_site_data
            Plot binding site data in a matplotlib violin plot.

        References
        ----------
        .. [1] Shrake, A.; Rupley, J. A., Environment and exposure to solvent of protein atoms. Lysozyme and insulin.
            Journal of Molecular Biology 1973, 79 (2), 351-371

        """
        MARTINI_CG_radii = {"BB": 0.26, "SC1": 0.23, "SC2": 0.23, "SC3": 0.23}

        self._check_calculation("Binding Site ID", self.compute_binding_nodes, print_data=False)

        if "Binding Site Surface Area" in self.dataset.columns:
            # keep existing data
            surface_area_per_residue = np.array(self.dataset["Binding Site Surface Area"].tolist())
        else:
            surface_area_per_residue = np.zeros(self._nresi_per_protein)
        if radii is None:
            radii_book = MARTINI_CG_radii
        else:
            radii_book = {**MARTINI_CG_radii, **radii}

        # calculate binding site surface area
        selected_bs_id = np.atleast_1d(np.array(binding_site_id, dtype=int)) if binding_site_id is not None \
            else np.arange(len(self._node_list), dtype=int)
        selected_bs_id_map = {bs_id: self._node_list[bs_id] for bs_id in selected_bs_id}
        returned_values = p_map(partial(calculate_surface_area_wrapper, binding_site_map=selected_bs_id_map,
                                        nprot=self._nprot, timeunit=self._timeunit, stride=self._stride,
                                        dt_traj=self._dt_traj, radii=radii_book), self._trajfile_list,
                                self._topfile_list, np.arange(len(self._trajfile_list), dtype=int),
                                num_cpus=num_cpus, desc="CALCULATE BINDING SITE SURFACE AREA")
        surface_data = []
        data_keys = []
        for returned_tuple in returned_values:
             for idx in np.arange(len(returned_tuple[0])):
                surface_data.append(returned_tuple[0][idx])
                data_keys.append(returned_tuple[1][idx])
        surface_area_data = pd.concat(surface_data, keys=data_keys)
        # update dataset
        for bs_id in selected_bs_id:
            nodes = self._node_list[bs_id]
            surface_area_per_residue[nodes] = surface_area_data["Binding Site {}".format(bs_id)].mean()
        self.dataset["Binding Site Surface Area"] = surface_area_per_residue
        # plot surface area
        if plot_data:
            if save_dir is not None:
                surface_area_dir = check_dir(save_dir)
            else:
                surface_area_dir = check_dir(self._save_dir, "Bound_Poses_{}".format(self._lipid))
            plot_surface_area(surface_area_data,
                              os.path.join(surface_area_dir,
                                           "Surface_Area_{}_timeseries.{}".format(self._lipid, fig_format)),
                              timeunit=self._timeunit, fig_close=fig_close)
            selected_columns = [column for column in surface_area_data.columns if column != "Time"]
            surface_data_noTimeSeries = surface_area_data[selected_columns]
            plot_binding_site_data(surface_data_noTimeSeries,
                                   os.path.join(surface_area_dir,
                                                "Surface_Area_{}_violinplot.{}".format(self._lipid, fig_format)),
                                   title="{}".format(self._lipid), ylabel=r"Surface Area (nm$^2$)",
                                   fig_close=fig_close)
        return surface_area_data


    #################################
    #    save and plot
    #################################
    def save_data(self, item, save_dir=None):
        """Assisting function for saving data.

        This function saves a couple of unprocessed interaction data to local disc. These data include:

        - ``Duration``: a python dictionary with residue IDs as its keys, which stores the durations of all contacts
          for residues.

        - ``Occupancy``: a python dictionary with residue IDs as its keys, which stores the lipid occupancy from each
          trajectory for residues.

        - ``Lipid Count``: a python dictionary with residue IDs as its keys, which stores the averaged lipid count
          from each trajectory for residues.

        - ``CorrCoef`` : a numpy ndarray that stores the interaction correlation matrix of residues.

        - ``Duration BS``: a python dictionary with binding site IDs as its keys, which stores the durations of all
          contacts for binding sites.

        - ``Occupancy BS``: a python dictionary with binding site IDs as its keys, which stores the lipid occupancy
          from each trajectory for binding sites.

        - ``Dataset``: a pandas.DataFrame object that stores interaction data for residues by row. The interaction
          data, including interaction with the residue and with the binding site to which the residue belongs, can be
          accessed easily.

        The python dictionary objects and numpy ndarray objects are saved to local disc in pickle, whereas the
        pandas.DataFrame is saved in csv format.

        The data will be saved under the directory specified by ``save_dir``. If None is given, a directory of
        Dataset_{lipid} will be created under the root directory initiated at the begining of the calculation and used to
        store the dataset.

        Parameters
        ----------
        item : {"Dataset", "Duration", "Occupancy", "Lipid Count", "CorrCoef", "Duration BS", "Occupancy BS"}
            The interaction data to save.

        save_dir : str, optional, default=None
            The directory for saving the data. By default, the data are saved in Dataset_{lipid} under the root directory
            defined at the initialization of ``LipidInteraction``.

        """
        data_dir = check_dir(save_dir, "Dataset_{}".format(self._lipid)) if save_dir is not None \
            else check_dir(self._save_dir, "Dataset_{}".format(self._lipid))

        if item.lower() == "duration":
            obj = self._duration
        elif item.lower() == "occupancy":
            obj = self._occupancy
        elif item.lower() == "lipid count":
            obj = self._lipid_count
        elif item.lower() == "corrcoef":
            obj = self.interaction_corrcoef
        elif item.lower() == "duration bs":
            obj = self._duration_BS
        elif item.lower() == "occupancy bs":
            obj = self._occupancy_BS
        if 'obj' in locals():
            with open(os.path.join(data_dir, "{}.pickle".format("_".join(item.split()))), "wb") as f:
                pickle.dump(obj, f, 2)

        if item.lower() == "dataset":
            self.dataset.to_csv(os.path.join(data_dir, "Dataset.csv"), header=True, index=False)

        return

    def save_coordinate(self, item, save_dir=None, fn_coord=None):
        """Save protein coordinates in PDB format with interaction data in the B factor column.

        In this method, the receptor coordinates, which are taken from the first frame of the trajectory, are written
        out in the PDB format with the specified interaction data stored in the B factor column. Supported interaction
        data include: ``Residence Time``, ``Duration``, ``Occupancy`` and ``Lipid Count``. These are interaction data
        for residues.

        By default, the coordinates will be saved in the directory of Coordinate_{lipid} under the root directory defined
        when the class ``LipidInteraction`` was initiated. It can be changed by providing to path to ``save_dir``.

        By default, the coordinates file is saved as Coordinate_{lipid}_{item}.pdb. This can be changed by providing
        a file name to ``fn_coord``.

        Parameters
        -----------
        item : {"Residence Time", "Duration", "Occupancy", "Lipid Count"}
            Interaction data to be stored in the B factor column.

        save_dir : str or None, default=None
            The directory for saving the coordinate file. By default, the coordinate file is saved at the directory of
            Coordinate_{lipid} under the root directory defined at the initialization of ``LipidInteraction``.

        fn_coord : str or None, default=None
            The file name of the written coordinate file. By default, the file is named as Coordinate_{lipid}_{item}.pdb

        See Also
        --------
        pylipid.util.write_PDB
            Write interaction data in bfactor columns.

        """
        coord_dir = check_dir(save_dir, "Coordinate_{}".format(self._lipid)) if save_dir is not None \
            else check_dir(self._save_dir, "Coordinate_{}".format(self._lipid))
        if fn_coord is None:
            fn_coord = "Coordinate_{}_{}.pdb".format(self._lipid, "_".join(item.split()))
        data = self.dataset[item].tolist()
        write_PDB(self._protein_ref, data, os.path.join(coord_dir, fn_coord), resi_offset=self._resi_offset)
        return

    def save_pymol_script(self, pdb_file, save_dir=None):
        """Save a pymol script that maps interactions onto protein structure in PyMol.

        This method will save a python script, named ``show_binding_site_info.py``,  which opens a PyMol session and maps
        the binding site information to the receptor coordinates in that session. The receptor coordinate is provided
        through the parameter ``pdb_file``. As PyMol only recognize atomistic structures, it needs a receptor atomistic
        structure before coarse-graining for those coarse-grained simulations. For the MARTINI simulations, the receptor
        atomistic coordinate can be the one used by `martinize.py`, which is a python script that converts atomistic
        structures to MARTINI corase-grained models. Regardless of the simulation models, the provided receptor atomistic
        coordinates should have the same topology as in the simulations.

        In the PyMol session, residues from the same binding sites are shown in the same color and shown in spheres with
        scales corresponding to their lipid residence times. This PyMol session can provide a quick overview of the binding
        sites and provide a better understanding of the structural details.

        By default, this script is saved under the directory Dataset_{lipid}, together with the file Dataset.csv, from
        which the script reads the interaction data. The directory for storing the script can be changed by providing a
        path to ``save_dir``.

        Parameters
        ----------
        pdb_file : str
            The file of receptor atomistic coordinates.
        save_dir : str, optional, default=None
            The directory for saving the python script. By default, the script is saved at the directory of
            Dataset_{lipid}, together with the file Dataset.csv, from which it reads the interaction data.

        See Also
        --------
        pylipid.util.write_pymol_script
            Write Python script that opens a PyMol session with binding site information.

        """
        script_dir = check_dir(save_dir) if save_dir is not None else \
            check_dir(os.path.join(self._save_dir, "Dataset_{}".format(self._lipid)))
        data_fname = os.path.join(script_dir, "Dataset.csv".format(self._lipid))
        if not os.path.isfile(data_fname):
            self.dataset.to_csv(data_fname, index=False, header=True)
        write_pymol_script(os.path.join(script_dir, "show_binding_site_info.py"), pdb_file, data_fname,
                           self._lipid, len(self._node_list))
        return

    def plot(self, item, save_dir=None, gap=200, fig_close=False, fig_format="pdf"):
        """Assisting function for plotting interaction data.

        This assiting function can make a couple of plots:
            - ``Residence Time``: plot residue residence times as a function of residue index.
            - ``Duration``: plot residue lipid durations as a function of residue index.
            - ``Occupancy``: plot residue lipid occupancies as a function of residue index.
            - ``Lipid Count``: plot residue lipid count as a function of residue index.
            - ``CorrCoef``: plot the Pearson correlation matrix of the lipid interactions for residues.

        By default, the figures are saved in the directory of Figure_{lipid} under the root directory defined
        when the class ``LipidInteraction`` was initiated. It can be changed by providing to path to ``save_dir``.

        The figures are named as {item}.{fig_format}. The formats supported by matplotlib.savefig() are allowed.

        For proteins with large gaps in their residue index, this method provides the parameter ``gap`` to plot data
        in multiple figures when the gap of residue index for two adjacent residues is larger than the specified value.

        For multichain proteins, this method will plot data in separate figure for each chain.

        Parameters
        ----------
        item : {"Residence Time", "Duration", "Occupancy", "Lipid Count", "CorrCoef"}
            The interaction data to be plotted.

        save_dir : str, default=None
            The directory for saving the figures. By default, figures are saved in the directory of Figures_{lipid}
            under the root directory defined at the initialization of ``LipidInteraction``.

        gap : int, default=200
            The gap in residue index to initiate a new figure.

        fig_close : bool, default=False
            This parameter control whether to close the plotted figures using plt.close(). It can save memory if
            many figures are generated.

        fig_format : str, default="pdf"
            Figure format. Support formats included in matplotlib.pyplot.

        See Also
        --------
        pylipid.plot.plot_residue_data
            Plot interactions as a function of residue index

        """
        figure_dir = check_dir(save_dir, "Figure_{}".format(self._lipid)) if save_dir is not None \
            else check_dir(self._save_dir, "Figure_{}".format(self._lipid))

        if item == "Residence Time":
            ylabel = "Res. Time (ns)" if self._timeunit == "ns" else r"Res. Time ($\mu$s)"
        elif item == "Duration":
            ylabel = "Duration (ns)" if self._timeunit == 'ns' else r"Duration ($\mu$s)"
        elif item == "Occupancy":
            ylabel = "Occupancy (%)"
        elif item == "Lipid Count":
            ylabel = "Lipid Count (num.)"
        if "ylabel" in locals():
            data = self.dataset[item]
            title = "{} {}".format(self._lipid, item)
            fig_fn = os.path.join(figure_dir, "{}.{}".format("_".join(item.split()), fig_format))
            residue_index = np.array([int(re.findall("^[0-9]+", residue)[0]) for residue in self._residue_list])
            plot_residue_data(residue_index, data, gap=gap, ylabel=ylabel, fn=fig_fn, title=title,
                              fig_close=fig_close)

        if item == "CorrCoef":
            residue_index = np.array([int(re.findall("^[0-9]+", residue)[0]) for residue in self._residue_list])
            plot_corrcoef(self.interaction_corrcoef, residue_index, fn=os.path.join(figure_dir,
                                                                                    f"CorrCoef.{fig_format}"),
                          title="{} Correlation Coeffient".format(self._lipid), fig_close=fig_close)

        return

    def plot_logo(self, item, letter_map=None, color_scheme="chemistry", gap=2000, save_dir=None,
                  fig_close=False, fig_format="pdf"):
        """Plot interactions using Logomaker.

        `Logomaker <https://logomaker.readthedocs.io/en/latest/>`_ is a Python package for generating publication-quality
        sequence logos. This method adopts Logomaker to plot the interacting amino acids. The height of the amino acids
        illustrates the strength of the interaction. This plot can show the amino acids composition of lipid interactions
        and make the checking of chemical properties of lipid interaction easier.

        ``color_scheme`` specifies the colormaps to illustrate the chemical properties of amino acids. See
        `here <https://logomaker.readthedocs.io/en/latest/examples.html#color-schemes>`_ for a list of color scheme
        supported by Logomaker.

        ``letter_map`` requires a python dictionary that maps the residue name to single letters (i.e. {"ALA": "A"}).
        The 20 common amino acids are defined in this method. Other amino acids/residue names need to be defined using
        this parameter.

        The figures are saved with the file name of {item}_logo.{fig_format}. The formats supported by matplotlib.savefig()
        are allowed.

        By default, the figures are saved in the directory of Figure_{lipid} under the root directory defined
        when the class ``LipidInteraction`` was initiated. It can be changed by providing to path to ``save_dir``.

        The rest of the parameters are the same as :meth:`~LipidInteraction.plot`.

        Parameters
        ----------
        item : {"Residence Time", "Duration", "Occupancy", "Lipid Count"}
            The interaction data to plot.

        letter_map : dict or None, optional, default=None
            A python dictionary that maps the residue names to single letters.

        color_scheme : str, optional, default="chemistry"
            Logomaker color schemes. The default is 'chemistry'.

        save_dir : str, default=None
            The directory for saving the figures. By default, figures are saved in the directory of Figures_{lipid}
            under the root directory defined at the initialization of ``LipidInteraction``.

        gap : int, default=200
            The gap in residue index to initiate a new figure.

        fig_close : bool, default=False
            This parameter control whether to close the plotted figures using plt.close(). It can save memory if
            many figures are generated.

        fig_format : str, default="pdf"
            Figure format. Support formats included in matplotlib.

        See Also
        --------
        pyipid.api.LipidInteraction.plot
            Assisting function for plotting interaction data.
        pylipid.plot.plot_residue_data_logo
            Plot interactions using `logomaker.Logo
            <https://logomaker.readthedocs.io/en/latest/implementation.html#logo-class>`_.

        """
        figure_dir = check_dir(save_dir, "Figure_{}".format(self._lipid)) if save_dir is not None \
            else check_dir(self._save_dir, "Figure_{}".format(self._lipid))

        single_letter = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
                         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
                         'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
        if letter_map is not None:
            single_letter.update(letter_map)

        resname_set = [re.findall("[a-zA-Z]+$", residue)[0] for residue in self._residue_list]
        residue_index = np.array([int(re.findall("^[0-9]+", residue)[0]) for residue in self._residue_list])

        if item == "Residence Time":
            ylabel = "Res. Time (ns)" if self._timeunit == "ns" else r"Res. Time ($\mu$s)"
        elif item == "Duration":
            ylabel = "Duration (ns)" if self._timeunit == 'ns' else r"Duration ($\mu$s)"
        elif item == "Occupancy":
            ylabel = "Occupancy (%)"
        elif item == "Lipid Count":
            ylabel = "Lipid Count (num.)"

        if "ylabel" in locals():
            data = self.dataset[item]
            title = "{} {} Logo".format(self._lipid, item)
            fig_fn = os.path.join(figure_dir, "{}_logo.{}".format("_".join(item.split()), fig_format))
            plot_residue_data_logo(residue_index, resname_set, data, ylabel=ylabel,
                                    fn=fig_fn, title=title, letter_map=letter_map, color_scheme=color_scheme,
                                    fig_close=fig_close)
        else:
            print("Invalid input for item!")

        return

    def write_site_info(self, sort_residue="Residence Time", save_dir=None, fn=None):
        """Write a report on binding site with lipid interaction information.

        This method writes a report of binding site information in a txt file. This report includes, for each binding
        site, the various interaction data and properties of the binding site, followed by a table of comprising residues
        and the interaction data of these residues. This table is sorted in an order determined by the item provided by
        ``sort_residue``. This report provides a quick and formatted view of the binding site information.

        By default, the report is saved in the root directory defined when the class ``LipidInteraction`` was initiated.
        It can be changed by providing to path to ``save_dir``.

        By default, the report is saved with a filename of BindingSites_Info_{lipid}.txt. It can be changed via the parameter
        ``fn``.

        Parameters
        ----------
        sort_residue : {"Residence Time", "Duration", "Occupancy", "Lipid Count"}, default="Residence Time"
            The item used for sorting the binding site residues.

        save_dir : str, default=None
            The directory for saving the report. By default the report is saved in the root directory defined when
            ``LipidInteraction`` was initialized.

        fn : str, default=None
            The filename of the report. By default the report is saved with the name BindingSites_Info_{lipid}.txt

        """
        if len(self._node_list) == 0:
            print("No binding site was detected!!")
        else:
            BS_dir = check_dir(save_dir) if save_dir is not None \
                else check_dir(self._save_dir)
            if fn is None:
                fn = "BindingSites_Info_{}.txt".format(self._lipid)
            self._check_calculation("Binding Site ID", self.compute_binding_nodes, print_data=False)
            self._check_calculation("Binding Site Koff", self.compute_site_koff, plot_data=False)
            mapping_funcs = {"Residence Time": self.compute_residue_koff,
                             "Duration": self.compute_residue_duration,
                             "Occupancy": self.compute_residue_occupancy,
                             "Lipid Count": self.compute_residue_lipidcount}
            self._check_calculation(sort_residue, mapping_funcs[sort_residue])
            with open(os.path.join(BS_dir, fn), "a") as f:
                f.write(f"## Network modularity {self._network_modularity:5.3f}")
                f.write("\n")
                for bs_id, nodes in enumerate(self._node_list):
                    text = self._format_BS_print_info(bs_id, nodes, sort_residue)
                    f.write(text)
        return

    def show_stats_per_traj(self, write_log=True, print_log=True, fn_log=None):
        """Show stats of lipid interaction per trajectory.

        This assisting function show some quick stats for each trajectory. These stats include the 10 residues
        that showed longest durations, the highest occupancies and the largest lipid count. This stats provide a quick
        way to check interactions from each trajectory. The stats info can be printed in sys.stdout or written in a file.

        Parameters
        ----------
        write_log : bool, default=True
            Whether to write the interaction stats in a log file.

        print_log : bool, default=True
            Whether to print the interaction stats in sys.stdout.

        fn_log : str, default=None
            The filename of stats log file. By default, the log file is saved with the name calculation_log_{lipid}.txt.
            This log file is saved in the root directory defined when the class ``LipidInteraction`` was initiated.

        """
        if fn_log is None:
            fn_log = "calculation_log_{}.txt".format(self.lipid)
        text = []
        text.append("###### Lipid: {}\n".format(self._lipid))
        text.append("###### Lipid Atoms: {}\n".format(self._lipid_atoms))
        text.append("###### Cutoffs: {}\n".format(self.cutoffs))
        text.append("###### nprot: {}\n".format(self._nprot))
        text.append("###### Trajectories:\n")
        for traj_fn in self._trajfile_list:
            text.append("  {}\n".format(traj_fn))
        text.append("###### Coordinates:\n")
        for top_fn in self._topfile_list:
            text.append("  {}\n".format(top_fn))
        # interaction information by trajectory
        for traj_idx, trajfile in enumerate(self._trajfile_list):
            # sort values
            durations = np.array(
                [np.concatenate(
                    self._duration[residue_id][traj_idx * self._nprot:(traj_idx + 1) * self._nprot]).mean()
                 for residue_id in self._protein_residue_id])
            duration_arg_idx = np.argsort(durations)[::-1]
            occupancies = np.array(
                [np.mean(self._occupancy[residue_id][traj_idx * self._nprot:(traj_idx + 1) * self._nprot])
                 for residue_id in self._protein_residue_id])
            occupancy_arg_idx = np.argsort(occupancies)[::-1]
            lipid_counts = np.array(
                [np.mean(self._lipid_count[residue_id][traj_idx * self._nprot:(traj_idx + 1) * self._nprot])
                 for residue_id in self._protein_residue_id])
            lipid_count_arg_idx = np.argsort(lipid_counts)[::-1]

            text.append(
                "\n########## {} interactions in \n########## {} \n".format(self._lipid, trajfile))
            text.append(
                "10 residues that showed longest average interaction durations ({}):\n".format(self._timeunit))
            for residue, duration in zip(self._residue_list[duration_arg_idx][:10], durations[duration_arg_idx][:10]):
                text.append("{:^8s} -- {:^8.3f}\n".format(residue, duration))
            text.append("10 residues that showed highest lipid occupancy (100%):\n")
            for residue, occupancy in zip(self._residue_list[occupancy_arg_idx][:10],
                                          occupancies[occupancy_arg_idx][:10]):
                text.append("{:^8s} -- {:^8.2f}\n".format(residue, occupancy))
            text.append("10 residues that have the largest number of surrounding lipids (count):\n")
            for residue, lipid_count in zip(self._residue_list[lipid_count_arg_idx][:10],
                                            lipid_counts[lipid_count_arg_idx][:10]):
                text.append("{:^8s} -- {:^8.2f}\n".format(residue, lipid_count))
            text.append("\n")
            text.append("\n")

        log = "".join(text)

        if write_log:
            with open("{}/{}".format(self._save_dir, fn_log), "a+") as f:
                f.write(log)

        if print_log:
            print(log)

        return

    ############################################
    #     assisting func
    ############################################

    def _format_BS_print_info(self, bs_id, nodes, sort_item):
        """Format binding site information."""
        Residue_property_book = {"ARG": "Pos. Charge", "HIS": "Pos. Charge", "LYS": "Pos. Charge",
                                 "ASP": "Neg. Charge", "GLU": "Neg. Charge",
                                 "SER": "Polar", "THR": "Polar", "ASN": "Polar", "GLN": "Polar",
                                 "CYS": "Special", "SEC": "Special", "GLY": "Special", "PRO": "Special",
                                 "ALA": "Hydrophobic", "VAL": "Hydrophobic", "ILE": "Hydrophobic", "LEU": "Hydrophobic",
                                 "MET": "Hydrophobic", "PHE": "Hydrophobic", "TYR": "Hydrophobic", "TRP": "Hydrophobic"}
        text = "# Binding site {}\n".format(bs_id)
        if "Binding Site Residence Time" in self.dataset.columns:
            text += "{:30s} {:10.3f} {:5s}\n".format(" Binding Site Residence Time:",
                                                     self._res_time_BS[bs_id], self._timeunit)
        if "Binding Site R Squared" in self.dataset.columns:
            text += "{:30s} {:10.3f}  R squared: {:7.4f}\n".format(" Binding Site Koff:", self._koff_BS[bs_id],
                                                                   self._r_squared_BS[bs_id])
        if "Binding Site Duration" in self.dataset.columns:
            text += "{:30s} {:10.3f} {:5s}\n".format(" Binding Site Duration:",
                                                     np.concatenate(self._duration_BS[bs_id]).mean(), self._timeunit)
        if "Binding Site Occupancy" in self.dataset.columns:
            text += "{:30s} {:10.3f} %\n".format(" Binding Site Occupancy:",
                                                 np.mean(self._occupancy_BS[bs_id]))
        if "Binding Site Lipid Count" in self.dataset.columns:
            text += "{:30s} {:10.3f}\n".format(" Binding Site Lipid Count:",
                                               np.mean(self._lipid_count_BS[bs_id]))
        if "Binding Site Surface Area" in self.dataset.columns:
            text += "{:30s} {:10.3f} nm^2\n".format(" Binding Site Surface Area:",
                            np.mean(self.dataset[self.dataset["Binding Site ID"]==bs_id]["Binding Site Surface Area"]))
        res_stats = {"Pos. Charge": 0, "Neg. Charge": 0, "Polar": 0, "Special": 0, "Hydrophobic": 0}
        # stats on the chemical properties of binding site residues
        for residue in self._residue_list[nodes]:
            res_stats[Residue_property_book[re.findall("[a-zA-Z]+$", residue)[0]]] += 1
        BS_num_resi = len(nodes)
        text += "{:20s} {:10s}\n".format(" Pos. Charge:", "/".join([str(res_stats["Pos. Charge"]), str(BS_num_resi)]))
        text += "{:20s} {:10s}\n".format(" Neg. Charge:", "/".join([str(res_stats["Neg. Charge"]), str(BS_num_resi)]))
        text += "{:20s} {:10s}\n".format(" Polar:", "/".join([str(res_stats["Polar"]), str(BS_num_resi)]))
        text += "{:20s} {:10s}\n".format(" Hydrophobic:", "/".join([str(res_stats["Hydrophobic"]), str(BS_num_resi)]))
        text += "{:20s} {:10s}\n".format(" Special:", "/".join([str(res_stats["Special"]), str(BS_num_resi)]))
        text += "{:^9s}{:^7s}{:^16s}{:^16s}{:^15s}{:^12s}{:^7s}{:^10s}\n".format("Residue", "Res ID",
                                                                                 "Res. Time ({})".format(
                                                                                     self._timeunit),
                                                                                 "Duration ({})".format(self._timeunit),
                                                                                 "Occupancy (%)", "Lipid Count",
                                                                                 "Koff", "R Squared")
        info_dict_set = self.dataset.iloc[nodes].sort_values(by=sort_item, ascending=False).to_dict("records")
        for info_dict in info_dict_set:
            info_dict_converted = defaultdict(float, info_dict)
            text += "{Residue:^9s}{Residue ID:^7d}{Residence Time:^16.3f}{Duration:^16.3f}" \
                    "{Occupancy:^15.3f}{Lipid Count:^12.3f}{Koff:^7.3f}{R Squared:^10.3f}\n".format(**info_dict_converted)
        text += "\n"
        text += "\n"
        return text

    def _check_calculation(self, item, calculation, *args, **kwargs):
        """Check BS calculation. """
        if item not in self.dataset.columns:
            print("#" * 60)
            print(f"{item} has not been calculated. Start the calculation.")
            calculation(*args, **kwargs)
            print("Finished the calculation of {}.".format(item))
            print("#" * 60)

