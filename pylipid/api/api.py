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
import pickle
import os
import re
import warnings
import mdtraj as md
import numpy as np
np.seterr(all='ignore')
from scipy.sparse import coo_matrix
from sklearn.decomposition import PCA
import pandas as pd
from tqdm import trange, tqdm
from ..func import cal_contact_residues
from ..func import Duration, cal_koff
from ..func import cal_lipidcount, cal_occupancy
from ..func import get_node_list
from ..func import collect_bound_poses, vectorize_poses, calculate_scores, write_bound_poses
from ..func import cluster_DBSCAN, cluster_KMeans
from ..func import calculate_site_surface_area
from ..plot import plot_koff
from ..plot import plot_surface_area, plot_binding_site_data
from ..plot import plot_residue_data, plot_corrcoef, plot_residue_data_logos
from ..util import check_dir, write_PDB, write_pymol_script, sparse_corrcoef, rmsd, get_traj_info


class LipidInteraction:
    def __init__(self, trajfile_list, cutoffs=[0.475, 0.7], lipid="CHOL", topfile_list=None, lipid_atoms=None,
                 nprot=1, resi_offset=0, save_dir=None, timeunit="us", stride=1, dt_traj=None):

        """The outer layer class that integrates calculations and handles workflow.

        `LipidInteraction` reads trajectory information via `mdtraj.load()`, and calculate interactions
        of the specified lipid for both protein residues and the calculated binding sites. 'LipidInteraction' also
        has a couple of assisting functions to plot the interaction data and save the generated data.

        Parameters
        ----------
        trajfile_list : str or a list of str
            Trajectory filename(s). Read by mdtraj.load() to obtain trajectory information.

        cutoffs : list of two scalar or a scalar, default=[0.475, 0.7]
            Cutoff value(s) for defining contacts. When a list of two scalar are provided, the dual-cutoff scheme
            will be used.

        lipid : str, default="CHOL"
            Lipid residue name.

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
            The root directory to store the data. All the generated dataset and created directories will be
            put under this directory.

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
    @property
    def residue_list(self):
        """Residue names."""
        return self._residue_list

    @property
    def node_list(self):
        """Residue ID list of binding site"""
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
        """Root directory of the generated data."""
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

        Use either residue_id or residue_name to obtain the information.
        Return the information in a pandas.DataFrame object.

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
        """Create contacting lipid index for residues at each frame.

        This function needs to run before any of the calculations.

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
                self._duration = {residue_id: [] for residue_id in self._protein_residue_id}
                self._occupancy = {residue_id: [] for residue_id in self._protein_residue_id}
                self._lipid_count = {residue_id: [] for residue_id in self._protein_residue_id}
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
                        md.compute_distances(traj, np.array(list(product(residue_atom_indices, lipid_atom_indices)))),
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
        """Calculate the durations of lipid contacts for residues

        Parameters
        ----------
        residue_id : int or list of int, default=None

        Returns
        -------
        durations : list, len(durations)=len(residue_id)

        """
        self._check_calculation("Residue", self.collect_residue_contacts)
        if residue_id is None:
            selected_residue_id = self._protein_residue_id
        else:
            selected_residue_id = np.atleast_1d(residue_id)

        for residue_id in selected_residue_id:
            self._duration[residue_id] = [
                        Duration(self._contact_residues_low[residue_id][traj_idx*protein_idx],
                                 self._contact_residues_high[residue_id][traj_idx*protein_idx],
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
        """Calculate the percentage of frames in which lipid contacts are formed for residues.

        Parameters
        ----------
        residue_id : int or list of int, default=None

        Returns
        -------
        occupancies : list, len(occupancies)=len(residue_id)

        """
        self._check_calculation("Residue", self.collect_residue_contacts)
        if residue_id is None:
            selected_residue_id = self._protein_residue_id
        else:
            selected_residue_id = np.atleast_1d(residue_id)
        for residue_id in selected_residue_id:
            self._occupancy[residue_id] = [cal_occupancy(self._contact_residues_low[residue_id][traj_idx*protein_idx])
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
        """Calculate the average number of surrounding lipids for residues.

        Parameters
        ----------
        residue_id : int or list of int, default=None

        Returns
        -------
        lipidcounts : list, len(lipidcount)=len(residue_id)

        """
        self._check_calculation("Residue", self.collect_residue_contacts)
        if residue_id is None:
            selected_residue_id = self._protein_residue_id
        else:
            selected_residue_id = np.atleast_1d(residue_id)
        for residue_id in selected_residue_id:
            self._lipid_count[residue_id] = [cal_lipidcount(self._contact_residues_low[residue_id][traj_idx * protein_idx])
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
                             save_dir=None, plot_data=True, fig_close=True):
        """Calculate interaction koff and residence time for residues.

        Parameters
        ----------
        residue_id : int or array_like or None, default=None
        nbootstrap : int, default=10
        initial_guess : array_like, default=None
        save_dir : str, default=None
        print_data : bool, default=True
        plot_data : bool, default=True
        fig_close : bool, default=True

        Returns
        ---------
        koff : scalar or list of scalar
        restime : scalar or list of scalar

        """
        self._check_calculation("Residue", self.compute_residue_koff)
        if plot_data:
            koff_dir = check_dir(save_dir, "Reisidue_koffs_{}".format(self._lipid)) if save_dir is not None \
                else check_dir(self._save_dir, "Residue_koffs_{}".format(self._lipid))
        self._check_calculation("Residue", self.collect_residue_contacts)
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

        for residue_id in tqdm(selected_residue_id,desc="CALCULATE KOFF FOR RESIDUES", total=len(selected_residue_id)):
            durations = np.concatenate(self._duration[residue_id])
            residue = residue_name_set[residue_id]
            if np.sum(durations) == 0:
                self._koff[residue_id] = 0
                self._res_time[residue_id] = 0
                self._r_squared[residue_id] = 0
                self._koff_boot[residue_id] = 0
                self._r_squared_boot[residue_id] = 0
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    koff, restime, properties = cal_koff(durations, t_total, timestep, nbootstrap, initial_guess)
                self._koff[residue_id] = koff
                self._res_time[residue_id] = restime
                self._r_squared[residue_id] = properties["r_squared"]
                self._koff_boot[residue_id] = np.mean(properties["ks_boot_set"], axis=0)[0]
                self._r_squared_boot[residue_id] = np.mean(properties["r_squared_boot_set"])
                if plot_data:
                    text = self._format_koff_text(properties)
                    plot_koff(durations, properties["delta_t_list"], properties["survival_rates"],
                              properties["n_fitted"], survival_rates_bootstraps=properties["survival_rates_boot_set"],
                              fig_fn=os.path.join(koff_dir, "{}.pdf".format(residue)), title=residue,
                              timeunit=self._timeunit, t_total=t_total, text=text, fig_close=fig_close)
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
        """Calculate binding sites via computing the community structures.

        Parameters
        ----------
        threshold : int, default=4
        save_dir : str or None, default=None
        print : bool, default=True

        Returns
        -------
        node_list: list
        modularity : float or None

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
            self._duration_BS = {bs_id:[] for bs_id in np.arange(len(self._node_list))}
            self._occupancy_BS = {bs_id:[] for bs_id in np.arange(len(self._node_list))}
            self._lipid_count_BS = {bs_id:[] for bs_id in np.arange(len(self._node_list))}
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

        Parameters
        ----------
        binding_site_id : int of list of int, default=None

        Returns
        -------
        durations_BS : list, len(durations_BS)=len(binding_site_id)

        """
        self._check_calculation("Residue", self.collect_residue_contacts)
        self._check_calculation("Binding Site ID", self.compute_binding_nodes, print_data=False)
        selected_bs_id = np.atleast_1d(binding_site_id) if binding_site_id is not None \
            else np.arange(len(self._node_list), dtype=int)
        for bs_id in selected_bs_id:
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
        """Calculate the percentage of frames in which lipid contacts are formed for binding sites.

        Parameters
        ----------
        binding_site_id : int or list of int, default=None

        Returns
        -------
        occupancy_BS : list, len(occupancy_BS)=len(binding_site_id)

        """
        self._check_calculation("Residue", self.collect_residue_contacts)
        self._check_calculation("Binding Site ID", self.compute_binding_nodes, print_data=False)
        selected_bs_id = np.atleast_1d(binding_site_id) if binding_site_id is not None \
            else np.arange(len(self._node_list), dtype=int)
        for bs_id in selected_bs_id:
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
        """Calculate the average number of surrounding lipids for binding sites.

        Parameters
        ----------
        binding_site_id : int or list of int, default=None

        Returns
        -------
        lipidcount_BS : list, len(lipidcount_BS)=len(binding_site_id)

        """
        self._check_calculation("Residue", self.collect_residue_contacts)
        self._check_calculation("Binding Site ID", self.compute_binding_nodes, print_data=False)
        selected_bs_id = np.atleast_1d(binding_site_id) if binding_site_id is not None \
            else np.arange(len(self._node_list), dtype=int)
        for bs_id in selected_bs_id:
            nodes = self._node_list[bs_id]
            lipidcount_BS = []
            for traj_idx in np.arange(len(self._trajfile_list), dtype=int):
                for protein_idx in np.arange(self._nprot, dtype=int):
                    list_to_take = traj_idx * self._nprot + protein_idx
                    n_frames = len(self._contact_residues_low[nodes[0]][list_to_take])
                    contact_BS_low = [np.unique(np.concatenate(
                        [self._contact_residues_low[node][list_to_take][frame_idx] for node in nodes]))
                        for frame_idx in np.arange(n_frames)]
                    lipidcount_BS.append(cal_occupancy(contact_BS_low))
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
                          save_dir=None, plot_data=True, fig_close=True):
        """Calculate interactions koff and residence time for binding sites.

        Parameters
        ----------
        binding_site_id : int or array_like or None, default=None
        nbootstrap : int, default=10
        initial_guess : array_like, default=None
        save_dir : str, default=None
        plot_data : bool, default=True
        fig_close : bool, default=True

        Returns
        ---------
        koffs : list
        restimes : list

        """
        self._check_calculation("Residue", self.collect_residue_contacts)
        self._check_calculation("Binding Site ID", self.compute_binding_nodes, print_data=False)
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
        # calculate koff for binding sites
        for bs_id in tqdm(selected_bs_id, desc="CALCULATE KOFF FOR BINDING SITES", total=len(selected_bs_id)):
            durations = np.concatenate(self._duration_BS[bs_id])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                koff_BS, restime_BS, properties_BS = cal_koff(durations, t_total, timestep, nbootstrap, initial_guess)
            self._koff_BS[bs_id] = koff_BS
            self._koff_BS_boot[bs_id] = np.mean(properties_BS["ks_boot_set"], axis=0)[0]
            self._res_time_BS[bs_id] = restime_BS
            self._r_squared_BS[bs_id] = properties_BS["r_squared"]
            self._r_squared_BS_boot[bs_id] = np.mean(properties_BS["r_squared_boot_set"])
            if plot_data:
                # plot BS koff and BS residence time
                text = self._format_koff_text(properties_BS)
                plot_koff(durations, properties_BS["delta_t_list"], properties_BS["survival_rates"],
                          properties_BS["n_fitted"], survival_rates_bootstraps=properties_BS["survival_rates_boot_set"],
                          fig_fn=os.path.join(BS_dir, f"BS_id{bs_id}.pdf"), title=f"Binding Site {bs_id}",
                          timeunit=self._timeunit, t_total=t_total, text=text, fig_close=fig_close)
        # update dataset
        for data, column_name in zip(
                [self._koff_BS, self._koff_BS_boot, self._res_time_BS, self._r_squared, self._r_squared_BS_boot],
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

    def analyze_bound_poses(self, binding_site_id=None, n_top_poses=3, pose_format="gro", score_weights=None,
                            kde_bw=0.15, pca_component=0.95, plot_rmsd=True, save_dir=None,
                            n_clusters="auto", eps=None, min_samples=None, metric="euclidean", fig_close=False):
        """Analyze bound poses for binding sites.

        This function can find representative bound poses, cluster the bound poses and calculate pose RMSD for
        binding sites.

        Parameters
        ----------
        binding_site_id : int or list_like or None, default=None
        n_top_poses : int, default=5
        pose_format : str, default="pdb"
        score_weights : dict or None, default=None
        kde_bw : scalar, default=0.15
        pca_component : int, float or ‘mle’, default=0.95
            Set the `n_components` value in `sklearn.decomposition.PCA`
        plot_rmsd : bool, default=True
        n_clusters : int or 'auto'
            If `n_clusters == 'auto'`, the number of clusters will be guessed based on density, whereas n_clusters=N
            will use KMeans to cluster the poses.
        save_dir : str or None, default=None

        Returns
        -------
        pose_pool : dict
            Coordinates of all bound poses in stored in a python dictionary {binding_site_id: pose coordinates}
        rmsd_data : pandas.DataFrame
            Bound poses RMSDs are stored by columns with column name of binding site id.

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
        pose_pool = collect_bound_poses(selected_bs_map, self._contact_residues_low, self._trajfile_list,
                                        self._topfile_list, self._lipid, stride=self._stride, nprot=self._nprot)
        # analyize bound poses
        RMSD_set = {}
        for bs_id in tqdm(selected_bs_id, desc="ANALYZE BOUND POSES"):
            # lipid_dist_per_atom shape: [n_lipid_atoms, n_bound_poses, n_BS_residues]
            if len(pose_pool[bs_id]) == 0:
                print(f"Binding Site {bs_id} is bogus! Possibly due to insufficient sampling.")
                continue
            lipid_dist_per_atom, pose_traj = vectorize_poses(pose_pool[bs_id], self._node_list[bs_id],
                                                             self._protein_ref, self._lipid_ref)
            if n_top_poses > 0:
                pose_dir_rank = check_dir(pose_dir, "BSid{}_rank".format(bs_id), print_info=False)
                atom_weights = {atom_idx: 1 for atom_idx in np.arange(self._lipid_ref.n_atoms)}
                if score_weights is not None:
                    translate = {atom_idx: score_weights[self._lipid_ref.atom(atom_idx)]
                                 for atom_idx in np.arange(self._lipid_ref.n_atoms)
                                 if self._lipid_ref.atom(atom_idx) in score_weights.keys()}
                    atom_weights.update(translate)
                scores = calculate_scores(lipid_dist_per_atom, kde_bw=kde_bw, pca_component=pca_component,
                                          score_weights=atom_weights)
                num_of_poses = min(n_top_poses, pose_traj.n_frames)
                pose_indices = np.argsort(scores)[::-1][:num_of_poses]
                write_bound_poses(pose_traj, pose_indices, pose_dir_rank, pose_prefix="BSid{}_top".format(bs_id),
                                  pose_format=pose_format)
            lipid_dist_per_pose = np.array([lipid_dist_per_atom[:, pose_id, :].ravel()
                                            for pose_id in np.arange(lipid_dist_per_atom.shape[1])])
            # cluster poses
            if n_clusters == 'auto':
                pose_dir_clustered = check_dir(pose_dir, "BSid{}_clusters".format(bs_id), print_info=False)
                transformed_data = PCA(n_components=pca_component).fit_transform(lipid_dist_per_pose)
                cluster_labels = cluster_DBSCAN(transformed_data, eps=eps, min_samples=min_samples,
                                                metric=metric)
                cluster_id_set = [label for label in np.unique(cluster_labels) if label != -1]
                selected_pose_id = [np.random.choice(np.where(cluster_labels == cluster_id)[0], 1)[0]
                                    for cluster_id in cluster_id_set]
                write_bound_poses(pose_traj, selected_pose_id, pose_dir_clustered,
                                  pose_prefix="BSid{}_cluster".format(bs_id), pose_format=pose_format)
            elif n_clusters > 0:
                pose_dir_clusters = check_dir(pose_dir, "BSid{}_clusters".format(bs_id), print_info=False)
                transformed_data = PCA(n_components=pca_component).fit_transform(lipid_dist_per_pose)
                cluster_labels = cluster_KMeans(transformed_data, n_clusters=n_clusters)
                cluster_id_set = np.unique(cluster_labels)
                selected_pose_id = [np.random.choice(np.where(cluster_labels == cluster_id)[0], 1)[0]
                                    for cluster_id in cluster_id_set]
                write_bound_poses(pose_traj, selected_pose_id, pose_dir_clusters,
                                  pose_prefix="BSid{}_cluster".format(bs_id), pose_format=pose_format)
            # calculate pose rmsd
            dist_mean = np.mean(lipid_dist_per_pose, axis=0)
            RMSD_set["Binding Site {}".format(bs_id)] = [rmsd(lipid_dist_per_pose[pose_id], dist_mean)
                                                         for pose_id in np.arange(len(lipid_dist_per_pose))]
            pose_rmsd_per_residue[self._node_list[bs_id]] = np.mean(RMSD_set["Binding Site {}".format(bs_id)])
        # update dataset
        self.dataset["Binding Site Pose RMSD"] = pose_rmsd_per_residue
        pose_rmsd_data = pd.DataFrame(
            dict([(bs_label, pd.Series(rmsd_set)) for bs_label, rmsd_set in RMSD_set.items()]))
        # plot RMSD
        if plot_rmsd and n_top_poses > 0:
            plot_binding_site_data(pose_rmsd_data, os.path.join(pose_dir, "Pose_RMSD_violinplot.pdf"),
                                   title="{}".format(self._lipid), ylabel="RMSD (nm)", fig_close=fig_close)
        return pose_pool, pose_rmsd_data

    def compute_surface_area(self, binding_site_id=None, radii=None, plot_data=True, save_dir=None, fig_close=False):
        """Calculate binding site surface areas.

        Parameters
        -----------
        binding_site_id : int or array_like or None, default=None
        radii : dict or None, default=None
        plot_data : bool, default=True
        save_dir : str or None, default=None

        """
        MARTINI_CG_radii = {"BB": 0.26, "SC1": 0.23, "SC2": 0.23, "SC3": 0.23}

        self._check_calculation("Binding Site ID", self.compute_binding_nodes, print_data=False)

        if "Binding Site Surface Area" in self.dataset.columns:
            # keep existing data
            surface_area_per_residue = np.array(self.dataset["Binding Site Surface Area"].tolist())
        else:
            surface_area_per_residue = np.zeros(self._nresi_per_protein)
        if plot_data:
            selected_bs_id = np.atleast_1d(np.array(binding_site_id, dtype=int)) if binding_site_id is not None \
                else np.arange(len(self._node_list), dtype=int)
        if radii is None:
            radii_book = MARTINI_CG_radii
        else:
            radii_book = {**MARTINI_CG_radii, **radii}

        # calculate binding site surface area
        selected_bs_id_map = {bs_id: self._node_list[bs_id] for bs_id in selected_bs_id}
        surface_area_data = calculate_site_surface_area(selected_bs_id_map, radii_book, self._trajfile_list,
                                                        self._topfile_list, self._nprot, self._timeunit, self._stride,
                                                        dt_traj=self._dt_traj)
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
                surface_area_dir = check_dir(self._save_dir)
            plot_surface_area(surface_area_data,
                              os.path.join(surface_area_dir, "Surface_Area_{}_timeseries.pdf".format(self._lipid)),
                              timeunit=self._timeunit, fig_close=fig_close)
            selected_columns = [column for column in surface_area_data.columns if column != "Time"]
            surface_data_noTimeSeries = surface_area_data[selected_columns]
            plot_binding_site_data(surface_data_noTimeSeries,
                                   os.path.join(surface_area_dir, "Surface_Area_{}_violinplot.pdf".format(self._lipid)),
                                   title="{}".format(self._lipid), ylabel=r"Surface Area (nm$^2$)",
                                   fig_close=fig_close)
        return surface_area_data

    #################################
    #    save and plot
    #################################
    def save_data(self, item, save_dir=None):
        """Assisting function for saving data.

        Parameters
        ----------
        item : {"Dataset", "Duration", "Occupancy", "Lipid Count", "CorrCoef", "Duration BS",
                "Occupancy BS"}
        save_dir : str, optional, default=None

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
            self.dataset.to_csv(os.path.join(data_dir, "dataset.csv"), header=True, index=False)

        return

    def save_coordinate(self, item, save_dir=None, fn_coord=None):
        """Save protein coordinates in PDB format with interaction data in the B factor column.

        parameters
        -----------
        item : {"Residence Time", "Duration", "Occupancy", "Lipid Count"}
        save_dir : str or None, default=None

        """
        coord_dir = check_dir(save_dir, "Coordinate_{}".format(self._lipid)) if save_dir is not None \
            else check_dir(self._save_dir, "Coordinate_{}".format(self._lipid))
        if fn_coord is None:
            fn_coord = "Coordinate_{}_{}.pdb".format(self._lipid, "_".join(item.split()))
        data = self.dataset[item].tolist()
        write_PDB(self._protein_ref, data, os.path.join(coord_dir, fn_coord), resi_offset=self._resi_offset)
        return

    def save_pymol_script(self, pdb_file, save_dir=None):
        """save a pymol script that maps interactions onto protein structure in PyMol.

        Parameters
        ----------
        pdb_file : str
        save_dir : str, optional, default=None

        """
        script_dir = check_dir(save_dir) if save_dir is not None else check_dir(self._save_dir)
        data_fname = os.path.join(script_dir, "Dataset_{}.csv".format(self._lipid))
        if not os.path.isfile(data_fname):
            self.dataset.to_csv(data_fname, index=False, header=True)
        write_pymol_script(os.path.join(script_dir, "show_binding_site_info.py"), pdb_file, data_fname,
                           self._lipid, len(self._node_list))
        return

    def plot(self, item, save_dir=None, gap=200, fig_close=False):
        """Assisting function for plotting interaction data.

        Plot interactions per residue or plot interaction correlation matrix.

        Parameters
        ----------
        item : {"Residence Time", "Duration", "Occupancy", "Lipid Count", "CorrCoef"}
        save_dir : str, default=None
        gap : int, default=200

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
            fig_fn = os.path.join(figure_dir, "{}.pdf".format("_".join(item.split())))
            residue_index = np.array([int(re.findall("^[0-9]+", residue)[0]) for residue in self._residue_list])
            plot_residue_data(residue_index, data, gap=gap, ylabel=ylabel, fn=fig_fn, title=title,
                              fig_close=fig_close)

        if item == "CorrCoef":
            residue_index = np.array([int(re.findall("^[0-9]+", residue)[0]) for residue in self._residue_list])
            plot_corrcoef(self.interaction_corrcoef, residue_index, fn=os.path.join(figure_dir, "CorrCoef.pdf"),
                          title="{} Correlation Coeffient".format(self._lipid), fig_close=fig_close)

        return

    def plot_logo(self, item, save_dir=None, gap=2000, letter_map=None, color_scheme="chemistry", fig_close=False):
        """Plot interactions using logomaker.

        Parameters
        ----------
        item : {"Residence Time", "Duration", "Occupancy", "Lipid Count"}
        save_dir : str or None, optional, default=None
        gap : int, optional, default=2000
        letter_map : dict or None, optional, default=None
        color_scheme : str, optional, default="chemistry"

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
            fig_fn = os.path.join(figure_dir, "{}_logo.pdf".format("_".join(item.split())))
            plot_residue_data_logos(residue_index, resname_set, data, ylabel=ylabel,
                                    fn=fig_fn, title=title, letter_map=letter_map, color_scheme=color_scheme,
                                    fig_close=fig_close)
        else:
            print("Invalid input for item!")

        return

    def write_site_info(self, sort_residue="Residence Time", save_dir=None, fn=None):
        """Write a report on binding site with lipid interaction information.

        The report contains information of the binding site residence time, durations, occupancy,
        lipid count, surface area and some stats on amino acids composition. A list of lipid interactions with
        comprising residues will follow after the binding site information, in a sorted order provided by
        `sort_residue`.

        Parameters
        ----------
        sort_residue : {"Residence Time", "Duration", "Occupancy", "Lipid Count"}, default="Residence Time"
        save_dir : str, default=None
        fn : str, default=None

        """
        if len(self._node_list) == 0:
            print("No binding site was detected!!")
        else:
            BS_dir = check_dir(save_dir) if save_dir is not None \
                else check_dir(self._save_dir)
            if fn is None:
                fn = "BindingSites_Info_{}.txt".format(self._lipid)
            self._check_calculation("Binding Site ID", self.compute_binding_nodes, print_data=False)
            self._check_calculation("Binding Site Koff", self.compute_site_koff, print_data=False)
            with open(os.path.join(BS_dir, fn), "a") as f:
                f.write(f"## Network modularity {self._network_modularity:5.3f}")
                f.write("\n")
                for bs_id, nodes in enumerate(self._node_list):
                    text = self._format_BS_print_info(bs_id, nodes, sort_residue)
                    f.write(text)
        return

    def show_stats_per_traj(self, write_log=True, print_log=True, fn_log=None):
        """Show stats of interactions per trajectory.

        Parameters
        ----------
        write_log : bool, default=True
        print_log : bool, default=True
        fn_log : str, default=None

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

    def _format_koff_text(self, properties):
        """Format text for koff plot. """
        tu = "ns" if self._timeunit == "ns" else r"$\mu$s"
        text = "{:18s} = {:.3f} {:2s}$^{{-1}} $\n".format("$k_{{off1}}$", properties["ks"][0], tu)
        text += "{:18s} = {:.3f} {:2s}$^{{-1}} $\n".format("$k_{{off2}}$", properties["ks"][1], tu)
        text += "{:14s} = {:.4f}\n".format("$R^2$", properties["r_squared"])
        ks_boot_avg = np.mean(properties["ks_boot_set"], axis=0)
        cv_avg = 100 * np.std(properties["ks_boot_set"], axis=0) / np.mean(properties["ks_boot_set"], axis=0)
        text += "{:18s} = {:.3f} {:2s}$^{{-1}}$ ({:3.1f}%)\n".format("$k_{{off1, boot}}$", ks_boot_avg[0],
                                                                     tu, cv_avg[0])
        text += "{:18s} = {:.3f} {:2s}$^{{-1}}$ ({:3.1f}%)\n".format("$k_{{off2, boot}}$", ks_boot_avg[1],
                                                                     tu, cv_avg[1])
        text += "{:14s} = {:.4f}\n".format("$R^2$$_{{boot}}$", np.mean(properties["r_squared_boot_set"]))
        text += "{:18s} = {:.3f} {:2s}".format("$Res. Time$", properties["res_time"], tu)
        return text

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
