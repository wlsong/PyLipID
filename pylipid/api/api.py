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
from ..funcs import cal_contact_residues
from ..funcs import cal_interaction_frequency, Duration, cal_koff
from ..funcs import get_node_list
from ..funcs import collect_bound_poses, vectorize_poses, calculate_scores, write_bound_poses
from ..funcs import cluster_DBSCAN, cluster_KMeans
from ..funcs import calculate_site_surface_area
from ..plots import plot_koff
from ..plots import plot_surface_area, plot_binding_site_data
from ..plots import plot_residue_data, plot_corrcoef, plot_residue_data_logos
from ..utils import check_dir, write_PDB, write_pymol_script, sparse_corrcoef, rmsd, get_traj_info


class LipidInteraction:
    def __init__(self, trajfile_list, topfile_list=None, stride=1, dt_traj=None, cutoffs=[0.55, 1.0],
                 lipid="POPC", lipid_atoms=None, nprot=1, resi_offset=0, save_dir=None, timeunit="us"):
        """The main class to handle the calculation.

        *api* reads trajectory information via `mdtraj.load()`, and calculate the interactions
        of specified lipids with the proteins or specified regions of the proteins in the systems.

        Parameters
        ----------
        trajfile_list : str or list of str
            A list of trajectory filenames. `mdtraj.load()` will load the trajectories one by one to read information.
        topfile_list : str or list of str, optional, default=None
            A list of topology filenames of the trajectories in *trajfile_list*. For those trajectory formats that
            do not include topology information, `mdtraj.load()` requires a structure to read in

        """
        self._trajfile_list = np.atleast_1d(trajfile_list)
        if len(np.atleast_1d(topfile_list)) == len(self._trajfile_list):
            self._topfile_list = np.atleast_1d(topfile_list)
        elif len(self._trajfile_list) > 1 and len(np.atleast_1d(topfile_list)) == 1:
            self._topfile_list = [topfile_list for dummy in self._trajfile_list]
        else:
            raise ValueError(
                "topfile_list should either have the same length as trajfile_list or have one valid file name")

        self._dt_traj = dt_traj
        self._cutoffs = np.sort(np.array(cutoffs, dtype=float))
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
        """Obtain a list of the protein residues with their residue names and index."""
        return self._residue_list

    @property
    def node_list(self):
        return self._node_list

    @property
    def lipid(self):
        return self._lipid

    @property
    def lipid_atoms(self):
        return self._lipid_atoms

    @property
    def cutoffs(self):
        return self._cutoffs

    @property
    def nprot(self):
        return self._nprot

    @property
    def stride(self):
        return self._stride

    @property
    def trajfile_list(self):
        return self._trajfile_list

    @property
    def topfile_list(self):
        return self._topfile_list

    @property
    def dt_traj(self):
        return self._dt_traj

    @property
    def resi_offset(self):
        return self._resi_offset

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def timeunit(self):
        return self._timeunit

    def koff(self, residue_id=None, residue_name=None):
        """Return koff of the specified residue"""
        if residue_id is not None and residue_name is not None:
            assert self.dataset[self.dataset["Residue ID"] == residue_id]["Residue"] == residue_name, \
                "residue_id and residue_name are pointing to different residues!"
            return self._koff[residue_id]
        elif residue_id is not None:
            return self._koff[residue_id]
        elif residue_name is not None:
            return self._koff[self._residue_map[residue_name]]

    def res_time(self, residue_id=None, residue_name=None):
        """Return residence time of the specified residue"""
        if residue_id is not None and residue_name is not None:
            assert self.dataset[self.dataset["Residue ID"] == residue_id]["Residue"] == residue_name, \
                "residue_id and residue_name are pointing to different residues!"
            return self._res_time[residue_id]
        elif residue_id is not None:
            return self._res_time[residue_id]
        elif residue_name is not None:
            return self._res_time[self._residue_map[residue_name]]

    def koff_bs(self, bs_id):
        """Return koff of the specified binding site. """
        return self._koff_BS[bs_id]

    def res_time_bs(self, bs_id):
        """Return residence time of the specified binding site. """
        return self._res_time_BS[bs_id]

    def residue(self, residue_id=None, residue_name=None, print_data=True):
        """Obtain the calculated information for a residue

        Use either residue_id or residue_name to obtain the information"""
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
        """Obtain the calculated information for a binding site. """
        df = self.dataset[self.dataset["Binding Site ID"] == binding_site_id]
        if print_data:
            text = self._format_BS_print_info(binding_site_id, self._node_list[binding_site_id], sort_residue)
            print(text)
        return df

    ########################################
    #     interaction calculation
    ########################################
    def collect_residue_contacts(self, write_log=True, print_log=True, fn_log=None):
        """Calculate durations, occupancy and lipid count from trajectories.

        """
        # initialise variables for interaction of residues
        self.durations = defaultdict(list)
        self.occupancy = defaultdict(list)
        self.lipid_count = defaultdict(list)
        self.contact_residues_high = defaultdict(list)
        self.contact_residues_low = defaultdict(list)
        self._residue_list = None
        self._protein_residue_id = []
        self.interaction_corrcoef = None
        self.dataset = None
        self._protein_ref = None
        self._lipid_ref = None
        self._T_total = []
        self._timesteps = []
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
                self._koff = np.zeros(self._nresi_per_protein)
                self._r_squared = np.zeros(self._nresi_per_protein)
                self._res_time = np.zeros(self._nresi_per_protein)
                self._koff_b = np.zeros(self._nresi_per_protein)
                self._residue_map = {resn: resi for resn, resi in zip(self._protein_residue_id, self._residue_list)}
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
                    contact_low, frame_id_set_low, lipid_id_set_low = cal_contact_residues(dist_matrix,
                                                                                           self._cutoffs[0])
                    contact_high, _, _ = cal_contact_residues(dist_matrix, self._cutoffs[1])
                    self.contact_residues_high[residue_id].append(contact_high)
                    self.contact_residues_low[residue_id].append(contact_low)
                    self.durations[residue_id].append(
                        Duration(contact_low, contact_high, timestep).cal_durations())
                    with warnings.catch_warnings():  # mute the warnings for empty list.
                        warnings.simplefilter("ignore")
                        occupancy, lipid_count = cal_interaction_frequency(contact_low)
                    self.occupancy[residue_id].append(occupancy)
                    self.lipid_count[residue_id].append(lipid_count)
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
        # wrap up trajectory information
        self._wrapup_in_logfile(write_log=write_log, print_log=print_log, fn_log=fn_log)

        self.dataset = pd.DataFrame({"Residue": [residue for residue in self._residue_list],
                                     "Residue ID": self._protein_residue_id,
                                     "Occupancy": [np.mean(self.occupancy[residue_id])
                                                   for residue_id in self._protein_residue_id],
                                     "Occupancy std": [np.std(self.occupancy[residue_id])
                                                       for residue_id in self._protein_residue_id],
                                     "Duration": [np.mean(np.concatenate(self.durations[residue_id]))
                                                  for residue_id in self._protein_residue_id],
                                     "Duration std": [np.std(np.concatenate(self.durations[residue_id]))
                                                      for residue_id in self._protein_residue_id],
                                     "Lipid Count": [np.mean(self.lipid_count[residue_id])
                                                     for residue_id in self._protein_residue_id],
                                     "Lipid Count std": [np.std(self.lipid_count[residue_id])
                                                         for residue_id in self._protein_residue_id]})

        return

    def compute_residue_koff(self, residue_id=None, nbootstrap=10, initial_guess=[1., 1., 1., 1.],
                             save_dir=None, print_data=True, plot_data=True, fig_close=True):
        """Calculate interaction koff for residues.

        Parameters
        ----------
        residue_id : int or array_like or None, optional, default=None
        nbootstrap : int, optional, default=10
        initial_guess : array_like, default=None
        save_dir : str, optional, default=None
        print_data : bool, optional, default=True
        plot_data : bool, optional, default=True
        fig_close : bool, optional, default=True

        Returns
        ---------
        koff : scalar
        restime : scalar

        """
        if plot_data:
            koff_dir = check_dir(save_dir, "Reisidue_koffs_{}".format(self._lipid)) if save_dir is not None \
                else check_dir(self._save_dir, "Residue_koffs_{}".format(self._lipid))
        self._check_BS_calculation("Residue", self.collect_residue_contacts, write_log=True, print_log=False)
        if len(set(self._residue_list)) != len(self._residue_list):
            residue_name_set = ["{}_ResidueID{}".format(residue, residue_id) for residue, residue_id in
                                zip(self._residue_list, self._protein_residue_id)]
        else:
            residue_name_set = self._residue_list
        if residue_id is not None:
            selected_residue_id = np.atleast_1d(residue_id)
        else:
            selected_residue_id = self._protein_residue_id
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

        for residue_id in selected_residue_id:
            if residue_id not in self.durations.keys():
                raise ValueError("Residue ID {} is not valid.".format(residue_id))
            durations = np.concatenate(self.durations[residue_id])
            residue = residue_name_set[residue_id]
            if np.sum(durations) == 0:
                self._koff[residue_id] = 0
                self._res_time[residue_id] = 0
                self._r_squared[residue_id] = 0
                self._koff_b[residue_id] = 0
                if print_data:
                    print("No interaction was detected for {}!".format(residue))
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    koff, restime, properties = cal_koff(durations, t_total, timestep, nbootstrap, initial_guess)
                self._koff[residue_id] = koff
                self._res_time[residue_id] = restime
                self._r_squared[residue_id] = properties["r_squared"]
                self._koff_b[residue_id] = np.mean(properties["ks_boot_set"], axis=0)[0]
                if plot_data:
                    text = self._format_koff_text(properties)
                    plot_koff(durations, properties["delta_t_list"], properties["survival_rates"],
                              properties["n_fitted"], survival_rates_bootstraps=properties["survival_rates_boot_set"],
                              fig_fn=os.path.join(koff_dir, "{}.pdf".format(residue)), title=residue,
                              timeunit=self._timeunit, t_total=t_total, text=text, fig_close=fig_close)
                if print_data:
                    print("{:15s}: {}".format("Residue", residue))
                    print("{:15s}: {}".format("Residue ID", residue_id))
                    print("{:15s}: {:.3f} {:2s}^(-1)".format("Koff", koff, self._timeunit))
                    print("{:15s}: {:.3f} {:2s}".format("Residence Time", restime, self._timeunit))
                    print("{:15s}: {:.3f} {:2s}".format("Duration", self.dataset.iloc[residue_id]["Duration"], self._timeunit))
                    print("{:15s}: {:.3f} %".format("Occupancy", self.dataset.iloc[residue_id]["Occupancy"]))
                    print("{:15s}: {:.3f}".format("Lipid Count", self.dataset.iloc[residue_id]["Lipid Count"]))
        # update dataset
        self.dataset["Koff"] = self._koff
        self.dataset["Residence Time"] = self._res_time
        self.dataset["R Squared"] = self._r_squared
        self.dataset["Koff Bootstrap avg"] = self._koff_b

        if len(selected_residue_id) == 1:
            return self._koff[selected_residue_id[0]], self._res_time[selected_residue_id[0]]
        else:
            return [self._koff[residue_id] for residue_id in selected_residue_id], \
                   [self._res_time[residue_id] for residue_id in selected_residue_id]

    def compute_binding_nodes(self, threshold=4, print_data=True):
        """Calculate binding sites.

        Parameters
        ----------
        threshold : int, optional, default=4
        save_dir : str or None, optional, default=None
        print : bool, optional, default=True

        Returns
        -------
        node_list: list of lists

        """
        corrcoef_raw = np.nan_to_num(self.interaction_corrcoef)
        corrcoef = np.copy(corrcoef_raw)
        node_list = get_node_list(corrcoef, threshold=threshold)
        self._node_list = node_list
        if len(self._node_list) == 0:
            print("No binding site detected!!")
        else:
            residue_BS_identifiers = np.ones(self._nresi_per_protein, dtype=int) * -1
            for bs_id, nodes in enumerate(self._node_list):
                residue_BS_identifiers[nodes] = bs_id
            # update dataset
            self.dataset["Binding Site ID"] = residue_BS_identifiers
            # initialise variable for binding site interactions
            self._durations_BS = [[] for dummy in self._node_list]
            self._occupancy_BS = [[] for dummy in self._node_list]
            self._lipid_count_BS = [[] for dummy in self._node_list]
            self._koff_BS = np.zeros(len(self._node_list))
            self._res_time_BS = np.zeros(len(self._node_list))
            self._r_squared_BS = np.zeros(len(self._node_list))
            if print_data:
                for bs_id, nodes in enumerate(self._node_list):
                    print("#" * 25)
                    print(f"Binding Site ID: {bs_id}")
                    print("{:>10s} -- {:<12s}".format("Residue", "Residue ID"))
                    for node in nodes:
                        print("{:>10s} -- {:<12d}".format(self._residue_list[node], self._protein_residue_id[node]))
                    print("#" * 25)
        return node_list

    def compute_site_koff(self, binding_site_id=None, nbootstrap=10, initial_guess=[1., 1., 1., 1.],
                          save_dir=None, print_data=True, plot_data=True, sort_residue="Residence Time",
                          fig_close=True):
        """Calculate interactions for binding sites.

        Parameters
        ----------
        binding_site_id : int or array_like or None, optional, default=None
        nbootstrap : int, optional, default=10
        initial_guess : array_like, default=None
        save_dir : str, optional, default=None
        print_data : bool, optional, default=True
        sort_residue : str, optional, default="Residence Time"
        plot_data : bool, optional, default=True

        Returns
        ---------
        koffs : list
        restimes : list

        """
        self._check_BS_calculation("Binding Site ID", self.compute_binding_nodes, print_data=False)
        if print_data or plot_data:
            self._check_BS_calculation("Residence Time", self.compute_residue_koff, print_data=False, plot_data=False)
        if plot_data:
            BS_dir = check_dir(save_dir, "Binding_Sites_koffs_{}".format(self._lipid)) if save_dir is not None \
                else check_dir(self._save_dir, "Binding_Sites_koffs_{}".format(self._lipid))

        selected_bs_id = np.atleast_1d(binding_site_id) if binding_site_id is not None \
            else np.arange(len(self._node_list), dtype=int)
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

        if "Binding Site Koff" in self.dataset.columns:
            # keep existing data
            koff_BS_per_residue = np.array(self.dataset["Binding Site Koff"].tolist())
            res_time_BS_per_residue = np.array(self.dataset["Binding Site Residence Time"].tolist())
            durations_BS_per_residue = np.array(self.dataset["Binding Site Duration"].tolist())
            occupancy_BS_per_residue = np.array(self.dataset["Binding Site Occupancy"].tolist())
            lipid_count_BS_per_residue = np.array(self.dataset["Binding Site Lipid Count"].tolist())
            r_squared_BS_per_residue = np.array(self.dataset["Binding Site R Squared"].tolist())
        else:
            koff_BS_per_residue = np.zeros(self._nresi_per_protein)
            res_time_BS_per_residue = np.zeros(self._nresi_per_protein)
            durations_BS_per_residue = np.zeros(self._nresi_per_protein)
            occupancy_BS_per_residue = np.zeros(self._nresi_per_protein)
            lipid_count_BS_per_residue = np.zeros(self._nresi_per_protein)
            r_squared_BS_per_residue = np.zeros(self._nresi_per_protein)

        # calculate interactions for binding sites
        for bs_id in tqdm(selected_bs_id, desc="CALCULATE INTERACTIONS FOR BINDING SITES",
                          total=len(selected_bs_id)):
            nodes = self._node_list[bs_id]
            # calculate durations, occupancy and lipid count
            for traj_idx in np.arange(len(self._trajfile_list), dtype=int):
                for protein_idx in np.arange(self._nprot, dtype=int):
                    list_to_take = traj_idx * self._nprot + protein_idx
                    n_frames = len(self.contact_residues_low[nodes[0]][list_to_take])
                    contact_BS_low = [np.unique(np.concatenate(
                        [self.contact_residues_low[node][list_to_take][frame_idx] for node in nodes]))
                        for frame_idx in np.arange(n_frames)]
                    contact_BS_high = [np.unique(np.concatenate(
                        [self.contact_residues_high[node][list_to_take][frame_idx] for node in nodes]))
                        for frame_idx in np.arange(n_frames)]
                    self._durations_BS[bs_id].append(
                        Duration(contact_BS_low, contact_BS_high, timestep).cal_durations())
                    occupancy_BS, lipid_count_BS = cal_interaction_frequency(contact_BS_low)
                    self._occupancy_BS[bs_id].append(occupancy_BS)
                    self._lipid_count_BS[bs_id].append(lipid_count_BS)
            # calculate BS koff and BS residence time
            durations = np.concatenate(self._durations_BS[bs_id])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                koff_BS, restime_BS, properties_BS = cal_koff(durations, t_total, timestep, nbootstrap, initial_guess)
            self._koff_BS[bs_id] = koff_BS
            self._res_time_BS[bs_id] = restime_BS
            self._r_squared_BS[bs_id] = properties_BS["r_squared"]
            koff_BS_per_residue[nodes] = koff_BS
            res_time_BS_per_residue[nodes] = restime_BS
            r_squared_BS_per_residue[nodes] = properties_BS["r_squared"]
            durations_BS_per_residue[nodes] = np.concatenate(self._durations_BS[bs_id]).mean()
            occupancy_BS_per_residue[nodes] = np.mean(self._occupancy_BS[bs_id])
            lipid_count_BS_per_residue[nodes] = np.mean(self._lipid_count_BS[bs_id])
            if plot_data:
                # plot BS koff and BS residence time
                text = self._format_koff_text(properties_BS)
                plot_koff(durations, properties_BS["delta_t_list"], properties_BS["survival_rates"],
                          properties_BS["n_fitted"], survival_rates_bootstraps=properties_BS["survival_rates_boot_set"],
                          fig_fn=os.path.join(BS_dir, f"BS_id{bs_id}.pdf"), title=f"Binding Site {bs_id}",
                          timeunit=self._timeunit, t_total=t_total, text=text, fig_close=fig_close)
        # update dataset
        self.dataset["Binding Site Koff"] = koff_BS_per_residue
        self.dataset["Binding Site Residence Time"] = res_time_BS_per_residue
        self.dataset["Binding Site R Squared"] = r_squared_BS_per_residue
        self.dataset["Binding Site Duration"] = durations_BS_per_residue
        self.dataset["Binding Site Occupancy"] = occupancy_BS_per_residue
        self.dataset["Binding Site Lipid Count"] = lipid_count_BS_per_residue
        # print binding site info
        if print_data:
            for bs_id in selected_bs_id:
                nodes = self._node_list[bs_id]
                text = self._format_BS_print_info(bs_id, nodes, sort_residue)
                print(text)

        if len(selected_bs_id) == 1:
            return self._koff_BS[bs_id], self._res_time_BS[bs_id]
        else:
            return [self._koff_BS[bs_id] for bs_id in selected_bs_id], \
                   [self._res_time_BS[bs_id] for bs_id in selected_bs_id]

    def analyze_bound_poses(self, binding_site_id=None, n_top_poses=3, pose_format="gro", score_weights=None,
                            kde_bw=0.15, pca_component=0.95, plot_rmsd=True, save_dir=None,
                            n_clusters="auto", eps=None, min_samples=None, metric="euclidean", fig_close=False):
        """Analyze binding poses.

        Parameters
        ----------
        binding_site_id : int or list_like or None, optional, default=None
        n_top_poses : int, optional, default=5
        pose_format : str, optional, default="gro"
        score_weights : dict or None, optional, default=None
        kde_bw : scalar, optional, default=0.15
        pca_component : int, float or ‘mle’, default=0.95
            Set the `n_components` value in `sklearn.decomposition.PCA`
        plot_rmsd : bool, default=True
        n_clusters : int or 'auto'
            If `n_clusters == 'auto'`,
        save_dir : str or None, optinal, default=None

        """
        self._check_BS_calculation("Binding Site ID", self.compute_binding_nodes, print_data=False)

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
        pose_pool = collect_bound_poses(selected_bs_map, self.contact_residues_low, self._trajfile_list,
                                        self._topfile_list, self._lipid, stride=self._stride, nprot=self._nprot)
        # analyize bound poses
        RMSD_set = {}
        for bs_id in tqdm(selected_bs_id, desc="ANALYZE BOUND POSES"):
            # lipid_dist_per_atom shape: [n_lipid_atoms, n_bound_poses, n_BS_residues]
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
        binding_site_id : int or array_like or None, optional, default=None
        radii : dict or None, optional, default=None
        plot_data : bool, optional, default=True
        save_dir : str or None, optional, default=None

        """
        MARTINI_CG_radii = {"BB": 0.26, "SC1": 0.23, "SC2": 0.23, "SC3": 0.23}

        self._check_BS_calculation("Binding Site ID", self.compute_binding_nodes, print_data=False)

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

    def write_site_info(self, sort_residue="Residence Time", save_dir=None, fn_info=None):
        """Write the binding site information in a txt file. """
        BS_dir = check_dir(save_dir) if save_dir is not None \
            else check_dir(self._save_dir)
        if fn_info is None:
            fn_info = "BindingSites_Info_{}.txt".format(self._lipid)
        self._check_BS_calculation("Binding Site ID", self.compute_binding_nodes, print_data=False)
        self._check_BS_calculation("Binding Site Koff", self.compute_site_koff, print_data=False)
        with open(os.path.join(BS_dir, fn_info), "a") as f:
            for bs_id, nodes in enumerate(self._node_list):
                text = self._format_BS_print_info(bs_id, nodes, sort_residue)
                f.write(text)
        return

    #################################
    #    save and plots
    #################################
    def save_data(self, item, save_dir=None):
        """Assisting function for save data set.

        Parameters
        ----------
        item : {"Dataset", "Duration", "Occupancy", "Lipid Count", "CorrCoef", "Duration BS",
                "Occupancy BS"}
        save_dir : str, optional, default=None

        """
        data_dir = check_dir(save_dir, "Dataset_{}".format(self._lipid)) if save_dir is not None \
            else check_dir(self._save_dir, "Dataset_{}".format(self._lipid))

        if item.lower() == "duration":
            obj = self.durations
        elif item.lower() == "occupancy":
            obj = self.occupancy
        elif item.lower() == "lipid count":
            obj = self.lipid_count
        elif item.lower() == "corrcoef":
            obj = self.interaction_corrcoef
        elif item.lower() == "duration bs":
            obj = self._durations_BS
        elif item.lower() == "occupancy bs":
            obj = self._occupancy_BS
        if 'obj' in locals():
            with open(os.path.join(data_dir, "{}.pickle".format("_".join(item.split()))), "wb") as f:
                pickle.dump(obj, f, 2)

        if item.lower() == "dataset":
            self.dataset.to_csv(os.path.join(data_dir, "dataset.csv"))

        return

    def save_coordinate(self, item, save_dir=None, fn_coord=None):
        """Save lipid interactions in the b factor column of a PDB coordinate file

        parameters
        -----------
        item : {"Residence Time", "Duration", "Occupancy", "Lipid Count"}
        save_dir : str or None, optional, default=None

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
            self.dataset.to_csv(data_fname)
        write_pymol_script(os.path.join(script_dir, "show_binding_site_info.py"), pdb_file, data_fname,
                           self._lipid, len(self._node_list))
        return

    def plot(self, item, save_dir=None, gap=200, fig_close=False):
        """Plot interactions.

        Parameters
        ----------
        item : {"Residence Time", "Duration", "Occupancy", "Lipid Count", "CorrCoef"}
        save_dir : str, optional, default=None
        gap : int, optional, default=200

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

    ############################################
    #     assisting func
    ############################################
    def _wrapup_in_logfile(self, write_log=True, print_log=True, fn_log=None):
        """Assisting function for formatting the log file text."""
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
                    self.durations[residue_id][traj_idx * self._nprot:(traj_idx + 1) * self._nprot]).mean()
                 for residue_id in self._protein_residue_id])
            duration_arg_idx = np.argsort(durations)[::-1]
            occupancies = np.array(
                [np.mean(self.occupancy[residue_id][traj_idx * self._nprot:(traj_idx + 1) * self._nprot])
                 for residue_id in self._protein_residue_id])
            occupancy_arg_idx = np.argsort(occupancies)[::-1]
            lipid_counts = np.array(
                [np.mean(self.lipid_count[residue_id][traj_idx * self._nprot:(traj_idx + 1) * self._nprot])
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
        text += "{:30s} {:10.3f} {:5s}\n".format(" Binding Site Residence Time:",
                                                 self._res_time_BS[bs_id], self._timeunit)
        text += "{:30s} {:10.3f}  R squared: {:7.4f}\n".format(" Binding Site Koff:", self._koff_BS[bs_id],
                                                               self._r_squared_BS[bs_id])
        text += "{:30s} {:10.3f} {:5s}\n".format(" Binding Site Duration:",
                                                 np.concatenate(self._durations_BS[bs_id]).mean(), self._timeunit)
        text += "{:30s} {:10.3f} %\n".format(" Binding Site Occupancy:",
                                             np.mean(self._occupancy_BS[bs_id]))
        text += "{:30s} {:10.3f}\n".format(" Binding Site Lipid Count:",
                                           np.mean(self._lipid_count_BS[bs_id]))
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
            text += "{Residue:^9s}{Residue ID:^7d}{Residence Time:^16.3f}{Duration:^16.3f}" \
                    "{Occupancy:^15.3f}{Lipid Count:^12.3f}{Koff:^7.3f}{R Squared:^10.3f}\n".format(**info_dict)
        text += "\n"
        text += "\n"
        return text

    def _check_BS_calculation(self, item, calculation, **kwargs):
        """Check BS calculation. """
        if item not in self.dataset.columns:
            print("#" * 60)
            print(f"{item} has not been calculated. Start the calculation.")
            calculation(**kwargs)
            print("Finished the calculation of {}.".format(item))
            print("#" * 60)
