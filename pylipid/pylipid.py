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
import warnings
import os
import re
import mdtraj as md
import numpy as np
from scipy.sparse import coo_matrix
import pandas as pd
from tqdm import trange, tqdm
from .funcs import get_traj_info, cal_contact_residues, sparse_corrcoef
from .funcs import cal_interaction_frequency, Duration, cal_koff
from .funcs import get_node_list, collect_binding_poses, write_binding_poses, calculate_surface_area
from .plots import plot_koff
from .plots import plot_surface_area, plot_binding_site_data
from .plots import plot_residue_data, plot_corrcoef, plot_residue_data_logos
from .util import check_dir, write_PDB, write_pymol_script


class LipidInteraction:
    def __init__(self, trajfile_list, topfile_list=None, stride=1, dt_traj=None, cutoffs=[0.55, 1.0],
                 lipid="POPC", lipid_atoms=None, nprot=1, resi_offset=0, save_dir=None, timeunit="us"):
        """The main class to handle the calculation.

        *LipidInteraction* reads trajectory information via `mdtraj.load()`, and calculate the interactions
        of specified lipids with the proteins or specified regions of the proteins in the systems.

        Parameters
        ----------
        trajfile_list : str or list of str
            A list of trajectory filenames. `mdtraj.load()` will load the trajectories one by one to read information.
        topfile_list : str or list of str, optional, default=None
            A list of topology filenames of the trajectories in *trajfile_list*. For those trajectory formats that
            do not include topology information, `mdtraj.load()` requires a structure to read in

        """
        self.trajfile_list = np.atleast_1d(trajfile_list)
        if len(np.atleast_1d(topfile_list)) == 1:
            self.topfile_list = [topfile_list for dummy in self.trajfile_list]
        else:
            assert len(self.trajfile_list) == len(topfile_list), \
                "The list of coordinates should be in the same order and length of list of trajectories!"
            self.topfile_list = np.atleast_1d(topfile_list)

        self.save_dir = check_dir(save_dir)
        self.dt_traj = dt_traj
        self.cutoffs = np.sort(np.array(cutoffs, dtype=float))
        self.lipid = lipid
        self.lipid_atoms = lipid_atoms
        self.nprot = int(nprot)
        self.timeunit = timeunit
        self.stride = int(stride)
        self.resi_offset = resi_offset

        return

    #############################################
    #     attributes
    #############################################
    @property
    def residue_list(self):
        """Obtain a list of the protein residues with their residue names and index."""
        return self._residue_list

    def residue(self, residue_id=None, residue_name=None, print_output=True):
        """Obtain the calculated information for a residue

        Use either residue_id or residue_name to obtain the information"""
        if residue_id is not None and residue_name is not None:
            assert self.dataset[self.dataset["Residue ID"]==residue_id]["Residue"] == residue_name, \
            "residue_id and residue_name are pointing to different residues!"
            df = self.dataset[self.dataset["Residue ID"]==residue_id]
        elif residue_id is not None:
            df = self.dataset[self.dataset["Residue ID"] == residue_id]
        elif residue_name is not None:
            df = self.dataset[self.dataset["Residue"] == residue_name]
        if print_output:
            print(df)
        return df

    def binding_site(self, binding_site_id, print_output=True, print_sort="Residence Time"):
        """Obtain the calculated information for a binding site. """
        df = self.dataset[self.dataset["Binding Site ID"] == binding_site_id]
        if print_output:
            self._format_BS_print_info(binding_site_id, self.node_list[binding_site_id], print_sort)
        return df

    ########################################
    #     interaction calculation
    ########################################
    def cal_interactions(self, save_dir=None):
        """Calculate durations, occupancy and lipid count from trajectories.

        Parameters
        ----------

        Returns
        ----------


        """
        if save_dir is None:
            self.save_dir = check_dir(self.save_dir, "Interaction_{}".format(self.lipid))
        else:
            self.save_dir = check_dir(save_dir, "Interaction_{}".format(self.lipid))

        # initialise variables for interaction of residues
        self.durations = defaultdict(list)
        self.occupancy = defaultdict(list)
        self.lipid_count = defaultdict(list)
        self.contact_residues_high = defaultdict(list)
        self.contact_residues_low = defaultdict(list)
        self._residue_list = []
        self._protein_residue_id = []
        self.interaction_corrcoef = None
        self.dataset = None
        self._protein_ref = None
        self._lipid_ref = None
        self._T_total = []
        self._timesteps = []
        for data in [self.koff, self.r_squared, self.res_time, self.koff_b]:
            data = np.zeros(self.nresi_per_protein)
        # initialise data for interaction matrix
        col = []
        row = []
        data = []
        ncol_start = 0
        # calculate interactions from trajectories
        for traj_idx in trange(len(self.trajfile_list), desc="CALCULATE INTERACTIONS FROM TRAJECTORIES",
                               total=len(self.trajfile_list)):
            traj = md.load(self.trajfile_list[traj_idx], top=self.topfile_list[traj_idx], stride=self.stride)
            traj_info, self._protein_ref, self._lipid_ref = get_traj_info(traj, self.lipid,
                                                                          lipid_atoms=self.lipid_atoms,
                                                                          resi_offset=self.resi_offset,
                                                                          nprot=self.nprot,
                                                                          protein_ref=self._protein_ref,
                                                                          lipid_ref=self._lipid_ref)
            if self.dt_traj is None:
                timestep = traj.timestep / 1000000.0 if self.timeunit == "us" else traj.timestep / 1000.0
            else:
                timestep = float(self.dt_traj * self.stride)
            self._T_total.append((traj.n_frames - 1) * timestep)
            self._timesteps.append(timestep)
            if len(self._protein_residue_id) == 0:
                self._protein_residue_id = traj_info["protein_residue_id"]
                self._residue_list = traj_info["residue_list"]
                self.nresi_per_protein = len(self._residue_list)
            else:
                assert len(self._protein_residue_id) == traj_info["protein_residue_id"], \
                    "Trajectory {} contains {} residues whereas trajectory {} contains {} residues".format(
                        traj_idx, len(traj_info["protein_residue_id"]), traj_idx - 1, len(self._protein_residue_id))
                self._protein_residue_id = traj_info["protein_residue_id"]
            ncol_per_protein = traj_info["lipid_atom_index_set"] * traj.n_frames
            for protein_idx in np.arange(self.nprot):
                for residue_id, residue_atom_indices in enumerate(traj_info["protein_residue_id"][protein_idx]):
                    # calculate interaction per residue
                    dist_matrix = np.array([np.min(
                        md.compute_distances(traj, np.array(list(product(residue_atom_indices, lipid_atom_indices)))),
                        axis=1) for lipid_atom_indices in traj_info["lipid_atom_index_set"]])
                    contact_low, frame_id_set_low, lipid_id_set_low = cal_contact_residues(dist_matrix, self.cutoffs[0])
                    contact_high, _, _ = cal_contact_residues(dist_matrix, self.cutoffs[1])
                    self.contact_residues_high[residue_id].append(contact_high)
                    self.contact_residues_low[residue_id].append(contact_low)
                    self.durations[residue_id].append(
                        Duration(contact_low, contact_high, timestep).cal_durations())
                    occupancy, lipid_count = cal_interaction_frequency(contact_low)
                    self.occupancy[residue_id].append(occupancy)
                    self.lipid_count[residue_id].append(lipid_count)
                    # update coordinates for coo_matrix
                    col.append([ncol_start + ncol_per_protein * protein_idx + lipid_id * traj.n_frames + \
                                frame_id for frame_id, lipid_id in zip(frame_id_set_low, lipid_id_set_low)])
                    row.append([residue_id for dummy in np.arange(len(frame_id_set_low))])
                    data.append(dist_matrix[lipid_id_set_low, frame_id_set_low])
            ncol_start += ncol_per_protein * self.nprot
        # calculate correlation coefficient matrix
        row = np.concatenate(row)
        col = np.concatenate(col)
        data = np.concatenate(data)
        contact_info = coo_matrix((data, (row, col)), shape=(self.nresi_per_protein, ncol_start))
        self.interaction_corrcoef = sparse_corrcoef(contact_info)

        # wrap up trajectory information
        self._wrapup_in_logfile()
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

        return self.dataset

    def cal_residue_koff(self, residue_id=None, nbootstrap=10, initial_guess=[1., 1., 1., 1.],
                         save_dir=None, print_output=True, plot=True):
        """Calculate interaction koff for residues.

        Parameters
        ----------
        residue_id : int or array_like or None, optional, default=None
        nbootstrap : int, optional, default=10
        initial_guess : array_like, default=None
        save_dir : str, optional, default=None
        print_output : bool, optional, default=True
        plot : bool, optional, default=True

        Returns
        ---------
        koff : float
        restime : float


        """
        if save_dir is not None:
            koff_dir = check_dir(save_dir, "Koffs_{}".format(self.lipid))
        else:
            koff_dir = check_dir(self.save_dir, "Koffs_{}".format(self.lipid))

        if len(self._residue_list) == 0:
            raise ValueError("Interaction durations haven't been calculated yet, please calculate \
                              interactions via cal_interactions().")

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
                self.koff[residue_id] = 0
                self.res_time[residue_id] = 0
                self.r_squared[residue_id] = 0
                self.koff_b[residue_id] = 0
                if print_output:
                    print("No interaction was detected for {}!".format(residue))
            else:
                koff, restime, properties = cal_koff(durations, t_total, timestep, nbootstrap, initial_guess)
                self.koff[residue_id] = koff
                self.res_time[residue_id] = restime
                self.r_squared[residue_id] = properties["r_squared"]
                self.koff_b[residue_id] = np.mean(properties["ks_boot_set"], axis=0)[0]

                if plot:
                    text = self._format_koff_text(properties)
                    plot_koff(durations, properties["delta_t_list"], properties["survival_func"],
                              properties["n_fitted"], survival_rates_bootstraps=properties["survival_func_boot_set"],
                              fig_fn=os.path.join(koff_dir, "{}.pdf".format(residue)), title=residue,
                              timeunit=self.timeunit, t_total=t_total, text=text)
                if print_output:
                    print("{:15s}: {}".format("Residue", residue))
                    print("{:15s}: {}".format("Residue ID", residue_id))
                    print("{:15s}: {:.3f} {:2s}^(-1)".format("Koff", koff, self.timeunit))
                    print("{:15s}: {:.3f} {:2s}".format("Residence Time", restime, self.timeunit))
                    print("{:15s}: {:.3f} {:2s}".format("Duration", self.dataset.iloc[residue_id]["Duration"]))
                    print("{:15s}: {:.3f} %".format("Occupancy", self.dataset.iloc[residue_id]["Occupancy"]))
                    print("{:15s}: {:.3f}".format("Lipid Count", self.dataset.iloc[residue_id]["Lipid Count"]))

        # update dataset
        self.dataset["Koff"] = self.koff
        self.dataset["Residence Time"] = self.res_time
        self.dataset["R Squared"] = self.r_squared
        self.dataset["Koff Bootstrap avg"] = self.koff_b

        return koff, restime

    def cal_binding_site(self, size=4, print_output=True):
        """Calculate binding sites.

        Parameters
        ----------
        size : int, optional, default=4
        save_dir : str or None, optional, default=None
        print : bool, optional, default=True

        Returns
        -------
        node_list: list of lists

        """
        corrcoef_raw = np.nan_to_num(self.interaction_corr)
        corrcoef = np.copy(corrcoef_raw)
        self.node_list = get_node_list(corrcoef, size=size)
        if len(self.node_list) == 0:
            print("No binding site detected!!")
        else:
            residue_BS_identifiers = np.ones(self.nresi_per_protein) * 999
            for bs_id, nodes in enumerate(self.node_list):
                residue_BS_identifiers[nodes] = bs_id
            # update dataset
            self.dataset["Binding Site ID"] = residue_BS_identifiers
            # initialise variable for binding site interactions
            self.durations_BS = [[] for dummy in self.node_list]
            self.occupancy_BS = [[] for dummy in self.node_list]
            self.lipid_count_BS = [[] for dummy in self.node_list]
            self.koff_BS = np.zeros(len(self.node_list))
            self.res_time_BS = np.zeros(len(self.node_list))
            self.r_squared_BS = np.zeros(len(self.node_list))

            if print_output:
                for bs_id, nodes in enumerate(self.node_list):
                    print(f"Binding Site ID: {bs_id}")
                    print("{:^10s}{:^12s}".format("Residue", "Residue ID"))
                    for node in nodes:
                        print("{:^10s}{:^12s}".format(self._residue_list[node], self._protein_residue_id[node]))

        return self.node_list

    def cal_site_koff(self, binding_site_id=None, nbootstrap=10, initial_guess=[1., 1., 1., 1.],
                      save_dir=None, print_output=True, sort_residue="Residence Time", plot=True):
        """Calculate interactions for binding sites.

        Parameters
        ----------
        binding_site_id : int or array_like or None, optional, default=None
        nbootstrap : int, optional, default=10
        initial_guess : array_like, default=None
        save_dir : str, optional, default=None
        print_output : bool, optional, default=True
        sort_residue : str, optional, default="Residence Time"
        plot : bool, optional, default=True

        Returns
        ---------
        koffs : list
        restimes : list

        """
        if "Binding Site ID" not in self.dataset.columns:
            raise ValueError(
                "cal_binding_site() has not been implemented yet or no binding site was found!"
            )

        if save_dir is not None:
            BS_dir = check_dir(save_dir, "Binding_Sites_Koffs_{}".format(self.lipid))
        else:
            BS_dir = check_dir(self.save_dir, "Binding_Sites_Koffs_{}".format(self.lipid))

        if binding_site_id is not None:
            selected_bs_id = np.atleast_1d(binding_site_id)
        else:
            selected_bs_id = np.arange(len(self.node_list))

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
            koff_BS_per_residue = self.dataset["Binding Site Koff"].to_list()
            res_time_BS_per_residue = self.dataset["Binding Site Residence Time"].to_list()
            durations_BS_per_residue = self.dataset["Binding Site Duration"].to_list()
            occupancy_BS_per_residue = self.dataset["Binding Site Occupancy"].to_list()
            lipid_count_BS_per_residue = self.dataset["Binding Site Lipid Count"].to_list()
            r_squared_BS_per_residue = self.dataset["Binding Site R Squared"].to_list()
        else:
            koff_BS_per_residue = np.zeros(self.nresi_per_protein)
            res_time_BS_per_residue = np.zeros(self.nresi_per_protein)
            durations_BS_per_residue = np.zeros(self.nresi_per_protein)
            occupancy_BS_per_residue = np.zeros(self.nresi_per_protein)
            lipid_count_BS_per_residue = np.zeros(self.nresi_per_protein)
            r_squared_BS_per_residue = np.zeros(self.nresi_per_protein)

        # calculate interactions for binding sites
        for bs_id in tqdm(selected_bs_id, desc="CALCULATE INTERACTIONS FOR BINDING SITES",
                          total=len(selected_bs_id)):
            nodes = self.node_list[bs_id]
            # calculate durations, occupancy and lipid count
            for traj_idx in np.arange(len(self.trajfile_list)):
                for protein_idx in np.arange(self.nprot):
                    list_to_take = traj_idx * self.nprot + protein_idx
                    n_frames = len(self.contact_residue_low[nodes[0]][list_to_take])
                    contact_BS_low = [np.unique(np.concatenate(
                        [self.contact_residues_low[node][list_to_take][frame_idx] for node in nodes]))
                        for frame_idx in np.arange(n_frames)]
                    contact_BS_high = [np.unique(np.concatenate(
                        [self.contact_residues_high[node][list_to_take][frame_idx] for node in nodes]))
                        for frame_idx in np.arange(n_frames)]
                    self.durations_BS[bs_id].append(
                        Duration(contact_BS_low, contact_BS_high, timestep).cal_durations())
                    occupancy_BS, lipid_count_BS = cal_interaction_frequency(contact_BS_low)
                    self.occupancy_BS[bs_id].append(occupancy_BS)
                    self.lipid_count_BS[bs_id].append(lipid_count_BS)
            # calculate BS koff and BS residence time
            durations = np.concatenate(self.durations_BS[bs_id])
            koff_BS, restime_BS, properties_BS = cal_koff(durations, t_total, timestep, nbootstrap, initial_guess)
            self.koff_BS[bs_id] = koff_BS
            self.res_time_BS[bs_id] = restime_BS
            self.r_squared_BS[bs_id] = properties_BS["r_squared"]

            if plot:
                # plot BS koff and BS residence time
                text = self._format_koff_text(properties_BS)
                plot_koff(durations, properties_BS["delta_t_list"], properties_BS["survival_func"],
                          properties_BS["n_fitted"], survival_rates_bootstraps=properties_BS["survival_func_boot_set"],
                          fig_fn=os.path.join(BS_dir, f"BS_id{bs_id}.pdf"), title=f"Binding Site {bs_id}",
                          timeunit=self.timeunit, t_total=t_total, text=text)

            koff_BS_per_residue[nodes] = koff_BS
            res_time_BS_per_residue[nodes] = restime_BS
            r_squared_BS_per_residue[nodes] = properties_BS["r_squared"]
            durations_BS_per_residue[nodes] = np.mean(durations)
            occupancy_BS_per_residue[nodes] = np.mean(self.occupancy_BS[bs_id])
            lipid_count_BS_per_residue[nodes] = np.mean(self.lipid_count_BS[bs_id])

        # update dataset
        self.dataset["Binding Site Koff"] = koff_BS_per_residue
        self.dataset["Binding Site Residence Time"] = res_time_BS_per_residue
        self.dataset["Binding Site R Squared"] = r_squared_BS_per_residue
        self.dataset["Binding Site Duration"] = durations_BS_per_residue
        self.dataset["Binding Site Occupancy"] = occupancy_BS_per_residue
        self.dataset["Binding Site Lipid Count"] = lipid_count_BS_per_residue

        # write and print binding site info
        for bs_id in selected_bs_id:
            nodes = self.node_list[bs_id]
            text = self._format_BS_print_info(bs_id, nodes, sort_residue)
            with open("{}/BindingSites_Info_{}.txt".format(BS_dir, self.lipid), "a") as f:
                f.write(text)
            if print_output:
                print(text)

        return [self.koff_BS[bs_id] for bs_id in selected_bs_id], [self.res_time_BS[bs_id] for bs_id in selected_bs_id]

    def gen_site_poses(self, binding_site_id=None, n_poses=5, pose_format="gro", score_weights=None,
                       kde_bw=0.15, save_dir=None):
        """Generate representative binding poses

        Parameters
        ----------
        binding_site_id : int or list_like or None, optional, default=None
        n_poses : int, optional, default=5
        pose_format : str, optional, default="gro"
        score_weights : dict or None, optional, default=None
        kde_bw : float, optional, default=0.15
        save_dir : str or None, optinal, default=None

        """
        if "Binding Site ID" not in self.dataset.columns:
            raise ValueError(
                "cal_binding_site() has not been implemented yet or no binding site was found!"
            )

        if save_dir is not None:
            pose_dir = check_dir(save_dir, "Binding_Poses_{}".format(self.lipid))
        else:
            pose_dir = check_dir(self.save_dir, "Binding_Poses_{}".format(self.lipid))

        if binding_site_id is not None:
            selected_bs_id = np.atleast_1d(binding_site_id)
        else:
            selected_bs_id = np.arange(len(self.node_list))

        selected_bs_map = {bs_id:nodes for bs_id, nodes in zip(selected_bs_id, self.node_list[selected_bs_id])}
        # store bound lipid poses
        pose_pool = collect_binding_poses(selected_bs_map, self.contact_residues_low, self.trajfile_list,
                                          self.topfile_list, self.stride, self.lipid, self.nprot)

        # generate representative binding poses
        write_binding_poses(pose_pool, selected_bs_map, self._protein_ref, self._lipid_ref, pose_dir,
                            n_poses=n_poses, pose_format=pose_format, kde_bw=kde_bw, score_weights=score_weights)

        return

    def cal_site_surface_area(self, binding_site_id=None, radii=None, plot=True, save_dir=None):
        """Calculate binding site surface areas.

        Parameters
        -----------
        binding_site_id : int or array_like or None, optional, default=None
        radii : dict or None, optional, default=None
        plot : bool, optional, default=True
        save_dir : str or None, optional, default=None

        """
        if "Binding Site ID" not in self.dataset.columns:
            raise ValueError(
                "cal_binding_site() has not been implemented yet or no binding site was found!"
            )

        if binding_site_id is not None:
            selected_bs_id = np.atleast_1d(np.array(binding_site_id, dtype=int))
        else:
            selected_bs_id = np.arange(len(self.node_list), dtype=int)

        MARTINI_CG_radii = {"BB": 0.26, "SC1": 0.23, "SC2": 0.23, "SC3": 0.23}

        if radii is None:
            radii_book = MARTINI_CG_radii
        else:
            radii_book = {**MARTINI_CG_radii, **radii}

        selected_bs_id_map = {bs_id: self.node_list[bs_id] for bs_id in selected_bs_id}

        # calculate binding site surface area
        surface_area = calculate_surface_area(selected_bs_id_map, radii_book, self.trajfile_list, self.topfile_list,
                                              self.nprot, self.timeunit, self.stride, dt_traj=self.dt_traj)

        # plot surface area
        if plot:
            if save_dir is not None:
                surface_area_dir = check_dir(save_dir, "Surface_Area_{}".format(self.lipid))
            else:
                surface_area_dir = check_dir(self.save_dir, "Surface_area_{}".format(self.lipid))

            plot_surface_area(surface_area, os.pathjoin(surface_area_dir, "Surface_Area_timeseries.pdf"),
                              timeunit=self.timeunit)

            selected_columns = [column for column in surface_area.columns if column != "Time"]
            surface_data = surface_area[selected_columns]
            plot_binding_site_data(surface_data, os.pathjoin(surface_area_dir, "Surface_Area_violinplot.pdf"),
                                   title="Surface Area Violin Plot", ylabel=r"Surface Area (nm$^2$)")

        return surface_area


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
        if save_dir is not None:
            data_dir = check_dir(save_dir, "Dataset_{}".format(self.lipid))
        else:
            data_dir = check_dir(self.save_dir, "Dataset_{}".format(self.lipid))

        if item.lower() == "duration":
            obj = self.durations
        elif item.lower() == "occupancy":
            obj = self.occupancy
        elif item.lower() == "lipid count":
            obj = self.lipid_count
        elif item.lower() == "corrcoef":
            obj = self.interaction_corrcoef
        elif item.lower() == "duration bs":
            obj = self.durations_BS
        elif item.lower() == "occupancy bs":
            obj = self.occupancy_BS
        if 'obj' in locals():
            with open(os.path.join(data_dir, "{}.pickle".format("_".join(item.split()))), "wb") as f:
                pickle.dump(obj, f, 2)

        if item.lower() == "dataset":
            self.dataset.to_csv(os.path.join(data_dir, "dataset.csv"))

        return

    def save_coordinate(self, item, save_dir=None):
        """Save lipid interactions in the b factor column of a PDB coordinate file

        parameters
        -----------
        item : {"Residence Time", "Duration", "Occupancy", "Lipid Count"}
        save_dir : str or None, optional, default=None

        """
        if save_dir is not None:
            coord_dir = check_dir(save_dir, "Coordinate_{}".format(self.lipid))
        else:
            coord_dir = check_dir(self.save_dir, "Coordinate_{}".format(self.lipid))

        data = self.dataset[item].tolist()
        write_PDB(self._protein_ref, data, os.path.join(coord_dir, "Coordinate_{}.pdb".format("_".join(item.split()))),
                  resi_offset=self.resi_offset)

        return

    def save_pymol_script(self, pdb_file, save_dir=None):
        """save a pymol script that maps interactions onto protein structure in PyMol.

        Parameters
        ----------
        pdb_file : str
        save_dir : str, optional, default=None

        """
        if save_dir is not None:
            script_dir = check_dir(save_dir)
        else:
            script_dir = check_dir(self.save_dir)

        if not os.path.isfile(os.path.join(save_dir, "Dataset_{}.csv".format(self.lipid))):
            data_fname = os.path.join(save_dir, "Dataset_{}.csv".format(self.lipid))
            self.dataset.to_csv(data_fname)

        write_pymol_script(os.path.join(script_dir, "show_binding_site_info.py"), pdb_file, data_fname,
                           self.lipid, len(self.node_list))

        return

    def plot(self, item, save_dir=None, gap=200):
        """Plot interactions.

        Parameters
        ----------
        item : {"Residence Time", "Duration", "Occupancy", "Lipid Count", "CorrCoef"}
        save_dir : str, optional, default=None
        gap : int, optional, default=200

        """
        if save_dir is not None:
            figure_dir = check_dir(save_dir, "Figure_{}".format(self.lipid))
        else:
            figure_dir = check_dir(self.save_dir, "Figure_{}".format(self.lipid))

        if item == "Residence Time":
            ylabel = "Res. Time (ns)" if self.timeunit == "ns" else r"Res. Time ($\mu$s)"
        elif item == "Duration":
            ylabel = "Duration (ns)" if self.timeunit == 'ns' else r"Duration ($\mu$s)"
        elif item == "Occupancy":
            ylabel = "Occupancy (%)"
        elif item == "Lipid Count":
            ylabel = "Lipid Count (num.)"

        if "ylabel" in locals():
            data = self.dataset[item]
            title = "{} {}".format(self.lipid, item)
            fig_fn = os.path.join(save_dir, "{}.pdf".format("_".join(item.split())))
            residue_index = np.array([int(re.findall("^[0-9]+", residue)[0]) for residue in self._residue_list])
            plot_residue_data(residue_index, data, gap=gap, ylabel=ylabel, fn=fig_fn, title=title)

        if item == "CorrCoef":
            residue_index = np.array([int(re.findall("^[0-9]+", residue)[0]) for residue in self._residue_list])
            plot_corrcoef(self.interaction_corrcoef, residue_index, fn=os.path.join(figure_dir, "CorrCoef.pdf"),
                          title="{} Correlation Coeffient".format(self.lipid))

        return

    def plot_logo(self, item, save_dir=None, gap=2000, letter_map=None, color_scheme="chemistry"):
        """Plot interactions using logomaker.

        Parameters
        ----------
        item : {"Residence Time", "Duration", "Occupancy", "Lipid Count"}
        save_dir : str or None, optional, default=None
        gap : int, optional, default=2000
        letter_map : dict or None, optional, default=None
        color_scheme : str, optional, default="chemistry"

        """
        if save_dir is not None:
            figure_dir = check_dir(save_dir, "Figure_{}".format(self.lipid))
        else:
            figure_dir = check_dir(self.save_dir, "Figure_{}".format(self.lipid))

        single_letter = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
                         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
                         'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
        if letter_map is not None:
            single_letter.update(letter_map)

        resname_set = [re.findall("[a-zA-Z]+$", residue)[0] for residue in self._residue_list]
        residue_index = np.array([int(re.findall("^[0-9]+", residue)[0]) for residue in self._residue_list])

        if item == "Residence Time":
            ylabel = "Res. Time (ns)" if self.timeunit == "ns" else r"Res. Time ($\mu$s)"
        elif item == "Duration":
            ylabel = "Duration (ns)" if self.timeunit == 'ns' else r"Duration ($\mu$s)"
        elif item == "Occupancy":
            ylabel = "Occupancy (%)"
        elif item == "Lipid Count":
            ylabel = "Lipid Count (num.)"

        if "ylabel" in locals():
            data = self.dataset[item]
            title = "{} {} Logo".format(self.lipid, item)
            fig_fn = os.path.join(save_dir, "{}_logo.pdf".format("_".join(item.split())))
            plot_residue_data_logos(residue_index, resname_set, data, ylabel=ylabel,
                                    fn=fig_fn, title=title, letter_map=letter_map, color_scheme=color_scheme)

        return

    ############################################
    #     assisting func
    ############################################
    def _wrapup_in_logfile(self):
        log_fn = "{}/calculation_log_{}.txt".format(self.save_dir, self.lipid)
        f = open(log_fn, "a+")
        f.write("###### Lipid: {}\n".format(self.lipid))
        f.write("###### Lipid Atoms: {}\n".format(self.lipid_atoms))
        f.write("###### Cutoffs: {}\n".format(self.cutoff))
        f.write("###### nprot: {}\n".format(self.nprot))
        f.write("###### Trajectories:\n")
        for traj_fn in self.trajfile_list:
            f.write("  {}\n".format(traj_fn))
        f.write("###### Coordinates:\n")
        for top_fn in self.topfile_list:
            f.write("  {}\n".format(top_fn))
        # write interaction information trajectory by trajectory
        for traj_idx, trajfile in enumerate(self.trajfile_list):
            print("\n########## {} interactions in \n########## {} \n".format(
                self.lipid, trajfile))
            f.write("\n########## {} interactions in \n########## {} \n".format(
                self.lipid, trajfile))
            durations = np.array(
                [np.concatenate(
                    self.durations[residue_id][traj_idx * self.nprot:(traj_idx + 1) * self.nprot]).mean()
                 for residue_id in self._protein_residue_id])
            duration_arg_idx = np.argsort(durations)[::-1]
            occupancies = np.array(
                [np.mean(self.occupancy[residue_id][traj_idx * self.nprot:(traj_idx + 1) * self.nprot])
                 for residue_id in self._protein_residue_id])
            occupancy_arg_idx = np.argsort(occupancies)[::-1]
            lipid_counts = np.array(
                [np.mean(self.lipid_count[residue_id][traj_idx * self.nprot:(traj_idx + 1) * self.nprot])
                 for residue_id in self._protein_residue_id])
            lipid_count_arg_idx = np.argsort(lipid_counts)[::-1]
            log_text = "10 residues that showed longest average interaction durations ({}):\n".format(self.timeunit)
            for residue, duration in zip(self._residue_list[duration_arg_idx][:10],
                                         durations[duration_arg_idx][:10]):
                log_text += "{:^8s} -- {:^8.3f}\n".format(residue, duration)
            log_text += "10 residues that showed highest lipid occupancy (100%):\n"
            for residue, occupancy in zip(self._residue_list[occupancy_arg_idx][:10],
                                          occupancies[occupancy_arg_idx][:10]):
                log_text += "{:^8s} -- {:^8.2f}\n".format(residue, occupancy)
            log_text += "10 residues that have the largest number of surrounding lipids (count):\n"
            for residue, lipid_count in zip(self._residue_list[lipid_count_arg_idx][:10],
                                            lipid_counts[lipid_count_arg_idx][:10]):
                log_text += "{:^8s} -- {:^8.2f}\n".format(residue, lipid_count)
            print(log_text)
            f.write(log_text)
        f.write("\n")
        f.close()
        return

    def _format_koff_text(self, properties):
        """Format text for koff plot. """
        tu = "ns" if self.timeunit == "ns" else r"$\mu$ s"
        text = "{:18s} = {:.3f} {:2s}$^{{-1}} $\n".format("$k_{{off1}}$", properties["ks"][0], tu)
        text += "{:18s} = {:.3f} {:2s}$^{{-1}} $\n".format("$k_{{off2}}$", properties["ks"][1], tu)
        text += "{:14s} = {:.4f}\n".format("$R^2$", properties["r_squared"])
        ks_boot_avg = np.mean(properties["ks_boot_set"], axis=0)
        cv_avg = 100 * np.std(properties["ks_boot_set"], axis=0) / np.mean(properties["ks_boot_set"], aixs=0)
        text += "{:18s} = {:.3f} {:2s}$^{{-1}}$ ({:3.1f}%)\n".format("$k_{{off1, boot}}$", tu,
                                                                     ks_boot_avg[0], cv_avg[0])
        text += "{:18s} = {:.3f} {:2s}$^{{-1}}$ ({:3.1f}%)\n".format("$k_{{off2, boot}}$", tu,
                                                                     ks_boot_avg[1], cv_avg[1])
        text += "{:18s} = {:.3f} {:2s}".format("Res. Time", properties["res_time"], tu)

        return text

    def _format_BS_print_info(self, bs_id, nodes, sort_item):
        """Format binding site information."""
        import re as _re

        Residue_property_book = {"ARG": "Pos. Charge", "HIS": "Pos. Charge", "LYS": "Pos. Charge",
                                 "ASP": "Neg. Charge", "GLU": "Neg. Charge",
                                 "SER": "Polar", "THR": "Polar", "ASN": "Polar", "GLN": "Polar",
                                 "CYS": "Special", "SEC": "Special", "GLY": "Special", "PRO": "Special",
                                 "ALA": "Hydrophobic", "VAL": "Hydrophobic", "ILE": "Hydrophobic", "LEU": "Hydrophobic",
                                 "MET": "Hydrophobic", "PHE": "Hydrophobic", "TYR": "Hydrophobic", "TRP": "Hydrophobic"}

        text = "# Binding site {}\n".format(bs_id)
        text += "{:30s} {:10.3f} {:5s}\n".format(" Binding Site Residence Time:",
                                                 self.res_time_BS[bs_id], self.timeunit)
        text += "{:30s} {:10.3f}  R squared: {:7.4f}\n".format(" Binding Site Koff:", self.koff_BS[bs_id],
                                                               self.r_squared_BS[bs_id])
        text += "{:30s} {:10.3f} {:5s}\n".format(" Binding Site Duration:", self.durations_BS[bs_id], self.timeunit)
        text += "{:30s} {:10.3f} %\n".format(" Binding Site Occupancy:", self.occupancy_BS[bs_id])
        text += "{:30s} {:10.3f}\n".format(" Binding Site Lipid Count:", self.lipid_count_BS[bs_id])
        res_stats = {"Pos. Charge": 0, "Neg. Charge": 0, "Polar": 0, "Special": 0, "Hydrophobic": 0}
        # stats on the chemical properties of binding site residues
        for residue in self._residue_list[nodes]:
            res_stats[Residue_property_book[_re.findall("[a-zA-Z]+$", residue)[0]]] += 1
        BS_num_resi = len(nodes)
        text += "{:20s} {:10s}\n".format(" Pos. Charge:", "/".join([str(res_stats["Pos. Charge"]), str(BS_num_resi)]))
        text += "{:20s} {:10s}\n".format(" Neg. Charge:", "/".join([str(res_stats["Neg. Charge"]), str(BS_num_resi)]))
        text += "{:20s} {:10s}\n".format(" Polar:", "/".join([str(res_stats["Polar"]), str(BS_num_resi)]))
        text += "{:20s} {:10s}\n".format(" Hydrophobic:", "/".join([str(res_stats["Hydrophobic"]), str(BS_num_resi)]))
        text += "{:20s} {:10s}\n".format(" Special:", "/".join([str(res_stats["Special"]), str(BS_num_resi)]))
        text += "{:^9s}{:^7s}{:^16s}{:^16s}{:^15s}{:^12s}{:^7s}{:^10s}\n".format("Residue", "Resid",
                                                                          "Res. Time ({})".format(self.timeunit),
                                                                          "Duration ({})".format(self.timeunit),
                                                                          "Occupancy (%)", "Lipid Count",
                                                                          "Koff", "R Squared")
        info_dict_set = self.dataset.iloc[nodes].sort_values(by=sort_item, ascending=False).to_dict("records")
        for info_dict in info_dict_set:
            text += "{Residue:^9s}{Residue ID:^7d}{Residence Time:^16.3f}{Duration:^16.3f}" \
                    "{Occupancy:^15.3f}{Lipid Count:^12.3f}{Koff:^7.3f}{R Squared:^10.3f}\n".format(**info_dict)

        text += "\n"
        text += "\n"

        return text

