#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 19:28:17 2019

@author: Wanling Song

"""
import mdtraj as md
import numpy as np
import pandas as pd
import argparse
import sys
from collections import defaultdict
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix
from scipy import sparse
from scipy import stats
import community
import warnings
from shutil import copyfile
import datetime
from itertools import product
import logomaker
import rmsd
warnings.simplefilter(action='ignore', category=FutureWarning)
np.seterr(all='ignore')

###################################
######  Parameter settings  #######
###################################

parser = argparse.ArgumentParser()
parser.add_argument("-f", nargs="+", metavar="./run/md.xtc", help="List of trajectories, seperated by space, \
                     Supports xtc, gro format. Used by mdtraj.load()")
parser.add_argument("-c", nargs="+", metavar="./run/system.gro", \
                    help="List of coordinates of trajectory, in the same order as -f, required when inputs of -f are xtc trajectories, \
                    Supported format: gro, pdb, etc., Used by mdtraj.load()")
parser.add_argument("-stride", default=1, metavar=1, help="Striding through trajectories. Only every stride-th will be analized." )
parser.add_argument("-dt", default=None, metavar="None", help="The time interval between two adjacent frames in the trajectories. \
                    If not specified, the mdtraj will deduce from the trajectories. This works for trajectories in format of e.g. xtc which \
                    include timestep information. For trajectories in dcd format, users have to provide the time interval manually, \
                    in a time unite consistent with -tu")
parser.add_argument("-tu", default="us", choices=["ns", "us"], metavar="us", \
                    help="Time unit for interaction duration calculation. Available options: ns, us. This will affect the unit of koff as well.")
parser.add_argument("-save_dir", default=None, metavar="None", help="The directory where all the generated results will be put in. \
                    The directory will be created if not existing. Using the current working directory if not specified.")
parser.add_argument("-cutoffs", nargs=2, default=(0.55, 1.0), metavar=(0.55, 1.0), \
                    help="Double cutoff seperated by space. In unit of nm. Default is 0.55 1.0. The double cutoffs are used to define lipid \
                    interactions. A continuous lipid contact with a given residue starts when the lipid moves to the given residue \
                    closer than the smaller cutoff; and ends when the lipid moves farther than the larger cutoff. The standard single \
                    cutoff can be acheived by setting the same value for both cutoffs.")
parser.add_argument("-lipids", nargs="+", metavar="POPC", default="POPC CHOL POP2", \
                    help="Lipid species to check, seperated by space. Should be consistent with residue names in your trajectories.")
parser.add_argument("-lipid_atoms", nargs="+", metavar="PO4", default=None, \
                    help="Lipid atoms to check, seperated by space. Should be consistent with the atom names in your trajectories.")
parser.add_argument("-radii", nargs="+", default=None, metavar="BB:0.26 SC1:0.23", help="Change/Define the radius of atoms/beads \
                    that is used for the calculation of binding site surface area. Supported syntax is BB:0.26, which defines the radius of \
                    bead BB as 0.26 nm, or CA:0.12 which defines the radius of atom CA as 0.12 nm. For atomistic simulations, the default radii are taken from\
                    mdtraj https://github.com/mdtraj/mdtraj/blob/master/mdtraj/geometry/sasa.py#L56. For coarse-grained \
                    simulations, this script defines the radius of the MARTINI 2 beads of BB as 0.26 nm and SC1/2/3 as 0.23 nm.")
parser.add_argument("-nprot", default=1, metavar="1", \
                    help="num. of proteins (or chains) in the simulation system. The calculated results will be averaged among these proteins \
                    (or chains). The proteins (or chains) need to be identical, otherwise the averaging will fail.")
parser.add_argument("-resi_offset", default=0, metavar="0", help="Shifting the residue index. It is useful if you need to change the residue \
                    index in your trajectories. For example, to change the residue indeces from 5,6,7,..., to 10,11,12,..., use -resi_offset 4. \
                    All the outputs, including plotted figures and saved coordinates, will be changed by this.")
parser.add_argument("-resi_list", nargs="+", default=[], metavar="1-10 20-30", help="The indices of residues on which the calculations are done. \
                    This option is useful for those proteins with large regions that don't require calculation. Skipping those calculations could \
                    save time and memory. Accepted syntax include 1/ defining a range, like 1-10 (both ends included); 2/ single residue index, \
                    like 25 26 17. All the selections are seperated by space. For example, -resi_list 1-10 20-30 40 45 46 means selecting \
                    residues 1-10, 20-30, 40, 45 and 46 for calculation. The residue indices are not affected by -resi_offset, i.e. they \
                    should be consistent with the indices in your trajectories.")
parser.add_argument("-nbootstrap", default=10, metavar=10, help="The number of samples for bootstrapping the calcultion of koff. \
                    The default is 10. The larger the number, the more time-consuming the calculation will be. The closer the bootstrapped \
                    residence time/koffs are to the original values, the more reliable those original values are. The bootstrapped results \
                    are ploted in each of the koff plots and plotted apposed to the original values in the figure showing residence time. ")
parser.add_argument("-save_dataset", nargs="?", default=True, const=True, metavar="True", help="Save dataset in Pickle. Default is True")
parser.add_argument("-gen_binding_poses", default=5, metavar=0, help="The num. of top-scored lipid binding poses to be generated for each binding \
                    site. The default is 5. A scoring function is generated for each binding site based on the sum of the probability density function of each atom/bead \
                    the lipid molecule. Score = sum(PDF(atom_i) * Weight(atom_i)) for atom_i in the lipid molecule. The weight function Weight(atom_i) \
                    is specified by the flag -score_weights.")
parser.add_argument("-save_pose_format", default="gro", metavar="gro", help="The format the generated lipid binding poses are written into. This function \
                    is carried out by mdtraj.save(), hence supports the formats that are included by mdtraj. ")
parser.add_argument("-score_weights", default=None, metavar="PO4:1 C1:1", help="The weight of each of the lipid atom/bead contributes to the scoring function. \
                    Top-rated lipid binding poses can be generated based on users' specification. The bounds poses of each binding site are scored based \
                    on the scoring function Score = sum(PDF(atom_i) * Weight(atom_i)) for atom_i in the lipid molecule.")
parser.add_argument("-letter_map", default=None, metavar="ARG:K GLY:G", help="Map the three-letter amino acids to one letter. This map is \
                    used in making logomaker figures (https://logomaker.readthedocs.io/en/latest/). The common 20 amino acids are defined \
                    by this script. Users need to use this flag to define maps for uncommon amino acids in their systems.")              
parser.add_argument("-pdb", default=None, metavar="None", help="Provide a PDB structure onto which the binding site information will be mapped. \
                    Using this flag will generate a 'show_binding_site_info.py' file in the -save_dir directory, which allows users to check the \
                    mapped binding site information in PyMol. Users can run the generated script by 'python show_binding_site_info.py' \
                    to open such a PyMol session.")
parser.add_argument("-pymol_gui", nargs="?", default=True, const=True, metavar="True", help="Show the PyMol session of binding site information \
                    at the end of the calcution. Need to be used in conjuction with -pdb.")
parser.add_argument("-chain", default=None, metavar="None", help="Select the chain of the structure provided by -pdb to which the binding \
                    site information mapped. This option is useful when the pdb structure has multiple chains. ")

args = parser.parse_args(sys.argv[1:])

##########################################
########## assisting functions ###########
##########################################

def get_atom_index_for_lipid(lipid, traj, part=None):
    whole_atom_index = traj.top.select("resname {}".format(lipid))
    if part != None:
        parts_atom_index = [traj.topology.atom(idx).index for idx in whole_atom_index if traj.topology.atom(idx).name in part]
        return parts_atom_index
    else:
        return whole_atom_index


class Durations():
    def __init__(self, contact_residues_low, contact_residue_high, dt):
        self.contact_low = contact_residues_low
        self.contact_high = contact_residue_high
        self.dt = dt

    def cal_duration(self):
        self.pointer = [np.zeros_like(self.contact_high[idx], dtype=np.int) for idx in range(len(self.contact_high))]
        durations = []
        for i in range(len(self.contact_low)):
            for j in range(len(self.contact_low[i])):
                pos = np.where(self.contact_high[i] == self.contact_low[i][j])[0][0]
                if self.pointer[i][pos] == 0:
                    durations.append(self.get_duration(i, pos))
        if len(durations) == 0:
            return [0]
        else:
            return durations

    def get_duration(self, i, j):
        count = 1
        self.pointer[i][j] = 1
        lipid_to_search = self.contact_high[i][j]
        for k in range(i+1, len(self.contact_high)):
            locations = np.where(self.contact_high[k] == lipid_to_search)[0]
            if len(locations) == 0:
                return count * self.dt
            else:
                pos = locations[0]
                self.pointer[k][pos] = 1
                count +=1
        return (count - 1) * self.dt


def cal_interaction_intensity(contact_residues_high):
    """
    The probablily of finding the lipids around the selected residue plus the number of
    lipids found around the selected residue, the average number of lipid per contact
    """
    contact_counts = [len(item) for item in contact_residues_high]
    mask = np.array(contact_counts) > 0
    contact_counts_nonzero = np.array(contact_counts)[mask]
    return 100 * len(contact_counts_nonzero)/len(contact_residues_high), np.nan_to_num(contact_counts_nonzero.mean())


def cal_sigma(durations, num_of_lipids, T_total, delta_t_range):
    sigma = {}
    for delta_t in delta_t_range:
        if delta_t == 0:
            sigma[delta_t] = 1
            sigma0 = float(sum([restime - delta_t for restime in durations if restime >= delta_t])) / ((T_total - delta_t) * num_of_lipids)
        else:
            try:
                sigma[delta_t] = float(sum([restime - delta_t for restime in durations if restime >= delta_t])) / ((T_total - delta_t) * num_of_lipids * sigma0)
            except ZeroDivisionError:
                sigma[delta_t] = 0
    return sigma


def cal_restime_koff(sigma, initial_guess):
    """
    fit the exponential curve y=A*e^(-k1*x)+B*e^(-k2*x)
    """
    delta_t_range = list(sigma.keys())
    delta_t_range.sort() # x
    hist_values = np.nan_to_num([sigma[delta_t] for delta_t in delta_t_range]) # y
    try:
        popt, pcov = curve_fit(bi_expo, np.array(delta_t_range, dtype=np.float128), np.array(hist_values, dtype=np.float128), p0=initial_guess, maxfev=100000)
        n_fitted = bi_expo(np.array(delta_t_range, dtype=np.float128), *popt)
        r_squared = 1 - np.sum((np.nan_to_num(n_fitted) - np.nan_to_num(hist_values))**2)/np.sum((hist_values - np.mean(hist_values))**2)
        ks = [abs(k) for k in popt[:2]]
        koff = np.min(ks)
        restime = 1/koff
    except RuntimeError:
        koff = 0
        restime = 0
        r_squared = 0
        popt = [0, 0, 0, 0]
    return restime, koff, r_squared, popt

def bi_expo(x, k1, k2, A, B):
    return A*np.exp(-k1*x) + B*np.exp(-k2*x)

def check_dir(save_dir, suffix=None):

    if save_dir == None:
        save_dir = os.getcwd()
    else:
        save_dir = os.path.abspath(save_dir)
    if suffix != None:
        save_dir = os.path.join(save_dir, suffix)
    if not os.path.isdir(save_dir):
        print("Creating new director: {}".format(save_dir))
        os.makedirs(save_dir)

    return save_dir

def sparse_corrcoef(A, B=None):

    if B is not None:
        A = sparse.vstack((A, B), format='csr')
    A = A.astype(np.float64)
    n = A.shape[1]
    # Compute the covariance matrix
    rowsum = A.sum(1)
    centering = rowsum.dot(rowsum.T.conjugate()) / n
    C = (A.dot(A.T.conjugate()) - centering) / (n - 1)
    # The correlation coefficients are given by
    # C_{i,j} / sqrt(C_{i} * C_{j})
    d = np.diag(C)
    coeffs = C / np.sqrt(np.outer(d, d))
    return coeffs


#####################################
####### Main Class object ###########
#####################################


class LipidInteraction():

    def __init__(self, trajfile_list, grofile_list=None, stride=1, dt=None, cutoff=[0.55, 1.0], \
                 lipid="POPC", lipid_atoms=None, nprot=1, resi_list=[], resi_offset=0, save_dir=None, timeunit="us"):
        if grofile_list != None:
            assert len(trajfile_list) == len(grofile_list), \
            "List of coordinates should be in the same order and length of list of trajectories!"

        self.save_dir = check_dir(save_dir)
        self.trajfile_list = trajfile_list
        self.grofile_list = grofile_list
        self.dt = dt
        self.nrepeats = len(self.trajfile_list)
        self.cutoff = np.sort(np.array(cutoff, dtype=float))
        self.lipid = lipid
        self.lipid_atoms = lipid_atoms
        self.nprot = int(nprot)
        self.timeunit = timeunit
        self.koff = {}
        self.sigmas = {}
        self.params = {}
        self.r_squared = {}
        self.res_time = {}
        self.koff_b = {}
        self.koff_b_cv = {}
        self.res_time_b = {}
        self.res_time_b_cv = {}
        self.r_squared_b = {}
        self.interaction_duration = defaultdict(list)
        self.interaction_occupancy = defaultdict(list)
        self.lipid_count = defaultdict(list)
        self.dist_matrix_set = []
        self.stride = stride
        self.resi_offset = resi_offset

        traj = md.load(self.trajfile_list[0], top=self.grofile_list[0], stride=self.stride)
        all_protein_atom_indices = traj.top.select("protein")
        self.natoms_per_protein = int(len(all_protein_atom_indices)/self.nprot)
        self.prot_atom_indices = all_protein_atom_indices[:self.natoms_per_protein]
        self.nresi_per_protein = traj.top.atom(self.prot_atom_indices[-1]).residue.index - \
                                 traj.top.atom(self.prot_atom_indices[0]).residue.index + 1
        self._protein_ref = traj[0].atom_slice(self.prot_atom_indices)
        one_lipid_indices = []
        for lipid_id in np.sort(traj.top.select("resn {}".format(self.lipid))):
            if len(one_lipid_indices) == 0:
                one_lipid_indices.append(lipid_id)
            if traj.top.atom(lipid_id).residue.index != traj.top.atom(one_lipid_indices[-1]).residue.index:
                break
            else:
                one_lipid_indices.append(lipid_id)
        self._lipid_ref = traj[0].atom_slice(np.unique(one_lipid_indices))
        if len(resi_list) == 0:
            residue_set = ["{}{}".format(traj.top.residue(resi).resSeq+resi_offset, traj.top.residue(resi).name) \
                           for resi in np.arange(self.nresi_per_protein)]
            self.residue_set = np.array(residue_set, dtype=str) # residue id in structure instead of builtin index in mdtraj
            self.protein_residue_indices_set = [] # atom indices for each residue
            for protein_idx in range(self.nprot):
                self.protein_residue_indices_set.append(np.unique([traj.top.atom(atom_idx).residue.index \
                                                                   for atom_idx in \
                                                                   all_protein_atom_indices[protein_idx*self.natoms_per_protein:(protein_idx+1)*self.natoms_per_protein]]))
        elif len(resi_list) > 0:
            resi_list = np.sort(np.array(np.hstack(resi_list), dtype=int))
            selected_residues_per_protein = np.unique([traj.top.atom(atom_idx).residue.index for atom_idx in self.prot_atom_indices \
                                                       if traj.top.atom(atom_idx).residue.resSeq in resi_list])
            residue_set = ["{}{}".format(traj.top.residue(resi).resSeq+resi_offset, traj.top.residue(resi).name) \
                           for resi in selected_residues_per_protein]
            self.residue_set = np.array(residue_set, dtype=str)
            self.protein_residue_indices_set = []
            for protein_idx in range(self.nprot):
                self.protein_residue_indices_set.append(np.array([resi \
                                                          for resi in selected_residues_per_protein + protein_idx*self.nresi_per_protein]))
        
        return


    def cal_interactions(self, save_dir=None, save_dataset=True, nbootstrap=10):

        if save_dir == None:
            self.save_dir = check_dir(self.save_dir, "Interaction_{}".format(self.lipid))
        else:
            self.save_dir = check_dir(save_dir, "Interaction_{}".format(self.lipid))

        with open("{}/calculation_log_{}.txt".format(self.save_dir, self.lipid), "w") as f:
            f.write("###### Lipid: {}\n".format(self.lipid))
            f.write("###### Lipid Atoms: {}\n".format(self.lipid_atoms))
            f.write("###### Cutoffs: {}\n".format(self.cutoff))
            f.write("###### nprot: {}\n".format(self.nprot))
            f.write("###### Trajectories:\n")
            for traj_fn in self.trajfile_list:
                f.write("  {}\n".format(traj_fn))
            f.write("###### Coordinates:\n")
            for gro_fn in self.grofile_list:
                f.write("  {}\n".format(gro_fn))
            f.write("\n")
            row = []
            col = []
            data = []
            self.num_of_lipids = []
            self.T_total = []
            self.timesteps = []
            self._lipid_resi_indices_set = []
            ncol_start = 0
            for traj_idx, trajfile in enumerate(self.trajfile_list):
                print("\n########## Start calculation of {} interaction in \n########## {} \n".format(self.lipid, self.trajfile_list[traj_idx]))
                f.write("\n###### Start calculation of {} interaction in \n###### {} \n".format(self.lipid, self.trajfile_list[traj_idx]))
                traj = md.load(trajfile, top=self.grofile_list[traj_idx], stride=self.stride)
                if self.dt == None:
                    timestep = traj.timestep/1000000.0 if self.timeunit == "us" else traj.timestep/1000.0
                else:
                    timestep = float(self.dt * self.stride)
                lipid_atom_indices = traj.top.select("resn {}".format(self.lipid))
                lipid_resi_indices = set()
                for atom in lipid_atom_indices:
                    lipid_resi_indices.add(traj.top.atom(atom).residue.index)
                lipid_resi_indices = list(lipid_resi_indices)
                lipid_resi_indices.sort()
                self._lipid_resi_indices_set.append(lipid_resi_indices)
                self.num_of_lipids.append(len(lipid_resi_indices))
                self.T_total.append((traj.n_frames - 1) * timestep)
                self.timesteps.append(timestep)
                ### build new traj if necessary ###
                if self.lipid_atoms != None:
                    lipid_haystack = get_atom_index_for_lipid(self.lipid, traj, part=self.lipid_atoms)
                    new_xyz = []
                    selected_atom_indices = np.hstack([traj.top.select("protein"), lipid_haystack])
                    for frame in traj.xyz:
                        new_frame = frame[selected_atom_indices]
                        new_xyz.append(new_frame)
                    reduced_frame = traj[0].atom_slice(selected_atom_indices)
                    reduced_top = reduced_frame.top
                    new_traj = md.Trajectory(new_xyz, reduced_top, time=traj.time, unitcell_lengths=traj.unitcell_lengths, \
                                             unitcell_angles=traj.unitcell_angles)
                    lipid_resi_indices = [new_traj.top.atom(new_traj.top.select("protein")[-1]).residue.index+1+idx \
                                          for idx in np.arange(self.num_of_lipids[-1])]
                else:
                    new_traj = traj
                ncol_per_protein = self.num_of_lipids[-1] * new_traj.n_frames
                for idx_protein in np.arange(self.nprot):
                    pairs = list(product(self.protein_residue_indices_set[idx_protein],lipid_resi_indices))
                    contact_dist_matrix, _ = md.compute_contacts(new_traj, pairs, scheme="closest", periodic=True)
                    self.dist_matrix_set.append(contact_dist_matrix)
                    for resid, residue in enumerate(self.residue_set):
                        dist_matrix_resi = contact_dist_matrix[:, resid*self.num_of_lipids[-1]:(resid+1)*self.num_of_lipids[-1]]
                        contact_residues_low = [[] for dummy in np.arange(new_traj.n_frames)]
                        contact_residues_high = [[] for dummy in np.arange(new_traj.n_frames)]
                        frame_id_set_low, lipid_id_set_low = np.where(dist_matrix_resi <= self.cutoff[0])
                        frame_id_set_high, lipid_id_set_high = np.where(dist_matrix_resi <= self.cutoff[1])
                        for frame_id, lipid_id in zip(frame_id_set_low, lipid_id_set_low):
                            contact_residues_low[frame_id].append(lipid_id)
                        for frame_id, lipid_id in zip(frame_id_set_high, lipid_id_set_high):
                            contact_residues_high[frame_id].append(lipid_id)
                        col.append([ncol_start + ncol_per_protein*idx_protein + lipid_id*new_traj.n_frames + \
                                    frame_id for frame_id, lipid_id in zip(frame_id_set_low, lipid_id_set_low)])
                        row.append([resid for dummy in np.arange(len(col[-1]))])
                        data.append(dist_matrix_resi[frame_id_set_low, lipid_id_set_low])
                        self.interaction_duration[residue].append(Durations(contact_residues_low, contact_residues_high, timestep).cal_duration())
                        occupancy, lipidcount = cal_interaction_intensity(contact_residues_high)
                        self.interaction_occupancy[residue].append(occupancy)
                        self.lipid_count[residue].append(lipidcount)
                ncol_start += ncol_per_protein * self.nprot

                ###############################################
                ###### get some statistics for this traj ######
                ###############################################
                durations = np.array([np.concatenate(self.interaction_duration[residue][-self.nprot:]).mean() for residue in self.residue_set])
                duration_arg_idx = np.argsort(durations)[::-1]
                occupancies = np.array([np.mean(self.interaction_occupancy[residue][-self.nprot:]) for residue in self.residue_set])
                occupancy_arg_idx = np.argsort(occupancies)[::-1]
                lipidcounts =  np.array([np.mean(self.lipid_count[residue][-self.nprot:]) for residue in self.residue_set])
                lipidcount_arg_idx = np.argsort(lipidcounts)[::-1]
                log_text = "10 residues that showed longest interaction (and their raw interaction durations):\n".format(int(idx_protein))
                for residue, duration in zip(self.residue_set[duration_arg_idx][:10], durations[duration_arg_idx][:10]):
                    log_text += "{:^8s} -- {:^8.3f}\n".format(residue, duration)
                log_text += "10 residues that showed highest lipid occupancy:\n"
                for residue, occupancy in zip(self.residue_set[occupancy_arg_idx][:10], occupancies[occupancy_arg_idx][:10]):
                    log_text += "{:^8s} -- {:^8.2f}\n".format(residue, occupancy)
                log_text += "10 residues that have the largest number of surrounding lipids:\n"
                for residue, lipidcount in zip(self.residue_set[lipidcount_arg_idx][:10], lipidcounts[lipidcount_arg_idx][:10]):
                    log_text += "{:^8s} -- {:^8.2f}\n".format(residue, lipidcount)
                print(log_text)
                f.write(log_text)

            row = np.concatenate(row)
            col = np.concatenate(col)
            data = np.concatenate(data)
            contact_info = coo_matrix((data, (row, col)), shape=(self.nresi_per_protein, ncol_start))
            self.interaction_covariance = sparse_corrcoef(contact_info)

        ###################################################
        ############ calculate and plot koffs #############
        ###################################################
        koff_dir = check_dir(self.save_dir, "Koffs_{}".format(self.lipid))
        for residue in self.residue_set:
            duration_raw = np.concatenate(self.interaction_duration[residue])
            if np.sum(duration_raw) > 0:
                bootstrap_results = self.bootstrap(duration_raw, residue, "{}/{}_{}.pdf".format(koff_dir, self.lipid, residue), \
                                                   nbootstrap=nbootstrap)
                self.sigmas[residue] = bootstrap_results["sigma"]
                self.koff[residue] = bootstrap_results["koff"]
                self.res_time[residue] = bootstrap_results["restime"]
                self.params[residue] = bootstrap_results["params"]
                self.r_squared[residue] = bootstrap_results["r_squared"]
                self.koff_b[residue] = bootstrap_results["koff_b_avg"]
                self.koff_b_cv[residue] = bootstrap_results["koff_b_cv"]
                self.res_time_b[residue] = bootstrap_results["res_time_b_avg"]
                self.res_time_b_cv[residue] = bootstrap_results["res_time_b_cv"]
                self.r_squared_b[residue] = bootstrap_results["r_squared_b_avg"]
            else:
                delta_t_range = np.arange(0, self.T_total[traj_idx], np.min(self.timesteps))
                self.sigmas[residue] = {key:value for key, value in zip(delta_t_range, np.zeros(len(delta_t_range)))}
                self.koff[residue] = 0
                self.res_time[residue] = 0
                self.params[residue] = [0, 0, 0, 0]
                self.r_squared[residue] = 0.0
                self.koff_b[residue] = 0
                self.koff_b_cv[residue] = 0
                self.res_time_b[residue] = 0
                self.res_time_b_cv[residue] = 0
                self.r_squared_b[residue] = 0.0

        ##############################################
        ########## wrapping up dataset ###############
        ##############################################
        T_max = np.max(self.T_total)
        Res_Time = np.array([self.res_time[residue] for residue in self.residue_set])
        Capped = Res_Time > T_max
        Res_Time[Capped] = T_max
        Res_Time_B = np.array([self.res_time_b[residue] for residue in self.residue_set])
        Capped = Res_Time_B > T_max
        Res_Time_B[Capped] = T_max
        dataset = pd.DataFrame({"Residue": [residue for residue in self.residue_set],
                                "Occupancy": np.array([np.mean(self.interaction_occupancy[residue]) \
                                                       for residue in self.residue_set]),
                                "Occupancy_std": np.array([np.std(self.interaction_occupancy[residue]) \
                                                           for residue in self.residue_set]),
                                "Duration": np.array([np.mean(np.concatenate(self.interaction_duration[residue])) \
                                                      for residue in self.residue_set]),
                                "Duration_std": np.array([np.std(np.concatenate(self.interaction_duration[residue])) \
                                                          for residue in self.residue_set]),
                                "Residence Time": Res_Time,
                                "Capped": Capped,
                                "R squared": np.array([self.r_squared[residue] for residue in self.residue_set]),
                                "Koff": np.array([self.koff[residue] for residue in self.residue_set]),
                                "Residence Time_boot": Res_Time_B,
                                "Residence Time_boot_cv": np.array([self.res_time_b_cv[residue] for residue in self.residue_set]),
                                "Koff_boot": np.array([self.koff_b[residue] for residue in self.residue_set]),
                                "Koff_boot_cv": np.array([self.koff_b_cv[residue] for residue in self.residue_set]),
                                "R squared_boot": np.array([self.r_squared_b[residue] for residue in self.residue_set]),
                                "LipidCount": np.array([np.mean(self.lipid_count[residue]) \
                                                         for residue in self.residue_set]),
                                "LipidCount_std": np.array([np.std(self.lipid_count[residue]) \
                                                             for residue in self.residue_set])})

        dataset.to_csv("{}/Interactions_{}.csv".format(self.save_dir, self.lipid), index=False)
        self.dataset = dataset

        reminder = """
NOTE:
Occupancy:     percentage of frames where lipid is in contact
               with the given residue (0-100%);
Duration:      Average length of a continuous interaction of lipid
               with the given residue (in unit of {timeunit});
LipidCount:    Average number of lipid surrounding the given residue within the longer cutoff;
Koff:          Koff of lipid with the given residue (in unit of ({timeunit})^(-1));
                """.format(**{"timeunit": self.timeunit})
        print(reminder)
        print()

        if save_dataset:
            dataset_dir = check_dir(self.save_dir, "Dataset")
            with open("{}/interaction_durations_{}.pickle".format(dataset_dir, self.lipid), "wb") as f:
                pickle.dump(self.interaction_duration, f, 2)
            with open("{}/sigmas_{}.pickle".format(dataset_dir, self.lipid), "wb") as f:
                pickle.dump(self.sigmas, f, 2)
            with open("{}/curve_fitting_params_{}.pickle".format(dataset_dir, self.lipid), "wb") as f:
                pickle.dump(self.params, f, 2)
            with open("{}/interaction_covariance_matrix_{}.pickle".format(dataset_dir, self.lipid), "wb") as f:
                pickle.dump(self.interaction_covariance, f, 2)
        return


    def bootstrap(self, durations, label, fig_fn, nbootstrap=10):
        """
        bootstrap durations to calculate koffs, return bootstrapped values
        """
        initial_guess = (1, 1, 1, 1)
        ##### prep for plotting ######
        plt.rcParams["font.size"] = 10
        plt.rcParams["font.weight"] = "bold"
        if self.timeunit == "ns":
            xlabel = "Duration (ns)"
        elif self.timeunit == "us":
            xlabel = r"Duration ($\mu s$)"
        fig = plt.figure(1, figsize=(8.2, 3.5))
        left, width = 0.0975, 0.23
        bottom, height = 0.17, 0.75
        left_h = left + width + 0.0375
        rect_scatter = [left, bottom, width, height]
        rect_histy = [left_h, bottom, width, height]
        axScatter = fig.add_axes(rect_scatter)
        axHisty = fig.add_axes(rect_histy)
        ######## start bootstrapping ######
        delta_t_range = np.arange(0, np.min(self.T_total), np.min(self.timesteps))
        duration_sampled_set = [np.random.choice(durations, size=len(durations)) for dummy in range(nbootstrap)]
        koff1_sampled_set = []
        koff2_sampled_set = []
        restime_sampled_set = []
        r_squared_sampled_set = []
        for duration_sampled in duration_sampled_set:
            sigma_sampled = cal_sigma(duration_sampled, np.mean(self.num_of_lipids), np.max(self.T_total), delta_t_range)
            hist_values_sampled = np.array([sigma_sampled[delta_t] for delta_t in delta_t_range])
            axHisty.plot(delta_t_range, hist_values_sampled, color="gray", alpha=0.5)
            restime_sampled, koff_sampled, r_squared_sampled, params_sampled = cal_restime_koff(sigma_sampled, initial_guess)
            n_fitted = bi_expo(np.array(delta_t_range), *params_sampled)
            r_squared_sampled = 1 - np.sum((np.nan_to_num(n_fitted) - np.nan_to_num(hist_values_sampled))**2)/np.sum((hist_values_sampled - np.mean(hist_values_sampled))**2)
            ks_sampled = [abs(k) for k in params_sampled[:2]]
            ks_sampled.sort()
            koff1_sampled_set.append(ks_sampled[0])
            koff2_sampled_set.append(ks_sampled[1])
            restime_sampled_set.append(restime_sampled)
            r_squared_sampled_set.append(r_squared_sampled)
        ######## plot original data #########
        sigma = cal_sigma(durations, np.mean(self.num_of_lipids), np.max(self.T_total), delta_t_range)
        x = np.sort(durations)
        y = np.arange(len(x)) + 1
        axScatter.scatter(x[::-1], y, label=label, s=10)
        axScatter.set_xlim(0, x[-1] * 1.1)
        axScatter.legend(loc="upper right", prop={"size": 10}, frameon=False)
        axScatter.set_ylabel("Sorted Index", fontsize=10, weight="bold")
        axScatter.set_xlabel(xlabel, fontsize=10, weight="bold")
        hist_values = np.array([sigma[delta_t] for delta_t in delta_t_range])
        axHisty.scatter(delta_t_range, hist_values, zorder=8, s=3, label="sigma func.")
        axHisty.yaxis.set_label_position("right")
        axHisty.yaxis.tick_right()
        axHisty.set_xlabel(r"$\Delta t$", fontsize=10, weight="bold")
        axHisty.set_ylabel("Probability", fontsize=10, weight="bold")
        axHisty.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axHisty.set_ylim(-0.1, 1.1)
        restime, koff, r_squared, params = cal_restime_koff(sigma, initial_guess)
        n_fitted = bi_expo(np.array(delta_t_range), *params)
        r_squared = 1 - np.sum((np.nan_to_num(n_fitted) - np.nan_to_num(hist_values))**2)/np.sum((hist_values - np.mean(hist_values))**2)
        ks = [abs(k) for k in params[:2]]
        ks.sort()
        axHisty.plot(delta_t_range, n_fitted, 'r--', linewidth=3, zorder=10, label="Fitted biexpo.")
        axHisty.legend(loc="upper right", prop={"size": 10}, frameon=False)
        ######### labels ############
        if self.timeunit == "ns":
            text = "{:18s} = {:.3f} ns$^{{-1}} $\n".format("$k_{{off1}}$", ks[0])
            text += "{:18s} = {:.3f} ns$^{{-1}} $\n".format("$k_{{off2}}$", ks[1])
            text += "{:14s} = {:.4f}\n".format("$R^2$", r_squared)
            text += "{:18s} = {:.3f} ns$^{{-1}}$ ({:3.1f}%)\n".format("$k_{{off1, boot}}$", np.mean(koff1_sampled_set), 100*np.std(koff1_sampled_set)/np.mean(koff1_sampled_set))
            text += "{:18s} = {:.3f} ns$^{{-1}}$ ({:3.1f}%)\n".format("$k_{{off2, boot}}$", np.mean(koff2_sampled_set), 100*np.std(koff2_sampled_set)/np.mean(koff2_sampled_set))
            text += "{:18s} = {:.4f}\n".format("$R^2$$_{{boot, avg}}$", np.mean(r_squared_sampled_set))
        elif self.timeunit == "us":
            text = "{:18s} = {:.3f} $\mu s^{{-1}} $\n".format("$k_{{off1}}$", ks[0])
            text += "{:18s} = {:.3f} $\mu s^{{-1}} $\n".format("$k_{{off2}}$", ks[1])
            text += "{:14s} = {:.4f}\n".format("$R^2$", r_squared)
            text += "{:18s} = {:.3f} $\mu s^{{-1}}$ ({:3.1f}%)\n".format("$k_{{off1, boot}}$", np.mean(koff1_sampled_set), 100*np.std(koff1_sampled_set)/np.mean(koff1_sampled_set))
            text += "{:18s} = {:.3f} $\mu s^{{-1}}$ ({:3.1f}%)\n".format("$k_{{off2, boot}}$", np.mean(koff2_sampled_set), 100*np.std(koff2_sampled_set)/np.mean(koff2_sampled_set))
            text += "{:18s} = {:.4f}\n".format("$R^2$$_{{boot, avg}}$", np.mean(r_squared_sampled_set))
        axHisty.text(1.4, 1.0, text, verticalalignment='top', horizontalalignment='left', transform=axHisty.transAxes, \
                     fontdict={"size": 8, "weight": "bold"})
        plt.savefig(fig_fn, dpi=300)
        plt.close()

        return {"koff": koff, "restime": restime, "sigma": sigma, "params": params, "r_squared": r_squared,
                "koff_b_avg": np.mean(koff1_sampled_set), "koff_b_cv": np.std(koff1_sampled_set)/np.mean(koff1_sampled_set),
                "res_time_b_avg": np.mean(restime_sampled_set), "res_time_b_cv": np.std(restime_sampled_set)/np.mean(restime_sampled_set),
                "r_squared_b_avg": np.mean(r_squared_sampled_set)}
    
    
    def cal_interaction_network(self, save_dir=None, pdb=None, chain=None, pymol_gui=True, save_dataset=True, nbootstrap=10, \
                                radii=None, gen_binding_poses=5, score_weights=None, save_pose_format="pdb"):

        Residue_property_book = {"ARG": "Pos. Charge", "HIS": "Pos. Charge", "LYS": "Pos. Charge",
                                 "ASP": "Neg. Charge", "GLU": "Neg. Charge",
                                 "SER": "Polar", "THR": "Polar", "ASN": "Polar", "GLN": "Polar",
                                 "CYS": "Special", "SEC": "Special", "GLY": "Special", "PRO": "Special",
                                 "ALA": "Hydrophobic", "VAL": "Hydrophobic", "ILE": "Hydrophobic", "LEU": "Hydrophobic",
                                 "MET": "Hydrophobic", "PHE": "Hydrophobic", "TYR": "Hydrophobic", "TRP": "Hydrophobic"}

        MARTINI_CG_radii = {"BB": 0.26, "SC1": 0.23, "SC2": 0.23, "SC3": 0.23}

        if radii == None:
            radii_book = MARTINI_CG_radii
        else:
            radii_book = {**MARTINI_CG_radii, **radii}

        if save_dir == None:
            save_dir = check_dir(self.save_dir, "Binding_Sites_{}".format(self.lipid))
        else:
            save_dir = check_dir(save_dir, "Binding_Sites_{}".format(self.lipid))
        
        residue_interaction_strength = self.dataset["Residence Time"]
        MIN = residue_interaction_strength.quantile(0.15)
        MAX = residue_interaction_strength.quantile(0.99)
        X = (MAX - residue_interaction_strength)/(MAX - MIN)
        residue_interaction_strength = 10 * ((1-np.exp(X))/(1 + np.exp(X))) + 1
        interaction_covariance = np.nan_to_num(self.interaction_covariance)
        #### refined network ###
        ##### determine cov_cutoff #####
        f = open("{}/BindingSites_Info_{}.txt".format(save_dir, self.lipid), "w")
        ##### write out info ######
        reminder = """
# Occupancy: percentage of frames where lipid is in contact with the given residue (0-100%);
# Duration/Residence Time: average length of a continuous interaction of lipid with the given residue (in unit of {timeunit});
# Koff: Koff of lipid with the given residue (in unit of ({timeunit})^(-1));
# Pos. Charge: ARG, HIS, LYS;
# Neg. Charge: ASP, GLU;
# Polar: SER, THR, ASN, GLN;
# Hydrophobic: ALA, VAL, ILE, LEU, MET, PHE, TYR, TRP;
# Special: CYS, SEC, GLY, PRO.
        """.format(**{"timeunit": self.timeunit})
        f.write(reminder)
        f.write("\n")
        binding_site_id = 0
        covariance_network =np.copy(interaction_covariance)
        covariance_network[covariance_network < 0.0] = 0.0
        residue_network_raw = nx.Graph(covariance_network)
        part = community.best_partition(residue_network_raw, weight='weight')
        values = [part.get(node) for node in residue_network_raw.nodes()]
        binding_site_identifiers = np.ones(len(self.residue_set), dtype=int) * 999
        self.interaction_duration_BS = defaultdict(list)
        self.interaction_occupancy_BS = defaultdict(list)
        self.lipid_count_BS = defaultdict(list)
        self.sigmas_BS = {}
        self.params_BS = {}
        BS_restime = np.zeros(len(self.residue_set))
        BS_koff = np.zeros(len(self.residue_set))
        BS_rsquared = np.zeros(len(self.residue_set))
        BS_duration = np.zeros(len(self.residue_set))
        BS_lipidcount = np.zeros(len(self.residue_set))
        BS_occupancy = np.zeros(len(self.residue_set))
        BS_area = defaultdict(list)
        BS_koff_b = np.zeros(len(self.residue_set))
        BS_koff_b_cv = np.zeros(len(self.residue_set))
        BS_restime_b = np.zeros(len(self.residue_set))
        BS_restime_b_cv = np.zeros(len(self.residue_set))
        BS_rsquared_b = np.zeros(len(self.residue_set))
        t_total_max = np.max(self.T_total)
        node_list_set = []
        for value in range(max(values)):
            node_list = [k for k,v in part.items() if v == value]
            if len(node_list) > 2:
                binding_site_identifiers[node_list] = binding_site_id
                node_list_set.append(node_list)
                binding_site_id += 1
        ########### cal site koff and surface area ############
        if len(node_list_set) > 0:
            self._coordinate_pool = [[] for dummy in np.arange(len(node_list_set))]
            for traj_idx, trajfile in enumerate(self.trajfile_list):
                traj = md.load(trajfile, top=self.grofile_list[traj_idx], stride=self.stride)
                if self.dt == None:
                    timestep = traj.timestep/1000000.0 if self.timeunit == "us" else traj.timestep/1000.0
                else:
                    timestep = float(self.dt)
                for idx_protein in np.arange(self.nprot):
                    dist_matrix_prot = self.dist_matrix_set[traj_idx*self.nprot+idx_protein]
                    for binding_site_id, node_list in enumerate(node_list_set):
                        BS_dist_matrix = np.column_stack([dist_matrix_prot[:, node*self.num_of_lipids[traj_idx]:(node+1)*self.num_of_lipids[traj_idx]] \
                                                          for node in node_list])
                        contact_BS_low_prev = [[] for dummy in np.arange(traj.n_frames)]
                        contact_BS_high_prev = [[] for dummy in np.arange(traj.n_frames)]
                        frame_id_set_low, column_id_set_low = np.where(BS_dist_matrix <= self.cutoff[0])
                        frame_id_set_high, column_id_set_high = np.where(BS_dist_matrix <= self.cutoff[1])
                        for frame_id, column_id in zip(frame_id_set_low, column_id_set_low):
                            contact_BS_low_prev[frame_id].append(column_id%self.num_of_lipids[traj_idx])
                        for frame_id, column_id in zip(frame_id_set_high, column_id_set_high):
                            contact_BS_high_prev[frame_id].append(column_id%self.num_of_lipids[traj_idx])
                        contact_BS_low = [np.unique(contact) for contact in contact_BS_low_prev]
                        contact_BS_high = [np.unique(contact) for contact in contact_BS_high_prev]
                        self.interaction_duration_BS[binding_site_id].append(Durations(contact_BS_low, contact_BS_high, timestep).cal_duration())
                        occupancy, lipidcount = cal_interaction_intensity(contact_BS_low)
                        self.interaction_occupancy_BS[binding_site_id].append(occupancy)
                        self.lipid_count_BS[binding_site_id].append(lipidcount)
                        ########### store lipid binding poses ############
                        protein_indices = traj.top.select("protein")[idx_protein*self.natoms_per_protein:(idx_protein+1)*self.natoms_per_protein]
                        ref_coords = np.copy(self._protein_ref.xyz[0])
                        if len(self._protein_ref.top.select("protein and name CA")) > 0:
                            alignment_indices = len(self._protein_ref.top.select("protein and name CA"))
                        else:
                            alignment_indices = np.arange(len(protein_indices))
                        ref_centroid = ref_coords[alignment_indices].mean(axis=0)
                        ref_coords_centered = ref_coords - ref_centroid
                        if gen_binding_poses > 0:
                            position_set = set()
                            lipid_index_set = column_id_set_low % self.num_of_lipids[traj_idx]
                            for framd_id, lipid_index in zip(frame_id_set_low, lipid_index_set):
                                position_set.add((frame_id, lipid_index)) ### remove duplicates ###
                            for frmae_id, lipid_index in position_set:
                                lipid_id = self._lipid_resi_indices_set[traj_idx][lipid_index]
                                lipid_indices = np.sort([atom.index for atom in traj.top.residue(lipid_id).atoms])
                                selected_coords = np.copy(traj.xyz[frame_id, np.hstack([protein_indices, lipid_indices])])
                                selected_coords_centered = selected_coords - selected_coords[alignment_indices].mean(axis=0)
                                rotation_matrix = rmsd.kabsch(selected_coords_centered[alignment_indices], ref_coords_centered[alignment_indices])
                                selected_coords_aligned = np.dot(selected_coords_centered, rotation_matrix) + ref_centroid
                                self._coordinate_pool[binding_site_id].append(selected_coords_aligned) 
                    ### calculate area ###
                    new_xyz = []
                    selected_protein_atom_idx = traj.top.select("protein")[idx_protein*self.natoms_per_protein:(idx_protein+1)*self.natoms_per_protein]
                    for frame in traj.xyz:
                        new_frame = frame[selected_protein_atom_idx]
                        new_xyz.append(new_frame)
                    reduced_frame = traj[0].atom_slice(selected_protein_atom_idx)
                    reduced_top = reduced_frame.top
                    new_traj = md.Trajectory(new_xyz, reduced_top, time=traj.time, unitcell_lengths=traj.unitcell_lengths, unitcell_angles=traj.unitcell_angles)
                    areas = md.shrake_rupley(new_traj, mode='residue', change_radii=radii_book)
                    for binding_site_id, node_list in enumerate(node_list_set):
                        BS_area[binding_site_id].append(areas[:, node_list].sum(axis=1))
        ########### write and plot results ###########
        for binding_site_id in np.arange(len(node_list_set)):
            duration_raw = np.concatenate(self.interaction_duration_BS[binding_site_id])
            mask = (binding_site_identifiers == binding_site_id)
            bootstrap_results = self.bootstrap(duration_raw, "BS id: {}".format(binding_site_id), "{}/BS_koff_id{}.pdf".format(save_dir, binding_site_id), nbootstrap=nbootstrap)
            self.sigmas_BS[binding_site_id] = bootstrap_results["sigma"]
            self.params_BS[binding_site_id] = bootstrap_results["params"]
            BS_restime[mask] = bootstrap_results["restime"]
            BS_koff[mask] = bootstrap_results["koff"]
            BS_rsquared[mask] = bootstrap_results["r_squared"]
            BS_koff_b[mask] = bootstrap_results["koff_b_avg"]
            BS_koff_b_cv[mask] = bootstrap_results["koff_b_cv"]
            BS_restime_b[mask] = bootstrap_results["res_time_b_avg"]
            BS_restime_b_cv[mask] = bootstrap_results["res_time_b_cv"]
            BS_rsquared_b[mask] = bootstrap_results["r_squared_b_avg"]
            ############# write results ###############
            f.write("# Binding site {}\n".format(binding_site_id))
            BS_restime[mask] = bootstrap_results["restime"] if bootstrap_results["restime"] <= t_total_max else t_total_max
            if bootstrap_results["restime"] <= t_total_max:
                f.write("{:20s} {:10.3f} {:5s}   R squared: {:7.4f}\n".format(" BS Residence Time:", bootstrap_results["restime"], self.timeunit, bootstrap_results["r_squared"]))
            else:
                f.write("{:20s} {:10.3f} {:5s}** R squared: {:7.4f}\n".format(" BS Residence Time:", t_total_max, self.timeunit, bootstrap_results["r_squared"]))
            f.write("{:20s} {:10.3f}\n".format(" BS koff:", bootstrap_results["koff"]))
            f.write("{:20s} {:10.3f} +- {:10.3f}\n".format(" BS koff Bootstrap:", bootstrap_results["koff_b_avg"], bootstrap_results["koff_b_cv"]))
            duration = np.mean(np.concatenate(self.interaction_duration_BS[binding_site_id]))
            BS_duration[mask] = duration
            f.write("{:20s} {:10.3f} {:5s}\n".format(" BS Duration:", duration, self.timeunit))
            occupancy = np.mean(self.interaction_occupancy_BS[binding_site_id])
            BS_occupancy[mask] = occupancy
            f.write("{:20s} {:10.3f} %\n".format(" BS Lipid Occupancy:", occupancy))
            lipidcount = np.mean(self.lipid_count_BS[binding_site_id])
            BS_lipidcount[mask] = lipidcount
            f.write("{:20s} {:10.3f}\n".format(" BS Lipid Count:", lipidcount))
            f.write("{:20s} {:10.3f} +- {:10.3f}\n".format(" BS Surface Area:", np.concatenate(BS_area[binding_site_id]).mean(), np.concatenate(BS_area[binding_site_id]).std()))
            res_stats = {"Pos. Charge": 0, "Neg. Charge": 0, "Polar": 0, "Special": 0, "Hydrophobic": 0}
            for residue in self.residue_set[mask]:
                res_stats[Residue_property_book[residue[-3:]]] += 1
            BS_num_resi = np.sum(mask)
            f.write("{:20s} {:10s}\n".format(" Pos. Charge:", "/".join([str(res_stats["Pos. Charge"]), str(BS_num_resi)])))
            f.write("{:20s} {:10s}\n".format(" Neg. Charge:", "/".join([str(res_stats["Neg. Charge"]), str(BS_num_resi)])))
            f.write("{:20s} {:10s}\n".format(" Polar:", "/".join([str(res_stats["Polar"]), str(BS_num_resi)])))
            f.write("{:20s} {:10s}\n".format(" Hydrophobic:", "/".join([str(res_stats["Hydrophobic"]), str(BS_num_resi)])))
            f.write("{:20s} {:10s}\n".format(" Special:", "/".join([str(res_stats["Special"]), str(BS_num_resi)])))
            f.write("{:^9s}{:^9s}{:^13s}{:^11s}{:^10s}{:^10s}{:^10s}{:^13s}{:^10s}{:^10s}\n".format("Residue", "Duration", "Duration std", \
                    "Res. Time", "R squared", "Occupancy", "Occu. std", "Lipid Count", "L. C. std", "Koff"))
            for residue in self.residue_set[mask]:
                f.write("{Residue:^9s}{Duration:^9.3f}{Duration_std:^13.3f}{Residence Time:^11.3f}{R squared:^10.4f}{Occupancy:^10.3f}{Occupancy_std:^10.3f}{LipidCount:^13.3f}{LipidCount_std:^10.3f}{Koff:^10.4f}\n".format(\
                        **self.dataset[self.dataset["Residue"]==residue].to_dict("records")[0] ))
            f.write("\n")
            f.write("\n")
        f.close()

        ######################## plot area stats ##########################
        bs_id_set = []
        bs_area_set = []
        for binding_site_id in BS_area.keys():
            bs_area_set.append(np.concatenate(BS_area[binding_site_id]))
            bs_id_set.append([binding_site_id for dummy in np.arange(len(np.concatenate(BS_area[binding_site_id])))])
        d_area = pd.DataFrame({"BS id": np.concatenate(bs_id_set), "Area (nm^2)": np.concatenate(bs_area_set)})
        plt.rcParams["font.size"] = 8
        plt.rcParams["font.weight"] = "bold"
        if len(BS_area.keys()) <= 8:
            fig, ax = plt.subplots(figsize=(4.5, 2.8))
        elif len(BS_area.keys()) > 8 and len(BS_area.keys()) <= 15:
            fig, ax = plt.subplots(figsize=(6.5, 2.8))
        else:
            fig, ax = plt.subplots(figsize=(9.5, 3))
        sns.violinplot(x="BS id", y="Area (nm^2)", data=d_area, palette="Set3", bw=.2, cut=1, linewidth=1, ax=ax)
        ax.set_xlabel("BS id", fontsize=8, weight="bold")
        ax.set_ylabel(r"Surface Area (unit$^2$)", fontsize=8, weight="bold")
        ax.set_title("{} Binding Site Surface Area".format(self.lipid), fontsize=8, weight="bold")
        plt.tight_layout()
        plt.savefig("{}/BS_surface_area.pdf".format(save_dir), dpi=300)
        plt.close()

        ################ update dataset ########################
        self.dataset["Binding site"]  = binding_site_identifiers
        self.dataset["BS Residence Time"] = BS_restime
        self.dataset["BS koff"] = BS_koff
        self.dataset["BS Duration"] = BS_duration
        self.dataset["BS Occupancy"] = BS_occupancy
        self.dataset["BS LipidCount"] = BS_lipidcount
        self.dataset["BS R squared"] = BS_rsquared
        self.dataset["BS Residence Time_boot"] = BS_restime_b
        self.dataset["BS Residence Time_boot_cv"] = BS_restime_b_cv
        self.dataset["BS koff_boot"] = BS_koff_b
        self.dataset["BS koff_boot_cv"] = BS_koff_b_cv
        self.dataset["BS R squared_boot"] = BS_rsquared_b
        self.dataset.to_csv("{}/Interactions_{}.csv".format(self.save_dir, self.lipid), index=False)
        ################ save dataset ###################
        if save_dataset:
            dataset_dir = check_dir(self.save_dir, "Dataset")
            with open("{}/BS_interaction_duration_{}.pickle".format(dataset_dir, self.lipid), "wb") as f:
                pickle.dump(self.interaction_duration_BS, f, 2)
            with open("{}/BS_sigmas_{}.pickle".format(dataset_dir, self.lipid), "wb") as f:
                pickle.dump(self.sigmas_BS, f, 2)
            with open("{}/BS_curve_fitting_params_{}.pickle".format(dataset_dir, self.lipid), "wb") as f:
                pickle.dump(self.params_BS, f, 2)
            with open("{}/BS_surface_area_{}.pickle".format(dataset_dir, self.lipid), "wb") as f:
                pickle.dump(BS_area, f, 2)
        
        ################## generate binding poses ################
        if gen_binding_poses > 0:
            coords_save_dir = check_dir(save_dir, "Binding_Poses")
            lipid_atom_map = {atom.index:atom.name for atom in self._lipid_ref.top.atoms}
            weights = {name:1 for index, name in lipid_atom_map.items()}
            if score_weights != None:
                for name, weight in score_weights.items():
                    if name in weights:
                        weights[name] *= weight
            
            binding_site_id_set = np.arange(len(self._coordinate_pool))
            for binding_site_id in binding_site_id_set:
                num_of_poses = gen_binding_poses if gen_binding_poses <= len(self._coordinate_pool[binding_site_id]) \
                    else len(self._coordinate_pool[binding_site_id])
                self._lipid_pool_per_atom = [np.array(self._coordinate_pool[binding_site_id])[:, idx-self._lipid_ref.n_atoms, :] \
                                             for idx in np.arange(self._lipid_ref.n_atoms)]
                self._kde_funcs = {}
                for atom_idx in np.arange(self._lipid_ref.n_atoms):
                    self._kde_funcs[atom_idx] = stats.gaussian_kde(self._lipid_pool_per_atom[atom_idx].T)
                ### evaluate binding poses ###
                failed = False
                try: 
                    scores = np.sum([weights[lipid_atom_map[idx]] * self._kde_funcs[idx].evaluate(self._lipid_pool_per_atom[idx].T) \
                                     for idx in np.arange(self._lipid_ref.n_atoms)], axis=0)
                except np.linalg.LinAlgError:
                    failed = True
                ### write binding poses ###
                if failed:
                    with open("{}/Errors.txt".format(coords_save_dir), "a") as f:
                        f.write("Error! PDF calculation showed error at Binding Site {}, probably due to insufficient number of binding events. \n")
                    selected_indices = np.arange(num_of_poses)
                else:
                    selected_indices = np.argsort(scores)[::-1][:num_of_poses]
                ###############################
                joined_top = self._protein_ref.top.join(self._lipid_ref.top)
                for pose_id in np.arange(num_of_poses, dtype=int):
                    new_traj = md.Trajectory(self._coordinate_pool[binding_site_id][selected_indices[pose_id]], joined_top, \
                                             time=self._protein_ref.time, unitcell_lengths=self._protein_ref.unitcell_lengths, \
                                             unitcell_angles=self._protein_ref.unitcell_angles)
                    new_traj.save("{}/BSid{}_No{}.{}".format(coords_save_dir, binding_site_id, pose_id, save_pose_format))
        
        ######################################################################
        ###### show binding site residues with scaled spheres in pymol #######
        ######################################################################
        if pdb != None:
            ############ check if pdb has a path to it ##########
            pdb_new_loc = os.path.join(self.save_dir, os.path.basename(pdb))
            copyfile(pdb, pdb_new_loc)
            ########### write out a pymol pml file ###############
            Selection = "tmp and chain {}".format(chain) if chain != None else "tmp"
            binding_site_id += 1
            text = """
import pandas as pd
import numpy as np
import pymol
from pymol import cmd
pymol.finish_launching()

dataset = pd.read_csv("{HOME_DIR}/Interactions_{LIPID}.csv")
residue_set = np.array(dataset["Residue"].tolist())
binding_site_id = {BINDING_SITE_ID}
binding_site_identifiers = np.array(dataset["Binding site"].tolist())

######### calculate scale ###############
residue_interaction_strength = dataset["Residence Time"]
MIN = residue_interaction_strength.quantile(0.15)
MAX = residue_interaction_strength.quantile(0.99)
X = (MAX - residue_interaction_strength)/(MAX - MIN)
SCALES = 1.5 * ((1-np.exp(X))/(1 + np.exp(X))) + 1.0

######################################
######## some pymol settings #########
cmd.set("cartoon_oval_length", 1.0)
cmd.set("cartoon_oval_width", 0.3)
cmd.set("cartoon_color", "white")
cmd.set("stick_radius", 0.35)
##################################
cmd.load("{PDB}", "tmp")
cmd.extract("Prot_{LIPID}", "{SELECTION}")
prefix = "Prot_{LIPID}"
cmd.delete("tmp")
cmd.hide("everything")
cmd.show("cartoon", prefix)
cmd.center(prefix)
cmd.orient(prefix)
colors = np.array([np.random.choice(np.arange(256, dtype=float), size=3) for dummy in range(binding_site_id)])
colors /= 255.0

            """.format(**{"HOME_DIR": self.save_dir, "LIPID": self.lipid, "BINDING_SITE_ID": binding_site_id, "PDB": pdb_new_loc, "SELECTION": Selection})
            text += r"""
for bs_id in np.arange(binding_site_id):
    selected_indices = np.where(binding_site_identifiers == bs_id)[0]
    selected_residues = residue_set[selected_indices]
    selected_resids = [residue[:-3] for residue in selected_residues]
    selected_resns = [residue[-3:] for residue in selected_residues]
    cmd.set_color("tmp_{}".format(bs_id), list(colors[bs_id]))
    for selected_index, selected_resid, selected_resn in zip(selected_indices, selected_resids, selected_resns):
        cmd.select("BS_{}_{}{}".format(bs_id, selected_resid, selected_resn), "{} and resid {} and (not name C+O+N)".format(prefix, selected_resid))
        cmd.show("spheres", "BS_{}_{}{}".format(bs_id, selected_resid, selected_resn))
        cmd.set("sphere_scale", SCALES[selected_index], selection="BS_{}_{}{}".format(bs_id, selected_resid, selected_resn))
        cmd.color("tmp_{}".format(bs_id), "BS_{}_{}{}".format(bs_id, selected_resid, selected_resn))
    cmd.group("BS_{}".format(bs_id), "BS_{}_*".format(bs_id))
            """
            with open("{}/show_binding_sites_info.py".format(self.save_dir), "w") as f:
                f.write(text)

            ##################  Launch a pymol session  #######################
            if pymol_gui:
                import pymol
                from pymol import cmd
                pymol.finish_launching(['pymol', '-q'])
                ##### do some pymol settings #####
                residue_interaction_strength = self.dataset["Residence Time"]
                MIN = residue_interaction_strength.quantile(0.15)
                MAX = residue_interaction_strength.quantile(0.99)
                X = (MAX - residue_interaction_strength)/(MAX - MIN)
                SCALES = 1.5 * ((1-np.exp(X))/(1 + np.exp(X))) + 1.0
                ##### do some pymol settings #####
                cmd.set("cartoon_oval_length", 1.0)
                cmd.set("cartoon_oval_width", 0.3)
                cmd.set("cartoon_color", "white")
                cmd.set("stick_radius", 0.35)
                ##################################
                cmd.load(pdb_new_loc, "tmp")
                cmd.extract("Prot_{}".format(self.lipid), Selection)
                prefix = "Prot_{}".format(self.lipid)
                cmd.delete("tmp")
                cmd.hide("everything")
                cmd.show("cartoon", prefix)
                cmd.center(prefix)
                cmd.orient(prefix)
                colors = np.array([np.random.choice(np.arange(256, dtype=float), size=3) for dummy in range(binding_site_id)])
                colors /= 255.0
                for bs_id in np.arange(binding_site_id):
                    selected_indices = np.where(binding_site_identifiers == bs_id)[0]
                    selected_residues = self.residue_set[selected_indices]
                    selected_resids = [residue[:-3] for residue in selected_residues]
                    selected_resns = [residue[-3:] for residue in selected_residues]
                    cmd.set_color("tmp_{}".format(bs_id), list(colors[bs_id]))
                    for selected_index, selected_resid, selected_resn in zip(selected_indices, selected_resids, selected_resns):
                        cmd.select("{}_BS_{}_{}{}".format(self.lipid, bs_id, selected_resid, selected_resn), "Prot and resid {} and (not name C+O+N)".format(selected_resid))
                        cmd.show("spheres", "{}_BS_{}_{}{}".format(self.lipid, bs_id, selected_resid, selected_resn))
                        cmd.set("sphere_scale", SCALES[selected_index], selection="{}_BS_{}_{}{}".format(self.lipid, bs_id, selected_resid, selected_resn))
                        cmd.color("tmp_{}".format(bs_id), "{}_BS_{}_{}{}".format(self.lipid, bs_id, selected_resid, selected_resn))
                    cmd.group("{}_BS_{}".format(self.lipid, bs_id), "{}_BS_{}_*".format(self.lipid, bs_id))
        return


    def plot_interactions(self, item="Duration",  save_dir=None, letter_map=None):

        if save_dir == None:
            save_dir = check_dir(self.save_dir, "Figures_{}".format(self.lipid))
        else:
            save_dir = check_dir(save_dir, "Figures_{}".format(self.lipid))
        ### single-letter dictionary ###
        single_letter = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
                         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
                         'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
        if letter_map != None:
            single_letter.update(letter_map)
        
        plt.rcParams["font.size"] = 8
        plt.rcParams["font.weight"] = "bold"

        data = self.dataset[item]
        resi = np.array([int(residue[:-3]) for residue in self.residue_set])
        width = 1
        sns.set_style("ticks", {'xtick.major.size': 5.0, 'ytick.major.size': 5.0})
        if item == "Residence Time":
            if len(data) <= 500:
                fig = plt.figure(figsize=(5.5, 5))
            elif len(data) > 500 and len(data) <= 1500:
                fig = plt.figure(figsize=(7.5, 5))
            else:
                fig = plt.figure(figsize=(9, 6))
            ax_R2 = fig.add_axes([0.18, 0.79, 0.75, 0.10])
            ax_capped = fig.add_axes([0.18, 0.71, 0.75, 0.05])
            ax_data = fig.add_axes([0.18, 0.50, 0.75, 0.18])
            ax_boot = fig.add_axes([0.18, 0.22, 0.75, 0.18])
            ax_boot_cv = fig.add_axes([0.18, 0.08, 0.75, 0.10])
            ax_boot.xaxis.tick_top()
            ax_boot.invert_yaxis()
            ax_boot_cv.invert_yaxis()
            for ax in [ax_data, ax_capped, ax_R2, ax_boot, ax_boot_cv]:
                ax.yaxis.set_ticks_position('left')
                ax.spines['right'].set_visible(False)
            for ax in [ax_capped, ax_R2, ax_boot_cv]:
                ax.xaxis.set_ticks_position('none')
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.set_xticklabels([])
            ax_data.spines['top'].set_visible(False)
            ax_boot.spines['bottom'].set_visible(False)
            if len(data) > 1000:
                ax_data.xaxis.set_major_locator(MultipleLocator(200))
                ax_data.xaxis.set_minor_locator(MultipleLocator(50))
                ax_boot.xaxis.set_major_locator(MultipleLocator(200))
                ax_boot.xaxis.set_minor_locator(MultipleLocator(50))
            elif len(data) <= 1000:
                ax_data.xaxis.set_major_locator(MultipleLocator(100))
                ax_data.xaxis.set_minor_locator(MultipleLocator(10))
                ax_boot.xaxis.set_major_locator(MultipleLocator(100))
                ax_boot.xaxis.set_minor_locator(MultipleLocator(10))
            if self.timeunit == "ns":
                timeunit = " (ns) "
            elif self.timeunit == "us":
                timeunit = r" ($\mu s$)"
            ax_data.bar(resi, data, width, linewidth=0, color="#F75C03")
            ax_data.set_ylabel("Res. Time {}".format(timeunit), fontsize=8, weight="bold", va="center")
            ax_data.set_xlabel("Residue Index", fontsize=8, weight="bold")
            ax_capped.plot(resi, self.dataset["Capped"]*1, linewidth=0, marker="+", markerfacecolor="#38040E", \
                           markeredgecolor="#38040E", markersize=2)
            ax_capped.set_ylim(0.9, 1.1)
            ax_capped.set_yticks([1.0])
            ax_capped.set_yticklabels(["Capped"], fontsize=8, weight="bold")
            ax_capped.set_xlim(ax_data.get_xlim())
            mask = self.dataset["R squared"] > 0
            ax_R2.plot(resi[mask], self.dataset["R squared"][mask], linewidth=0, marker="+", markerfacecolor="#0FA3B1", markeredgecolor="#0FA3B1", \
                       markersize=2)
            ax_R2.set_xlim(ax_data.get_xlim())
            ax_R2.set_ylabel(r"$R^2$", fontsize=8, weight="bold", va="center")
            ax_R2.set_title("{} {}".format(self.lipid, item), fontsize=8, weight="bold")

            ax_boot.bar(resi, self.dataset["Residence Time_boot"], width, linewidth=0, color="#F75C03")
            ax_boot.set_xlim(ax_data.get_xlim())
            ax_boot.set_ylabel("Res. Time \n Boot. {}".format(timeunit), fontsize=8, weight="bold", va="center")
            ax_boot.set_xticklabels([])
            mask = self.dataset["R squared_boot"] > 0
            mask = self.dataset["Residence Time_boot_cv"] > 0
            ax_boot_cv.plot(resi[mask], self.dataset["Residence Time_boot_cv"][mask], linewidth=0, marker="+", markerfacecolor="#0FA3B1", markeredgecolor="#F7B538",
                            markersize=2)
            ax_boot_cv.set_ylabel("Coef. Var.", fontsize=8, weight="bold", va="center")
            ax_boot_cv.set_xlim(ax_data.get_xlim())
            for ax in [ax_data, ax_capped, ax_R2, ax_boot, ax_boot_cv]:
                ax.yaxis.set_label_coords(-0.15, 0.5, transform=ax.transAxes)
            plt.savefig("{}/{}_{}.pdf".format(save_dir, "_".join(item.split()), self.lipid), dpi=300)
            plt.close()
        
            ###### logomater #####
            SL_resn = [single_letter[residue[-3:]] for residue in self.residue_set]
            df = pd.DataFrame({"Resid": resi, "Resn": SL_resn, "Data": data})
            matrix = df.pivot(index="Resid", columns='Resn', values="Data").fillna(0)
            n_rows = 1+resi[-1]//100
            length = 100 - resi[0]
            fig, axes = plt.subplots(n_rows, 1, figsize=(4.5, 1.3*n_rows), sharey=True)
            plt.subplots_adjust(hspace=0.5)
            for idx, ax in enumerate(axes):
                if idx == (n_rows - 1):
                    logomaker.Logo(matrix[idx*length:], color_scheme="chemistry", ax=ax)
                    ax.set_xlabel("Residue Index", fontsize=8, weight="bold")
                else:
                    logomaker.Logo(matrix[idx*length:(idx+1)*length], color_scheme="chemistry", ax=ax)
                ax.xaxis.set_major_locator(MultipleLocator(20))
                ax.xaxis.set_minor_locator(MultipleLocator(1))
                ax.set_xlim(idx*100, (idx+1)*100)
                ax.set_ylim(0, data.max()*1.1)
                ax.set_ylabel("Res. Time {}".format(timeunit), fontsize=8, weight="bold", va="center")
                for label in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
                    plt.setp(label, size=8, weight="bold")
            plt.savefig("{}/{}_logo_{}.pdf".format(save_dir, "_".join(item.split()), self.lipid), dpi=300)
            

        else:
            fig, ax = plt.subplots(1, 1, figsize=(4.5,2.8))
            ax.bar(resi, data, width, linewidth=0, color=sns.xkcd_rgb["red"])
            sns.despine(fig, top=True, right=True, trim=False)
            if len(data) > 1000:
                ax.xaxis.set_major_locator(MultipleLocator(200))
                ax.xaxis.set_minor_locator(MultipleLocator(50))
            elif len(data) <= 1000:
                ax.xaxis.set_major_locator(MultipleLocator(100))
                ax.xaxis.set_minor_locator(MultipleLocator(10))
            ax.set_xlabel("Residue Index", fontsize=8, weight="bold")
            if self.timeunit == "ns":
                timeunit = " (ns) "
            elif self.timeunit == "us":
                timeunit = r" ($\mu s$)"
            if item == "Duration":
                ylabel = item + timeunit
            elif item == "Occupancy":
                ylabel = item + " 100% "
            elif item == "LipidCount":
                ylabel = "Num. of Lipids"
            ax.set_ylabel(ylabel, fontsize=8, weight="bold")
            for label in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
                plt.setp(label, fontsize=8, weight="bold")
            ax.set_title("{} {}".format(self.lipid, item), fontsize=8, weight="bold")
            plt.tight_layout()
            plt.savefig("{}/{}_{}.pdf".format(save_dir, "_".join(item.split()), self.lipid), dpi=300)
            plt.close()
            
            ###### logomater #####
            SL_resn = [single_letter[residue[-3:]] for residue in self.residue_set]
            df = pd.DataFrame({"Resid": resi, "Resn": SL_resn, "Data": data})
            matrix = df.pivot(index="Resid", columns='Resn', values="Data").fillna(0)
            n_rows = 1 + resi[-1]//100 - resi[0]//100
            start = (resi[0]//100)*100
            length = start + 100 - resi[0]
            fig, axes = plt.subplots(n_rows, 1, figsize=(4.5, 1.3*n_rows), sharey=True)
            plt.subplots_adjust(hspace=0.5)
            for idx, ax in enumerate(axes):
                if idx == (n_rows - 1):
                    logomaker.Logo(matrix[(idx-1)*100 + length:], color_scheme="chemistry", ax=ax)
                    ax.set_xlabel("Residue Index", fontsize=8, weight="bold")
                elif idx == 0:
                    logomaker.Logo(matrix[:length], color_scheme="chemistry", ax=ax)
                else:
                    logomaker.Logo(matrix[(idx-1)*100+length:idx*100+length], color_scheme="chemistry", ax=ax)
                ax.xaxis.set_major_locator(MultipleLocator(20))
                ax.xaxis.set_minor_locator(MultipleLocator(1))
                ax.set_xlim(idx*100+start, (idx+1)*100+start)
                ax.set_ylim(0, data.max()*1.05)
                ax.set_ylabel(ylabel, fontsize=8, weight="bold", va="center")
                for label in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
                    plt.setp(label, fontsize=8, weight="bold")
            plt.savefig("{}/{}_logo_{}.pdf".format(save_dir, "_".join(item.split()), self.lipid), dpi=300)
            
        return


    def write_to_pdb(self, item, save_dir=None):

        if save_dir == None:
            save_dir = check_dir(self.save_dir, "Coordinates_{}".format(self.lipid))
        else:
            save_dir = check_dir(save_dir, "Coordinates_{}".format(self.lipid))

        ##### load coords ######
        tmp_traj = md.load(self.trajfile_list[0], top=self.grofile_list[0], stride=self.stride)
        data = self.dataset[item]
        coords = tmp_traj.xyz[0][self.prot_atom_indices]
        table, _ = tmp_traj.top.to_dataframe()
        atom_idx_set = table.serial[self.prot_atom_indices]
        resid_set = table.resSeq[self.prot_atom_indices] + self.resi_offset
        atom_name_set = table.name[self.prot_atom_indices]
        resn_set = table.resName[self.prot_atom_indices]
        chainID = [chr(65+int(idx)) for idx in table.chainID]
        data_expanded = np.zeros(len(self.prot_atom_indices))
        for idx_set, value in zip(self.protein_residue_indices_set[0], data):
            data_expanded[np.array(idx_set)] = value
        ######## write out coords ###########
        fn = "{}/Coords_{}.pdb".format(save_dir, "_".join(item.split()))
        with open(fn, "w") as f:
            for idx in np.arange(len(self.prot_atom_indices)):
                f.write("{HEADER:6s}{ATOM_ID:5d} {ATOM_NAME:^4s}{SPARE:1s}{RESN:3s} {CHAIN_ID:1s}{RESI:4d}{SPARE:1s}   {COORDX:8.3f}{COORDY:8.3f}{COORDZ:8.3f}{OCCUP:6.2f}{BFACTOR:6.2f}\n".format(**{\
                        "HEADER": "ATOM",
                        "ATOM_ID": atom_idx_set[idx],
                        "ATOM_NAME": atom_name_set[idx],
                        "SPARE": "",
                        "RESN": resn_set[idx],
                        "CHAIN_ID": chainID[idx],
                        "RESI": resid_set[idx],
                        "COORDX": coords[idx, 0] * 10,
                        "COORDY": coords[idx, 1] * 10,
                        "COORDZ": coords[idx, 2] * 10,
                        "OCCUP": 1.0,
                        "BFACTOR": data_expanded[idx]}))
            f.write("TER")
        return


######################################################
########### Load params and do calculation ###########
######################################################

if __name__ == '__main__':

    trajfile_list = args.f
    grofile_list = args.c
    lipid_set = args.lipids
    cutoff = [float(data) for data in args.cutoffs]
    save_dir = check_dir(args.save_dir)
    ######################### process resi_list ##########################
    resi_list = []
    if len(args.resi_list) > 0:
        for item in args.resi_list:
            if "-" in item:
                item_list = item.split("-")
                resi_list.append(np.arange(int(item_list[0]), int(item_list[-1])+1))
            else:
                resi_list.append(int(item))
        resi_list = np.hstack(resi_list)
    #######################################################################
    ######## write a backup file of params for reproducibility ############
    fn = os.path.join(save_dir, "pylipid_backup_{}.txt".format(datetime.datetime.now().strftime("%Y_%m_%d_%H%M")))
    with open(fn, "w") as f:
        f.write("##### Record params for reproducibility #####\n")
        f.write("python {}\n".format(" ".join(sys.argv)))
    #######################################################################
    ############################ change of radii ##########################
    ##### mdtraj default radii:
    ##### https://github.com/mdtraj/mdtraj/blob/b28df2cd6e5c35fa006fe3c24728857880793abb/mdtraj/geometry/sasa.py#L56
    if args.radii == None:
        radii_book = None
    else:
        radii_book = {}
        for item in args.radii:
            radius = item.split(":")
            radii_book[radius[0]] = float(radius[1])
    #######################################################################
    ################# score weight for kde calculation ####################
    if args.score_weights == None:
        score_weights = None
    else:
        score_weights = {}
        for item in args.score_weights:
            weight = item.split(":")
            score_weights[weight[0]] = float(weight[1])
    #######################################################################
    ################# map three letter to single letter ###################
    letter_map = None
    if args.letter_map != None:
        letter_map = {}
        for item in args.letter_map:
            letter_map[item.split(":")[0]] = item.split(":")[1]
    #######################################################################
    for lipid in lipid_set:
        li = LipidInteraction(trajfile_list, grofile_list, stride=args.stride, dt=args.dt, cutoff=cutoff, lipid=lipid, \
                              lipid_atoms=args.lipid_atoms, nprot=args.nprot, timeunit=args.tu, resi_offset=int(args.resi_offset), \
                              resi_list=resi_list, save_dir=args.save_dir)
        li.cal_interactions(save_dataset=args.save_dataset, nbootstrap=int(args.nbootstrap))
        li.plot_interactions(item="Duration", letter_map=letter_map)
        li.plot_interactions(item="Residence Time", letter_map=letter_map)
        li.plot_interactions(item="Occupancy", letter_map=letter_map)
        li.plot_interactions(item="LipidCount", letter_map=letter_map)
        li.write_to_pdb(item="Duration")
        li.write_to_pdb(item="Residence Time")
        li.write_to_pdb(item="Occupancy")
        li.write_to_pdb(item="LipidCount")
        li.cal_interaction_network(pdb=args.pdb, chain=args.chain, save_dataset=args.save_dataset, \
                                   pymol_gui=args.pymol_gui, radii=radii_book, gen_binding_poses=int(args.gen_binding_poses), \
                                   score_weights=score_weights, save_pose_format=args.save_pose_format)


