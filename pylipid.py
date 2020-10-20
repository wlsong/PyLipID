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
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from sklearn.decomposition import PCA
import community
import warnings
from shutil import copyfile
import datetime
from itertools import product
import logomaker
import re
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
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
                    that is used for the calculation of binding site surface area. Values need to be in the unit of nm. Supported syntax is \
                    BB:0.26, which defines the radius of bead BB as 0.26 nm, or CA:0.12 which defines the radius of atom CA as 0.12 nm. For \
                    atomistic simulations, the default radii are taken from \
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
parser.add_argument("-chain_breaks", nargs="+", default=[], metavar="100 281 420", help="Start a new chain at the X-th residue (starting at 1) in \
                    the trajectory topology. This identifier is independent of the residue index but checks the residue order in the topology. \
                    Multiple chain breaks are supported. This option is useful when the simulation system contains \
                    multiple differnt chains, or users want to see the difference between chains even if these chains are identical. Using this flag \
                    will generate seperate figures for each of the chains. But the binding site detection will still treat the proteins in the \
                    system collectively, i.e. those binding sites that cover at multiple chains will be identified.")
parser.add_argument("-nbootstrap", default=10, metavar=10, help="The number of samples for bootstrapping the calcultion of koff. \
                    The default is 10. The larger the number, the more time-consuming the calculation will be. The closer the bootstrapped \
                    residence time/koffs are to the original values, the more reliable those original values are. The bootstrapped results \
                    are ploted in each of the koff plots and plotted apposed to the original values in the figure showing residence time. ")
parser.add_argument("-save_dataset", nargs="?", default=True, const=True, metavar="True", help="Save dataset in Pickle. Default is True")
parser.add_argument("-gen_binding_poses", default=5, metavar=5, help="The num. of top-scored lipid binding poses to be generated for each binding \
                    site. The default is 5. A scoring function is generated for each binding site based on the sum of the probability density function of each atom/bead \
                    the lipid molecule. Score = sum(PDF(atom_i) * Weight(atom_i)) for atom_i in the lipid molecule. The weight function Weight(atom_i) \
                    is specified by the flag -score_weights.")
parser.add_argument("-save_pose_format", default="gro", metavar="gro", help="The format the generated lipid binding poses are written into. This function \
                    is carried out by mdtraj.save(), hence supports the formats that are included by mdtraj. ")
parser.add_argument("-score_weights", nargs="+", default=None, metavar="PO4:1 C1:1", help="The weight of each of the lipid atom/bead contributes to the scoring function. \
                    Top-rated lipid binding poses can be generated based on users' specification. The bounds poses of each binding site are scored based \
                    on the scoring function Score = sum(PDF(atom_i) * Weight(atom_i)) for atom_i in the lipid molecule.")
parser.add_argument("-letter_map", nargs="+", default=None, metavar="ARG:K GLY:G", help="Map the three-letter amino acids to one letter. This map is \
                    used in making logomaker figures (https://logomaker.readthedocs.io/en/latest/). The common 20 amino acids are defined \
                    by this script. Users need to use this flag to define maps for uncommon amino acids in their systems.")
parser.add_argument("-pdb", default=None, metavar="None", help="Provide a PDB structure onto which the binding site information will be mapped. \
                    Using this flag will generate a 'show_binding_site_info.py' file in the -save_dir directory, which allows users to check the \
                    mapped binding site information in PyMol. Users can run the generated script by 'python show_binding_site_info.py' \
                    to open such a PyMol session.")

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


def cal_sigma(durations, num_of_contacts, T_total, delta_t_range):
    sigma = {}
    for delta_t in delta_t_range:
        if delta_t == 0:
            sigma[delta_t] = 1
            sigma0 = float(sum([restime - delta_t for restime in durations if restime >= delta_t])) / ((T_total - delta_t) * num_of_contacts)
        else:
            try:
                sigma[delta_t] = float(sum([restime - delta_t for restime in durations if restime >= delta_t])) / ((T_total - delta_t) * num_of_contacts * sigma0)
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
        self.contact_residues_high = defaultdict(list)
        self.contact_residues_low = defaultdict(list)
        self.stride = int(stride)
        self.resi_offset = resi_offset
        self.resi_list = resi_list
        self.residue_set = []
        self._protein_ref = None
        self._lipid_ref = None

        return


    def _get_traj_stats(self, traj, lipid, lipid_atoms):

        lipid_atom_indices = traj.top.select("resn {}".format(self.lipid))
        lipid_resi_indices = set()
        for atom in lipid_atom_indices:
            lipid_resi_indices.add(traj.top.atom(atom).residue.index)
        num_of_lipids = len(lipid_resi_indices)
        lipid_resi_indices = list(lipid_resi_indices)
        lipid_resi_indices.sort()
        lipid_resi_indices_original = lipid_resi_indices

        if self._lipid_ref == None:
            one_lipid_indices = []
            for lipid_id in np.sort(traj.top.select("resn {}".format(self.lipid))):
                if len(one_lipid_indices) == 0:
                    one_lipid_indices.append(lipid_id)
                elif traj.top.atom(lipid_id).residue.index != traj.top.atom(one_lipid_indices[-1]).residue.index:
                    break
                else:
                    one_lipid_indices.append(lipid_id)
            self._lipid_ref = traj[0].atom_slice(np.unique(one_lipid_indices))

        if lipid_atoms != None:
            lipid_haystack = get_atom_index_for_lipid(lipid, traj, part=lipid_atoms)
            selected_atom_indices = np.hstack([traj.top.select("protein"), lipid_haystack])
            new_xyz = [frame[selected_atom_indices] for frame in traj.xyz]
            reduced_frame = traj[0].atom_slice(selected_atom_indices)
            reduced_top = reduced_frame.top
            new_traj = md.Trajectory(new_xyz, reduced_top, time=traj.time, unitcell_lengths=traj.unitcell_lengths, \
                                     unitcell_angles=traj.unitcell_angles)
            lipid_resi_indices = [new_traj.top.atom(new_traj.top.select("protein")[-1]).residue.index+1+idx \
                                  for idx in np.arange(num_of_lipids)]
        else:
            new_traj = traj
        all_protein_atom_indices = new_traj.top.select("protein")
        natoms_per_protein = int(len(all_protein_atom_indices)/self.nprot)
        prot_atom_indices = all_protein_atom_indices[:natoms_per_protein]
        selected_protein_resi_set = []
        if len(self.resi_list) == 0:
            residue_set = ["{}{}".format(new_traj.top.residue(resi).resSeq+self.resi_offset, new_traj.top.residue(resi).name) \
                           for resi in np.arange(new_traj.top.atom(prot_atom_indices[0]).residue.index, \
                                                 new_traj.top.atom(prot_atom_indices[-1]).residue.index + 1)]
            residue_set = np.array(residue_set, dtype=str) # residue id in structure instead of builtin index in mdtraj
            for protein_idx in range(self.nprot):
                selected_protein_resi_set.append(np.unique([new_traj.top.atom(atom_idx).residue.index \
                                                              for atom_idx in \
                                                              all_protein_atom_indices[protein_idx*natoms_per_protein:(protein_idx+1)*natoms_per_protein]]))
        elif len(self.resi_list) > 0:
            resi_list = np.sort(np.array(np.hstack(self.resi_list), dtype=int))
            for protein_idx in range(self.nprot):
                selected_protein_resi_set.append(np.unique([new_traj.top.atom(atom_idx).residue.index \
                                                                   for atom_idx in \
                                                                   all_protein_atom_indices[protein_idx*natoms_per_protein:(protein_idx+1)*natoms_per_protein] \
                                                                   if new_traj.top.atom(atom_idx).residue.resSeq in resi_list]))
            residue_set = ["{}{}".format(new_traj.top.residue(resi).resSeq+self.resi_offset, new_traj.top.residue(resi).name) \
                           for resi in selected_protein_resi_set[0]]
            residue_set = np.array(residue_set, dtype=str)
        protein_resi_rank = selected_protein_resi_set[0] - new_traj.top.atom(all_protein_atom_indices[0]).residue.index
        if self._protein_ref == None:
            self._protein_ref = new_traj[0].atom_slice(prot_atom_indices)

        return new_traj, {"protein_resi_rank": protein_resi_rank, "selected_protein_resi_set": selected_protein_resi_set,
                          "residue_set": residue_set, "num_of_lipids": num_of_lipids,
                          "lipid_resi_indices": lipid_resi_indices, "lipid_resi_indices_original": lipid_resi_indices_original}


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
            self.lipid_resi_set = []
            self.T_total = []
            self.timesteps = []
            self.protein_resi_rank = []
            ncol_start = 0
            for traj_idx, trajfile in enumerate(self.trajfile_list):
                print("\n########## Start calculation of {} interaction in \n########## {} \n".format(self.lipid, self.trajfile_list[traj_idx]))
                f.write("\n###### Start calculation of {} interaction in \n###### {} \n".format(self.lipid, self.trajfile_list[traj_idx]))
                traj = md.load(trajfile, top=self.grofile_list[traj_idx], stride=self.stride)
                if self.dt == None:
                    timestep = traj.timestep/1000000.0 if self.timeunit == "us" else traj.timestep/1000.0
                else:
                    timestep = float(self.dt * self.stride)
                self.T_total.append((traj.n_frames - 1) * timestep)
                self.timesteps.append(timestep)
                new_traj, traj_stats = self._get_traj_stats(traj, self.lipid, self.lipid_atoms)
                if len(self.protein_resi_rank) == 0:
                    self.protein_resi_rank = traj_stats["protein_resi_rank"]
                self.lipid_resi_set.append(traj_stats["lipid_resi_indices_original"])
                if len(self.residue_set) == 0:
                    self.residue_set = traj_stats["residue_set"] 
                    self.nresi_per_protein = len(self.residue_set)
                elif len(traj_stats["residue_set"]) != len(self.residue_set):
                    raise IndexError("Protein configurations are different among repeats. Different number of residues detected!")
                ncol_per_protein = traj_stats["num_of_lipids"] * new_traj.n_frames
                for idx_protein in np.arange(self.nprot):
                    for resid, residue_index in enumerate(traj_stats["selected_protein_resi_set"][idx_protein]):
                        pairs = list(product([residue_index], traj_stats["lipid_resi_indices"]))
                        dist_matrix_resi, _ = md.compute_contacts(new_traj, pairs, scheme="closest", periodic=True)
                        contact_residues_low = [[] for dummy in np.arange(new_traj.n_frames)]
                        contact_residues_high = [[] for dummy in np.arange(new_traj.n_frames)]
                        frame_id_set_low, lipid_id_set_low = np.where(dist_matrix_resi <= self.cutoff[0])
                        frame_id_set_high, lipid_id_set_high = np.where(dist_matrix_resi <= self.cutoff[1])
                        for frame_id, lipid_id in zip(frame_id_set_low, lipid_id_set_low):
                            contact_residues_low[frame_id].append(int(lipid_id))
                        for frame_id, lipid_id in zip(frame_id_set_high, lipid_id_set_high):
                            contact_residues_high[frame_id].append(int(lipid_id))
                        col.append([ncol_start + ncol_per_protein*idx_protein + lipid_id*new_traj.n_frames + \
                                    frame_id for frame_id, lipid_id in zip(frame_id_set_low, lipid_id_set_low)])
                        contact_low = [np.array(contact, dtype=int) for contact in contact_residues_low]
                        contact_high = [np.array(contact, dtype=int) for contact in contact_residues_high]
                        row.append([resid for dummy in np.arange(len(frame_id_set_low))])
                        data.append(dist_matrix_resi[frame_id_set_low, lipid_id_set_low])
                        self.contact_residues_high[resid].append(contact_high)
                        self.contact_residues_low[resid].append(contact_low)
                        self.interaction_duration[resid].append(Durations(contact_low, contact_high, timestep).cal_duration())
                        occupancy, lipidcount = cal_interaction_intensity(contact_high)
                        self.interaction_occupancy[resid].append(occupancy)
                        self.lipid_count[resid].append(lipidcount)
                ncol_start += ncol_per_protein * self.nprot

                ###############################################
                ###### get some statistics for this traj ######
                ###############################################
                durations = np.array([np.concatenate(self.interaction_duration[resid][-self.nprot:]).mean() for resid in np.arange(self.nresi_per_protein)])
                duration_arg_idx = np.argsort(durations)[::-1]
                occupancies = np.array([np.mean(self.interaction_occupancy[resid][-self.nprot:]) for resid in np.arange(self.nresi_per_protein)])
                occupancy_arg_idx = np.argsort(occupancies)[::-1]
                lipidcounts =  np.array([np.mean(self.lipid_count[resid][-self.nprot:]) for resid in np.arange(self.nresi_per_protein)])
                lipidcount_arg_idx = np.argsort(lipidcounts)[::-1]
                log_text = "10 residues that showed longest average interaction durations ({}):\n".format(self.timeunit)
                for residue, duration in zip(traj_stats["residue_set"][duration_arg_idx][:10], durations[duration_arg_idx][:10]):
                    log_text += "{:^8s} -- {:^8.3f}\n".format(residue, duration)
                log_text += "10 residues that showed highest lipid occupancy (100%):\n"
                for residue, occupancy in zip(traj_stats["residue_set"][occupancy_arg_idx][:10], occupancies[occupancy_arg_idx][:10]):
                    log_text += "{:^8s} -- {:^8.2f}\n".format(residue, occupancy)
                log_text += "10 residues that have the largest number of surrounding lipids (count):\n"
                for residue, lipidcount in zip(traj_stats["residue_set"][lipidcount_arg_idx][:10], lipidcounts[lipidcount_arg_idx][:10]):
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
        if len(set(self.residue_set)) != len(self.residue_set):
            residue_name_set = ["{}_{}".format(residue, resi_rank) for residue, resi_rank in zip(self.residue_set, self.protein_resi_rank)]
        else:
            residue_name_set = self.residue_set
        for resid, residue in enumerate(residue_name_set):
            duration_raw = np.concatenate(self.interaction_duration[resid])
            if np.sum(duration_raw) > 0:
                bootstrap_results = self.bootstrap(duration_raw, residue, "{}/{}_{}.pdf".format(koff_dir, self.lipid, residue), \
                                                   nbootstrap=nbootstrap)
                self.sigmas[resid] = bootstrap_results["sigma"]
                self.koff[resid] = bootstrap_results["koff"]
                self.res_time[resid] = bootstrap_results["restime"]
                self.params[resid] = bootstrap_results["params"]
                self.r_squared[resid] = bootstrap_results["r_squared"]
                self.koff_b[resid] = bootstrap_results["koff_b_avg"]
                self.koff_b_cv[resid] = bootstrap_results["koff_b_cv"]
                self.res_time_b[resid] = bootstrap_results["res_time_b_avg"]
                self.res_time_b_cv[resid] = bootstrap_results["res_time_b_cv"]
                self.r_squared_b[resid] = bootstrap_results["r_squared_b_avg"]
            else:
                delta_t_range = np.arange(0, self.T_total[traj_idx], np.min(self.timesteps))
                self.sigmas[resid] = {key:value for key, value in zip(delta_t_range, np.zeros(len(delta_t_range)))}
                self.koff[resid] = 0
                self.res_time[resid] = 0
                self.params[resid] = [0, 0, 0, 0]
                self.r_squared[resid] = 0.0
                self.koff_b[resid] = 0
                self.koff_b_cv[resid] = 0
                self.res_time_b[resid] = 0
                self.res_time_b_cv[resid] = 0
                self.r_squared_b[resid] = 0.0

        ##############################################
        ########## wrapping up dataset ###############
        ##############################################
        T_max = np.max(self.T_total)
        Res_Time = np.array([self.res_time[resid] for resid in np.arange(self.nresi_per_protein)])
        Capped = Res_Time > T_max
        Res_Time[Capped] = T_max
        Res_Time_B = np.array([self.res_time_b[resid] for resid in np.arange(self.nresi_per_protein)])
        Capped = Res_Time_B > T_max
        Res_Time_B[Capped] = T_max
        dataset = pd.DataFrame({"Residue": [residue for residue in self.residue_set],
                                "Residue idx": self.protein_resi_rank,
                                "Occupancy": np.array([np.mean(self.interaction_occupancy[resid]) \
                                                       for resid in np.arange(self.nresi_per_protein)]),
                                "Occupancy_std": np.array([np.std(self.interaction_occupancy[resid]) \
                                                           for resid in np.arange(self.nresi_per_protein)]),
                                "Duration": np.array([np.mean(np.concatenate(self.interaction_duration[resid])) \
                                                      for resid in np.arange(self.nresi_per_protein)]),
                                "Duration_std": np.array([np.std(np.concatenate(self.interaction_duration[resid])) \
                                                          for resid in np.arange(self.nresi_per_protein)]),
                                "Residence Time": Res_Time,
                                "Capped": Capped,
                                "R squared": np.array([self.r_squared[resid] for resid in np.arange(self.nresi_per_protein)]),
                                "Koff": np.array([self.koff[resid] for resid in np.arange(self.nresi_per_protein)]),
                                "Residence Time_boot": Res_Time_B,
                                "Residence Time_boot_cv": np.array([self.res_time_b_cv[resid] for resid in np.arange(self.nresi_per_protein)]),
                                "Koff_boot": np.array([self.koff_b[resid] for resid in np.arange(self.nresi_per_protein)]),
                                "Koff_boot_cv": np.array([self.koff_b_cv[resid] for resid in np.arange(self.nresi_per_protein)]),
                                "R squared_boot": np.array([self.r_squared_b[resid] for resid in np.arange(self.nresi_per_protein)]),
                                "LipidCount": np.array([np.mean(self.lipid_count[resid]) \
                                                         for resid in np.arange(self.nresi_per_protein)]),
                                "LipidCount_std": np.array([np.std(self.lipid_count[resid]) \
                                                             for resid in np.arange(self.nresi_per_protein)])})

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
        initial_guess = (1., 1., 1., 1.)
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
            sigma_sampled = cal_sigma(duration_sampled, len(duration_sampled), np.max(self.T_total), delta_t_range)
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
        sigma = cal_sigma(durations, len(durations), np.max(self.T_total), delta_t_range)
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


    def cal_interaction_network(self, save_dir=None, pdb=None, save_dataset=True, nbootstrap=10, \
                                radii=None, gen_binding_poses=5, score_weights=None, save_pose_format="pdb", kde_bw=0.15):

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

        f = open("{}/BindingSites_Info_{}.txt".format(save_dir, self.lipid), "w")
        ##### write out info ######
        reminder = """
# Occupancy: percentage of frames where lipid is in contact with the given residue (0-100%);
# Duration/Residence Time: average length of a continuous interaction of lipid with the given residue (in unit of {timeunit});
# Koff: Koff of lipid with the given residue/binding site (in unit of ({timeunit})^(-1));
# Pos. Charge: ARG, HIS, LYS;
# Neg. Charge: ASP, GLU;
# Polar: SER, THR, ASN, GLN;
# Hydrophobic: ALA, VAL, ILE, LEU, MET, PHE, TYR, TRP;
# Special: CYS, SEC, GLY, PRO.
        """.format(**{"timeunit": self.timeunit})
        f.write(reminder)
        f.write("\n")
        binding_site_id = 0
        interaction_covariance = np.nan_to_num(self.interaction_covariance)
        covariance_network = np.copy(interaction_covariance)
        covariance_network[covariance_network < 0.0] = 0.0
        residue_network_raw = nx.Graph(covariance_network)
        part = community.best_partition(residue_network_raw, weight='weight')
        values = [part.get(node) for node in residue_network_raw.nodes()]
        binding_site_identifiers = np.ones(self.nresi_per_protein, dtype=int) * 999
        self.interaction_duration_BS = defaultdict(list)
        self.interaction_occupancy_BS = defaultdict(list)
        self.lipid_count_BS = defaultdict(list)
        self.sigmas_BS = {}
        self.params_BS = {}
        BS_restime = np.zeros(self.nresi_per_protein)
        BS_koff = np.zeros(self.nresi_per_protein)
        BS_rsquared = np.zeros(self.nresi_per_protein)
        BS_duration = np.zeros(self.nresi_per_protein)
        BS_lipidcount = np.zeros(self.nresi_per_protein)
        BS_occupancy = np.zeros(self.nresi_per_protein)
        BS_koff_b = np.zeros(self.nresi_per_protein)
        BS_koff_b_cv = np.zeros(self.nresi_per_protein)
        BS_restime_b = np.zeros(self.nresi_per_protein)
        BS_restime_b_cv = np.zeros(self.nresi_per_protein)
        BS_rsquared_b = np.zeros(self.nresi_per_protein)
        BS_surface_area = np.zeros(self.nresi_per_protein)
        t_total_max = np.max(self.T_total)
        node_list_set = []
        for value in range(max(values)):
            node_list = [k for k,v in part.items() if v == value]
            if len(node_list) >= 3:
                binding_site_identifiers[node_list] = binding_site_id
                node_list_set.append(node_list)
                binding_site_id += 1
        ########### cal site koff and surface area ############
        if len(node_list_set) > 0:
            surface_area_all = defaultdict(list)
            self._coordinate_pool = [[] for dummy in np.arange(len(node_list_set))]
            for traj_idx, trajfile in enumerate(self.trajfile_list):
                traj = md.load(trajfile, top=self.grofile_list[traj_idx], stride=self.stride)
                if self.dt == None:
                    timestep = traj.timestep/1000000.0 if self.timeunit == "us" else traj.timestep/1000.0
                else:
                    timestep = float(self.dt)
                protein_indices_all = traj.top.select("protein")
                natoms_per_protein = int(len(protein_indices_all)/self.nprot)
                for idx_protein in np.arange(self.nprot):
                    protein_indices = protein_indices_all[idx_protein*natoms_per_protein:(idx_protein+1)*natoms_per_protein]
                    for binding_site_id, node_list in enumerate(node_list_set):
                        list_to_take = traj_idx*self.nprot+idx_protein
                        contact_BS_low = [np.unique(np.concatenate([self.contact_residues_low[node][list_to_take][frame_idx] for node in node_list])) \
                                          for frame_idx in range(traj.n_frames)]
                        contact_BS_high = [np.unique(np.concatenate([self.contact_residues_high[node][list_to_take][frame_idx] for node in node_list])) \
                                           for frame_idx in range(traj.n_frames)]
                        self.interaction_duration_BS[binding_site_id].append(Durations(contact_BS_low, contact_BS_high, timestep).cal_duration())
                        occupancy, lipidcount = cal_interaction_intensity(contact_BS_high)
                        self.interaction_occupancy_BS[binding_site_id].append(occupancy)
                        self.lipid_count_BS[binding_site_id].append(lipidcount)
                        ########### store lipid binding poses ############
                        for frame_id in range(len(contact_BS_low)):
                            if len(contact_BS_low[frame_id]) > 0:
                                for lipid_id in contact_BS_low[frame_id]:
                                    lipid_index = self.lipid_resi_set[traj_idx][lipid_id]
                                    lipid_indices = np.sort([atom.index for atom in traj.top.residue(lipid_index).atoms])
                                    self._coordinate_pool[binding_site_id].append([np.copy(traj.xyz[frame_id, np.hstack([protein_indices, lipid_indices])]), \
                                                                                   np.copy(traj.unitcell_angles[frame_id]), \
                                                                                   np.copy(traj.unitcell_lengths[frame_id])])
                    ### calculate area ###
                    new_xyz = []
                    for frame in traj.xyz:
                        new_frame = frame[protein_indices]
                        new_xyz.append(new_frame)
                    reduced_frame = traj[0].atom_slice(protein_indices)
                    reduced_top = reduced_frame.top
                    if reduced_top.residue(0).index != 0:
                        starting_index = reduced_top.residue(0).index
                        for residue in reduced_top.residues:
                            residue.index -= starting_index
                    new_traj = md.Trajectory(new_xyz, reduced_top, time=traj.time, unitcell_lengths=traj.unitcell_lengths, unitcell_angles=traj.unitcell_angles)
                    areas = md.shrake_rupley(new_traj, mode='residue', change_radii=radii_book)
                    for binding_site_id, node_list in enumerate(node_list_set):
                        surface_area_all[binding_site_id].append(areas[:, node_list].sum(axis=1))
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
                bs_area = np.concatenate(surface_area_all[binding_site_id]).mean()
                BS_surface_area[mask] = bs_area
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
                f.write("{:20s} {:10.3f} nm^2 +- {:10.3f}\n".format(" BS Surface Area:", bs_area, np.concatenate(surface_area_all[binding_site_id]).std()))
                res_stats = {"Pos. Charge": 0, "Neg. Charge": 0, "Polar": 0, "Special": 0, "Hydrophobic": 0}
                for residue in self.residue_set[mask]:
                    res_stats[Residue_property_book[re.findall("[a-zA-Z]+$", residue)[0]]] += 1
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
            for binding_site_id in surface_area_all.keys():
                bs_area_set.append(np.concatenate(surface_area_all[binding_site_id]))
                bs_id_set.append([binding_site_id for dummy in np.arange(len(bs_area_set[-1]))])
            d_area = pd.DataFrame({"BS id": np.concatenate(bs_id_set), "Area (nm^2)": np.concatenate(bs_area_set)})
            plt.rcParams["font.size"] = 8
            plt.rcParams["font.weight"] = "bold"
            if len(surface_area_all.keys()) <= 8:
                fig, ax = plt.subplots(figsize=(4.5, 2.8))
            elif len(surface_area_all.keys()) > 8 and len(surface_area_all.keys()) <= 15:
                fig, ax = plt.subplots(figsize=(6.5, 2.8))
            else:
                fig, ax = plt.subplots(figsize=(9.5, 3))
            sns.violinplot(x="BS id", y="Area (nm^2)", data=d_area, palette="Set3", bw=.2, cut=1, linewidth=1, ax=ax)
            ax.set_xlabel("BS id", fontsize=8, weight="bold")
            ax.set_ylabel(r"Surface Area (nm$^2$)", fontsize=8, weight="bold")
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
        self.dataset["BS Surface Area"] = BS_surface_area
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
                pickle.dump(surface_area_all, f, 2)
        ################## generate binding poses ################
        if gen_binding_poses > 0 and len(node_list_set) > 0:
            coords_save_dir = check_dir(save_dir, "Binding_Poses")
            lipid_atom_map = {atom.index:atom.name for atom in self._lipid_ref.top.atoms}
            weights = {name:1 for index, name in lipid_atom_map.items()}
            if score_weights != None:
                weights.update(score_weights)
            binding_site_id_set = np.arange(len(self._coordinate_pool))
            if len(self.resi_list) == 0:
                selected_protein_atoms = [[atom.index for atom in residue.atoms] for residue in self._protein_ref.top.residues]
            else:
                selected_protein_atoms = [[atom.index for atom in residue.atoms] for residue in self._protein_ref.top.residues \
                                                   if residue.resSeq in self.resi_list]
            lipid_atoms = [self._protein_ref.n_atoms + atom_idx for atom_idx in np.arange(self._lipid_ref.n_atoms)]
            joined_top = self._protein_ref.top.join(self._lipid_ref.top)

            for binding_site_id in binding_site_id_set:
                num_of_poses = gen_binding_poses if gen_binding_poses <= len(self._coordinate_pool[binding_site_id]) \
                    else len(self._coordinate_pool[binding_site_id])
                node_list = node_list_set[binding_site_id]
                new_traj = md.Trajectory([frame[0] for frame in self._coordinate_pool[binding_site_id]], joined_top, \
                                          time=np.arange(len(self._coordinate_pool[binding_site_id])), \
                                          unitcell_angles=[frame[1] for frame in self._coordinate_pool[binding_site_id]], \
                                          unitcell_lengths=[frame[2] for frame in self._coordinate_pool[binding_site_id]])
                dist_per_atom = [[md.compute_distances(new_traj, list(product([lipid_atoms[idx]], selected_protein_atoms[resi])), periodic=True).min(axis=1) \
                                  for resi in node_list] for idx in np.arange(self._lipid_ref.n_atoms)]

                kde_funcs = {}
                try:
                    for atom_idx in np.arange(self._lipid_ref.n_atoms):
                        transformed_data = PCA(n_components=0.95).fit_transform(np.array(dist_per_atom[atom_idx]).T)
                        var_type = ""
                        bw = []
                        for dummy in range(len(transformed_data[0])):
                            var_type += "c"
                            bw.append(kde_bw)
                        kde_funcs[atom_idx] = KDEMultivariate(data=transformed_data, \
                                                                    var_type=var_type, bw=bw)
                    ### evaluate binding poses ###
                    scores = np.sum([weights[lipid_atom_map[idx]] * kde_funcs[idx].pdf() \
                                    for idx in np.arange(self._lipid_ref.n_atoms)], axis=0)
                    selected_indices = np.argsort(scores)[::-1][:num_of_poses]
                    ###############################
                    for pose_id in np.arange(num_of_poses, dtype=int):
                        new_traj[selected_indices[pose_id]].save("{}/BSid{}_No{}.{}".format(coords_save_dir, \
                                                                                            binding_site_id, pose_id, save_pose_format))
                except ValueError:
                    with open("{}/Error.txt".format(coords_save_dir), "a+") as error_file:
                        error_file.write("BSid {}: Pose generation error -- possibly due to insufficient number of binding event.\n".format(binding_site_id))

        ######################################################################
        ###### show binding site residues with scaled spheres in pymol #######
        ######################################################################
        if pdb != None:
            ############ check if pdb has a path to it ##########
            pdb_new_loc = os.path.join(self.save_dir, os.path.basename(pdb))
            copyfile(pdb, pdb_new_loc)
            ########### write out a pymol pml file ###############
            binding_site_id += 1
            text = """
import numpy as np
import re
import pymol
from pymol import cmd
pymol.finish_launching()

########## set the sphere scales to corresponding value ##########
value_to_show = "Residence Time"

###### reading data from csv file ##########
binding_site_id = {BINDING_SITE_ID}

fn_data = "{HOME_DIR}/Interactions_{LIPID}.csv"
with open(fn_data, "r") as f:
    data_lines = f.readlines()
    
column_names = data_lines[0].strip().split(",")
for column_idx, column_name in enumerate(column_names):
    if column_name == "Residue":
        column_id_residue_set = column_idx
    elif column_name == "Residue idx":
        column_id_residue_index = column_idx
    elif column_name == "Binding_site":
        column_id_BS = column_idx
    elif column_name == value_to_show:
        column_id_value_to_show = column_idx

residue_set = []
residue_rank_set = []
binding_site_identifiers = []
values_to_show = []
for line in data_lines[1:]:
    data_list = line.strip().split(",")
    residue_set.append(data_list[column_id_residue_set])
    residue_rank_set.append(data_list[column_id_residue_index])
    binding_site_identifiers.append(data_list[column_id_BS])
    values_to_show.append(data_list[column_id_value_to_show])

############## read information from pdb coordinates ##########
pdb_file = "{PDB}"
with open(pdb_file, "r") as f:
    pdb_lines = f.readlines()

residue_identifiers = set()    
for line in pdb_lines:
    if line.strip()[:4] == "ATOM":
        residue_identifiers.add((line.strip()[22:26].strip(), line.strip()[17:20].strip(), line.strip()[21].strip()))
##                                    residue index,              resname,                     chain id     

######### calculate scale ###############
values_to_show = np.array(values_to_show)
MIN = np.percentile(values_to_show, 0)
MAX = np.percentile(values_to_show, 100)
X = (values_to_show - np.percentile(values_to_show, 50))/(MAX - MIN)
SCALES = 2.5/(1 + np.exp(-X * 10))

######## some pymol settings #########
cmd.set("retain_order", 1)
cmd.set("cartoon_oval_length", 1.0)
cmd.set("cartoon_oval_width", 0.3)
cmd.set("cartoon_color", "white")
cmd.set("stick_radius", 0.35)

##################################
cmd.load("{PDB}", "Prot_{LIPID}")
prefix = "Prot_{LIPID}"
cmd.hide("everything")
cmd.show("cartoon", prefix)
cmd.center(prefix)
cmd.orient(prefix)
colors = np.array([np.random.choice(np.arange(256, dtype=float), size=3) for dummy in range(binding_site_id)])
colors /= 255.0
            """.format(**{"HOME_DIR": self.save_dir, "LIPID": self.lipid, "BINDING_SITE_ID": binding_site_id, "PDB": pdb_new_loc})
            text += r"""
residue_set = np.array(residue_set, dtype=str)
residue_rank_set = np.array(residue_rank_set, dtype=int)
binding_site_identifiers = np.array(binding_site_identifiers, dtype=int)
residue_identifiers = list(residue_identifiers)
for bs_id in np.arange(binding_site_id):
    cmd.set_color("tmp_{}".format(bs_id), list(colors[bs_id]))
    for entry_id in np.where(binding_site_identifiers == bs_id)[0]:
        selected_residue = residue_set[entry_id]
        selected_residue_rank = residue_rank_set[entry_id]
        identifier_from_pdb = residue_identifiers[selected_residue_rank]
        if re.findall("[a-zA-Z]+$", selected_residue)[0] != identifier_from_pdb[1]:
            raise IndexError("The {}th residue in the provided pdb file ({}{}) is different from that in the simulations ({})!".format(entry_id+1, 
                                                                                                                                     identifier_from_pdb[0], 
                                                                                                                                     identifier_from_pdb[1],
                                                                                                                                     selected_residue))
        if identifier_from_pdb[2] != " ":
            cmd.select("BSid{}_{}".format(bs_id, selected_residue), "chain {} and resid {} and (not name C+O+N)".format(identifier_from_pdb[2], 
                                                                                                                      identifier_from_pdb[0]))
        else:
            cmd.select("BSid{}_{}".format(bs_id, selected_residue), "resid {} and (not name C+O+N)".format(identifier_from_pdb[0]))
        cmd.show("spheres", "BSid{}_{}".format(bs_id, selected_residue))
        cmd.set("sphere_scale", SCALES[entry_id], selection="BSid{}_{}".format(bs_id, selected_residue))
        cmd.color("tmp_{}".format(bs_id), "BSid{}_{}".format(bs_id, selected_residue))
    cmd.group("BSid{}".format(bs_id), "BSid{}_*".format(bs_id))
            
            """
            with open("{}/show_binding_sites_info.py".format(self.save_dir), "w") as f:
                f.write(text)

        return


    def plot_interactions(self, item="Duration",  save_dir=None, letter_map=None, chain_breaks=[]):

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
        if len(chain_breaks) == 0:
            chain_break_points = [0, len(self.dataset)]
            no_break = True
        else:
            chain_break_points = [0]
            for points in chain_breaks:
                chain_break_points.append(points)
            chain_break_points.append(len(self.dataset))
            no_break = False

        plt.rcParams["font.size"] = 8
        plt.rcParams["font.weight"] = "bold"

        for point_idx in np.arange(1, len(chain_break_points), dtype=int):
            dataset = self.dataset[chain_break_points[point_idx-1]:chain_break_points[point_idx]]
            data = dataset[item]
            if len(data) == 0:
                continue
            resi = np.array([int(re.findall("^[0-9]+", residue)[0]) for residue in self.residue_set])[chain_break_points[point_idx-1]:chain_break_points[point_idx]]
            SL_resn = [single_letter[re.findall("[a-zA-Z]+$", residue)[0]] for residue in self.residue_set][chain_break_points[point_idx-1]:chain_break_points[point_idx]]
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
                elif len(data) <= 1000 and len(data) > 100:
                    ax_data.xaxis.set_major_locator(MultipleLocator(100))
                    ax_data.xaxis.set_minor_locator(MultipleLocator(10))
                    ax_boot.xaxis.set_major_locator(MultipleLocator(100))
                    ax_boot.xaxis.set_minor_locator(MultipleLocator(10))
                elif len(data) <= 100:
                    ax_data.xaxis.set_major_locator(MultipleLocator(10))
                    ax_data.xaxis.set_minor_locator(MultipleLocator(1))
                    ax_boot.xaxis.set_major_locator(MultipleLocator(10))
                    ax_boot.xaxis.set_minor_locator(MultipleLocator(1))
                if self.timeunit == "ns":
                    timeunit = " (ns) "
                elif self.timeunit == "us":
                    timeunit = r" ($\mu s$)"
                ax_data.bar(resi, data, width, linewidth=0, color="#F75C03")
                ax_data.set_ylabel("Res. Time {}".format(timeunit), fontsize=8, weight="bold", va="center")
                ax_data.set_xlabel("Residue Index", fontsize=8, weight="bold")
                ax_capped.plot(resi, dataset["Capped"]*1, linewidth=0, marker="+", markerfacecolor="#38040E", \
                               markeredgecolor="#38040E", markersize=2)
                ax_capped.set_ylim(0.9, 1.1)
                ax_capped.set_yticks([1.0])
                ax_capped.set_yticklabels(["Capped"], fontsize=8, weight="bold")
                ax_capped.set_xlim(ax_data.get_xlim())
                mask = dataset["R squared"] > 0
                ax_R2.plot(resi[mask], dataset["R squared"][mask], linewidth=0, marker="+", markerfacecolor="#0FA3B1", markeredgecolor="#0FA3B1", \
                           markersize=2)
                ax_R2.set_xlim(ax_data.get_xlim())
                ax_R2.set_ylabel(r"$R^2$", fontsize=8, weight="bold", va="center")
                ax_R2.set_title("{} {}".format(self.lipid, item), fontsize=8, weight="bold")

                ax_boot.bar(resi, dataset["Residence Time_boot"], width, linewidth=0, color="#F75C03")
                ax_boot.set_xlim(ax_data.get_xlim())
                ax_boot.set_ylabel("Res. Time \n Boot. {}".format(timeunit), fontsize=8, weight="bold", va="center")
                ax_boot.set_xticklabels([])
                mask = dataset["R squared_boot"] > 0
                mask = dataset["Residence Time_boot_cv"] > 0
                ax_boot_cv.plot(resi[mask], dataset["Residence Time_boot_cv"][mask], linewidth=0, marker="+", markerfacecolor="#0FA3B1", markeredgecolor="#F7B538",
                                markersize=2)
                ax_boot_cv.set_ylabel("Coef. Var.", fontsize=8, weight="bold", va="center")
                ax_boot_cv.set_xlim(ax_data.get_xlim())
                for ax in [ax_data, ax_capped, ax_R2, ax_boot, ax_boot_cv]:
                    ax.yaxis.set_label_coords(-0.15, 0.5, transform=ax.transAxes)
                if no_break:
                    plt.savefig("{}/{}_{}.pdf".format(save_dir, "_".join(item.split()), self.lipid), dpi=300)
                else:
                    plt.savefig("{}/{}_{}_{}.pdf".format(save_dir, "_".join(item.split()), self.lipid, str(point_idx)), dpi=300)
                plt.close()

            else:
                fig, ax = plt.subplots(1, 1, figsize=(4, 2.2))
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
                if no_break:
                    plt.savefig("{}/{}_{}.pdf".format(save_dir, "_".join(item.split()), self.lipid), dpi=300)
                else:
                    plt.savefig("{}/{}_{}_{}.pdf".format(save_dir, "_".join(item.split()), self.lipid, str(point_idx)), dpi=300)
                plt.close()
            
            ###### logomater #####
            if item == "Residence Time":
                ylabel = "Res. Time {}".format(timeunit)
            df = pd.DataFrame({"Resid": resi, "Resn": SL_resn, "Data": data})
            matrix = df.pivot(index="Resid", columns='Resn', values="Data").fillna(0)
            n_rows = 1 + resi[-1]//100 - resi[0]//100
            start = (resi[0]//100)*100
            length = start + 100 - resi[0]
            fig, axes = plt.subplots(n_rows, 1, figsize=(4.5, 1.3*n_rows), sharey=True)
            plt.subplots_adjust(hspace=0.5, left=0.2)
            for idx, ax in enumerate(np.atleast_1d(axes)):
                try:
                    if idx == (n_rows - 1):
                        logomaker.Logo(matrix[(idx-1)*100 + length:], color_scheme="chemistry", ax=ax)
                        ax.set_xlabel("Residue Index", fontsize=8, weight="bold")
                    elif idx == 0:
                        logomaker.Logo(matrix[:length], color_scheme="chemistry", ax=ax)
                    else:
                        logomaker.Logo(matrix[(idx-1)*100+length:idx*100+length], color_scheme="chemistry", ax=ax)
                except logomaker.src.error_handling.LogomakerError:
                    pass
                ax.xaxis.set_major_locator(MultipleLocator(20))
                ax.xaxis.set_minor_locator(MultipleLocator(1))
                ax.set_xlim(idx*100+start, (idx+1)*100+start)
                ax.set_ylim(0, data.max()*1.05)
                ax.set_ylabel(ylabel, fontsize=8, weight="bold", va="center")
                for label in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
                    plt.setp(label, fontsize=8, weight="bold")
            plt.tight_layout()
            if no_break:
                plt.savefig("{}/{}_logo_{}.pdf".format(save_dir, "_".join(item.split()), self.lipid), dpi=300)
            else:
                plt.savefig("{}/{}_logo_{}_{}.pdf".format(save_dir, "_".join(item.split()), self.lipid, str(point_idx)), dpi=300)
            plt.close()

        return


    def write_to_pdb(self, item, save_dir=None):

        if save_dir == None:
            save_dir = check_dir(self.save_dir, "Coordinates_{}".format(self.lipid))
        else:
            save_dir = check_dir(save_dir, "Coordinates_{}".format(self.lipid))
        ##### load coords ######
        data = self.dataset[item]
        coords = self._protein_ref.xyz[0]
        table, _ = self._protein_ref.top.to_dataframe()
        atom_idx_set = table.serial
        resid_set = table.resSeq + self.resi_offset
        atom_name_set = table.name
        resn_set = table.resName
        chainID = [chr(65+int(idx)) for idx in table.chainID]
        data_expanded = np.zeros(len(table))
        residue_indices = np.array([atom.residue.index for atom in self._protein_ref.top.atoms])
        for value, selected_residue_index in zip(data, self._selected_residue_indices):
            locations = np.where(residue_indices == selected_residue_index)[0]
            data_expanded[locations] = value
        ######## write out coords ###########
        fn = "{}/Coords_{}.pdb".format(save_dir, "_".join(item.split()))
        with open(fn, "w") as f:
            for idx in np.arange(self._protein_ref.n_atoms):
                coords_dictionary = {"HEADER": "ATOM",
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
                                    "BFACTOR": data_expanded[idx]}
                row = "{HEADER:6s}{ATOM_ID:5d} ".format(**coords_dictionary) +\
                      "{ATOM_NAME:^4s}{SPARE:1s}{RESN:3s} ".format(**coords_dictionary) +\
                      "{CHAIN_ID:1s}{RESI:4d}{SPARE:1s}   ".format(**coords_dictionary) +\
                      "{COORDX:8.3f}{COORDY:8.3f}{COORDZ:8.3f}{OCCUP:6.2f}{BFACTOR:6.2f}\n".format(**coords_dictionary)
                f.write(row)
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
    #######################################################################
    ######## write a backup file of params for reproducibility ############
    fn = os.path.join(save_dir, "pylipid_backup_{}.txt".format(datetime.datetime.now().strftime("%Y_%m_%d_%H%M")))
    with open(fn, "w") as f:
        f.write("##### Record params for reproducibility #####\n")
        f.write("python {}\n".format(" ".join(sys.argv)))
    ######################################################################
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
    ################# process chain breaks ################################
    chain_breaks = [] if len(args.chain_breaks) == 0 else [int(num)-1 for num in args.chain_breaks]
    #######################################################################
    for lipid in lipid_set:
        li = LipidInteraction(trajfile_list, grofile_list, stride=int(args.stride), dt=args.dt, cutoff=cutoff, lipid=lipid, \
                              lipid_atoms=args.lipid_atoms, nprot=args.nprot, timeunit=args.tu, resi_offset=int(args.resi_offset), \
                              resi_list=resi_list, save_dir=args.save_dir)
        li.cal_interactions(save_dataset=args.save_dataset, nbootstrap=int(args.nbootstrap))
        li.plot_interactions(item="Duration", letter_map=letter_map, chain_breaks=chain_breaks)
        li.plot_interactions(item="Residence Time", letter_map=letter_map, chain_breaks=chain_breaks)
        li.plot_interactions(item="Occupancy", letter_map=letter_map, chain_breaks=chain_breaks)
        li.plot_interactions(item="LipidCount", letter_map=letter_map, chain_breaks=chain_breaks)
        li.write_to_pdb(item="Duration")
        li.write_to_pdb(item="Residence Time")
        li.write_to_pdb(item="Occupancy")
        li.write_to_pdb(item="LipidCount")
        li.cal_interaction_network(pdb=args.pdb, save_dataset=args.save_dataset, \
                                   radii=radii_book, gen_binding_poses=int(args.gen_binding_poses), \
                                   score_weights=score_weights, save_pose_format=args.save_pose_format)


