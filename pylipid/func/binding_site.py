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

"""This module contains functions for calculating binding sites"""

import os
from collections import defaultdict
from itertools import product
import community
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import trange
import mdtraj as md
from statsmodels.nonparametric.kernel_density import KDEMultivariate as kde
from sklearn.decomposition import PCA


__all__ = ["get_node_list", "collect_bound_poses", "vectorize_poses", "calculate_scores", "write_bound_poses",
           "calculate_site_surface_area"]


def get_node_list(corrcoef, threshold=4):
    """Calculate binding sites from correlation coeffient matrix.

    Binding sites are calculated as the community structures of the network built
    on the correlation coefficient matrix.

    Parameters
    -----------
    corrcoef : ndarray(n, n)
    threshold : int, default=4

    Returns
    --------
    node_list : list of lists
    modularity : float or None

    """
    # TODO: check negative values in corrcoef_matrix. Come up with better solutions.
    corrcoef[corrcoef < 0.0] = 0.0 # network edge can't take negative values. Residues with
                                   # negative correlationsare are forced to separate to different binding sites.
    graph = nx.Graph(corrcoef)
    partition = community.best_partition(graph, weight='weight')
    values = [partition.get(node) for node in graph.nodes()]
    node_list = []
    for value in range(max(values)):
        nodes = [k for k, v in partition.items() if v == value]
        if len(nodes) >= threshold:
            node_list.append(nodes)
    if len(node_list) > 0:
        modularity = community.modularity(partition, graph)
    else:
        modularity = None
    return node_list, modularity


def collect_bound_poses(binding_site_map, contact_residue_index, trajfile_list, topfile_list,
                        lipid, stride=1, nprot=1):
    """Collected the bound poses based on contact_residue_index.

    Parameters
    ----------
    binding_site_map : dict
    contact_residue_index : dict
    trajfile_list : list of str
    topfile_list : list of str
    lipid : str
    stride : int,default=1
    nprot : int, default=1

    Returns
    ---------
    pose_pool : dict
        Coordinates of all bound poses in stored in a python dictionary {binding_site_id: pose coordinates}

    See also
    --------
    pylipid.func.cal_contact_residues
        The function that calculates the contact residues from distance matrix

    """
    trajfile_list = np.atleast_1d(trajfile_list)
    topfile_list = np.atleast_1d(topfile_list)
    stride = int(stride)
    nprot = int(nprot)
    pose_pool = defaultdict(list)
    for traj_idx in np.arange(len(trajfile_list)):
        traj = md.load(trajfile_list[traj_idx], top=topfile_list[traj_idx], stride=stride)
        protein_indices_all = traj.top.select("protein")
        natoms_per_protein = int(len(protein_indices_all) / nprot)
        # obtain lipid residue index
        lipid_residue_index_set = set()
        for atom_idx in traj.top.select("resn {}".format(lipid)):
            lipid_residue_index_set.add(traj.top.atom(atom_idx).residue.index)
        lipid_residue_index_set = list(lipid_residue_index_set)
        lipid_residue_index_set.sort()
        # store collect poses
        for protein_idx in np.arange(nprot):
            protein_indices = protein_indices_all[
                              protein_idx * natoms_per_protein:(protein_idx + 1) * natoms_per_protein]
            for bs_id, nodes in binding_site_map.items():
                list_to_take = traj_idx * nprot + protein_idx
                contact_BS = [np.unique(np.concatenate(
                    [contact_residue_index[node][list_to_take][frame_idx] for node in nodes]))
                    for frame_idx in range(traj.n_frames)]
                for frame_id in range(len(contact_BS)):
                    if len(contact_BS[frame_id]) > 0:
                        for lipid_id in contact_BS[frame_id]:
                            lipid_residue_index = lipid_residue_index_set[int(lipid_id)]
                            lipid_atom_indices = np.sort(
                                [atom.index for atom in traj.top.residue(lipid_residue_index).atoms])
                            pose_pool[bs_id].append(
                                [np.copy(traj.xyz[frame_id, np.hstack([protein_indices, lipid_atom_indices])]),
                                 np.copy(traj.unitcell_angles[frame_id]), np.copy(traj.unitcell_lengths[frame_id])])
        return pose_pool


def vectorize_poses(bound_poses, binding_nodes, protein_ref, lipid_ref):
    """Calculate distance matrix for bound poses.

    Parameters
    ----------
    bound_poses :
    binding_nodes :
    protein_ref :
    lipid_ref :

    Returns
    -------
    dist_matrix :
    pose_traj : mdtraj.Trajectory

    """
    # prepare topology
    protein_atom_indices = [[atom.index for atom in residue.atoms] for residue in protein_ref.top.residues]
    lipid_atoms = [protein_ref.n_atoms + atom_idx for atom_idx in np.arange(lipid_ref.n_atoms)]
    joined_top = protein_ref.top.join(lipid_ref.top)

    # start pose generation
    pose_traj = md.Trajectory([frame[0] for frame in bound_poses], joined_top,
                             time=np.arange(len(bound_poses)),
                             unitcell_angles=[frame[1] for frame in bound_poses],
                             unitcell_lengths=[frame[2] for frame in bound_poses])
    # calculate distance to binding site residues for each lipid atom
    dist_per_atom = np.array(
        [np.array([md.compute_distances(pose_traj,
                                        list(product([lipid_atoms[idx]], protein_atom_indices[node])),
                                        periodic=True).min(axis=1) for node in binding_nodes]).T
         for idx in np.arange(lipid_ref.n_atoms)])  # shape: [n_lipid_atoms, n_poses, n_BS_residues]
    return dist_per_atom, pose_traj


def calculate_scores(data, kde_bw=0.15, pca_component=0.95, score_weights=None):
    """Calculate scores based on probability density.

    Parameters
    ----------
    data : ndarray, shape=(n_atoms, n_samples, n_dims)
    kde_bw : scalar
    pca_component : scalar
    score_weights : None or dict
        A dictionary that contains the weight for n_dims, {n_dim: weight}

    Returns
    ---------
    scores : ndarray, shape=(n_samples,)

    See also
    --------
    pylipid.func.collect_bound_poses
    pylipid.func.vectorize_poses

    """
    weights = {atom_idx: 1 for atom_idx in np.arange(np.shape(data)[0])}
    if score_weights is not None:
        weights.update(score_weights)

    kde_funcs = {}
    try:
        for atom_idx in np.arange(np.shape(data)[0]):
            transformed_data = PCA(n_components=pca_component).fit_transform(data[atom_idx])
            var_type = ""
            bw = []
            for dummy in range(len(transformed_data[0])):
                var_type += "c"
                bw.append(kde_bw)
            kde_funcs[atom_idx] = kde(data=transformed_data, var_type=var_type, bw=bw)
        # evaluate binding poses
        scores = np.sum([weights[atom_idx] * kde_funcs[atom_idx].pdf()
                         for atom_idx in np.arange(np.shape(data)[0])], axis=0)
        return scores

    except ValueError:
        print("Pose generation error -- possibly due to insufficient number of binding event.")


def write_bound_poses(pose_traj, pose_indices, save_dir, pose_prefix="BoundPose", pose_format="pdb"):
    """Write bound poses specified by pose_indices.

    Parameters
    ----------
    pose_traj : mdtraj.Trajectory
    pose_indices : array_like
    save_dir : str
    pose_prefix : str, optional, default=BoundPose"
    pose_format : str, optional, default="gro"

    """
    for idx, pose_id in enumerate(pose_indices):
        pose_traj[pose_id].save(os.path.join(save_dir, "{}{}.{}".format(pose_prefix, idx, pose_format)))
    return


def calculate_site_surface_area(binding_site_map, radii_book, trajfile_list, topfile_list,
                                nprot, timeunit, stride, dt_traj=None):
    """Calculate the surface area (in unit of nm^2) as a function of time.

    Parameters
    ----------
    binding_site_map : dict
    trajfile : str or list of str
    topfile : str or list of str
    nprot : int
    timeunit : str, {ns, us}
    stride : int
    dt_traj : scalar or None, optional, default=None

    Returns
    -------
    surface_area : pandas.DataFrame

    """
    surface_data = []
    data_keys = []
    for traj_idx in trange(len(trajfile_list), desc="CALCULATE SURFACE AREA PER TRAJ", total=len(trajfile_list)):
        trajfile, topfile = trajfile_list[traj_idx], topfile_list[traj_idx]
        traj = md.load(trajfile, top=topfile, stride=stride)
        if dt_traj is None:
            traj_times = traj.time / 1000000.0 if timeunit == "us" else traj.time / 1000.0
        else:
            traj_times = float(dt_traj * stride) * np.arange(traj.n_frames)
        protein_indices_all = traj.top.select("protein")
        natoms_per_protein = int(len(protein_indices_all) / nprot)
        for protein_idx in np.arange(nprot):
            protein_indices = protein_indices_all[
                              protein_idx * natoms_per_protein:(protein_idx + 1) * natoms_per_protein]
            new_traj = traj.atom_slice(protein_indices, inplace=False)
            area_all = md.shrake_rupley(new_traj, mode='residue', change_radii=radii_book)
            selected_data = [area_all[:, nodes].sum(axis=1) for bs_id, nodes in binding_site_map.items()]
            selected_data.append(traj_times)
            column_names = ["Binding Site {}".format(bs_id) for bs_id, nodes in binding_site_map.items()]
            column_names.append("Time")
            surface_data.append(pd.DataFrame(np.array(selected_data).T, columns=column_names))
            data_keys.append((traj_idx, protein_idx))
    surface_area = pd.concat(surface_data, keys=data_keys)
    return surface_area
