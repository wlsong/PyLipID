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
from ..util import check_dir, rmsd
from .clusterer import cluster_DBSCAN, cluster_KMeans


__all__ = ["get_node_list", "collect_bound_poses", "vectorize_poses", "calculate_scores", "write_bound_poses",
           "calculate_surface_area_wrapper", "analyze_pose_wrapper"]


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
                        lipid, protein_ref, lipid_ref, stride=1, nprot=1):
    """Collected the bound poses based on contact_residue_index.

    Parameters
    ----------
    binding_site_map : dict
    contact_residue_index : dict
    trajfile_list : list of str
    topfile_list : list of str
    lipid : str
    protein_ref : mdtraj.Trajectory
    lipid_ref : mdtraj.Trajectory
    stride : int,default=1
    nprot : int, default=1

    Returns
    ---------
    pose_traj : dict
        Coordinates of all bound poses in stored in a python dictionary {binding_site_id: mdtraj.Trajectory}

    See also
    --------
    pylipid.func.cal_contact_residues
        The function that calculates the contact residues from distance matrix

    """
    joined_top = protein_ref.top.join(lipid_ref.top)
    trajfile_list = np.atleast_1d(trajfile_list)
    topfile_list = np.atleast_1d(topfile_list)
    stride = int(stride)
    nprot = int(nprot)
    pose_pool = defaultdict(list)
    pose_info = defaultdict(list)
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
                            pose_info[bs_id].append((traj_idx, protein_idx, lipid_residue_index, frame_id*traj.timestep))
    pose_traj = {}
    for bs_id in pose_pool.keys():
        pose_traj[bs_id] = md.Trajectory([frame[0] for frame in pose_pool[bs_id]], joined_top,
                                          time=None,
                                          unitcell_angles=[frame[1] for frame in pose_pool[bs_id]],
                                          unitcell_lengths=[frame[2] for frame in pose_pool[bs_id]])

    return pose_traj, pose_info


def vectorize_poses(bound_poses, binding_nodes, protein_atom_indices, lipid_atom_indices):
    """Calculate distance matrix for bound poses.

    Parameters
    ----------
    bound_poses :
    binding_nodes :
    protein_atom_indices :
    lipid_atom_indcies :

    Returns
    -------
    dist_matrix :

    """
    # calculate distance to binding site residues for each lipid atom
    dist_per_atom = np.array(
        [np.array([md.compute_distances(bound_poses,
                                        list(product([lipid_atom_indices[idx]], protein_atom_indices[node])),
                                        periodic=True).min(axis=1) for node in binding_nodes]).T
         for idx in np.arange(len(lipid_atom_indices))])  # shape: [n_lipid_atoms, n_poses, n_BS_residues]
    return dist_per_atom


def calculate_scores(data, kde_bw=0.15, pca_component=0.90, score_weights=None):
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


def calculate_surface_area(trajfile_list, topfile_list, binding_site_map, nprot=1, timeunit="us",
                            stride=1, dt_traj=None, radii_book=None):
    """Calculate surface area in a serial manner.

    Parameters
    ----------
    trajfile_lsit : list of str
    topfile_list : list of str
    binding_site_map : dict,
    nprot : int, default=1
    timeunit : str, default='us'
    stride : int, default=1
    dr_traj : float or None, default=None
    radii_book : dict, default=None

    Returns
    -------
    surface_area : pandas.DataFrame

    """
    surface_data = []
    data_keys = []
    for traj_idx in trange(len(trajfile_list), desc="CALCULATE BINDING SITE SURFACE AREA PER TRAJ"):
        traj = md.load(trajfile_list[traj_idx], top=topfile_list[traj_idx])
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
    surface_area_data = pd.concat(surface_data, keys=data_keys)
    return surface_area_data


def calculate_surface_area_wrapper(trajfile, topfile, traj_idx, binding_site_map={}, nprot=1, timeunit="us",
                                   stride=1, dt_traj=None, radii_book=None):
    """A wrapper function for calculating surface area. """
    traj = md.load(trajfile, top=topfile, stride=stride)
    if dt_traj is None:
        traj_times = traj.time / 1000000.0 if timeunit == "us" else traj.time / 1000.0
    else:
        traj_times = float(dt_traj * stride) * np.arange(traj.n_frames)
    protein_indices_all = traj.top.select("protein")
    natoms_per_protein = int(len(protein_indices_all) / nprot)
    surface_data = []
    data_keys = []
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
    return surface_data, data_keys


def analyze_pose_wrapper(bs_id, poses_of_the_site, nodes_of_the_site, pose_info_of_the_site, pose_dir=None,
                         n_top_poses=3, protein_atom_indices=None, lipid_atom_indices=None, atom_weights=None,
                         kde_bw=0.15, pca_component=0.90, pose_format="gro", n_clusters="auto",
                         eps=None, min_samples=None, metric="euclidean", trajfile_list=None):
    """A wrapper that ranks poses, clusters poses and calculates pose RMSD. """
    if len(poses_of_the_site) == 0:
        print(f"No Bound pose collected from Binding Site {bs_id}! Possibly due to insufficient sampling.")
        return []
    ## rank poses ##
    pose_dir_rank = check_dir(pose_dir, "BSid{}_rank".format(bs_id), print_info=False)
    # lipid_dist_per_atom shape: [n_lipid_atoms, n_bound_poses, n_BS_residues]
    lipid_dist_per_atom = vectorize_poses(poses_of_the_site, nodes_of_the_site,
                                          protein_atom_indices, lipid_atom_indices)
    scores = calculate_scores(lipid_dist_per_atom, kde_bw=kde_bw, pca_component=pca_component,
                              score_weights=atom_weights)
    num_of_poses = min(n_top_poses, poses_of_the_site.n_frames)
    selected_pose_indices = np.argsort(scores)[::-1][:num_of_poses]
    if len(selected_pose_indices) > 0:
        write_bound_poses(poses_of_the_site, selected_pose_indices, pose_dir_rank, pose_prefix="BSid{}_top".format(bs_id),
                          pose_format=pose_format)
        _write_pose_info([pose_info_of_the_site[int(pose_idx)] for pose_idx in selected_pose_indices],
                         f"{pose_dir_rank}/pose_info.txt", trajfile_list)
    ## cluster poses ##
    lipid_dist_per_pose = np.array([lipid_dist_per_atom[:, pose_id, :].ravel()
                                    for pose_id in np.arange(lipid_dist_per_atom.shape[1])])
    pose_dir_clustered = check_dir(pose_dir, "BSid{}_clusters".format(bs_id), print_info=False)
    transformed_data = PCA(n_components=pca_component).fit_transform(lipid_dist_per_pose)
    if n_clusters == 'auto':
        _, core_sample_indices = cluster_DBSCAN(transformed_data, eps=eps, min_samples=min_samples,
                                                metric=metric)
        selected_pose_indices = [np.random.choice(i_core_sample, 1)[0] for i_core_sample in core_sample_indices]
    elif n_clusters > 0:
        cluster_labels = cluster_KMeans(transformed_data, n_clusters=n_clusters)
        cluster_id_set = np.unique(cluster_labels)
        selected_pose_indices = [np.random.choice(np.where(cluster_labels == cluster_id)[0], 1)[0]
                            for cluster_id in cluster_id_set]
    if len(selected_pose_indices) > 0:
        write_bound_poses(poses_of_the_site, selected_pose_indices, pose_dir_clustered,
                          pose_prefix="BSid{}_cluster".format(bs_id), pose_format=pose_format)
        _write_pose_info([pose_info_of_the_site[int(pose_idx)] for pose_idx in selected_pose_indices],
                              f"{pose_dir_clustered}/pose_info.txt", trajfile_list)
    ## calculate RMSD ##
    dist_mean = np.mean(lipid_dist_per_pose, axis=0)
    pose_rmsds = [rmsd(lipid_dist_per_pose[pose_id], dist_mean)
                  for pose_id in np.arange(len(lipid_dist_per_pose))]
    return pose_rmsds


def _write_pose_info(selected_pose_info, fn, trajfile_list):
    """Write pose information for each selected pose. """
    with open(fn, "w") as f:
        for idx, info in enumerate(selected_pose_info):
            f.write("POSE ID : {}\n".format(int(idx)))
            f.write("TRAJ FN : {}\n".format(trajfile_list[info[0]]))
            f.write("PROT ID : {}\n".format(info[1]))
            f.write("LIPID ID: {}\n".format(info[2]))
            f.write("TIME    : {:8.3f} ps\n".format(info[3]))
            f.write("\n")
            f.write("\n")