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
import community
import networkx as nx
import numpy as np
from tqdm import trange, tqdm
import mdtraj as md


__all__ = ["get_node_list", "collect_binding_poses", "write_binding_poses"]


def get_node_list(corrcoef, size=4):
    """Calculate binding sites from correlation coeffient matrix.

    Binding sites are calculated as the community structures of the network built
    on the correlation coefficient matrix.

    Parameters
    -----------
    corrcoef : ndarray(n, n)
    size : int, optional, default=4

    Returns
    --------
    node_list : list of lists

    """
    # TODO: check negative values in corrcoef_matrix. Come up with better solutions.
    corrcoef[corrcoef < 0.0] = 0.0 # network edge can't take negative values. Residues with
                                   # negative correlationsare are forced to separate to different binding sites.
    residue_network = nx.Graph(corrcoef)
    part = community.best_partition(residue_network, weight='weight')
    values = [part.get(node) for node in residue_network.nodes()]
    node_list = []
    for value in range(max(values)):
        nodes = [k for k, v in part.items() if v == value]
        if len(nodes) >= size:
            node_list.append(nodes)

    return node_list


def collect_binding_poses(binding_site_map, contact_list, trajfile_list, topfile_list, stride, lipid, nprot):
    """Collected the bound poses based on contact_list.

    Parameters
    ----------
    binding_site_map : dict
    contact_list : list of lists
    trajfile_list : list of str
    topfile_list : list of str
    lipid : str
    nprot : int

    Returns
    ---------
    pose_pool : dict

    See also
    --------
    pylipid.funcs.cal_contact_residues
        The function that calculates the contact residues from distance matrix

    """
    from collections import defaultdict as _dict

    pose_pool = _dict(list)
    for traj_idx in trange(len(trajfile_list), desc="COLLECT BINDING POSES", total=len(trajfile_list)):
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
                    [contact_list[node][list_to_take][frame_idx] for node in nodes]))
                    for frame_idx in range(traj.n_frames)]
                for frame_id in range(len(contact_BS)):
                    if len(contact_BS[frame_id]) > 0:
                        for lipid_id in contact_BS[frame_id]:
                            lipid_residue_index = lipid_residue_index_set[lipid_id]
                            lipid_atom_indices = np.sort(
                                [atom.index for atom in traj.top.residue(lipid_residue_index).atoms])
                            pose_pool[bs_id].append(
                                [np.copy(traj.xyz[frame_id, np.hstack([protein_indices, lipid_atom_indices])]),
                                 np.copy(traj.unitcell_angles[frame_id]), np.copy(traj.unitcell_lengths[frame_id])])

        return pose_pool


def write_binding_poses(pose_pool, binding_site_map, protein_ref, lipid_ref, save_dir,
                        n_poses=5, pose_format="gro", kde_bw=0.15, score_weights=None):
    """Write representative binding poses for binding sites.

    Parameters
    ----------
    pose_pool : dict
    binding_site_map : dict
    protein_ref : md.Trajectory object
    lipid_ref : md.Trajectory object
    save_dir : str
    n_poses : int, optional, default=5
    psoe_format : str, optional, default="gro"
    kde_bw : float, optional, default=0.15
    score_weights : dict or None, optional, default=None

    Notes
    --------

    """
    from statsmodels.nonparametric.kernel_density import KDEMultivariate as _kde
    from sklearn.decomposition import PCA as _pca
    from itertools import product as _product

    # update weights for calculated scoring functions.
    lipid_atom_map = {atom.index: atom.name for atom in lipid_ref.top.atoms}
    weights = {name: 1 for index, name in lipid_atom_map.items()}
    if score_weights is not None:
        weights.update(score_weights)

    # prepare topology
    protein_atom_indices = [[atom.index for atom in residue.atoms] for residue in protein_ref.top.residues]
    lipid_atoms = [protein_ref.n_atoms + atom_idx for atom_idx in np.arange(lipid_ref.n_atoms)]
    joined_top = protein_ref.top.join(lipid_ref.top)

    # start pose generation
    for bs_id in tqdm(pose_pool.keys(), desc='WRITE BINDING POSES', total=len(pose_pool.keys())):
        num_of_poses = n_poses if n_poses <= len(pose_pool[bs_id]) else len(pose_pool[bs_id])
        nodes = binding_site_map[bs_id]
        new_traj = md.Trajectory([frame[0] for frame in pose_pool[bs_id]], joined_top,
                                 time=np.arange(pose_pool[bs_id]),
                                 unitcell_angles=[frame[1] for frame in pose_pool[bs_id]],
                                 unitcell_lengths=[frame[2] for frame in pose_pool[bs_id]])
        # calculate distance to binding site residues for each lipid atom
        dist_per_atom = [[md.compute_distances(new_traj,
                                               list(_product([lipid_atoms[idx]], protein_atom_indices[node])),
                                               periodic=True).min(axis=1)
                          for node in nodes] for idx in np.arange(lipid_ref.n_atoms)]

        # calculate probability density function using the distance vectors.
        kde_funcs = {}
        try:
            for atom_idx in np.arange(lipid_ref.n_atoms):
                transformed_data = _pca(n_components=0.95).fit_transform(np.array(dist_per_atom[atom_idx]).T)
                var_type = ""
                bw = []
                for dummy in range(len(transformed_data[0])):
                    var_type += "c"
                    bw.append(kde_bw)
                kde_funcs[atom_idx] = _kde(data=transformed_data, var_type=var_type, bw=bw)
            # evaluate binding poses
            scores = np.sum([weights[lipid_atom_map[idx]] * kde_funcs[idx].pdf()
                             for idx in np.arange(lipid_ref.n_atoms)], axis=0)
            selected_indices = np.argsort(scores)[::-1][:num_of_poses]
            ###############################
            for pose_id in np.arange(num_of_poses, dtype=int):
                new_traj[selected_indices[pose_id]].save(os.path.join(save_dir, "BSid{}_No{}.{}".format(bs_id, pose_id,
                                                                                                        pose_format)))
        except ValueError:
            # exception for num. of poses being smaller than n_components of kde
            with open(os.path.join(save_dir, "Error.txt"), "a") as error_file:
                error_file.write(
                    "BSid {}: Pose generation error -- possibly due to insufficient number of binding event.\n".format(
                        bs_id))

    return


def calculate_surface_area(binding_site_map, radii_book, trajfile_list, topfile_list,
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
    dt_traj : float or None, optional, default=None

    Returns
    -------
    surface_area : pandas.DataFrame

    """
    import pandas as _pd

    surface_data = []
    data_keys = []
    for traj_idx in trange(len(trajfile_list), desc="CALCULATE SURFACE AREA", total=len(trajfile_list)):
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
            column_names = [f"Binding Site {bs_id}" for bs_id, nodes in binding_site_map.items()]
            column_names.append("Time")
            surface_data.append(_pd.DataFrame(np.array(selected_data).T, columns=column_names))
            data_keys.append((traj_idx, protein_idx))
    surface_area = _pd.concat(surface_data, keys=data_keys)

    return surface_area


