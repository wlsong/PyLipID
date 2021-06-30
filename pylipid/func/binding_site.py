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
import mdtraj as md
from statsmodels.nonparametric.kernel_density import KDEMultivariate as kde
from sklearn.decomposition import PCA
from ..util import check_dir, rmsd
from .clusterer import cluster_DBSCAN, cluster_KMeans


__all__ = ["get_node_list", "collect_bound_poses", "vectorize_poses", "calculate_scores", "write_bound_poses",
           "calculate_surface_area_wrapper", "analyze_pose_wrapper"]


def get_node_list(corrcoef, threshold=4):
    r"""Calculate community structures from interaction network.

    The interaction network is built using the correlation coeffient matrix, in which the edges are the Pearson correlation
    of the two connecting nodes. The community structures of this network is calculated using the Louvain algorithm [1]_,
    which find high modularity network partitions. The modularity is defined as [2]_:

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

    By default, this method returns communities containing at least 4 nodes. This setting can be changed by using
    the parameter ``threshold``.

    Parameters
    -----------
    corrcoef : ndarray(n, n)
        The Pearson correlation matrix.

    threshold : int, default=4
        Size of communities. Only communities with more nodes than the threshold will be returned.

    Returns
    --------
    node_list : list of lists
        A list of community nodes.

    modularity : float or None
        The modularity of network partition. It measure the quality of network partition. The value is between 1 and
        -1. The bigger the modularity, the better the partition.

    References
    ----------
    .. [1] Blondel, V. D.; Guillaume, J.-L.; Lambiotte, R.; Lefebvre, E., Fast unfolding of communities in large
           networks. Journal of Statistical Mechanics: Theory and Experiment 2008, 2008 (10), P10008

    .. [2] Newman, M. E. J., Analysis of weighted networks. Physical Review E 2004, 70 (5), 056131.

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
    """Collected the bound poses from trajectories.

    The bound poses of binding sites specified in the ``binding_site_map`` are collected from trajectories based on the
    contacting lipid residue index that is provided by ``contact_residue_index``.

    In implementation, the ``binding_site_map provides`` a dictionary that informs of what residues constitues the selected
    binding sites. The ``contact_residue_index`` is the lipid index generated by
    :meth:`pylipid.api.LipidInteraction.collect_residue_contacts`, which is a python dictionary that records,
    for each residue, the contacting lipid indices at each trajectory frame. The contacting lipids at each trajectory frame
    are calculated by merging, with duplicates removed, the contacting lipid molecules of the binding site residues. The
    coordinates of the contacting lipid molecules, together with the protein coordinates that the lipid poses bind are taken
    from the trajectories and made into a MDtraj.Trajectory object, which contains coordinates and topology information of
    the protein-lipid complex and the unitcell information of the simulation box at the time the poses were taken.

    ``protein_ref`` and ``lipid_ref`` are the MDtraj topology object (mdtraj.Topology) of a copy of the  protein molecule
    and a copy of the lipid molecule respectively. These two are combined to generate the topology object for the
    protein-lipid complex which is required for mdtraj.Trajectory object of the bound poses.

    The simulation trajectories are read in this function to obtain the lipid and receptor coordinates. Therefore, the
    order of the ``trajfile_list`` and ``topfile_list`` and the settings of ``stride`` and ``nprot`` should keep consistent
    with the settings that generated the contacting lipid index ``contact_residue_index``.

    Parameters
    ----------
    binding_site_map : dict
        A python dictionary with the selected binding site IDs as its keys and the binding site residue indices as the
        corresponding values. {binding site ID: a list of binding site residue indices}

    contact_residue_index : dict
        A python dictionary that records contacting lipid molecule indices for residues. It should has residue indices
        as its keys and a list of contacting lipid molecule residue indices for each of the trajectory frame as their
        corresponding values.

    trajfile_list : list of str
        A list of trajectory files.

    topfile_list : list of str
        A list of topology files.

    lipid : str
        Residue name of the lipid to check.

    protein_ref : mdtraj.Topology or mdtraj.Trajectory
        A MDtraj object of a copy of the protein molecule that contains topology information of the protein.

    lipid_ref : mdtraj.Trajectory
        A MDtraj object of a copy of the lipid molecule that contains topology information of the lipid molecule.

    stride : int,default=1
        Analyze every stride-th frame of the trajectories. Should keep consistent with the settings that generated the
        contacting lipid index ``contact_residue_index``.

    nprot : int, default=1
        Number of copies of proteins in the simulation systems. Should keep consistent with the settings that generated
        the contacting lipid index ``contact_residue_index``.

    Returns
    ---------
    pose_traj : dict
        A python dictionary that stores bound pose coordinates. This python dictionary has the selected binding site IDs
        as its keys and each corresponding value contains a mdtraj.Trajectory object that contains the bound lipid coordinates
        with the receptor coordinates the lipid bind to.

    pose_info : dict
        A python dictionary that stores the corresponding information to the lipid poses in ``pose_traj``. The pose
        information is a tuple of four figures, which correspond to traj_idx, protein_idx, lipid_residue_index,
        frame_id*traj.timestep.

    See also
    --------
    pylipid.func.cal_contact_residues
        The function that calculates the contact residues from distance matrix
    pylipid.api.LipidInteraction.collect_residue_contacts
        The function that creates contact lipid index.
    pylipid.api.LipidInteraction.analyze_bound_poses
        The function that analyzes lipid bound poses.

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
    """Convert the bound poses to distance vectors.

    This function takes the protein-lipid complex structure, and translate the bound lipid poses to distance vectors,
    which contains the minimum distances of each of the lipid molecule atom to the binding site residues. It uses
    the function `mdtraj.compute_distances <https://mdtraj.org/1.9.4/api/generated/mdtraj.compute_distances.html>`_
    to calculate atom-wise distances.

    Parameters
    ----------
    bound_poses : mdtraj.Trajectory
        mdtraj.Trajectory object that contain protein-lipid complex coordinates.

    binding_nodes : list
        Binding site residue indices.

    protein_atom_indices : list
        A list of residue atom indices in the protein-lipid complex structure, i.e. [[0,1,2],[3,4,5],[6,7]] means the
        first residue contains atom 0,1,2, and the second residue contains atom 3,4,5.

    lipid_atom_indcies : list
        The atom indices of the lipid molecule in the protein-lipid complex structure.

    Returns
    -------
    dist_matrix : numpy.ndarray
        The distance matrix of the bound poses, in the shape of [n_lipid_atoms, n_poses, n_binding_site_residues]

    """
    # calculate distance to binding site residues for each lipid atom
    dist_per_atom = np.array(
        [np.array([md.compute_distances(bound_poses,
                                        list(product([lipid_atom_indices[idx]], protein_atom_indices[node])),
                                        periodic=True).min(axis=1) for node in binding_nodes]).T
         for idx in np.arange(len(lipid_atom_indices))])  # shape: [n_lipid_atoms, n_poses, n_BS_residues]
    return dist_per_atom


def calculate_scores(dist_matrix, kde_bw=0.15, pca_component=0.90, score_weights=None):
    r"""Calculate scores based on probability density.

    This function first lower the dimension of dist_matrix by using a
    `PCA <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_. Then the distribution of
    the distance vectors for each atom is estimated using
    `KDEMultivariate <https://www.statsmodels.org/devel/generated/statsmodels.nonparametric.kernel_density.KDEMultivariate.html>`_.

    The score of a lipid pose is calculated based on the probability density function of the atom positions in the binding
    site and weights given to the atoms:

    .. math::
        \text { score }=\sum_{i} W_{i} \cdot \hat{f}_{i, H}(D)

    where :math:`W_{i}` is the weight given to atom i of the lipid molecule, H is the bandwidth and
    :math:`\hat{f}_{i, H}(D)` is a multivariate kernel density etimation of the position of atom i in the specified
    binding site. :math:`\hat{f}_{i, H}(D)` is calculated from all the bound lipid poses in that binding site.

    Parameters
    ----------
    dist_matrix : numpy.ndarray, shape=(n_lipid_atoms, n_poses, n_binding_site_residues)
        The distance vectors describing the position of bound poses in the binding site. This dist_matrix can be
        generated by :meth:`~vectorize_poses`.

    kde_bw : scalar, default=0.15
        The bandwidth for kernel density estimation. Used by
        `KDEMultivariate <https://www.statsmodels.org/devel/generated/statsmodels.nonparametric.kernel_density.KDEMultivariate.html>`_.
        By default, the bandwidth is set to 0.15nm which roughly corresponds to the vdw radius of MARTINI 2 beads.

    pca_component : scalar, default=0.9
        The number of components to keep. if ``0 < pca_component<1``, select the number of components such that the
        amount of variance that needs to be explained is greater than the percentage specified by n_components. It is used
        by `PCA <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_.

    score_weights : None or dict
        A dictionary that contains the weight for n_lipid_atoms, {idx_atom: weight}

    Returns
    ---------
    scores : numpy.ndarray, shape=(n_samples,)
        Scores for bound poses.

    See also
    --------
    pylipid.func.collect_bound_poses
        Collect bound poses from trajectories.
    pylipid.func.vectorize_poses
        Convert bound poses to distance vectors.

    """
    weights = {atom_idx: 1 for atom_idx in np.arange(np.shape(dist_matrix)[0])}
    if score_weights is not None:
        weights.update(score_weights)

    kde_funcs = {}
    try:
        for atom_idx in np.arange(np.shape(dist_matrix)[0]):
            transformed_data = PCA(n_components=pca_component).fit_transform(dist_matrix[atom_idx])
            var_type = ""
            bw = []
            for dummy in range(len(transformed_data[0])):
                var_type += "c"
                bw.append(kde_bw)
            kde_funcs[atom_idx] = kde(data=transformed_data, var_type=var_type, bw=bw)
        # evaluate binding poses
        scores = np.sum([weights[atom_idx] * kde_funcs[atom_idx].pdf()
                         for atom_idx in np.arange(np.shape(dist_matrix)[0])], axis=0)
        return scores

    except ValueError:
        print("Pose generation error -- possibly due to insufficient number of binding event.")


def write_bound_poses(pose_traj, pose_indices, save_dir, pose_prefix="BoundPose", pose_format="pdb"):
    """Write selected bound poses to disc.

    The ``pose_traj`` is a mdtraj.Trajectory object. The selected bound poses, i.e. selected frames of this mdtraj.Trajectory
    object, are written to disc by mdtraj.Trajectory.save() function.

    Parameters
    ----------
    pose_traj : mdtraj.Trajectory
        Bound poses coordinates object.

    pose_indices : array_like
        The indices of the poses, i.e. the indices of the frames, to be written.

    save_dir : str
        The directory to save the written coordinate files.

    pose_prefix : str, optional, default=BoundPose"
        The prefix of the coordinate files.

    pose_format : str, optional, default="gro"
        The format of the coordinate files. Support formats included by mdtraj.Trajectory.save().

    """
    for idx, pose_id in enumerate(pose_indices):
        pose_traj[pose_id].save(os.path.join(save_dir, "{}{}.{}".format(pose_prefix, idx, pose_format)))
    return


def analyze_pose_wrapper(bs_id, bound_poses, binding_nodes, pose_info, pose_dir=None,
                         n_top_poses=3, protein_atom_indices=None, lipid_atom_indices=None, atom_weights=None,
                         kde_bw=0.15, pca_component=0.90, pose_format="gro", n_clusters="auto",
                         eps=None, min_samples=None, metric="euclidean", trajfile_list=None):
    r"""A wrapper function that ranks poses, clusters poses and calculates pose RMSD.

    A wrapper function is to assist the calculation using `p_tqdm <https://github.com/swansonk14/p_tqdm>`_, which is
    a multiprocessing library incorporated with progress bars. So some of the setting or parameters are used to
    assist the use of multiprocessing.

    This function uses :meth:`~vectorize_poses` to convert bound poses into distance vectors, which measure the relative
    position of lipid atoms in their binding site. Then it uses :meth:`~calculate_scores` to calculate the density distributions
    of lipid atoms and score the bound poses based on the probability density function of the position of lipid atoms. Based on
    the scores, a couple of top-ranked poses (determined by ``n_top_poses``) are written out in a coordinate format
    determined by ``pose_format``. The top-ranked poses for a binding site are saved in the directory of BSidP{bs_id}_rank.

    This function also clusters the bound poses. The bound poses are clustered using their distance vectoris, i.e. their
    relative positions in the binding site. If ``n_clusters`` is set to `auto`, this function will cluster the bound poses
    using :meth:`pylipid.func.cluster_DBSCAN`. DBSCAN finds clusters of core samples of high density. If ``n_clusters``
    is given an integer larger than 0, this function will cluster the lipid bound poses using :meth:`pylipid.func.cluster_KMeans`.
    The KMeans cluster separates the samples into `n` clusters of equal variances, via minimizing the `inertia`.

    For writing out the cluster poses, this function will randomly select one pose from each cluster in the case of
    using KMeans or one from the core samples of each cluster when DBSCAN is used, and writes the selected protein-lipid
    coordinates in the provided pose format ``pose_format``. The clustered poses are saved in the directory BSid{bs_id}_clusters.

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
    bs_id : int
        Binding Site ID. Used in creating directory for stoing poses.

    bound_poses : mdtraj.Trajectory
        A mdtraj.Trajectory object that contains protein-lipid complex coordinates.

    binding_nodes : list
        A list of residue indices of binding site residues.

    pose_info : list
        A list of tuples of information of the bound poses in ``bound_poses``. A info tuple contains four figures,
        which correspond to traj_idx, protein_idx, lipid_residue_index, frame_id*traj.timestep. These pose_info tuples
        and the bound poses are generated by :meth:`~collect_bound_poses`.

    pose_dir : str or None
        The directory to save representative bound poses and clustered poses. If None, save at the current working directory.

    n_top_poses : int, default=3
        Number of representative bound poses selected to write to disc.

    protein_atom_indices : list
        A list of residue atom indices in the protein-lipid complex structure, i.e. [[0,1,2],[3,4,5],[6,7]] means the
        first residue contains atom 0,1,2, and the second residue contains atom 3,4,5. Used when vectorizing the bound poses
        by :meth:`~vectorize_poses`.

    lipid_atom_indcies : list
        The atom indices of the lipid molecule in the protein-lipid complex structure. Used when vectorizing the bound poses
        by :meth:`~vectorize_poses`.

    atom_weights : None or dict
        A dictionary that contains the weight for n_lipid_atoms, {idx_atom: weight}. Used when scoring the bound poses by
        :meth:`~calculate_scores`.

    kde_bw : calar, default=0.15
        The bandwidth for kernel density estimation. Used when estimating the kernel density of lipid atom positions by
        :meth:`~calculate_scores`. By default, the bandwidth is set to 0.15nm which roughly corresponds  to the vdw
        radius of MARTINI 2 beads.

    pca_component : scalar, default=0.9
        The number of components to keep. if ``0 < pca_component<1``, select the number of components such that the
        amount of variance that needs to be explained is greater than the percentage specified by n_components. It is used
        to decrease the dimentions of distance vectors when estimating the kernel density, by :meth:`~calculate_scores`.

    pose_format : str, optional, default="gro"
        The format of the coordinate files. Support formats included by mdtraj.Trajectory.save(). Used by
        :meth:`~write_bound_poses`.

    n_clusters : int or "auto", default="auto"
        Number of clusters to find in the bound poses. When set to `auto`, :meth:`pylipid.func.cluster_DBSCAN` will be
        used for clustering. When a integer is given, :meth:`pylipid.func.cluster_KMeans` will be used.

    eps : float or None, default=None
        The maximum distance between two samples for one to be considered as in the neighborhood of the other. Used by
        :meth:`pylipid.func.cluster_DBSCAN`.

    min_samples : int or None, default=None
        The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. Used by
        :meth:`pylipid.func.cluster_DBSCAN`.

    metric : string, or callable, default=’euclidean’
        The metric to use when calculating distance between instances in a feature array. Used by :meth:`pylipid.func.cluster_DBSCAN`.

    trajfile_list : str or None, default=None
        Used to write pose information, i.e. what trajectory, which timestep and which lipid molecule the written pose
        was taken from.

    Returns
    -------
    pose_rmsds : list
        A list of bound pose RMSDs.

    See Also
    --------
    pylipid.func.vectorize_poses
        Convert the bound poses to distance vectors.
    pylipid.func.calculate_scores
        Calculate scores based on probability density.
    pylipid.func.cluster_DBSCAN
        Cluster data using DBSCAN.
    pylipid.func.cluster_KMeans
        Cluster data using KMeans.

    """
    if len(bound_poses) == 0:
        print(f"No Bound pose collected from Binding Site {bs_id}! Possibly due to insufficient sampling.")
        return []
    ## rank poses ##
    pose_dir_rank = check_dir(pose_dir, "BSid{}_rank".format(bs_id), print_info=False)
    # lipid_dist_per_atom shape: [n_lipid_atoms, n_bound_poses, n_BS_residues]
    lipid_dist_per_atom = vectorize_poses(bound_poses, binding_nodes,
                                          protein_atom_indices, lipid_atom_indices)
    scores = calculate_scores(lipid_dist_per_atom, kde_bw=kde_bw, pca_component=pca_component,
                              score_weights=atom_weights)
    num_of_poses = min(n_top_poses, bound_poses.n_frames)
    selected_pose_indices = np.argsort(scores)[::-1][:num_of_poses]
    if len(selected_pose_indices) > 0:
        write_bound_poses(bound_poses, selected_pose_indices, pose_dir_rank, pose_prefix="BSid{}_top".format(bs_id),
                          pose_format=pose_format)
        _write_pose_info([pose_info[int(pose_idx)] for pose_idx in selected_pose_indices],
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
        write_bound_poses(bound_poses, selected_pose_indices, pose_dir_clustered,
                          pose_prefix="BSid{}_cluster".format(bs_id), pose_format=pose_format)
        _write_pose_info([pose_info[int(pose_idx)] for pose_idx in selected_pose_indices],
                              f"{pose_dir_clustered}/pose_info.txt", trajfile_list)
    ## calculate RMSD ##
    dist_mean = np.mean(lipid_dist_per_pose, axis=0)
    pose_rmsds = [rmsd(lipid_dist_per_pose[pose_id], dist_mean)
                  for pose_id in np.arange(len(lipid_dist_per_pose))]
    return pose_rmsds


def calculate_surface_area_wrapper(trajfile, topfile, traj_idx, binding_site_map={}, nprot=1, timeunit="us",
                                   stride=1, dt_traj=None, radii=None):
    """A wrapper function for calculating surface area.

    A wrapper function is to assist the calculation using `p_tqdm <https://github.com/swansonk14/p_tqdm>`_, which is
    a multiprocessing library incorporated with progress bars. So some of the setting or parameters are used to
    assist the use of multiprocessing.

    In this function, the provided trajectory is read by mdtraj.read() and the solvent accessible surface of each residue
    is calculated by `mdtraj.shrake_rupley <https://mdtraj.org/1.9.4/api/generated/mdtraj.shrake_rupley.html>`_. The
    binding site surface area is calculated as the sum of surface areas of its comprising residues. The binding sites are
    defined by ``binding_site_map`` which is a python dictionary object that includes binding site IDs as its keys and
    the residue indices of the binding site residues as the corresponding values.

    The calculation of surface area requires a definition of the protein atom radius. MDtraj defines the radii for common
    atoms (see `here <https://github.com/mdtraj/mdtraj/blob/master/mdtraj/geometry/sasa.py#L56>`_). The radius of the BB
    bead in MARTINI2 is defined as 0.26 nm, the SC1/SC2/SC3 are defined as 0.23 nm in this function. Use the param
    ``radii`` to define or change of definition of atom radius.

    The returned data ``surface_area`` is a list of pandas.DataFrame objects, and each of these pandas.DataFrame object
    contains surface area data for one copy of the proteins in the simulation system. That is, if the system has N copies
    of the receptor, ``surface_area`` will have N pandas.DataFrame objects. Each pandas.DataFrame contains the surface
    areas as a function of time for the selected binding site are shown by column with the column name of
    "Binding Site {idx}". This pandas.DataFrame object also has a "Time" column which records the timestep at which the
    surface areas are measured. The other returned data ``data_keys`` contains a list of tuples of (traj_idx, protein_idx),
    which corresponds to the list of pandas.DataFrame objects of ``surface_area``. The reason for having this arrangement
    is to assist the generation of a big pandas.DataFrame that includes surface area data from all trajectories in
    :meth:`pylipid.api.LipidInteraction.compute_surface_area` when used in combination with ``p_tqdm``.

    Parameters
    ----------
    trajfile : str
        Trajectory filename.

    topfile : str
        Topology filename.

    traj_idx : idx
        The index of the given trajectory in the original trajectory list.

    binding_site_map : dict
        A python dictionary to defines the selected binding site IDs and the residue indices of the binding site residues

    nprot : int, default=1
        Number of copies of the receptor in the simulation system.

    timeunit : {"us", "ns"}
        The time unit used for reporting the timestpes.

    stride : int, default=1
        Analyze every stride-th frame from the trajectory.

    dt_traj : int or None
        Timestep of trajectories. It is required when trajectories do not have timestep information. Not needed for
        trajectory formats of e.g. xtc, trr etc. If None, timestep information will take from trajectories.

    radii : dict or None
        Protein atom radii {atom name: radius}. The radius is reported in nanometer.

    Returns
    -------
    surface_area : list
        A list of {nprot} pandas.DataFrame objects. Each pandas.DataFrame object records the surface areas as a function
        of time for the selected binding sites on one copy of the receptor in the system.

    data_keys : list
        A list of tuples (traj_idx, protein_idx) which are the traj index and protein index information of the
        pandas.DataFrame objects in ``surface_area``.

    See Also
    --------
    pylipid.api.LipidInteraction.compute_surface_area
        Calculate binding site surface area from a list of trajectories.

    """
    MARTINI_CG_radii = {"BB": 0.26, "SC1": 0.23, "SC2": 0.23, "SC3": 0.23}
    if radii is None:
        radii_book = MARTINI_CG_radii
    else:
        radii_book = {**MARTINI_CG_radii, **radii}
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