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

"""This module contains functions that deals with trajectories. """

from collections import defaultdict
import numpy as np


__all__ = ["get_traj_info"]


def get_traj_info(traj, lipid, lipid_atoms=None, resi_offset=0, nprot=1, protein_ref=None, lipid_ref=None):
    """Get trajectory information regarding atom/residue index and topologies.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        A mdtraj.Trajectory object.
    lipid : str
        The residue name of the lipid to check.
    lipid_atoms : a list of str; opt
        The names of lipid atoms that are used to define lipid interaction and lipid binding sites.
        Default is None, that is all the lipid atoms will be used for calculation.
    resi_offset : int, optional, default=0
        Shift of residue index. The new residue index (i.e. the original index + resi_offset) will
        be used in all the generated data.
    nprot : int, optional, default=1
        Number of protein copies in the systems. If nprot >= 2, the protein copies need to be identical,
        and the generated data will be the averages of the copies.
    protein_ref : None or mdtraj.Trajectory, optional, default=None
        A mdtraj.Trajectory object that stores the topology and coordinates of a copy of the protein structure.
    lipid_ref : None or mdtraj.Trajectory, optional, default=None
        A mdtraj.Trajectory object that stores the topology and coordinates of a lipid molecule structure.

    Returns
    -------
    traj_info : dict
        A dictionary that contains the topology information of `traj`.
    protein_ref :  mdtraj.Trajectory
        A mdtraj.Trajectory object that stores the topology and coordinates of a copy of the protein structure.
    lipid_ref : mdtraj.Trajectory
        A mdtraj.Trajectory object that stores the topology and coordinates of a lipid molecule structure.

    """
    lipid_index_dict = defaultdict(list)
    lipid_atom_indices = traj.top.select("resn {}".format(lipid))
    if lipid_atoms is not None:
        for atom_index in lipid_atom_indices:
            if traj.top.atom(atom_index).name in lipid_atoms:
                lipid_index_dict[traj.top.atom(atom_index).residue.index].append(atom_index)
    else:
        for atom_index in lipid_atom_indices:
            lipid_index_dict[traj.top.atom(atom_index).residue.index].append(atom_index)
    lipid_residue_atomid_list = [lipid_index_dict[resi] for resi in np.sort(list(lipid_index_dict.keys()))]
    # get protein atom indices
    all_protein_atom_indices = traj.top.select("protein")
    natoms_per_protein = int(len(all_protein_atom_indices)/nprot)
    protein_residue_atomid_list = []
    for protein_idx in np.arange(nprot):
        chain_index_dict = defaultdict(list)
        for atom_index in all_protein_atom_indices[protein_idx*natoms_per_protein:(protein_idx+1)*natoms_per_protein]:
            chain_index_dict[traj.top.atom(atom_index).residue.index].append(atom_index)
        protein_residue_atomid_list.append([chain_index_dict[resi] for resi in np.sort(list(chain_index_dict.keys()))])
    protein_residue_id = np.arange(len(protein_residue_atomid_list[0]), dtype=int)
    residue_list = np.array(["{}{}".format(traj.top.atom(residue_atoms[0]).residue.resSeq+resi_offset,
                                           traj.top.atom(residue_atoms[0]).residue.name)
                             for residue_atoms in protein_residue_atomid_list[0]], dtype=str)
    if lipid_ref is None:
        one_lipid_indices = []
        for lipid_id in np.sort(traj.top.select("resn {}".format(lipid))):
            if len(one_lipid_indices) == 0:
                one_lipid_indices.append(lipid_id)
            elif traj.top.atom(lipid_id).residue.index != traj.top.atom(one_lipid_indices[-1]).residue.index:
                break
            else:
                one_lipid_indices.append(lipid_id)
        lipid_ref = traj[0].atom_slice(np.unique(one_lipid_indices))
    if protein_ref is None:
        protein_ref = traj[0].atom_slice(all_protein_atom_indices[:natoms_per_protein])

    traj_info = {"protein_residue_atomid_list": protein_residue_atomid_list,
                 "lipid_residue_atomid_list": lipid_residue_atomid_list,
                 "protein_residue_id": protein_residue_id, "residue_list": residue_list}

    return traj_info, protein_ref, lipid_ref
