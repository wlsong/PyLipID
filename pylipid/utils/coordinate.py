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

"""This module contains functions for writing coordinates in PDB format"""

import numpy as np

__all__ = ["write_PDB"]

def write_PDB(prot_obj, bfactor, pdb_fn, resi_offset=0):
    """Write interaction data in bfactor columns.

    Parameters
    ----------
    prot_obj : mdtraj.TrajectoryObject
    bfactor : array_like
    pdb_fn : str
    resi_offset : int, optional, default=0

    """
    coords = prot_obj.xyz[0]
    table = prot_obj.top.to_dataframe()[0]
    atom_idx_set = table.serial
    resid_set = table.resSeq + resi_offset
    atom_name_set = table.name
    resn_set = table.resName
    chainID = [chr(65 + int(idx)) for idx in table.chainID]
    atom_residue_map = {atom_idx: prot_obj.top.atom(atom_idx).residue.index
                        for atom_idx in np.arange(prot_obj.n_atoms)}
    ######## write out coords ###########
    with open(pdb_fn, "w") as f:
        for idx in np.arange(prot_obj.n_atoms):
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
                                 "BFACTOR": bfactor[atom_residue_map[idx]]}
            row = "{HEADER:6s}{ATOM_ID:5d} ".format(**coords_dictionary) + \
                  "{ATOM_NAME:^4s}{SPARE:1s}{RESN:3s} ".format(**coords_dictionary) + \
                  "{CHAIN_ID:1s}{RESI:4d}{SPARE:1s}   ".format(**coords_dictionary) + \
                  "{COORDX:8.3f}{COORDY:8.3f}{COORDZ:8.3f}{OCCUP:6.2f}{BFACTOR:6.2f}\n".format(**coords_dictionary)
            f.write(row)
        f.write("TER")

    return