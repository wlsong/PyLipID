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

"""This module contains functions that write PyMol scripts."""

import os
from shutil import copyfile

__all__ = ["write_pymol_script"]


def write_pymol_script(fname, pdb_fname, data_fname, lipid, n_binding_site):
    """Write PyMol script.

    Parameters
    ----------
    fname : str
    pdb_fname : str
    data_fname : str
    lipid : str
    n_binding_site : int

    """
    script_fname = os.path.abspath(fname)
    save_dir, _ = os.path.split(script_fname)
    pdb_new_fname = os.path.join(save_dir, os.path.basename(pdb_fname))
    copyfile(os.path.abspath(pdb_fname), pdb_new_fname)
    data_fname = os.path.abspath(data_fname)

    text = """
import numpy as np
import re
import pymol
from pymol import cmd
pymol.finish_launching()

########## files to process ##########
csv_file = "{CSV_FILE}"
pdb_file = "{PDB}"

########## set the sphere scales to corresponding value ##########
value_to_show = "Residence Time"

###### reading data from csv file ##########
num_of_binding_site = {N_BINDING_SITE}

with open(csv_file, "r") as f:
    data_lines = f.readlines()

column_names = data_lines[0].strip().split(",")
for column_idx, column_name in enumerate(column_names):
    if column_name == "Residue":
        column_id_residue_list = column_idx
    elif column_name == "Residue ID":
        column_id_residue_index = column_idx
    elif column_name == "Binding Site ID":
        column_id_BS = column_idx
    elif column_name == value_to_show:
        column_id_value_to_show = column_idx

residue_list = []
residue_rank_set = []
binding_site_identifiers = []
values_to_show = []
for line in data_lines[1:]:
    data_list = line.strip().split(",")
    residue_list.append(data_list[column_id_residue_list])
    residue_rank_set.append(data_list[column_id_residue_index])
    binding_site_identifiers.append(float(data_list[column_id_BS]))
    values_to_show.append(data_list[column_id_value_to_show])

############## read information from pdb file ##########
with open(pdb_file, "r") as f:
    pdb_lines = f.readlines()
residue_identifiers = []
for line in pdb_lines:
    line_stripped = line.strip()
    if line_stripped[:4] == "ATOM":
        identifier = (line_stripped[22:26].strip(), line_stripped[17:20].strip(), line_stripped[21].strip())
##                           residue index,              resname,                     chain id
        if len(residue_identifiers) == 0:
            residue_identifiers.append(identifier)
        elif identifier != residue_identifiers[-1]:
            residue_identifiers.append(identifier)

######### set sphere scales ###############
values_to_show = np.array(values_to_show, dtype=float)
MIN = np.percentile(np.unique(values_to_show), 5)
MAX = np.percentile(np.unique(values_to_show), 100)
X = (values_to_show - np.percentile(np.unique(values_to_show), 50))/(MAX - MIN)
SCALES = 1/(0.5 + np.exp(-X * 5))

######## some pymol settings #########
cmd.set("retain_order", 1)
cmd.set("cartoon_oval_length", 1.0)
cmd.set("cartoon_oval_width", 0.3)
cmd.set("cartoon_color", "white")
cmd.set("stick_radius", 0.35)

##################################
cmd.load(pdb_file, "Prot_{LIPID}")
prefix = "Prot_{LIPID}"
cmd.hide("everything")
cmd.show("cartoon", prefix)
cmd.center(prefix)
cmd.orient(prefix)
colors = np.array([np.random.choice(np.arange(256, dtype=float), size=3) for dummy in range(num_of_binding_site)])
colors /= 255.0
            """.format(**{"SAVE_DIR": save_dir, "LIPID": lipid, "N_BINDING_SITE": n_binding_site,
                          "PDB": pdb_new_fname, "CSV_FILE": data_fname})

    text += r"""
residue_list = np.array(residue_list, dtype=str)
residue_rank_set = np.array(residue_rank_set, dtype=int)
binding_site_identifiers = np.array(binding_site_identifiers, dtype=int)
residue_identifiers = list(residue_identifiers)
for bs_id in np.arange(num_of_binding_site):
    cmd.set_color("tmp_{}".format(bs_id), list(colors[bs_id]))
    for entry_id in np.where(binding_site_identifiers == bs_id)[0]:
        selected_residue = residue_list[entry_id]
        selected_residue_rank = residue_rank_set[entry_id]
        identifier_from_pdb = residue_identifiers[selected_residue_rank]
        if re.findall("[a-zA-Z]+$", selected_residue)[0] != identifier_from_pdb[1]:
            raise IndexError(
            "The {}th residue in the provided pdb file ({}{}) is different from that in the simulations ({})!".format(
                                                                                            entry_id+1,
                                                                                            identifier_from_pdb[0],
                                                                                            identifier_from_pdb[1],
                                                                                            selected_residue)
                                                                                            )
        if identifier_from_pdb[2] != " ":
            cmd.select("BSid{}_{}".format(bs_id, selected_residue), 
            "chain {} and resid {} and (not name C+O+N)".format(identifier_from_pdb[2], identifier_from_pdb[0]))
        else:
            cmd.select("BSid{}_{}".format(bs_id, selected_residue), 
            "resid {} and (not name C+O+N)".format(identifier_from_pdb[0]))
        cmd.show("spheres", "BSid{}_{}".format(bs_id, selected_residue))
        cmd.set("sphere_scale", SCALES[entry_id], selection="BSid{}_{}".format(bs_id, selected_residue))
        cmd.color("tmp_{}".format(bs_id), "BSid{}_{}".format(bs_id, selected_residue))
    cmd.group("BSid{}".format(bs_id), "BSid{}_*".format(bs_id))

            """
    with open(fname, "w") as f:
        f.write(text)

    return