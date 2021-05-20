#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 18:30:37 2021

@author: wlsong
"""

from pymol import cmd, stored


def listselection(selection, HOH="Y"):
	'''
  Function that prints out the number of residues in a selection

  Arguments:
  selection (str): a selection-expression
  HOH (optional, str): 'Y' - including water molecules
                       'N' - excluding water molecules

	Usage: listselection(selection, ['Y'/'N'])
	'''
	printedselection=""
	extra=""
	counter=0
	sel=selection

	if HOH=="N":
		sel=selection+" and not resn HOH"
		extra=", without HOH"
  
	objs=cmd.get_object_list(sel)

	for a in range(len(objs)):
		m1=cmd.get_model(sel+" and "+objs[a])
		for x in range(len(m1.atom)):
			if m1.atom[x-1].resi!=m1.atom[x].resi:
				printedselection+="%s/%s/%s/%s\n" % (objs[a], m1.atom[x].chain, m1.atom[x].resn, m1.atom[x].resi)
				counter+=1
				
	print("Residues in '%s%s': %s" % (selection, extra, counter))
	print(printedselection)
		
cmd.extend('listselection',listselection)

cmd.fetch("4ZSG.pdb", "complex")
cmd.select("lig", "resn 4QX")
cmd.select("pocket_prot", "byres poly within 5 of lig")
cmd.select("pocket_wat", "byres resn HOH within 5 of lig")
cmd.select("pocket", "lig or pocket_prot or pocket_wat")
cmd.extract("pocket", "pocket")
cmd.delete("complex")
cmd.h_add())

cmd.listselection("resn 4qx", HOH="N")
cmd.listselection("resn 4qx", HOH="Y")
