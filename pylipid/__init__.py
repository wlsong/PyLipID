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

"""PyLipID is a python library for anaylysing protein-lipid interactions.
It calculates:
- lipid interactions with the proteins in terms of their duration, residence time, occupancy, num. of lipids surrounding
given residues and koff;
- lipid binding sites via interaction networks.
- various binding kinetics, e.g. lipid residence time, koff, etc, for each binding site.
- lipid binding site surface area via Shrake-Rupley algorithm (Shrake, A; Rupley, JA. (1973) J Mol Biol 79 (2): 351–71)
- probablity density functions of bound lipid and generates representative binding poses for each binding site based on
the calculated PDF.

It plots:
- lipid interactions (in terms of duration, residence time, occupancy, and num. of surroudning lipids) with the protein
as a function of protein residue indeces.
- the calculated lipid koff for each protein residue.
- the calculated lipid koff for each binding site.
- surface area for each binding site.

It generates:
- protein coordinates in pdb formate in which such data as residuence time, koff, duration and occupancy are recorded in
the b factor column.
- representative binding poses for each binding site based on scoring functions that use probability density functions of
the bound lipids.

"""



