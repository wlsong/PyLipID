![Github Logo](https://github.com/wlsong/PyLipID/blob/master/header.png)

## Introduction 
**pylipid.py**: is a toolkit to calculate lipid interactions with membrane proteins. 

It calculates: 
- lipid interactions with the proteins in terms of their duration, residence time, occupancy, num. of lipids surrounding given residues and koff;
- lipid binding sites via interaction networks. 
- various binding kinetics, e.g. lipid residence time, koff, etc, for each binding site. 
- lipid binding site surface area via Shrake-Rupley algorithm (Shrake, A; Rupley, JA. (1973) J Mol Biol 79 (2): 351–71)
- probablity density functions of bound lipid and generates representative binding poses for each binding site based on the calcuated PDF. 

It plots:
- lipid interactions (in terms of duration, residence time, occupancy, and num. of surroudning lipids) with the protein as a function of protein residue indeces. 
- the calculated lipid koff for each protein residue. 
- the calculated lipid koff for each binding site.
- surface area for each binding site. 

It generates:
- protein coordinates in pdb formate in which such data as residuence time, koff, duration and occupancy are recorded in the b factor column. 
- representative binding poses for each binding site based on scoring functions that use probability density functions of the bound lipids. 

It can also map in a PyMol session the calculated binding sites to a pdb structure users provide through -pdb. When the flag -pdb is provided, pylipid.py will write out a python script 'show_binding_site_info.py' that allows users to open up a PyMol session, in which residues that belong to the same binding site are shown in spheres with sizes corresponding to their calculated residence time. 

For definition of residence time, please refer to:
- García, Angel E.Stiller, Lewis. Computation of the mean residence time of water in the hydration shells of biomolecules. 1993. Journal of Computational Chemistry;
- Duncan AL, Corey RA, Sansom MSP. Defining how multiple lipid species interact with inward rectifier potassium (Kir2) channels. 2020. Proc Natl Acad Sci U S A.

To alleviate the 'cage-rattling' phenomenon of the beads dynamics in coarse-grained simulations, pylipid uses a dual-cutoff scheme. This scheme defines the start of a continuous interaction of a lipid molecule with a given object when any atom/bead of the lipid molecule moves within the smaller cutoff; and the end of such a contunuous interaction when all of the atoms/beas of the lipid molecule move out of the larger cutoff. Such a dual-cutoff scheme can also be applied to atomistic simulations. The recommended dual-cutoff for coarse-grained simulations is **0.55 1.0** nm, and that for atomistic simulations is **0.35 0.55** nm. But it's reccommended for users to do some tests on their systems. Users can use the same value for both cutoffs to achieve a single cutoff scheme. 
