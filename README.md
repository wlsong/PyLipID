![Github Logo](https://github.com/wlsong/PyLipID/blob/master/header.tiff)

## Introduction 
**pylipid.py**: is a toolkit to calculate lipid interactions with membrane proteins. 
It calculates: 
- lipid interactions with the proteins in terms of their duration, residence time, occupancy, num. of lipids surrounding given residues and koff;
- lipid binding sites via interaction networks. 

It plots:
- lipid interactions (in terms of duration, residence time, occupancy, and num. of surroudning lipids) with the protein as a function of protein residues. 
- the calculated lipid koff to each protein residue. 
- interaction network of lipid binding sites. 

It can also map the calculated binding sites to a pdb structure you provide through -pdb. Using the -pdb flag will generate a python script that allows users to view the binding site information mapped to the provided structures in PyMol. In Such a PyMol session, residues that belong to the same binding site are grouped under the same tab and shown in spheres with sizes corresponding to their residence time of the lipid of study. The -pymol_gui flag allows users to launch such a PyMol session at the end the calculation automatically. 

For definition of residence time, please refer to:
- García, Angel E.Stiller, Lewis. Computation of the mean residence time of water in the hydration shells of biomolecules. 1993. Journal of Computational Chemistry;
- Duncan AL, Corey RA, Sansom MSP. Defining how multiple lipid species interact with inward rectifier potassium (Kir2) channels. 2020. Proc Natl Acad Sci U S A.

## Installation:
pylipid.py requires following packages:
- python
- mdtraj
- numpy
- pandas
- matplotlib
- seaborn
- networkx
- scipy
- pymol
- python-louvain

To create a compatible python environment but not to mess up with your global python settings, we recommend building an independent env called PyLipID using [conda](https://www.anaconda.com/distribution/). 
To create this PyLipID environment using the provided env.yml, assuming you have installed conda in your system:
```
conda env create -f env.yml
```
Now your python env PyLipID is all set. Whenever you want to use the script, activate PyLipID first by
```
conda activate PyLipID
```
When you want to get back to your default global python env:
``` 
conda deactivate
```
Remove this env from your system by:
```
conda env remove --name PyLipID
```

## Usage:

**-f**: Trajectories to check. Can be a list of trajectories with similar system settings. Read in by mdtraj.load().

**-c**: Structural information of the trajectories given to -f. Read in by mdtraj.load(). Supported format include gro, pdb xyz, etc. 

**-stride**: Stride through trajectories. Only every stride-th frame will be analyzed.

**-dt**: Define the time interval between two adjacent frames in the trajectories. If not specified, the mdtraj will deduce from the trajectories. This works for trajectories in format of e.g. xtc which include timestep information. For trajectories in dcd format, users have to provide the time interval manually, in a time unite consistent with -tu

**-tu**: Time unit of all the calculations. Available options include ns and us. 

**-save_dir**: Directory where all the results will be located. Will use current working directory if not specified. 

**-cutoffs**: The double cutoffs used to define lipid interactions. A continuous lipid contact with a given residue starts when the lipid moves to the given residue closer than the smaller cutoff; and ends when the lipid moves farther than the larger cutoff. The standard single cutoff can be acheived by setting the same value for both cutoffs. 

**-lipids**:  Lipid species to check, seperated by space. Should keep consistent with residue name in your trajectories.

**-lipid_atoms**: Lipid atoms to check, seperated by space. Should be consistent with the atom names in your trajectories.

**-radii**: Change/Define the radii of atoms/beads that are used for the calculation of binding site surface area. Supported syntax is BB:0.26, which defines the radius of bead BB as 0.26 nm, or CA:0.12 which defines the radius of atom CA as 0.12 nm. For atomistic simulations, the default radii are taken from mdtraj https://github.com/mdtraj/mdtraj/blob/master/mdtraj/geometry/sasa.py#L56. For coarse-grained simulations, this script defines the radius of the MARTINI 2 beads of BB as 0.26 nm and SC1/2/3 as 0.23 nm.  

**-nprot**: num. of proteins (or chains) in the simulation system. The calculated results will be averaged among these proteins (or chains). The proteins (or chains) need to be identical, otherwise the averaging will fail.

**-resi_offset**: Shifting the residue index. It is useful if you need to change the residue index in your trajectories. For example, to change the residue indeces from 5,6,7,..., to 10,11,12,..., use -resi_offset 4. All the outputs, including plotted figures and saved coordinates, will be changed by this.

**-resi_list**: The indices of residues selected for calculation. The calculation will be done only on these selected residues. This option is useful for those proteins with large regions/domains that don't require calculation, and skipping those regions/domains could save time and memory. Accepted syntax include 1/ defining a range of residue indeces, like 1-10 (both ends included); 2/ single residue index, like 25 26 17. The selections are seperated by space. For example, -resi_list 1-10 20-30 40 45 46 means selecting residues 1-10, 20-30, 40, 45 and 46 for calculation. The residue indices are not affected by -resi_offset, i.e. they should be consistent with the indices in your trajectories.

**-nbootstrap**: The number of samples for bootstrapping the calcultion of koff. The default is 10. The larger the number, the more time-consuming the calculation will be. The closer the bootstrapped residence time/koffs are to the original values, the more reliable those original values are. The bootstrapped results are plotted in each of the koff plots and plotted apposed to the original values in the figure showing residence time. 

**-save_dataset**: Save dataset in pickle. Default is True. 

**-pdb**: Provide a PDB structure onto which the binding site information will be mapped. Using this flag will generate a 'show_binding_site_info.py' file in the -save_dir directory, which allows users to check the mapped binding site information in PyMol. Users can run the generated script by 'python show_binding_site_info.py' to open such a PyMol session.

**-pymol_gui**: Show the PyMol session of binding site information at the end of calculation. Need to be used in conjuction with -pdb.

**-chain**: Select the chain of the structure provided by -pdb to which the binding site information mapped. This option is useful when the pdb structure has multiple chains. 



## Examples: 
A standard check on lipid interactions:
```
conda activate PyLipID
python pylipid.py -f ./run_1/md.xtc ./run_2/md.xtc -c ./run_1/protein_lipids.gro ./run_2/protein_lipids.gro 
-cutoffs 0.55 1.0 -lipids POPC CHOL POP2 -nprot 1 -save_dataset 
```
To map the calculated binding site information onto a structure (which should be consistent with your trajectory configuration), use the flag -pdb:
```
python pylipid.py -f ./run_1/md.xtc ./run_2/md.xtc -c ./run_1/protein_lipids.gro ./run_2/protein_lipids.gro 
-cutoffs 0.55 1.0 -lipids POPC CHOL POP2 -nprot 1 -save_dataset -pdb XXXX.pdb
```
Replace 'XXXX.pdb' with the pdb file you chose. By default, a PyMol session with the calculated binding site information will show up at the end of the calculation. To switch off this PyMol GUI, use -pymol_gui False. This binding site information with PyMol display is stored in a python script 'show_binding_site_info.py' which allows users to re-open this PyMol session by the command 'python show_binding_site_info.py'.
For phospholipids, it's recommended to use only the headgroup atoms for a better definition of binding events:
```
python pylipid.py -f ./run_1/md.xtc ./run_2/md.xtc -c ./run_1/protein_lipids.gro ./run_2/protein_lipids.gro 
-cutoffs 0.55 1.0 -lipids POP2 -lipid_atoms C1 C2 C3 C4 PO4 P1 P2 -nprot 1 -save_dataset -pdb XXXX.pdb 
-pymol_gui False
```
To specify a couple of regions for the calculation, use -resi_list:
```
python pylipid.py f ./run_1/md.xtc ./run_2/md.xtc -c ./run_1/protein_lipids.gro ./run_2/protein_lipids.gro 
-cutoffs 0.55 1.0 -lipids POPC CHOL POP2 -nprot 1 -resi_list 10-30 50-70 100-130 -save_dataset -pdb XXXX.pdb -pymol_gui False
```
The recommended dual-cutoff for coarse-grained simulations is **0.55 1.0**, and that for atomistic simulations is **0.35 0.55** . But it's always reccommended for users to do some tests on their systems. 


## Developers:
- Wanling Song
- Anna Duncan
- Robin Corey
- Bertie Ansell
