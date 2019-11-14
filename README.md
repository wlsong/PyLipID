# PyLipID

## Introduction 
**pylipid.py**: is a toolkit to calculate lipid interactions with membrane proteins. 
It calculates: 
- lipid interactions with the proteins in terms of their duration, residence time, occupancy, num. of lipids surrounding given residues and koff;
- lipid binding sites via interaction networks. 

It plots:
- lipid interaction with the protein as a function of protein residues. 
- the calculated lipid koff to each protein residue. 
- interaction network of lipid binding sites. 

It can also map the calculated binding sites to a pdb structure you provide through -pdb. Residues belonging to the same binding site are grouped under the same tab and shown in spheres with sizes corresponding to their residence time with the lipid of study.  

For definition of residence time, please refer to:
- García, Angel E.Stiller, Lewis. Computation of the mean residence time of water in the hydration shells of biomolecules. 1993. Journal of Computational Chemistry;
- Arnarez, C., et al. Evidence for cardiolipin binding sites on the membrane-exposed surface of the cytochrome bc1. 2013. J Am Chem Soc

## Installation:
pylipid.py is tested against 
- python = 3.6
- mdtraj = 1.9
- numpy = 1.14
- pandas = 0.23
- matplotlib = 3.1
- seaborn = 0.8
- networkx = 2.1
- scipy = 1.1
- pymol = 1.9
- networkx = 2.1

To create a compatible python environment and to not mess up with your global python env, we recommend creating an independent environment called PyLipID using [conda](https://www.anaconda.com/distribution/). 
To create this PyLipID environment, assuming you have installed conda in your system:
```
conda create -n PyLipID python=2.6 mdtraj=1.9 numpy=1.14 pandas=0.23 matplotlib=3.1 seaborn=0.8 scipy=1.1 network=2.1
conda init bash # your shell name. Supported shells include bash, fish,tcsh, zsh etc. See conda init --help for information
source ~/.bashrc
```
To install pymol and community, we need to:
```
conda activate PyLipID 
# now we are in PyLipID env and what we do in the following is only effective to this env
conda install -c samoturk pymol
pip install python-louvain
```
Now your python env PyLipID is all set. Whenever you want to use the script, activate PyLipID first by
```
conda activate PyLipID
```
When you want to get back to your default global python env:
``` 
conda deactivate PyLipID
```
Remove this env from your system by:
```
conda env remove --name PyLipID
```

## Usage:

**-f**: Trajectories to check. Can be a list of trajectories with similar system settings. Read in by mdtraj.load().

**-c**: structural information of the trajectories given to -f. Read in by mdtraj.load(). Supported format include gro, pdb xyz, etc. 

**-stride**: stride through trajectories. Only every stride-th frame will be analyzed.

**-tu**: time unit of all the calculations. Available options include ns and us. 

**-save_dir**: directory where all the results will be located. Will use current working directory if not specified. 

**-cutoffs**: the double cutoffs used to define lipid interactions. A continuous lipid contact with a given residue starts when the lipid
gets closer to the given residue than the smaller cutoff and ends when the lipid gets farther than the larger cutoff. 

**-lipids**: specify the lipid residue name 

**-lipid_atoms**: specify the atoms to check

**-nprot**: num. of proteins in the system

**-resi_offset**: Shift the residue index of the protein. Can be useful when a protein with missing residues at its N-terminus was martinized 
to Martini force field, as martinize.py shift the residue index of the first residue to 1 regardless of its original index. 

**-save_dataset**: save dataset in pickle. 

**-pdb**: Provide a PDB structure onto which the binding site information will be mapped. Using this flag will open a pymol session at the end of calculation and also save a python file "show_binding_site_info.py" in the -save_dir directory. No pymol session will be opened nor python file written out if not specified.

**-chain**: The chain in the -pdb structure onto which binding site infomation should be mapped. This is useful when the pdb structure you provide by -pdb has multiple chains. 

Usage example: 
```
conda activate PyLipID
python pylipid.py -f ./run_1/md.xtc ./run_2/md.xtc -c ./run_1/protein_lipids.gro ./run_2/protein_lipids.gro 
-cutoffs 0.55 1.4 -lipids POPC CHOL POP2 -nprot 1 -resi_offset 5 -save_dataset -pdb XXXX.pdb -chain A
```
For phospholipids, it's recommended to use only the headgroup atoms to detect lipid binding sites:
```
python pylipid.py -f ./run_1/md.xtc ./run_2/md.xtc -c ./run_1/protein_lipids.gro ./run_2/protein_lipids.gro 
-cutoffs 0.55 1.4 -lipids POP2 -lipid_atoms C1 C2 C3 C4 PO4 P1 P2 -nprot 1 -resi_offset 5 -save_dataset -pdb XXXX.pdb -chain A
```

