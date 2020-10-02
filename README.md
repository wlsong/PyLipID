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

It can also map in a PyMol session the calculated binding sites to a pdb structure users provide through -pdb. In Such a PyMol session, residues that belong to the same binding site are grouped under the same tab and shown in spheres with sizes corresponding to their residence time of the lipid of study. The -pymol_gui flag allows users to launch such a PyMol session at the end the calculation automatically. 

For definition of residence time, please refer to:
- García, Angel E.Stiller, Lewis. Computation of the mean residence time of water in the hydration shells of biomolecules. 1993. Journal of Computational Chemistry;
- Duncan AL, Corey RA, Sansom MSP. Defining how multiple lipid species interact with inward rectifier potassium (Kir2) channels. 2020. Proc Natl Acad Sci U S A.

To alleviate the 'cage-rattling' phenomenon of the beads dynamics in coarse-grained simulations, pylipid uses a dual-cutoff scheme. This scheme defines the start of a continuous interaction of a lipid molecule with a given object when any atom/bead of the lipid molecule moves within the smaller cutoff; and the end of such a contunuous interaction when all of the atoms/beas of the lipid molecule move out of the larger cutoff. Such a dual-cutoff scheme can also be applied to atomistic simulations. The recommended dual-cutoff for coarse-grained simulations is **0.55 1.0** nm, and that for atomistic simulations is **0.35 0.55** nm. But it's reccommended for users to do some tests on their systems. Users can use the same value for both cutoffs to achieve a single cutoff scheme. 


## Installation:
pylipid.py requires following packages:
- python=3.7
- mdtraj
- numpy
- pandas
- matplotlib
- seaborn
- networkx
- scipy
- pymol (if -pymol_gui True)
- python-louvain
- logomaker
- statsmodels
- scikit-learn

To create a compatible python environment but not to mess up with your global python settings, we recommend building an independent env called PyLipID using [conda](https://www.anaconda.com/distribution/). 
To create this PyLipID environment using the provided env.yml, assuming you have installed [conda](https://www.anaconda.com/distribution/) in your system:
```
conda env create -f env.yml
```
Now your python env PyLipID is all set. Whenever you want to use the script, activate PyLipID first by
```
conda activate PyLipID
```
To get back to your default global python env:
``` 
conda deactivate
```
Remove this env from your system by:
```
conda env remove --name PyLipID
```


## Examples: 
All the **pylipid.py** flags can be checked via 'python pylipid.py -h'.

A standard calculation of lipid interactions using **pylipid.py**:
```
conda activate PyLipID
python pylipid.py -f ./run_1/md.xtc ./run_2/md.xtc -c ./run_1/protein_lipids.gro ./run_2/protein_lipids.gro 
-cutoffs 0.55 1.0 -lipids POPC CHOL POP2 -nprot 1 -save_dataset 
```

**pylipid.py** uses graph theory and community analysis to calcualte lipid binding sites. The script can also view the calcualted binding sites in PyMol, provided that users provide a pdb file of the protein structure in the atomistic model (because PyMol only recognises atomistic coordinate models) via the flag -pdb. For the coarse-grained simulations, either provide the atomistic protein structure before coarse-graining or use an atomistic structure that is converted back from coarse-grained models. Users need to make sure that the provided protein coordinates are consistent with the configuration in the simulations in terms of the residue indices and ordering of the protein. An example of using the flag -pdb: 
```
python pylipid.py -f ./run_1/md.xtc ./run_2/md.xtc -c ./run_1/protein_lipids.gro ./run_2/protein_lipids.gro 
-cutoffs 0.55 1.0 -lipids POPC CHOL POP2 -nprot 1 -save_dataset -pdb XXXX.pdb
```
Replace 'XXXX.pdb' with the pdb file of your chose. By default, a PyMol session with the calculated binding site information will show up at the end of the calculation. To switch off this PyMol GUI, use -pymol_gui False. This binding site information with PyMol display is stored in a python script 'show_binding_site_info.py' which allows users to re-open this PyMol session by the command 'python show_binding_site_info.py'.

Due to the smoothened energy potentials, coarse-grained force fields often render the tails of phosphalipids too flexible, which could lead to poor characterisation of binding sites. When the behaviours of the tails are not the main focus, users can just check the binding of headgroups for a better characterisation of binding dynamics. Users can use the flag -lipid_atoms to specify lipid atoms/beads for calculation. An example of calculating the binding of PIP2 in MARTINI 2 (POP2) using only the headgroup beads: 
```
python pylipid.py -f ./run_1/md.xtc ./run_2/md.xtc -c ./run_1/protein_lipids.gro ./run_2/protein_lipids.gro 
-cutoffs 0.55 1.0 -lipids POP2 -lipid_atoms C1 C2 C3 C4 PO4 P1 P2 -nprot 1 -save_dataset -pdb XXXX.pdb 
-pymol_gui False
```

**pylipid.py** allows user to specify a couple of regions for the calculation via the flag -resi_list. Supported syntax includes: 1/ use "-" to indicate a range of the protein residue index (both ends included); or 2/ specify individual residue index seperated by space: 
```
python pylipid.py f ./run_1/md.xtc ./run_2/md.xtc -c ./run_1/protein_lipids.gro ./run_2/protein_lipids.gro 
-cutoffs 0.55 1.0 -lipids POPC CHOL POP2 -nprot 1 -resi_list 5 7 8 10-30 50-70 100-130 -save_dataset -pdb XXXX.pdb -pymol_gui False
```

**pylipid.py** calculates the surface area of each binding site. By default, the script uses atom radii defined by mdtraj (https://github.com/mdtraj/mdtraj/blob/master/mdtraj/geometry/sasa.py#L56) for calculation. The script also defines the radii of MARTINI coarse-grained beads BB as 0.26 nm and SC1/2/3 as 0.23 nm. To change or define radii of atoms/beads, use -radii and specify radius in unit of nm. For example, to change the radius of MARINI coarse-grained beads BB to 0.28 nm and SC1 to 0.22 nm: 
```
python pylipid.py f ./run_1/md.xtc ./run_2/md.xtc -c ./run_1/protein_lipids.gro ./run_2/protein_lipids.gro 
-cutoffs 0.55 1.0 -lipids POPC CHOL POP2 -nprot 1 -radii BB:0.28 SC1:0.22
```

**pylipid.py** calculates the probability density functions of bound lipids and uses such functions to score all the bound lipid poses in the trajectories. Then the script ranks the binding poses based on the calculated scores and writes out the top ranking lipid binding poses along with the receptor coordinates that the pose binds to. By default, pylipid.py writes 5 top ranking lipid poses for each binding site, and users can use -gen_binding_poses to change how many to be generated. The script writes the binding pose coordinates in pdb format by default, and users can change the saved coordinate formate to those that are suported by mdtraj via -save_pose_format. For phospholipids, it's recommended to give higher weights to lipid headgroups in the scoring functions. Use -score_weights to change the weights. The following example shows how to generate 10 top ranking poses for each binding site, to save the binding poses in gro format and to give higher weight to the headgroup beads of PIP2 in the MARTINI force field:
```
python pylipid.py f ./run_1/md.xtc ./run_2/md.xtc -c ./run_1/protein_lipids.gro ./run_2/protein_lipids.gro 
-cutoffs 0.55 1.0 -lipids POP2 -nprot 1 -gen_binding_poses 10 -save_pose_format gro -score_weights PO4:10 P1:10:P2:10 C1:10 C2:10 C3:10
```
The calculation of probability density uses the function of [KDEMultivariate](https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.kernel_density.KDEMultivariate.html) from statsmodels. This calculation can take some time (a couple of hours) for atomistic simulations or long coarse-grained simulations where the collected binding data is large (either due to higher granularity or a larger number of binding events). To speed up the calculation, users can decrease the volume of data by using the flag -stride to stride throught trajectories, i.e. analyse only every X-th of the trajectory frame. In addition, if getting the bound lipid coordinates is not the focus, users can use -gen_binding_poses 0 to switch off the binding pose generation process. 


## Developers:
- Wanling Song
- Anna Duncan
- Robin Corey
- Bertie Ansell


## Thanks for reading to the end, much respect!
Writing scripts is about fixing one bug after another.

Be brave, be real and keep going, homie!
![Github Logo](https://github.com/wlsong/wlsong/blob/master/resources/dino.gif)

