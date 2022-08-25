# vemethod-simulation-frictions
This repository contains files required to replicate the study "How Important Are Study Designs? A Simulation-Based Assessment of VE Estimation Bias with Time-Varying Vaccine Coverage, and Heterogeneous Testing and Baseline Attack Rates" by Jing Lian Suah, Naor Bar-Zeev and Maria Deloria Knoll.

Link to the preprint can be found [here](test).

This repository contains the following.
1. Python scripts required to replicate the study. These are in the main directory.
2. Output files from scripts. These are in the ```Output``` directory.
3. Latest draft article of the study, which is in the ```Article``` directory.
4. Latest slides summarising the study, which is in the ```Presentation``` directory.

# Ethical consideration
As this research uses only simulated data, and does not involve any human participants, no institutional review board (IRB) approval is required.

# Declaration of interests
JLS received support for attending academic meetings from AstraZeneca for work outside this paper. NB-Z received research grants from Merck, personal fees from Merck, and a research grant from Johnson & Johnson, all for unrelated work outside the scope of this paper. MDK received reports grants from Merck, personal fees from Merck, and grants from Pfizer, outside the submitted work.

# Computation
1. Simulation scripts (```VEMethod_Sim1b_Parallel_NoCI_CloudVersion.py``` and ```VEMethod_Sim1b_Parallel_NoCI_CloudVersion_ReSeed.py```) were run on a AWS EC2 c6i.32xLarge Amazon Linux instance (128 vCPUs); entire process took 415516 seconds (115.42 hours).
2. Post-simulation analyses (the remaining ```py``` files with ```VEMethod_``` suffixes) were run on Windows 10, 4-core (8 logical processors) 10th gen i7 and 11th gen i5 local machines, as no parallel processing is required.

# Replication
0. ```git clone suahjl/vemethod-simulation-frictions```
1. ```pip install -r requirements.txt```
2. ```VEMethod_Sim1b_Parallel_NoCI_CloudVersion```
3. ```VEMethod_Sim1b_Parallel_NoCI_CloudVersion_ReSeed```
4. Execute the following in any order.
	- ```VEMethod_Sim1b_Heatmap_GreekLetters.py```
	- ```VEMethod_Sim1b_PureDesignBias_Heatmap.py```
	- ```VEMethod_Sim1b_WaveSpecific_Heatmap.py```
	- ```VEMethod_Drivers1b.py```
	- ```VEMethod_RelDirection1b.py```

<br/> \*Note 1: You may consider running the simulation using different seeds.
<br/> \*Note 2: All output files are placed in the 'Output' folder for ease of navigation.

# Large (>100MB) output files
Due to file size limits on GitHub, 3 large output files can be accessed via dropbox [here](https://www.dropbox.com/sh/7sxgwfymrbkexb9/AADc4E3wb-FEsMr7SMIRqH4Ba?dl=0). These files are:
1. The full VE estimation bias heatmap of all study designs and all parameter sets from the simulation: *VEMethod_Sim1b_Parallel_NoCI_Wide_Gradient*
2. The above but with VE estimation biases rescaled to that the survival analysis cohort with true outcomes being observable: *VEMethod_Sim1b_Parallel_NoCI_Wide_Relative_Gradient*
3. The above but using absolute VE estimation biases: *VEMethod_Sim1b_Parallel_NoCI_Wide_Relative_Absolute Gradient*