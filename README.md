# vemethod-simulation-frictions
Research project on VE methodology using simulation-based technique and theoretical model with behavioural frictions

# Ethical consideration:
This research uses only simulated data, hence does not require IRB review.

# Computational notes:
1. Simulation files (steps 0 and 1) were run on AWS EC2 c6i.32xLarge Amazon Linux instance (128 vCPUs); entire process took 415516 seconds (115.42 hours).
2. Post-simulation analyses (steps 2 onwards) were run on Windows 10, 4-core (8 logical processors) 10th gen i7 and 11th gen i5 local machines, as no parallel processing is required.

# Replication notes:
1. Steps 0 and 1 can be run as-is, provided all dependencies (modules in the script itself) have been installed.
2. Run step 0 as-is. This script simulates data using the theoretical model in the paper using all parameter combinations detailed.
3. Run step 1 as-is. This script re-runs select simulations whose legal parameter combinations returned singular matrix errors using a different seed, and merges both data frames.
4. Run step 2 after changing the directory line (path_method = 'xx') to your own desired output directory, the output from step 1 should be in the same directory.
<br/> \*Note: You may consider running the simulation using different seeds.

# Large (>100MB) output files
Due to file size limits on GitHub, 3 large output files can be accessed via dropbox [here](https://www.dropbox.com/sh/7sxgwfymrbkexb9/AADc4E3wb-FEsMr7SMIRqH4Ba?dl=0). These files are:
1. The full VE estimation bias heatmap of all study designs and all parameter sets from the simulation: *VEMethod_Sim1b_Parallel_NoCI_Wide_Gradient*
2. The above but with VE estimation biases rescaled to that the survival analysis cohort with true outcomes being observable: *VEMethod_Sim1b_Parallel_NoCI_Wide_Relative_Gradient*
3. The above but using absolute VE estimation biases: *VEMethod_Sim1b_Parallel_NoCI_Wide_Relative_Absolute Gradient*