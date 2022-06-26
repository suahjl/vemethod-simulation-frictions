#### Ad Hoc: Replace all parameters with unicode greek letters
#### Regenerate heatmaps
#### Easier as the original version already made the LaTeX-style columns indices
#### Post-26Jun2022: No more telegram_send (too mega-sized)

import gc
import pandas as pd
import numpy as np
from datetime import date, timedelta
import itertools
import time
import telegram_send
import dataframe_image as dfi
import shutil
from tqdm import tqdm

time_start = time.time()

### 0 --- Preliminaries
path_method = 'D:/Users/ECSUAH/Desktop/Quant/HealthEconomy/VaccinesAssessment/VEMethod/' # DOUBLE CHECK THIS
VE_consol_all  = pd.read_parquet(path_method + 'VEMethod_Sim1b_Parallel_CloudVersion_NoCI_FIN.parquet') # take in raw output

### I --- Cleaning
print('I --- Cleaning')
## Replace LaTex-style columns with unicode Greek characters
print('Replace LaTex-style columns with unicode Greek characters')
param_latex = ['N', 'T', 'Theta_v', 'p_v', 'Theta_tau', 'p_tau', 'ktau', 'alpha_b', 'kalpha', 'mew_ve']
param_greek = ['N', 'T', '\u0398(v)', 'p(v)', '\u0398(\u03C4)', 'p(\u03C4)', 'k(\u03C4)', '\u03B1(b)', 'k(\u03B1)', '\u03BC(VE)']
dict_param = dict(zip(param_latex, param_greek))
dict_param_reverse = dict(zip(param_greek, param_latex))
VE_consol_all = VE_consol_all.rename(columns=dict_param)
## Extreme values
print('Extreme values')
ubound = 20
lbound = -20
# VE_consol_all.loc[VE_consol_all['Bias'] >= ubound, 'Bias'] = np.nan # indicative of inability to estimate
# VE_consol_all.loc[VE_consol_all['Bias'] <= lbound, 'Bias'] = np.nan # indicative of inability to estimate
## LaTeX version
print('LaTeX version')
VE_consol_all_reverse = VE_consol_all.rename(columns=dict_param_reverse) # for rendering html

### II --- Replicate original script
print('II --- Replicate original script')
VE_consol_all_wide = VE_consol_all.pivot(index=param_greek, columns='Design', values='Bias') # only central measure, no CIs
VE_consol_all_wide_reverse = VE_consol_all_reverse.pivot(index=param_latex, columns='Design', values='Bias') # only central measure, no CIs (reverse version)
VE_consol_all_wide.to_csv(path_method + 'VEMethod_Sim1b_Parallel_NoCI_Wide.csv', index=True)
n_sim = len(VE_consol_all_wide.index) # number of simulations done (length of parameter set)
## Export raw table
print('Export raw table')
# with open(path_method + 'VEMethod_Sim1b_Parallel_NoCI_Wide.csv', 'rb') as f:
#     telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
#                        files=[f],
#                        captions=['VEMethod_Sim1b_Parallel_NoCI_Wide\n\n' +
#                                  '*Extreme values have been blanked out'])
## Generate heat map
print('Generate heat map')
VE_consol_all_wide_grad = VE_consol_all_wide_reverse.style.background_gradient(cmap='coolwarm', vmin=lbound, vmax=ubound)
with open(path_method + 'VEMethod_Sim1b_Parallel_NoCI_Wide_Gradient.html', 'w') as f:
    f.write(VE_consol_all_wide_grad.render())
telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
                   captions=['Number of simulations completed: ' + str(n_sim) + '\n\n' +
                             'Simulated bias in VE estimates across study designs for multiple parameter sets under simultaneous epidemic and vaccination rollout' + '\n\n' +
                             'File Name: VEMethod_Sim1b_Parallel_NoCI_Wide_Gradient.html'])
VE_consol_all_wide_grad.to_excel(path_method + 'VEMethod_Sim1b_Parallel_NoCI_Wide_Gradient.xlsx', sheet_name='Gradient', float_format='%.2f', na_rep='', index=True)
telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
                   captions=['Number of simulations completed: ' + str(n_sim) + '\n\n' +
                             'Simulated bias in VE estimates across study designs for multiple parameter sets under simultaneous epidemic and vaccination rollout' + '\n\n' +
                             'File Name: VEMethod_Sim1b_Parallel_NoCI_Wide_Gradient.xlsx'])

### III --- Generate relative bias (benchmark against survival true y)
print('III --- Generate relative bias (benchmark against survival true y)')
ref_design='Cohort (immortal time correction): True y'
list_design = list(VE_consol_all['Design'].unique())
VE_consol_all_wide_reverse_rel = VE_consol_all_wide_reverse.copy()
for i in tqdm(list_design):
    VE_consol_all_wide_reverse_rel.loc[VE_consol_all_wide_reverse_rel[i] == np.inf, i] = np.nan
for i in tqdm(list_design):
    VE_consol_all_wide_reverse_rel.loc[VE_consol_all_wide_reverse_rel[i].isna(), i] = 0
list_design.remove(ref_design)
for i in tqdm(list_design):
    VE_consol_all_wide_reverse_rel[i] = VE_consol_all_wide_reverse_rel[i] / VE_consol_all_wide_reverse_rel[ref_design]
VE_consol_all_wide_reverse_rel[ref_design] = VE_consol_all_wide_reverse_rel[ref_design] / VE_consol_all_wide_reverse_rel[ref_design]
VE_consol_all_wide_rel_grad = VE_consol_all_wide_reverse_rel.style.background_gradient(cmap='coolwarm', vmin=lbound, vmax=ubound)
with open(path_method + 'VEMethod_Sim1b_Parallel_NoCI_Wide_Relative_Gradient.html', 'w') as f:
    f.write(VE_consol_all_wide_rel_grad.render())
telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
                   captions=['Number of simulations completed: ' + str(n_sim) + '\n\n' +
                             'Simulated bias in VE estimates relative to ' + ref_design + '\n\n' +
                             'File Name: VEMethod_Sim1b_Parallel_NoCI_Wide_Relative_Gradient.html'])
VE_consol_all_wide_rel_grad.to_excel(path_method + 'VEMethod_Sim1b_Parallel_NoCI_Wide_Relative_Gradient.xlsx',
                                     sheet_name='Gradient',
                                     float_format='%.2f',
                                     na_rep='',
                                     index=True)
telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
                   captions=['Number of simulations completed: ' + str(n_sim) + '\n\n' +
                             'Simulated bias in VE estimates relative to ' + ref_design + '\n\n' +
                             'File Name: VEMethod_Sim1b_Parallel_NoCI_Wide_Relative_Gradient.xlsx'])

### IV --- Generate relative ABSOLUTE bias (benchmark against survival true y)
print('IV --- Generate relative ABSOLUTE bias (benchmark against survival true y)')
ref_design='Cohort (immortal time correction): True y'
list_design = list(VE_consol_all['Design'].unique())
VE_consol_all_wide_reverse_rel_abs = VE_consol_all_wide_reverse.copy()
for i in tqdm(list_design):
    VE_consol_all_wide_reverse_rel_abs[i] = np.abs(VE_consol_all_wide_reverse_rel_abs[i]) # express as absolutes, then same as prev block
for i in tqdm(list_design):
    VE_consol_all_wide_reverse_rel_abs.loc[VE_consol_all_wide_reverse_rel_abs[i] == np.inf, i] = np.nan
for i in tqdm(list_design):
    VE_consol_all_wide_reverse_rel_abs.loc[VE_consol_all_wide_reverse_rel_abs[i].isna(), i] = 0
list_design.remove(ref_design)
for i in tqdm(list_design):
    VE_consol_all_wide_reverse_rel_abs[i] = VE_consol_all_wide_reverse_rel_abs[i] / VE_consol_all_wide_reverse_rel_abs[ref_design]
VE_consol_all_wide_reverse_rel_abs[ref_design] = VE_consol_all_wide_reverse_rel_abs[ref_design] / VE_consol_all_wide_reverse_rel_abs[ref_design]
VE_consol_all_wide_rel_abs_grad = VE_consol_all_wide_reverse_rel_abs.style.background_gradient(cmap='Blues', vmin=0, vmax=10) # diff scheme and range since >=0
with open(path_method + 'VEMethod_Sim1b_Parallel_NoCI_Wide_Relative_Absolute_Gradient.html', 'w') as f:
    f.write(VE_consol_all_wide_rel_abs_grad.render())
telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
                   captions=['Number of simulations completed: ' + str(n_sim) + '\n\n' +
                             'ABSOLUTE simulated bias in VE estimates relative to ' + ref_design + '\n\n' +
                             'File Name: VEMethod_Sim1b_Parallel_NoCI_Wide_Relative_Absolute_Gradient.html'])
VE_consol_all_wide_rel_abs_grad.to_excel(path_method + 'VEMethod_Sim1b_Parallel_NoCI_Wide_Relative_Absolute_Gradient.xlsx',
                                         sheet_name='Gradient',
                                         float_format='%.2f',
                                         na_rep='',
                                         index=True)
with open(path_method + 'VEMethod_Sim1b_Parallel_NoCI_Wide_Relative_Absolute_Gradient.xlsx', 'rb') as f:
    telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
                       captions=['Number of simulations completed: ' + str(n_sim) + '\n\n' +
                                 'ABSOLUTE simulated bias in VE estimates relative to ' + ref_design + '\n\n' +
                                 'File Name: VEMethod_Sim1b_Parallel_NoCI_Wide_Relative_Absolute_Gradient.xlsx'])

### End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
