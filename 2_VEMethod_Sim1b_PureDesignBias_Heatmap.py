#### Generate heatmap of design biases with only 'perfect parameter assumptions'

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

import statsmodels.formula.api as smf
from linearmodels import PanelOLS

import matplotlib.pyplot as plt
import seaborn as sns

time_start = time.time()

### 0 --- Preliminaries
## Path
path_method = 'D:/Users/ECSUAH/Desktop/Quant/HealthEconomy/VaccinesAssessment/VEMethod/' # check this
## File name
file_sim = 'VEMethod_Sim1b_Parallel_CloudVersion_NoCI_FIN.parquet' # take the raw version (all latex)
## Data frame
df = pd.read_parquet(path_method + file_sim)
n_row = len(df.index) # number of simulations done
## Pandas global settings
pd.set_option('precision', 0)

### 0 --- Setup
## Parameters
param_latex = ['N', 'T', 'Theta_v', 'p_v', 'Theta_tau', 'p_tau', 'ktau', 'alpha_b', 'kalpha', 'mew_ve']
param_greek = ['N', 'T', '\u0398(v)', 'p(v)', '\u0398(\u03C4)', 'p(\u03C4)', 'k(\u03C4)', '\u03B1(b)', 'k(\u03B1)', '\u03BC(VE)']
dict_param = dict(zip(param_latex, param_greek))
param_latex_ideal = ['p_v', 'Theta_tau', 'p_tau', 'ktau', 'kalpha']
param_latex_variable = ['T', 'mew_ve', 'Theta_v']
param_latex_fixed = ['N', 'alpha_b']
## Reference
ref_design='Cohort: True y'
list_design = list(df['Design'].unique())
list_design.remove(ref_design) # move ref to number 0
list_design.insert(0, ref_design) # move ref to number 0
n_list_design = list(range(len(list_design)))
dict_design_n = dict(zip(list_design, n_list_design))
## Rename columns (designs)
df['Design'] = df['Design'].replace(dict_design_n)

### I --- Cleaning
## Extreme values
ubound = 20
lbound = -20
# df.loc[df['Bias'] >= ubound, 'Bias'] = np.nan # indicative of inability to estimate
# df.loc[df['Bias'] <= lbound, 'Bias'] = np.nan # indicative of inability to estimate
## Choose parameter values
for i in param_latex_ideal: df = df[df[i] == 1]
for i in param_latex_variable: df[i] = df[i].astype('str').replace('\.0', '', regex=True)
## Trim columns
df = df[['Design', 'Bias'] + param_latex_variable]
## Pivot
df_wide = df.pivot(index=param_latex_variable, columns='Design', values='Bias') # only central measure, no CIs

### II --- Functions
def telsendimg(path='', cap=''):
    with open(path, 'rb') as f:
        telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
                           images=[f],
                           captions=[cap])
def heatmap(input=pd.DataFrame(), mask=None, colourmap='vlag', outputfile='', title='', lb=lbound, ub=ubound):
    fig = plt.figure()
    sns.heatmap(input,
                mask=mask,
                annot=True,
                cmap=colourmap,
                center=0,
                annot_kws={'size':7},
                vmin=lb,
                vmax=ub,
                xticklabels=True,
                yticklabels=True,)
    plt.title(title)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=7)
    fig.tight_layout()
    fig.savefig(path_method + outputfile)
    return fig

### III --- Heatmap
fig = heatmap(input=df_wide,
              outputfile='VEMethod_Sim1b_PureDesignBias_Heatmap.png',
              title="Simulated bias in VE estimates across study designs\n under 'perfect' parameter assumptions\n (T, \u03BC(VE), \u0398(v) on vertical axis)")
telsendimg(path_method + 'VEMethod_Sim1b_PureDesignBias_Heatmap.png',
           cap="Simulated bias in VE estimates across study designs\n under 'perfect' parameter assumptions\n (T, \u03BC(VE), \u0398(v) on vertical axis)")

### End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')