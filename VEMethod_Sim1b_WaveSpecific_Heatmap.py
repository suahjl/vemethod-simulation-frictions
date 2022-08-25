# Generate heatmap of design biases with only specific parameter sets reflective of past SARS-CoV-2 waves

import gc
import pandas as pd
import numpy as np
from datetime import date, timedelta
import itertools
import time
# import telegram_send
import dataframe_image as dfi
import shutil
from tqdm import tqdm

import statsmodels.formula.api as smf
from linearmodels import PanelOLS

import matplotlib.pyplot as plt
import seaborn as sns

time_start = time.time()

# 0 --- Preliminaries
# Path
path_method = 'Output/' # check this
# File name
file_sim = 'VEMethod_Sim1b_Parallel_CloudVersion_NoCI_FIN.parquet' # take the raw version (all latex)
# Data frame
df = pd.read_parquet(path_method + file_sim)
n_row = len(df.index)  # number of simulations done
# Pandas global settings
pd.set_option('precision', 0)

# 0 --- Setup
# Parameters
param_latex = ['N', 'T', 'Theta_v', 'p_v', 'Theta_tau', 'p_tau', 'ktau', 'alpha_b', 'kalpha', 'mew_ve']
param_latex2 = ['N', 'Theta_v', 'p_v', 'Theta_tau', 'p_tau', 'ktau', 'alpha_b', 'kalpha', 'mew_ve', 'T']  # T is placed last
param_greek = ['N', 'T', '\u0398(v)', 'p(v)', '\u0398(\u03C4)', 'p(\u03C4)', 'k(\u03C4)', '\u03B1(b)', 'k(\u03B1)', '\u03BC(VE)']
dict_param = dict(zip(param_latex, param_greek))
param_latex_variable = ['p_v', 'Theta_v', 'Theta_tau', 'p_tau', 'ktau', 'kalpha', 'mew_ve']

# I --- Cleaning
# Extreme values
ubound = 20
lbound = -20
# df.loc[df['Bias'] >= ubound, 'Bias'] = np.nan  # indicative of inability to estimate
# df.loc[df['Bias'] <= lbound, 'Bias'] = np.nan  # indicative of inability to estimate
# Wave-specific sets
cond_wild = ((df['p_v'] == 0.15) &
             (df['Theta_v'] == 0.3) &
             (df['Theta_tau'] == 1) &
             (df['p_tau'] == 1) &
             (df['ktau'] == 1) &
             (df['kalpha'] == 1.25) &
             (df['mew_ve'] == 0.9))

cond_alpha = ((df['p_v'] == 0.15) &
              (df['Theta_v'] == 0.3) &
              (df['Theta_tau'] == 0.75) &
              (df['p_tau'] == 1) &
              (df['ktau'] == 1) &
              (df['kalpha'] == 1.25) &
              (df['mew_ve'] == 0.9))

cond_delta = ((df['p_v'] == 0.3) &
              (df['Theta_v'] == 0.5) &
              (df['Theta_tau'] == 0.75) &
              (df['p_tau'] == 0.75) &
              (df['ktau'] == 1) &
              (df['kalpha'] == 1) &
              (df['mew_ve'] == 0.7))

cond_omicron1 = ((df['p_v'] == 0.5) &
                 (df['Theta_v'] == 0.7) &
                 (df['Theta_tau'] == 0.75) &
                 (df['p_tau'] == 0.5) &
                 (df['ktau'] == 0.75) &
                 (df['kalpha'] == 0.75) &
                 (df['mew_ve'] == 0.5))

cond_omicron2 = ((df['p_v'] == 0.85) &
                 (df['Theta_v'] == 0.7) &
                 (df['Theta_tau'] == 0.5) &
                 (df['p_tau'] == 0.5) &
                 (df['ktau'] == 0.75) &
                 (df['kalpha'] == 0.75) &
                 (df['mew_ve'] == 0.5))

cond_severe_alpha = ((df['p_v'] == 0.15) &
                     (df['Theta_v'] == 0.3) &
                     (df['Theta_tau'] == 1) &
                     (df['p_tau'] == 1) &
                     (df['ktau'] == 1) &
                     (df['kalpha'] == 1) &
                     (df['mew_ve'] == 0.9))

cond_severe_delta = ((df['p_v'] == 0.3) &
                     (df['Theta_v'] == 0.5) &
                     (df['Theta_tau'] == 1) &
                     (df['p_tau'] == 1) &
                     (df['ktau'] == 1) &
                     (df['kalpha'] == 1) &
                     (df['mew_ve'] == 0.9))

cond_severe_omicron1 = ((df['p_v'] == 0.5) &
                        (df['Theta_v'] == 0.7) &
                        (df['Theta_tau'] == 1) &
                        (df['p_tau'] == 1) &
                        (df['ktau'] == 1) &
                        (df['kalpha'] == 1) &
                        (df['mew_ve'] == 0.9))

cond_severe_omicron2 = ((df['p_v'] == 0.85) &
                        (df['Theta_v'] == 0.7) &
                        (df['Theta_tau'] == 1) &
                        (df['p_tau'] == 1) &
                        (df['ktau'] == 1) &
                        (df['kalpha'] == 1) &
                        (df['mew_ve'] == 0.9))

df = df[(cond_wild | cond_alpha | cond_delta | cond_omicron1 | cond_omicron2 |
         cond_severe_alpha | cond_severe_delta | cond_severe_omicron1 | cond_severe_omicron2)] # severe alpha = severe wild

df.loc[cond_wild, 'Wave'] = 'Scenario A'
df.loc[cond_alpha, 'Wave'] = 'Scenario B'
df.loc[cond_delta, 'Wave'] = 'Scenario C'
df.loc[cond_omicron1, 'Wave'] = 'Scenario D'
df.loc[cond_omicron2, 'Wave'] = 'Scenario E'
df.loc[cond_severe_alpha, 'Wave'] = 'Severe Outcomes (Scenario A & B)'
df.loc[cond_severe_delta, 'Wave'] = 'Severe Outcomes (Scenario C)'
df.loc[cond_severe_omicron1, 'Wave'] = 'Severe Outcomes (Scenario D)'
df.loc[cond_severe_omicron2, 'Wave'] = 'Severe Outcomes (Scenario E)'
# Order of columns
wave = df.pop('Wave')
df.insert(0, wave.name, wave)
# Sort T
df = df.sort_values(by='T', ascending=True).reset_index(drop=False) # sort, then pivot later
# Pivot
df_wide = df.pivot(index=['Wave']+param_latex2, columns='Design', values='Bias') # only central measure, no CIs

# III --- Heatmap
df_wide_grad = df_wide.style.background_gradient(cmap='coolwarm', vmin=lbound, vmax=ubound)
with open(path_method + 'VEMethod_Sim1b_WaveSpecific_Heatmap.html', 'w') as f:
    f.write(df_wide_grad.render())
df_wide_grad.to_excel(path_method + 'VEMethod_Sim1b_WaveSpecific_Heatmap.xlsx', sheet_name='Gradient', float_format='%.2f', na_rep='', index=True)
