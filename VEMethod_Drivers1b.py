# Comparing VE methodologies
# Time-varying vaccine coverage
# 'Leaky' vaccines, heterogeneous VE, heterogeneous testing and vax preferences
# Drivers1b: Use simulated biases from Sim1b (designs M and parameters Xi) to estimate relative importance of Xi by M

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
file_sim = 'VEMethod_Sim1b_Parallel_CloudVersion_NoCI_FIN.parquet'  # take the raw version (all latex)
# Data frame
df = pd.read_parquet(path_method + file_sim)
n_row = len(df.index)  # number of simulations done

# 0 --- Setup
# max and min of estimated bias
bias_lb = -60
bias_ub = 60
# For design dummies
nM = df['Design'].nunique()
# Parameters
param_latex = ['N', 'T', 'Theta_v', 'p_v', 'Theta_tau', 'p_tau', 'ktau', 'alpha_b', 'kalpha', 'mew_ve']
param_greek = ['N', 'T', '\u0398(v)', 'p(v)', '\u0398(\u03C4)', 'p(\u03C4)', 'k(\u03C4)', '\u03B1(b)', 'k(\u03B1)', '\u03BC(VE)']
dict_param = dict(zip(param_latex, param_greek))
list_Xi = ['T', 'Theta_v', 'p_v', 'Theta_tau', 'p_tau', 'ktau', 'kalpha', 'mew_ve']  # excludes N, and alpha_b since these are static
list_Bias_Xi = ['Bias'] + list_Xi
list_M = list(df['Design'].unique())
# list_Mnum = list(range(0, len(list_M)))
# dict_M = dict(zip(list_M, list_Mnum))

# 0 --- Additional cleaning
# Range of bias considered
cond_bias = ((df['Bias'] >= bias_lb) & (df['Bias'] <= bias_ub))
df = df[cond_bias]
df_wide = df.pivot(index=param_latex, columns='Design', values='Bias')  # calculate number of simulations kept with 'usable' bias
n_sim = len(df_wide.index)
# Redefine ktau and kalpha as absolute deviation from 1
df['ktau'] = np.abs(df['ktau'] - 1)  # ktau
df['kalpha'] = np.abs(df['kalpha'] - 1)  # kalpha
# Redefine p_v and p_tau as shortfall from 1 (deviation from static coverage and perfect test-willing testing)
df['p_v'] = np.abs(df['p_v'] - 1)
df['p_tau'] = np.abs(df['p_tau'] - 1)  # p_tau
# Redefine Theta_tau and Theta_v as shortall from 1 (deviation from full test-willing population and vax-willing population)
df['Theta_tau'] = np.abs(df['Theta_tau'] - 1)  # Theta_tau
df['Theta_v'] = np.abs(df['Theta_v'] - 1) # Theta_v
# Redefine mew_ve as shortfall from 1 (deviation from perfect vaccine)
df['mew_ve'] = np.abs(df['mew_ve'] - 1)  # mew_ve

# 0 --- Define functions


def OLS(formula, data=df, method='pinv', cov='HC3', dropM=True):
    mod = smf.ols(formula=formula, data=data)
    result = mod.fit(method=method, cov_type=cov)
    beta = result.params
    ci = result.conf_int()
    est = pd.concat([beta, ci], axis=1)
    if dropM == True: est = est.iloc[nM:, :]
    elif dropM == False: est = est.iloc[1:, :]
    est = est.reset_index()
    est.columns = ['Parameters', 'Point Estimate', 'Lower Bound', 'Upper Bound']
    return mod, result, est


def FE(formula, data=df, index=[''], dropM=True):
    d = df.set_index(index) # sets the entity
    mod = PanelOLS.from_formula(formula, d)
    result = mod.fit(cov_type='clustered', cluster_entity=True)
    beta = pd.DataFrame(result.params)
    ci = pd.DataFrame(result.conf_int())
    est = pd.concat([beta, ci], axis=1)
    if dropM == True: est = est.iloc[nM:, :]
    elif dropM == False: est = est.iloc[1:, :]
    est = est.reset_index()
    est.columns = ['Parameters', 'Point Estimate', 'Lower Bound', 'Upper Bound']
    return mod, result, est


def heatmap(input=pd.DataFrame(),
            mask=None,
            colourmap='vlag',
            vmin=-20,
            vmax=20,
            outputfile='',
            brackettext='All (With Design Dummies)',
            title='Yes'):
    fig = plt.figure()
    sns.heatmap(input, mask=mask, annot=True, cmap=colourmap, center=0, annot_kws={'size':14}, vmin=vmin, vmax=vmax)
    if title == 'Yes':
        plt.title(brackettext + ':\n' +  'Association Between Parameters \u039E and Absolute Bias')
    elif title == 'Rescaled':
        plt.title(brackettext + ':\n' + 'Rescaled Association Between Parameters \u039E and Absolute Bias')
    elif title == 'No':
        pass
    elif title == 'Robustness':
        plt.title('Average and Largest Sensitivity \n' +
                  'Between Parameters \u039E and Absolute Bias \n' +
                  '(Parameter With Largest Sensitivity \n on Vertical Axis)',
                  loc='left',
                  fontsize=7)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    fig.tight_layout()
    fig.savefig(path_method + outputfile)
    return fig

# I --- Regression setup
# Number of sims kept
print('VEMethod_Drivers1b' + '\n\n' +
      'Number of "valid" simulations: ' + str(n_sim) + '\n' +
      'Bias = [' + str(bias_lb) + ', ' + str(bias_ub) + ']')  # number of simulations with 'usable' bias
# Equations
M = 'C(Design)'
Xi = '+'.join(list_Xi)
eqn_li='Bias ~ + ' + M + ' + ' + Xi
eqn_li_Mspec='Bias ~ ' + Xi

# II.A --- Estimating OLS regression + Heatmaps
print('II.A --- Estimating OLS regression + Heatmaps')
# Linear
mod, result, est = OLS(formula=eqn_li)
est = est.round(3)
est['Parameters'] = est['Parameters'].replace(dict_param)
est = est.set_index('Parameters')
# dfi.export(est, path_method + 'VEMethod_Drivers1b_FEest_Li.png')
fig = heatmap(input=est,
              outputfile='VEMethod_Drivers1b_FEest_Li_Heatmap.png',
              title='Yes')
# M-specific
est_consol = pd.DataFrame(columns=['Design', 'Parameters', 'Point Estimate', 'Lower Bound', 'Upper Bound'])
R = 1
for M in tqdm(list_M):
    d = df[df['Design'] == M]
    mod, result, est = OLS(formula=eqn_li_Mspec, data=d, dropM=False)
    est = est.round(3)
    est['Parameters'] = est['Parameters'].replace(dict_param)
    est['Design'] = M
    est_consol = pd.concat([est_consol, est], axis=0)  # consolidate DFs before setting index in M-spec DF
    del est['Design']  # don't need this for the M-spec heatmap
    est = est.set_index('Parameters')
    fig = heatmap(input=est,
                  outputfile='VEMethod_Drivers1b_FEest_Li_Mspec_Heatmap' + str(R) + '.png',
                  brackettext=M,
                  title='Yes')  # switch on/off title
    R += 1
# 'Robustness' heatmap: average beta and highest beta parameter
# Average beta (absolute)
d = est_consol.copy()
d['Point Estimate'] = np.abs(d['Point Estimate'])
est_consol_avg = d.groupby('Design')['Point Estimate'].agg('mean').reset_index(drop=False)
est_consol_avg = est_consol_avg.rename(columns={'Point Estimate': 'Avg. Abs. Sens.'})
# Highest beta-parameter
est_consol_max = pd.DataFrame(columns=['Design', 'Parameters', 'Point Estimate'])
for M in tqdm(list_M):
    d = est_consol[est_consol['Design'] == M]
    d['Absolute Point Estimate'] = np.absolute(d['Point Estimate'])
    d = d.loc[d['Absolute Point Estimate'] == d['Absolute Point Estimate'].max(),
              ['Design', 'Parameters', 'Point Estimate']]
    est_consol_max = pd.concat([est_consol_max, d], axis=0)
est_consol_max = est_consol_max.reset_index(drop=True)
est_consol_max = est_consol_max.rename(columns={'Point Estimate': 'Largest Sens.',
                                                'Parameters': 'Largest Sens. Param.'})
# Altogether
est_consol_robustness = est_consol_avg.merge(est_consol_max, on='Design', how='outer')
est_consol_robustness = est_consol_robustness.set_index(['Design', 'Largest Sens. Param.'])
est_consol_robustness = est_consol_robustness[['Avg. Abs. Sens.', 'Largest Sens.']] # order of columns
mask = np.zeros_like(est_consol_robustness)
# mask[:,0] = True # set all rows in column 1 to True
fig = heatmap(input=est_consol_robustness,
              outputfile='VEMethod_Drivers1b_FEest_Li_Mspec_Robustness_Heatmap.png',
              title='No') # No title

# Interim --- Pause to not trigger flood control
# print('Interim --- Pause to not trigger flood control')
# time.sleep(30)

# II.B --- 'Realistic' Values
print("II.B --- 'Realistic' Values")
# Function and parameter values
real_p_v = 0.5
real_Theta_tau = 0.75
real_Theta_v = 0.7
real_p_tau = 0.5
real_ktau = 0.75
real_kalpha = 0.75
real_mew_ve = 0.5

# cond_omicron1 = ((df['p_v'] == 0.5) &
#              (df['Theta_tau'] == 0.75) &
#              (df['p_tau'] == 0.5) &
#              (df['ktau'] == 0.75) &
#              (df['kalpha'] == 0.75))


def realism(frame,
            p_v=real_p_v,
            Theta_tau=real_Theta_tau,
            Theta_v=real_Theta_v,
            p_tau=real_p_tau,
            ktau=real_ktau,
            kalpha=real_kalpha,
            mew_ve = real_mew_ve): # run this function before converting LaTeX into Greek
    frame.loc[frame['Parameters'] == 'p_v',
              ['Point Estimate', 'Lower Bound', 'Upper Bound']] = frame[['Point Estimate', 'Lower Bound',
                                                                         'Upper Bound']] * (1-p_v)  # p_v = 0.15
    frame.loc[frame['Parameters'] == 'Theta_v',
              ['Point Estimate', 'Lower Bound', 'Upper Bound']] = frame[['Point Estimate', 'Lower Bound',
                                                                         'Upper Bound']] * (1-Theta_v)  # Theta_v = 0.7
    frame.loc[frame['Parameters'] == 'Theta_tau',
              ['Point Estimate', 'Lower Bound', 'Upper Bound']] = frame[['Point Estimate', 'Lower Bound',
                                                                         'Upper Bound']] * (1-Theta_tau)  # Theta_tau = 0.75
    frame.loc[frame['Parameters'] == 'p_tau',
              ['Point Estimate', 'Lower Bound', 'Upper Bound']] = frame[['Point Estimate', 'Lower Bound',
                                                                         'Upper Bound']] * (1-p_tau)  # p_tau = 0.5
    frame.loc[frame['Parameters'] == 'ktau',
              ['Point Estimate', 'Lower Bound', 'Upper Bound']] = frame[['Point Estimate', 'Lower Bound',
                                                                         'Upper Bound']] * (1-ktau)  # ktau = 0.75
    frame.loc[frame['Parameters'] == 'kalpha',
              ['Point Estimate', 'Lower Bound', 'Upper Bound']] = frame[['Point Estimate', 'Lower Bound',
                                                                         'Upper Bound']] * (1-kalpha)  # kalpha = 0.75
    frame.loc[frame['Parameters'] == 'mew_ve',
              ['Point Estimate', 'Lower Bound', 'Upper Bound']] = frame[['Point Estimate', 'Lower Bound',
                                                                        'Upper Bound']] * (1-mew_ve)  # mew_ve = 0.5
    return frame


# M-specific (may need to run twice; switch on the delete line if so)
# del est_realistic_consol
est_realistic_consol = pd.DataFrame(columns=['Design', 'Parameters', 'Point Estimate', 'Lower Bound', 'Upper Bound'])
R = 1
for M in tqdm(list_M):
    d = df[df['Design'] == M]
    mod, result, est_realistic = OLS(formula=eqn_li_Mspec, data=d, dropM=False)
    est_realistic = est_realistic.round(3)
    est_realistic = realism(frame=est_realistic)  # rescale to realistic parameter values
    est_realistic['Parameters'] = est_realistic['Parameters'].replace(dict_param)
    est_realistic['Design'] = M
    est_realistic_consol = pd.concat([est_realistic_consol, est_realistic], axis=0)  # consolidate DFs before setting index in M-spec DF
    del est_realistic['Design']  # don't need this for the M-spec heatmap
    est_realistic = est_realistic.set_index('Parameters')
    fig = heatmap(input=est_realistic,
                  outputfile='VEMethod_Drivers1b_FEest_Realistic_Li_Mspec_Heatmap' + str(R) + '.png',
                  brackettext=M,
                  title='Rescaled')  # switch on/off title
    R += 1

# Interim --- Pause to not trigger flood control
# print('Interim --- Pause to not trigger flood control')
# time.sleep(30)

# II.C --- 'Realistic' Values (Reduced Bias)
print("II.C --- 'Realistic' Values (Reduced Bias)")
# Function and parameter values
real2_p_v = 0.5
real2_Theta_tau = 0.75
real2_Theta_v = 0.7
real2_p_tau = 0.75
real2_ktau = 0.95
real2_kalpha = 0.75
real2_mew_ve = 0.5

# cond_omicron1 = ((df['p_v'] == 0.5) &
#              (df['Theta_tau'] == 0.75) &
#              (df['p_tau'] == 0.5) &
#              (df['ktau'] == 0.75) &
#              (df['kalpha'] == 0.75))'

# M-specific (may need to run twice; switch on the delete line if so)
# del est_realistic2_consol
est_realistic2_consol = pd.DataFrame(columns=['Design', 'Parameters', 'Point Estimate', 'Lower Bound', 'Upper Bound'])
R = 1
for M in tqdm(list_M):
    d = df[df['Design'] == M]
    mod, result, est_realistic2 = OLS(formula=eqn_li_Mspec, data=d, dropM=False)
    est_realistic2 = est_realistic2.round(3)
    est_realistic2 = realism(frame=est_realistic2,
                             p_v=real2_p_v,
                             Theta_tau=real2_Theta_tau,
                             Theta_v=real2_Theta_v,
                             p_tau=real2_p_tau,
                             ktau=real2_ktau,
                             kalpha=real2_kalpha,
                             mew_ve=real2_mew_ve)  # rescale to realistic parameter values
    est_realistic2['Parameters'] = est_realistic2['Parameters'].replace(dict_param)
    est_realistic2['Design'] = M
    est_realistic2_consol = pd.concat([est_realistic2_consol, est_realistic2], axis=0)  # consolidate DFs before setting index in M-spec DF
    del est_realistic2['Design']  # don't need this for the M-spec heatmap
    est_realistic2 = est_realistic2.set_index('Parameters')
    fig = heatmap(input=est_realistic2,
                  outputfile='VEMethod_Drivers1b_FEest_Realistic2_Li_Mspec_Heatmap' + str(R) + '.png',
                  brackettext=M,
                  title='Rescaled')  # switch on/off title
    R += 1

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')