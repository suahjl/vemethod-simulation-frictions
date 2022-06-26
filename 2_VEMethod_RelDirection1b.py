#### Comparing VE methodologies
#### Time-varying vaccine coverage
#### 'Leaky' vaccines, heterogeneous VE, heterogeneous testing and vax preferences
#### RelDirection1b: Use simulated biases from Sim1b to check consistency in directional differences between designs M

import gc
import pandas as pd
import numpy as np
import time
from datetime import date, timedelta
import itertools
import time
import telegram_send
import dataframe_image as dfi
import shutil
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff

time_start = time.time()

### 0 --- Preliminaries
## Path
path_method = 'D:/Users/ECSUAH/Desktop/Quant/HealthEconomy/VaccinesAssessment/VEMethod/' # check this
## File name
file_sim = 'VEMethod_Sim1b_Parallel_CloudVersion_NoCI_FIN.parquet' # take the raw version (all latex)
## Data frame
df = pd.read_parquet(path_method + file_sim)
n_row = len(df.index) # number of simulations done

### 0 --- Setup
## WHAT TO RUN?
run1a = 1
run1b = 1
run1c = 1
run1d = 1
## Key objects
param_latex = ['N', 'T', 'Theta_v', 'p_v', 'Theta_tau', 'p_tau', 'ktau', 'alpha_b', 'kalpha', 'mew_ve']
ref_design='Cohort: True y'
list_design = list(df['Design'].unique())
list_design.remove(ref_design) # move ref to number 0
list_design.insert(0, ref_design) # move ref to number 0
n_list_design = list(range(len(list_design)))
dict_design_n = dict(zip(list_design, n_list_design))
## Rename columns (designs)
df['Design'] = df['Design'].replace(dict_design_n)
## Change bias to absolute bias
df['Bias'] = np.abs(df['Bias'])
## Wide version (for later)
df_wide = df.pivot(index=param_latex, columns='Design', values='Bias').reset_index()
## Temp (change to loop for sustainability): on / off for blanking out extreme values
option_blank = 0
if option_blank == 1:
    print('Extreme values are DROPPED')
    df.loc[(df['Bias'] > 20), 'Bias'] = np.nan # already in absolutes
else:
    print('Extreme values are KEPT')
## Functions
def telsendimg(path='', cap=''):
    with open(path, 'rb') as f:
        telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
                           images=[f],
                           captions=[cap])
def telsendfile(path='', cap=''):
    with open(path, 'rb') as f:
        telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
                           files=[f],
                           captions=[cap])

### I.A --- Check distribution of biases (majority lies in xx to xx): percentiles
if run1a == 1:
    ## All
    df_perc = pd.DataFrame(columns=['Percentile', 'AbsoluteBias'])
    for i in tqdm(list(np.arange(5,100,5))):
        d = pd.DataFrame(zip([0],[0]), columns=['Percentile', 'AbsoluteBias'])
        d['Percentile'] = i
        d['AbsoluteBias'] = np.percentile(df['Bias'].dropna(), i)
        df_perc = pd.concat([df_perc, d])
    df_perc.to_html(path_method + 'VEMethod_RelDirection1b_Perc.html', index=False)
    df_perc.to_csv(path_method + 'VEMethod_RelDirection1b_Perc.csv', index=False)
    dfi.export(df_perc, path_method + 'VEMethod_RelDirection1b_Perc.png')
    telsendimg(path=path_method + 'VEMethod_RelDirection1b_Perc.png',
               cap='Percentiles of absolute VE (aggregate)')
    telsendfile(path=path_method + 'VEMethod_RelDirection1b_Perc.html',
                cap='Percentiles of absolute VE (aggregate)')
    telsendfile(path=path_method + 'VEMethod_RelDirection1b_Perc.csv',
                cap='Percentiles of absolute VE (aggregate)')
    ## By study design
    df_perc_M = pd.DataFrame(columns=['Design', 'Percentile', 'AbsoluteBias'])
    for i, j in tqdm(zip(n_list_design, list_design)):
        d_temp = df[df['Design'] == i]
        for k in list(np.arange(5,100,5)):
            d = pd.DataFrame(zip([0], [0], [0]), columns=['Design', 'Percentile', 'AbsoluteBias'])
            d['Design'] = j
            d['Percentile'] = k
            d['AbsoluteBias'] = np.percentile(d_temp['Bias'].dropna(), k)
            df_perc_M = pd.concat([df_perc_M, d])
    df_perc_M.to_html(path_method + 'VEMethod_RelDirection1b_Perc_M.html', index=False)
    df_perc_M.to_csv(path_method + 'VEMethod_RelDirection1b_Perc_M.csv', index=False)
    telsendfile(path=path_method + 'VEMethod_RelDirection1b_Perc_M.html',
                cap='Percentiles of absolute VE (by study design)')
    telsendfile(path=path_method + 'VEMethod_RelDirection1b_Perc_M.csv',
                cap='Percentiles of absolute VE (by study design)')

### I.B --- Check distribution of biases: kernel density
if run1b == 1:
    ## All
    fig = ff.create_distplot([df['Bias'].dropna()],
                             group_labels=['All'],
                             bin_size=0.1,
                             curve_type='kde',
                             show_hist=False)
    fig.update_layout(title='Distribution of Estimation Bias (All Designs)',
                      plot_bgcolor='white',
                      font=dict(color='black'))
    fig.write_image(path_method + 'VEMethod_RelDirection1b_Dist.png')
    telsendimg(path=path_method + 'VEMethod_RelDirection1b_Dist.png',
               cap='Distribution of Estimation Bias (All Designs)')
    ## By study design
    for i in tqdm(n_list_design):
        if i == 0:
            list_bias_M = [df.loc[df['Design'] == i, 'Bias'].reset_index(drop=True).dropna().to_list()]
        elif i > 0:
            list_bias_M = list_bias_M + [df.loc[df['Design'] == i, 'Bias'].reset_index(drop=True).dropna().to_list()]
    fig = ff.create_distplot(list_bias_M,
                             group_labels=list_design,
                             bin_size=0.5,
                             curve_type='kde',
                             show_hist=False)
    fig.update_layout(title='Distribution of Estimation Bias (By Designs)',
                      plot_bgcolor='white',
                      font=dict(color='black'))
    fig.write_image(path_method + 'VEMethod_RelDirection1b_Dist_M.png')
    telsendimg(path=path_method + 'VEMethod_RelDirection1b_Dist_M.png',
               cap='Distribution of Estimation Bias (By Designs)')

### I.C --- Relative ranks by Xi
if run1c == 1:
    df_rank = pd.DataFrame(df.groupby(param_latex)['Bias'].rank()).rename(columns={'Bias': 'Rank'})
    df = df.merge(df_rank, left_index=True, right_index=True, how='left')
    tab_rank = pd.DataFrame(list_design, columns=['Design'])
    for i in tqdm(['RankMean', 'RankSD', 'RankSkew', 'RankKurtosis']):
        tab_rank[i] = 0 # placeholder
    for i,j in tqdm(zip(n_list_design, list_design)):
        d = df[df['Design'] == i]
        tab_rank.loc[tab_rank['Design'] == j, 'RankMean'] = '{:.1f}'.format(d['Rank'].mean())
        tab_rank.loc[tab_rank['Design'] == j, 'RankSD'] = '{:.1f}'.format(d['Rank'].std())
        tab_rank.loc[tab_rank['Design'] == j, 'RankSkew'] = '{:.1f}'.format(d['Rank'].skew())
        tab_rank.loc[tab_rank['Design'] == j, 'RankKurtosis'] = '{:.1f}'.format(d['Rank'].kurtosis())
    dfi.export(tab_rank, path_method + 'VEMethod_RelDirection1b_Rank.png')
    telsendimg(path=path_method + 'VEMethod_RelDirection1b_Rank.png',
               cap='Distributional Measures of Ranks of Absolute Bias Between Study Designs By Parameter Set \u039E')

### I.D --- Pairwise ranks
if run1d == 1:
    m0 = list(range(len(n_list_design))) # manual
    m1 = list(range(1, len(n_list_design)))
    m2 = list(range(2, len(n_list_design)))
    m3 = list(range(3, len(n_list_design)))
    m4 = list(range(4, len(n_list_design)))
    m5 = list(range(5, len(n_list_design)))
    m6 = list(range(6, len(n_list_design)))
    m7 = list(range(7, len(n_list_design)))
    m8 = list(range(8, len(n_list_design)))
    m9 = list(range(9, len(n_list_design)))
    ## Column names
    # 0
    k = 1
    for i in tqdm(m1):
        if k == 1: p0 = ['0-' + str(i)]
        elif k > 1: p0 = p0 + ['0-' + str(i)]
        k += 1
    # 1
    k = 1
    for i in tqdm(m2):
        if k == 1: p1 = ['1-' + str(i)]
        elif k > 1: p1 = p1 + ['1-' + str(i)]
        k += 1
    # 2
    k = 1
    for i in tqdm(m3):
        if k == 1: p2 = ['2-' + str(i)]
        elif k > 1: p2 = p2 + ['2-' + str(i)]
        k += 1
    # 3
    k = 1
    for i in tqdm(m4):
        if k == 1: p3 = ['3-' + str(i)]
        elif k > 1: p3 = p3 + ['3-' + str(i)]
        k += 1
    # 4
    k = 1
    for i in tqdm(m5):
        if k == 1: p4 = ['4-' + str(i)]
        elif k > 1: p4 = p4 + ['4-' + str(i)]
        k += 1
    # 5
    k = 1
    for i in tqdm(m6):
        if k == 1: p5 = ['5-' + str(i)]
        elif k > 1: p5 = p5 + ['5-' + str(i)]
        k += 1
    # 6
    k = 1
    for i in tqdm(m7):
        if k == 1: p6 = ['6-' + str(i)]
        elif k > 1: p6 = p6 + ['6-' + str(i)]
        k += 1
    # 7
    k = 1
    for i in tqdm(m8):
        if k == 1: p7 = ['7-' + str(i)]
        elif k > 1: p7 = p7 + ['7-' + str(i)]
        k += 1
    # 8
    k = 1
    for i in tqdm(m9):
        if k == 1: p8 = ['8-' + str(i)]
        elif k > 1: p8 = p8 + ['8-' + str(i)]
        k += 1
    # all
    p_all = p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8
    ## Pairwise comparison
    df_pair = df_wide[param_latex + n_list_design]
    for i in p_all: df_pair[i] = 0 # placeholder values
    # 0
    ref = 0
    for i, j in tqdm(zip(p0, m1)):
        df_pair.loc[(df_pair[ref] >= df_pair[j]), i] = 1 # ref has higher abs bias
        df_pair.loc[(df_pair[ref] < df_pair[j]), i] = -1 # ref has lower abs bias
    # 1
    ref = 1
    for i, j in tqdm(zip(p1, m2)):
        df_pair.loc[(df_pair[ref] >= df_pair[j]), i] = 1  # ref has higher abs bias
        df_pair.loc[(df_pair[ref] < df_pair[j]), i] = -1  # ref has lower abs bias
    # 2
    ref = 2
    for i, j in tqdm(zip(p2, m3)):
        df_pair.loc[(df_pair[ref] >= df_pair[j]), i] = 1  # ref has higher abs bias
        df_pair.loc[(df_pair[ref] < df_pair[j]), i] = -1  # ref has lower abs bias
    # 3
    ref = 3
    for i, j in tqdm(zip(p3, m4)):
        df_pair.loc[(df_pair[ref] >= df_pair[j]), i] = 1  # ref has higher abs bias
        df_pair.loc[(df_pair[ref] < df_pair[j]), i] = -1  # ref has lower abs bias
    # 4
    ref = 4
    for i, j in tqdm(zip(p4, m5)):
        df_pair.loc[(df_pair[ref] >= df_pair[j]), i] = 1  # ref has higher abs bias
        df_pair.loc[(df_pair[ref] < df_pair[j]), i] = -1  # ref has lower abs bias
    # 5
    ref = 5
    for i, j in tqdm(zip(p5, m6)):
        df_pair.loc[(df_pair[ref] >= df_pair[j]), i] = 1  # ref has higher abs bias
        df_pair.loc[(df_pair[ref] < df_pair[j]), i] = -1  # ref has lower abs bias
    # 6
    ref = 6
    for i, j in tqdm(zip(p6, m7)):
        df_pair.loc[(df_pair[ref] >= df_pair[j]), i] = 1  # ref has higher abs bias
        df_pair.loc[(df_pair[ref] < df_pair[j]), i] = -1  # ref has lower abs bias
    # 7
    ref = 7
    for i, j in tqdm(zip(p7, m8)):
        df_pair.loc[(df_pair[ref] >= df_pair[j]), i] = 1  # ref has higher abs bias
        df_pair.loc[(df_pair[ref] < df_pair[j]), i] = -1  # ref has lower abs bias
    # 8
    ref = 8
    for i, j in tqdm(zip(p8, m9)):
        df_pair.loc[(df_pair[ref] >= df_pair[j]), i] = 1  # ref has higher abs bias
        df_pair.loc[(df_pair[ref] < df_pair[j]), i] = -1  # ref has lower abs bias
    ## Tabulate
    # %
    k = 1
    for i in p_all:
        if k == 1:
            df_paircomp = df_pair[i].value_counts(normalize=True) # don't reset index
        elif k > 1:
            d = df_pair[i].value_counts(normalize=True) # don't reset index to use concat
            df_paircomp = pd.concat([df_paircomp, d], axis=1)
        k += 1
    df_paircomp = 100 * df_paircomp.transpose().fillna(0)
    df_paircomp = df_paircomp.rename(columns={1:'Left>=Right',
                                              0:'EitherIsNAN',
                                              -1:'Left<Right'})
    # N
    k = 1
    for i in p_all:
        if k == 1:
            df_paircomp_N = df_pair[i].value_counts(normalize=False)  # don't reset index
        elif k > 1:
            d = df_pair[i].value_counts(normalize=False)  # don't reset index to use concat
            df_paircomp_N = pd.concat([df_paircomp_N, d], axis=1)
        k += 1
    df_paircomp_N = df_paircomp_N.transpose().fillna(0)
    df_paircomp_N = df_paircomp_N.rename(columns={1: 'Left>=Right',
                                              0: 'EitherIsNAN',
                                              -1: 'Left<Right'})
    ## Generate heat map
    # %
    fig = plt.figure()
    sns.heatmap(df_paircomp,
                mask=False,
                annot=True,
                cmap='Blues',
                annot_kws={'size': 5},
                vmin=0,
                vmax=100,
                xticklabels=True,
                yticklabels=True)
    plt.title('Pairwise Rank (% of Parameter Sets \u039E) of Estimation Bias (Y-Axis: Study Designs)',
              fontsize=8)
    fig.tight_layout()
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=6)
    fig.savefig(path_method + 'VEmethod_RelDirection1b_Pairwise.png')
    telsendimg(path=path_method + 'VEmethod_RelDirection1b_Pairwise.png',
               cap='Pairwise Rank (% of Parameter Sets \u039E) of Estimation Bias (Y-Axis: Study Designs)')
    # N
    fig = plt.figure()
    sns.heatmap(df_paircomp_N,
                mask=False,
                annot=True,
                cmap='Blues',
                annot_kws={'size': 5},
                vmin=0, # no vmax for N version
                xticklabels=True,
                yticklabels=True,
                fmt='g') # integers
    plt.title('Pairwise Rank (N of Parameter Sets \u039E) of Estimation Bias (Y-Axis: Study Designs)',
              fontsize=8)
    fig.tight_layout()
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=6)
    fig.savefig(path_method + 'VEmethod_RelDirection1b_Pairwise_N.png')
    telsendimg(path=path_method + 'VEmethod_RelDirection1b_Pairwise_N.png',
               cap='Pairwise Rank (N of Parameter Sets \u039E) of Estimation Bias (Y-Axis: Study Designs)')
    ## Table as legend
    df_dict_design = pd.DataFrame(dict_design_n.items(), columns=['StudyDesign', 'Encoding'])
    dfi.export(df_dict_design, path_method + 'VEmethod_RelDirection1b_DictDesign.png')
    telsendimg(path=path_method + 'VEmethod_RelDirection1b_DictDesign.png',
               cap='Reference: Study Design and Encoding')

### End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')