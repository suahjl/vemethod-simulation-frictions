# Comparing VE methodologies
# Time-varying vaccine coverage
# 'Leaky' vaccines, heterogeneous VE, heterogeneous testing and vax preferences
# Sim1: Simulate bias across methods M and parameters Xi
# b: beta distribution to simulate individual-level VE
# Parallel processing
# Skip all unnecessary CIs (to reduce computational requirements)
# CloudVersion: larger N, more combos, parquet brotli compression, simple directories
# ReSeed: new seed for legal combos delivering singular matrix errors

import gc
import pandas as pd
import numpy as np
from datetime import date, timedelta
import itertools
import time
# import telegram_send
from tqdm import tqdm

import statsmodels.formula.api as smf

import multiprocess as mp

time_start = time.time()

# 0 --- Preliminaries
# Original data frame
df_original = pd.read_parquet('Output/VEMethod_Sim1b_Parallel_CloudVersion_NoCI.parquet')
# Grand consol VE
VE_consol_all = pd.DataFrame(columns=['N', 'T', 'Theta_v', 'p_v', 'Theta_tau', 'p_tau', 'ktau', 'alpha_b', 'kalpha', 'mew_ve',
                                      'Design', 'Bias', 'VE'])

# 0 --- Parameters
np.random.seed(1237246261) # check seed

# list_N = [1000] # how many people (keep fixed)
# list_T = [20, 30, 40, 50, 60]  # how many days
# list_Theta_v = [0.3, 0.4, 0.5, 0.6, 0.7]  # vaxers share (keep fixed)
# list_p_v = [0.15, 0.3, 0.4, 0.5, 0.6, 0.85, 1]  # probability of getting vaccinated on day x
# list_Theta_tau = [0.5, 0.65, 0.75, 0.85, 1]  # testers share
# list_p_tau = [0.5, 0.65, 0.75, 0.85, 1]  # testing probability if theta_tau = 1
# list_ktau = [0.5, 0.75, 0.9, 1, 1.1, 1.25, 1.5]  # relative propensity to test between v = 1 and v = 0 (captures selection in testing)
# list_alpha_b = [0.025]  # baseline daily infection risk (keep fixed)
# list_kalpha = [0.5, 0.75, 0.9, 1, 1.1, 1.25, 1.5]   # asymmetry in baseline infection risk between vax-willing and vax-unwilling
# list_mew_ve = [0.5, 0.6, 0.7, 0.8, 0.9]  # peak of VE distribution (beta distribution) (if 0.5, then mode = mean)

# paramslist = list(itertools.product(list_N, list_T, list_Theta_v, list_p_v, list_Theta_tau, list_p_tau, list_ktau, list_alpha_b, list_kalpha, list_mew_ve))
paramslist = [(1000, 20, 0.7, 0.85, 0.65, 0.65, 1.1, 0.025, 1.5, 0.9),
              (1000, 20, 0.3, 0.3, 1, 0.85, 0.9, 0.025, 0.5, 0.9),
              (1000, 20, 0.4, 0.15, 1, 0.5, 1.5, 0.025, 0.75, 0.6),
              (1000, 30, 0.4, 0.15, 0.65, 0.75, 0.5, 0.025, 1, 0.8),
              (1000, 40, 0.3, 0.15, 0.5, 0.75, 1.1, 0.025, 1.1, 0.9),
              (1000, 40, 0.5, 0.15, 0.65, 0.65, 0.9, 0.025, 1, 0.5)]
# IMPORTANT CHECKS
for params in paramslist:
    N = params[0]
    T = params[1]
    Theta_v = params[2]
    p_v = params[3]
    Theta_tau = params[4]
    p_tau = params[5]
    ktau = params[6]
    alpha_b = params[7]
    kalpha = params[8]
    mew_ve = params[9]
    d = df_original[(df_original['N'] == N) &
                    (df_original['T'] == T) &
                    (df_original['Theta_v'] == Theta_v) &
                    (df_original['p_v'] == p_v) &
                    (df_original['Theta_tau'] == Theta_tau) &
                    (df_original['p_tau'] == p_tau) &
                    (df_original['ktau'] == ktau) &
                    (df_original['alpha_b'] == alpha_b) &
                    (df_original['kalpha'] == kalpha) &
                    (df_original['mew_ve'] == mew_ve)]
    if len(d) == 0: print('INDEED NOT THERE')
    elif len(d) > 0: raise NotImplementedError
    elif len(d) < 0: raise NotImplementedError

# Pandas global settings
pd.options.display.float_format = '{:.3f}'.format # display max 2 dp for floats (doesn't affect actual float value)

# I --- Define the entire simulation as a mega-sized function


def snsdcomeback(params):
    # Standalone dependencies
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    import gc
    import telegram_send
    import traceback

    # Seed
    np.random.seed(1237246261) # check seed

    # Estimation functions
    eqn = 'yhat~v'
    opt_method = 'newton'
    opt_maxiter = 100
    n_treatment_D = 2
    eqn_ph = 'ttevent~v'
    col_status = 'yhat'
    eqn_nb = 'yhat~v'


    def logitVE(equation=eqn,
                method=opt_method,
                maxiter=opt_maxiter,
                data=pd.DataFrame(),
                n_keep=n_treatment_D):
        import statsmodels.formula.api as smf
        _mod = smf.logit(equation, data=data)
        _result = _mod.fit(method=method, maxiter=maxiter)
        print(_result.summary2())
        _VE = 100 * (1 - np.exp(_result.params))
        _VE = _VE.iloc[1:n_keep]
        _VE = pd.DataFrame(_VE)
        _VE.rename(columns={0: 'VE'}, inplace=True)
        _VE = _VE.round(2)
        return _VE, _result, _mod


    def coxVE(equation=eqn_ph,
              status=col_status,
              method=opt_method,
              maxiter=opt_maxiter,
              data=pd.DataFrame(),
              n_keep=n_treatment_D):
        import statsmodels.formula.api as smf
        _mod = smf.phreg(equation, data=data, status=status)
        _result = _mod.fit(method=method, maxiter=maxiter)
        print(_result.summary())
        _VE = 100 * (1 - pd.DataFrame(np.exp(_result.params)))  # possibly singular object
        _VE = _VE.iloc[0:n_keep - 1]  # no constant
        _VE = pd.DataFrame(_VE)
        _VE.rename(columns={0: 'VE'}, inplace=True)
        _VE = _VE.round(2)
        return _VE, _result, _mod


    def nbVE(equation=eqn_nb,
             offset='PD_v',
             method=opt_method,
             maxiter=opt_maxiter,
             data=pd.DataFrame(),
             n_keep=n_treatment_D):
        import statsmodels.formula.api as smf
        _mod = smf.negativebinomial(equation, data=data, offset=np.log(data[offset]))
        _result = _mod.fit(method=method, maxiter=maxiter)
        print(_result.summary2())
        _VE = 100 * (1 - np.exp(_result.params))
        _VE = _VE.iloc[1:n_keep]
        _VE = pd.DataFrame(_VE)
        _VE.rename(columns={0: 'VE'}, inplace=True)
        _VE = _VE.round(2)
        return _VE, _result, _mod

    def ve_betadist(mu=0.5, alpha=9, n=1000):  # placeholder figures (redefined in the main function itself)
        _beta = (alpha / mu) - alpha
        _x = pd.Series(np.random.beta(alpha, _beta, n))
        return _x

    # Equivalent to 'nested loops'
    N = params[0]
    T = params[1]
    Theta_v = params[2]
    p_v = params[3]
    Theta_tau = params[4]
    p_tau = params[5]
    ktau = params[6]
    alpha_b = params[7]
    kalpha = params[8]
    mew_ve = params[9]

    # The actual simulation
    try:
        print('N = ' + str(N) + '; ' + \
              'T = ' + str(T) + '; ' + \
              'Theta_v = ' + str(Theta_v) + '; ' + \
              'p_v = ' + str(p_v) + '; ' + \
              'Theta_tau = ' + str(Theta_tau) + '; ' + \
              'p_tau = ' + str(p_tau) + '; ' + \
              'ktau = ' + str(ktau) + '; ' + \
              'alpha_b = ' + str(alpha_b) + '; ' + \
              'kalpha = ' + str(kalpha) + '; ' + \
              'mew_ve = ' + str(mew_ve))
        # 0 --- Consol frame
        VE_consol = pd.DataFrame(columns=['Design', 'VE'])  # so that paradoxical runs produce empty DF as output
        # 0 --- Exclude paradoxical combinations
        if (p_tau * ktau > 1):
            print('Impossible combo, skip')  # don't send updates to avoid flood control
            # return VE_consol
        else:
            # I --- Simulate static environment
            # true population line listing (Ni)
            df_i = pd.DataFrame(columns=['id', 'theta_v', 'theta_tau', 've'])
            df_i['id'] = pd.Series(list(range(1, N + 1)))  # assign IDs
            df_i['theta_v'] = np.random.binomial(n=1, p=Theta_v, size=N)  # assign vax-preference parameter
            df_i['theta_tau'] = np.random.binomial(n=1, p=Theta_tau, size=N)  # assign test-preference parameter
            df_i['ve'] = ve_betadist(mu=mew_ve, alpha=9, n=N)  # latent individual VE if jab is taken (1b: beta dist)
            print(pd.crosstab(df_i['theta_v'], df_i['theta_tau']))
            # true population panel data set
            df_it = df_i.copy()
            df_it['t'] = 1
            for i in tqdm(range(2, T + 1)):
                d = df_i.copy()
                d['t'] = i
                df_it = pd.concat([df_it, d], axis=0)
            df_it = df_it.sort_values(by=['id', 't'], ascending=[True, True])
            df_it = df_it.reset_index(drop=True)
            # true panel subsets
            cond_t0 = (df_it['theta_tau'] == 0)
            cond_v0 = (df_it['theta_v'] == 0)
            df_it_t0v0 = df_it[(cond_t0 & cond_v0)].copy()
            df_it_t1v0 = df_it[(~cond_t0 & cond_v0)].copy()
            df_it_t0v1 = df_it[(cond_t0 & ~cond_v0)].copy()
            df_it_t1v1 = df_it[(~cond_t0 & ~cond_v0)].copy()

            # II --- Simulate dynamic environment
            # quadrant 1: never-testers, never-vaxers
            # simulation
            k = 1
            for i in tqdm(range(1, T + 1)):
                cond = (df_it_t0v0['t'] == i)
                d = df_it_t0v0[cond].copy()
                n = d['id'].nunique()  # number of unique IDs in this tvsubset-time
                # vax
                d['v'] = 0  # all never-vaxers
                # vax timing
                d['tv'] = 0  # all never-vaxers
                # test
                d['tau'] = 0  # all never-testers
                # infection
                d['y'] = np.random.binomial(n=1, p=alpha_b * kalpha, size=n)  # all never vaxers
                d.loc[d['y'] == 0, 'y'] = np.nan  # for sorting later
                # observed infection
                d['yhat'] = 0  # never-testers
                # piece together
                if k == 1:
                    d_t0v0 = d.copy()
                else:
                    d_t0v0 = pd.concat([d_t0v0, d], axis=0)
                k += 1
            # true infections y: keeping first infection
            d_t0v0 = d_t0v0.sort_values(by=['id', 'y', 't'], ascending=[True, False, True]).reset_index(
                drop=True)  # sort by ID, infection, time
            d = d_t0v0.groupby(['id', 'y'])['t'].min().reset_index(drop=False).rename(columns={'t': 'ty'})
            d = d[d['y'] == 1].reset_index(drop=True)  # keep only infected ones
            d_t0v0 = d_t0v0.merge(d[['id', 'ty']], on=['id'], how='left')
            cond = (d_t0v0['t'] == d_t0v0['ty'])
            d_t0v0.loc[cond, 'y'] = 1
            d_t0v0.loc[~cond, 'y'] = 0  # vectorised fillna
            d_t0v0['ty'] = d_t0v0['ty'].fillna(0)  # 0 date for those who were not infected by T
            # observed infections timing tyhat
            d_t0v0['tyhat'] = 0  # all never-testers
            # final subset
            d_t0v0 = d_t0v0.sort_values(by=['id', 't'], ascending=[True, True])  # sort by ID and time

            # quadrant 2: testers, never-vaxers
            # simulation
            k = 1
            for i in tqdm(range(1, T + 1)):
                cond = (df_it_t1v0['t'] == i)
                d = df_it_t1v0[cond].copy()
                n = d['id'].nunique()  # number of unique IDs in this tvsubset-time
                # vax
                d['v'] = 0  # all never-vaxers
                # vax timing
                d['tv'] = 0  # all never-vaxers
                # test
                d['tau'] = np.random.binomial(n=1, p=p_tau * ktau, size=n)  # since all unvax, scale by ktau
                # infection
                d['y'] = np.random.binomial(n=1, p=alpha_b * kalpha, size=n)  # all never vaxers
                d.loc[d['y'] == 0, 'y'] = np.nan  # for sorting later
                # observed infection
                d.loc[d['tau'] == 1, 'yhat'] = d['y']
                d.loc[d['tau'] == 0, 'yhat'] = 0
                # piece together
                if k == 1:
                    d_t1v0 = d.copy()
                else:
                    d_t1v0 = pd.concat([d_t1v0, d], axis=0)
                k += 1
            # true infections y: keeping first infection
            d_t1v0 = d_t1v0.sort_values(by=['id', 'y', 't'], ascending=[True, False, True]).reset_index(
                drop=True)  # sort by ID, infection, time
            d = d_t1v0.groupby(['id', 'y'])['t'].min().reset_index(drop=False).rename(columns={'t': 'ty'})
            d = d[d['y'] == 1].reset_index(drop=True)  # keep only infected ones
            d_t1v0 = d_t1v0.merge(d[['id', 'ty']], on=['id'], how='left')
            cond = (d_t1v0['t'] == d_t1v0['ty'])
            d_t1v0.loc[cond, 'y'] = 1
            d_t1v0.loc[~cond, 'y'] = 0  # vectorised fillna
            d_t1v0['ty'] = d_t1v0['ty'].fillna(0)  # 0 date for those who were not infected by T
            # observed infections yhat: resolving clash with true infections
            cond = ((d_t1v0['y'] == 0) & (d_t1v0['yhat'] == 1))  # clash: if true state is null, but not observed state
            d_t1v0.loc[cond, 'yhat'] = 0  # cannot observe infection when true state is null
            d_t1v0.loc[d_t1v0['yhat'].isna(), 'yhat'] = 0
            # observed infections timing tyhat
            d_t1v0 = d_t1v0.sort_values(by=['id', 'yhat', 't'], ascending=[True, False, True]).reset_index(
                drop=True)  # sort by ID, infection, time
            d = d_t1v0.groupby(['id', 'yhat'])['t'].min().reset_index(drop=False).rename(columns={'t': 'tyhat'})
            d = d[d['yhat'] == 1].reset_index(drop=True)  # keep only tested positive ones
            d_t1v0 = d_t1v0.merge(d[['id', 'tyhat']], on=['id'], how='left')
            d_t1v0.loc[d_t1v0['tyhat'].isna(), 'tyhat'] = 0  # those who stayed undetected till end of time
            # final subset
            d_t1v0 = d_t1v0.sort_values(by=['id', 't'], ascending=[True, True])  # sort by ID and time

            # quadrant 3: never-testers, vaxers
            # simulation: vax first
            k = 1
            for i in tqdm(range(1, T + 1)):
                cond = (df_it_t0v1['t'] == i)
                d = df_it_t0v1[cond].copy()
                n = d['id'].nunique()  # number of unique IDs in this tvsubset-time
                # vax
                d['v'] = np.random.binomial(n=1, p=p_v, size=n)  # whether someone decides to get vaxed on day t
                d.loc[d['v'] == 0, 'v'] = np.nan  # for sorting later
                # piece together
                if k == 1:
                    d_t0v1_v = d.copy()
                else:
                    d_t0v1_v = pd.concat([d_t0v1_v, d], axis=0)
                k += 1
            # vax status and timing v and tv: making v = 1 persistent
            d_t0v1_v = d_t0v1_v.sort_values(by=['id', 'v', 't'], ascending=[True, False, True]).reset_index(
                drop=True)  # sort by ID, vax, time
            d = d_t0v1_v[d_t0v1_v['v'] == 1].reset_index(drop=True)  # keep only vaccinated ones
            d = d.groupby(['id', 'v'])['t'].min().reset_index(drop=False).rename(columns={'t': 'tv'})
            d_t0v1_v = d_t0v1_v.merge(d[['id', 'tv']], on=['id'], how='left')
            cond1 = (d_t0v1_v['t'] >= d_t0v1_v['tv'])
            cond2 = (d_t0v1_v['t'] < d_t0v1_v['tv'])
            cond3 = (d_t0v1_v['tv'].isna())
            d_t0v1_v.loc[cond1, 'v'] = 1
            d_t0v1_v.loc[cond2, 'v'] = 0
            d_t0v1_v.loc[cond3, 'v'] = 0
            d_t0v1_v['tv'] = d_t0v1_v['tv'].fillna(0)  # 0 vax timing for those who were not vaccinated by T
            # simulation: others
            id = list(d['id'].unique())
            k = 1
            for i in tqdm(range(1, T + 1)):
                cond = (d_t0v1_v['t'] == i)  # start from the one with only vax
                d = d_t0v1_v[cond].copy()
                n = d['id'].nunique()  # number of unique IDs in this tvsubset-time
                # test
                d['tau'] = 0  # all never-testers
                # infection
                for j in id:  # row operation: unique VEs by ID
                    if (d.loc[d['id'] == j, 'v'].reset_index(drop=True)[0] == 1):
                        ve = d.loc[((d['v'] == 1) & (d['id'] == j)), 've'].reset_index(drop=True)[0]
                    else:
                        ve = 0
                    # print(str(ve)) # for checking
                    d.loc[((d['v'] == 1) & (d['id'] == j)), 'y'] = \
                        np.random.binomial(n=1, p=((1 - ve) * alpha_b),
                                           size=1)  # kalpha doesn't kick in since vax-willing
                if d['id'].count() > 0:
                    d.loc[d['y'] == 0, 'y'] = np.nan  # for sorting later
                elif d['id'].count() == 0:
                    pass  # avoid error due to empty data frame for the case of theta_tau = 1
                # observed infection
                d['yhat'] = 0  # never-testers
                # piece together
                if k == 1:
                    d_t0v1 = d.copy()
                else:
                    d_t0v1 = pd.concat([d_t0v1, d], axis=0)
                k += 1
            # true infections y: keeping first infection
            if d_t0v1['id'].count() > 0:
                d_t0v1 = d_t0v1.sort_values(by=['id', 'y', 't'], ascending=[True, False, True]).reset_index(
                    drop=True)  # sort by ID, infection, time
                d = d_t0v1.groupby(['id', 'y'])['t'].min().reset_index(drop=False).rename(columns={'t': 'ty'})
                d = d[d['y'] == 1].reset_index(drop=True)  # keep only infected ones
                d_t0v1 = d_t0v1.merge(d[['id', 'ty']], on=['id'], how='left')
                cond = (d_t0v1['t'] == d_t0v1['ty'])
                d_t0v1.loc[cond, 'y'] = 1
                d_t0v1.loc[~cond, 'y'] = 0  # vectorised fillna
                d_t0v1['ty'] = d_t0v1['ty'].fillna(0)  # 0 date for those who were not infected by T
                # observed infections timing tyhat
                d_t0v1['tyhat'] = 0  # all never testers
                # final subset
                d_t0v1 = d_t0v1.sort_values(by=['id', 't'], ascending=[True, True])  # sort by ID and time
            elif d_t0v1['id'].count() == 0:  # avoid errors due to empty df when theta_tau = 1
                d = pd.DataFrame(columns=['id', 'ty', 'tyhat', 'y', 'yhat'])
                d_t0v1 = d_t0v1.merge(d[['id', 'ty']], on=['id'], how='left')
                d_t0v1 = d_t0v1.reset_index(drop=True)

            # quadrant 4: testers, vaxers
            # simulation: vax first
            k = 1
            for i in tqdm(range(1, T + 1)):
                cond = (df_it_t1v1['t'] == i)
                d = df_it_t1v1[cond].copy()
                n = d['id'].nunique()  # number of unique IDs in this tvsubset-time
                # vax
                d['v'] = np.random.binomial(n=1, p=p_v, size=n)  # whether someone decides to get vaxed on day t
                d.loc[d['v'] == 0, 'v'] = np.nan  # for sorting later
                # piece together
                if k == 1:
                    d_t1v1_v = d.copy()
                else:
                    d_t1v1_v = pd.concat([d_t1v1_v, d], axis=0)
                k += 1
            # vax status v: making v = 1 persistent
            d_t1v1_v = d_t1v1_v.sort_values(by=['id', 'v', 't'], ascending=[True, False, True]).reset_index(
                drop=True)  # sort by ID, vax, time
            d = d_t1v1_v[d_t1v1_v['v'] == 1].reset_index(drop=True)  # keep only vaccinated ones
            d = d.groupby(['id', 'v'])['t'].min().reset_index(drop=False).rename(columns={'t': 'tv'})
            d_t1v1_v = d_t1v1_v.merge(d[['id', 'tv']], on=['id'], how='left')
            cond1 = (d_t1v1_v['t'] >= d_t1v1_v['tv'])
            cond2 = (d_t1v1_v['t'] < d_t1v1_v['tv'])
            cond3 = (d_t1v1_v['tv'].isna())
            d_t1v1_v.loc[cond1, 'v'] = 1
            d_t1v1_v.loc[cond2, 'v'] = 0
            d_t1v1_v.loc[cond3, 'v'] = 0
            d_t1v1_v['tv'] = d_t1v1_v['tv'].fillna(0)  # 0 vax timing for those who were not vaccinated by T
            # simulation: others
            id = list(d['id'].unique())
            k = 1
            for i in tqdm(range(1, T + 1)):
                cond = (d_t1v1_v['t'] == i)  # start from the one with only vax
                d = d_t1v1_v[cond].copy()
                n = d['id'].nunique()  # number of unique IDs in this tvsubset-time
                # test (vax)
                d_v = d[d['v'] == 1].reset_index(drop=True)
                n_v = d_v['id'].nunique()
                d_v['tau'] = np.random.binomial(n=1, p=p_tau, size=n_v)
                # test (unvax)
                d_uv = d[d['v'] == 0].reset_index(drop=True)
                n_uv = d_uv['id'].nunique()
                d_uv['tau'] = np.random.binomial(n=1, p=p_tau * ktau,
                                                 size=n_uv)  # if unvaccinated, then scale testing prob by ktau
                # test (all)
                d = pd.concat([d_v, d_uv], axis=0)  # back to OG
                d = d.reset_index(drop=True)
                if d['id'].nunique() == n:
                    pass  # row(d_pre) = row(d_v) + row(d_uv) must hold
                else:
                    raise ArithmeticError  # row(d_pre) = row(d_v) + row(d_uv) must hold
                # infection
                for j in id:  # row operation: unique VEs by ID
                    if (d.loc[d['id'] == j, 'v'].reset_index(drop=True)[0] == 1):
                        ve = d.loc[((d['v'] == 1) & (d['id'] == j)), 've'].reset_index(drop=True)[0]
                    else:
                        ve = 0
                    # print(str(ve)) # for checking
                    d.loc[((d['v'] == 1) & (d['id'] == j)), 'y'] = \
                        np.random.binomial(n=1, p=((1 - ve) * alpha_b),
                                           size=1)  # kalpha doesn't kick in since vax-willing
                d.loc[d['y'] == 0, 'y'] = np.nan  # for sorting later
                # observed infection
                d.loc[d['tau'] == 1, 'yhat'] = d['y']
                d.loc[d['tau'] == 0, 'yhat'] = 0
                # piece together
                if k == 1:
                    d_t1v1 = d.copy()
                else:
                    d_t1v1 = pd.concat([d_t1v1, d], axis=0)
                k += 1
            # true infections y: keeping first infection
            d_t1v1 = d_t1v1.sort_values(by=['id', 'y', 't'], ascending=[True, False, True]).reset_index(
                drop=True)  # sort by ID, infection, time
            d = d_t1v1.groupby(['id', 'y'])['t'].min().reset_index(drop=False).rename(columns={'t': 'ty'})
            d = d[d['y'] == 1].reset_index(drop=True)  # keep only infected ones
            d_t1v1 = d_t1v1.merge(d[['id', 'ty']], on=['id'], how='left')
            cond = (d_t1v1['t'] == d_t1v1['ty'])
            d_t1v1.loc[cond, 'y'] = 1
            d_t1v1.loc[~cond, 'y'] = 0  # vectorised fillna
            d_t1v1['ty'] = d_t1v1['ty'].fillna(0)  # 0 date for those who were not infected by T
            # observed infections yhat: resolving clash with true infections
            cond = ((d_t1v1['y'] == 0) & (d_t1v1['yhat'] == 1))  # clash: if true state is null, but not observed state
            d_t1v1.loc[cond, 'yhat'] = 0  # cannot observe infection when true state is null
            d_t1v1.loc[d_t1v1['yhat'].isna(), 'yhat'] = 0
            # observed infections timing tyhat
            d_t1v1 = d_t1v1.sort_values(by=['id', 'yhat', 't'], ascending=[True, False, True]).reset_index(
                drop=True)  # sort by ID, infection, time
            d = d_t1v1.groupby(['id', 'yhat'])['t'].min().reset_index(drop=False).rename(columns={'t': 'tyhat'})
            d = d[d['yhat'] == 1].reset_index(drop=True)  # keep only tested positive ones
            d_t1v1 = d_t1v1.merge(d[['id', 'tyhat']], on=['id'], how='left')
            d_t1v1.loc[d_t1v1['tyhat'].isna(), 'tyhat'] = 0  # those who stayed undetected till end of time
            # final subset
            d_t1v1 = d_t1v1.sort_values(by=['id', 't'], ascending=[True, True])  # sort by ID and time

            # Piece everything together
            df = pd.concat([d_t0v0, d_t0v1, d_t1v0, d_t1v1], axis=0)  # main data frame
            df = df.astype('int')

            # View aggregates
            df_agg = df.groupby(['t'])['v', 'y', 'yhat'].sum().reset_index()
            for i in ['y', 'yhat']:
                df_agg[i] = df_agg[i].cumsum()

            # Clear memory
            del d_t0v0
            del d_t0v1
            del d_t1v0
            del d_t1v1
            del d_t0v1_v
            del d_t1v1_v
            del d
            del df_it_t0v0
            del df_it_t0v1
            del df_it_t1v0
            del df_it_t1v1
            gc.collect()

            # III --- Simulate study designs
            # Cohort: true infections (ct)
            # for true infected people, keep the day of true infection
            cond_y = (df['ty'] > 0)
            cond_t = (df['t'] == df['ty'])
            cond_end = (df['t'] == T)
            d_y = df[cond_y & cond_t]
            # for true uninfected, keep the last day
            d_n = df[~cond_y & cond_end]
            # merge
            df_ct = pd.concat([d_y, d_n], axis=0)
            del d_y
            del d_n
            gc.collect()
            df_ct = df_ct.sort_values(by=['id', 't'], ascending=[True, True])

            # Cohort (immortal time correction): true infections (cts)
            # for true infected people, keep the day of true infection
            cond_y = (df['ty'] > 0)
            cond_t = (df['t'] == df['ty'])
            cond_end = (df['t'] == T)
            d_y = df[cond_y & cond_t]
            d_y.loc[(d_y['ty'] >= d_y['tv']), 'ttevent'] = d_y['ty'] - d_y['tv'] + 1  # ty-tv for vax; ty-0 for unvax
            d_y = d_y.reset_index(drop=True)  # deals with the duplicate axis error
            d_y.loc[(d_y['tv'] == 0), 'ttevent'] = d_y['ty']  # if unvaccinated, time-to-event = ty - 0
            # for true uninfected, keep the last day
            d_n = df[~cond_y & cond_end].reset_index(drop=True)  # deals with the duplicate axis error
            d_n.loc[(d_n['tv'] > 0), 'ttevent'] = T - d_n[
                'tv'] + 1  # if vaccinated by day T, then time-at-risk starts from vax date
            d_n.loc[(d_n['tv'] == 0), 'ttevent'] = T  # if unvaccinated by day T, then time-at-risk starts from 0
            # for infected while vaccinated, generate pre-vax (immortal time)
            cond_imt_yv = ((df['ty'] > df['tv']) & (df['t'] == (df['tv'] - 1)))  # take t-1 from vax date
            d_imt_yv = df[cond_imt_yv]
            d_imt_yv['ttevent'] = d_imt_yv['tv'] - 1  # should be the as d_imt_yv['t']
            # for uninfected but vaccinated eventually, generate pre-vax (immortal time)
            cond_imt_nyv = (~cond_y & (df['tv'] > 0) & (df['t'] == (df['tv'] - 1)))
            d_imt_nyv = df[cond_imt_nyv]
            d_imt_nyv['ttevent'] = d_imt_nyv['tv'] - 1  # should be the as d_imt_yv['t']
            # merge
            df_cts = pd.concat([d_y, d_n, d_imt_yv, d_imt_nyv], axis=0)
            del d_y
            del d_n
            del d_imt_yv
            del d_imt_nyv
            gc.collect()
            df_cts = df_cts.sort_values(by=['id', 't'], ascending=[True, True])

            # Cohort (aggregated): true infections (cta)
            # Generate count variable
            df_cta = df_ct.copy()  # start from granular cohort data
            df_cta['count'] = 1
            # Split into ty-indexed and tv-indexed data frames
            d_y = df_cta[df_cta['y'] == 1]
            d_y = d_y.groupby(['ty', 'v'])['count'].agg('count').reset_index().rename(
                columns={'count': 'y', 'ty': 't'})  # by vax status
            d_y.loc[d_y['y'].isna(), 'y'] = 0  # vectorised fillna
            d_v = df_cta[df_cta['v'] == 1]
            d_v = d_v.groupby(['tv', 'v'])['count'].agg('count').reset_index().rename(
                columns={'count': 'n_v', 'tv': 't'})  # daily vax count
            checkt = d_v['t'].max().astype('int')
            if checkt == T:  # match
                pass
            elif checkt < T:  # super fast rollout
                appendt = T - checkt
                addt = pd.DataFrame(list(range(checkt + 1, checkt + appendt + 1)), columns=['t'])
                d_v = pd.concat([d_v, addt], axis=0).reset_index(drop=True)
            d_v = d_v.sort_values(by=['t'], ascending=[True]).reset_index(drop=True)
            d_v.loc[d_v['v'].isna(), 'v'] = 1  # missing
            d_v.loc[d_v['n_v'].isna(), 'n_v'] = 0  # missing
            d_v['N_v'] = d_v['n_v'].cumsum()
            d_v['PD_v'] = d_v['N_v'].cumsum()
            # Generate vprime + concat with d_v
            d_vprime = d_v.copy()
            d_vprime['v'] = 0
            d_vprime['N_v'] = N - d_vprime['N_v']
            d_vprime['PD_v'] = d_vprime['N_v'].cumsum()
            d_v = pd.concat([d_v, d_vprime], axis=0)
            d_v = d_v.reset_index(drop=True)
            del d_v['n_v']
            # Merge
            df_cta = d_y.merge(d_v, on=['t', 'v'], how='outer')
            df_cta = df_cta.sort_values(by=['t', 'v'], ascending=[True, True]).reset_index(drop=True)
            del d_y
            del d_v
            del d_vprime
            gc.collect()
            # Missing
            df_cta.loc[df_cta['y'].isna(), 'y'] = 0
            # Scale y against PD and N
            df_cta['y_N'] = df_cta['y'] / df_cta['N_v']  # same as regressing levels on levels and offset
            df_cta['y_PD'] = df_cta['y'] / df_cta['PD_v']  # same as regressing levels on levels and offset

            # Cohort: observed (co)
            # for observed infected people, keep the day of observed infection
            cond_y = (df['tyhat'] > 0)
            cond_t = (df['t'] == df['tyhat'])
            cond_end = (df['t'] == T)
            d_y = df[cond_y & cond_t]
            # for observed uninfected, keep the last day
            d_n = df[~cond_y & cond_end]
            # merge
            df_co = pd.concat([d_y, d_n], axis=0)
            del d_y
            del d_n
            gc.collect()
            df_co = df_co.sort_values(by=['id', 't'], ascending=[True, True])

            # Cohort (immortal time correction): observed infections (cos)
            # for true infected people, keep the day of true infection
            cond_y = (df['tyhat'] > 0)
            cond_t = (df['t'] == df['tyhat'])
            cond_end = (df['t'] == T)
            d_y = df[cond_y & cond_t]
            d_y.loc[(d_y['tyhat'] >= d_y['tv']), 'ttevent'] = d_y['tyhat'] - d_y[
                'tv'] + 1  # if vaccinated, time-to-event = tyhat - tv; tyhat < tv is not possible
            d_y = d_y.reset_index(drop=True)  # deals with the duplicate axis error
            d_y.loc[(d_y['tv'] == 0), 'ttevent'] = d_y['tyhat']  # if unvaccinated, time-to-event = tyhat - 0
            # for true uninfected, keep the last day
            d_n = df[~cond_y & cond_end].reset_index(drop=True)  # deals with the duplicate axis error
            d_n.loc[(d_n['tv'] > 0), 'ttevent'] = T - d_n[
                'tv'] + 1  # if vaccinated by day T, then time-at-risk starts from vax date
            d_n.loc[(d_n['tv'] == 0), 'ttevent'] = T  # if unvaccinated by day T, then time-at-risk starts from 0
            # for infected while vaccinated, generate pre-vax (immortal time)
            cond_imt_yv = ((df['tyhat'] > df['tv']) & (df['t'] == (df['tv'] - 1)))  # take t-1 from vax date
            d_imt_yv = df[cond_imt_yv]
            d_imt_yv['ttevent'] = d_imt_yv['tv'] - 1  # should be the as d_imt_yv['t']
            # for uninfected but vaccinated eventually, generate pre-vax (immortal time)
            cond_imt_nyv = (~cond_y & (df['tv'] > 0) & (df['t'] == (df['tv'] - 1)))
            d_imt_nyv = df[cond_imt_nyv]
            d_imt_nyv['ttevent'] = d_imt_nyv['tv'] - 1  # should be the as d_imt_yv['t']
            # merge
            df_cos = pd.concat([d_y, d_n, d_imt_yv, d_imt_nyv], axis=0)
            del d_y
            del d_n
            del d_imt_yv
            del d_imt_nyv
            gc.collect()
            df_cos = df_cos.sort_values(by=['id', 't'], ascending=[True, True])

            # Cohort (aggregated): observed infections (coa)
            # Generate count variable
            df_coa = df_ct.copy()  # start from granular cohort data
            df_coa['count'] = 1
            # Split into tyhat-indexed and tv-indexed data frames
            d_y = df_coa[df_coa['yhat'] == 1]
            d_y = d_y.groupby(['tyhat', 'v'])['count'].agg('count').reset_index().rename(
                columns={'count': 'yhat', 'tyhat': 't'})  # by vax status
            d_y.loc[d_y['yhat'].isna(), 'yhat'] = 0  # vectorised fillna
            d_v = df_coa[df_coa['v'] == 1]
            d_v = d_v.groupby(['tv', 'v'])['count'].agg('count').reset_index().rename(
                columns={'count': 'n_v', 'tv': 't'})  # daily vax count
            checkt = d_v['t'].max().astype('int')
            if checkt == T:  # match
                pass
            elif checkt < T:  # super fast rollout
                appendt = T - checkt
                addt = pd.DataFrame(list(range(checkt + 1, checkt + appendt + 1)),
                                    columns=['t'])
                d_v = pd.concat([d_v, addt], axis=0).reset_index(drop=True)
            d_v = d_v.sort_values(by=['t'], ascending=[True]).reset_index(drop=True)
            d_v.loc[d_v['v'].isna(), 'v'] = 1  # missing
            d_v.loc[d_v['n_v'].isna(), 'n_v'] = 0  # missing
            d_v['N_v'] = d_v['n_v'].cumsum()
            d_v['PD_v'] = d_v['N_v'].cumsum()
            # Generate vprime + concat with d_v
            d_vprime = d_v.copy()
            d_vprime['v'] = 0
            d_vprime['N_v'] = N - d_vprime['N_v']
            d_vprime['PD_v'] = d_vprime['N_v'].cumsum()
            d_v = pd.concat([d_v, d_vprime], axis=0)
            d_v = d_v.reset_index(drop=True)
            del d_v['n_v']
            # Merge
            df_coa = d_y.merge(d_v, on=['t', 'v'], how='outer')
            df_coa = df_coa.sort_values(by=['t', 'v'], ascending=[True, True]).reset_index(drop=True)
            del d_y
            del d_v
            del d_vprime
            gc.collect()
            # Missing
            df_coa.loc[df_coa['yhat'].isna(), 'yhat'] = 0
            # Scale y against PD and N
            df_coa['yhat_N'] = df_coa['yhat'] / df_coa['N_v']  # same as regressing levels on levels and offset
            df_coa['yhat_PD'] = df_coa['yhat'] / df_coa['PD_v']  # same as regressing levels on levels and offset

            # TND: first pos, first neg
            df_tf = df.copy()
            # generate max vax status
            d = df_tf.groupby(['id'])['v'].max().reset_index(drop=False).rename(columns={'v': 'vmax'})
            df_tf = df_tf.merge(d[['id', 'vmax']], on='id', how='left')
            # for observed infected people, keep day of test
            cond_y = (df_tf['tyhat'] > 0)
            cond_t = (df_tf['t'] == df_tf['tyhat'])
            cond_tau = (df_tf['tau'] == 1)
            d_y = df_tf[cond_y & cond_t]
            # generate the day of first negative test for all observed uninfected + vax
            cond_v = (df_tf['vmax'] == 1)  # vaccinated people
            cond_tv = (df_tf['t'] >= df_tf['tv'])
            d = df_tf[
                ~cond_y & cond_v & cond_tv & cond_tau]  # keep only post-vax periods with tests taken if never observed infected
            d = d.groupby(['id'])['t'].min().reset_index(drop=False).rename(columns={'t': 'tfn_v'})
            df_tf = df_tf.merge(d[['id', 'tfn_v']], on='id', how='left')  # first neg timing if ever-vaxed
            # generate the day of first negative test for all observed uninfected + unvax
            d = df_tf[~cond_y & ~cond_v & cond_tau]  # keep everything for the unvax if never observed infected
            d = d.groupby(['id'])['t'].min().reset_index(drop=False).rename(columns={'t': 'tfn_u'})
            df_tf = df_tf.merge(d[['id', 'tfn_u']], on='id', how='left')
            # combine first negative test dates
            df_tf.loc[df_tf['tfn_v'].isna(), 'tfn'] = df_tf['tfn_u']
            df_tf.loc[df_tf['tfn_u'].isna(), 'tfn'] = df_tf['tfn_v']
            del df_tf['tfn_u']
            del df_tf['tfn_v']
            # for observed uninfected people, keep first neg test for unvax, but first neg test when v = 1 for the vaxed
            cond_tfn = (df_tf['t'] == df_tf['tfn'])
            d_n = df_tf[cond_tfn]
            # merge
            df_tf = pd.concat([d_y, d_n], axis=0)
            del d_y
            del d_n
            gc.collect()
            df_tf = df_tf.sort_values(by=['id', 't'], ascending=[True, True])
            df_tf.loc[df_tf['tfn'].isna(), 'tfn'] = 0  # filling missing (for those who tested positive)

            # TND: first pos, multiple neg
            df_tm = df.copy()
            # for observed infected people, keep day of test (same as first pos, first neg version)
            cond_y = (df_tm['tyhat'] > 0)
            cond_t = (df_tm['t'] == df_tm['tyhat'])
            cond_tau = (df_tm['tau'] == 1)
            d_y = df_tm[cond_y & cond_t]
            # for observed infected people, keep all negative tests before the first positive test
            cond_tmn = (df_tm['t'] < df_tm['tyhat'])
            d_y_pre = df_tm[cond_y & cond_tmn & cond_tau]
            # for observed uninfected people, keep all negative test
            d_n = df_tm[~cond_y & cond_tau]
            # merge
            df_tm = pd.concat([d_y, d_y_pre, d_n], axis=0)
            del d_y
            del d_y_pre
            del d_n
            gc.collect()
            df_tm = df_tm.sort_values(by=['id', 't'], ascending=[True, True])

            # IV --- Estimation
            # Cohort: true
            VE, result, mod = logitVE(equation='y~v', data=df_ct)
            VE['Design'] = 'Cohort: True y'
            VE_consol = pd.concat([VE_consol, VE], axis=0)

            # Cohort (immortal time correction): true
            VE, result, mod = coxVE(data=df_cts, status='y')
            VE['Design'] = 'Cohort (immortal time correction): True y'
            VE_consol = pd.concat([VE_consol, VE], axis=0)

            # Cohort (aggregate; PD): true
            VE, result, mod = nbVE(equation='y~v', data=df_cta, offset='PD_v')
            VE['Design'] = 'Cohort (aggregate; PD): True y'
            VE_consol = pd.concat([VE_consol, VE], axis=0)

            # Cohort (aggregate; N): true
            VE, result, mod = nbVE(equation='y~v', data=df_cta, offset='N_v')
            VE['Design'] = 'Cohort (aggregate; N): True y'
            VE_consol = pd.concat([VE_consol, VE], axis=0)

            # Cohort: observed
            VE, result, mod = logitVE(data=df_co)
            VE['Design'] = 'Cohort: Observed y'
            VE_consol = pd.concat([VE_consol, VE], axis=0)

            # Cohort (immortal time correction): observed
            VE, result, mod = coxVE(data=df_cts)
            VE['Design'] = 'Cohort (immortal time correction): Observed y'
            VE_consol = pd.concat([VE_consol, VE], axis=0)

            # Cohort (aggregate; PD): observed
            VE, result, mod = nbVE(equation='yhat~v', data=df_coa, offset='PD_v')
            VE['Design'] = 'Cohort (aggregate; PD): Observed y'
            VE_consol = pd.concat([VE_consol, VE], axis=0)

            # Cohort (aggregate; N): observed
            VE, result, mod = nbVE(equation='yhat~v', data=df_coa, offset='N_v')
            VE['Design'] = 'Cohort (aggregate; N): Observed y'
            VE_consol = pd.concat([VE_consol, VE], axis=0)

            # TND: first pos, first neg
            VE, result, mod = logitVE(data=df_tf)
            VE['Design'] = 'TND: First Pos, First Neg'
            VE_consol = pd.concat([VE_consol, VE], axis=0)

            # TND: first pos, multiple neg
            VE, result, mod = logitVE(data=df_tm)
            VE['Design'] = 'TND: First Pos, Multiple Neg'
            VE_consol = pd.concat([VE_consol, VE], axis=0)

            # Label parameters
            VE_consol['N'] = N
            VE_consol['T'] = T
            VE_consol['Theta_v'] = Theta_v
            VE_consol['p_v'] = p_v
            VE_consol['Theta_tau'] = Theta_tau
            VE_consol['p_tau'] = p_tau
            VE_consol['ktau'] = ktau
            VE_consol['alpha_b'] = alpha_b
            VE_consol['kalpha'] = kalpha
            VE_consol['mew_ve'] = mew_ve

            # Calculate bias
            VE_consol['Bias'] = VE_consol['VE'] - 100 * mew_ve
            return VE_consol
    except:
        # Fails
        print('Error at \n\n' +
              'N = ' + str(N) + '; \n' +
              'T = ' + str(T) + '; \n' +
              'Theta_v = ' + str(Theta_v) + '; \n' +
              'p_v = ' + str(p_v) + '; \n' +
              'Theta_tau = ' + str(Theta_tau) + '; \n' +
              'p_tau = ' + str(p_tau) + '; \n' +
              'ktau = ' + str(ktau) + '; \n' +
              'alpha_b = ' + str(alpha_b) + '; \n' +
              'kalpha = ' + str(kalpha) + '; \n' +
              'mew_ve = ' + str(mew_ve) + '\n\n' +
              traceback.format_exc())

### II --- Simulate data set + study designs
__file__ = 'workaround.py'
pool = mp.Pool() # no arguments = fastest relative to nested loops in MWE
output = pool.map(snsdcomeback,paramslist)
VE_consol_all = pd.concat(output)
print('0_VEMethod_Sim1b_Parallel_CloudVersion_ReSeed_NoCI\n\n' +
      'SIMULATION ONLY: Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds')
print('\n----- SIMULATION ONLY: Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')

### III.A --- Export raw ReSeed file
VE_consol_all = VE_consol_all.reset_index(drop=True) # generates unique indices
VE_consol_all.to_parquet('VEMethod_Sim1b_Parallel_CloudVersion_ReSeed_NoCI.parquet', compression='brotli', index=False)

### III.B --- Merge ReSeed frame with original frame
df = pd.concat([df_original, VE_consol_all], axis=0)
df = df.reset_index(drop=True) # regenerates unique indices
for params in paramslist:
    N = params[0]
    T = params[1]
    Theta_v = params[2]
    p_v = params[3]
    Theta_tau = params[4]
    p_tau = params[5]
    ktau = params[6]
    alpha_b = params[7]
    kalpha = params[8]
    mew_ve = params[9]
    d = df[(df['N'] == N) &
           (df['T'] == T) &
           (df['Theta_v'] == Theta_v) &
           (df['p_v'] == p_v) &
           (df['Theta_tau'] == Theta_tau) &
           (df['p_tau'] == p_tau) &
           (df['ktau'] == ktau) &
           (df['alpha_b'] == alpha_b) &
           (df['kalpha'] == kalpha) &
           (df['mew_ve'] == mew_ve)]
    if len(d) == 0: raise NotImplementedError
    elif len(d) > 0: print('NOW THERE, GOOD TO GO')
    elif len(d) < 0: raise NotImplementedError
df.to_parquet('Output/VEMethod_Sim1b_Parallel_CloudVersion_NoCI_FIN.parquet', compression='brotli', index=False)

### End
print('0_VEMethod_Sim1b_Parallel_CloudVersion_ReSeed_NoCI: COMPLETED')
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')