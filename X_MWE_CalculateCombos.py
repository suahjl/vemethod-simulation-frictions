#### Calculate number of combos & estimated time to be run in simulation

import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools

### Start
list_N = [1000] # how many people (keep fixed)
list_T = [20, 30, 40, 50, 60] # how many days
list_Theta_v = [0.3, 0.4, 0.5, 0.6, 0.7] # vaxers share (keep fixed)
list_p_v = [0.15, 0.3, 0.4, 0.5, 0.6, 0.85, 1] # probability of getting vaccinated on day x
list_Theta_tau = [0.5, 0.65, 0.75, 0.85, 1] # testers share
list_p_tau = [0.5, 0.65, 0.75, 0.85, 1] # testing probability if theta_tau = 1
list_ktau = [0.5, 0.75, 0.9, 1, 1.1, 1.25, 1.5] # relative propensity to test between v = 1 and v = 0 (captures selection in testing)
list_alpha_b = [0.025] # baseline daily infection risk (keep fixed)
list_kalpha = [0.5, 0.75, 0.9, 1, 1.1, 1.25, 1.5]  # asymmetry in baseline infection risk between vax-willing and vax-unwilling
list_mew_ve = [0.5, 0.6, 0.7, 0.8, 0.9] # peak of VE distribution (beta distribution) (if 0.5, then mode = mean)

# list_N = [1000] # how many people (keep fixed)
# list_T = [20, 30, 40, 50, 60] # how many days
# list_Theta_v = [0.3, 0.4, 0.5, 0.6, 0.7] # vaxers share (keep fixed)
# list_p_v = [0.15, 0.3, 0.5, 0.85, 1] # probability of getting vaccinated on day x
# list_Theta_tau = [0.5, 0.65, 0.75, 0.85, 1] # testers share
# list_p_tau = [0.5, 0.65, 0.75, 0.85, 1] # testing probability if theta_tau = 1
# list_ktau = [0.5, 0.75, 1, 1.25, 1.5] # relative propensity to test between v = 1 and v = 0 (captures selection in testing)
# list_alpha_b = [0.015, 0.02, 0.025] # baseline daily infection risk (keep fixed)
# list_kalpha = [0.5, 0.75, 1, 1.25, 1.5] # asymmetry in baseline infection risk between vax-willing and vax-unwilling
# list_mew_ve = [0.5, 0.6, 0.7, 0.8, 0.9] # peak of VE distribution (beta distribution) (if 0.5, then mode = mean)

# list_N = [2000] # how many people (keep fixed)
# list_T = [20, 30, 40, 50, 60] # how many days
# list_Theta_v = [0.3, 0.4, 0.5, 0.6, 0.7] # vaxers share (keep fixed)
# list_p_v = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1] # probability of getting vaccinated on day x
# list_Theta_tau = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1] # testers share
# list_p_tau = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1] # testing probability if theta_tau = 1
# list_ktau = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5] # relative propensity to test between v = 1 and v = 0 (captures selection in testing)
# list_alpha_b = [0.01, 0.015, 0.02, 0.025] # baseline daily infection risk (keep fixed)
# list_kalpha = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5] # asymmetry in baseline infection risk between vax-willing and vax-unwilling
# list_mew_ve = [0.5, 0.6, 0.7, 0.8, 0.9] # peak of VE distribution (beta distribution) (if 0.5, then mode = mean)

# list_N = [2000] # how many people (keep fixed)
# list_T = [20, 30, 40, 50, 60] # how many days
# list_Theta_v = [0.3, 0.7] # vaxers share (keep fixed)
# list_p_v = [0.3, 1] # probability of getting vaccinated on day x
# list_Theta_tau = [0.5] # testers share
# list_p_tau = [0.5] # testing probability if theta_tau = 1
# list_ktau = [0.5, 1, 1.5] # relative propensity to test between v = 1 and v = 0 (captures selection in testing)
# list_alpha_b = [0.025] # baseline daily infection risk (keep fixed)
# list_kalpha = [0.5, 1, 1.5] # asymmetry in baseline infection risk between vax-willing and vax-unwilling
# list_mew_ve = [0.5, 0.7, 0.9] # peak of VE distribution (if 0.5, then mode = mean) (keep fixed)

paramslist = list(itertools.product(list_N, list_T, list_Theta_v, list_p_v, list_Theta_tau, list_p_tau, list_ktau, list_alpha_b, list_kalpha, list_mew_ve))
sec_per_combo = 0.462 # 0.4 (N=1000); 0.73 (N=2000)
N_combo = len(paramslist)
N_combo_illegal = 0
for params in tqdm(paramslist):
    draw_p_tau = params[5]
    draw_ktau = params[6]
    if (draw_p_tau * draw_ktau > 1):
        N_combo_illegal += 1
total_sec = sec_per_combo * (N_combo - N_combo_illegal)
total_min = total_sec / 60
total_hr = total_min / 60
total_day = total_hr / 24
cost_hr = 6.4512 # USD per hour
total_cost = cost_hr * total_hr

print('Total combos: ' + str(N_combo) + '\n' +
      'Total LEGAL combos: ' + str(N_combo - N_combo_illegal) + '\n' +
      'Seconds per combo: ' + str(sec_per_combo) + '\n' +
      'Total seconds: ' + str('{:.2f}'.format(total_sec)) + '\n' +
      'Total minutes: ' + str('{:.2f}'.format(total_min)) + '\n' +
      'Total hours: ' + str('{:.2f}'.format(total_hr)) + '\n' +
      'Total days: ' + str('{:.2f}'.format(total_day)) + '\n' +
      'Quoted hourly cost (USD/hr): ' + str('{:.2f}'.format(cost_hr)) + '\n' +
      'Estimated total cost (USD): ' + str('{:.2f}'.format(total_cost)))

### End