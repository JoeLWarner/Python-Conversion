# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:10:35 2025

@author: jw4e24
"""

from eigenvalue_equation import nslabmodes_f
from silica_my import silica_n
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

l = 780e-9
k0 = 2 * np.pi / l
d = 2e-6
TE = False

nm = 1
dn_wg = 5e-3
dn = dn_wg / 10000
dlist  = [d, d, d]

def find_root(beta, birefringence):
    return nslabmodes_f(beta, k0, n, dlist, TE)

dn_wg_values = np.linspace(0.001, 0.01, 10)
neffTM = []
beta_solutions = []
for value in dn_wg_values:
    n_core = silica_n(l) + value
    n_cladding = silica_n(l)
    n = [n_cladding, n_core, n_cladding]
    beta_range = np.linspace(k0 * (np.min(n) + dn), k0 * (np.max(n) - dn), 1000, dtype=np.float64)
    B0vals = nslabmodes_f(beta_range, k0, n, dlist, TE=False)
    zero_crossings1 = np.where(np.diff(np.signbit(B0vals.real)))[0]
    for idx in zero_crossings1:
        result = root_scalar(
            find_root,
            args=(0,),  
            bracket=[beta_range[idx], beta_range[idx + 1]],
            method='brentq',
            rtol=1e-12
        )
        if result.converged:
            beta_solutions.append(result.root)
for beta in beta_solutions:
    #print("TM", beta)
    neffTM.append(beta/k0)
TE = True
neffTEtemp = []
beta_solutionsTE = []
birefringence = 2e-4
for value in dn_wg_values:
    n_core = silica_n(l) + value
    n_cladding = silica_n(l)
    n = [n_cladding, n_core, n_cladding]
    beta_range = np.linspace(k0 * (np.min(n) + dn), k0 * (np.max(n) - dn), 1000, dtype=np.float64)
    B0vals = nslabmodes_f(beta_range, k0, n, dlist, TE)
    zero_crossings2 = np.where(np.diff(np.signbit(B0vals.real)))[0]
    for idx in zero_crossings2:
        result = root_scalar(
            find_root,
            args=(0,),  
            bracket=[beta_range[idx], beta_range[idx + 1]],
            method='brentq',
            rtol=1e-12  
        )
        if result.converged:
            beta_solutionsTE.append(result.root)
for index, beta in enumerate(beta_solutionsTE):
    neffTEtemp.append(beta/k0)
neffTE = [val + birefringence for val in neffTEtemp]

neff_diffs = np.subtract(neffTE, neffTM) # this line is just here to check the neff diffs in the variable explorer
CL = ((np.pi/2) / k0) / np.subtract(neffTE, neffTM)

plt.plot(neff_diffs, CL, 'o-', markersize='3')
plt.xlabel("Effective Refractive Index Difference")
plt.ylabel("Coherence Length L (m)")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.grid()
plt.show()

fig, ax1 = plt.subplots()

ax1.set_xlabel('Index Difference')
ax1.plot(dn_wg_values, neffTM, 'd-', markersize='3', label='TM neff', color='blue')
ax1.plot(dn_wg_values, neffTE,'d-', markersize='3', label='TE neff', color='orange')
ax1.set_ylabel('Effective Refractive Indexes')
ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax1.grid()

ax2 = ax1.twinx() 
ax2.plot(dn_wg_values, neff_diffs, label='neff Differences', linestyle='dotted',color='green')
ax2.set_ylabel('neff Differences')
ax2.tick_params(axis='y')
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax2.grid()

fig.legend(loc=2)
fig.tight_layout()  
plt.show()


    
    
    
    
    
    
    
    
    