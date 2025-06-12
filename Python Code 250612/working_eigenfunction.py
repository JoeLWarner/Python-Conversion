# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:45:37 2025

@author: jw4e24
"""

import numpy as np
from silica_my import silica_n
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from mode_solution import calculate_mode_shape
from eigenvalue_equation import nslabmodes_f
from scipy.optimize import curve_fit

l = 213e-9
k0 = 2 * np.pi / l
d = 2e-6
TE = True

nm = 1

n_core = silica_n(l) + 0.005
n_cladding = silica_n(l)
n = [n_cladding, n_core, n_cladding]

dn_wg = 5e-3
dn = dn_wg / 10000
beta_range = np.linspace(k0 * (np.min(n) + dn), k0 * (np.max(n) - dn), 1000, dtype=np.float64)

dlist  = [d, d, d]
B0_values = nslabmodes_f(beta_range, k0, n, dlist, TE)
    
    # Find zero crossings
zero_crossings = np.where(np.diff(np.signbit(B0_values.real)))[0]


def find_root(beta):
    return nslabmodes_f(beta, k0, n, dlist, TE)

beta_solutionsTE = []
for idx in zero_crossings:
    result = root_scalar(
        find_root,
        bracket=[beta_range[idx], beta_range[idx + 1]],
        method='brentq',
        rtol=1e-12  
    )
    if result.converged:
        beta_solutionsTE.append(result.root)

plt.figure(figsize=(10, 6))
plt.plot(beta_range, B0_values.real, 'b-', label='B0')
#plt.plot(beta_solutions, 
#         np.zeros_like(beta_solutions), 
#         'ro', 
#         label='Roots')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('β (inverse propagation constant)')
plt.ylabel('B0')
plt.title('Slab Waveguide Mode Solutions')
plt.grid(True)
plt.legend()
plt.show()

# Print solutions
print("\nFound solutions (effective indices):")
for beta in beta_solutionsTE:
    print(f"n_eff = {beta/k0:.6f}")
    print(f"Beta = {beta:.6f}")
    if TE == True:
        print("TE Mode")
    else:
        print("TM Mode")
        
maxb = [beta / k0 for beta in beta_solutionsTE]
print("fundamental mode is", max(maxb), "at wavelength", l)
#########mode solution########
##############################


#######Gaussian Fit#####################
########################################

def gaussian(x, A, x0, w_Gauss):
    return A * np.exp(-((x - x0) / w_Gauss) ** 2)

def fit_gaussian(x, E, d):
    p0 = [np.max(E), d / 2, d]  # Center at d/2, width starting at d
    popt, _ = curve_fit(gaussian, x, E, p0=p0)
    A_fit, x0_fit, w_Gauss_fit = popt  
    return A_fit, x0_fit, w_Gauss_fit

x = np.linspace(-d, 2*d, 1000)

ETE = calculate_mode_shape(x, beta, k0, n, [d], TE=True)
n_core = silica_n(l) + 0.005
n_cladding = silica_n(l)
n = [n_cladding, n_core, n_cladding]
ETM = calculate_mode_shape(x, beta, k0, n, [d], TE=False)

A_fit_TE, x0_fit_TE, w_Gauss_TE = fit_gaussian(x, ETE, d)
A_fit_TM, x0_fit_TM, w_Gauss_TM = fit_gaussian(x, ETM, d)

E_Gauss_TE = gaussian(x, A_fit_TE, x0_fit_TE, w_Gauss_TE)
E_Gauss_TM = gaussian(x, A_fit_TM, x0_fit_TM, w_Gauss_TM)

plt.figure(figsize=(8, 6))
plt.plot(x, ETE, label='Mode Field', color='b')
plt.plot(x, E_Gauss_TE, 'r--', label="Gaussian Fit (TE)")
plt.xlabel('Position (µm)')
plt.ylabel('Electric Field Amplitude')
plt.title('Mode Shape and Gaussian Fit, TE mode')
plt.legend()
plt.grid(True)

plt.figure(figsize=(8,6))
plt.plot(x, ETM, label='Mode Field', color='b')
plt.plot(x, E_Gauss_TM, 'r--', label="Gaussian Fit (TM)")
plt.xlabel('Position (µm)')
plt.ylabel('Electric Field Amplitude')
plt.title('Mode Shape and Gaussian Fit, TM mode')
plt.legend()
plt.grid(True)
plt.show()

print(f"Gaussian waist (TE): {w_Gauss_TE:.6e} m")
print(f"Gaussian waist (TM): {w_Gauss_TM:.6e} m")


