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

l = 780e-9
k0 = 2 * np.pi / l
d = 2e-6
TE = False

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

#########mode solution########
##############################
x = np.linspace(-d, 2*d, 1000)

#######Gaussian Fit#####################
########################################

def fit_gaussian(x, E):
    def gaussian(x, A, x0, sigma):
        return A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
    # Initial parameter estimates: peak amplitude, center, width
    p0 = [np.max(E), x[np.argmax(E)], np.std(x) / 2]
    popt, _ = curve_fit(gaussian, x, E, p0=p0)
    return popt  # Returns (A, x0, sigma)


ETE = calculate_mode_shape(x, beta, k0, n, [d], TE=True)
n_core = silica_n(l) + 0.005
n_cladding = silica_n(l)
birefringence = 0
n = [n_cladding, n_core+birefringence, n_cladding]
ETM = calculate_mode_shape(x, beta, k0, n, [d], TE=False)

A_fit1, x0_fit1, sigma_fit1 = fit_gaussian(x, ETE)
A_fit2, x0_fit2, sigma_fit2 = fit_gaussian(x, ETM)

plt.figure(figsize=(8, 6))
plt.plot(x * 1e6, ETE, label='Mode Field', color='b')
plt.plot(x * 1e6, A_fit1 * np.exp(-((x - x0_fit1) ** 2) / (2 * sigma_fit1 ** 2)), 'r--', label='Gaussian Fit')
plt.xlabel('Position (µm)')
plt.ylabel('Electric Field Amplitude')
plt.title('Mode Shape and Gaussian Fit, TE mode')
plt.legend()
plt.grid(True)

plt.figure(figsize=(8,6))
plt.plot(x * 1e6, ETM, label='Mode Field', color='b')
plt.plot(x * 1e6, A_fit2 * np.exp(-((x - x0_fit2) ** 2) / (2 * sigma_fit2 ** 2)), 'r--', label='Gaussian Fit')
plt.xlabel('Position (µm)')
plt.ylabel('Electric Field Amplitude')
plt.title('Mode Shape and Gaussian Fit, TM mode')
plt.legend()
plt.grid(True)
plt.show()

print(f"Gaussian Fit Parameters for TE:\nAmplitude: {A_fit1:.4f}, Center: {x0_fit1:.4e} m, Width (sigma): {sigma_fit1:.4e} m")
print(f"Gaussian Fit Parameters for TM:\nAmplitude: {A_fit2:.4f}, Center: {x0_fit2:.4e} m, Width (sigma): {sigma_fit2:.4e} m")


