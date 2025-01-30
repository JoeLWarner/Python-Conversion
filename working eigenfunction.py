# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:45:37 2025

@author: jw4e24
"""

import numpy as np
from silica_my import silica_n
#from eigsolver import find_beta_roots
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from mode_solution import calculate_mode_shape
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

def nslabmodes_f(beta, k0, n, d, TE):
    """
    Calculate TE/TM modes for multilayer slab waveguide.
    
    Parameters:
    beta (array-like): Propagation constant values to evaluate
    k0 (float): Wave number in free space (2π/λ)
    n (array-like): Refractive indices of layers
    d (array-like): Thickness of layers (except for outer layers)
    TE (bool): True for TE modes, False for TM modes
    
    Returns:
    array: B0 values for eigenvalue equation
    """
    # Convert inputs to numpy arrays if they aren't already
    beta = np.asarray(beta, dtype=np.complex128)
    n = np.asarray(n, dtype=np.complex128)
    d = np.asarray(d, dtype=np.complex128)
    
    # Number of layers
    nn = len(n)
    
    # Initialize A0 and B0
    A0 = np.zeros_like(beta, dtype=np.complex128)
    B0 = np.ones_like(beta, dtype=np.complex128)
    
    # Loop through layers
    for in_ in range(nn-1):
        # Calculate transverse propagation constants
        kappa0 = np.sqrt(k0**2 * n[in_]**2 - beta**2)
        kappa1 = np.sqrt(k0**2 * n[in_+1]**2 - beta**2)
        
        # Calculate impedance ratio
        kk = kappa0/kappa1
        
        # Modify for TM modes
        if not TE:
            kk = kk * (n[in_+1]**2)/(n[in_]**2)
        
        # Calculate transfer matrix elements
        A1 = 0.5*(1 + kk)*A0 + 0.5*(1 - kk)*B0
        B1 = 0.5*(1 - kk)*A0 + 0.5*(1 + kk)*B0
        
        # Apply phase if not at last interface
        if in_ < nn-2:
            phase = 1j * kappa1 * d[in_]
            A1 = np.exp(phase) * A1
            B1 = np.exp(-phase) * B1
        
        # Update for next iteration
        A0 = A1
        B0 = B1
    
    return B0
dlist  = [d, d, d]
B0_values = nslabmodes_f(beta_range, k0, n, dlist, TE)
    
    # Find zero crossings
zero_crossings = np.where(np.diff(np.signbit(B0_values.real)))[0]


def find_root(beta):
    return nslabmodes_f(beta, k0, n, dlist, TE)

beta_solutions = []
for idx in zero_crossings:
    # Use root_scalar with bracket method (similar to fzero)
    result = root_scalar(
        find_root,
        bracket=[beta_range[idx], beta_range[idx + 1]],
        method='brentq',
        rtol=1e-12  # Similar to MATLAB's TolX
    )
    if result.converged:
        beta_solutions.append(result.root)

plt.figure(figsize=(10, 6))
plt.plot(beta_range, B0_values.real, 'b-', label='B0')
plt.plot(beta_solutions, 
         np.zeros_like(beta_solutions), 
         'ro', 
         label='Roots')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('β (inverse propagation constant)')
plt.ylabel('B0')
plt.title('Slab Waveguide Mode Solutions')
plt.grid(True)
plt.legend()
plt.show()

# Print solutions
print("\nFound solutions (effective indices):")
for beta in beta_solutions:
    print(f"n_eff = {beta/k0:.6f}")
    print(f"Beta = {beta:.6f}")
    if TE == True:
        print("TE Mode")
    else:
        print("TM Mode")

#########mode solution########
##############################
beta_fundamental = beta_solutions[0]  # Take fundamental mode

x = np.linspace(-2*d, 2*d, 1000)

    # Calculate field using the beta solution found earlier
beta = beta_solutions[0]  # Use first found mode
def plot_mode_shape(x, E, TE, l, d, n):
    plt.figure(figsize=(10, 6))
    plt.plot(x, E, 'b-', label=f'{"TE" if TE else "TM"} Mode')
    plt.xlabel('Position (m)')
    plt.ylabel('Electric Field')
    plt.title(f'{"TE" if TE else "TM"} Mode Shape\n' + 
             f'λ={l*1e9:.0f}nm\n' +
             f'n_core={n[1]:.4f}, n_clad={n[0]:.4f}')
    vlines = [0, d]
    for lines in vlines:
        plt.axvline(x=lines,color='k', linestyle='--', alpha=0.5)
    plt.grid(True)
    plt.legend()
    plt.show()

def calculate_and_plot_mode(x, beta, l, k0, d_core, TE, n_core, n_clad):
    n = [n_clad, n_core, n_clad]  # refractive index profile
    d = [d_core]  # only need to specify the core thickness
    E = calculate_mode_shape(x, beta, k0, n, d, TE)
    plot_mode_shape(x, E, TE, l, d, n)
    

x = np.linspace(-d, 2*d, 1000)

#calculate_and_plot_mode(x, beta, l, k0, d, TE, n[1], n[0])
#TE = True
#calculate_and_plot_mode(x, beta, l, k0, d, TE, n[1], n[0])
########End of mode solution############
########################################

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
ETM = calculate_mode_shape(x, beta, k0, n, [d], TE=False)

A_fit1, x0_fit1, sigma_fit1 = fit_gaussian(x, ETE)
A_fit2, x0_fit2, sigma_fit2 = fit_gaussian(x, ETM)

plt.figure(figsize=(8, 6))
plt.plot(x * 1e6, ETE, label='Mode Field', color='b')
plt.plot(x * 1e6, A_fit1 * np.exp(-((x - x0_fit1) ** 2) / (2 * sigma_fit1 ** 2)), 'r--', label='Gaussian Fit')
plt.xlabel('Position (µm)')
plt.ylabel('Electric Field Amplitude')
plt.title('Mode Shape and Gaussian Fit')
plt.legend()
plt.grid(True)

plt.figure(figsize=(8,6))
plt.plot(x * 1e6, ETM, label='Mode Field', color='b')
plt.plot(x * 1e6, A_fit2 * np.exp(-((x - x0_fit2) ** 2) / (2 * sigma_fit2 ** 2)), 'r--', label='Gaussian Fit')
plt.xlabel('Position (µm)')
plt.ylabel('Electric Field Amplitude')
plt.title('Mode Shape and Gaussian Fit')
plt.legend()
plt.grid(True)
plt.show()




