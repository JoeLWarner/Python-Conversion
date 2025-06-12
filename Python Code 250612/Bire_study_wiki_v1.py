#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 14:09:35 2025

@author: joel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Birefringent layer comparison including Stokes parameters to identify circular polarization.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

# Physical constants
C_LIGHT = 2.998e8  # Speed of light

# ----------------------------------------------------------------------------
# Permittivity tensor and dispersion
# ----------------------------------------------------------------------------
def create_permittivity_tensor(n_o, n_e, optic_axis=[0,1,0]):
    c_hat = np.array(optic_axis)/np.linalg.norm(optic_axis)
    I = np.eye(3)
    cc = np.outer(c_hat, c_hat)
    return n_o**2*I + (n_e**2 - n_o**2)*cc

# Closed-form effective indices (eqs. 7–8)
def calc_effective_indices(theta, n_o, n_e):
    n_ord = n_o
    n_ext = n_o*n_e/np.sqrt(n_o**2*np.cos(theta)**2 + n_e**2*np.sin(theta)**2)
    return n_ord, n_ext

# ----------------------------------------------------------------------------
# Propagation through slab
# ----------------------------------------------------------------------------
def propagate_through_biref(E_in, k_dir, thickness, n_o, n_e, wavelength, optic_axis=[0,1,0]):
    # compute angle
    k_hat = np.array(k_dir)/np.linalg.norm(k_dir)
    c_hat = np.array(optic_axis)/np.linalg.norm(optic_axis)
    theta = np.arccos(np.abs(np.dot(k_hat, c_hat)))
    # effective indices
    n_ord, n_ext = calc_effective_indices(theta, n_o, n_e)
    # path length
    path = thickness/abs(np.dot(k_hat, [0,0,1]))
    # phases
    k0 = 2*np.pi/wavelength
    phi_o = k0*n_ord*path
    phi_e = k0*n_ext*path
    # decompose approximately: Ey->extra, Ex->ord
    Ex, Ey, Ez = E_in
    Ex_out = Ex*np.exp(1j*phi_o)
    Ey_out = Ey*np.exp(1j*phi_e)
    Ez_out = Ez*np.exp(1j*phi_o)
    return np.array([Ex_out, Ey_out, Ez_out])

# ----------------------------------------------------------------------------
# Stokes calculation
# ----------------------------------------------------------------------------
def compute_stokes(E):
    Ex, Ey = E[0], E[1]
    S0 = abs(Ex)**2 + abs(Ey)**2
    S1 = abs(Ex)**2 - abs(Ey)**2
    S2 = 2*np.real(Ex*np.conj(Ey))
    S3 = 2*np.imag(Ex*np.conj(Ey))
    return S0, S1, S2, S3

# ----------------------------------------------------------------------------
# Compute accurate retardance and Stokes parameters
# ----------------------------------------------------------------------------
lam = 780e-9
n_e = 1.486
n_o = 1.658
k0 = 2*np.pi/lam
# design thickness for quarter-wave at normal incidence
thickness = np.abs((np.pi/2) / (k0 * (n_e - n_o)))

def compute_accurate_and_stokes(angles_deg, n_o=n_o, n_e=n_e,
                                 thickness=thickness, wavelength=lam):
    angles = np.array(angles_deg)
    thetas = np.deg2rad(angles)
    k0 = 2*np.pi/wavelength

    phase_accur = np.zeros_like(thetas)
    stokes_s3   = np.zeros_like(thetas)

    # input polarization 45° linear
    E_in = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0], dtype=complex)

    for i, theta in enumerate(thetas):
        # accurate phase
        n_ord, n_ext = calc_effective_indices(theta, n_o, n_e)
        path = thickness/np.cos(theta)
        phase_accur[i] = k0 * (n_ext - n_ord) * path

        # propagate and compute Stokes
        k_dir = [np.sin(theta), 0, np.cos(theta)]
        E_out = propagate_through_biref(E_in, k_dir, thickness, n_o, n_e, wavelength)
        S0, S1, S2, S3 = compute_stokes(E_out)
        stokes_s3[i] = S3/S0 if S0 > 0 else np.nan

    # Plot only the accurate results
    fig, axs = plt.subplots(2,1, figsize=(8,10))

    # Phase retardance vs angle
    axs[0].plot(angles, phase_accur * 180/np.pi, 'r-s', label='Accurate Δφ')
    axs[0].axhline(90, color='k', linestyle='--', label='Target 90°')
    axs[0].axhline(-90, color='k', linestyle='--', label='Target -90°')
    axs[0].set(title='Accurate Retardance vs Angle', xlabel='Angle (°)', ylabel='Δφ (°)')
    axs[0].legend()
    axs[0].grid(True)

    # Circularity (S3/S0) vs angle
    axs[1].plot(angles, stokes_s3, 'k-d', label='S3/S0')
    axs[1].axhline(1, color='r', linestyle='--')
    axs[1].axhline(-1, color='r', linestyle='--')
    axs[1].set(title='Circular Polarization Metric vs Angle', xlabel='Angle (°)', ylabel='S3/S0')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
    
    
    return angles, phase_accur, stokes_s3

if __name__ == '__main__':
    angles = np.linspace(0, 80, 17)
    compute_accurate_and_stokes(angles)

