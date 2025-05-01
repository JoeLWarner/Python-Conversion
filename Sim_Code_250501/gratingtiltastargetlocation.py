#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 16:00:42 2025

@author: joel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script computes the full grating index distribution (ngp) for a range
of target beam propagation directions given by:
    ntar = np.array([ntar_x, 1]) / ||[ntar_x, 1]||
with ntar_x varying from 0 to 1.
Only the final pcolormesh figure is produced.

@author: joel
"""

import numpy as np
import matplotlib.pyplot as plt
from silica_my import silica_n
from Grating_angles_2D import grating_angles_2D_f2
from scatteringrate2D import scatteringrate2D_f

# -----------------------------
# Basic Parameters
# -----------------------------
l = 780e-9  
k0 = 2 * np.pi / l
pol = 1  # TE polarization

neff_TE = 1.455879904809160
w0_TE = 2.145549895305066e-06
neff = neff_TE
w0 = w0_TE
beta = neff * k0  
dng = 1  # Grating index contrast
nsil = silica_n(l)  # Silica refractive index 

# -----------------------------
# Grid & Target Beam Parameters
# -----------------------------
xv = np.linspace(-20e-6, 20e-6, 4000)  # grid in the grating plane
dx = xv[1] - xv[0]

# Gaussian target beam parameters
w2 = 2.5e-6
zfoc = 20e-6
E20 = 0.1

# -----------------------------
# Vary ntar_x from 0 to 1
# -----------------------------
ntar_x_vals = [0.0, 0.25, 0.5, 0.75, 1.0]

# Create subplots for each value of ntar_x
num_plots = len(ntar_x_vals)
fig, axs = plt.subplots(1, num_plots, figsize=(4*num_plots, 6), sharey=True)

for i, ntar_x in enumerate(ntar_x_vals):
    # Construct ntar vector with first element ntar_x and second element 1
    ntar = np.array([ntar_x, 1.0]) / np.linalg.norm([ntar_x, 1.0])
    
    # Rotate coordinates for the tilted Gaussian beam
    zzr = xv * ntar[0]
    xxr = np.sqrt(np.maximum(xv**2 - zzr**2, 0))  # protect against negatives inside sqrt
    zzr_shifted = zzr - zfoc
    zR = np.pi * w2**2 / l * nsil
    w2z = w2 * np.sqrt(1 + (zzr_shifted / zR)**2)
    eta = 0.5 * np.arctan(zzr_shifted / zR)
    Rzi = zzr_shifted / (zzr_shifted**2 + zR**2)
    
    # Compute the target (TE) field E2 at z=0 in the grating plane
    E2 = E20 * np.sqrt(w2 / w2z) * np.exp(-(xxr / w2z)**2) * \
         np.exp(1j * (k0 * nsil * zzr_shifted + k0 * nsil * xxr**2 * Rzi / 2 - eta))
    
    # Compute central grating properties based on ntar (used later in the index profile)
    lamgrat0, alphatilt0 = grating_angles_2D_f2(ntar[0], ntar[1], beta, k0 * nsil)
    
    # Compute local grating properties from finite differences of E2
    kx = -1j * (E2[2:] - E2[:-2]) / (2 * dx)
    kx = kx / E2[1:-1]
    kx = np.concatenate(([kx[0]], kx, [kx[-1]]))
    kx = np.real(kx)
    kz_local = np.sqrt((k0 * nsil)**2 - kx**2)
    lamgrat, alphatilt = grating_angles_2D_f2(kx, kz_local, beta, k0 * nsil)
    
    # Compute scattering rate for TE polarization
    al1, Ex1, Ey1, Ez1 = scatteringrate2D_f(lamgrat, alphatilt0, pol, k0, neff, nsil, w0, dng)
    E1norm = np.sqrt(Ex1**2 + Ey1**2 + Ez1**2)
    
    # Compute grating strength (normalized)
    dng1 = np.abs(E2) / E1norm
    dng1 /= np.max(dng1)
    
    # Pump depletion calculations
    loss = np.cumsum(dng1**2 * al1) * dx
    dngbar_max = np.sqrt(1 / np.max(loss))
    conv_eff = 0.99
    dngbar = dngbar_max * np.sqrt(conv_eff)
    P = 1 - dngbar**2 * loss
    dng_full = np.real(dng1 * dngbar / np.sqrt(P))
    
    # Compute phase term for the grating index (depends on ntar[0])
    phi = np.angle(E2 * np.exp(-1j * k0 * nsil * (ntar[0] * xv)))
    
    # Build 2D map for the full grating index distribution (ngp)
    zv = np.linspace(-3e-6, 3e-6, int(6e-6 / 0.05e-6))
    xxplot, zzplot = np.meshgrid(xv, zv)
    phip = np.outer(np.ones_like(zv), phi)
    dng_fullp = np.outer(np.ones_like(zv), dng_full)
    ngp = dng_fullp * np.sin((2 * np.pi / lamgrat0) * (xxplot - zzplot * np.tan(alphatilt0)) - phip)
    
    # Plot the pcolormesh for this ntar_x
    im = axs[i].pcolormesh(xxplot, zzplot, ngp, shading='auto', cmap='RdBu_r')
    axs[i].set_title(f'ntar_x = {ntar_x:.2f}')
    axs[i].set_xlabel('x (m)')
    if i == 0:
        axs[i].set_ylabel('z (m)')
    axs[i].axis('equal')

fig.suptitle('Full Grating Index Distribution for Varying ntar_x', fontsize=16)
#fig.colorbar(im, ax=axs.ravel().tolist(), label='Grating Index')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
