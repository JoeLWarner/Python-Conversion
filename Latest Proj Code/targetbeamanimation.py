#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script propagates a target field (computed via a tilted Gaussian) using FFT
and animates the final pcolormesh plot as the target beam direction is varied.
The target beam direction is controlled by:
    ntar = [ntar_x, 1] / norm([ntar_x,1]),
with ntar_x varying between 0 and 1.
A horizontal red dashed line is drawn at the focus (at z = zfoc * ntar[1]) and
updates as ntar_x changes.

@author: joel
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftshift
import matplotlib.animation as animation
from silica_my import silica_n  # assumes you have this module

# -----------------------------
# Parameters
# -----------------------------
l = 780e-9  
k0 = 2 * np.pi / l
neff_TE = 1.455879904809160
w0_TE = 2.145549895305066e-06
nsil = silica_n(l)  # Silica refractive index

# Use TE parameters:
neff = neff_TE
w0 = w0_TE
beta = neff * k0  
dng = 1

# -----------------------------
# Grid and target beam parameters
# -----------------------------
xv = np.linspace(-20e-6, 20e-6, 4000)   # grating-plane x-grid
dx = xv[1] - xv[0]

# Target (tilted Gaussian) beam parameters
w2 = 2.5e-6
zfoc = 20e-6  # focal distance
E20 = 0.1

# Use only the central portion for FFT propagation:
nx = 2 * (len(xv) // 2)
x = xv[:nx]
dx = x[1] - x[0]
dkx = 2 * np.pi / (nx * dx)
kx = np.arange(-nx/2, nx/2) * dkx

# Build kz array allowing for evanescent components:
kz = np.zeros_like(kx, dtype=complex)
mask_real = (nsil**2 * k0**2) >= (kx**2)
kz[mask_real] = np.sqrt((nsil**2 * k0**2) - kx[mask_real]**2)
kz[~mask_real] = 1j * np.sqrt(kx[~mask_real]**2 - (nsil**2 * k0**2))

# Propagation distances (z from 0 to 50 microns)
z = np.arange(0, 51) * 1e-6
nz = len(z)

# Pre-allocate field propagation array:
Etot = np.zeros((nz, nx), dtype=complex)

# Pre-calculate constant parameter for the Gaussian beam:
zR = np.pi * w2**2 / l * nsil

# -----------------------------
# Animation: vary ntar_x from 0 to 1
# -----------------------------
ntar_x_values = np.linspace(-1, 1, 50)  # 20 values between 0 and 1

# Set up the figure
fig, ax = plt.subplots(figsize=(8,6))
ntar_x_init = ntar_x_values[0]
ntar = np.array([ntar_x_init, 1.0]) / np.linalg.norm([ntar_x_init, 1.0])

# Compute the tilted Gaussian target field (E2) at z = 0:
zzr = xv * ntar[0]                 
xxr = np.sqrt(np.maximum(xv**2 - zzr**2, 0))  
zzr_shift = zzr - zfoc             
w2z = w2 * np.sqrt(1 + (zzr_shift / zR)**2)
eta = 0.5 * np.arctan(zzr_shift / zR)
Rzi = zzr_shift / (zzr_shift**2 + zR**2)
E2 = E20 * np.sqrt(w2 / w2z) * np.exp( -(xxr / w2z)**2 ) \
     * np.exp(1j * (k0 * nsil * zzr_shift + k0 * nsil * xxr**2 * Rzi/2 - eta))

# Use the central portion for propagation
E0 = E2[:nx]

# Compute the FFT of the initial field
E0k = fftshift(fft(fftshift(E0)))
for ii in range(1, nz):
    Ek = E0k * np.exp(1j * kz * z[ii])
    E = fftshift(ifft(fftshift(Ek)))
    Etot[ii, :] = E
Etot[0, :] = E0

# Create initial pcolormesh plot
im = ax.pcolormesh(x, z, np.abs(Etot)**2, shading='auto', cmap='viridis')
ax.set_xlabel('x (m)')
ax.set_ylabel('z (m)')
title_text = ax.set_title('Propagated Field Intensity, ntar_x = {:.2f}'.format(ntar_x_init))
cb = fig.colorbar(im, ax=ax, label='Intensity (a.u.)')

# Create a horizontal dashed line at the focus z coordinate.
# For a beam with ntar = [ntar_x, 1]/||[ntar_x,1]||, the focus is at z_focus = zfoc * ntar[1].
z_focus = zfoc * ntar[1]
line, = ax.plot([x.min(), x.max()], [z_focus, z_focus], 'r--', linewidth=2)

def update(frame):
    # Get current ntar_x and update ntar vector
    ntar_x = ntar_x_values[frame]
    ntar = np.array([ntar_x, 1.0]) / np.linalg.norm([ntar_x, 1.0])
    
    # Recompute the tilted Gaussian target field E2 at z = 0
    zzr = xv * ntar[0]
    xxr = np.sqrt(np.maximum(xv**2 - zzr**2, 0))
    zzr_shift = zzr - zfoc
    w2z = w2 * np.sqrt(1 + (zzr_shift / zR)**2)
    eta = 0.5 * np.arctan(zzr_shift / zR)
    Rzi = zzr_shift / (zzr_shift**2 + zR**2)
    E2 = E20 * np.sqrt(w2 / w2z) * np.exp( -(xxr / w2z)**2 ) \
         * np.exp(1j * (k0 * nsil * zzr_shift + k0 * nsil * xxr**2 * Rzi/2 - eta))
    E0 = E2[:nx]
    
    # FFT propagation
    E0k = fftshift(fft(fftshift(E0)))
    for ii in range(1, nz):
        Ek = E0k * np.exp(1j * kz * z[ii])
        E = fftshift(ifft(fftshift(Ek)))
        Etot[ii, :] = E
    Etot[0, :] = E0
    
    # Update the pcolormesh with new propagated intensity.
    im.set_array((np.abs(Etot)**2).ravel())
    title_text.set_text('Propagated Field Intensity, ntar_x = {:.2f}'.format(ntar_x))
    
    # Update the horizontal dashed line position.
    # The focus is defined as z_focus = zfoc * ntar[1].
    #z_focus = zfoc * ntar[1]
    #line.set_data([x.min(), x.max()], [z_focus, z_focus])
    
    return [im, title_text, line]

anim = animation.FuncAnimation(fig, update, frames=len(ntar_x_values), interval=50, blit=False)
plt.show()
