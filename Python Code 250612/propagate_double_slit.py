#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 13:17:44 2025

@author: joel
"""


import numpy as np
import matplotlib.pyplot as plt
from silica_my import silica_n
from Grating_angles_2D import grating_angles_2D_f2
from scatteringrate2D import scatteringrate2D_f
from scipy.fftpack import fft, ifft, fftshift


l = 780e-9  
k0 = 2 * np.pi / l
pol = 1  # polarization: 1 for TE (s), 2 for TM (p)

neff_TE = 1.455879904809160
w0_TE = 2.145549895305066e-06
neff_TM = 1.455869613276399
w0_TM = 2.155888764131035e-06

neff = neff_TE if pol == 1 else neff_TM
w0 = w0_TE if pol == 1 else w0_TM

beta = neff * k0  
dng = 1  # Grating index contrast i think this is 1 for calibration but should be 2e-3 for realistic
nsil = silica_n(l)  # Silica refractive index 

#  target beam section
xv = np.linspace(-60e-6, 60e-6, 12000)  #  create grid in the grating plane

# tilted Gaussian target beam
ntar = np.array([0, 1]) / np.linalg.norm([0, 1])
w2 = 3e-6
zfoc = 50e-6
E20 = 0.1

#align reference frame with the direction of the guassian propagation 
zzr = xv * ntar[0]  # z in rotated frame
xxr = np.sqrt(xv**2 - zzr**2)  # x in rotated frame
zzr -= zfoc
#defining gasussian properties
zR = np.pi * w2**2 / (l) * nsil
w2z = w2 * np.sqrt(1 + (zzr / zR)**2)
eta = 0.5 * np.arctan(zzr / zR) * 0.5
Rzi = zzr / (zzr**2 + zR**2)

#E2 = E20 * np.sqrt(w2 / w2z) * np.exp(-(xxr / w2z) ** 2) * np.exp(1j * (k0 * nsil * zzr + k0 * nsil * xxr**2 * Rzi / 2 - eta))

#code changed sligtly for super gaussian 
p = 1  # Super-Gaussian exponent, p > 2
E2 = E20 * np.sqrt(w2 / w2z) * np.exp(-(xxr / w2z) ** (2*p)) * np.exp(1j * (k0 * nsil * zzr + k0 * nsil * xxr**2 * Rzi / 2 - eta))

# target field plot
# =============================================================================
# plt.figure(1)
# plt.plot(xv, np.abs(E2)**2)
# plt.xlabel('x (m)')
# plt.ylabel('|E2|^2')
# plt.title('Target Field |E2|^2 at z=0')
# plt.show()
# 
# plt.figure(2)
# plt.plot(xv, np.angle(E2))
# plt.xlabel('x (m)')
# plt.ylabel('Phase(E2)')
# plt.title('Target Field Phase at z=0')
# plt.show()
# =============================================================================

nx = len(xv)
x = xv[:nx]
E0 = E2[:nx]  

# Define double-slit in the x domain.
obstacle = np.zeros_like(x)  # fully blocking initially
slit_width = 200 
slit_separation = 200  
center = len(x) // 2

# first slit (values 1 where the slit is open)
obstacle[center - slit_separation//2 - slit_width//2 : center - slit_separation//2 + slit_width//2] = 1
# second slit
obstacle[center + slit_separation//2 - slit_width//2 : center + slit_separation//2 + slit_width//2] = 1

dx = x[1] - x[0]
dkx = 2 * np.pi / (nx * dx)
kx = np.arange(-nx/2, nx/2) * dkx

kz = np.zeros_like(kx, dtype=complex)
mask_real = (nsil**2 * k0**2) >= (kx**2)
kz[mask_real] = np.sqrt((nsil**2 * k0**2) - kx[mask_real]**2)
kz[~mask_real] = 1j * np.sqrt(kx[~mask_real]**2 - (nsil**2 * k0**2))

z = np.arange(0, 81) * 1e-6  # from 0 to 80 microns
nz = len(z)
mid_index = nz // 2

# from z = 0 to z_mid
# fft without mask until double slit point 
E0k = np.fft.fftshift(np.fft.fft(np.fft.fftshift(E0)))
Etot_first = np.zeros((mid_index + 1, nx), dtype=complex)
Etot_first[0, :] = E0

for ii in range(1, mid_index + 1):
    Ek = E0k * np.exp(1j * kz * z[ii])
    E = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Ek)))
    Etot_first[ii, :] = E

# Field at z = z[mid_index]
E_mid = Etot_first[-1, :]
E_mid_masked = E_mid * obstacle

# from z = z_mid to z_final 
E_mid_masked_k = np.fft.fftshift(np.fft.fft(np.fft.fftshift(E_mid_masked)))
# Define relative z for segment 2 (starting at 0)
z2 = z[mid_index:]
z2_rel = z2 - z2[0]  # relative propagation distance
n2 = len(z2)

Etot_second = np.zeros((n2, nx), dtype=complex)
Etot_second[0, :] = E_mid_masked  

for ii in range(1, n2):
    Ek = E_mid_masked_k * np.exp(1j * kz * z2_rel[ii])
    E = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Ek)))
    Etot_second[ii, :] = E

Etot_total = np.vstack((Etot_first, Etot_second[1:, :]))
z_total = np.concatenate((z[:mid_index+1], z2[1:]))

plt.figure(figsize=(8, 6))
plt.pcolormesh(x, z_total, np.abs(Etot_total)**2, shading='auto')
plt.xlabel('x (m)')
plt.ylabel('z (m)')
plt.title('Propagation with Double Slit at z = {:.1e} m'.format(z[mid_index]))
plt.colorbar(label='Intensity (a.u.)')
plt.show()

