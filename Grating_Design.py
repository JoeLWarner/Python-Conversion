#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:33:07 2025

@author: joel
"""

import numpy as np
import matplotlib.pyplot as plt
from silica_my import silica_n
from Grating_angles_2D import grating_angles_2D_f2
from scatteringrate2D import scatteringrate2D_f

# Parameters
l = 780e-9  # Wavelength (m)
k0 = 2 * np.pi / l
pol = 1  # Polarization: 1 for TE (s), 2 for TM (p)

# Mode parameters (from nslabmodes)
neff_TE = 1.455879904809160
w0_TE = 2.145549895305066e-06
neff_TM = 1.455869613276399
w0_TM = 2.155888764131035e-06

# Assign effective index and waist based on polarization
neff = neff_TE if pol == 1 else neff_TM
w0 = w0_TE if pol == 1 else w0_TM

beta = neff * k0  # Propagation constant
dng = 1  # Grating index contrast i think this is 1 for calibration but should be 2e-3 for realistic
nsil = silica_n(l)  # Silica refractive index (cladding)

# Define target beam
xv = np.linspace(-20e-6, 20e-6, 4000)  # Grid in the grating plane

# Define a tilted Gaussian target beam
ntar = np.array([1, 1]) / np.linalg.norm([1, 1])
w2 = 2.5e-6
zfoc = 50e-6
E20 = 0.1

zzr = xv * ntar[0]  # z in rotated frame
xxr = np.sqrt(xv**2 - zzr**2)  # x in rotated frame
zzr -= zfoc

zR = np.pi * w2**2 / (l) * nsil
w2z = w2 * np.sqrt(1 + (zzr / zR)**2)
eta = 0.5 * np.arctan(zzr / zR)
Rzi = zzr / (zzr**2 + zR**2)

E2 = E20 * np.sqrt(w2 / w2z) * np.exp(-(xxr / w2z) ** 2) * np.exp(1j * (k0 * nsil * zzr + k0 * nsil * xxr**2 * Rzi / 2 - eta))

# Plot Target Field
plt.figure(1)
plt.plot(xv, np.abs(E2)**2)
plt.xlabel('x (m)')
plt.ylabel('|E2|^2')
plt.title('Target Field |E2|^2 at z=0')
plt.show()

plt.figure(2)
plt.plot(xv, np.angle(E2))
plt.xlabel('x (m)')
plt.ylabel('Phase(E2)')
plt.title('Target Field Phase at z=0')
plt.show()
#all fine to here

# Calculate central grating properties
lamgrat0, alphatilt0 = grating_angles_2D_f2(ntar[0], ntar[1], beta, k0 * nsil)

# Calculate grid spacing
dx = xv[1] - xv[0]  # Distance between points

# Central difference for kx
kx = -1j * (E2[2:] - E2[:-2]) / (2*dx)
kx = kx / E2[1:-1]
kx = np.concatenate(([kx[0]], kx, [kx[-1]]))
kx = np.real(kx)
kz = np.sqrt((k0*nsil)**2 - kx**2)

# Compute local grating properties
lamgrat, alphatilt = grating_angles_2D_f2(kx, kz, beta, k0 * nsil)

# Plot kx, kz, lambda, and alpha tilt
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].plot(xv, kx)
axs[0, 0].set_title('kx of Local Target Field')
axs[0, 0].set_xlabel('x (m)')
axs[0, 0].set_ylabel('kx')

axs[0, 1].plot(xv, kz)
axs[0, 1].set_title('kz of Local Target Field')
axs[0, 1].set_xlabel('x (m)')
axs[0, 1].set_ylabel('kz')

axs[1, 0].plot(xv, lamgrat)
axs[1, 0].set_title('Local Grating Period')
axs[1, 0].set_xlabel('x (m)')
axs[1, 0].set_ylabel('Lambda')

axs[1, 1].plot(xv, np.degrees(alphatilt))
axs[1, 1].set_title('Local Optimum Grating Tilt')
axs[1, 1].set_xlabel('x (m)')
axs[1, 1].set_ylabel('Tilt Angle (deg)')

plt.tight_layout()
plt.show()

# Calculate scattering rate
al1, Ex1, Ey1, Ez1 = scatteringrate2D_f(lamgrat, alphatilt0, pol, k0, neff, nsil, w0, dng)
E1norm = np.sqrt(Ex1**2 + Ey1**2 + Ez1**2)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].plot(xv, Ex1/E1norm)
axs[0, 0].set_title('Scattered Ex component')
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('Ex')
axs[0, 1].plot(xv, Ey1/E1norm)
axs[0, 1].set_title('Scattered Ey component')
axs[0, 1].set_xlabel('x')
axs[0, 1].set_ylabel('Ey')
axs[1, 0].plot(xv, Ez1/E1norm)
axs[1, 0].set_title('Scattered Ez component')
axs[1, 0].set_xlabel('x')
axs[1, 0].set_ylabel('Ez')

plt.tight_layout()
plt.show()


plt.figure(5)
plt.plot(xv, al1)
plt.xlabel('x (m)')
plt.ylabel('Alpha')
plt.title('Scattering Efficiency α (1/m) for dng=1')
plt.show()

# Compute and normalize grating strength
dng1 = np.abs(E2) / E1norm
dng1 /= np.max(dng1)

plt.figure(7)
plt.plot(xv, dng1)
plt.xlabel('x (m)')
plt.ylabel('dng')
plt.title('Grating dng (normalized)')
plt.show()

# Compute pump depletion
conv_eff = 0.99
loss = np.cumsum(dng1**2 * al1) * (xv[1] - xv[0])
dngbar_max = np.sqrt(1 / np.max(loss))
dngbar = dngbar_max * np.sqrt(conv_eff)
P = 1 - dngbar**2 * loss
dng_full = np.real(dng1 * dngbar / np.sqrt(P))

fig, axs = plt.subplots(2, 1, figsize=(10, 8))
axs[0].plot(xv, P)
axs[0].set_xlabel('x (m)')
axs[0].set_ylabel('P')
axs[0].set_title('Pump Power')

axs[1].plot(xv, dng_full)
axs[1].set_xlabel('x (m)')
axs[1].set_ylabel('dng')
axs[1].set_title('Grating dng with Pump Depletion')
plt.tight_layout()
plt.show()


# Compute full grating
phi = np.angle(E2 * np.exp(-1j * k0 * nsil * (ntar[0] * xv)))
ng = dng_full * np.sin(2 * np.pi / lamgrat0 * xv - phi)

plt.figure(8)
plt.plot(xv, ng)
plt.xlabel('x (m)')
plt.ylabel('Grating Index')
plt.title('Full Grating Index Profile')
plt.show()
