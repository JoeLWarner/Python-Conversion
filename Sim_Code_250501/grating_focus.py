#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:01:33 2025

@author: joel
"""

import numpy as np
import matplotlib.pyplot as plt
from silica_my import silica_n
from Grating_angles_2D import grating_angles_2D_f2
from scatteringrate2D import scatteringrate2D_f
from scipy.integrate import cumtrapz, trapz

lam = 780e-9        # wavelength [m]
k0 = 2 * np.pi / lam  # vacuum propagation constant [1/m]
c = 3e8             # speed of light [m/s]

# Mode parameters (from nslabmodes.m)
neff_TE = 1.455879904809160   # effective index for TE mode (pol=1)
w0_TE   = 2.145549895305066e-6  # waist of fundamental TE mode
neff_TM = 1.455869613276399   # effective index for TM mode (pol=2)
w0_TM   = 2.155888764131035e-6  # waist of fundamental TM mode

#%% Define target intensity distribution

phitar = 135 * np.pi / 180   # target propagation angle [rad]
dtar = 0.01                  # distance from centre of grating to target [m]
efficiency = 0.9             # fraction of pump power converted into output beam

# Create a grid of z-points (propagation direction)
zv = np.arange(-1, 1.01, 0.01) * 0.001  # from -1e-3 to 1e-3 [m]

wtar = 300e-6  # waist of the emitted beam [m]
Itar = np.exp(-(zv / wtar) ** 2)  # intensity profile (Gaussian)

# Normalize target intensity to the desired efficiency using numerical integration
Itar = Itar * efficiency / trapz(Itar, zv)

# Plot target intensity distribution
plt.plot(zv, Itar, 'b-')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel('z (m)')
plt.ylabel('Emitted field intensity (TE, target)')
plt.title(f'Efficiency: {efficiency}   Input power: 1')
plt.grid(True)

#%% Design of grating for TE pump

pol = 1  # pol = 1 for TE (horizontal input polarization)

if pol == 1:
    neff = neff_TE
    w0 = w0_TE
else:
    neff = neff_TM
    w0 = w0_TM

beta = neff * k0  # propagation constant of the mode

dng = 1  # grating index contrast (set to 1 for design calculation)
# silica refractive index – note: in the MATLAB code lam*1e6 was used (µm conversion)
nsil = silica_n(lam)

# Position of target point (in grating plane coordinates)
ztar = -dtar * np.cos(phitar)
ytar = dtar * np.sin(phitar)

# Calculate required emission angle for each z-point in the grating
# (Here, phiz is the local angle required such that the reflected ray reaches the target)
phiz = np.arccos( -(ztar - zv) / np.sqrt((ztar - zv)**2 + ytar**2) )

# From the emission angle, compute the required grating period.
# k_z is the z-component of the wave vector in the cladding:
kz = -np.cos(phiz) * nsil * k0
# The grating must provide the difference between the mode propagation constant and k_z:
kgrat = beta - kz
Lambda = 2 * np.pi / kgrat  # required grating period at each z

# Choose the grating tilt angle (here constant over z) as:
theta = phitar / 2  + 0 * zv  # vector with same shape as zv

# Calculate reflectance of the grating (for dng=1)
# The scatteringrate2D_f function should accept Lambda (array), theta (array), pol, k0, neff, nsil, w0, dng.
al, Ex, Ey, Ez = scatteringrate2D_f(Lambda, theta, pol, k0, neff, nsil, w0, dng)
alTE = al.copy()  # save the TE scattering rate

# Calculate the required grating index modulation (dngv) to get the correct output field strength.
# Note: MATLAB’s cumtrapz returns an array of the cumulative integral; here we use scipy.integrate.cumtrapz
# We set initial=0 so that the output array has the same length as zv.
cumItar = cumtrapz(Itar, zv, initial=0)
# Avoid division by zero (or negative values) by ensuring denominator stays positive.
denom = 1 - cumItar
# dngv = sqrt( 1./al .* Itar ./ (1-cumtrapz(Itar)) )
dngv = np.sqrt( np.divide(Itar, al * denom, out=np.zeros_like(Itar), where=(al*denom)>0) )

# Plot the designed grating properties
fig, axs = plt.subplots(3, sharex=True)
axs[0].plot(zv, phiz * 180/np.pi)
axs[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#axs[0].set_xlabel('z (m)')
axs[0].set_ylabel(r'$\phi$ (deg)')
axs[0].grid(True)

axs[1].plot(zv, Lambda)
axs[1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#axs[1].set_xlabel('z (m)')
axs[1].set_ylabel(r'$L_g$ (m)')
axs[1].grid(True)

axs[2].plot(zv, dngv)
axs[2].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axs[2].set_xlabel('z (m)')
axs[2].set_ylabel('dng')
axs[2].grid(True)

plt.tight_layout()
#%% What does this grating do to a TM pump?

pol = 2  # now use TM polarization

if pol == 1:
    neff = neff_TE
    w0 = w0_TE
else:
    neff = neff_TM
    w0 = w0_TM

beta = neff * k0  # update propagation constant

dng = 1  # for design, use dng=1
nsil = silica_n(lam)  # update silica index

# Get TM scattering rate (with the same Lambda and theta as before)
al, Ex, Ey, Ez = scatteringrate2D_f(Lambda, theta, pol, k0, neff, nsil, w0, dng)
alTM = al.copy()

# Calculate the TM reflected intensity (arbitrary units)
cum_al = cumtrapz(al * dngv**2, zv, initial=0)
Itm = np.exp(-cum_al) * al * dngv**2

# Set pump power so that at the maximum of the TE reflected field, the TM field
# has the same intensity (for creating circular polarization, for example)
Imax = Itar.max()
ii = Itar.argmax()
P0TM = Imax / Itm[ii] if Itm[ii] != 0 else 0
Itm = Itm * P0TM
print(P0TM)
# Overall efficiency for the TM pump
efficiency_TM = 1 - np.exp(-trapz(al * dngv**2, zv))

# Plot the emitted field intensities for TM (solid) and target TE (dashed)
plt.figure(3)
plt.plot(zv, Itm, label='TM')
plt.plot(zv, Itar, '--', label='TE, target')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel('z (m)')
plt.ylabel('Emitted field intensity')
#plt.title(f'Efficiency: {efficiency_TM:.3f}   Input power: {P0TM:.3f}')
plt.legend()
plt.grid(True)

# Plot the reflectance (loss rate) for TE and TM
plt.figure(4)
plt.plot(zv, alTE * dngv**2, label='TE')
plt.plot(zv, alTM * dngv**2, label='TM')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel('z (m)')
plt.ylabel('Reflectance (1/m)')
plt.legend()
plt.grid(True)

plt.show()


