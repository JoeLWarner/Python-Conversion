#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 13:47:00 2025

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
zv = np.arange(-1, 1.01, 0.001) * 0.001  # from -1e-3 to 1e-3 [m]

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

#% Design of grating for TE pump

pol = 1  # pol = 1 for TE (horizontal input polarization)

if pol == 1:
    neff = neff_TE
    w0 = w0_TE
else:
    neff = neff_TM
    w0 = w0_TM

beta = neff * k0 
dng = 1 
nsil = silica_n(lam)

ztar = -dtar * np.cos(phitar)
ytar = dtar * np.sin(phitar)

phiz = np.arccos( -(ztar - zv) / np.sqrt((ztar - zv)**2 + ytar**2) )
kz = np.cos(phiz) * nsil * k0
kgrat = beta - kz
Lambda = 2 * np.pi / kgrat 
theta = phitar / 2  + 0 * zv 
al_TE, Ex_TE, Ey_TE, Ez_TE = scatteringrate2D_f(Lambda, theta, pol, k0, neff, nsil, w0, dng)

######TM wave through te grating

pol = 2
if pol == 1:
    neff = neff_TE
    w0 = w0_TE
else:
    neff = neff_TM
    w0 = w0_TM

beta = neff * k0  
dng = 1  #
nsil = silica_n(lam)
# Get TM scattering rate (with the same Lambda and theta as before)
al_TM, Ex_TM, Ey_TM, Ez_TM = scatteringrate2D_f(Lambda, theta, pol, k0, neff, nsil, w0, dng)

def calculate_emission_angle(Lambda, theta, k0, neff, nsil):
    beta = neff * k0
    kgrat = 2 * np.pi / Lambda  # grating momentum
    k_z = beta - kgrat
    phi = np.arccos(k_z / (nsil * k0))
    return phi

phiTE = calculate_emission_angle(Lambda, theta, k0, neff_TE, nsil)
phiTM = calculate_emission_angle(Lambda, theta, k0, neff_TM, nsil)

angle_diff = (phiTM - phiTE) * 180/np.pi

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(zv, phiTE * 180/np.pi, label='TE emission angle')
ax1.plot(zv, phiTM * 180/np.pi, label='TM emission angle')
ax1.plot(zv, phiz * 180/np.pi, '--', label='Design target angle')

ax2.plot(zv, angle_diff, label='Emission angle difference', color='red')

ax1.set_xlabel('z (m)')
ax1.set_ylabel('Emission angle (degrees)')
ax2.set_ylabel('Angle difference (degrees)')
ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax1.legend()
ax2.legend(loc='lower left')
ax1.grid(True)
ax1.tick_params(axis="x", which="both", direction="in", pad=5, top=True, bottom=True)
ax1.tick_params(axis="y", which="both", direction="in", pad=5, right=True, left=True)
plt.show()

avg_diff = np.mean(angle_diff)
print(f"Average difference between TM and TE emission angles: {avg_diff:.4f} degrees")

diffs = []
temp_TE = 1.45
temp_TM = 1.45
x_values = []  # To store the percentage differences

plt.figure(4)
for i in range(0, 150):
    phiTE = calculate_emission_angle(Lambda, theta, k0, temp_TE, nsil)
    phiTM = calculate_emission_angle(Lambda, theta, k0, temp_TM, nsil)
    temp_TE += temp_TE*0.001  
    percentage_diff = ((temp_TE - temp_TM) / temp_TM) * 100
    angle_difflist = (phiTM - phiTE) * 180/np.pi
    avg_difflist = np.mean(angle_difflist)
    diffs.append(avg_difflist)
    x_values.append(percentage_diff)

plt.plot(x_values, diffs)
plt.grid()
plt.xlabel('Percentage difference in neff (%)')
plt.ylabel('Difference in emission angle (degs)')
plt.tick_params(axis="x", which="both", direction="in", pad=5, top=True, bottom=True)
plt.tick_params(axis="y", which="both", direction="in", pad=5, right=True, left=True)
plt.xlim(left=0)
plt.ylim(bottom=0)

