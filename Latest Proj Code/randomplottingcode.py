#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 14:03:09 2025

@author: joel
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz, trapz
from silica_my import silica_n
from Grating_angles_2D import grating_angles_2D_f2
from scatteringrate2D import scatteringrate2D_f
from scattertester import scatteringrate2D_rect
from matplotlib import rcParams
from matplotlib.ticker import FormatStrFormatter

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
#dng has been made realistic for these simulations
dng = 1e-3  # Grating index contrast i think this is 1 for calibration but should be 1 - 2e-3 for realistic
nsil = silica_n(l)  #silica ref index

xv = np.linspace(-20e-6, 20e-6, 4000)  # Grid in the grating plane

# tilted Gaussian target beam
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

lamgrat0, alphatilt0 = grating_angles_2D_f2(ntar[0], ntar[1], beta, k0 * nsil)


grating_lengths = [10e-3, 10e-6]
diff_angles = [30, 60, 90]
dng_values = np.linspace(0, 3e-3, 100)

###############
# relfectivity as a function of grating modulation 
###############


sixlist = []
ninelist = []
onelist = []
'''

plt.figure(figsize=(8, 6))
for dangle in diff_angles:
    theta = np.radians(dangle / 2)
    Lambda = l / (neff * (1 + np.cos(2 * theta)))
    alphas = []
    for dng in dng_values:
        al1, _, _, _ = scatteringrate2D_f(Lambda, theta, pol, k0, neff, nsil, w0, dng)
        #alphas.append(al1/100)
        r = 1 - np.exp(- al1 * 10e-3)
        alphas.append(r * 100)
        if dangle == 45:
            sixlist.append(r * 100)
        if dangle == 60:
            ninelist.append(r * 100)
        if dangle == 90:
            onelist.append(r * 100)
    if dangle == 60:
        plt.plot(dng_values, alphas, label=f'Diffraction angle {dangle}', marker='d', markersize=3.8)
    else:
        plt.plot(dng_values, alphas, label=f'Diffraction angle {dangle}')
        #plt.axvline(1.2e-3, color='black', linestyle='dashdot')


plt.xlabel(r'Grating index modulation, $\Delta n_g$')
plt.ylabel('Reflectance (%)')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.legend()
#plt.grid(True)
plt.show()
'''
# Set matplotlib to use LaTeX
#rcParams['text.usetex'] = True
rcParams['font.family'] = 'sans-serif'
plt.figure(figsize=(8, 7))

for i, dangle in enumerate(diff_angles):
    theta = np.radians(dangle / 2)
    Lambda = l / (neff * (1 + np.cos(2 * theta)))
    alphas = []
    
    for dng in dng_values:
        al1, _, _, _ = scatteringrate2D_f(Lambda, theta, pol, k0, neff, nsil, w0, dng)
        r = 1 - np.exp(- al1 * 10e-3)
        alphas.append(r * 100)
    
    # Plot with styling based on the angle value
    if dangle == 90:
        plt.plot(dng_values, alphas, 'r--', linewidth=2, label=f'{dangle}$^{{\circ}}$')
        plt.plot(dng_values[::10], np.array(alphas)[::10], 'rs', markersize=0, markerfacecolor='none', markeredgewidth=1.5)
    elif dangle == 30:
        plt.plot(dng_values, alphas, 'k-', linewidth=2, label=f'{dangle}$^{{\circ}}$')
        plt.plot(dng_values[::10], np.array(alphas)[::10], 'kx', markersize=0, markeredgewidth=2)
    elif dangle == 60:
        plt.plot(dng_values, alphas, 'b-', linewidth=2, label=f'{dangle}$^{{\circ}}$')
        plt.plot(dng_values[::10], np.array(alphas)[::10], 'bo', markersize=0, markerfacecolor='none', markeredgewidth=1.5)
    else:
        plt.plot(dng_values, alphas, label=f'{dangle}$^{{\circ}}$')

plt.xlabel(r'Index modulation, $\Delta\rm{n_g}$', fontsize=24, labelpad=10)
plt.ylabel('Reflectance / [%]', fontsize=20)

plt.xlim(min(dng_values), max(dng_values))
plt.ylim(0, 100)

# Get current axes and set tick positions and labels:
ax = plt.gca()
tick_positions = [0, 1e-3, 2e-3, 3e-3]
ax.set_xticks(tick_positions)
ax.set_xticklabels(['0', '1', '2', '3'], fontsize=18)
ax.tick_params(axis="x", which="both", direction="in", pad=5, top=True, bottom=True)
ax.tick_params(axis="y", which="both", direction="in", pad=5, right=True, left=True)


plt.yticks(fontsize=18)
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()

