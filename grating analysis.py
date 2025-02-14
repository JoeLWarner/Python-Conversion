#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:51:04 2025

@author: joel
"""

import numpy as np
import matplotlib.pyplot as plt
from silica_my import silica_n
from Grating_angles_2D import grating_angles_2D_f2
from scatteringrate2D import scatteringrate2D_f

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

l0 = 355e-9 #grating period capital Lambda
theta = np.linspace(10, 80, 500)*np.pi/180
lambgrate = l0

al1, Ex1, Ey1, Ez1 = scatteringrate2D_f(lambgrate, theta, pol, k0, neff, nsil, w0, dng)


plt.figure(1)
plt.plot(theta*180/np.pi, al1)
plt.xlabel(r'$\theta$')
plt.ylabel('Alpha')
plt.title('Scattering Efficiency α (1/m) for dng=1')
plt.show()

grating_lengths = [10e-3, 10e-5]
diff_angles = [60, 90, 120]
dng_values = np.linspace(0, 5e-3, 100)

# relfectivity as a function of grating modulation 
plt.figure(figsize=(8, 6))
for dangle in diff_angles:
    theta = np.radians(dangle / 2)
    Lambda = l / (neff * (1 + np.cos(2 * theta)))
    alphas = []
    for dng in dng_values:
        al1, _, _, _ = scatteringrate2D_f(Lambda, theta, pol, k0, neff, nsil, w0, dng)
        alphas.append(al1)
    plt.plot(dng_values, alphas, label=f'Diffraction angle {dangle}')
plt.xlabel('Grating index modulation, dng')
plt.ylabel('Reflectance, alpha (1/m)')
plt.title('Reflectance vs. Grating Index Modulation')
plt.legend()
plt.grid(True)
plt.show()

#reflectivity as a function of diffraction angle. 

plt.figure(figsize=(8, 6))
for angle in diff_angles:
    theta = np.radians(angle / 2)
    Lambda = l / (neff * (1 + np.cos(2 * theta)))
    delta = np.radians(10)
    diff_range = np.linspace(theta-delta, theta+delta, 100)
    alpharange_list = []
    for phi in diff_range:
        alphas = []
        alpharange, _, _, _ = scatteringrate2D_f(Lambda, phi, pol, k0, neff, nsil, w0, dng)
        alpharange_list.append(alpharange * grating_lengths[0])
    plt.plot(np.degrees(diff_range), alpharange_list, label=f'Diffraction angle {angle}')
    
plt.xlabel('Grating Tilt (Degrees)')
plt.ylabel('Reflectivity')
plt.show()  
    
    
    
    
    
    
    
    
    
    
    
    