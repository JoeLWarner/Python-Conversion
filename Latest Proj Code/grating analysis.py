#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:51:04 2025

@author: joel
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz, trapz
from silica_my import silica_n
from Grating_angles_2D import grating_angles_2D_f2
from scatteringrate2D import scatteringrate2D_f
from scattertester import scatteringrate2D_rect

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


grating_lengths = [10e-3, 10e-5]
diff_angles = [60, 90, 120]
dng_values = np.linspace(0, 5e-3, 100)

###############
# relfectivity as a function of grating modulation 
###############

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
    if dangle == 60:
        plt.plot(dng_values, alphas, label=f'Diffraction angle {dangle}', marker='d', markersize=3.8)
    else:
        plt.plot(dng_values, alphas, label=f'Diffraction angle {dangle}')
        #plt.axvline(1.2e-3, color='black', linestyle='dashdot')
plt.xlabel(r'Grating index modulation, $\Delta n_g$')
plt.ylabel('Reflectance (%)')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.title('Reflectance vs. Grating Index Modulation')
plt.legend()
#plt.grid(True)
plt.show()


##################
#reflectance over grating lengths 
#################

dng = 1e-3
theta = np.radians(30)  
Lambda = l / (neff * (1 + np.cos(2 * theta)))  
grating_lengths = np.linspace(10e-6, 10e-3, 100)
reflectance_percent = []
for L in grating_lengths:
    alpha, _, _, _ = scatteringrate2D_f(Lambda, theta, pol, k0, neff, nsil, w0, 1.2e-3)
    R = 1 - np.exp(-np.mean(alpha) * L)
    reflectance_percent.append(R * 100)  
# Plot results
plt.figure(figsize=(10, 6))
plt.plot(grating_lengths*1e3, reflectance_percent, 'b-', linewidth=2)
plt.xlabel("Grating Length (mm)")
plt.ylabel("Reflectance (%)")
plt.title("Reflectance vs. Grating Length, dng = 1.2e-3")
plt.grid(True)
plt.show()

#################
#reflectivity as a function of diffraction angle. 
#################

plt.figure(figsize=(8, 6))
for angle in diff_angles:
    theta = np.radians(angle / 2)
    Lambda = l / (neff * (1 + np.cos(2 * theta)))
    delta = np.radians(10)
    diff_range = np.linspace(theta-delta, theta+delta, 100)
    alpharange_list = []
    for phi in diff_range:
        alpharange, _, _, _ = scatteringrate2D_f(Lambda, phi, pol, k0, neff, nsil, w0, dng)
        alpharange_list.append(alpharange*0.01)
    plt.plot(np.degrees(diff_range), alpharange_list, label=f'Diffraction angle {angle}')
    
plt.xlabel('Grating Tilt (Degrees)')
plt.ylabel('Reflectivity')
plt.legend()
plt.show()  

####################
#reflectance when wavelengths are changed and grating is optimised 
####################

neff_values = [1.5396167978018032, 1.459147, 1.455880, 1.453487, 1.4449355552629308]
waist_values = [9.484380e-07, 1.832930e-06, 2.145548e-06, 2.456798e-06, 3.983035e-06]
wave_range = [213e-9, 650e-9, 780e-9, 910e-9, 1550e-9]
op_value = [1, 0.7, 0.3, 0.9, 1.0]
colours = ['black', 'red', 'blue', 'lime', 'red']
plt.figure(figsize=(8, 6))

for i, l in enumerate(wave_range):
    k0 = 2 * np.pi / l
    nsil = silica_n(l)
    for angle in diff_angles:
        theta = np.radians(angle / 2)
        neff = neff_values[i]
        Lambda = l / (neff * (1 + np.cos(2 * theta)))
        delta = np.radians(10)
        diff_range = np.linspace(theta - delta, theta + delta, 100)
        alpharange_list = []
        for phi in diff_range:
            alpharange, _, _, _ = scatteringrate2D_f(Lambda, phi, 1, k0, neff, nsil, waist_values[i], dng)
            alpharange_list.append(alpharange * 0.01)
        plt.plot(np.degrees(diff_range), alpharange_list, 
                 label=f'{angle}°, λ = {l*1e9:.0f} nm', 
                 alpha=op_value[i], 
                 color=colours[i])

plt.xlabel('Grating Tilt (Degrees)')
plt.ylabel('Reflectivity')
plt.grid(True)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
plt.show()
    


#####################
# measuring reflectance when the wavelength changes for a grating optimised for a certain wavelength
#####################
lam_design = 780e-9               
neff_design = 1.455880              
waist_design = 2.145548e-6           
wavelengths = np.linspace(650e-9, 910e-9, 100)
diff_angles = [60, 90, 120]

plt.figure(figsize=(8,6))
for angle in diff_angles:
    theta = np.radians(angle / 2)
    # grating period optimized for 780nm
    Lambda = lam_design / (neff_design * (1 + np.cos(2 * theta)))
    reflectances = []
    for l in wavelengths:
        k0 = 2 * np.pi / l
        nsil = silica_n(l) 
        #k0 and nsil are varied but the grating params are kept as if wavelength was 780nm 
        alpha, _, _, _ = scatteringrate2D_f(Lambda, theta, pol, k0, neff_design, nsil, waist_design, dng)
        R = 1 - np.exp(-alpha * 0.0001)
        reflectances.append(R * 100)
        
    plt.plot(wavelengths * 1e9, reflectances, label=f'Diffraction angle {angle}°')

plt.axvline(780, linestyle = 'dashdot', color = 'black')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance (%)')
plt.title('Reflectance vs. Wavelength for a Grating Optimized at 780 nm')
plt.legend()
plt.grid(True)
plt.show()
    
###################
#reflectance on grating period and emission angle
###################
grating_period_range = np.linspace(300e-9, 2000e-9, 1701)

fixedg_alphas = []
faTM = []
phi_list = []
k0 = 2 * np.pi / l
for period in grating_period_range:
    kgrat = 2*np.pi / period 
    kx = beta - kgrat
    phi = np.pi - np.arccos(kx / (nsil*k0))
    phi_list.append(phi)
    theta = phi/2
    afg, _, _, _ = scatteringrate2D_f(period, theta, 1, k0, neff, nsil, w0, dng)
    afgTM, _, _, _ = scatteringrate2D_f(period, theta, 2, k0, neff_TM, nsil, w0_TM, dng)
    fixedg_alphas.append(afg)
    faTM.append(afgTM)
plt.plot(grating_period_range, fixedg_alphas, label='TE reflectance')
plt.plot(grating_period_range, faTM, label='TM reflectance')
plt.xlabel('Grating period (m)')
plt.ylabel('Reflectance (1/m)')
plt.legend()
plt.show() 

plt.plot(np.degrees(phi_list), fixedg_alphas, label='TE')
plt.plot(np.degrees(phi_list), faTM, label='TM')
plt.xlabel('Emission Angle (degs)')
plt.ylabel('Reflectance (1/m)')
plt.legend()
plt.show()


####################    
##relfection on waveguide width changed by changing the gaussian width
####################

w_values = np.linspace(0.1e-6, 6e-6, 100)
ldom = 213e-9
nsildom = silica_n(ldom)
for angle in diff_angles:
    theta = np.radians(angle / 2)
    Lambda = l / (1.5396167978018032 * (1 + np.cos(2 * theta)))
    width_ref = []
    for width in w_values:
        al1, _, _, _ = scatteringrate2D_f(Lambda, theta, pol, k0, 1.5396167978018032, nsildom, width, 1e-3)
        R = 1 - np.exp(-al1 * 100e-6)
        width_ref.append(R) 
    plt.plot(w_values, width_ref, label=f'Tilt {angle/2}')

plt.xlabel('Waveguide Width w0 (µm)')
plt.ylabel('Reflectance')
plt.title('Reflectance vs waveguide width')
plt.legend()
plt.show()



    










    
    