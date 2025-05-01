#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 10:10:29 2025

@author: joel
"""

import numpy as np
import matplotlib.pyplot as plt
from silica_my import silica_n
from Grating_angles_2D import grating_angles_2D_f2
from scatteringrate2D import scatteringrate2D_f
from scipy.integrate import cumtrapz, trapz
from waveguideschematic import schematic

lam = 780e-9        
k0 = 2 * np.pi / lam
c = 3e8             

neff_TE = 1.455879904809160   
w0_TE   = 2.145549895305066e-6 
neff_TM = 1.455869613276399   
w0_TM   = 2.155888764131035e-6 

#%% Define target intensity distribution

phitar = 90 * np.pi / 180   
dtar = 0.01 #[m]
efficiency = 0.9            # fraction of pump power converted into output beam

zv = np.arange(-1, 1.01, 0.01) * 0.001  # from -1e-3 to 1e-3 [m]

wtar = 300e-6  # waist of emitted beam [m]
Itar = np.exp(-(zv / wtar) ** 2)  # intensity profile 
Itar = Itar * efficiency / trapz(Itar, zv)

plt.plot(zv, Itar, 'b-')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel('z (m)')
plt.ylabel('Emitted field intensity (TE, target)')
plt.title(f'Efficiency: {efficiency}   Input power: 1')
plt.grid(True)

if 1==1:
    fig, ax = schematic(phitar)
    plt.show()
#%% Design of grating for TE pump

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
kz = -np.cos(phiz) * nsil * k0
# The grating must provide the difference between the mode propagation constant and k_z:
kgrat = beta - kz
Lambda = 2 * np.pi / kgrat  

theta = phitar / 2  + 0 * zv  

al, Ex, Ey, Ez = scatteringrate2D_f(Lambda, theta, pol, k0, neff, nsil, w0, dng)
alTE = al.copy()  
cumItar = cumtrapz(Itar, zv, initial=0)
denom = 1 - cumItar
dngv = np.sqrt( np.divide(Itar, al * denom, out=np.zeros_like(Itar), where=(al*denom)>0) )

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


def calculate_reflection_percentage(r_fresnel):
    reflectance_percent = np.abs(r_fresnel)**2 * 100
    return reflectance_percent


#% Calculate Fresnel reflection at core-cladding interface

#n_core = neff    
#n_clad = nsil    

n_core = 1.5    
n_clad = 1    

theta_inc = np.pi/2 - phiz

if pol == 1:  
    cos_theta_t = np.sqrt(1 - (n_core/n_clad * np.sin(theta_inc))**2)
    r_fresnel = np.where(
        np.abs(np.sin(theta_inc)) <= n_clad/n_core,  
        (n_core*np.cos(theta_inc) - n_clad*cos_theta_t) / (n_core*np.cos(theta_inc) + n_clad*cos_theta_t),
        1.0  
    )
else:  
    cos_theta_t = np.sqrt(1 - (n_core/n_clad * np.sin(theta_inc))**2)
    r_fresnel = np.where(
        np.abs(np.sin(theta_inc)) <= n_clad/n_core,  
        (n_clad*np.cos(theta_inc) - n_core*cos_theta_t) / (n_clad*np.cos(theta_inc) + n_core*cos_theta_t),
        1.0  
    )


T_fresnel = 1 - np.abs(r_fresnel)**2

plt.figure()
plt.plot(zv, np.abs(r_fresnel)**2)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel('z (m)')
plt.ylabel('Fresnel Reflection r_s')
plt.grid(True)

# =============================================================================
# 
# fig, axs = plt.subplots(2, sharex=True)
# axs[0].plot(zv, np.abs(r_fresnel)**2)
# axs[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# axs[0].set_ylabel('Reflection coefficient')
# axs[0].grid(True)
# 
# axs[1].plot(zv, T_fresnel)
# axs[1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# axs[1].set_xlabel('z (m)')
# axs[1].set_ylabel('Transmission coefficient')
# axs[1].grid(True)
# 
# plt.tight_layout()
# =============================================================================

# adjust grating design to account for Fresnel losses
# The effective output intensity needs to be boosted by 1/T_fresnel to compensate for the Fresnel losses

# Adjust target intensity profile to account for Fresnel transmission
#Itar_adjusted = Itar / T_fresnel
# Renormalize to maintain the overall efficiency
#Itar_adjusted = Itar_adjusted * efficiency / trapz(Itar_adjusted, zv)

# Recalculate the grating index modulation
# =============================================================================
# cumItar_adjusted = cumtrapz(Itar_adjusted, zv, initial=0)
# denom_adjusted = 1 - cumItar_adjusted
# dngv_adjusted = np.sqrt(np.divide(Itar_adjusted, al * denom_adjusted, 
#                                 out=np.zeros_like(Itar_adjusted), 
#                                 where=(al*denom_adjusted)>0))
# =============================================================================

Itar_Transmitted = Itar*T_fresnel


plt.figure()
plt.plot(zv, Itar, 'b-', label='Without')
plt.plot(zv, Itar_Transmitted, 'r-', label='With reflection')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel('z (m)')
plt.ylabel('Intensity')
plt.title('Target intensity with and without reflection')
plt.legend()
plt.grid(True)

# =============================================================================
# plt.figure()
# plt.plot(zv, dngv, 'b-', label='Original dng')
# #plt.plot(zv, dngv_adjusted, 'r-', label='Adjusted dng')
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# plt.xlabel('z (m)')
# plt.ylabel('dng')
# plt.title('Grating index modulation adjustment')
# plt.legend()
# plt.grid(True)
# 
# =============================================================================
reflection_percentage = calculate_reflection_percentage(r_fresnel)

mean_reflection = np.mean(reflection_percentage)
max_reflection = np.max(reflection_percentage)
min_reflection = np.min(reflection_percentage)

plt.figure()
plt.plot(np.degrees(theta_inc), reflection_percentage)
plt.title('Reflection percentage against incident angle')
plt.xlabel('Incident angle')
plt.ylabel('Reflection Percentage (%)')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.grid()

plt.figure(figsize=(10, 6))
plt.plot(zv, reflection_percentage)
plt.title(f'Fresnel Reflection Percentage\n'
          f'Mean: {mean_reflection:.4f}%  '
          f'Max: {max_reflection:.4f}%  '
          f'Min: {min_reflection:.4f}%')
plt.xlabel('z (m)')
plt.ylabel('Reflection Percentage (%)')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.grid(True)

print("Reflection Percentage Analysis:")
print(f"Mean Reflection: {mean_reflection:.4f}%")
print(f"Maximum Reflection: {max_reflection:.4f}%")
print(f"Minimum Reflection: {min_reflection:.4f}%")

#how much does the angle change after scatrering 
theta_trans = np.arcsin(np.clip((n_core/n_clad) * np.sin(theta_inc), -1, 1))
#convert back tp phi reference 
phi_trans = np.pi/2 - theta_trans

angle_change = phi_trans - phiz   # positive means the transmitted beam is more vertical

plt.figure(figsize=(10, 6))
plt.plot(zv*1e3, np.degrees(phiz), 'b-', label='Original Scattered Angle (phiz)')
plt.plot(zv*1e3, np.degrees(phi_trans), 'r--', label='Transmitted Angle in Cladding (phi_trans)')
plt.xlabel('Grating coordinate z (mm)')
plt.ylabel('Angle (degrees)')
plt.title('Comparison of Scattered vs. Transmitted Angles')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(zv*1e3, np.degrees(angle_change), 'k-')
plt.xlabel('Grating coordinate z (mm)')
plt.ylabel('Angle change (degrees)')
plt.title('Difference: Transmitted Angle - Scattered Angle')
plt.grid(True)
plt.show()

