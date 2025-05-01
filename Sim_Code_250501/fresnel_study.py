#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 13:57:33 2025

@author: joel
"""

import numpy as np
import matplotlib.pyplot as plt
from silica_my import silica_n
from Grating_angles_2D import grating_angles_2D_f2
from scatteringrate2D import scatteringrate2D_f
from scipy.integrate import cumtrapz, trapz
from waveguideschematic import schematic

lam = 780e-9        # wavelength [m]
k0 = 2 * np.pi / lam  # vacuum propagation constant [1/m]
c = 3e8             # speed of light [m/s]

# Mode parameters (from nslabmodes.m)
neff_TE = 1.455879904809160   # effective index for TE mode (pol=1)
w0_TE   = 2.145549895305066e-6  # waist of fundamental TE mode
neff_TM = 1.455869613276399   # effective index for TM mode (pol=2)
w0_TM   = 2.155888764131035e-6  # waist of fundamental TM mode

#%% Define target intensity distribution

phitar = 90 * np.pi / 180   # target propagation angle [rad]
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

theta_t = phiz

# For TE:
n_eff_TE_val = neff_TE  # effective index for TE mode
# Compute the incidence angle (inside the waveguide) using Snell's law:
#   n_eff * sin(theta_i) = nsil * sin(theta_t)
theta_i_TE = np.arcsin(np.clip((nsil / n_eff_TE_val) * np.sin(theta_t), -1, 1))
# Fresnel reflection coefficient for TE:
r_TE = (n_eff_TE_val * np.cos(theta_i_TE) - nsil * np.cos(theta_t)) / \
     (n_eff_TE_val * np.cos(theta_i_TE) + nsil * np.cos(theta_t))
#r_TE = - np.sin(theta_i_TE-theta_t) / np.sin(theta_i_TE+theta_t)      
reflected_fraction_TE = np.abs(r_TE)**2

# Plot the reflection fractions vs. grating coordinate (zv)
plt.figure(figsize=(8,6))
plt.plot(zv, reflected_fraction_TE, 'b-', label='TE Reflection Fraction')
#plt.plot(zv, reflected_fraction_TM, 'r--', label='TM Reflection Fraction')
plt.xlabel('z (m)')
plt.ylabel('Reflected Fraction |r|^2')
plt.title('Fresnel Reflection Fraction vs. Grating Coordinate')
plt.legend()
plt.grid(True)
plt.show()


##new stuff 
grating_tilt = phitar/2

# ------------------------------------------------------------------
# Calculate the "Interface Reflection Coefficient" (variable along z)
# ------------------------------------------------------------------
# At each grating slice (position z in zv), the light must be scattered at a local emission angle 
# so that it will hit the target. We assume that this local emission angle, phiz, is given by:
phiz = np.arccos( -( -dtar*np.cos(phitar) - zv ) / np.sqrt(( -dtar*np.cos(phitar) - zv )**2 + (dtar*np.sin(phitar))**2) )
# Explanation: For a given grating slice at position z (along the grating), the horizontal distance 
# from that slice to the target is: (target_x - z). Here we assume the target is located at 
# target_x = -dtar*cos(phitar) (if z = 0 corresponds to grating center). You may need to adjust this 
# formula for your geometry.
#
# For each slice, the emitted (transmitted) angle in the cladding is assumed to be phiz.
# Now, using Snell's law we relate the incidence angle in the core (theta_i) and the emission angle 
# (theta_t = phiz):
theta_t = phiz
theta_i = np.arcsin(np.clip((nsil / neff_TE) * np.sin(theta_t), -1, 1))

# Now compute the TE Fresnel reflection coefficient:
r_TE_interface = (neff_TE * np.cos(theta_i) - nsil * np.cos(theta_t)) / (neff_TE * np.cos(theta_i) + nsil * np.cos(theta_t))
R_TE_interface = np.abs(r_TE_interface)**2  # intensity reflection coefficient

# ------------------------------------------------------------------
# Calculate the "Grating Reflection Coefficient" (constant along z)
# ------------------------------------------------------------------
# This coefficient is computed using the fixed grating tilt, so we set the emission angle equal 
# to grating_tilt.
theta_t_grating = grating_tilt
theta_i_grating = np.arcsin(np.clip((nsil / neff_TE) * np.sin(theta_t_grating), -1, 1))
r_TE_grating = (neff_TE * np.cos(theta_i_grating) - nsil * np.cos(theta_t_grating)) / \
               (neff_TE * np.cos(theta_i_grating) + nsil * np.cos(theta_t_grating))
R_TE_grating = np.abs(r_TE_grating)**2

# ------------------------------------------------------------------
# Plot the results
# ------------------------------------------------------------------
plt.figure(figsize=(10,6))
plt.plot(zv, R_TE_interface, 'b-', label='Interface Reflection Coefficient')
plt.axhline(R_TE_grating, color='r', linestyle='--', label='Grating Reflection Coefficient (constant)')
plt.xlabel('Grating coordinate z (m)')
plt.ylabel('Reflection Fraction |r|^2')
plt.title('Comparison of Reflection Coefficients')
plt.legend()
plt.grid(True)
plt.show()

# Print average values:
print(f"Average Interface Reflection (TE): {np.mean(R_TE_interface):.4e}")
print(f"Grating Reflection (TE): {R_TE_grating:.4e}")