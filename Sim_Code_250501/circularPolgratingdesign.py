#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 15:18:59 2025

@author: joel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Circular Polarization Generation Using Dual Optimized Gratings

This script:
  1. Defines a target intensity distribution and target location.
  2. Designs a grating optimized for TE polarization and computes the TE scattered field.
  3. Designs a separate grating optimized for TM polarization by specifying a target 
     field that is shifted by 90° relative to TE.
  4. At the target location, extracts the scattered fields from each design.
  5. Scales the TM field (if necessary) so that its amplitude matches the TE field.
  6. Simulates the time evolution of the combined fields (TE along y and TM along z) 
     and plots the resulting polarization ellipse and time evolution curves.
     
For circular polarization, the TE and TM fields should have equal amplitude and a 90° phase difference.
  
@author: joel (adapted)
@date: Feb 17 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz, trapz
from silica_my import silica_n
from scatteringratetest import scatteringratetest   # Use the updated scattering function

#==============================================================================
# 1. Define Target Intensity Distribution and Target Location
#==============================================================================
lam = 780e-9                # wavelength [m]
k0 = 2 * np.pi / lam        # free-space propagation constant [1/m]

# Mode parameters from nslabmodes.m
neff_TE = 1.455879904809160    # effective index for TE mode
w0_TE   = 2.145549895305066e-6  # TE mode waist
neff_TM = 1.455869613276399     # effective index for TM mode
w0_TM   = 2.155888764131035e-6  # TM mode waist

# Target parameters:
phitar = 135 * np.pi / 180    # target propagation angle [rad]
dtar = 0.01                 # distance from grating center to target [m]
efficiency = 0.9            # desired output efficiency

# Define grating coordinate along z (this is our design coordinate)
zv = np.arange(-1, 1.01, 0.01) * 1e-3  # z from -1e-3 m to +1e-3 m

wtar = 300e-6  # waist of the target beam [m]
# Define target amplitude profile (square-root of intensity)
Itar_amplitude = np.exp(-(zv / wtar)**2)
# Normalize so that total target intensity matches efficiency:
norm_factor = np.sqrt(efficiency / trapz(Itar_amplitude**2, zv))
Itar_amplitude = Itar_amplitude * norm_factor

# For the TE target, the target field is simply:
# E_target_TE(z) = Itar_amplitude(z) * exp(i * propagation_phase(z))
# Calculate propagation phase: assume propagation in silica (n_silica = silica_n(lam))
n_silica = silica_n(lam)
propagation_phase = n_silica * k0 * np.abs(zv - dtar)  # simple model: phase ~ k_silica * distance
E_target_TE = Itar_amplitude * np.exp(1j * propagation_phase)

# For the TM target, we want an extra 90° phase shift:
target_phase_offset = np.pi/2
E_target_TM = Itar_amplitude * np.exp(1j * (propagation_phase + target_phase_offset))

# For design, we are mostly using the intensity profiles (|E_target|^2), so:
Itar_TE = np.abs(E_target_TE)**2  # should equal Itar_amplitude^2
Itar_TM = np.abs(E_target_TM)**2  # same intensity profile

# Plot target amplitudes and phases for verification:
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(zv*1e3, Itar_amplitude, 'b-', label='Target Amplitude')
plt.xlabel('z (mm)')
plt.ylabel('Amplitude')
plt.title('Target Field Amplitude')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(zv*1e3, np.angle(E_target_TE), 'b-', label='TE Target Phase')
plt.plot(zv*1e3, np.angle(E_target_TM), 'r--', label='TM Target Phase')
plt.plot(zv*1e3, np.angle(E_target_TM)-np.angle(E_target_TE), 'g-.', label='Phase Difference')
plt.xlabel('z (mm)')
plt.ylabel('Phase (rad)')
plt.title('Target Field Phases')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Define target point in the grating plane (for phase matching)
ztar = -dtar * np.cos(phitar)
ytar = dtar * np.sin(phitar)

#==============================================================================
# 2. Design TE-Optimized Grating and Compute TE Scattered Field
#==============================================================================
pol = 1  # TE polarization for design
neff = neff_TE
w0 = w0_TE
beta = neff * k0   # guided mode propagation constant
dng = 1            # grating index contrast (design value)
nsil = silica_n(lam)

# Compute required emission angle at each z so that light from that z reaches the target:
phiz = np.arccos( -(ztar - zv) / np.sqrt((ztar - zv)**2 + ytar**2) )

# Compute free-space z-component of the wavevector (in the cladding):
kz = -np.cos(phiz) * nsil * k0

# Phase matching: grating must supply extra momentum: kgrat = beta - kz.
kgrat = beta - kz
Lambda_TE = 2 * np.pi / kgrat  # local grating period for TE

# Choose grating tilt angle (assumed constant over z, typically half the target angle):
theta_TE = phitar/2 + 0*zv

# Compute TE scattering fields using the scattering function (for pol=1):
al_TE, Ex_TE, Ey_TE, Ez_TE = scatteringratetest(Lambda_TE, theta_TE, 1, k0, neff, nsil, w0, dng)
alTE = al_TE.copy()

# Compute required local grating modulation (dngv) so that the integrated scattered intensity matches the TE target:
cumItar_TE = cumtrapz(Itar_TE, zv, initial=0)
denom_TE = 1 - cumItar_TE
dngv_TE = np.sqrt(np.divide(Itar_TE, alTE*denom_TE, out=np.zeros_like(Itar_TE), where=(alTE*denom_TE)>0))

#==============================================================================
# 3. Design TM-Optimized Grating (with 90° Target Phase) and Compute TM Scattered Field
#==============================================================================
pol = 2  # TM polarization for design
neff = neff_TM
w0 = w0_TM
beta = neff * k0   # updated for TM
nsil = silica_n(lam)

# Use the same target spatial profile as TE, but now for TM we want the extra phase (E_target_TM)
# (Intensity profile remains the same.)
phiz_TM = np.arccos( -(ztar - zv) / np.sqrt((ztar - zv)**2 + ytar**2) )
kz_TM = -np.cos(phiz_TM) * nsil * k0
kgrat_TM = beta - kz_TM
Lambda_TM = 2 * np.pi / kgrat_TM  # grating period for TM design

# Choose grating tilt for TM (for simplicity, use same as TE):
theta_TM = phitar/2 + 0*zv

# Compute TM scattering fields using scatteringratetest with target_phase_offset = π/2:
al_TM, Ex_TM, Ey_TM, Ez_TM = scatteringratetest(Lambda_TM, theta_TM, 2, k0, neff, nsil, w0, dng, target_phase_offset=target_phase_offset)
alTM = al_TM.copy()

# Compute modulation for TM grating using the TM target intensity profile:
cumItar_TM = cumtrapz(Itar_TM, zv, initial=0)
denom_TM = 1 - cumItar_TM
dngv_TM = np.sqrt(np.divide(Itar_TM, alTM*denom_TM, out=np.zeros_like(Itar_TM), where=(alTM*denom_TM)>0))

#==============================================================================
# 4. Extract Fields at the Target Location and Compare
#==============================================================================
# Choose the representative target location (where the target amplitude is maximum)
idx = Itar_amplitude.argmax()

# From TE-optimized grating, extract TE field (assumed along y):
E_TE_target = Ey_TE[idx]  # complex field from TE grating

# From TM-optimized grating, extract TM field (assumed along z):
E_TM_target = Ez_TM[idx]  # complex field from TM grating (should have extra 90° phase)

# Compute amplitudes and phases:
A_TE = np.abs(E_TE_target)
phi_TE = np.angle(E_TE_target)

A_TM = np.abs(E_TM_target)
phi_TM = np.angle(E_TM_target)

delta_phi = phi_TM - phi_TE

print("At target (z =", zv[idx], "m):")
print("TE (from TE-opt): amplitude = {:.3e}, phase = {:.3f} rad".format(A_TE, phi_TE))
print("TM (from TM-opt): amplitude = {:.3e}, phase = {:.3f} rad, Δphase = {:.3f} rad".format(A_TM, phi_TM, delta_phi))

#==============================================================================
# 5. (Optional) Apply Pump Scaling to TM Field if Needed
#==============================================================================
# If the amplitudes differ, you may want to scale the TM field so that A_TE = A_TM.
# Here we compute a scaling factor.
pump_scaling = A_TE / A_TM if A_TM > 0 else 1.0
E_TM_target_scaled = E_TM_target * pump_scaling
A_TM_scaled = np.abs(E_TM_target_scaled)
phi_TM_scaled = np.angle(E_TM_target_scaled)
delta_phi_scaled = phi_TM_scaled - phi_TE

print("\nAfter TM pump scaling:")
print("Scaled TM amplitude = {:.3e}".format(A_TM_scaled))
print("Scaled relative phase (TM - TE) = {:.3f} rad".format(delta_phi_scaled))

#==============================================================================
# 6. Simulate Time Evolution & Plot Polarization Ellipse
#==============================================================================
omega = 2*np.pi  # normalized angular frequency for demonstration
t = np.linspace(0, 4, 1000)

# Time evolution for TE field (assumed along y):
E_y_t = A_TE * np.cos(omega*t + phi_TE)

# Time evolution for TM field (assumed along z) from TM-opt design:
E_z_t = A_TM_scaled * np.cos(omega*t + phi_TM_scaled)

# Plot time evolution of the two components:
plt.figure(figsize=(8,4))
plt.plot(t, E_y_t, 'b-', label='TE (E_y)')
plt.plot(t, E_z_t, 'r--', label='TM (E_z, scaled)')
plt.xlabel('Normalized Time')
plt.ylabel('Electric Field Amplitude (a.u.)')
plt.title('Time Evolution of TE and TM Fields at Target')
plt.legend()
plt.grid(True)
plt.show()

# Plot polarization ellipse (TE vs. TM):
plt.figure(figsize=(6,6))
plt.plot(E_y_t, E_z_t, 'k-')
plt.xlabel('E_y (TE)')
plt.ylabel('E_z (TM, scaled)')
plt.title('Polarization Ellipse at Target (z = {:.2f} mm)'.format(zv[idx]*1e3))
plt.axis('equal')
plt.grid(True)
plt.show()

