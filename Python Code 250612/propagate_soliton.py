#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 10:59:34 2025

@author: joel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 14:02:09 2025

@author: joel
"""


import numpy as np
import matplotlib.pyplot as plt
from silica_my import silica_n


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
xv = np.linspace(-60e-6, 60e-6, 20000)  #  create grid in the grating plane

# tilted Gaussian target beam
ntar = np.array([0, 1]) / np.linalg.norm([0, 1])
w2 = 3e-6
zfoc = 26e-6
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
p = 1  
#E2 = E20 * np.sqrt(w2 / w2z) * np.exp(-(xxr / w2z) ** (2*p)) * np.exp(1j * (k0 * nsil * zzr + k0 * nsil * xxr**2 * Rzi / 2 - eta))


def sech(x):
    return 1/np.cosh(x)
soliton_width = 2e-6  # width parameter
amplitude = 0.1       # amplitude

nx = len(xv)
x = xv[:nx]
# Create initial field (1D fundamental soliton profile)
E0_soliton = amplitude * sech(x/soliton_width)
E0 = E0_soliton[:nx]  

dx = x[1] - x[0]
dkx = 2 * np.pi / (nx * dx)
kx = np.arange(-nx/2, nx/2) * dkx

kz = np.zeros_like(kx, dtype=complex)
mask_real = (nsil**2 * k0**2) >= (kx**2)
kz[mask_real] = np.sqrt((nsil**2 * k0**2) - kx[mask_real]**2)
kz[~mask_real] = 1j * np.sqrt(kx[~mask_real]**2 - (nsil**2 * k0**2))

z = np.arange(0, 80) * 1e-6  
nz = len(z)
#mid_index = nz // 2  # Insert the lens halfway
mid_index = nz // 3

#  z = 0 to z = z[mid_index]
E0k = np.fft.fftshift(np.fft.fft(np.fft.fftshift(E0)))
Etot_first = np.zeros((mid_index + 1, nx), dtype=complex)
Etot_first[0, :] = E0

for i in range(1, mid_index + 1):
    Ek = E0k * np.exp(1j * kz * z[i])
    E = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Ek)))
    Etot_first[i, :] = E

# Field at the lens plane (z = z[mid_index])
E_mid = Etot_first[-1, :]
n2 = 2.7e-20

f = 30e-6  
# lens phase factor:
lens_phase = np.exp(-1j * nsil * k0 / (2 * f) * x**2)
E_mid_lensed = E_mid * lens_phase

# z = z[mid_index] to z = z_final
E_mid_lensed_k = np.fft.fftshift(np.fft.fft(np.fft.fftshift(E_mid_lensed)))
z2 = z[mid_index:]
z2_rel = z2 - z2[0]  
n2 = len(z2)

Etot_second = np.zeros((n2, nx), dtype=complex)
Etot_second[0, :] = E_mid_lensed

for ii in range(1, n2):
    Ek = E_mid_lensed_k * np.exp(1j * kz * z2_rel[ii])
    E = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Ek)))
    Etot_second[ii, :] = E

Etot_total = np.vstack((Etot_first, Etot_second[1:, :]))
z_total = np.concatenate((z[:mid_index+1], z2[1:]))

plt.figure(figsize=(8, 6))
plt.pcolormesh(x, z_total, np.abs(Etot_total)**2, shading='auto', cmap='jet')
plt.xlabel('x (m)')
plt.ylabel('z (m)')
plt.title('Propagation with a Lens Inserted at z = {:.1e} m'.format(z[mid_index]))
plt.colorbar(label='Intensity (a.u.)')
plt.show()

z_focus_target = zfoc + f 
idx_focus = np.argmin(np.abs(z_total - z_focus_target))

idx_initial = 2  
idx_lens = mid_index  

selected_indices = [idx_initial, idx_lens, 50]
selected_z_values = [z_total[i] for i in selected_indices]
max_intensity_at_each_z = np.max(np.abs(Etot_total)**2, axis=1) # intensity values along z 
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

for j, idx in enumerate(selected_indices):
    intensity_profile = np.abs(Etot_total[idx, :])**2
    half_max = np.max(intensity_profile) / 2.0
    indices_above_half = np.where(intensity_profile >= half_max)[0]
    fwhm_val = x[indices_above_half[-1]] - x[indices_above_half[0]]
    axs[j].plot(x*1e6, intensity_profile, 'b-', linewidth=2)
    axs[j].axhline(half_max, color='r', linestyle='--', label='Half-Maximum')
    axs[j].axvline(x[indices_above_half[0]]*1e6, color='g', linestyle='--', label='FWHM boundaries')
    axs[j].axvline(x[indices_above_half[-1]]*1e6, color='g', linestyle='--')
    
    axs[j].set_xlabel('x (µm)', fontsize=12)
    axs[j].set_ylabel('Intensity (a.u.)', fontsize=12)
    axs[j].grid(True)
    axs[j].set_title('z = {:.1e} m\nFWHM = {:.1e} m'.format(z_total[idx], fwhm_val), fontsize=12)
    axs[j].legend()
    print("Max intensity at", z_total[idx], "with value",  max_intensity_at_each_z[idx])
plt.tight_layout()
plt.show()

# =============================================================================
# import matplotlib.animation as animation
# fig, ax = plt.subplots(figsize=(8, 6))
# line, = ax.plot([], [], 'r-', lw=2)
# ax.set_xlim(x[0]*1e6, x[-1]*1e6)
# ax.set_ylim(0, np.max(np.abs(Etot_total)**2)*1.1)
# ax.set_xlabel('x (µm)')
# ax.set_ylabel('Intensity (a.u.)')
# title = ax.set_title('Intensity Profile at z = {:.1e} m'.format(z_total[0]))
# 
# def init():
#     line.set_data([], [])
#     return line,
# 
# def animate(i):
#     intensity_profile = np.abs(Etot_total[i, :])**2
#     line.set_data(x*1e6, intensity_profile)
#     title.set_text('Intensity Profile at z = {:.1e} m'.format(z_total[i]))
#     return line,
# 
# ani = animation.FuncAnimation(fig, animate, frames=len(z_total),
#                               init_func=init, blit=False, interval=50)
# plt.show()
# =============================================================================
# =============================================================================
# plt.figure(figsize=(8, 6))
# # Plot lines of constant phase (wavefronts)
# num_levels = 20
# plt.contour(x, z_total, np.angle(Etot_total), levels=np.linspace(-np.pi, np.pi, num_levels), 
#             colors='white', alpha=0.8, linewidths=0.5)
# # Background intensity
# plt.pcolormesh(x, z_total, np.abs(Etot_total)**2, shading='auto')
# plt.xlabel('x (m)')
# plt.ylabel('z (m)')
# plt.title('Intensity with Wavefront Lines')
# plt.colorbar(label='Intensity (a.u.)')
# plt.show()
# =============================================================================

plt.figure(figsize=(8, 6))
phase_data = np.angle(Etot_total)  # Extract phase information (-π to π)
plt.pcolormesh(x, z_total, phase_data, cmap='twilight', shading='auto')
plt.xlabel('x (m)')
plt.ylabel('z (m)')
plt.title('Phase Evolution During Propagation')
plt.colorbar(label='Phase (rad)')
plt.show()

