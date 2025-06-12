#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 14:02:09 2025
@author: joel
"""
import numpy as np
import matplotlib.pyplot as plt
from silica_my import silica_n

# Common parameters
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

# Grid setup
xv = np.linspace(-60e-6, 60e-6, 20000)  # Create grid in the grating plane

# Tilted Gaussian target beam parameters
ntar = np.array([0, 1]) / np.linalg.norm([0, 1])
w2 = 3e-6  # Beam waist
zfoc = 30e-6  # Focus position
E20 = 0.1

# Align reference frame with the direction of the gaussian propagation 
zzr = xv * ntar[0]  # z in rotated frame
xxr = np.sqrt(xv**2 - zzr**2)  # x in rotated frame
zzr -= zfoc

# Defining gaussian properties
zR = np.pi * w2**2 / (l) * nsil  # Rayleigh range
w2z = w2 * np.sqrt(1 + (zzr / zR)**2)  # Beam width at z
eta = 0.5 * np.arctan(zzr / zR) * 0.5  # Gouy phase
Rzi = zzr / (zzr**2 + zR**2)  # Inverse radius of curvature

# Super Gaussian field
p = 1  # Gaussian order (p=1 is standard Gaussian)
E2 = E20 * np.sqrt(w2 / w2z) * np.exp(-(xxr / w2z) ** (2*p)) * np.exp(1j * (k0 * nsil * zzr + k0 * nsil * xxr**2 * Rzi / 2 - eta))

# Extract field for propagation
nx = len(xv)
x = xv[:nx]
E0 = E2[:nx]
dx = x[1] - x[0]

# Angular spectrum method setup
dkx = 2 * np.pi / (nx * dx)
kx = np.arange(-nx/2, nx/2) * dkx
kz = np.zeros_like(kx, dtype=complex)
mask_real = (nsil**2 * k0**2) >= (kx**2)
kz[mask_real] = np.sqrt((nsil**2 * k0**2) - kx[mask_real]**2)
kz[~mask_real] = 1j * np.sqrt(kx[~mask_real]**2 - (nsil**2 * k0**2))

# Z-axis setup for propagation
z = np.arange(0, 80) * 1e-6
nz = len(z)
mid_index = 25  # Lens position index

# Lens parameters
f = 35e-6  # Focal length

# ANGULAR SPECTRUM METHOD PROPAGATION
# First part: from z=0 to lens plane
E0k = np.fft.fftshift(np.fft.fft(np.fft.fftshift(E0)))
Etot_first = np.zeros((mid_index + 1, nx), dtype=complex)
Etot_first[0, :] = E0

for i in range(1, mid_index + 1):
    Ek = E0k * np.exp(1j * kz * z[i])
    E = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Ek)))
    Etot_first[i, :] = E

# Field at the lens plane
E_mid = Etot_first[-1, :]
lens_phase = np.exp(-1j * nsil * k0 / (2 * f) * x**2)
E_mid_lensed = E_mid * lens_phase

# Second part: from lens plane to end
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

# Complete propagation result
Etot_total = np.vstack((Etot_first, Etot_second[1:, :]))
z_total = np.concatenate((z[:mid_index+1], z2[1:]))

# ABCD MATRIX METHOD IMPLEMENTATION
def q_parameter(z, w0, wavelength, n):
    """Calculate q parameter at position z for a Gaussian beam"""
    zR = np.pi * w0**2 * n / wavelength
    return z + 1j * zR

def propagate_q(q, ABCD_matrix):
    """Propagate q parameter through an ABCD system"""
    A = ABCD_matrix[0, 0]
    B = ABCD_matrix[0, 1]
    C = ABCD_matrix[1, 0]
    D = ABCD_matrix[1, 1]
    return (A*q + B) / (C*q + D)

def beam_width_from_q(q, wavelength, n):
    """Calculate beam width from q parameter"""
    return np.sqrt(-wavelength / (np.pi * n * np.imag(1/q)))

def intensity_from_q(x, q, amplitude, wavelength, n):
    """Calculate Gaussian beam intensity from q parameter"""
    w = beam_width_from_q(q, wavelength, n)
    w0_q = beam_width_from_q(q_parameter(0, w2, wavelength, n), wavelength, n)
    
    # Calculate radius of curvature
    if np.real(q) == 0:
        R = float('inf')
    else:
        R = np.real(q) / (np.real(q)**2 + np.imag(q)**2)
    
    k = 2 * np.pi * n / wavelength
    
    # Phase components
    if R == float('inf'):
        curvature_phase = 0
    else:
        curvature_phase = k * x**2 / (2 * R)
    
    gouy_phase = np.arctan2(np.real(q), np.imag(q))
    
    # Complex field
    field = amplitude * (w0_q / w) * np.exp(-(x/w)**2) * np.exp(1j * (curvature_phase - gouy_phase))
    return field

# Setup for ABCD propagation
wavelength = l
z_abcd = z_total  # Use same z positions as angular spectrum method
n_points = 1000  # Number of x points for ABCD calculation (can be different from angular spectrum)
x_abcd = np.linspace(-60e-6, 60e-6, n_points)

# Initial q parameter at z=0 (beam waist)
# Adjust initial q parameter to match your beam setup
initial_q = q_parameter(-zfoc, w2, wavelength, nsil)

# Define ABCD matrices for free space and lens
def free_space_matrix(d, n):
    """ABCD matrix for free space propagation distance d in medium with index n"""
    return np.array([[1, d/n], [0, 1]])

def thin_lens_matrix(f):
    """ABCD matrix for a thin lens with focal length f"""
    return np.array([[1, 0], [-1/f, 1]])

# Calculate beam parameters at each z position
Etot_abcd = np.zeros((len(z_abcd), len(x_abcd)), dtype=complex)

# Initial field
q = initial_q
Etot_abcd[0, :] = intensity_from_q(x_abcd, q, E20, wavelength, nsil)

# Propagate step by step
lens_z = z[mid_index]  # Lens position
for i in range(1, len(z_abcd)):
    # Distance since last step
    dz = z_abcd[i] - z_abcd[i-1]
    
    # Check if we're crossing the lens
    if z_abcd[i-1] < lens_z and z_abcd[i] >= lens_z:
        # Propagate to lens
        pre_lens_dist = lens_z - z_abcd[i-1]
        abcd_to_lens = free_space_matrix(pre_lens_dist, nsil)
        q = propagate_q(q, abcd_to_lens)
        
        # Apply lens
        abcd_lens = thin_lens_matrix(f)
        q = propagate_q(q, abcd_lens)
        
        # Propagate rest of the way
        post_lens_dist = z_abcd[i] - lens_z
        abcd_after_lens = free_space_matrix(post_lens_dist, nsil)
        q = propagate_q(q, abcd_after_lens)
    else:
        # Regular propagation
        abcd = free_space_matrix(dz, nsil)
        q = propagate_q(q, abcd)
    
    # Calculate field at this position
    Etot_abcd[i, :] = intensity_from_q(x_abcd, q, E20, wavelength, nsil)

# Visualization
plt.figure(figsize=(12, 10))

# Plot 1: Angular Spectrum Method
plt.subplot(2, 2, 1)
plt.pcolormesh(x, z_total, np.abs(Etot_total)**2, shading='auto', cmap='jet')
plt.axhline(y=z[mid_index], color='white', linestyle='--', label='Lens Position')
plt.axhline(y=zfoc, color='red', linestyle='--', label='Initial Focus')
plt.xlabel('x (m)')
plt.ylabel('z (m)')
plt.title('Angular Spectrum Method')
plt.colorbar(label='Intensity (a.u.)')
#plt.legend()

# Plot 2: ABCD Matrix Method
plt.subplot(2, 2, 2)
plt.pcolormesh(x_abcd, z_abcd, np.abs(Etot_abcd)**2, shading='auto', cmap='jet')
plt.axhline(y=z[mid_index], color='white', linestyle='--', label='Lens Position')
plt.axhline(y=zfoc, color='red', linestyle='--', label='Initial Focus')
plt.xlabel('x (m)')
plt.ylabel('z (m)')
plt.title('ABCD Matrix Method')
plt.colorbar(label='Intensity (a.u.)')
#plt.legend()

plt.tight_layout()
#plt.savefig('gaussian_propagation_comparison.png', dpi=300)
plt.show()
plt.pcolormesh(x, z_total, np.angle(Etot_total), shading='auto', cmap='twilight')
plt.colorbar(label='Phase (rad)')
plt.show()

plt.pcolormesh(x_abcd, z_abcd, np.angle(Etot_abcd), shading='auto', cmap='twilight')
plt.colorbar(label='Phase (rad)')
plt.show()
# =============================================================================
# # Quantitative comparison at specific planes
# z_compare_indices = [0, mid_index, -1]  # Start, lens, end
# z_labels = ['Initial plane', 'Lens plane', 'Final plane']
# 
# plt.figure(figsize=(15, 5))
# for i, (idx, label) in enumerate(zip(z_compare_indices, z_labels)):
#     plt.subplot(1, 3, i+1)
#     
#     # Normalize for better comparison
#     as_intensity = np.abs(Etot_total[idx])**2
#     as_intensity = as_intensity / np.max(as_intensity)
#     
#     # For ABCD, we need to interpolate to match x coordinates
#     abcd_intensity = np.abs(Etot_abcd[idx])**2
#     abcd_intensity = abcd_intensity / np.max(abcd_intensity)
#     abcd_interp = np.interp(x, x_abcd, abcd_intensity)
#     
#     plt.plot(x*1e6, as_intensity, 'b-', label='Angular Spectrum')
#     plt.plot(x*1e6, abcd_interp, 'r--', label='ABCD Matrix')
#     plt.xlabel('x (μm)')
#     plt.ylabel('Normalized Intensity')
#     plt.title(label)
#     plt.legend()
#     plt.grid(True)
# 
# plt.tight_layout()
# #plt.savefig('gaussian_propagation_comparison_slices.png', dpi=300)
# plt.show()
# =============================================================================
