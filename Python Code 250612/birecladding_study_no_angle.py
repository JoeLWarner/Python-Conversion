#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 14:49:06 2025

@author: joel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physically accurate model of propagation through a waveguide with birefringent cladding
"""

import numpy as np
import matplotlib.pyplot as plt
from silica_my import silica_n

# --------------------------------------------------------
# 0) BASIC PARAMETERS
# --------------------------------------------------------
lam    = 780e-9
k0     = 2*np.pi/lam
nsil   = silica_n(lam)
n_air  = 1.0

# Mode waists from your slab‐mode solver
w0_TE  = 2.145549895305066e-6
w0_TM  = 2.155888764131035e-6

# Common tilted‐beam parameters
ntar   = np.array([0.2,1])/np.linalg.norm([0.2,1])
zfoc   = 50e-6     # focal shift in the slab before the boundary
E20    = 0.1
p      = 1        # super‐Gaussian order

# Define birefringent cladding parameters
biref_start = -20e-6   # Position (relative to interface where beam exits the waveguide)
biref_end = -5e-6      # End before reaching the interface to ensure clear field patterns
biref_thickness = abs(biref_end - biref_start)

# Define birefringence - different indices for TE and TM
n_TE_ordinary = nsil  # Ordinary index (same as silica for smooth transition)
delta_n = 0.01        # Birefringence amount
n_TE_extraordinary = n_TE_ordinary + delta_n  # TE sees extraordinary index
n_TM = n_TE_ordinary  # TM sees ordinary index

print(f"Birefringent region thickness: {biref_thickness*1e6:.2f} µm")
print(f"TE index (extraordinary): {n_TE_extraordinary:.5f}")
print(f"TM index (ordinary): {n_TM:.5f}")
print(f"Birefringence (Δn): {delta_n:.5f}")

# Calculate expected phase shift
expected_phase_shift = k0 * delta_n * biref_thickness
expected_phase_degrees = expected_phase_shift * 180/np.pi
print(f"Expected additional phase shift for TE: {expected_phase_shift:.4f} rad ({expected_phase_degrees:.2f}°)")

# Spatial grid
xv     = np.linspace(-60e-6, 60e-6, 4096)
nx     = xv.size
dx     = xv[1]-xv[0]

# Propagation grid
# We'll propagate from deep in the waveguide to beyond the air interface
# First define the waveguide propagation region (negative z values)
z_start = -70e-6    # Start position
z_interface = 0     # Air interface position
z_end = 80e-6       # End position in air

# Create non-uniform z grid with finer resolution in the birefringent region
# This gives better accuracy for the phase evolution
z_before_biref = np.linspace(z_start, biref_start, 30)  # Before birefringent region
z_biref = np.linspace(biref_start, biref_end, 100)      # Birefringent region (higher sampling)
z_after_biref = np.linspace(biref_end, z_interface, 20) # After birefringent region to interface
z_air = np.linspace(z_interface, z_end, 81)             # In air (match original sampling)

# Combine z grids
zprop = np.unique(np.concatenate([z_before_biref, z_biref, z_after_biref, z_air]))
nz = zprop.size

# Indices for regions
idx_biref_start = np.argmin(np.abs(zprop - biref_start))
idx_biref_end = np.argmin(np.abs(zprop - biref_end))
idx_interface = np.argmin(np.abs(zprop - z_interface))

print(f"Total propagation steps: {nz}")
print(f"Birefringent region: steps {idx_biref_start} to {idx_biref_end} (total {idx_biref_end-idx_biref_start} steps)")

# --------------------------------------------------------
# 1) BUILD TE & TM GAUSSIANS AT THE INITIAL POSITION
# --------------------------------------------------------
def build_tilted_gaussian(w2, z_pos):
    """Build a tilted Gaussian at a given z position"""
    # rotate coords (for ntar=[0,1] this makes zzr constant)
    zzr = xv*ntar[0]        # = 0 here
    xxr = np.sqrt(xv**2 - zzr**2)
    
    # shift to put waist at z = zfoc
    # We account for the current z_pos
    rel_z = z_pos - zfoc
    zzr -= rel_z

    # Rayleigh range in silica
    zR = np.pi * w2**2 / lam * nsil
    w2z = w2 * np.sqrt(1 + (zzr/zR)**2)
    eta = 0.5*np.arctan(zzr/zR)*0.5
    Rzi = zzr/(zzr**2 + zR**2)

    E = (E20 * np.sqrt(w2/w2z)
         * np.exp(- (xxr/w2z)**(2*p))
         * np.exp(1j*(k0*nsil*zzr
                     + k0*nsil*(xxr**2)*Rzi/2
                     - eta)))
    return E

# Build initial fields at the starting position
E_TE = build_tilted_gaussian(w0_TE, z_start)
E_TM = build_tilted_gaussian(w0_TM, z_start)

# --------------------------------------------------------
# 2) PROPAGATION FUNCTIONS
# --------------------------------------------------------
def angular_spectrum_step(E_in, dz, n_medium):
    """Propagate field by angular spectrum method for a small step dz"""
    # FFT of input field
    E_k = np.fft.fftshift(np.fft.fft(np.fft.fftshift(E_in)))
    
    # Calculate kx components
    dkx = 2*np.pi/(nx*dx)
    kx = np.arange(-nx/2, nx/2)*dkx
    
    # Calculate kz components in the medium
    kz = np.zeros_like(kx, dtype=complex)
    mask = (n_medium**2 * k0**2) >= kx**2
    kz[mask] = np.sqrt(n_medium**2 * k0**2 - kx[mask]**2)
    kz[~mask] = 1j*np.sqrt(kx[~mask]**2 - n_medium**2 * k0**2)
    
    # Apply propagation phase
    H = np.exp(1j * kz * dz)
    E_k_out = E_k * H
    
    # IFFT to get propagated field
    E_out = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(E_k_out)))
    
    return E_out

def apply_fresnel(E_in_TE, E_in_TM):
    """Apply Fresnel transmission at silica-air interface"""
    # FFT of input fields
    E_k_TE = np.fft.fftshift(np.fft.fft(np.fft.fftshift(E_in_TE)))
    E_k_TM = np.fft.fftshift(np.fft.fft(np.fft.fftshift(E_in_TM)))
    
    # Calculate kx components
    dkx = 2*np.pi/(nx*dx)
    kx = np.arange(-nx/2, nx/2)*dkx
    
    # Calculate kz in silica and air
    kz_sil = np.zeros_like(kx, dtype=complex)
    mask_sil = (nsil**2 * k0**2) >= kx**2
    kz_sil[mask_sil] = np.sqrt(nsil**2 * k0**2 - kx[mask_sil]**2)
    kz_sil[~mask_sil] = 1j*np.sqrt(kx[~mask_sil]**2 - nsil**2 * k0**2)
    
    kz_air = np.zeros_like(kx, dtype=complex)
    mask_air = (n_air**2 * k0**2) >= kx**2
    kz_air[mask_air] = np.sqrt(n_air**2 * k0**2 - kx[mask_air]**2)
    kz_air[~mask_air] = 1j*np.sqrt(kx[~mask_air]**2 - n_air**2 * k0**2)
    
    # Fresnel cosines
    cos1 = kz_sil/(nsil*k0)
    cos2 = kz_air/(n_air*k0)
    
    # Fresnel Ts, Tp
    Ts = 2*nsil*cos1/(nsil*cos1 + n_air*cos2)
    Tp = 2*nsil*cos1/(n_air*cos1 + nsil*cos2)
    
    # Apply Fresnel transmission
    E_k_TE_air = E_k_TE * Ts
    E_k_TM_air = E_k_TM * Tp
    
    return E_k_TE_air, E_k_TM_air

# --------------------------------------------------------
# 3) STEP-BY-STEP PROPAGATION THROUGH ALL REGIONS
# --------------------------------------------------------
# Initialize arrays to store fields at all z positions
Etot_TE = np.zeros((nz, nx), dtype=complex)
Etot_TM = np.zeros((nz, nx), dtype=complex)

# Store initial fields
Etot_TE[0,:] = E_TE
Etot_TM[0,:] = E_TM

# Track current fields
E_TE_current = E_TE.copy()
E_TM_current = E_TM.copy()

# Keep track of accumulated phase difference for monitoring
phase_diff_center = np.zeros(nz)

# Loop through propagation steps
for i in range(1, nz):
    dz = zprop[i] - zprop[i-1]  # Step size
    z_current = zprop[i]
    
    # Check if we're at the interface (apply Fresnel)
    if i-1 < idx_interface <= i:
        # We're crossing the interface - apply Fresnel
        E_k_TE_air, E_k_TM_air = apply_fresnel(E_TE_current, E_TM_current)
        
        # Calculate remaining propagation in air
        remaining_dz = zprop[i] - z_interface
        
        # Propagate in air using angular spectrum
        # First convert back to spatial domain
        E_TE_interface = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(E_k_TE_air)))
        E_TM_interface = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(E_k_TM_air)))
        
        # Then propagate remaining distance
        E_TE_current = angular_spectrum_step(E_TE_interface, remaining_dz, n_air)
        E_TM_current = angular_spectrum_step(E_TM_interface, remaining_dz, n_air)
    
    else:
        # Regular propagation step
        if idx_biref_start <= i <= idx_biref_end:
            # We're in the birefringent region
            # TE sees extraordinary index
            E_TE_current = angular_spectrum_step(E_TE_current, dz, n_TE_extraordinary)
            # TM sees ordinary index
            E_TM_current = angular_spectrum_step(E_TM_current, dz, n_TM)
        elif i <= idx_interface:
            # Regular silica region
            E_TE_current = angular_spectrum_step(E_TE_current, dz, nsil)
            E_TM_current = angular_spectrum_step(E_TM_current, dz, nsil)
        else:
            # Air region (after interface)
            E_TE_current = angular_spectrum_step(E_TE_current, dz, n_air)
            E_TM_current = angular_spectrum_step(E_TM_current, dz, n_air)
    
    # Store current fields
    Etot_TE[i,:] = E_TE_current
    Etot_TM[i,:] = E_TM_current
    
    # Calculate center of beam and phase difference there
    # This helps us track the phase evolution
    intensity_TE = np.abs(E_TE_current)**2
    center_idx = np.argmax(intensity_TE)
    phi_TE = np.angle(E_TE_current[center_idx])
    phi_TM = np.angle(E_TM_current[center_idx])
    phase_diff = (phi_TE - phi_TM + np.pi) % (2*np.pi) - np.pi
    phase_diff_center[i] = phase_diff

# --------------------------------------------------------
# 4) POST-PROCESSING & VISUALIZATION
# --------------------------------------------------------
# Calculate phase differences throughout propagation
phi_TE = np.angle(Etot_TE)
phi_TM = np.angle(Etot_TM)
dphi = (phi_TE - phi_TM + np.pi) % (2*np.pi) - np.pi

# Define mask for π/2 phase difference with tolerance
tolerance = 0.1
mask_pi_2 = np.abs(dphi - np.pi/2) < tolerance

# A) Show regions - including birefringent section and interface
plt.figure(figsize=(12, 8))
plt.axvline(x=biref_start*1e6, color='g', linestyle='--', label='Birefringent region start')
plt.axvline(x=biref_end*1e6, color='g', linestyle='--', label='Birefringent region end')
plt.axvline(x=z_interface*1e6, color='k', linestyle='-', label='Silica-Air interface')

# Plot combined TE & TM intensities
plt.pcolormesh(zprop*1e6, xv*1e6, np.abs(Etot_TE).T**2, 
               shading='auto', cmap='Reds', alpha=0.5)
plt.pcolormesh(zprop*1e6, xv*1e6, np.abs(Etot_TM).T**2, 
               shading='auto', cmap='Blues', alpha=0.5)

plt.xlabel('z (µm)', fontsize=12)
plt.ylabel('x (µm)', fontsize=12)
plt.title('TE (red) & TM (blue) Propagation with Birefringent Region', fontsize=14)
plt.legend(loc='upper left')
plt.colorbar(label='Intensity (a.u.)')
plt.tight_layout()
#plt.savefig('physical_model_propagation.png', dpi=300)
plt.show()

# A) Show regions - including birefringent section and interface
plt.figure(figsize=(12, 8))
plt.axvline(x=biref_start*1e6, color='k', linestyle='--', label='Biref start')
plt.axvline(x=z_interface*1e6,   color='k', linestyle='--', label='Biref end')
plt.axvline(x=z_interface*1e6, color='k', linestyle='-',  label='Interface')

# TE intensity as translucent red mesh
TE_intensity = np.abs(Etot_TE).T**2
pcm = plt.pcolormesh(zprop*1e6,
                     xv*1e6,
                     TE_intensity,
                     shading='auto',
                     cmap='Reds',
                     alpha=0.4)

# TM intensity as blue contour lines
TM_intensity = np.abs(Etot_TM).T**2
levels = np.linspace(0, TM_intensity.max(), 8)[1:]
ctr = plt.contour(zprop*1e6,
                  xv*1e6,
                  TM_intensity,
                  levels=levels,
                  colors='blue',
                  linewidths=1)

# proxy artists for the legend
from matplotlib.patches import Patch
from matplotlib.lines   import Line2D
legend_handles = [
    Patch(facecolor='red', edgecolor='none', alpha=0.4, label='TE intensity'),
    Line2D([0],[0], color='blue', lw=2,                   label='TM intensity'),
]
plt.legend(handles=legend_handles, loc='upper left')

plt.xlabel('z (µm)', fontsize=12)
plt.ylabel('x (µm)', fontsize=12)
plt.title('TE (red) & TM (blue) Propagation with Birefringent Region', fontsize=14)
plt.colorbar(pcm, label='TE Intensity (a.u.)')
plt.tight_layout()
plt.show()
##the new plotting code stops here!!###########

# B) Phase difference throughout propagation
plt.figure(figsize=(12, 8))
plt.axvline(x=biref_start*1e6, color='g', linestyle='--', label='Birefringent region start')
plt.axvline(x=z_interface*1e6, color='k', linestyle='--', label='Birefringent region end')
#plt.axvline(x=z_interface*1e6, color='k', linestyle='-', label='Silica-Air interface')

plt.pcolormesh(zprop*1e6, xv*1e6, dphi.T, 
               shading='auto', cmap='twilight', vmin=-np.pi, vmax=np.pi)
plt.colorbar(label='Phase Difference (rad)')
plt.xlabel('z (µm)', fontsize=12)
plt.ylabel('x (µm)', fontsize=12)
plt.title('Phase Difference (TE-TM) Throughout Propagation', fontsize=14)
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('physical_model_phase_diff.png', dpi=300)
plt.show()

# =============================================================================
# # C) Phase difference along beam center
# plt.figure(figsize=(10, 6))
# plt.axvline(x=biref_start*1e6, color='g', linestyle='--', label='Birefringent region start')
# plt.axvline(x=biref_end*1e6, color='g', linestyle='--', label='Birefringent region end')
# plt.axvline(x=z_interface*1e6, color='k', linestyle='-', label='Silica-Air interface')
# 
# # Plot phase difference at beam center
# plt.plot(zprop*1e6, phase_diff_center, 'b-', linewidth=2, label='Phase diff at beam center')
# plt.axhline(y=np.pi/2, color='r', linestyle='--', label='π/2')
# plt.axhline(y=np.pi/4, color='r', linestyle=':', label='π/4')
# 
# plt.xlabel('z (µm)', fontsize=12)
# plt.ylabel('Phase Difference (rad)', fontsize=12)
# plt.title('Phase Difference at Beam Center vs. Propagation Distance', fontsize=14)
# plt.ylim(-np.pi, np.pi)
# plt.grid(True, alpha=0.3)
# plt.legend(loc='best')
# plt.tight_layout()
# plt.savefig('physical_model_center_phase.png', dpi=300)
# plt.show()
# 
# =============================================================================
# =============================================================================
# # D) Intensity with π/2 contours
# plt.figure(figsize=(12, 8))
# plt.axvline(x=biref_start*1e6, color='g', linestyle='--', label='Birefringent region start')
# plt.axvline(x=biref_end*1e6, color='g', linestyle='--', label='Birefringent region end')
# plt.axvline(x=z_interface*1e6, color='k', linestyle='-', label='Silica-Air interface')
# 
# # Plot TE intensity
# plt.pcolormesh(zprop*1e6, xv*1e6, np.abs(Etot_TE).T**2, 
#                shading='auto', cmap='Greys')
# 
# =============================================================================
# Add π/2 phase contours
# Create masks for significant intensity
intensity_threshold = 0.05
intensity_mask = np.zeros_like(Etot_TE, dtype=bool)
for i in range(nz):
    intensity_slice = np.abs(Etot_TE[i,:])**2
    max_intensity = np.max(intensity_slice)
    if max_intensity > 0:  # Avoid division by zero
        intensity_mask[i,:] = intensity_slice > (max_intensity * intensity_threshold)

# Combine with phase mask
mask_pi_2_beam = mask_pi_2 & intensity_mask
pi_2_z, pi_2_x = np.where(mask_pi_2_beam.T)  # Note the transpose for correct coordinates

# =============================================================================
# plt.scatter(zprop[pi_2_x]*1e6, xv[pi_2_z]*1e6, s=2, c='red', label='Δφ ≈ π/2')
# 
# plt.xlabel('z (µm)', fontsize=12)
# plt.ylabel('x (µm)', fontsize=12)
# plt.title('TE Intensity with π/2 Phase Difference Contours', fontsize=14)
# plt.legend(loc='upper left')
# plt.colorbar(label='Intensity (a.u.)')
# plt.tight_layout()
# plt.savefig('physical_model_pi2_contours.png', dpi=300)
# plt.show()
# =============================================================================

# E) Analysis of phase accumulation in birefringent region
biref_region_z = zprop[idx_biref_start:idx_biref_end+1]
phase_diff_biref = phase_diff_center[idx_biref_start:idx_biref_end+1]
total_phase_accumulated = phase_diff_biref[-1] - phase_diff_biref[0]
# =============================================================================
# 
# plt.figure(figsize=(10, 6))
# plt.plot(biref_region_z*1e6, phase_diff_biref, 'b-', linewidth=2)
# plt.axhline(y=phase_diff_biref[0], color='r', linestyle=':', label='Initial phase')
# plt.axhline(y=phase_diff_biref[-1], color='r', linestyle='--', label='Final phase')
# 
# plt.xlabel('z (µm)', fontsize=12)
# plt.ylabel('Phase Difference (rad)', fontsize=12)
# plt.title(f'Phase Accumulation in Birefringent Region: {total_phase_accumulated:.4f} rad', fontsize=14)
# plt.grid(True, alpha=0.3)
# plt.legend(loc='best')
# plt.tight_layout()
# plt.savefig('physical_model_phase_accumulation.png', dpi=300)
# plt.show()
# 
# =============================================================================
print(f"\nPhase analysis in birefringent region:")
print(f"Initial phase difference: {phase_diff_biref[0]:.4f} rad")
print(f"Final phase difference: {phase_diff_biref[-1]:.4f} rad")
print(f"Total phase accumulated: {total_phase_accumulated:.4f} rad ({total_phase_accumulated*180/np.pi:.2f}°)")
print(f"Theoretical prediction: {expected_phase_shift:.4f} rad ({expected_phase_degrees:.2f}°)")
print(f"Agreement: {100*(1-abs(total_phase_accumulated-expected_phase_shift)/expected_phase_shift):.1f}%")