#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 16:36:03 2025

@author: joel
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from silica_my import silica_n

lam    = 780e-9
k0     = 2*np.pi/lam
nsil   = silica_n(lam)
n_air  = 1

w0_TE  = 2.145549895305066e-6
w0_TM  = 2.155888764131035e-6

angles_deg = [0, 15, 30, 45, 60]  

ntar_vectors = []

for angle_deg in angles_deg:
    angle_rad = angle_deg * np.pi / 180
    
    # Create target vector [x,z] components
    # For angle=0, we get [0,1] (pure vertical)
    ntar = np.array([np.sin(angle_rad), np.cos(angle_rad)])
    ntar_vectors.append(ntar)

zfoc = 50e-6  
E20 = 0.1
p = 1

biref_start = -25e-6   # Position (relative to interface where beam exits the waveguide)
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

# Spatial grid
xv     = np.linspace(-60e-6, 60e-6, 2048)  
nx     = xv.size
dx     = xv[1]-xv[0]

z_start = -40e-6    
z_interface = biref_end 
z_end = 40e-6       

# Create non-uniform z grid with finer resolution in the birefringent region
z_before_biref = np.linspace(z_start, biref_start, 20)  # Before birefringent region
z_biref = np.linspace(biref_start, biref_end, 50)      # Birefringent region (higher sampling)
z_after_biref = np.linspace(biref_end, z_interface, 15) # After birefringent region to interface
z_air = np.linspace(z_interface, z_end, 40)             # In air (match original sampling)

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
def build_tilted_gaussian(w2, z_pos, ntar):
    """Build a tilted Gaussian at a given z position with target direction ntar"""
    # rotate coords
    zzr = xv*ntar[0]
    xxr = np.sqrt(xv**2 - zzr**2 + 1e-20)  # Add small constant to avoid sqrt of negative
    
    # shift to put waist at z = zfoc
    # We account for the current z_pos
    rel_z = z_pos - zfoc
    zzr -= rel_z

    # Rayleigh range in silica
    zR = np.pi * w2**2 / lam * nsil
    w2z = w2 * np.sqrt(1 + (zzr/zR)**2)
    eta = 0.5*np.arctan(zzr/zR)*0.5
    Rzi = zzr/(zzr**2 + zR**2 + 1e-20)  # Add small constant to avoid division by zero

    E = (E20 * np.sqrt(w2/w2z)
         * np.exp(- (xxr/w2z)**(2*p))
         * np.exp(1j*(k0*nsil*zzr
                     + k0*nsil*(xxr**2)*Rzi/2
                     - eta)))
    return E

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
# 3) FUNCTION TO RUN PROPAGATION FOR A SPECIFIC ANGLE
# --------------------------------------------------------
def propagate_with_angle(angle_idx):
    """Run propagation simulation for a specific angle"""
    ntar = ntar_vectors[angle_idx]
    angle_deg = angles_deg[angle_idx]
    
    # Calculate expected phase shift for this angle
    # For birefringent medium with optic axis aligned with interface:
    # Effective path length increases with angle as d/cos(angle)
    # But projection of E-field onto extraordinary axis decreases with cos(angle)
    # The net effect depends on the orientation of the optic axis
    
    # Here we assume optic axis is vertical (parallel to interface):
    path_factor = 1.0 / np.cos(angle_deg * np.pi/180)        # Path length effect
    expected_phase_shift = k0 * delta_n * biref_thickness * path_factor
    expected_phase_degrees = expected_phase_shift * 180/np.pi
    
    print(f"\nAngle {angle_deg}° simulation:")
    print(f"Target vector: [{ntar[0]:.3f}, {ntar[1]:.3f}]")
    print(f"Path length factor: {path_factor:.3f}")
    print(f"Expected phase shift: {expected_phase_shift:.4f} rad ({expected_phase_degrees:.2f}°)")

    # Build initial fields at the starting position for this angle
    E_TE = build_tilted_gaussian(w0_TE, z_start, ntar)
    E_TM = build_tilted_gaussian(w0_TM, z_start, ntar)
    
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
        center_idx = np.argmax(np.abs(E_TE_current)**2)
        phi_TE     = np.angle(E_TE_current[center_idx])
        phi_TM     = np.angle(E_TM_current[center_idx])
        phase_diff_center[i] = phi_TE - phi_TM

    # unwrap the center-beam phase so it grows smoothly past ±π
    phase_diff_center = np.unwrap(phase_diff_center)

    # extract only the birefringent region
    phase_diff_biref = phase_diff_center[idx_biref_start:idx_biref_end+1]
    total_phase_accumulated = phase_diff_biref[-1] - phase_diff_biref[0]

    print(f" Initial phase difference: {phase_diff_biref[0]:.4f} rad")
    print(f" Final   phase difference: {phase_diff_biref[-1]:.4f} rad")
    print(f" Accumulated phase: {total_phase_accumulated:.4f} rad ({np.degrees(total_phase_accumulated):.1f}°)")
    print(f" Agreement with theory: {100*(1-abs(total_phase_accumulated-expected_phase_shift)/expected_phase_shift):.1f}%")

    # compute full 2D wrapped phase fields then unwrap along z
    phi_TE_2d = np.angle(Etot_TE)
    phi_TM_2d = np.angle(Etot_TM)
    dphi      = np.unwrap(phi_TE_2d - phi_TM_2d, axis=0)
    
    print(f"Initial phase difference: {phase_diff_biref[0]:.4f} rad")
    print(f"Final phase difference: {phase_diff_biref[-1]:.4f} rad")
    print(f"Total phase accumulated: {total_phase_accumulated:.4f} rad ({total_phase_accumulated*180/np.pi:.2f}°)")
    print(f"Agreement with theory: {100*(1-abs(total_phase_accumulated-expected_phase_shift)/expected_phase_shift):.1f}%")
    
    return {
        'angle': angle_deg,
        'Etot_TE': Etot_TE,
        'Etot_TM': Etot_TM,
        'dphi': dphi,
        'phase_diff_center': phase_diff_center,
        'biref_phase_diff': phase_diff_biref,
        'total_phase_accumulated': total_phase_accumulated,
        'expected_phase_shift': expected_phase_shift
    }

# --------------------------------------------------------
# 4) RUN SIMULATION FOR ALL ANGLES AND ANALYZE RESULTS
# --------------------------------------------------------
results = []
for angle_idx in range(len(angles_deg)):
    results.append(propagate_with_angle(angle_idx))

# --------------------------------------------------------
# 5) VISUALIZATION
# --------------------------------------------------------

phase_cmap = LinearSegmentedColormap.from_list('phase_cmap', 
                                              ['blue', 'white', 'red'], 
                                              N=256)

# A) Phase accumulation vs angle
plt.figure(figsize=(10, 6))
angles = [r['angle'] for r in results]
measured_phases = [r['total_phase_accumulated'] for r in results]
expected_phases = [r['expected_phase_shift'] for r in results]

plt.plot(angles, np.array(measured_phases)*180/np.pi, 'bo-', label='Simulated')
plt.plot(angles, np.array(expected_phases)*180/np.pi, 'r--', label='Theoretical')

plt.xlabel('Angle (degrees)', fontsize=12)
plt.ylabel('Phase Accumulation (degrees)', fontsize=12)
plt.title('Phase Accumulation vs. Angle in Birefringent Region', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
#%%
# B) Intensity profiles at different angles
fig = plt.figure(figsize=(15, 12))
gs = gridspec.GridSpec(len(angles_deg), 2, width_ratios=[2, 1])

# =============================================================================
# for i, result in enumerate(results):
#     # Propagation plot
#     ax1 = plt.subplot(gs[i, 0])
#     
#     # Plot combined TE & TM intensities
#     ax1.pcolormesh(zprop*1e6, xv*1e6, np.abs(result['Etot_TE']).T**2, 
#                    shading='auto', cmap='Reds', alpha=0.5)
#     ax1.pcolormesh(zprop*1e6, xv*1e6, np.abs(result['Etot_TM']).T**2, 
#                    shading='auto', cmap='Blues', alpha=0.5)
#     
#     ax1.axvline(x=biref_start*1e6, color='k', linestyle='-')
#     ax1.axvline(x=biref_end*1e6, color='k', linestyle='-')
#     ax1.axvline(x=z_interface*1e6, color='k', linestyle='-')
#     
#     if i == 0:
#         ax1.text(biref_start*1e6 - 2, 35, 'Birefringent\nregion', color='k', ha='right')
#         ax1.text(z_interface*1e6 + 2, 50, 'Air', color='k', ha='left')
#     
#     ax1.set_title(f'Angle = {result["angle"]}°', fontsize=12)
#     ax1.set_xlabel('z (µm)', fontsize=10)
#     ax1.set_ylabel('x (µm)', fontsize=10)
#     
#     # Phase profile in birefringent region
#     ax2 = plt.subplot(gs[i, 1])
#     
#     biref_region_z = zprop[idx_biref_start:idx_biref_end+1]
#     phase_diff_biref = result['phase_diff_center'][idx_biref_start:idx_biref_end+1]
#     
#     ax2.plot(biref_region_z*1e6, phase_diff_biref, 'b-', linewidth=2)
#     ax2.axhline(y=phase_diff_biref[0], color='r', linestyle=':', label='Initial')
#     ax2.axhline(y=phase_diff_biref[-1], color='r', linestyle='--', label='Final')
#     
#     ax2.set_xlim(biref_start*1e6, biref_end*1e6)
#     ax2.set_ylim(-np.pi, np.pi)
#     ax2.xaxis.set_major_locator(MultipleLocator(5))
#     ax2.grid(True, alpha=0.3)
#     
#     total_phase = result['total_phase_accumulated']
#     ax2.set_title(f'Δφ = {total_phase:.3f} rad ({total_phase*180/np.pi:.1f}°)', fontsize=12)
#     ax2.set_xlabel('z (µm)', fontsize=10)
#     
#     if i == 0:
#         ax2.set_ylabel('Phase Difference (rad)', fontsize=10)
#         ax2.legend(loc='lower right', fontsize=8)
# =============================================================================
for i, result in enumerate(results):
    ax1 = plt.subplot(gs[i, 0])

    # TE: filled, semi-transparent red
    te = np.abs(result['Etot_TE']).T**2
    pcm_te = ax1.pcolormesh(zprop*1e6, xv*1e6, te,
                            shading='auto',
                            cmap='Reds',
                            alpha=0.4)
    
    # TM: draw as blue contours on top
    tm = np.abs(result['Etot_TM']).T**2
    # pick a reasonable set of levels, e.g. 10 levels between 0 and max
    levels = np.linspace(0, tm.max(), 10)[1:]
    ctr_tm = ax1.contour(zprop*1e6, xv*1e6, tm,
                         levels=levels,
                         colors='blue',
                         linewidths=1)
    
    # add proxy artists so the legend knows which is which
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    if i == 0:
        legend_items = [
            Patch(facecolor='red', edgecolor='none', alpha=0.4, label='TE intensity'),
            Line2D([0],[0], color='blue', lw=2, label='TM intensity'),
        ]
        ax1.legend(handles=legend_items, loc='upper right')

    ax1.axvline(x=biref_start*1e6, color='k', linestyle='-')
    ax1.axvline(x=biref_end*1e6,   color='k', linestyle='-')
    ax1.axvline(x=z_interface*1e6, color='k', linestyle='-')
    ax1.set_title(f'Angle = {result["angle"]}°')
    ax1.set_xlabel('z (µm)')
    ax1.set_ylabel('x (µm)')


plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.show()

# =============================================================================
# # C) Phase distribution at the output
# plt.figure(figsize=(12, 8))
# 
# # Find index closest to air interface
# output_idx = idx_interface + 5  # A bit after the interface
# 
# for i, result in enumerate(results):
#     # Get intensity and phase at output
#     intensity_TE = np.abs(result['Etot_TE'][output_idx])**2
#     intensity_TM = np.abs(result['Etot_TM'][output_idx])**2
#     total_intensity = intensity_TE + intensity_TM
#     
#     # Normalize
#     total_intensity /= np.max(total_intensity)
#     
#     # Get phase difference
#     phi_diff = result['dphi'][output_idx]
#     
#     # Plot only where intensity is significant
#     mask = total_intensity > 0.01
#     plt.plot(xv[mask]*1e6, phi_diff[mask], '-', label=f'{result["angle"]}°')
# 
# plt.axhline(y=np.pi/2, color='k', linestyle='--', label='π/2')
# plt.axhline(y=np.pi/4, color='k', linestyle=':', label='π/4')
# plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
# 
# plt.xlabel('x (µm)', fontsize=12)
# plt.ylabel('Phase Difference (rad)', fontsize=12)
# plt.title('Phase Difference (TE-TM) at Output vs. Position', fontsize=14)
# plt.ylim(-np.pi, np.pi)
# plt.grid(True, alpha=0.3)
# plt.legend(loc='upper right')
# plt.tight_layout()
# plt.show()
# =============================================================================

# D) Effective birefringence vs. angle
# =============================================================================
# plt.figure(figsize=(10, 6))
# 
# angles = np.array(angles_deg)
# measured_phases = np.array([r['total_phase_accumulated'] for r in results])
# expected_phases = np.array([r['expected_phase_shift'] for r in results])
# 
# # Calculate effective birefringence (phase / (k0 * thickness))
# measured_dn_eff = measured_phases / (k0 * biref_thickness)
# expected_dn_eff = expected_phases / (k0 * biref_thickness)
# 
# plt.plot(angles, measured_dn_eff, 'bo-', label='Simulated')
# plt.plot(angles, expected_dn_eff, 'r--', label='Theoretical')
# plt.plot(angles, delta_n * np.ones_like(angles), 'k:', label='Original Δn')
# 
# plt.xlabel('Angle (degrees)', fontsize=12)
# plt.ylabel('Effective Birefringence (Δn_eff)', fontsize=12)
# plt.title('Effective Birefringence vs. Angle', fontsize=14)
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show()
# =============================================================================

# Summary
print("\nSummary of Phase Control vs. Angle:")
print("-----------------------------------")
print("Angle (°) | Phase Shift (°) | Effective Δn | % of Original Δn")
print("-----------------------------------------------------------")
for i, result in enumerate(results):
    angle = result['angle']
    phase_deg = result['total_phase_accumulated'] * 180/np.pi
    dn_eff = result['total_phase_accumulated'] / (k0 * biref_thickness)
    pct_original = dn_eff / delta_n * 100
    print(f"{angle:8.1f} | {phase_deg:14.2f} | {dn_eff:11.5f} | {pct_original:15.1f}")