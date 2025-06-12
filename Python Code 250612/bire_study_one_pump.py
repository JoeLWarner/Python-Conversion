#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 14:54:51 2025

@author: joel
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from silica_my import silica_n

# Wavelength and constants
lam = 780e-9
k0 = 2*np.pi/lam
nsil = silica_n(lam)
n_air = 1

# Beam parameters
w0_pump = 2.15e-6  # Single pump beam waist

# We'll only use a single 45° polarized beam
angle_deg = 30  # Fixed angle for our beam
angle_rad = angle_deg * np.pi / 180
ntar = np.array([np.sin(angle_rad), np.cos(angle_rad)])

# Propagation parameters
zfoc = 50e-6
E20 = 0.1  # Field amplitude
p = 1      # Gaussian order

# Birefringent region (quarter-wave plate)
biref_start = -25e-6
biref_end = -5e-6
biref_thickness = abs(biref_end - biref_start)

# Calculate required birefringence for π/2 (90°) phase shift at 45°
# For a QWP: phase_shift = k0 * delta_n * thickness / cos(angle) = π/2
# Solving for delta_n:
target_phase_shift = np.pi/2  # 90° (quarter wave)
path_factor = 1.0 / np.cos(angle_rad)
delta_n = target_phase_shift / (k0 * biref_thickness * path_factor)

n_TE_ordinary = nsil  # Ordinary index (same as silica)
n_TE_extraordinary = n_TE_ordinary + delta_n  # TE sees extraordinary index
n_TM = n_TE_ordinary  # TM sees ordinary index

print(f"Birefringent region thickness: {biref_thickness*1e6:.2f} µm")
print(f"Required birefringence (Δn) for QWP: {delta_n:.6f}")
print(f"TE index (extraordinary): {n_TE_extraordinary:.6f}")
print(f"TM index (ordinary): {n_TM:.6f}")

# Spatial grid
xv = np.linspace(-60e-6, 60e-6, 2048)
nx = xv.size
dx = xv[1]-xv[0]

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
# FUNCTIONS
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

def calculate_polarization_parameters(E_TE, E_TM):
    """Calculate Stokes parameters and polarization state"""
    # Get amplitudes and phases
    A_TE = np.abs(E_TE)
    A_TM = np.abs(E_TM)
    phi_TE = np.angle(E_TE)
    phi_TM = np.angle(E_TM)
    
    # Calculate phase difference (unwrapped)
    delta_phi = phi_TE - phi_TM
    
    # Stokes parameters
    S0 = A_TE**2 + A_TM**2  # Total intensity
    S1 = A_TE**2 - A_TM**2  # Linear horizontal/vertical polarization
    S2 = 2 * A_TE * A_TM * np.cos(delta_phi)  # Linear +45/-45 polarization
    S3 = 2 * A_TE * A_TM * np.sin(delta_phi)  # Circular polarization
    
    # Normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        norm = np.where(S0 > 0, 1/S0, 0)
    s1 = S1 * norm
    s2 = S2 * norm
    s3 = S3 * norm
    
    # Calculate polarization parameters
    # Degree of polarization (1=fully polarized, 0=unpolarized)
    DOP = np.sqrt(S1**2 + S2**2 + S3**2) / (S0 + 1e-20)
    
    # Ellipticity - range [-1, 1] where:
    # -1 = left circular, 0 = linear, 1 = right circular
    ellipticity = S3 / (np.sqrt(S1**2 + S2**2 + S3**2) + 1e-20)
    
    # Orientation angle in degrees (0° = horizontal, 90° = vertical)
    orientation = 0.5 * np.arctan2(S2, S1) * 180/np.pi
    
    return {
        'S0': S0, 'S1': S1, 'S2': S2, 'S3': S3,
        's1': s1, 's2': s2, 's3': s3,
        'DOP': DOP, 'ellipticity': ellipticity, 
        'orientation': orientation, 'phase_diff': delta_phi
    }

def run_qwp_simulation():
    """Run simulation for a 45° polarized beam through a quarter-wave plate"""
    print(f"\nSimulating 45° polarized beam through quarter-wave plate:")
    print(f"Target vector: [{ntar[0]:.3f}, {ntar[1]:.3f}]")
    
    # Create TE and TM components with equal amplitude for 45° polarization
    # The factor 1/sqrt(2) ensures each component has equal power
    initial_pump = build_tilted_gaussian(w0_pump, z_start, ntar)
    E_TE = initial_pump / np.sqrt(2)
    E_TM = initial_pump / np.sqrt(2)
    
    # Initialize arrays to store fields at all z positions
    Etot_TE = np.zeros((nz, nx), dtype=complex)
    Etot_TM = np.zeros((nz, nx), dtype=complex)

    # Store initial fields
    Etot_TE[0,:] = E_TE
    Etot_TM[0,:] = E_TM

    # Track current fields
    E_TE_current = E_TE.copy()
    E_TM_current = E_TM.copy()

    # Keep track of raw phase differences for unwrapping
    phase_diff_center_raw = np.zeros(nz)
    
    # Arrays to store polarization state throughout propagation
    ellipticity = np.zeros(nz)
    orientation = np.zeros(nz)
    s3_values = np.zeros(nz)  # Normalized S3 (measure of circular polarization)

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
                # We're in the birefringent region (quarter-wave plate)
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
        
        # Calculate center of beam 
        intensity_TE = np.abs(E_TE_current)**2
        center_idx = np.argmax(intensity_TE)
        
        # Get TE and TM fields at beam center
        E_TE_center = E_TE_current[center_idx]
        E_TM_center = E_TM_current[center_idx]
        
        # Calculate polarization parameters at beam center
        pol_params = calculate_polarization_parameters(
            np.array([E_TE_center]), np.array([E_TM_center]))
        
        # Store raw phase difference
        phi_TE = np.angle(E_TE_center)
        phi_TM = np.angle(E_TM_center)
        phase_diff = phi_TE - phi_TM
        phase_diff_center_raw[i] = phase_diff
        
        # Store polarization state parameters
        ellipticity[i] = pol_params['ellipticity'][0]
        orientation[i] = pol_params['orientation'][0]
        s3_values[i] = pol_params['s3'][0]
    
    # Unwrap the phase differences
    phase_diff_center = np.unwrap(phase_diff_center_raw)
    
    # Calculate polarization state throughout
    pol_at_interfaces = {
        'initial': calculate_polarization_parameters(
            Etot_TE[0, np.argmax(np.abs(Etot_TE[0])**2)].reshape(1),
            Etot_TM[0, np.argmax(np.abs(Etot_TM[0])**2)].reshape(1)
        ),
        'before_biref': calculate_polarization_parameters(
            Etot_TE[idx_biref_start-1, np.argmax(np.abs(Etot_TE[idx_biref_start-1])**2)].reshape(1),
            Etot_TM[idx_biref_start-1, np.argmax(np.abs(Etot_TM[idx_biref_start-1])**2)].reshape(1)
        ),
        'after_biref': calculate_polarization_parameters(
            Etot_TE[idx_biref_end, np.argmax(np.abs(Etot_TE[idx_biref_end])**2)].reshape(1),
            Etot_TM[idx_biref_end, np.argmax(np.abs(Etot_TM[idx_biref_end])**2)].reshape(1)
        ),
        'at_interface': calculate_polarization_parameters(
            Etot_TE[idx_interface-1, np.argmax(np.abs(Etot_TE[idx_interface-1])**2)].reshape(1),
            Etot_TM[idx_interface-1, np.argmax(np.abs(Etot_TM[idx_interface-1])**2)].reshape(1)
        ),
        'final': calculate_polarization_parameters(
            Etot_TE[-1, np.argmax(np.abs(Etot_TE[-1])**2)].reshape(1),
            Etot_TM[-1, np.argmax(np.abs(Etot_TM[-1])**2)].reshape(1)
        )
    }
    
    # Calculate phase change through birefringent region (QWP)
    biref_phase_diff = phase_diff_center[idx_biref_start:idx_biref_end+1]
    total_phase_accumulated = biref_phase_diff[-1] - biref_phase_diff[0]
    
    print(f"Initial phase difference: {biref_phase_diff[0]:.4f} rad")
    print(f"Final phase difference: {biref_phase_diff[-1]:.4f} rad")
    print(f"Total phase accumulated: {total_phase_accumulated:.4f} rad ({total_phase_accumulated*180/np.pi:.2f}°)")
    print(f"Target for quarter-wave: {np.pi/2:.4f} rad (90.00°)")
    print(f"Agreement with target: {100*(1-abs(total_phase_accumulated-np.pi/2)/(np.pi/2)):.1f}%")
    
# =============================================================================
#     # Report polarization states
#     print("\nPolarization Parameters:")
#     print(f"Initial: Ellipticity = {pol_at_interfaces['initial']['ellipticity'][0]:.3f}, " +
#           f"Orientation = {pol_at_interfaces['initial']['orientation'][0]:.1f}°")
#     print(f"After QWP: Ellipticity = {pol_at_interfaces['after_biref']['ellipticity'][0]:.3f}, " +
#           f"Orientation = {pol_at_interfaces['after_biref']['orientation'][0]:.1f}°")
#     print(f"Final: Ellipticity = {pol_at_interfaces['final']['ellipticity'][0]:.3f}, " +
#           f"Orientation = {pol_at_interfaces['final']['orientation'][0]:.1f}°")
#     
#     # Interpret polarization state
#     if abs(pol_at_interfaces['after_biref']['ellipticity'][0]) > 0.9:
#         circ_type = "left" if pol_at_interfaces['after_biref']['ellipticity'][0] < 0 else "right"
#         print(f"After birefringent region: Nearly {circ_type}-circular polarization")
# =============================================================================
    
    return {
        'Etot_TE': Etot_TE,
        'Etot_TM': Etot_TM,
        'phase_diff_center': phase_diff_center,
        'ellipticity': ellipticity,
        'orientation': orientation,
        's3_values': s3_values,
        'pol_at_interfaces': pol_at_interfaces,
        'biref_phase_diff': biref_phase_diff,
        'total_phase_accumulated': total_phase_accumulated
    }

# --------------------------------------------------------
# RUN SIMULATION AND VISUALIZATION
# --------------------------------------------------------

# Run the simulation
results = run_qwp_simulation()

# Create visualization
plt.figure(figsize=(15, 12))
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])

# Plot polarization parameters throughout propagation
ax1 = plt.subplot(gs[0, :])
ax1.plot(zprop*1e6, results['s3_values'], 'r-', linewidth=2, 
         label='S3 (Circular Polarization)')
ax1.plot(zprop*1e6, results['ellipticity'], 'b--', linewidth=2, 
         label='Ellipticity')

ax1.axvline(x=biref_start*1e6, color='k', linestyle='-')
ax1.axvline(x=biref_end*1e6, color='k', linestyle='-')
ax1.axvline(x=z_interface*1e6, color='k', linestyle='-')

ax1.fill_between([biref_start*1e6, biref_end*1e6], -1.2, 1.2, 
                 color='lightgray', alpha=0.3, label='Birefringent Region (QWP)')

ax1.set_ylim(-1.1, 1.1)
ax1.set_xlim(z_start*1e6, z_end*1e6)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')
ax1.set_title('Polarization Evolution', fontsize=14)
ax1.set_ylabel('Parameter Value', fontsize=12)
ax1.text(biref_start*1e6 - 2, 0.9, 'Linear 45°', ha='right')
ax1.text(biref_end*1e6 + 2, 0.9, 'Circular', ha='left')

# Phase difference plot
ax2 = plt.subplot(gs[1, :])
phase_in_deg = results['phase_diff_center'] * 180/np.pi
ax2.plot(zprop*1e6, phase_in_deg, 'g-', linewidth=2)

ax2.axvline(x=biref_start*1e6, color='k', linestyle='-')
ax2.axvline(x=biref_end*1e6, color='k', linestyle='-')
ax2.axvline(x=z_interface*1e6, color='k', linestyle='-')

ax2.fill_between([biref_start*1e6, biref_end*1e6], 
                 np.min(phase_in_deg)-10, np.max(phase_in_deg)+10, 
                 color='lightgray', alpha=0.3)

ax2.grid(True, alpha=0.3)
ax2.set_title('Phase Difference Between TE and TM Components', fontsize=14)
ax2.set_ylabel('Phase Difference (degrees)', fontsize=12)
ax2.set_xlim(z_start*1e6, z_end*1e6)

# Plot combined TE & TM intensities
ax3 = plt.subplot(gs[2, :])
ax3.pcolormesh(zprop*1e6, xv*1e6, np.abs(results['Etot_TE']).T**2, 
             shading='auto', cmap='Reds', alpha=0.5, label='TE')
ax3.pcolormesh(zprop*1e6, xv*1e6, np.abs(results['Etot_TM']).T**2, 
             shading='auto', cmap='Blues', alpha=0.5, label='TM')

ax3.axvline(x=biref_start*1e6, color='k', linestyle='-')
ax3.axvline(x=biref_end*1e6, color='k', linestyle='-')
ax3.axvline(x=z_interface*1e6, color='k', linestyle='-')

ax3.text(biref_start*1e6 - 2, 35, 'QWP\nRegion', color='k', ha='right')
ax3.text(z_interface*1e6 + 2, 50, 'Air', color='k', ha='left')

ax3.set_title('Beam Propagation', fontsize=14)
ax3.set_xlabel('z (µm)', fontsize=12)
ax3.set_ylabel('x (µm)', fontsize=12)

# Adjust layout and save
plt.tight_layout()
plt.subplots_adjust(hspace=0.35)
plt.show()

# Bonus: Create visualization of polarization ellipses at key points
z_points = [
    (0, "Initial"),
    (idx_biref_start, "Before QWP"),
    (idx_biref_end, "After QWP"),
    (nz-1, "Final")
]

plt.figure(figsize=(16, 4))
for i, (z_idx, title) in enumerate(z_points):
    ax = plt.subplot(1, 4, i+1)
    
    # Get fields at beam center for this z position
    intensity = np.abs(results['Etot_TE'][z_idx])**2
    center_idx = np.argmax(intensity)
    
    E_TE = results['Etot_TE'][z_idx, center_idx]
    E_TM = results['Etot_TM'][z_idx, center_idx]
    
    A_TE = np.abs(E_TE)
    A_TM = np.abs(E_TM)
    phi_TE = np.angle(E_TE)
    phi_TM = np.angle(E_TM)
    delta_phi = phi_TE - phi_TM
    
    # Plot polarization ellipse
    t = np.linspace(0, 2*np.pi, 100)
    Ex = A_TE * np.cos(t)
    Ey = A_TM * np.cos(t - delta_phi)
    
    ax.plot(Ex, Ey, 'b-', linewidth=2)
    ax.plot([0, 0], [-1.5*A_TM, 1.5*A_TM], 'k--', alpha=0.3)
    ax.plot([-1.5*A_TE, 1.5*A_TE], [0, 0], 'k--', alpha=0.3)
    ax.set_xlim(-1.5*max(A_TE, A_TM), 1.5*max(A_TE, A_TM))
    ax.set_ylim(-1.5*max(A_TE, A_TM), 1.5*max(A_TE, A_TM))
    ax.set_aspect('equal')
    
    # Get polarization parameters
    pol_params = calculate_polarization_parameters(
        np.array([E_TE]), np.array([E_TM]))
    
    ellip = pol_params['ellipticity'][0]
    orient = pol_params['orientation'][0]
    s3 = pol_params['s3'][0]
    
    # Determine polarization type
    if abs(ellip) < 0.1:
        pol_type = "Linear"
    elif abs(ellip) > 0.9:
        pol_type = "Circular" + (" (L)" if ellip < 0 else " (R)")
    else:
        pol_type = "Elliptical" + (" (L)" if ellip < 0 else " (R)")
    
    ax.set_title(f"{title}\n{pol_type}\nε={ellip:.2f}, θ={orient:.1f}°")
    
    # Add z position info
    ax.text(0.5, -0.15, f"z = {zprop[z_idx]*1e6:.1f} µm", 
            transform=ax.transAxes, ha='center')
    
plt.tight_layout()
plt.show()

# Additional plot - Poincaré sphere trajectory
# (This would require a 3D plot showing the evolution on the Poincaré sphere,
# with S1, S2, S3 as coordinates)