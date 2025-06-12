#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 11:55:16 2025

@author: joel
"""

"""
Study of angular dependence of birefringent cladding layer acting as QWP
Modified from original code to analyze how TM mode behavior changes with angle
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from silica_my import silica_n
from matplotlib.ticker import FuncFormatter, MultipleLocator

# Wavelength and constants
lam = 780e-9
k0 = 2*np.pi/lam
nsil = silica_n(lam)
n_air = 1

# Beam parameters
w0_pump = 2.15e-6  # Single pump beam waist

# Birefringent region parameters
biref_start = -25e-6
biref_end = -5e-6
biref_thickness = abs(biref_end - biref_start)

# Propagation parameters
zfoc = 50e-6
E20 = 0.1  # Field amplitude
p = 1      # Gaussian order

# Interface and propagation grid
z_start = -40e-6
z_interface = biref_end
z_end = 40e-6

# Spatial grid
xv = np.linspace(-60e-6, 60e-6, 2048)
nx = xv.size
dx = xv[1]-xv[0]

# Create non-uniform z grid
z_before_biref = np.linspace(z_start, biref_start, 20)  # Before birefringent region
z_biref = np.linspace(biref_start, biref_end, 50)      # Birefringent region (higher sampling)
z_after_biref = np.linspace(biref_end, z_interface, 15) # After birefringent region to interface
z_air = np.linspace(z_interface, z_end, 40)             # In air

# Combine z grids
zprop = np.unique(np.concatenate([z_before_biref, z_biref, z_after_biref, z_air]))
nz = zprop.size

# Indices for regions
idx_biref_start = np.argmin(np.abs(zprop - biref_start))
idx_biref_end = np.argmin(np.abs(zprop - biref_end))
idx_interface = np.argmin(np.abs(zprop - z_interface))

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

def build_plane_wave(z_pos, ntar, E0=E20):
    """
    Build a plane wave at z = z_pos propagating along ntar = [sinθ, cosθ].
    """
    # phase accumulated over x at the entry plane
    # k0·nsil is the wavevector in the medium
    phase = k0 * nsil * (ntar[0] * xv + ntar[1] * z_pos)
    return E0 * np.exp(1j * phase)
# =============================================================================
# def build_plane_wave(z_pos, ntar, E0=E20):
#     """
#     Build a plane wave at z = z_pos propagating along ntar = [sinθ, cosθ]
#     using coordinate rotation similar to the Gaussian beam approach.
#     """
#     # rotate coords - same approach as in build_tilted_gaussian
#     zzr = xv*ntar[0]
#     xxr = np.sqrt(xv**2 - zzr**2 + 1e-20)  # Add small constant to avoid sqrt of negative
#     
#     # shift to match position similar to Gaussian method
#     rel_z = z_pos - zfoc
#     zzr -= rel_z
# 
#     # Create plane wave with rotated coordinates
#     # Use only the linear phase term (no curvature or Gouy phase)
#     phase = k0 * nsil * zzr
#     
#     return E0 * np.exp(1j * phase)
# =============================================================================

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

def calculate_tm_projection(angle_rad):
    """
    Calculate how TM field components project onto the birefringent axes
    
    For a birefringent layer with one axis aligned with TE (y-axis),
    we need to calculate how the TM field components project onto the principal axes.
    """
    # Define propagation direction vector
    k_vec = np.array([np.sin(angle_rad), 0, np.cos(angle_rad)])
    
    # For TM polarization, E is perpendicular to both k and y
    # First, get the unit vector perpendicular to k and y
    E_TM_direction = np.cross(k_vec, np.array([0, 1, 0]))
    E_TM_direction = E_TM_direction / np.linalg.norm(E_TM_direction)
    
    # The E-field for TM has components in both x and z directions
    # But only the component along the x-axis (perpendicular to both y and z)
    # will be affected by the birefringence
    
    # Calculate projection onto x-axis (orthogonal to TE)
    projection_on_x = E_TM_direction[0]
    
    # Calculate projection onto z-axis (propagation direction)
    projection_on_z = E_TM_direction[2]
    
    return {
        'E_TM_direction': E_TM_direction,
        'projection_on_x': projection_on_x,
        'projection_on_z': projection_on_z,
        'angle_rad': angle_rad,
        'angle_deg': angle_rad * 180/np.pi
    }



def design_qwp_for_reference_angle(ref_angle_deg):
    """
    Design a QWP for a specific reference angle
    Returns the required birefringence (delta_n)
    """
    ref_angle_rad = ref_angle_deg * np.pi / 180
    
    # Calculate required birefringence for π/2 (90°) phase shift at reference angle
    target_phase_shift = np.pi/2  # 90° (quarter wave)
    path_factor = 1.0 / np.cos(ref_angle_rad)  # Path length increase due to angle
    
    # Calculate required delta_n
    delta_n = target_phase_shift / (k0 * biref_thickness * path_factor)
    
    # Ordinary and extraordinary indices
    n_ordinary = nsil
    n_extraordinary = n_ordinary + delta_n
    
    return {
        'delta_n': delta_n,
        'n_ordinary': n_ordinary,
        'n_extraordinary': n_extraordinary,
        'reference_angle_deg': ref_angle_deg,
        'reference_angle_rad': ref_angle_rad,
        'target_phase_shift': target_phase_shift
    }
# =============================================================================
# def design_qwp_for_reference_angle(ref_angle_deg, delta_n):
#     """
#     Design a QWP for a specific reference angle given a birefringence delta_n.
#     Returns the required thickness to achieve a quarter-wave (π/2 phase shift).
#     
#     Parameters:
#     - ref_angle_deg: Reference angle in degrees
#     - delta_n: Birefringence (difference in refractive indices)
#     - k0: Wave number (2π / wavelength)
#     - nsil: Ordinary refractive index (e.g., index of silica)
#     
#     Returns:
#     - dict with calculated thickness and related parameters
#     """
#     ref_angle_rad = ref_angle_deg * np.pi / 180
#     target_phase_shift = np.pi / 2  # Quarter-wave (90°) phase shift
#     path_factor = 1.0 / np.cos(ref_angle_rad)  # Effective path length increase
# 
#     # Calculate the required thickness
#     biref_thickness = target_phase_shift / (k0 * delta_n * path_factor)
# 
#     n_ordinary = nsil
#     n_extraordinary = n_ordinary + delta_n
# 
#     return {
#         'biref_thickness': biref_thickness,
#         'delta_n': delta_n,
#         'n_ordinary': n_ordinary,
#         'n_extraordinary': n_extraordinary,
#         'reference_angle_deg': ref_angle_deg,
#         'reference_angle_rad': ref_angle_rad,
#         'target_phase_shift': target_phase_shift
#     }
# =============================================================================


def run_qwp_simulation(angle_deg, qwp_params):
    """Run simulation at specified angle using the QWP parameters"""
    angle_rad = angle_deg * np.pi / 180
    ntar = np.array([np.sin(angle_rad), np.cos(angle_rad)])
    
    # Extract QWP parameters
    n_ordinary = qwp_params['n_ordinary']
    n_extraordinary = qwp_params['n_extraordinary']
    
    # For this study:
    # TE sees extraordinary index (E_y aligned with extraordinary axis)
    # TM sees ordinary index for the component perpendicular to TE
    n_TE = n_extraordinary
    n_TM = n_ordinary
    
    # Create initial field with equal TE and TM components (45° polarization)
    #initial_pump = build_tilted_gaussian(w0_pump, z_start, ntar)
    initial_pump = build_plane_wave(z_start, ntar)
    E_TE = initial_pump / np.sqrt(2)
    E_TM = initial_pump / np.sqrt(2)
    
    # Initialize arrays to store fields
    Etot_TE = np.zeros((nz, nx), dtype=complex)
    Etot_TM = np.zeros((nz, nx), dtype=complex)
    Etot_TE[0,:] = E_TE
    Etot_TM[0,:] = E_TM

    # Track current fields
    E_TE_current = E_TE.copy()
    E_TM_current = E_TM.copy()

    # Arrays to store polarization state
    phase_diff_center_raw = np.zeros(nz)
    ellipticity = np.zeros(nz)
    orientation = np.zeros(nz)
    s3_values = np.zeros(nz)

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
            E_TE_interface = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(E_k_TE_air)))
            E_TM_interface = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(E_k_TM_air)))
            
            # Then propagate remaining distance
            E_TE_current = angular_spectrum_step(E_TE_interface, remaining_dz, n_air)
            E_TM_current = angular_spectrum_step(E_TM_interface, remaining_dz, n_air)
        
        else:
            # Regular propagation step
            if idx_biref_start <= i <= idx_biref_end:
                # We're in the birefringent region (quarter-wave plate)
                E_TE_current = angular_spectrum_step(E_TE_current, dz, n_TE)
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
    
    # Calculate polarization state at key points
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
        'final': calculate_polarization_parameters(
            Etot_TE[-1, np.argmax(np.abs(Etot_TE[-1])**2)].reshape(1),
            Etot_TM[-1, np.argmax(np.abs(Etot_TM[-1])**2)].reshape(1)
        )
    }
    
    # Calculate phase change through birefringent region
    biref_phase_diff = phase_diff_center[idx_biref_start:idx_biref_end+1]
    total_phase_accumulated = biref_phase_diff[-1] - biref_phase_diff[0]
    
    return {
        'Etot_TE': Etot_TE,
        'Etot_TM': Etot_TM,
        'phase_diff_center': phase_diff_center,
        'ellipticity': ellipticity,
        'orientation': orientation,
        's3_values': s3_values,
        'pol_at_interfaces': pol_at_interfaces,
        'biref_phase_diff': biref_phase_diff,
        'total_phase_accumulated': total_phase_accumulated,
        'angle_deg': angle_deg
    }

def pi_formatter(y, pos):
    # Convert degrees to radians
    rad = np.deg2rad(y)
    # Convert to multiples of π/2
    multiple = rad / (np.pi/2)
    
    # Format the label based on the multiple value
    if multiple == 0:
        return "0"
    elif multiple.is_integer():
        if multiple == 1:
            return "π/2"
        elif multiple == 2:
            return "π"
        elif multiple == -1:
            return "-π/2"
        elif multiple == -2:
            return "-π"
        else:
            return f"{int(multiple)}π/2"
    else:
        # For non-integer multiples, round to 1 decimal place
        return f"{multiple:.1f}π/2"

# MAIN STUDY FUNCTION
def angular_dependence_study():
    """Study how QWP performance varies with angle"""
    
    print("Angular Dependence Study of QWP Performance")
    print("-" * 50)
    
    # 1. Design QWP for reference angle (90°)
    reference_angle = 0
    print(f"Designing QWP for reference angle: {reference_angle}°")
    
    dn = 0.172
    qwp_params = design_qwp_for_reference_angle(reference_angle)
    print(f"Birefringent layer parameters:")
    print(f"  Thickness: {biref_thickness*1e6:.2f} µm")
    print(f"  Target phase shift: {qwp_params['target_phase_shift']*180/np.pi:.1f}°")
    print(f"  Required birefringence (Δn): {qwp_params['delta_n']:.6f}")
    print(f"  TE index (extraordinary): {qwp_params['n_extraordinary']:.6f}")
    print(f"  TM index (ordinary): {qwp_params['n_ordinary']:.6f}")
    print()
    
    # 2. Study angle range
    angles = np.linspace(0, 80, 20)  # from 0° to 90° in 10 steps
    
    # Arrays to store results
    phase_accumulation = np.zeros_like(angles)
    s3_after_qwp = np.zeros_like(angles)
    ellipticity_after_qwp = np.zeros_like(angles)
    tm_x_projection = np.zeros_like(angles)
    tm_z_projection = np.zeros_like(angles)
    
    # Process each angle
    results_by_angle = {}
    
    for i, angle in enumerate(angles):
        #print(f"Processing angle: {angle:.1f}°...")
        
        # Calculate TM projection metrics
        tm_proj = calculate_tm_projection(angle * np.pi / 180)
        tm_x_projection[i] = tm_proj['projection_on_x']
        tm_z_projection[i] = tm_proj['projection_on_z']
        
        # Run simulation at this angle
        result = run_qwp_simulation(angle, qwp_params)
        results_by_angle[angle] = result
        
        # Extract key metrics
        phase_accumulation[i] = result['total_phase_accumulated'] * 180 / np.pi  # in degrees
        s3_after_qwp[i] = result['pol_at_interfaces']['after_biref']['s3'][0]
        ellipticity_after_qwp[i] = result['pol_at_interfaces']['after_biref']['ellipticity'][0]
        
        # Print summary
# =============================================================================
#         print(f"  - Total phase accumulated: {phase_accumulation[i]:.2f}°")
#         print(f"  - S3 after QWP: {s3_after_qwp[i]:.4f}")
#         print(f"  - Ellipticity after QWP: {ellipticity_after_qwp[i]:.4f}")
#         print(f"  - TM projection on x-axis: {tm_x_projection[i]:.4f}")
#         print(f"  - TM projection on z-axis: {tm_z_projection[i]:.4f}")
#         print()
# =============================================================================
    
    # ---------------------------------------------------------
    # VISUALIZATIONS
    # ---------------------------------------------------------
    
    # Figure 1: TM field vector alignment with birefringent axes
    plt.figure(figsize=(10, 6))
    
    # Plot projections of TM field
    plt.plot(angles, tm_x_projection, 'bo-', label='TM Projection on x-axis')
    plt.plot(angles, tm_z_projection, 'ro-', label='TM Projection on z-axis')
    
    plt.axvline(x=reference_angle, color='k', linestyle='--', label=f'Reference Angle ({reference_angle}°)')
    
    plt.xlabel('Scattering Angle (degrees)')
    plt.ylabel('Field Component Magnitude (normalized)')
    plt.title('TM Field Projections vs. Scattering Angle', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Figure 2: QWP Performance Metrics
    plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    
    # Phase accumulation
    ax1 = plt.subplot(gs[0])
    ax1.yaxis.set_major_formatter(FuncFormatter(pi_formatter))
    degrees_per_pi_half = 90  # π/2 = 90 degrees
    ax1.yaxis.set_major_locator(MultipleLocator(degrees_per_pi_half))
    ax1.plot(angles, phase_accumulation, 'bo-', linewidth=2)
    ax1.axhline(y=90, color='r', linestyle='--', label='Target (90°)')
    ax1.axvline(x=reference_angle, color='k', linestyle='--', label=f'Reference Angle ({reference_angle}°)')
    
    ax1.set_ylabel('Phase Accumulation (degrees)')
    ax1.set_title('Phase Accumulation vs. Scattering Angle', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Polarization state
    ax2 = plt.subplot(gs[1])
    #ax2.plot(angles, s3_after_qwp, 'go-', linewidth=2, label='S3 (Circular Polarization)')
    ax2.plot(angles, ellipticity_after_qwp, 'mo-', linewidth=2, label='Ellipticity')
    
    ax2.axhline(y=1, color='r', linestyle='--', label='Perfect Circular')
    ax2.axvline(x=reference_angle, color='k', linestyle='--')
    
    ax2.set_xlabel('Scattering Angle (degrees)')
    ax2.set_ylabel('Polarization Metrics')
    ax2.set_title('Polarization State vs. Scattering Angle', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.1, 1.1)
    ax2.legend(loc="lower left")
    
    plt.tight_layout()
    plt.show()
    
    # Figure 3: Poincaré sphere visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create points for each angle
    s1_points = np.array([results_by_angle[angle]['pol_at_interfaces']['after_biref']['s1'][0] for angle in angles])
    s2_points = np.array([results_by_angle[angle]['pol_at_interfaces']['after_biref']['s2'][0] for angle in angles])
    s3_points = np.array([results_by_angle[angle]['pol_at_interfaces']['after_biref']['s3'][0] for angle in angles])
    
    # Plot trajectory on Poincaré sphere
    ax.plot(s1_points, s2_points, s3_points, 'bo-', markersize=8)
    
    # Add points and labels for each angle
    for i, angle in enumerate(angles):
        ax.text(s1_points[i], s2_points[i], s3_points[i], f'{angle:.0f}°', fontsize=9)
    
    # Plot sphere wireframe
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.2)
    
    # Mark key points on sphere
    ax.scatter([0], [0], [1], color='r', s=100, label='Right Circular')
    ax.scatter([0], [0], [-1], color='g', s=100, label='Left Circular')
    ax.scatter([1], [0], [0], color='b', s=50, label='Linear Horizontal')
    ax.scatter([-1], [0], [0], color='c', s=50, label='Linear Vertical')
    ax.scatter([0], [1], [0], color='m', s=50, label='Linear +45°')
    ax.scatter([0], [-1], [0], color='y', s=50, label='Linear -45°')
    
    # Set labels and title
    ax.set_xlabel('S1')
    ax.set_ylabel('S2')
    ax.set_zlabel('S3')
    ax.set_title('Polarization State on Poincaré Sphere\nfor Different Scattering Angles', fontsize=14)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()

    print("Study complete.")
    return results_by_angle

def visualize_polarization_evolution(results_by_angle, selected_angles=None):
    """Create a pcolormap showing polarization evolution along the beam path
    
    Parameters:
    -----------
    results_by_angle : dict
        Results from angular_dependence_study indexed by angles
    selected_angles : list, optional
        List of angles to plot, if None will use 3-4 representative angles
    """
    if selected_angles is None:
        # Choose a few representative angles if not specified
        all_angles = sorted(results_by_angle.keys())
        selected_angles = [all_angles[0], all_angles[len(all_angles)//3], 
                          all_angles[2*len(all_angles)//3], all_angles[-1]]
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(selected_angles), 3, figsize=(15, 4*len(selected_angles)))
    
    # Set up common colormap ranges
    s3_vmin, s3_vmax = -1, 1  # S3 (circular polarization)
    ellip_vmin, ellip_vmax = -1, 1  # Ellipticity
    orient_vmin, orient_vmax = -90, 90  # Orientation in degrees
    
    # Plot titles and colorbars
    titles = ['S3 (Circular Polarization)', 'Ellipticity', 'Orientation (degrees)']
    cmaps = ['RdBu', 'RdBu', 'jet']
    
    for i, angle in enumerate(selected_angles):
        result = results_by_angle[angle]
        
        # Get fields and calculate polarization parameters across all space
        E_TE = result['Etot_TE']
        E_TM = result['Etot_TM']
        
        # Create meshgrid for pcolor
        X, Z = np.meshgrid(xv*1e6, zprop*1e6)  # convert to microns
        
        # Initialize arrays for polarization parameters
        s3 = np.zeros_like(X)
        ellipticity = np.zeros_like(X)
        orientation = np.zeros_like(X)
        
        # Calculate polarization parameters for each point
        for z_idx in range(nz):
            # Calculate polarization parameters across x dimension for this z
            E_TE_z = E_TE[z_idx, :]
            E_TM_z = E_TM[z_idx, :]
            
            # Normalize intensities to avoid issues in low-intensity regions
            intensity = np.abs(E_TE_z)**2 + np.abs(E_TM_z)**2
            mask = intensity > 0.01 * np.max(intensity)  # Only calculate where intensity is significant
            
            pol_params = calculate_polarization_parameters(E_TE_z, E_TM_z)
            
            # Store values where intensity is significant
            s3[z_idx, mask] = pol_params['s3'][mask]
            ellipticity[z_idx, mask] = pol_params['ellipticity'][mask]
            orientation[z_idx, mask] = pol_params['orientation'][mask]
        
        # Plot the three polarization parameters
        for j, (param, title, cmap, vmin, vmax) in enumerate(zip(
                [s3, ellipticity, orientation],
                titles, cmaps, 
                [s3_vmin, ellip_vmin, orient_vmin],
                [s3_vmax, ellip_vmax, orient_vmax])):
            
            ax = axes[i, j] if len(selected_angles) > 1 else axes[j]
            
            # Plot polarization parameter
            pcm = ax.pcolormesh(X, Z, param, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
            
            # Mark regions
            ax.axhline(biref_start*1e6, color='green', linestyle='-', linewidth=1.5, 
                      label='Birefringent region')
            ax.axhline(biref_end*1e6, color='green', linestyle='-', linewidth=1.5)
            ax.axhline(z_interface*1e6, color='blue', linestyle='-', linewidth=1.5, 
                      label='Silica-Air Interface')
            
            # Add colorbar
            cbar = plt.colorbar(pcm, ax=ax)
            
            # Set titles and labels
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(f'z (μm)\nAngle = {angle:.1f}°')
            if i == len(selected_angles)-1:
                ax.set_xlabel('x (μm)')
            
            # Only show legend for first plot
            if i == 0 and j == 0:
                ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.show()

# Run the study
if __name__ == "__main__":
    results = angular_dependence_study()
    #visualize_polarization_evolution(results, [0, 10, 20])
    phiTE = np.angle(results[0]['Etot_TE'])
    phiTM = np.angle(results[0]['Etot_TM'])
    plt.figure(figsize=(8,6))
    plt.pcolormesh(zprop*1e6, xv*1e6, phiTE.T - phiTM.T,
                   shading='auto',
                   cmap='twilight',          # cyclic colormap for phase
                   vmin=-np.pi, vmax=+np.pi)
    plt.colorbar(label='TE phase (rad)')
    plt.xlabel('z (µm)')
    plt.ylabel('x (µm)')
    plt.tight_layout()
    plt.show()
