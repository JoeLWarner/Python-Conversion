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

def calculate_tm_projection(angle_rad, fast_axis, slow_axis):
    """
    Calculate how TM and TE field components project onto the birefringent axes.
    
    angle_rad : float
        Propagation (incidence) angle in radians.
    fast_axis, slow_axis : array-like, shape (2,)
        Unit vectors (in the x–z plane) giving the in-plane fast and slow axes.
        
    Returns a dict with projections of fields onto the fast/slow axes
    """
    # propagation vector in x–z
    k_vec = np.array([np.sin(angle_rad), 0, np.cos(angle_rad)])
    
    # TE field is always along y-axis (perpendicular to x-z plane)
    E_TE = np.array([0, 1, 0])
    
    # TM field is perpendicular to both k and TE
    E_TM = np.cross(k_vec, E_TE)
    E_TM = E_TM / np.linalg.norm(E_TM)  # Normalize to unit vector
    
    # For projection, we need to extend fast/slow axes to 3D
    fast_axis_3d = np.array([fast_axis[0], 0, fast_axis[1]])
    slow_axis_3d = np.array([slow_axis[0], 0, slow_axis[1]])
    
    # Normalize 3D axes (should already be normalized but just to be safe)
    fast_axis_3d = fast_axis_3d / np.linalg.norm(fast_axis_3d)
    slow_axis_3d = slow_axis_3d / np.linalg.norm(slow_axis_3d)
    
    # Calculate projections using dot products
    proj_fast_TE = np.dot(E_TE, fast_axis_3d)
    proj_slow_TE = np.dot(E_TE, slow_axis_3d)
    proj_fast_TM = np.dot(E_TM, fast_axis_3d)
    proj_slow_TM = np.dot(E_TM, slow_axis_3d)
    
    return {
        'E_TM_direction': E_TM,
        'proj_fast_TM': proj_fast_TM,
        'proj_slow_TM': proj_slow_TM,
        'proj_fast_TE': proj_fast_TE,
        'proj_slow_TE': proj_slow_TE,
        'angle_rad': angle_rad,
        'angle_deg': np.degrees(angle_rad)
    }

axis_angle = 180

def design_qwp_for_reference_angle(ref_angle_deg, axis_angle_deg=axis_angle):
    """
    Design a QWP for a specific reference incidence angle
    and an in‐plane optic‐axis rotation.
    Returns the required birefringence (delta_n) and the fast/slow axes.
    """
    # convert to radians
    ref_rad  = np.deg2rad(ref_angle_deg)
    axis_rad = np.deg2rad(axis_angle_deg)
    
    # target quarter‐wave
    target_phase_shift = np.pi/2
    
    # path‐length factor in the birefringent layer
    path_factor = 1.0/np.cos(ref_rad)
    
    # compute Δn
    delta_n = target_phase_shift/(k0 * biref_thickness * path_factor)
    
    # indices
    n_o = nsil
    n_e = n_o + delta_n
    
    # build in‐plane fast & slow axes in the x-z plane
    # The 3D axes are:
    # fast_axis_3d = [sin(axis_rad), 0, cos(axis_rad)]
    # slow_axis_3d = [-cos(axis_rad), 0, sin(axis_rad)]
    
    # In the x-z plane representation:
    fast_axis = np.array([np.sin(axis_rad), np.cos(axis_rad)])
    slow_axis = np.array([-np.cos(axis_rad), np.sin(axis_rad)])
    
    return {
        'delta_n': delta_n,
        'n_ordinary': n_o,
        'n_extraordinary': n_e,
        'reference_angle_deg': ref_angle_deg,
        'reference_angle_rad': ref_rad,
        'axis_angle_deg': axis_angle_deg,
        'axis_angle_rad': axis_rad,
        'fast_axis': fast_axis,
        'slow_axis': slow_axis,
        'target_phase_shift': target_phase_shift
    }



def run_qwp_simulation(angle_deg, qwp_params, axis_angle_deg=axis_angle):
    """Run simulation at specified angle using the QWP parameters,
       with in-plane optic axis rotated by axis_angle_deg."""
    angle_rad = np.deg2rad(angle_deg)
    ntar = np.array([np.sin(angle_rad), np.cos(angle_rad)])
    axis_rad = np.deg2rad(axis_angle_deg)

    # build rotated fast & slow axes in the x–y plane
    fast_axis = np.array([ np.sin(axis_rad),  np.cos(axis_rad) ])   # unit vector
    slow_axis = np.array([-np.cos(axis_rad),  np.sin(axis_rad) ])   # +90° ccw

    # Extract QWP indices
    n_o = qwp_params['n_ordinary']
    n_e = qwp_params['n_extraordinary']

    # TE/TM indices *before* projection:
    # we will project each instant TE/TM onto fast/slow below
    n_TE = None
    n_TM = None

    # initial 45° pump in TE/TM basis
    initial_pump = build_tilted_gaussian(w0_pump, z_start, ntar)
    #initial_pump = build_plane_wave(z_start, ntar)
    E_TE = initial_pump / np.sqrt(2)
    E_TM = initial_pump / np.sqrt(2)

    # storage
    Etot_TE = np.zeros((nz, nx), dtype=complex)
    Etot_TM = np.zeros((nz, nx), dtype=complex)
    Etot_TE[0] = E_TE
    Etot_TM[0] = E_TM

    E_TE_current = E_TE.copy()
    E_TM_current = E_TM.copy()

    phase_diff_center_raw = np.zeros(nz)
    ellipticity = np.zeros(nz)
    orientation  = np.zeros(nz)
    s3_values    = np.zeros(nz)

    for i in range(1, nz):
        dz = zprop[i] - zprop[i-1]

        # if crossing to air
        if i-1 < idx_interface <= i:
            E_k_TE_air, E_k_TM_air = apply_fresnel(E_TE_current, E_TM_current)
            rem = zprop[i] - z_interface
            E_TE_interface = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(E_k_TE_air)))
            E_TM_interface = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(E_k_TM_air)))
            E_TE_current = angular_spectrum_step(E_TE_interface, rem, n_air)
            E_TM_current = angular_spectrum_step(E_TM_interface, rem, n_air)

        else:
            # in birefringent region?
            if idx_biref_start <= i <= idx_biref_end:
                # project current TE/TM onto fast/slow axes
                E_vec = np.stack([E_TE_current, E_TM_current], axis=0)  # shape (2,nx)
                # build 2×2 projection matrix from (TE,TM)→(fast,slow)
                P = np.stack([fast_axis, slow_axis], axis=1)  # shape (2,2)
                # but TE-field vector is [0,1] in x–y, TM is perpendicular to ntar; 
                # since we carry only scalar envelopes, we assume basis alignment:
                # treat E_TE_current scalar as ŷ-component, E_TM_current as x̂-component
                # so decomposition is just dot with axes:
                E_fast = fast_axis[1]*E_TE_current + fast_axis[0]*E_TM_current
                E_slow = slow_axis[1]*E_TE_current + slow_axis[0]*E_TM_current

                # propagate along each principal axis
                E_fast = angular_spectrum_step(E_fast, dz, n_e)
                E_slow = angular_spectrum_step(E_slow, dz, n_o)

                # recombine back into TE/TM (invert the 2×2)
                invP = np.linalg.inv(P)
                E_TE_current = invP[1,0]*E_fast + invP[1,1]*E_slow
                E_TM_current = invP[0,0]*E_fast + invP[0,1]*E_slow

            elif i <= idx_interface:
                E_TE_current = angular_spectrum_step(E_TE_current, dz, nsil)
                E_TM_current = angular_spectrum_step(E_TM_current, dz, nsil)
            else:
                E_TE_current = angular_spectrum_step(E_TE_current, dz, n_air)
                E_TM_current = angular_spectrum_step(E_TM_current, dz, n_air)

        Etot_TE[i] = E_TE_current
        Etot_TM[i] = E_TM_current

        # beam-center metrics
        center_idx = np.argmax(np.abs(E_TE_current)**2)
        E_TE_c = E_TE_current[center_idx]
        E_TM_c = E_TM_current[center_idx]

        pp = calculate_polarization_parameters(
            np.array([E_TE_c]), np.array([E_TM_c]))
        phi_TE = np.angle(E_TE_c)
        phi_TM = np.angle(E_TM_c)

        phase_diff_center_raw[i] = phi_TE - phi_TM
        ellipticity[i]  = pp['ellipticity'][0]
        orientation[i]  = pp['orientation'][0]
        s3_values[i]    = pp['s3'][0]

    phase_diff_center = np.unwrap(phase_diff_center_raw)

    pol_at_interfaces = {
        name: calculate_polarization_parameters(
            Etot_TE[idx, :][np.argmax(np.abs(Etot_TE[idx])**2)].reshape(1),
            Etot_TM[idx, :][np.argmax(np.abs(Etot_TM[idx])**2)].reshape(1))
        for name, idx in [
            ('initial',      0),
            ('before_biref', idx_biref_start-1),
            ('after_biref',  idx_biref_end),
            ('final',        nz-1)]
    }

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



# MAIN STUDY FUNCTION
def angular_dependence_study(axis_angle_deg=axis_angle):
    """Study how QWP performance varies with angle and optic‐axis orientation."""
    print("Angular Dependence Study of QWP Performance")
    print("-" * 50)

    # 1. Design QWP for reference angle (0°) with given in‐plane optic‐axis orientation
    reference_angle = 0
    print(f"Designing QWP for reference angle: {reference_angle}° and axis at {axis_angle_deg}° in‐plane")
    qwp_params = design_qwp_for_reference_angle(reference_angle, axis_angle_deg)
    
    fast_axis = qwp_params['fast_axis']
    slow_axis = qwp_params['slow_axis']

    print("Birefringent layer parameters:")
    print(f"  Thickness: {biref_thickness*1e6:.2f} µm")
    print(f"  Target phase shift: {qwp_params['target_phase_shift']*180/np.pi:.1f}°")
    print(f"  Required birefringence (Δn): {qwp_params['delta_n']:.6f}")
    print(f"  Fast‐axis (x,z): {fast_axis}")
    print(f"  Slow‐axis (x,z): {slow_axis}\n")

    # 2. Study angle range
    angles = np.linspace(0, 80, 9)  # Avoid 90° which causes numerical issues

    # Arrays to store results
    phase_accumulation       = np.zeros_like(angles)
    s3_after_qwp             = np.zeros_like(angles)
    ellipticity_after_qwp    = np.zeros_like(angles)
    proj_fast_TM             = np.zeros_like(angles)
    proj_slow_TM             = np.zeros_like(angles)
    proj_fast_TE             = np.zeros_like(angles)
    proj_slow_TE             = np.zeros_like(angles)

    results_by_angle = {}

    for i, angle in enumerate(angles):
        print(f"Processing angle: {angle:.1f}°...")

        angle_rad = np.deg2rad(angle)
        # TM/TE projections onto the in‐plane axes
        tm_info = calculate_tm_projection(angle_rad, fast_axis, slow_axis)
        proj_fast_TM[i] = tm_info['proj_fast_TM']
        proj_slow_TM[i] = tm_info['proj_slow_TM']
        proj_fast_TE[i] = tm_info['proj_fast_TE']
        proj_slow_TE[i] = tm_info['proj_slow_TE']

        # Run full QWP simulation
        result = run_qwp_simulation(angle, qwp_params)
        results_by_angle[angle] = result

        # Key metrics at exit of birefringent layer
        phase_accumulation[i]    = result['total_phase_accumulated'] * 180/np.pi
        s3_after_qwp[i]          = result['pol_at_interfaces']['after_biref']['s3'][0]
        ellipticity_after_qwp[i] = result['pol_at_interfaces']['after_biref']['ellipticity'][0]

        print(f"  • Phase ∆: {phase_accumulation[i]:.2f}°")
        print(f"  • S3 after QWP: {s3_after_qwp[i]:.3f}")
        print(f"  • Ellipticity after QWP: {ellipticity_after_qwp[i]:.3f}")
        print(f"  • TM→fast: {proj_fast_TM[i]:.3f}, TM→slow: {proj_slow_TM[i]:.3f}")
        print(f"  • TE→fast: {proj_fast_TE[i]:.3f}, TE→slow: {proj_slow_TE[i]:.3f}\n")

    # ---------------------------------------------------------
    # VISUALIZATIONS
    # ---------------------------------------------------------
    # 1) TM / TE projections
    plt.figure(figsize=(8,5))
    plt.plot(angles, proj_fast_TM, 'bo-', label='TM→fast')
    plt.plot(angles, proj_slow_TM, 'ro-', label='TM→slow')
    plt.plot(angles, proj_fast_TE, 'b--', label='TE→fast')
    plt.plot(angles, proj_slow_TE, 'r--', label='TE→slow')
    plt.axvline(reference_angle, color='k', ls='--')
    plt.xlabel('Incidence Angle (°)')
    plt.ylabel('Projection onto axis')
    plt.title(f'Field Projections onto In‐Plane Axes (axis at {axis_angle_deg}°)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2) Phase accumulation
    plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    
    # Phase accumulation
    ax1 = plt.subplot(gs[0])
    ax1.plot(angles, phase_accumulation, 'bo-', linewidth=2)
    ax1.axhline(y=90, color='r', linestyle='--', label='Target (90°)')
    ax1.axvline(x=reference_angle, color='k', linestyle='--', label=f'Reference Angle ({reference_angle}°)')
    
    ax1.set_ylabel('Phase Accumulation (degrees)')
    ax1.set_title('Phase Accumulation vs. Scattering Angle', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Polarization state
    ax2 = plt.subplot(gs[1])
    ax2.plot(angles, s3_after_qwp, 'go-', linewidth=2, label='S3 (Circular Polarization)')
    ax2.plot(angles, ellipticity_after_qwp, 'mo-', linewidth=2, label='Ellipticity')
    
    ax2.axhline(y=1, color='r', linestyle='--', label='Perfect Circular')
    ax2.axvline(x=reference_angle, color='k', linestyle='--')
    
    ax2.set_xlabel('Scattering Angle (degrees)')
    ax2.set_ylabel('Polarization Metrics')
    ax2.set_title('Polarization State vs. Scattering Angle', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.1, 1.1)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

    # 3) Polarization on Poincaré
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    s1 = np.array([results_by_angle[a]['pol_at_interfaces']['after_biref']['s1'][0] for a in angles])
    s2 = np.array([results_by_angle[a]['pol_at_interfaces']['after_biref']['s2'][0] for a in angles])
    s3 = np.array([results_by_angle[a]['pol_at_interfaces']['after_biref']['s3'][0] for a in angles])
    ax.plot(s1, s2, s3, 'o-', lw=2)
    for xi, yi, zi, a in zip(s1, s2, s3, angles):
        ax.text(xi, yi, zi, f'{a:.0f}°')
    # sphere wireframe
    u,v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    X = np.cos(u)*np.sin(v); Y = np.sin(u)*np.sin(v); Z = np.cos(v)
    ax.plot_wireframe(X, Y, Z, color='gray', alpha=0.2)
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel('S1'); ax.set_ylabel('S2'); ax.set_zlabel('S3')
    ax.set_title('Exit Polarization on Poincaré Sphere')
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
