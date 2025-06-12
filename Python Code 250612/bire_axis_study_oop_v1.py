#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 14:48:16 2025

@author: joel
"""

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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
from silica_my import silica_n
from tqdm.notebook import tqdm

# Wavelength and constants
lam = 780e-9
k0 = 2*np.pi/lam
nsil = silica_n(lam)
n_air = 1
neff = 1.455879904809160

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

def calculate_tm_projection(angle_rad, optic_axis=np.array([0, 0, 1])):
    """
    Given an incident angle (in the x–z plane) and an optic-axis unit vector,
    return the TM field direction and its projections onto:
      • the extraordinary axis (optic_axis)
      • the ordinary subspace (perp to optic_axis)
    """
    # 1) propagation direction in 3D
    k_vec = np.array([np.sin(angle_rad), 0.0, np.cos(angle_rad)])
    k_hat = k_vec / np.linalg.norm(k_vec)

    # 2) TM field is perpendicular to both k̂ and the TE (ŷ) direction
    E_TM_dir = np.cross(k_hat, np.array([0.0, 1.0, 0.0]))
    E_TM_dir /= np.linalg.norm(E_TM_dir)

    # 3) projection onto extraordinary (optic_axis) and ordinary subspace
    optic_axis = optic_axis / np.linalg.norm(optic_axis)
    proj_extra = np.dot(E_TM_dir, optic_axis)
    # ordinary component is what's left in magnitude, but for sign pick any orthonormal complement:
    proj_ord = np.sqrt(max(0.0, 1 - proj_extra**2))

    return {
        'E_TM_direction': E_TM_dir,
        'projection_extraordinary': proj_extra,
        'projection_ordinary':   proj_ord,
        'angle_rad':             angle_rad,
        'angle_deg':             angle_rad * 180/np.pi
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

def run_qwp_simulation(angle_deg, qwp_params):
    """Run simulation at specified angle using a pure out-of-plane optic axis"""
    angle_rad = angle_deg * np.pi / 180
    ntar = np.array([np.sin(angle_rad), np.cos(angle_rad)])

    # Extract QWP parameters
    n_ordinary = qwp_params['n_ordinary']
    n_extraordinary = qwp_params['n_extraordinary']

    # For an out-of-plane axis (along ẑ), both TE (ȳ) and TM (x̂) lie in the slow (ordinary) subspace:
    n_TE = n_ordinary
    n_TM = n_ordinary

    # Build the initial 45°-polarized pump
    initial_pump = build_tilted_gaussian(w0_pump, z_start, ntar)
    E_TE = initial_pump / np.sqrt(2)
    E_TM = initial_pump / np.sqrt(2)

    # Allocate storage
    Etot_TE = np.zeros((nz, nx), dtype=complex)
    Etot_TM = np.zeros((nz, nx), dtype=complex)
    Etot_TE[0, :] = E_TE
    Etot_TM[0, :] = E_TM

    E_TE_current = E_TE.copy()
    E_TM_current = E_TM.copy()

    phase_diff_center_raw = np.zeros(nz)
    ellipticity = np.zeros(nz)
    orientation = np.zeros(nz)
    s3_values = np.zeros(nz)

    # Propagate step by step
    for i in range(1, nz):
        dz = zprop[i] - zprop[i-1]

        # At the silica-air interface, apply Fresnel
        if i-1 < idx_interface <= i:
            E_k_TE_air, E_k_TM_air = apply_fresnel(E_TE_current, E_TM_current)
            remaining_dz = zprop[i] - z_interface

            # back to spatial
            E_TE_interface = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(E_k_TE_air)))
            E_TM_interface = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(E_k_TM_air)))

            E_TE_current = angular_spectrum_step(E_TE_interface, remaining_dz, n_air)
            E_TM_current = angular_spectrum_step(E_TM_interface, remaining_dz, n_air)

        else:
            # Regular step
            if idx_biref_start <= i <= idx_biref_end:
                # In the birefringent layer: both polarizations see the ordinary index
                E_TE_current = angular_spectrum_step(E_TE_current, dz, n_TE)
                E_TM_current = angular_spectrum_step(E_TM_current, dz, n_TM)
            elif i <= idx_interface:
                # In silica before interface
                E_TE_current = angular_spectrum_step(E_TE_current, dz, nsil)
                E_TM_current = angular_spectrum_step(E_TM_current, dz, nsil)
            else:
                # In air after interface
                E_TE_current = angular_spectrum_step(E_TE_current, dz, n_air)
                E_TM_current = angular_spectrum_step(E_TM_current, dz, n_air)

        Etot_TE[i, :] = E_TE_current
        Etot_TM[i, :] = E_TM_current

        # Compute polarization at beam center
        center_idx = np.argmax(np.abs(E_TE_current)**2)
        E_TE_c = E_TE_current[center_idx]
        E_TM_c = E_TM_current[center_idx]

        pol = calculate_polarization_parameters(
            np.array([E_TE_c]), np.array([E_TM_c])
        )
        ellipticity[i] = pol['ellipticity'][0]
        orientation[i] = pol['orientation'][0]
        s3_values[i] = pol['s3'][0]

        phi_TE = np.angle(E_TE_c)
        phi_TM = np.angle(E_TM_c)
        phase_diff_center_raw[i] = phi_TE - phi_TM

    # Unwrap and package results
    phase_diff_center = np.unwrap(phase_diff_center_raw)
    pol_at_interfaces = {
        name: calculate_polarization_parameters(
            Etot_TE[idx, np.argmax(np.abs(Etot_TE[idx])**2)].reshape(1),
            Etot_TM[idx, np.argmax(np.abs(Etot_TM[idx])**2)].reshape(1)
        )
        for name, idx in {
            'initial': 0,
            'before_biref': idx_biref_start-1,
            'after_biref': idx_biref_end,
            'final': nz-1
        }.items()
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


def angular_dependence_study(optic_axis=np.array([0, 0, 1])):
    """Study how QWP performance varies with angle and a given optic‐axis orientation"""
    
    print("Angular Dependence Study of QWP Performance")
    print("-" * 50)
    
    # 1. Design QWP for reference angle (0°)
    reference_angle = 0
    print(f"Designing QWP for reference angle: {reference_angle}°")
    qwp_params = design_qwp_for_reference_angle(reference_angle)
    print(f"Birefringent layer parameters:")
    print(f"  Thickness: {biref_thickness*1e6:.2f} µm")
    print(f"  Target phase shift: {qwp_params['target_phase_shift']*180/np.pi:.1f}°")
    print(f"  Required birefringence (Δn): {qwp_params['delta_n']:.6f}")
    print(f"  TE index (extraordinary): {qwp_params['n_extraordinary']:.6f}")
    print(f"  TM index (ordinary): {qwp_params['n_ordinary']:.6f}")
    print()
    
    # 2. Study angle range
    angles = np.linspace(0, 90, 10)
    phase_accumulation        = np.zeros_like(angles)
    s3_after_qwp              = np.zeros_like(angles)
    ellipticity_after_qwp     = np.zeros_like(angles)
    proj_extraordinary        = np.zeros_like(angles)
    proj_ordinary             = np.zeros_like(angles)
    results_by_angle          = {}
    
    for i, angle in enumerate(angles):
        print(f"Processing angle: {angle:.1f}°...")
        rad = angle * np.pi/180
        
        # TM projections onto this optic_axis
        tm_proj = calculate_tm_projection(rad, optic_axis=optic_axis)
        proj_extraordinary[i] = tm_proj['projection_extraordinary']
        proj_ordinary[i]      = tm_proj['projection_ordinary']
        
        # Run the QWP sim
        result = run_qwp_simulation(angle, qwp_params)
        results_by_angle[angle] = result
        
        # extract metrics
        phase_accumulation[i]    = result['total_phase_accumulated']*180/np.pi
        s3_after_qwp[i]          = result['pol_at_interfaces']['after_biref']['s3'][0]
        ellipticity_after_qwp[i] = result['pol_at_interfaces']['after_biref']['ellipticity'][0]
        
        print(f"  • Phase:    {phase_accumulation[i]:.2f}°")
        print(f"  • S3:       {s3_after_qwp[i]:.3f}")
        print(f"  • Ellip.:   {ellipticity_after_qwp[i]:.3f}")
        print(f"  • projₑ:    {proj_extraordinary[i]:.3f}")
        print(f"  • projₒ:    {proj_ordinary[i]:.3f}\n")
    
    # --- Visualization of TM projection ---
    plt.figure(figsize=(8,5))
    plt.plot(angles, proj_extraordinary, 'o-', label='Extraordinary')
    plt.plot(angles, proj_ordinary,      's--', label='Ordinary')
    plt.axvline(reference_angle, color='k', ls=':')
    plt.xlabel('Incidence Angle (°)')
    plt.ylabel('|E| projection')
    plt.title('TM Field Projections onto QWP Axes')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Figure 2: QWP Performance Metrics
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


# Run the study
if __name__ == "__main__":
    angles = np.linspace(0, 90, 10) 
    results = angular_dependence_study()
# =============================================================================
#     #visualize_polarization_evolution(results, [0, 10, 20])
#     #visualize_polarization_evolution(results, selected_angles=[0, 10, 20], 
#     #                           apply_mask=False, show_phase_diff=True)
#     phiTE = np.angle(results[0]['Etot_TE'])
#     phiTM = np.angle(results[0]['Etot_TM'])
#     plt.figure(figsize=(8,6))
#     plt.pcolormesh(zprop*1e6, xv*1e6, phiTE.T - phiTM.T,
#                    shading='auto',
#                    cmap='twilight',          # cyclic colormap for phase
#                    vmin=-np.pi, vmax=+np.pi)
#     plt.colorbar(label='TE phase (rad)')
#     plt.xlabel('z (µm)')
#     plt.ylabel('x (µm)')
#     plt.tight_layout()
#     plt.show()
# =============================================================================
