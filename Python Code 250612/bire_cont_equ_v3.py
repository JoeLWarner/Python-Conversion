# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 11:08:40 2025

@author: joeld
"""

import numpy as np
import matplotlib.pyplot as plt

# Material properties
n_silica = 1.46    # silica refractive index
n_air = 1.0        # air refractive index
n_o = 1.658        # calcite ordinary ray
n_e = 1.486        # calcite extraordinary ray

def solve_8x8_system(wavelength_nm, theta_inc, calcite_thickness_um, A1s, A1p, crystal_rotation_deg):
    """
    Solve the complete 8x8 boundary condition system for a 3-layer structure
    
    The 8 unknowns are:
    [B1s, B1p, A2s, B2s, A2p, B2p, A3s, A3p]
    
    Parameters:
    wavelength_nm: wavelength in nanometers
    theta_inc: incident angle in radians
    calcite_thickness_um: calcite layer thickness in micrometers
    A1s: incident s-wave amplitude
    A1p: incident p-wave amplitude 
    crystal_rotation_deg: rotation of crystal optical axis in degrees (default 0)
                         0° = fast axis along s-direction
                         45° = fast axis at 45° to s-direction
                         90° = fast axis along p-direction
    
    Returns:
    field_amplitudes: dictionary with all field amplitudes
    output_fields: complex amplitudes of transmitted fields [Es_out, Ep_out]
    phase_difference: phase difference between s and p output (degrees)
    reflectance: [Rs, Rp] reflection coefficients
    transmittance: [Ts, Tp] transmission coefficients
    crystal_info: dictionary with crystal rotation information
    """
    
    # Convert to SI units
    wavelength = wavelength_nm * 1e-9
    k0 = 2 * np.pi / wavelength
    calcite_thickness = calcite_thickness_um * 1e-6
    
    # Convert crystal rotation to radians
    crystal_rotation_rad = crystal_rotation_deg * np.pi / 180
    
    # Transform incident fields to crystal coordinate system
    # The crystal's fast and slow axes are rotated by crystal_rotation_deg
    # from the lab s,p coordinates
    cos_rot = np.cos(crystal_rotation_rad)
    sin_rot = np.sin(crystal_rotation_rad)
    
    # Rotation matrix to transform from lab (s,p) to crystal (fast,slow) coordinates
    # [E_fast]   [cos(θ)  sin(θ)] [E_s]
    # [E_slow] = [-sin(θ) cos(θ)] [E_p]
    E_fast_incident = cos_rot * A1s + sin_rot * A1p
    E_slow_incident = -sin_rot * A1s + cos_rot * A1p
    
    # In calcite, fast axis corresponds to extraordinary ray (lower n_e)
    # and slow axis corresponds to ordinary ray (higher n_o)
    A1_extraordinary = E_fast_incident  # Extraordinary ray amplitude
    A1_ordinary = E_slow_incident       # Ordinary ray amplitude
    
    # Calculate wave vector components
    kx = n_silica * k0 * np.sin(theta_inc)
    # Check for total internal reflection
    if kx/k0 > n_air:
        print(f"Warning: Total internal reflection may occur at exit interface")
    
    # z-components of wave vectors in each medium
    kz1 = k0 * np.sqrt(n_silica**2 - (kx/k0)**2 + 0j)  # silica
    kz2_o = k0 * np.sqrt(n_o**2 - (kx/k0)**2 + 0j)     # calcite ordinary
    kz2_e = k0 * np.sqrt(n_e**2 - (kx/k0)**2 + 0j)     # calcite extraordinary  
    kz3 = k0 * np.sqrt(n_air**2 - (kx/k0)**2 + 0j)     # air
    
    # Phase factors through calcite layer
    beta_o = kz2_o * calcite_thickness
    beta_e = kz2_e * calcite_thickness
    
    # Set up 8x8 matrix system
    # Variables: [B1s, B1p, A2s, B2s, A2p, B2p, A3s, A3p]
    
    A = np.zeros((8, 8), dtype=complex)
    b = np.zeros(8, dtype=complex)
    
    # Boundary conditions at interface 1 (silica/calcite, z=0)
    
    # Equation 1: Ex continuity for s-polarization
    # A1s + B1s = A2s + B2s
    A[0, 0] = 1   # B1s
    A[0, 2] = -1  # A2s
    A[0, 3] = -1  # B2s
    b[0] = -A1s
    
    # Equation 2: Hy continuity for s-polarization  
    # kz1*(A1s - B1s) = kz2_o*(A2s - B2s)
    A[1, 0] = -kz1    # B1s
    A[1, 2] = -kz2_o  # A2s
    A[1, 3] = kz2_o   # B2s
    b[1] = -kz1 * A1s
    
    # Equation 3: Ey continuity for p-polarization
    # A1p + B1p = A2p + B2p
    A[2, 1] = 1   # B1p
    A[2, 4] = -1  # A2p
    A[2, 5] = -1  # B2p
    b[2] = -A1p
    
    # Equation 4: Hx continuity for p-polarization
    # (A1p - B1p)*kz1/n_silica^2 = (A2p - B2p)*kz2_e/n_e^2
    A[3, 1] = -kz1 / n_silica**2     # B1p
    A[3, 4] = -kz2_e / n_e**2        # A2p
    A[3, 5] = kz2_e / n_e**2         # B2p
    b[3] = -kz1 * A1p / n_silica**2
    
    # Boundary conditions at interface 2 (calcite/air, z=calcite_thickness)
    
    # Equation 5: Field continuity - extraordinary ray to output s-component
    # A2_extraordinary*exp(i*beta_e) + B2_extraordinary*exp(-i*beta_e) transforms back to lab frame
    # The extraordinary ray contributes to both s and p components in lab frame
    A[4, 2] = cos_rot * np.exp(1j * beta_e)    # A2_extraordinary to A3s
    A[4, 3] = cos_rot * np.exp(-1j * beta_e)   # B2_extraordinary to A3s
    A[4, 4] = -sin_rot * np.exp(1j * beta_o)   # A2_ordinary to A3s
    A[4, 5] = -sin_rot * np.exp(-1j * beta_o)  # B2_ordinary to A3s
    A[4, 6] = -1                               # A3s
    b[4] = 0
    
    # Equation 6: Magnetic field continuity for s-component
    A[5, 2] = cos_rot * kz2_e * np.exp(1j * beta_e)    # A2_extraordinary
    A[5, 3] = -cos_rot * kz2_e * np.exp(-1j * beta_e)   # B2_extraordinary
    A[5, 4] = sin_rot * kz2_o * np.exp(1j * beta_o)    # A2_ordinary
    A[5, 5] = -sin_rot * kz2_o * np.exp(-1j * beta_o)   # B2_ordinary
    A[5, 6] = -kz3                                      # A3s
    b[5] = 0
    
    # Equation 7: Field continuity - extraordinary ray to output p-component
    A[6, 2] = sin_rot * np.exp(1j * beta_e)    # A2_extraordinary to A3p
    A[6, 3] = sin_rot * np.exp(-1j * beta_e)   # B2_extraordinary to A3p
    A[6, 4] = cos_rot * np.exp(1j * beta_o)    # A2_ordinary to A3p
    A[6, 5] = cos_rot * np.exp(-1j * beta_o)   # B2_ordinary to A3p
    A[6, 7] = -1                               # A3p
    b[6] = 0
    
    # Equation 8: Magnetic field continuity for p-component
    A[7, 2] = sin_rot * kz2_e * np.exp(1j * beta_e) / n_e**2     # A2_extraordinary
    A[7, 3] = -sin_rot * kz2_e * np.exp(-1j * beta_e) / n_e**2   # B2_extraordinary
    A[7, 4] = -cos_rot * kz2_o * np.exp(1j * beta_o) / n_o**2    # A2_ordinary
    A[7, 5] = cos_rot * kz2_o * np.exp(-1j * beta_o) / n_o**2    # B2_ordinary
    A[7, 7] = -kz3 / n_air**2                                    # A3p
    b[7] = 0
    
    # Solve the linear system
    solution = np.linalg.solve(A, b)
    
    # Extract field amplitudes
    B1s, B1p, A2_extraordinary, B2_extraordinary, A2_ordinary, B2_ordinary, A3s, A3p = solution
    
    # Store all field amplitudes
    field_amplitudes = {
        'A1s': A1s,                           # incident s-wave
        'A1p': A1p,                           # incident p-wave
        'B1s': B1s,                           # reflected s-wave
        'B1p': B1p,                           # reflected p-wave
        'A2_extraordinary': A2_extraordinary, # forward extraordinary wave in calcite
        'B2_extraordinary': B2_extraordinary, # backward extraordinary wave in calcite  
        'A2_ordinary': A2_ordinary,           # forward ordinary wave in calcite
        'B2_ordinary': B2_ordinary,           # backward ordinary wave in calcite
        'A3s': A3s,                           # transmitted s-wave
        'A3p': A3p                            # transmitted p-wave
    }
    
    # Crystal rotation information
    crystal_info = {
        'rotation_deg': crystal_rotation_deg,
        'rotation_rad': crystal_rotation_rad,
        'E_fast_incident': E_fast_incident,
        'E_slow_incident': E_slow_incident,
        'cos_rot': cos_rot,
        'sin_rot': sin_rot
    }
    
    # Output fields (transmitted)
    output_fields = np.array([A3s, A3p])
    
    # Phase difference between s and p output
    phase_s = np.angle(A3s)
    phase_p = np.angle(A3p) 
    phase_difference = (phase_p - phase_s) * 180 / np.pi
    
# =============================================================================
#     # Normalize phase difference to [-180, 180] range
#     while phase_difference > 180:
#         phase_difference -= 360
#     while phase_difference < -180:
#         phase_difference += 360
# =============================================================================
    
    # Calculate reflection and transmission coefficients
    Rs = abs(B1s)**2 / abs(A1s)**2 if A1s != 0 else 0
    Rp = abs(B1p)**2 / abs(A1p)**2 if A1p != 0 else 0
    Ts = abs(A3s)**2 * np.real(kz3) / (abs(A1s)**2 * np.real(kz1)) if A1s != 0 else 0
    Tp = abs(A3p)**2 * np.real(kz3) / (abs(A1p)**2 * np.real(kz1)) if A1p != 0 else 0
    
    reflectance = np.array([Rs, Rp])
    transmittance = np.array([Ts, Tp])
    print(f"⟹ Output amplitudes: Es={A3s:.4f}, Ep={A3p:.4f}")
    print(f"⟹ Phase ret.: {phase_difference:.1f}°") 
    
    return field_amplitudes, output_fields, phase_difference, reflectance, transmittance, crystal_info

# Example usage and testing
if __name__ == "__main__":
    
    fa, out, Δφ, R, T, CI = solve_8x8_system(
        wavelength_nm=633,
        theta_inc= 0.0,  # normal incidence
        calcite_thickness_um=0.928,
        A1s=1.0,  # s-polarized incident light
        A1p=1.0,   # no p-polarized component
        crystal_rotation_deg=0
    )

print(f"Transmitted phase difference = {Δφ:.2f}°")