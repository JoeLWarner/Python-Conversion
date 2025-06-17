# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 17:36:19 2025

@author: joeld

Fixed 3-Layer Birefringent System Solver
Solves boundary conditions for silica/calcite/air structure
"""

import numpy as np
import matplotlib.pyplot as plt

# Material properties
n_silica = 1.46    # silica refractive index
n_air = 1.0        # air refractive index
n_o = 1.658        # calcite ordinary ray
n_e = 1.486        # calcite extraordinary ray

def solve_8x8_system(wavelength_nm, theta_inc, calcite_thickness_um, A1s, A1p):
    """
    Solve the complete 8x8 boundary condition system for a 3-layer structure
    
    The 8 unknowns are:
    [B1s, B1p, A2s, B2s, A2p, B2p, A3s, A3p]
    where A1s = 1 (incident s-wave), A1p = 0 (no incident p-wave)
    
    Parameters:
    wavelength_nm: wavelength in nanometers
    theta_inc: incident angle in radians
    calcite_thickness_um: calcite layer thickness in micrometers
    
    Returns:
    field_amplitudes: dictionary with all field amplitudes
    output_fields: complex amplitudes of transmitted fields [Es_out, Ep_out]
    phase_difference: phase difference between s and p output (degrees)
    reflectance: [Rs, Rp] reflection coefficients
    transmittance: [Ts, Tp] transmission coefficients
    """
    
    # Convert to SI units
    wavelength = wavelength_nm * 1e-9
    k0 = 2 * np.pi / wavelength
    calcite_thickness = calcite_thickness_um * 1e-6
    
    # Calculate wave vector components
    kx = n_silica * k0 * np.sin(theta_inc)
    
    # z-components of wave vectors in each medium
    kz1 = k0 * np.sqrt(n_silica**2 - (kx/k0)**2)  # silica
    kz2_o = k0 * np.sqrt(n_o**2 - (kx/k0)**2)     # calcite ordinary
    kz2_e = k0 * np.sqrt(n_e**2 - (kx/k0)**2)     # calcite extraordinary  
    kz3 = k0 * np.sqrt(n_air**2 - (kx/k0)**2)     # air
    
    # Phase factors through calcite layer
    beta_o = kz2_o * calcite_thickness
    beta_e = kz2_e * calcite_thickness
    
    # Set up 8x8 matrix system
    # Variables: [B1s, B1p, A2s, B2s, A2p, B2p, A3s, A3p]
    # We know A1s = 1 (incident s-wave), A1p = 0 (no incident p-wave)
    
    A = np.zeros((8, 8), dtype=complex)
    b = np.zeros(8, dtype=complex)
    
    # Boundary conditions at interface 1 (silica/calcite, z=0)
    
    # Equation 1: Ex continuity for s-polarization
    # A1s + B1s = A2s + B2s  =>  1 + B1s = A2s + B2s
    A[0, 0] = 1   # B1s
    A[0, 2] = -1  # A2s
    A[0, 3] = -1  # B2s
    b[0] = -A1s     # -A1s
    
    # Equation 2: Hy continuity for s-polarization  
    # kz1*(A1s - B1s) = kz2_o*(A2s - B2s)  =>  kz1*(1 - B1s) = kz2_o*(A2s - B2s)
    A[1, 0] = -kz1    # B1s
    A[1, 2] = -kz2_o  # A2s
    A[1, 3] = kz2_o   # B2s
    b[1] = -kz1 * A1s      # -kz1*A1s
    
    # Equation 3: Ey continuity for p-polarization
    # A1p + B1p = A2p + B2p  =>  0 + B1p = A2p + B2p
    A[2, 1] = 1   # B1p
    A[2, 4] = -1  # A2p
    A[2, 5] = -1  # B2p
    b[2] = -A1p      # -A1p = 0
    
    # Equation 4: Hx continuity for p-polarization
    # (A1p - B1p)*kz1/n_silica^2 = (A2p - B2p)*kz2_e/n_e^2
    # (0 - B1p)*kz1/n_silica^2 = (A2p - B2p)*kz2_e/n_e^2
    A[3, 1] = -kz1 / n_silica**2     # B1p
    A[3, 4] = -kz2_e / n_e**2        # A2p
    A[3, 5] = kz2_e / n_e**2         # B2p
    b[3] = -kz1 * A1p / n_silica**2
    
    # Boundary conditions at interface 2 (calcite/air, z=calcite_thickness)
    
    # Equation 5: Ex continuity for s-polarization
    # A2s*exp(i*beta_o) + B2s*exp(-i*beta_o) = A3s
    A[4, 2] = np.exp(1j * beta_o)   # A2s
    A[4, 3] = np.exp(-1j * beta_o)  # B2s
    A[4, 6] = -1                    # A3s
    b[4] = 0
    
    # Equation 6: Hy continuity for s-polarization
    # kz2_o*(A2s*exp(i*beta_o) - B2s*exp(-i*beta_o)) = kz3*A3s
    A[5, 2] = kz2_o * np.exp(1j * beta_o)   # A2s
    A[5, 3] = -kz2_o * np.exp(-1j * beta_o) # B2s
    A[5, 6] = -kz3                          # A3s
    b[5] = 0
    
    # Equation 7: Ey continuity for p-polarization
    # A2p*exp(i*beta_e) + B2p*exp(-i*beta_e) = A3p
    A[6, 4] = np.exp(1j * beta_e)   # A2p
    A[6, 5] = np.exp(-1j * beta_e)  # B2p
    A[6, 7] = -1                    # A3p
    b[6] = 0
    
    # Equation 8: Hx continuity for p-polarization
    # (A2p*exp(i*beta_e) - B2p*exp(-i*beta_e))*kz2_e/n_e^2 = A3p*kz3/n_air^2
    A[7, 4] = kz2_e * np.exp(1j * beta_e) / n_e**2    # A2p
    A[7, 5] = -kz2_e * np.exp(-1j * beta_e) / n_e**2  # B2p
    A[7, 7] = -kz3 / n_air**2                         # A3p
    b[7] = 0
    
    # Solve the linear system
    solution = np.linalg.solve(A, b)
    
    # Extract field amplitudes
    B1s, B1p, A2s, B2s, A2p, B2p, A3s, A3p = solution
    
    # Store all field amplitudes
    field_amplitudes = {
        'A1s': A1s,    # incident s-wave (given)
        'A1p': A1p,    # incident p-wave (given) 
        'B1s': B1s,    # reflected s-wave
        'B1p': B1p,    # reflected p-wave
        'A2s': A2s,    # forward s-wave in calcite
        'B2s': B2s,    # backward s-wave in calcite  
        'A2p': A2p,    # forward p-wave in calcite
        'B2p': B2p,    # backward p-wave in calcite
        'A3s': A3s,    # transmitted s-wave
        'A3p': A3p     # transmitted p-wave
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
    Rs = abs(B1s)**2  # s-polarization reflectance
    Rp = abs(B1p)**2  # p-polarization reflectance  
    Ts = abs(A3s)**2 * np.real(kz3) / np.real(kz1)  # s-polarization transmittance
    Tp = abs(A3p)**2 * np.real(kz3) / np.real(kz1)  # p-polarization transmittance
    
    reflectance = np.array([Rs, Rp])
    transmittance = np.array([Ts, Tp])   
    print(f"⟹ Output amplitudes: Es={A3s:.4f}, Ep={A3p:.4f}")
    print(f"⟹ Phase ret.: {phase_difference:.1f}°") 
    return field_amplitudes, output_fields, phase_difference, reflectance, transmittance


# Example usage and testing
if __name__ == "__main__":
    
    fa, out, Δφ, R, T = solve_8x8_system(
    wavelength_nm=633,
    theta_inc=0,
    calcite_thickness_um=0.928,
    A1s=1,
    A1p=1
)

print(f"Transmitted phase difference = {Δφ:.2f}°")

