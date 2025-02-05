# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:37:19 2025

@author: jw4e24
"""

import numpy as np

def calculate_mode_shape(x, beta, k0, n, d, TE=True):
    """
    Calculate mode shape for arbitrary slab waveguide structure
    
    Parameters:
    x: array - position coordinates
    beta: float - propagation constant
    k0: float - wave number in vacuum
    n: list - refractive indices of layers
    d: list - thicknesses of layers (except first and last semi-infinite regions)
    TE: bool - True for TE mode, False for TM mode
    
    Returns:
    E: array - electric field distribution (Ey for TE, Ex for TM)
    """
    nn = len(n)  # number of layers
    # Initialize coefficients
    A0 = 0
    B0 = 1
    
    # Initialize coefficient arrays for forward and backward prop wave amplitudes
    if TE:
        At = [A0]
        Bt = [B0]
    else: #TM scaled with 1/n^2 factor from wave equation 
        At = [A0/n[0]**2]
        Bt = [B0/n[0]**2]
    
    # Calculate transverse wave vector between two regions at a time and calculates kk for the continuity equations later 
    for in_idx in range(nn-1):
        # Use complex sqrt to handle evanescent fields
        kappa0 = np.sqrt(complex(k0**2 * n[in_idx]**2 - beta**2))
        kappa1 = np.sqrt(complex(k0**2 * n[in_idx+1]**2 - beta**2))
        
        # Calculate impedance ratio
        kk = kappa0/kappa1
        
        if not TE: #TM is scaled again 
            kk = kk * n[in_idx+1]**2/n[in_idx]**2
        
        # Calculate new coefficients
        A1 = 0.5*(1 + kk)*A0 + 0.5*(1 - kk)*B0
        B1 = 0.5*(1 - kk)*A0 + 0.5*(1 + kk)*B0
        
        # Apply phase shift if not at last interface, phase is calcuated as phi = k_x . region thickness, applied with exponential 
        if in_idx < nn-2:
            phase = kappa1*d[in_idx]
            A1 = A1 * np.exp(1j*phase)
            B1 = B1 * np.exp(-1j*phase)
        
        # Update coefficients for next iteration
        A0 = A1
        B0 = B1
        
        # Store coefficients
        if TE: #if TE dont scale otherwise scale by 1/n^2 again 
            At.append(A0)
            Bt.append(B0)
        else:
            At.append(A0/n[in_idx+1]**2)
            Bt.append(B0/n[in_idx+1]**2)
    
    # Calculate field at each point
    E = np.zeros_like(x, dtype=complex)
    
    # Field in first region (x <= 0)
    kappa = np.sqrt(complex(k0**2 * n[0]**2 - beta**2))
    ix = x <= 0
    E[ix] = Bt[0] * np.exp(-1j * kappa * x[ix])
    
    # Field in intermediate regions
    for in_idx in range(1, nn-1):
        x1 = np.sum(d[:in_idx-1]) if in_idx > 0 else 0
        x2 = np.sum(d[:in_idx])
        kappa = np.sqrt(complex(k0**2 * n[in_idx]**2 - beta**2))
        ix = (x > x1) & (x <= x2)
        E[ix] = (At[in_idx] * np.exp(1j * kappa * (x[ix] - x2)) + 
                 Bt[in_idx] * np.exp(-1j * kappa * (x[ix] - x2)))
    
    # Field in last region (x > x1)
    x1 = np.sum(d)
    kappa = np.sqrt(complex(k0**2 * n[-1]**2 - beta**2))
    ix = x > x1
    E[ix] = At[-1] * np.exp(1j * kappa * (x[ix] - x1))
    
    return np.real(E)  # Return real part of the field

