# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:37:19 2025

@author: jw4e24
"""

import numpy as np

'''
def calculate_mode_shape(x, beta, k0, n, d, TE):
    """
    Calculate the field distribution (Ey for TE mode, Ex for TM mode)
    across all regions of the waveguide.
    """
    nn = len(n)  # Number of layers
    
    # Convert inputs to complex numbers
    beta = beta + 0j
    n = np.array(n, dtype=complex)
    x = np.array(x, dtype=complex)
    
    # Initialize starting coefficients
    A0 = 0 + 0j
    B0 = 1 + 0j
    
    # Initialize arrays to store coefficients
    At = [A0/n[0]**2] if not TE else [A0]
    Bt = [B0/n[0]**2] if not TE else [B0]
    
    # Calculate transfer matrices through all layers
    for in_ in range(nn-1):
        # Use complex sqrt to avoid warnings
        arg0 = complex(k0**2 * n[in_]**2 - beta**2)
        arg1 = complex(k0**2 * n[in_+1]**2 - beta**2)
        kappa0 = np.sqrt(arg0)
        kappa1 = np.sqrt(arg1)
        
        # Calculate impedance ratio
        kk = kappa0/kappa1
        
        # Modify for TM modes
        if not TE:
            kk = kk * n[in_+1]**2/n[in_]**2
        
        # Calculate transfer matrix elements
        A1 = 0.5*(1 + kk)*A0 + 0.5*(1 - kk)*B0
        B1 = 0.5*(1 - kk)*A0 + 0.5*(1 + kk)*B0
        
        # Apply phase if not at last interface
        if in_ < nn-2:
            A1 = np.exp(1j * kappa1 * d[in_]) * A1
            B1 = np.exp(-1j * kappa1 * d[in_]) * B1
        
        # Update coefficients
        A0 = A1
        B0 = B1
        
        # Store coefficients for each layer
        if TE:
            At.append(A0)
            Bt.append(B0)
        else:
            At.append(A0/n[in_+1]**2)
            Bt.append(B0/n[in_+1]**2)
    
    # Calculate E-field at all positions
    E = np.zeros_like(x, dtype=complex)
    
    # First region (left side)
    arg = complex(k0**2 * n[0]**2 - beta**2)
    kappa = np.sqrt(arg)
    ix = x <= 0
    E[ix] = Bt[0] * np.exp(-1j * kappa * x[ix])
    
    # Middle regions
    for in_ in range(1, nn-1):
        x1 = np.sum(d[0:in_-1]) if in_ > 0 else 0
        x2 = np.sum(d[0:in_])
        arg = complex(k0**2 * n[in_]**2 - beta**2)
        kappa = np.sqrt(arg)
        ix = (x > x1) & (x <= x2)
        E[ix] = (At[in_] * np.exp(1j * kappa * (x[ix] - x2)) + 
                 Bt[in_] * np.exp(-1j * kappa * (x[ix] - x2)))
    
    # Last region (right side)
    x1 = np.sum(d)
    arg = complex(k0**2 * n[-1]**2 - beta**2)
    kappa = np.sqrt(arg)
    ix = x > x1
    E[ix] = At[-1] * np.exp(1j * kappa * (x[ix] - x1))
    
    return E
'''
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
    
    # Initialize coefficient arrays
    if TE:
        At = [A0]
        Bt = [B0]
    else:
        At = [A0/n[0]**2]
        Bt = [B0/n[0]**2]
    
    # Calculate transfer matrices and coefficients for each interface
    for in_idx in range(nn-1):
        # Use complex sqrt to handle evanescent fields
        kappa0 = np.sqrt(complex(k0**2 * n[in_idx]**2 - beta**2))
        kappa1 = np.sqrt(complex(k0**2 * n[in_idx+1]**2 - beta**2))
        
        # Calculate impedance ratio
        kk = kappa0/kappa1
        
        if not TE:
            kk = kk * n[in_idx+1]**2/n[in_idx]**2
        
        # Calculate new coefficients
        A1 = 0.5*(1 + kk)*A0 + 0.5*(1 - kk)*B0
        B1 = 0.5*(1 - kk)*A0 + 0.5*(1 + kk)*B0
        
        # Apply phase shift if not at last interface
        if in_idx < nn-2:
            phase = kappa1*d[in_idx]
            A1 = A1 * np.exp(1j*phase)
            B1 = B1 * np.exp(-1j*phase)
        
        # Update coefficients for next iteration
        A0 = A1
        B0 = B1
        
        # Store coefficients
        if TE:
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

