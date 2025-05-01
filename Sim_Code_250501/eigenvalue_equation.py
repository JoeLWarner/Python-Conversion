# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:08:07 2025

@author: jw4e24
"""

import numpy as np

def nslabmodes_f(beta, k0, n, d, TE):
    """
    Calculate TE/TM modes for multilayer slab waveguide.
    
    Parameters:
    beta (array-like): Propagation constant values to evaluate
    k0 (float): Wave number in free space (2π/λ)
    n (array-like): Refractive indices of layers
    d (array-like): Thickness of layers (except for outer layers)
    TE (bool): True for TE modes, False for TM modes
    
    Returns:
    array: B0 values for eigenvalue equation
    """
    # Convert inputs to numpy arrays if they aren't already
    beta = np.asarray(beta, dtype=np.complex128)
    n = np.asarray(n, dtype=np.complex128)
    d = np.asarray(d, dtype=np.complex128)
    
    # Number of layers
    nn = len(n)
    
    # Initialize A0 and B0
    A0 = np.zeros_like(beta, dtype=np.complex128)
    B0 = np.ones_like(beta, dtype=np.complex128)
    
    # Loop through layers
    for in_ in range(nn-1):
        # Calculate transverse propagation constants
        kappa0 = np.sqrt(k0**2 * n[in_]**2 - beta**2)
        kappa1 = np.sqrt(k0**2 * n[in_+1]**2 - beta**2)
        
        # Calculate impedance ratio
        kk = kappa0/kappa1
        
        # Modify for TM modes
        if not TE:
            kk = kk * (n[in_+1]**2)/(n[in_]**2)
        
        # Calculate transfer matrix elements
        A1 = 0.5*(1 + kk)*A0 + 0.5*(1 - kk)*B0
        B1 = 0.5*(1 - kk)*A0 + 0.5*(1 + kk)*B0
        
        # Apply phase if not at last interface
        if in_ < nn-2:
            phase = 1j * kappa1 * d[in_]
            A1 = np.exp(phase) * A1
            B1 = np.exp(-phase) * B1
        
        # Update for next iteration
        A0 = A1
        B0 = B1
    
    return B0