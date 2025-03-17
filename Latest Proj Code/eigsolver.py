# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:35:08 2025

@author: jw4e24
"""
import numpy as np
from scipy.optimize import fsolve


def eigenvalue_equation1(beta, k0, n, d, TE):
    """
    Solve the eigenvalue equation for TE/TM modes in a slab waveguide.
    """
    kappa_core = np.sqrt(np.maximum(0, k0**2 * n[1]**2 - beta**2))
    kappa_clad = np.sqrt(np.maximum(1e-12, beta**2 - k0**2 * n[0]**2))  # Avoid division by zero

    if TE:
        lhs = kappa_core / n[1]
        rhs = kappa_clad / n[0]
    else:
        lhs = kappa_core / (n[1]**3)
        rhs = kappa_clad / (n[0]**3)

    return np.tan(kappa_core * d) - lhs / rhs

def eigenvalue_equation2(beta, k0, n, d, TE):
    """
    Solve the eigenvalue equation for TE/TM modes using a transfer matrix approach.
    """
    nn = len(n)  # Number of layers
    A0 = np.zeros_like(beta)
    B0 = np.ones_like(beta)

    for i in range(nn - 1):
        kappa0 = np.sqrt(np.maximum(0, k0**2 * n[i]**2 - beta**2))
        kappa1 = np.sqrt(np.maximum(1e-12, k0**2 * n[i + 1]**2 - beta**2))  # Avoid division by zero

        kk = np.where(kappa1 != 0, kappa0 / kappa1, 0)
        if not TE:
            kk *= (n[i + 1] / n[i])**2

        A1 = 0.5 * (1 + kk) * A0 + 0.5 * (1 - kk) * B0
        B1 = 0.5 * (1 - kk) * A0 + 0.5 * (1 + kk) * B0

        if i < nn - 2:
            A1 *= np.exp(1j * kappa1 * d).real
            B1 *= np.exp(-1j * kappa1 * d).real

        A0, B0 = A1, B1

    return np.real(B0)

def eigenvalue_equation3(beta, k0, n, d, TE):
    """
    Solve the eigenvalue equation using the transfer matrix approach to match MATLAB.
    """
    nn = len(n)  # Number of layers
    A0 = np.zeros_like(beta)
    B0 = np.ones_like(beta)

    for i in range(nn - 1):
        kappa0 = np.sqrt(np.maximum(0, k0**2 * n[i]**2 - beta**2))
        kappa1 = np.sqrt(np.maximum(1e-12, k0**2 * n[i + 1]**2 - beta**2))  # Avoid division by zero

        kk = np.where(kappa1 != 0, kappa0 / kappa1, 0)
        if not TE:
            kk *= (n[i + 1] / n[i])**2

        A1 = 0.5 * (1 + kk) * A0 + 0.5 * (1 - kk) * B0
        B1 = 0.5 * (1 - kk) * A0 + 0.5 * (1 + kk) * B0

        if i < nn - 2:
            A1 *= np.exp(1j * kappa1 * d[i]).real
            B1 *= np.exp(-1j * kappa1 * d[i]).real

        A0, B0 = A1, B1

    return np.real(B0)
