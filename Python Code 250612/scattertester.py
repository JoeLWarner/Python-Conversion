#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:46:58 2025

@author: joel
"""
import numpy as np

def scatteringrate2D_rect(Lambda, theta, pol, k0, neff, nsil, w0, h0, dng):
    """
    Example scattering rate for a rectangular waveguide mode.
    
    This is a modified version of your scatteringrate2D_f function,
    which now accepts two parameters for the mode profile:
      - w0: mode width (horizontal)
      - h0: mode height (vertical)
      
    The reflectance might depend on both dimensions via the mode overlap.
    This example simply multiplies the original dependence by an extra factor 
    that depends on h0. Replace this with your physical model.
    
    Args:
        Lambda: grating period (array or scalar)
        theta: grating tilt angle (array or scalar)
        pol: polarization (1 for TE, 2 for TM)
        k0: vacuum wavenumber
        neff: effective refractive index
        nsil: cladding refractive index
        w0: mode width (m)
        h0: mode height (m)
        dng: grating index modulation
        
    Returns:
        al: scattering (or loss) rate (1/m)
        Ex, Ey, Ez: scattered field components (placeholders here)
    """
    # Use the original expression (here taken from your previous code) for the dependence on w0:
    al = (np.sqrt(2*np.pi)/w0) * (1/(2*np.cos(theta)**2)**2) * (np.pi/Lambda)**2 
    al *= (w0/np.sin(2*theta))**2 * (dng/neff)**2
    al *= np.exp(-2*(w0/np.sin(2*theta))**2 * (k0*neff*np.cos(theta)**2 - np.pi/Lambda)**2)
    
    # Now include an additional dependence on the core height h0.
    # This is just one possible way to include h0. For instance, you might assume that
    # the effective scattering scales with the mode confinement in the vertical direction as well.
    al *= np.exp(-2*(h0/np.sin(2*theta))**2 * (k0*neff*np.cos(theta)**2 - np.pi/Lambda)**2)
    
    # For simplicity, set the scattered fields to zero (or add a proper model if desired)
    Ex = np.zeros_like(np.atleast_1d(al))
    Ey = np.zeros_like(np.atleast_1d(al))
    Ez = np.zeros_like(np.atleast_1d(al))
    
    return al, Ex, Ey, Ez
