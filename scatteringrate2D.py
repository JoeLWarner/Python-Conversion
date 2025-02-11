#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:40:51 2025

@author: joel
"""

import numpy as np

def scatteringrate2D_f(Lambda, theta, pol, k0, neff, nsil, w0, dng):
    """
    Formula of power loss from tilted grating
    
    Args:
        Lambda (array): Local grating period (m)
        theta (array): Tilt angle (rad)
        pol (int): 1 for horizontal polarization, 2 for vertical polarization
        k0 (float): Wave number
        neff (float): Effective refractive index
        nsil (float): Silicon refractive index
        w0 (float): Beam waist
        dng (float): Grating strength
        
    Returns:
        tuple: (al, Ex, Ey, Ez) where
            al: energy loss rate (1/m)
            Ex, Ey, Ez: scattered electric field
    """
    beta = k0 * neff
    
    phi = np.arccos(-(beta - 2*np.pi/Lambda)/(k0*nsil))
    
    alphas = (np.sqrt(2*np.pi)/w0) * (1/(2*np.cos(theta)**2)**2) * (np.pi/Lambda)**2 
    alphas *= (w0/np.sin(2*theta))**2 * (dng/neff)**2
    alphas *= np.exp(-2*(w0/np.sin(2*theta))**2 * (beta*np.cos(theta)**2 - np.pi/Lambda)**2)
    alphas *= np.sin(phi)
    
    alphap = alphas * np.cos(2*theta)**2
    
    Es3 = -1/(2*np.cos(theta)**2) * w0/np.sin(2*theta) * dng/neff * np.pi**1.5/Lambda
    Es3 *= np.exp(-(w0/(2*np.sin(2*theta)))**2 * (2*beta*np.cos(theta)**2 - 2*np.pi/Lambda)**2)
    
    Ep3 = np.cos(2*theta) * Es3
    
    if pol == 1:
        al = alphas
        Ex = np.zeros_like(Es3)
        Ey = Es3
        Ez = np.zeros_like(Es3)
    else:
        al = alphap
        Ex = Ep3 * np.sin(phi)
        Ey = np.zeros_like(Ep3)
        Ez = Ep3 * np.cos(phi)
    
    return al, Ex, Ey, Ez