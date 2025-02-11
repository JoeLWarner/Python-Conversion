#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:36:02 2025

@author: joel
"""

import numpy as np

def grating_angles_2D_f2(xtar, ztar, beta, kn):
    """
    Calculate the grating period and grating tilt angle.
    
    Parameters:
    xtar (array): Direction of target beam propagation in x.
    ztar (array): Direction of target beam propagation in z.
    beta (float): Propagation constant of waveguide pump mode.
    kn (float): Propagation constant in cladding.
    
    Returns:
    lamgrat (array): Grating period.
    alphatilt (array): Grating tilt angle (radians).
    """
    # Compute grating period using phase matching condition
    koutx = kn * xtar / np.sqrt(xtar**2 + ztar**2)
    kinx = beta
    kgratx = kinx - koutx  # Grating K
    lamgrat = 2 * np.pi / kgratx  # Grating period
    
    # Compute required grating tilt angle
    alphatilt = np.arccos(np.sqrt(kgratx / (2 * beta)))
    
    return lamgrat, alphatilt