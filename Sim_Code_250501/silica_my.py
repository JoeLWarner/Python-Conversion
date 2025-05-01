# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:18:27 2025

@author: jw4e24
"""

# l is wavelength in um
import numpy as np

def silica_n(wavelength_um):
    wavelength_um = wavelength_um * 1e6
    # Sellmeier coefficients
    c1 = 0.6961663
    l1 = 0.0684043
    c2 = 0.4079426
    l2 = 0.1162414
    c3 = 0.8974794
    l3 = 9.896161
    
    # Calculate terms in Sellmeier equation
    f1 = c1 * wavelength_um**2 / (wavelength_um**2 - l1**2)
    f2 = c2 * wavelength_um**2 / (wavelength_um**2 - l2**2)
    f3 = c3 * wavelength_um**2 / (wavelength_um**2 - l3**2)
    
    # Calculate refractive index
    n = np.sqrt(1 + f1 + f2 + f3)
    return n
    

