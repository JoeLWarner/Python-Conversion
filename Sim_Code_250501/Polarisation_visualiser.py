#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 14:37:50 2025

@author: joel
"""

import numpy as np
import matplotlib.pyplot as plt

# Assume these values come from your simulation:
A_TE = 1.0       # TE amplitude (normalized)
A_TM = 1       # TM amplitude (normalized); ideally equal to A_TE for circular polarization
phi_diff = np.pi  # Relative phase difference (TM phase - TE phase)

# Normalized angular frequency for demonstration (optical frequencies are too high to plot directly)
omega = 2 * np.pi  # one period corresponds to time T = 1

# Create a time array over one period
t = np.linspace(0, 1, 1000)

# Time evolution of the fields:
E_y = A_TE * np.cos(omega * t)            # TE component (assumed along y)
E_z = A_TM * np.cos(omega * t + phi_diff)   # TM component (assumed along z)

# Plot time evolution of the TE and TM components
plt.figure(figsize=(8, 4))
plt.plot(t, E_y, label='E_y (TE)')
plt.plot(t, E_z, label='E_z (TM)')
plt.xlabel('Normalized Time')
plt.ylabel('Electric Field Amplitude (a.u.)')
plt.title('Time Evolution of TE and TM Field Components')
plt.legend()
plt.grid(True)
plt.show()

# Plot the polarization ellipse (E_y vs. E_z)
plt.figure(figsize=(6, 6))
plt.plot(E_y, E_z, 'k-')
plt.xlabel('E_y (TE)')
plt.ylabel('E_z (TM)')
plt.title('Polarization Ellipse')
plt.axis('equal')
plt.grid(True)
plt.show()
