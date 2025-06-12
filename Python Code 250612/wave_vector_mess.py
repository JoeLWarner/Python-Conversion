#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 13:24:52 2025

@author: joel
"""

import numpy as np
import matplotlib.pyplot as plt

# Define spatial grid
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(x, y)

# Parameters
k0 = 2 * np.pi / 2.0  # central wavenumber (wavelength = 2)
theta = np.pi/2  # direction of propagation for plane wave (30 degrees)
kx = k0 * np.cos(theta)
ky = k0 * np.sin(theta)

# Plane wave: u = exp(i(kx*x + ky*y))
plane_wave = np.cos(kx * X + ky * Y)

# Gaussian beam: Gaussian envelope * plane wave
w0 = 3.0  # beam waist
gaussian_envelope = np.exp(-(X**2 + Y**2) / w0**2)
gaussian_wave = gaussian_envelope * np.cos(kx * X + ky * Y)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plane wave plot
axs[0].imshow(plane_wave, extent=[x.min(), x.max(), y.min(), y.max()], cmap='RdBu')
axs[0].quiver(0, 0, kx, ky, scale=10, color='black')
axs[0].set_title('Plane Wave with Wave Vector')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')

# Gaussian beam plot
axs[1].imshow(gaussian_wave, extent=[x.min(), x.max(), y.min(), y.max()], cmap='RdBu')
# Arrows showing variation of local wavevector direction (simulated)
for xi in np.linspace(-6, 6, 5):
    for yi in np.linspace(-6, 6, 5):
        local_kx = kx + 0.08 * xi  # simulate spreading
        local_ky = ky + 0.08 * yi
        axs[1].quiver(xi, yi, local_kx, local_ky, scale=10, color='black')

axs[1].set_title('Gaussian Beam with Varying Wave Vectors')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')

plt.tight_layout()
plt.show()
