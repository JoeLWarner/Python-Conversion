#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 14:31:57 2025

@author: joel
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the potential function for saddle points and pseudopotential
def saddle_potential(x, y, time_sign=1):
    return time_sign * (x**2 - y**2)

def pseudo_potential(x, y):
    return x**2 + y**2

# Create grid
x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)

# Calculate potentials
V1 = saddle_potential(X, Y, time_sign=1)
V2 = saddle_potential(X, Y, time_sign=-1)
V3 = pseudo_potential(X, Y)

# Create plots
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
titles = ["Saddle Potential (RF Phase A)", "Saddle Potential (RF Phase B)", "Time-Averaged Pseudopotential"]

for ax, V, title in zip(axs, [V1, V2, V3], titles):
    cp = ax.contourf(X, Y, V, levels=50, cmap='coolwarm')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(cp, ax=ax)

plt.tight_layout()
plt.show()

# Compute potentials
Z1 = saddle_potential(X, Y, time_sign=1)
Z2 = saddle_potential(X, Y, time_sign=-1)
Z3 = pseudo_potential(X, Y)

# Set up figure
fig = plt.figure(figsize=(18, 5))

# Plot saddle potential (RF phase A)
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.plot_surface(X, Y, Z1, cmap='coolwarm', edgecolor='none')
ax1.set_title("Saddle Potential (RF Phase A)")
ax1.set_yticklabels([])
ax1.set_xticklabels([])
ax1.set_zticklabels([])

#ax1.set_xlabel("x")
#ax1.set_ylabel("y")
#ax1.set_zlabel("Potential")

# Plot saddle potential (RF phase B)
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.plot_surface(X, Y, Z2, cmap='coolwarm', edgecolor='none')
ax2.set_title("Saddle Potential (RF Phase B)")
ax2.set_yticklabels([])
ax2.set_xticklabels([])
ax2.set_zticklabels([])
#ax2.set_xlabel("x")
#ax2.set_ylabel("y")
#ax2.set_zlabel("Potential")

# Plot time-averaged pseudopotential
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.plot_surface(X, Y, Z3, cmap='coolwarm', edgecolor='none')
ax3.set_title("Time-Averaged Pseudopotential")
ax3.set_yticklabels([])
ax3.set_xticklabels([])
ax3.set_zticklabels([])
#ax3.set_xlabel("x")
#ax3.set_ylabel("y")
#ax3.set_zlabel("Potential")

plt.subplots_adjust(wspace=-0.3, hspace=0)

plt.tight_layout()
plt.show()