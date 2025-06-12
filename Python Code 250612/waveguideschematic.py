# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 15:25:23 2025

@author: joel
"""

import numpy as np
import matplotlib.pyplot as plt

def schematic(phitar, 
                                  waveguide_length=40e-6,
                                  waveguide_width=5e-6,
                                  grating_length=20e-6,
                                  grating_thickness=0.5e-6,
                                  dtar=10e-6,            # target distance from grating center (m)
                                  grating_start_x=10e-6,
                                  num_lines=10,
                                  line_length_factor=1.0):
    grating_tilt = phitar / 2
    print(np.degrees(grating_tilt))
    grating_center_x = waveguide_length / 2
    grating_center_y = waveguide_width  # top edge of the waveguide
    target_x = grating_center_x - dtar * np.cos(phitar)
    target_y = grating_center_y + dtar * np.sin(phitar)

    fig, ax = plt.subplots(figsize=(8, 4))
    waveguide_rect = plt.Rectangle((0, 0), waveguide_length, waveguide_width, 
                                   edgecolor='black', facecolor='lightgray', lw=2)
    ax.add_patch(waveguide_rect)
    ax.text(waveguide_length/2, waveguide_width/2, 'Waveguide', 
            ha='center', va='center', fontsize=10)
    default_line_length = waveguide_width / np.cos(grating_tilt)
    line_length = line_length_factor * default_line_length

    line_spacing = grating_length / (num_lines - 1)
    for i in range(num_lines):
        x0 = grating_start_x + i * line_spacing
        y0 = 0 
        x1 = x0 + line_length * np.sin(grating_tilt)
        y1 = y0 + line_length * np.cos(grating_tilt)
        ax.plot([x0, x1], [y0, y1], color='blue', lw=1)

    ax.plot(target_x, target_y, 'ro', markersize=12, label='Target')
    ax.text(target_x, target_y, ' Target', color='red', ha='left', va='bottom', fontsize=10)

    ax.plot([grating_center_x, target_x], [waveguide_width/2, target_y], 'r-', lw=1)

    ax.set_xlim(-5e-6, waveguide_length+5e-6)
    ax.set_ylim(-1e-6, target_y+5e-6)
    ax.set_xlabel('z (m)')
    ax.set_ylabel('x (m)')
    ax.grid(True)
    #ax.legend()

    return fig, ax

