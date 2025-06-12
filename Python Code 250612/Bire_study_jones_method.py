#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 29 12:10:35 2025

@author: joel
"""

#!/usr/bin/env python3
"""
2×2 Jones-layer TMM for silica–calcite–air QWP with arbitrary in-plane optic axis angle φ.
This handles a rotation of the birefringent axis in the plane of incidence via basis rotations.

Dependencies:
    numpy, matplotlib

Run:
    python jones_qwp_rotated.py

Outputs phase difference and transmission vs. incidence angle.
"""
import numpy as np
import matplotlib.pyplot as plt

# ── 1) Materials & design ─────────────────────────────────────────────────────
λ0   = 590e-9      # design wavelength [m]
n1   = 1.45        # silica
n_o  = 1.658       # calcite ordinary index
n_e  = 1.486       # calcite extraordinary index
n3   = 1.0         # air
# set in-plane optic-axis rotation angle (radians)
phi  = np.deg2rad(0)  

d_qwp = λ0/(4*abs(n_e - n_o))  # QWP thickness [m]
print(f"Calcite QWP thickness: {d_qwp*1e6:.3f} μm")

# ── 2) Utility: basis rotation & effective index ─────────────────────────────
def R(phi):
    """Rotation matrix for Jones vector in s,p basis."""
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s], [s, c]])

def neff(phi):
    """Effective extraordinary index for in-plane axis at angle phi."""
    return 1.0/np.sqrt((np.cos(phi)**2)/(n_e**2) + (np.sin(phi)**2)/(n_o**2))

# ── 3) Fresnel coefficients ────────────────────────────────────────────────────
def fresnel_s(n_i, n_t, θ_i, θ_t):
    return (n_i*np.cos(θ_i) - n_t*np.cos(θ_t)) / (n_i*np.cos(θ_i) + n_t*np.cos(θ_t))

def fresnel_p(n_i, n_t, θ_i, θ_t):
    return (n_t*np.cos(θ_i) - n_i*np.cos(θ_t)) / (n_t*np.cos(θ_i) + n_i*np.cos(θ_t))

# ── 4) Jones matrix with axis rotation ─────────────────────────────────────────
def jones_qwp(theta_inc, phi):
    # rotate into crystal principal basis
    Rp = R(phi)
    Rm = R(-phi)

    # angles in lab: θ1 incidence, then ordinary and extraordinary
    θ1 = theta_inc
    θ_o = np.arcsin(n1*np.sin(θ1)/n_o)
    # use effective extraordinary index for both interface and propagation
    n_eff = neff(theta_inc)
    θ_e = np.arcsin(n1*np.sin(θ1)/n_eff)

    # interface 1 in principal basis
    rs_o = fresnel_s(n1,  n_o,   θ1,    θ_o)
    ts_o = 1 + rs_o
    rp_e = fresnel_p(n1, n_eff, θ1,    θ_e)
    tp_e = 1 + rp_e
    I12 = np.diag([ts_o, tp_e])

    # propagation in principal basis
    k0 = 2*np.pi/λ0
    δ_o = k0 * n_o   * d_qwp * np.cos(θ_o)
    δ_e = k0 * n_eff* d_qwp * np.cos(θ_e)
    P   = np.diag([np.exp(1j*δ_o), np.exp(1j*δ_e)])

    # interface 2 in principal basis
    rs_o2 = fresnel_s(n_o,   n3,   θ_o, θ1)
    ts_o2 = 1 + rs_o2
    rp_e2 = fresnel_p(n_eff, n3, θ_e, θ1)
    tp_e2 = 1 + rp_e2
    I23   = np.diag([ts_o2, tp_e2])

    # full Jones: rotate into principal, apply I12→P→I23, rotate back
    return Rm.dot(I23).dot(P).dot(I12).dot(Rp)

# ── 5) Sweep angles & evaluate ─────────────────────────────────────────────────
thetas = np.linspace(0, 65, 601)*np.pi/180
phase = []
Ttot  = []
for θ in thetas:
    J = jones_qwp(θ, phi)
    # launch 45° linear in lab: [Es; Ep] = [1,1]
    Eout = J.dot(np.array([1,1]))
    # phase difference Ey relative Ex
    φ = np.angle(Eout[1]*np.conj(Eout[0]))
    phase.append(np.degrees((φ+np.pi)%(2*np.pi) - np.pi))
    # total transmitted intensity
    Ttot.append(np.abs(Eout[0])**2 + np.abs(Eout[1])**2)
phase = np.array(phase)
Ttot  = np.array(Ttot)

phi_range = np.linspace(0, 360, 100)*np.pi/180
phase2 = []
Ttot2  = []
for phis in phi_range:
    J2 = jones_qwp(0, phis)
    Eout2 = J2.dot(np.array([1,1]))
    φ2 = np.angle(Eout2[1]*np.conj(Eout2[0]))
    phase2.append(np.degrees((φ2+np.pi)%(2*np.pi) - np.pi))
    # total transmitted intensity
    Ttot2.append(np.abs(Eout2[0])**2 + np.abs(Eout2[1])**2)
phase2 = np.array(phase2)
Ttot2  = np.array(Ttot2)

plt.plot(phi_range/np.pi*180, phase2)  
plt.xlabel("axis rotation angle")
plt.grid()

    

# ── 6) Plot ─────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots()
ax1.plot(thetas*180/np.pi, phase, 'b-')
ax1.axhline(90, color='r', ls='--', label='90°')
ax1.axhline(-90, color='r', ls='--', label='90°')
ax1.set_xlabel('Incidence angle (deg)')
ax1.set_ylabel('Phase Δ (deg)', color='b')
ax1.grid(True)
#ax2 = ax1.twinx()
#ax2.plot(thetas*180/np.pi, Ttot, 'g-')
#ax2.set_ylabel('Transmission', color='g')
#ax2.set_ylim(0,1.1)
plt.title(f'Rotated-axis QWP @ {λ0*1e9:.0f} nm, φ={np.degrees(phi):.0f}°')
plt.tight_layout()
plt.show()

# normal-incidence check
print(f"Normal incidence: Δφ={phase[0]:.1f}°, T={Ttot[0]:.3f}")
