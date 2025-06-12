#!/usr/bin/env python3
"""
2×2 Jones-layer TMM for silica–calcite–air QWP with arbitrary in-plane optic axis angle φ.
This properly handles rotation of the birefringent axis in the plane of incidence.

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
phi  = np.deg2rad(0)   # Start with 0° to check against your original case

d_qwp = λ0/(4*abs(n_e - n_o))  # QWP thickness [m]
print(f"Calcite QWP thickness: {d_qwp*1e6:.3f} μm")

# ── 2) Utility functions ─────────────────────────────────────────────────────
def R(angle):
    """Rotation matrix for 2D coordinate transformation."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])

def get_rotated_indices(phi):
    """
    Get the effective indices for s and p polarizations when crystal is rotated by phi.
    For uniaxial crystal with optic axis rotated in-plane by angle phi:
    - Crystal axes: ordinary at angle phi, extraordinary at angle phi+90°
    """
    # In rotated crystal:
    # s-direction projection onto crystal axes
    s_proj_o = np.cos(phi)  # projection of s onto ordinary axis
    s_proj_e = np.sin(phi)  # projection of s onto extraordinary axis
    
    # p-direction projection onto crystal axes  
    p_proj_o = -np.sin(phi)  # projection of p onto ordinary axis
    p_proj_e = np.cos(phi)   # projection of p onto extraordinary axis
    
    return s_proj_o, s_proj_e, p_proj_o, p_proj_e

def effective_index_general(n_o, n_e, cos_theta_o, cos_theta_e):
    """
    Calculate effective index for arbitrary direction in uniaxial crystal.
    For direction making angles with ordinary and extraordinary axes.
    """
    # This is a simplified model - in reality, you need the full tensor approach
    # For small rotations, this gives reasonable approximation
    return n_o  # This needs refinement for the general case

# ── 3) Fresnel coefficients ────────────────────────────────────────────────────
def fresnel_s(n_i, n_t, θ_i, θ_t):
    return (n_i*np.cos(θ_i) - n_t*np.cos(θ_t)) / (n_i*np.cos(θ_i) + n_t*np.cos(θ_t))

def fresnel_p(n_i, n_t, θ_i, θ_t):
    return (n_t*np.cos(θ_i) - n_i*np.cos(θ_t)) / (n_t*np.cos(θ_i) + n_i*np.cos(θ_t))

# ── 4) Jones matrix with proper axis rotation ──────────────────────────────────
def jones_qwp_rotated(theta_inc, phi):
    """
    Calculate Jones matrix for rotated QWP.
    
    Physical picture:
    1. Incident s,p components hit rotated crystal
    2. Each component gets split between ordinary and extraordinary rays
    3. These propagate with different phase velocities
    4. At exit, they recombine and exit into air
    """
    
    θ1 = theta_inc
    
    # Get projections of s,p onto crystal axes
    s_proj_o, s_proj_e, p_proj_o, p_proj_e = get_rotated_indices(phi)
    
    # Calculate refraction angles for ordinary and extraordinary rays
    θ_o = np.arcsin(n1*np.sin(θ1)/n_o)
    θ_e = np.arcsin(n1*np.sin(θ1)/n_e)  # Use true n_e, not effective
    
    # Fresnel coefficients for interfaces
    # Interface 1: silica -> calcite
    rs_o1 = fresnel_s(n1, n_o, θ1, θ_o)
    ts_o1 = 1 + rs_o1
    rp_o1 = fresnel_p(n1, n_o, θ1, θ_o)
    tp_o1 = 1 + rp_o1
    
    rs_e1 = fresnel_s(n1, n_e, θ1, θ_e)
    ts_e1 = 1 + rs_e1
    rp_e1 = fresnel_p(n1, n_e, θ1, θ_e)
    tp_e1 = 1 + rp_e1
    
    # Phase accumulation in crystal
    k0 = 2*np.pi/λ0
    δ_o = k0 * n_o * d_qwp * np.cos(θ_o)
    δ_e = k0 * n_e * d_qwp * np.cos(θ_e)
    
    # Interface 2: calcite -> air
    # Note: angles are same as interface 1 by reciprocity
    rs_o2 = fresnel_s(n_o, n3, θ_o, θ1)
    ts_o2 = 1 + rs_o2
    rp_o2 = fresnel_p(n_o, n3, θ_o, θ1)
    tp_o2 = 1 + rp_o2
    
    rs_e2 = fresnel_s(n_e, n3, θ_e, θ1)
    ts_e2 = 1 + rs_e2
    rp_e2 = fresnel_p(n_e, n3, θ_e, θ1)
    tp_e2 = 1 + rp_e2
    
    # Build Jones matrix components
    # For s-polarized input:
    # - s_proj_o fraction goes via ordinary ray
    # - s_proj_e fraction goes via extraordinary ray
    J11 = (s_proj_o * ts_o1 * np.exp(1j*δ_o) * ts_o2 * s_proj_o + 
           s_proj_e * ts_e1 * np.exp(1j*δ_e) * ts_e2 * s_proj_e)
    
    J21 = (s_proj_o * tp_o1 * np.exp(1j*δ_o) * tp_o2 * p_proj_o + 
           s_proj_e * tp_e1 * np.exp(1j*δ_e) * tp_e2 * p_proj_e)
    
    # For p-polarized input:
    J12 = (p_proj_o * ts_o1 * np.exp(1j*δ_o) * ts_o2 * s_proj_o + 
           p_proj_e * ts_e1 * np.exp(1j*δ_e) * ts_e2 * s_proj_e)
    
    J22 = (p_proj_o * tp_o1 * np.exp(1j*δ_o) * tp_o2 * p_proj_o + 
           p_proj_e * tp_e1 * np.exp(1j*δ_e) * tp_e2 * p_proj_e)
    
    return np.array([[J11, J12], [J21, J22]])

# ── 5) Sweep angles & evaluate ─────────────────────────────────────────────────
thetas = np.linspace(0, 65, 601)*np.pi/180
phase = []
Ttot  = []

for θ in thetas:
    J = jones_qwp_rotated(θ, phi)
    # launch 45° linear in lab: [Es; Ep] = [1,1]/√2
    Ein = np.array([1, 1])/np.sqrt(2)
    Eout = J.dot(Ein)
    
    # phase difference Ep relative to Es
    φ_diff = np.angle(Eout[1]) - np.angle(Eout[0])
    phase.append(np.degrees((φ_diff + np.pi) % (2*np.pi) - np.pi))
    
    # total transmitted intensity
    Ttot.append(np.abs(Eout[0])**2 + np.abs(Eout[1])**2)

phase = np.array(phase)
Ttot  = np.array(Ttot)

# ── 6) Plot ─────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(thetas*180/np.pi, phase, 'b-', linewidth=2, label='Phase difference')
ax1.axhline(90, color='r', ls='--', linewidth=2, label='90°')
ax1.axhline(-90, color='r', ls='--', linewidth=2, alpha=0.7)
ax1.set_xlabel('Incidence angle (deg)')
ax1.set_ylabel('Phase difference (deg)', color='b')
ax1.grid(True, alpha=0.3)
ax1.legend()

#ax2 = ax1.twinx()
#ax2.plot(thetas*180/np.pi, Ttot, 'g-', linewidth=2, label='Transmission')
#ax2.set_ylabel('Transmission', color='g')
#ax2.legend(loc='upper right')

plt.title(f'Rotated QWP @ {λ0*1e9:.0f} nm, optic axis angle φ = {np.degrees(phi):.1f}°')
plt.tight_layout()
plt.show()

phi_range = np.linspace(0, 360, 100)*np.pi/180
phase2 = []
Ttot2  = []
for phis in phi_range:
    J2 = jones_qwp_rotated(0, phis)
    Eout2 = J2.dot(np.array([1,1]))
    φ2 = np.angle(Eout2[1]) - np.angle(Eout2[0])
    phase2.append(np.degrees((φ2+np.pi)%(2*np.pi) - np.pi))
    # total transmitted intensity
    Ttot2.append(np.abs(Eout2[0])**2 + np.abs(Eout2[1])**2)
phase2 = np.array(phase2)
Ttot2  = np.array(Ttot2)

plt.plot(phi_range/np.pi*180, phase2) 
plt.axhline(90, color='r', ls='--', linewidth=2, label='90°') 
plt.axhline(-90, color='r', ls='--', linewidth=2, label='90°') 
plt.title("Phase difference vs in-plane axis rotation")
plt.ylabel("Phase difference (deg)")
plt.xlabel("axis rotation angle (deg)")
plt.grid()


# ── 7) Analysis ─────────────────────────────────────────────────────────────
print(f"\nRotation angle φ = {np.degrees(phi):.1f}°")
print(f"Normal incidence: Δφ = {phase[0]:.1f}°, T = {Ttot[0]:.3f}")
print(f"At 30°: Δφ = {phase[300]:.1f}°, T = {Ttot[300]:.3f}")
print(f"At 60°: Δφ = {phase[500]:.1f}°, T = {Ttot[500]:.3f}")

# Compare different rotation angles
print(f"\nProjection coefficients for φ = {np.degrees(phi):.1f}°:")
s_o, s_e, p_o, p_e = get_rotated_indices(phi)
print(f"s-pol: {s_o:.3f}×ordinary + {s_e:.3f}×extraordinary")
print(f"p-pol: {p_o:.3f}×ordinary + {p_e:.3f}×extraordinary")