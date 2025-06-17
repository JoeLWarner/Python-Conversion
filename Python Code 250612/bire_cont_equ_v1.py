# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 01:38:54 2025

@author: joeld
"""

import numpy as np
import matplotlib.pyplot as plt

def fresnel_iso(n_i, n_t, theta_i, polarization):
    """
    Compute Fresnel coefficients for isotropic interface.
    n_i, n_t: refractive indices incident and transmitted
    theta_i: incident angle (rad)
    polarization: 's' or 'p'
    Returns r, t
    """
    sin_theta_t = n_i / n_t * np.sin(theta_i)
    cos_theta_t = np.sqrt(1 - sin_theta_t**2, dtype=complex)
    cos_theta_i = np.cos(theta_i)
    if polarization == 's':
        rs = (n_i * cos_theta_i - n_t * cos_theta_t) / (n_i * cos_theta_i + n_t * cos_theta_t)
        ts = (2 * n_i * cos_theta_i) / (n_i * cos_theta_i + n_t * cos_theta_t)
        return rs, ts
    elif polarization == 'p':
        rp = (n_t * cos_theta_i - n_i * cos_theta_t) / (n_t * cos_theta_i + n_i * cos_theta_t)
        tp = (2 * n_i * cos_theta_i) / (n_t * cos_theta_i + n_i * cos_theta_t)
        return rp, tp
    else:
        raise ValueError("Polarization must be 's' or 'p'.")


def unit_vectors(theta, n):
    """
    For isotropic medium with refractive index n and incident angle theta (rad),
    returns k-hat, s-hat, p-hat, and impedance Z.
    """
    kx = np.sin(theta)
    kz = -np.cos(theta)
    k_hat = np.array([kx, 0, kz], dtype=complex)
    s_hat = np.array([0, 1, 0], dtype=complex)
    p_hat = np.cross(s_hat, k_hat)
    p_hat /= np.linalg.norm(p_hat)
    eta0 = 376.730313668
    Z = eta0 / (n * np.cos(theta))
    return k_hat, s_hat, p_hat, Z


def calc_continuity_matrix(wavelength, theta_inc_deg,
                            d1, d2, n0, n1, no, ne, phi_axis_deg):
    """
    Solve boundary conditions for air (n0) -> silica (n1, d1)
    -> calcite (no, ne, d2, axis phi) -> air (n0).
    Returns transmitted Jones amplitudes [t_s, t_p] and phase difference.
    """
    theta0 = np.deg2rad(theta_inc_deg)
    phi_axis = np.deg2rad(phi_axis_deg)

    # Medium 0 (air)
    k0, s0, p0, Z0_s = unit_vectors(theta0, n0)
    theta1 = np.arcsin(n0/n1 * np.sin(theta0))
    k1, s1, p1, Z1_s = unit_vectors(theta1, n1)
    Z1_p = Z1_s
    # Medium 3 same as medium 0 (air)
    s3, p3, Z3_s = s0, p0, Z0_s
    Z3_p = Z3_s

    # Layer 2 (calcite) wavevector (ordinary)
    theta2o = np.arcsin(n0/no * np.sin(theta0))
    k2 = np.array([np.sin(theta2o), 0, -np.cos(theta2o)], dtype=complex)

    # Optical axis orientation
    a_hat = np.array([np.cos(phi_axis), np.sin(phi_axis), 0], dtype=complex)
    e_o = np.cross(k2, a_hat)
    e_o /= np.linalg.norm(e_o)
    e_e = np.cross(e_o, k2)
    e_e /= np.linalg.norm(e_e)

    # Impedances in calcite
    eta0 = 376.730313668
    Z2_o = eta0 / (no * np.abs(k2[2]))
    Z2_e = eta0 / (ne * np.abs(k2[2]))

    # Phase accumulation
    k1z = k1[2] * (2*np.pi/n1) / wavelength
    k2o_z = k2[2] * (2*np.pi/no) / wavelength
    k2e_z = k2[2] * (2*np.pi/ne) / wavelength
    phi1 = k1z * d1
    phi2o = k2o_z * d2
    phi2e = k2e_z * d2

    # Build matrix
    A = np.zeros((12,12), dtype=complex)
    b = np.zeros(12, dtype=complex)
    # Interface 0-1
    A[0,0]=1; A[0,2]=-1; A[0,3]=-1; b[0]=-1
    A[1,0]=-1/Z0_s; A[1,2]=1/Z1_s; A[1,3]=-1/Z1_s; b[1]=-1/Z0_s
    A[2,1]=1; A[2,4]=-1; A[2,5]=-1; b[2]=0
    A[3,1]=-Z0_s; A[3,4]=Z1_p; A[3,5]=-Z1_p; b[3]=0
    # Interface 1-2 projections
    p_oo=np.dot(s1,e_o); p_eo=np.dot(s1,e_e)
    p_op=np.dot(p1,e_o); p_ep=np.dot(p1,e_e)
    A[4,2]=np.exp(1j*phi1); A[4,3]=np.exp(-1j*phi1)
    A[4,6]=-p_oo;A[4,7]=-p_oo;A[4,8]=-p_eo;A[4,9]=-p_eo; b[4]=0
    A[5,2]=np.exp(1j*phi1)/Z1_s;A[5,3]=-np.exp(-1j*phi1)/Z1_s
    A[5,6]=-p_oo/Z2_o;A[5,7]=p_oo/Z2_o;A[5,8]=-p_eo/Z2_e;A[5,9]=p_eo/Z2_e; b[5]=0
    A[6,4]=np.exp(1j*phi1);A[6,5]=np.exp(-1j*phi1)
    A[6,6]=-p_op;A[6,7]=-p_op;A[6,8]=-p_ep;A[6,9]=-p_ep; b[6]=0
    A[7,4]=np.exp(1j*phi1)/Z1_p;A[7,5]=-np.exp(-1j*phi1)/Z1_p
    A[7,6]=-p_op/Z2_o;A[7,7]=p_op/Z2_o;A[7,8]=-p_ep/Z2_e;A[7,9]=p_ep/Z2_e; b[7]=0
    # Interface 2-3 projections
    p_po=np.dot(e_o,s3); p_pe=np.dot(e_e,s3)
    p_qo=np.dot(e_o,p3); p_qe=np.dot(e_e,p3)
    A[8,6]=np.exp(1j*phi2o)*p_po;A[8,7]=np.exp(-1j*phi2o)*p_po
    A[8,8]=np.exp(1j*phi2e)*p_pe;A[8,9]=np.exp(-1j*phi2e)*p_pe;A[8,10]=-1; b[8]=0
    A[9,6]=np.exp(1j*phi2o)*p_po/Z2_o;A[9,7]=-np.exp(-1j*phi2o)*p_po/Z2_o
    A[9,8]=np.exp(1j*phi2e)*p_pe/Z2_e;A[9,9]=-np.exp(-1j*phi2e)*p_pe/Z2_e;A[9,10]=-1/Z3_s; b[9]=0
    A[10,6]=np.exp(1j*phi2o)*p_qo;A[10,7]=np.exp(-1j*phi2o)*p_qo
    A[10,8]=np.exp(1j*phi2e)*p_qe;A[10,9]=np.exp(-1j*phi2e)*p_qe;A[10,11]=-1; b[10]=0
    A[11,6]=np.exp(1j*phi2o)*p_qo/Z2_o;A[11,7]=-np.exp(-1j*phi2o)*p_qo/Z2_o
    A[11,8]=np.exp(1j*phi2e)*p_qe/Z2_e;A[11,9]=-np.exp(-1j*phi2e)*p_qe/Z2_e;A[11,11]=-1/Z3_p; b[11]=0

    x=np.linalg.solve(A,b)
    t_s, t_p = x[10], x[11]
    delta_phi = np.angle(t_p) - np.angle(t_s)
    return np.array([t_s, t_p]), delta_phi*180/np.pi

if __name__=='__main__':
    # Parameters
    wavelength = 632.8e-9  # m
    theta = 0.0           # deg
    d1 = 500e-9            # m
    d2 = 10e-9             # m calcite thickness
    n0 = 1.0
    n1 = 1.45
    no = 1.658
    ne = 1.486
    phi_axis = 0        # deg, crystal axis orientation

    E_out, dphi = calc_continuity_matrix(
        wavelength, theta, d1, d2, n0, n1, no, ne, phi_axis
    )
    print(f"Transmitted s: {E_out[0]:.4f}, p: {E_out[1]:.4f}")
    print(f"Phase diff: {dphi:.4f} deg")
    
angles = np.linspace(0, 360, 100)
phases = []
for rotate in angles:
    E_out, dphi = calc_continuity_matrix(
        wavelength, theta, d1, d2, n0, n1, no, ne, rotate
    )
    phases.append(dphi)
    
plt.plot(angles, phases)
plt.show()