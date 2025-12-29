"""
Stage 1: Physical Modeling - Implement environmental models: Earth's magnetic field and gravity gradient torque.
"""

import numpy as np


# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def magnetic_field(t, orbit_params):
    """
    Calculate the Earth's magnetic field vector in the ECI frame.
    
    Parameters
    ----------
    t : float
        Time since epoch (s).
    orbit_params : dict
        Dictionary containing 'altitude', 'inclination', and 'omega_orbit'.
    
    Returns
    -------
    B_eci : np.ndarray
        3x1 magnetic field vector in ECI frame (Tesla).
    """
    return result  # <- Return hint for the model


def _gravity_gradient_torque(r_body, inertia):
    """
    Calculate the gravity gradient torque in the body frame.
    
    Parameters
    ----------
    r_body : np.ndarray
        3x1 position vector in the body frame (m).
    inertia : np.ndarray
        3x3 inertia matrix (kg·m²).
    
    Returns
    -------
    tau_gg : np.ndarray
        3x1 gravity gradient torque vector (N·m).
    """
    return result  # <- Return hint for the model

# =============================================================================
# GOLD SOLUTION
# =============================================================================
def _gold_magnetic_field(t, orbit_params):
    R_e = 6371200.0
    M = 7.94e22
    mu0_over_4pi = 1e-7
    omega_e = 7.292115e-5
    tilt = np.radians(11.5)
    
    r = R_e + orbit_params['altitude']
    inc = np.radians(orbit_params['inclination'])
    theta = orbit_params['omega_orbit'] * t
    
    r_eci = r * np.array([
        np.cos(theta),
        np.sin(theta) * np.cos(inc),
        np.sin(theta) * np.sin(inc)
    ])
    r_mag = np.linalg.norm(r_eci)
    r_hat = r_eci / r_mag
    
    phi_m0 = np.radians(-70.0)
    phi_m = phi_m0 + omega_e * t
    
    m_eci = M * np.array([
        np.sin(tilt) * np.cos(phi_m),
        np.sin(tilt) * np.sin(phi_m),
        np.cos(tilt)
    ])
    
    B_eci = mu0_over_4pi * (1.0 / r_mag**3) * (3.0 * np.dot(m_eci, r_hat) * r_hat - m_eci)
    return B_eci

def _gold_get_orbit_pos(t, orbit_params):
    R_e = 6371200.0
    r = R_e + orbit_params['altitude']
    inc = np.radians(orbit_params['inclination'])
    theta = orbit_params['omega_orbit'] * t
    r_eci = r * np.array([
        np.cos(theta),
        np.sin(theta) * np.cos(inc),
        np.sin(theta) * np.sin(inc)
    ])
    return r_eci

def _gold_gravity_gradient_torque(r_body, inertia):
    mu = 3.986004418e14
    r_norm = np.linalg.norm(r_body)
    tau_gg = (3.0 * mu / r_norm**5) * np.cross(r_body, np.dot(inertia, r_body))
    return tau_gg

# =============================================================================
# TEST CASES
# =============================================================================
def test_cases():
    return [
        {
            "setup": """
t = 1000.0
params_eq = {'altitude': 500000, 'inclination': 0.0, 'omega_orbit': 0.0011}
params_pol = {'altitude': 500000, 'inclination': 90.0, 'omega_orbit': 0.0011}
""",
            "call": "[magnetic_field(t, params_eq), magnetic_field(t, params_pol)]",
            "gold_call": "[_gold_magnetic_field(t, params_eq), _gold_magnetic_field(t, params_pol)]"
        },
        {
            "setup": """
t = 0.0
params_low = {'altitude': 300000, 'inclination': 45.0, 'omega_orbit': 0.0011}
params_high = {'altitude': 2000000, 'inclination': 45.0, 'omega_orbit': 0.0011}
params_polar = {'altitude': 500000, 'inclination': 90.0, 'omega_orbit': 0.0011}
""",
            "call": "[magnetic_field(t, params_low), magnetic_field(t, params_polar)]",
            "gold_call": "[_gold_magnetic_field(t, params_low), _gold_magnetic_field(t, params_polar)]"
        },
        {
            "setup": """
inertia = np.diag([1.0, 2.0, 3.0])
r_aligned = np.array([7000000.0, 0, 0])  # Aligned with principal axis
r_unaligned = np.array([7000000.0, 7000000.0, 0]) / np.sqrt(2) # 45 deg offset
""",
            "call": "[_gravity_gradient_torque(r_aligned, inertia), _gravity_gradient_torque(r_unaligned, inertia)]",
            "gold_call": "[_gold_gravity_gradient_torque(r_aligned, inertia), _gold_gravity_gradient_torque(r_unaligned, inertia)]"
        }
    ]

