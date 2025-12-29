"""
Stage 1: Physical Modeling - Define the state-space derivative (ODE) for the nonlinear dynamics of a dual-spin satellite.
The state vector x includes the attitude quaternion q and the body angular velocity w.
The dynamics include the constant angular momentum of the internal momentum wheel.

Equations:
q_dot = 0.5 * Omega(w) * q
w_dot = I^-1 * (tau_total - w x (I*w + h_w))
where h_w is the momentum wheel angular momentum vector.
"""

import numpy as np


# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def dynamics_ode(t, x, u, inertia, h_w, B_eci, r_eci):
    """
    Calculate the derivative of the state vector.
    
    Parameters
    ----------
    t : float
        Current time (s).
    x : np.ndarray
        Current state [q0, q1, q2, q3, wx, wy, wz] (7 elements).
        q is the quaternion (ECI to Body), w is the body angular velocity.
    u : np.ndarray
        Control input (magnetic dipole moment m in body frame, 3 elements).
    inertia : np.ndarray
        3x3 inertia matrix (kg·m²).
    h_w : float
        Momentum wheel angular momentum magnitude (N·m·s), assumed along body z-axis.
    B_eci : np.ndarray
        3x1 magnetic field in ECI frame (Tesla).
    r_eci : np.ndarray
        3x1 position vector in ECI frame (m), used for gravity gradient.
    
    Returns
    -------
    dxdt : np.ndarray
        Derivative of the state vector [dq/dt, dw/dt] (7 elements).
    """
    return result  # <- Return hint for the model

# =============================================================================
# GOLD SOLUTION
# =============================================================================
def _gold_dynamics_ode(t, x, u, inertia, h_w, B_eci, r_eci):
    q = x[0:4]
    w = x[4:7]
    
    q_norm = np.linalg.norm(q)
    if q_norm > 1e-10:
        q = q / q_norm
    
    q0, q1, q2, q3 = q
    
    C = np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 + q0*q3), 2*(q1*q3 - q0*q2)],
        [2*(q1*q2 - q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 + q0*q1)],
        [2*(q1*q3 + q0*q2), 2*(q2*q3 - q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])
    
    B_body = C @ B_eci
    tau_mag = np.cross(u, B_body)
    
    mu = 3.986004418e14
    r_body = C @ r_eci
    r_norm = np.linalg.norm(r_body)
    tau_gg = (3.0 * mu / r_norm**5) * np.cross(r_body, np.dot(inertia, r_body))
    
    tau_total = tau_mag + tau_gg
    h_w_vec = np.array([0.0, 0.0, float(h_w)])
    
    H_total = np.dot(inertia, w) + h_w_vec
    w_dot = np.linalg.solve(inertia, tau_total - np.cross(w, H_total))
    
    dqdt = 0.5 * np.array([
        -q1*w[0] - q2*w[1] - q3*w[2],
         q0*w[0] + q2*w[2] - q3*w[1],
         q0*w[1] - q1*w[2] + q3*w[0],
         q0*w[2] + q1*w[1] - q2*w[0]
    ])
    
    return np.concatenate([dqdt, w_dot])

# =============================================================================
# TEST CASES
# =============================================================================
def test_cases():
    return [
        {
            "setup": """
t = 0.0
x = np.array([1.0, 0, 0, 0, 0.1, 0.2, 0.3])
u = np.zeros(3)
inertia = np.diag([0.1, 0.15, 0.2])
h_w = 0.0
B_eci = np.zeros(3)
r_eci = np.array([1e10, 0, 0]) # Negligible gravity gradient
""",
            "call": "dynamics_ode(t, x, u, inertia, h_w, B_eci, r_eci)",
            "gold_call": "_gold_dynamics_ode(t, x, u, inertia, h_w, B_eci, r_eci)"
        },
        {
            "setup": """
t = 0.0
x = np.array([1.0, 0, 0, 0, 0.01, 0.01, 0.1])
u = np.zeros(3)
inertia = np.diag([0.03, 0.03, 0.015])
h_w = 0.05 # Paper value
B_eci = np.zeros(3)
r_eci = np.array([7e6, 0, 0])
""",
            "call": "dynamics_ode(t, x, u, inertia, h_w, B_eci, r_eci)",
            "gold_call": "_gold_dynamics_ode(t, x, u, inertia, h_w, B_eci, r_eci)"
        },
        {
            "setup": """
t = 0.0
x = np.array([1.0, 0, 0, 0, 0, 0, 0])
u = np.array([0.1, 0.1, 0.1])
inertia = np.diag([0.1, 0.1, 0.1])
h_w = 0.0
B_eci = np.array([0, 0, 5e-5])
r_eci = np.array([1e10, 0, 0])
""",
            "call": "dynamics_ode(t, x, u, inertia, h_w, B_eci, r_eci)",
            "gold_call": "_gold_dynamics_ode(t, x, u, inertia, h_w, B_eci, r_eci)"
        }
    ]

