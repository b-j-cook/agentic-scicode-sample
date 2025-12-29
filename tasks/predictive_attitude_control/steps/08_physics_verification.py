"""
Stage 3: Optimal Control Synthesis - Physical Verification.
Implement a function to verify the physical consistency of the simulation, 
including angular momentum conservation and control torque orthogonality.
"""

import numpy as np


# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def verify_physics(x_history, u_history, B_body_history, tau_eci_history, dt, inertia, h_w):
    """
    Verify physical consistency of the simulation.
    Checks for:
    1. Maximum deviation in ECI angular momentum (Impulse-Momentum balance).
    2. Maximum absolute value of control torque projection on magnetic field.
    
    Parameters
    ----------
    x_history : np.ndarray (T, 7)
        History of states [q, w].
    u_history : np.ndarray (T-1, 3)
        History of magnetic dipole moments.
    B_body_history : np.ndarray (T-1, 3)
        History of magnetic field vectors in the body frame.
    tau_eci_history : np.ndarray (T-1, 3)
        History of total external torque in ECI frame.
    dt : float
        Time step (s).
    inertia : np.ndarray (3, 3)
        Inertia matrix.
    h_w : float
        Momentum wheel angular momentum.
        
    Returns
    -------
    verification_metrics : dict
        - 'momentum_error': float, max violation of Delta H = integral(tau) dt.
        - 'orthogonality_error': float, max (m x B) . B.
    """
    return result  # <- Return hint for the model


# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _gold_verify_physics(x_history, u_history, B_body_history, tau_eci_history, dt, inertia, h_w):
    T = x_history.shape[0]
    H_eci_actual = np.zeros((T, 3))
    h_w_vec = np.array([0.0, 0.0, float(h_w)])
    
    for i in range(T):
        q = x_history[i, 0:4]
        w = x_history[i, 4:7]
        
        # Rotation matrix ECI to Body
        q0, q1, q2, q3 = q
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-10:
            q0, q1, q2, q3 = q / q_norm
            
        C = np.array([
            [1 - 2*(q2**2 + q3**2), 2*(q1*q2 + q0*q3), 2*(q1*q3 - q0*q2)],
            [2*(q1*q2 - q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 + q0*q1)],
            [2*(q1*q3 + q0*q2), 2*(q2*q3 - q0*q1), 1 - 2*(q1**2 + q2**2)]
        ])
        
        h_body = inertia @ w + h_w_vec
        H_eci_actual[i] = C.T @ h_body
        
    H_expected = np.zeros((T, 3))
    H_expected[0] = H_eci_actual[0]
    for i in range(T - 1):
        # tau_eci_history[i] is the average torque for the interval
        H_expected[i+1] = H_expected[i] + tau_eci_history[i] * dt
        
    error_history = np.linalg.norm(H_eci_actual - H_expected, axis=1)
    momentum_error = float(np.max(error_history))
    
    # Orthogonality check: (m x B) . B should be 0
    ortho_errors = np.zeros(len(u_history))
    for i in range(len(u_history)):
        m = u_history[i]
        B = B_body_history[i]
        tau_mag = np.cross(m, B)
        ortho_errors[i] = np.abs(np.dot(tau_mag, B))
        
    orthogonality_error = float(np.max(ortho_errors))
    
    return {
        'momentum_error': momentum_error,
        'orthogonality_error': orthogonality_error
    }


# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
inertia = np.diag([0.1, 0.1, 0.1])
h_w = 0.01
dt = 0.1
x_history = np.zeros((5, 7))
x_history[:, 0] = 1.0 # Identity
x_history[:, 6] = 0.1 # Constant wz
# u_history has one fewer row because control is applied between state updates
u_history = np.zeros((4, 3))
B_body_history = np.zeros((4, 3))
tau_eci_history = np.zeros((4, 3))
""",
            "call": "verify_physics(x_history, u_history, B_body_history, tau_eci_history, dt, inertia, h_w)",
            "gold_call": "_gold_verify_physics(x_history, u_history, B_body_history, tau_eci_history, dt, inertia, h_w)"
        },
        {
            "setup": """
inertia = np.diag([0.1, 0.1, 0.1])
h_w = 0.0
dt = 1.0
x_history = np.zeros((3, 7))
x_history[:, 0] = 1.0
x_history[0, 4:7] = np.array([0, 0, 0])
x_history[1, 4:7] = np.array([1, 0, 0]) # wz increased by 1
x_history[2, 4:7] = np.array([2, 0, 0]) # wz increased by 1
u_history = np.zeros((2, 3))
B_body_history = np.zeros((2, 3))
tau_eci_history = np.array([[0.1, 0, 0], [0.1, 0, 0]]) # torque = 0.1, Delta H = 0.1*1.0
""",
            "call": "verify_physics(x_history, u_history, B_body_history, tau_eci_history, dt, inertia, h_w)",
            "gold_call": "_gold_verify_physics(x_history, u_history, B_body_history, tau_eci_history, dt, inertia, h_w)"
        },
        {
            "setup": """
inertia = np.eye(3)
h_w = 0.0
dt = 0.1
x_history = np.zeros((2, 7))
x_history[:, 0] = 1.0
u_history = np.array([[1.0, 0, 0]])
B_body_history = np.array([[0, 1.0, 0]])
tau_eci_history = np.zeros((1, 3))
""",
            "call": "verify_physics(x_history, u_history, B_body_history, tau_eci_history, dt, inertia, h_w)",
            "gold_call": "_gold_verify_physics(x_history, u_history, B_body_history, tau_eci_history, dt, inertia, h_w)"
        }
    ]
