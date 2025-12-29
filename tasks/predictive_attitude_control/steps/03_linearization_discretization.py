"""
Stage 2: Linearization & Discretization - Linearize the nonlinear dynamics and discretize the resulting LTV system.
"""

import numpy as np
from scipy.linalg import expm


# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def linearize_dynamics(x_ref, u_ref, inertia, h_w, B_body, r_body):
    """
    Compute the linearized state-space matrices for the 6D error state.
    
    The error state is defined as [delta_theta, delta_omega], where delta_theta 
    is a 3D small-angle representation of attitude error and delta_omega 
    is the angular velocity error.
    
    Parameters
    ----------
    x_ref : np.ndarray
        Reference state [q, w] (7 elements) around which to linearize.
    u_ref : np.ndarray
        Reference control input [m] (3 elements).
    inertia : np.ndarray
        3x3 inertia matrix (kg·m²).
    h_w : float
        Momentum wheel angular momentum (N·m·s).
    B_body : np.ndarray
        Magnetic field vector in the body frame (Tesla).
    r_body : np.ndarray
        Position vector in the body frame (m).
        
    Returns
    -------
    A : np.ndarray
        6x6 continuous-time state matrix.
    B : np.ndarray
        6x3 continuous-time control matrix.
    """
    return result  # <- Return hint for the model


def _discretize_ltv(A, B, dt):
    """
    Discretize the continuous state-space matrices using Zero-Order Hold (ZOH).
    
    Parameters
    ----------
    A : np.ndarray
        nxn continuous-time state matrix.
    B : np.ndarray
        nxm continuous-time control matrix.
    dt : float
        Discretization time step (s).
        
    Returns
    -------
    Ak : np.ndarray
        nxn discrete-time state matrix.
    Bk : np.ndarray
        nxm discrete-time control matrix.
    """
    return result  # <- Return hint for the model

# =============================================================================
# GOLD SOLUTION
# =============================================================================
def _gold_linearize_dynamics(x_ref, u_ref, inertia, h_w, B_body, r_body):
    w_ref = x_ref[4:7]
    h_w_vec = np.array([0.0, 0.0, float(h_w)])
    
    def _skew(v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    
    A11 = -_skew(w_ref)
    A12 = np.eye(3)
    
    H_ref = np.dot(inertia, w_ref) + h_w_vec
    term_w = _skew(H_ref) - _skew(w_ref) @ inertia
    
    mu = 3.986004418e14
    r_norm = np.linalg.norm(r_body)
    if r_norm > 1e-6:
        K = 3.0 * mu / r_norm**5
        rb_skew = _skew(r_body)
        Irb_skew = _skew(np.dot(inertia, r_body))
        A21_tau = K * (rb_skew @ inertia - Irb_skew) @ rb_skew
    else:
        A21_tau = np.zeros((3, 3))
    
    # Add magnetic torque derivative w.r.t. attitude: [m]x [B]x
    A21_tau += _skew(u_ref) @ _skew(B_body)
    
    A21 = np.linalg.solve(inertia, A21_tau)
        
    A22 = np.linalg.solve(inertia, term_w)
    
    A = np.block([
        [A11, A12],
        [A21, A22]
    ])
    
    B_w = np.linalg.solve(inertia, -_skew(B_body))
    B = np.block([
        [np.zeros((3, 3))],
        [B_w]
    ])
    
    return A, B

def _gold_discretize_ltv(A, B, dt):
    n = A.shape[0]
    m = B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    Phi = expm(M * dt)
    Ak = Phi[:n, :n]
    Bk = Phi[:n, n:]
    return Ak, Bk

# =============================================================================
# TEST CASES
# =============================================================================
def test_cases():
    return [
        {
            "setup": """
x_ref = np.array([1.0, 0, 0, 0, 0, 0, 0.1])
u_ref = np.zeros(3)
inertia = np.diag([0.1, 0.1, 0.05])
h_w = 0.01
B_body = np.array([0, 3e-5, 0])
r_body = np.array([7e6, 0, 0])
""",
            "call": "linearize_dynamics(x_ref, u_ref, inertia, h_w, B_body, r_body)",
            "gold_call": "_gold_linearize_dynamics(x_ref, u_ref, inertia, h_w, B_body, r_body)"
        },
        {
            "setup": """
x_ref = np.array([0.99, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
u_ref = np.array([0.05, 0.05, 0.05])
inertia = np.diag([0.2, 0.2, 0.1])
h_w = 0.05
B_body = np.array([1e-5, 2e-5, 3e-5])
r_body = np.array([4e6, 4e6, 4e6])
""",
            "call": "linearize_dynamics(x_ref, u_ref, inertia, h_w, B_body, r_body)",
            "gold_call": "_gold_linearize_dynamics(x_ref, u_ref, inertia, h_w, B_body, r_body)"
        },
        {
            "setup": """
A = np.array([
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [-1, -2, -3, -0.1, -0.2, -0.3],
    [0.1, 0.2, 0.3, -1, -1, -1],
    [1, 1, 1, 0, 0, 0]
])
B = np.zeros((6, 3))
B[3:6, :] = np.eye(3)
dt = 1e-4
""",
            "call": "_discretize_ltv(A, B, dt)",
            "gold_call": "_gold_discretize_ltv(A, B, dt)"
        }
    ]

