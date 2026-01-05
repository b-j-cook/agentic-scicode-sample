"""
Physical verification of the optimal dynamical stabilization model.
Checks for symplecticity of the monodromy matrix, trace conditions for periodic modes,
and consistency with the Schrödinger analogy.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def verify_stabilization_physics(T, u_plus, u_minus, t_s, expected_avg_u, eigenvalues):
    """
    Verifies the physical consistency of the stabilization model.
    
    Checks:
    1. Determinant of the monodromy matrix (should be 1.0).
    2. Trace of the monodromy matrix (should be <= 2.0 for stability).
    3. Consistency between the calculated average stiffness and the Schrödinger energy level.
    
    Parameters
    ----------
    T : float
        Period.
    u_plus : float
        Positive stiffness.
    u_minus : float
        Negative stiffness.
    t_s : float
        Switching time.
    expected_avg_u : float
        The average stiffness calculated for this profile.
    eigenvalues : np.ndarray
        Eigenvalues from the Schrödinger analogy.
        
    Returns
    -------
    verification_results : tuple
        - det_error: float, abs(det(M) - 1).
        - trace_error: float, max(0, abs(trace(M)) - 2).
        - analogy_error: float, abs(expected_avg_u - (-2 * eigenvalues[0])).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _get_transition_matrix(u, dt):
    """Helper: compute analytic transition matrix for constant stiffness."""
    if u > 0:
        omega = np.sqrt(u)
        cos_w = np.cos(omega * dt)
        sin_w = np.sin(omega * dt)
        return np.array([
            [cos_w, sin_w / omega],
            [-omega * sin_w, cos_w]
        ])
    elif u < 0:
        gamma = np.sqrt(-u)
        cosh_g = np.cosh(gamma * dt)
        sinh_g = np.sinh(gamma * dt)
        return np.array([
            [cosh_g, sinh_g / gamma],
            [gamma * sinh_g, cosh_g]
        ])
    else:
        return np.array([
            [1.0, dt],
            [0.0, 1.0]
        ])

def _compute_monodromy_matrix(T, u_plus, u_minus, t_s):
    """Helper: compute Floquet monodromy matrix."""
    Phi1 = _get_transition_matrix(u_plus, t_s)
    Phi2 = _get_transition_matrix(u_minus, T - 2 * t_s)
    Phi3 = _get_transition_matrix(u_plus, t_s)
    return Phi3 @ (Phi2 @ Phi1)

def _gold_verify_stabilization_physics(T, u_plus, u_minus, t_s, expected_avg_u, eigenvalues):
    M = _compute_monodromy_matrix(T, u_plus, u_minus, t_s)
    det_M = np.linalg.det(M)
    trace_M = np.trace(M)
    
    det_error = np.abs(det_M - 1.0)
    trace_error = max(0.0, np.abs(trace_M) - 2.0)
    
    # In the large T limit, <u_min> ~ -2 * E_0
    # where E_0 is the ground state energy of the Schrödinger analogy.
    analogy_error = np.abs(expected_avg_u - (-2 * eigenvalues[0]))
    
    return float(det_error), float(trace_error), float(analogy_error)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
T = 10.0
u_plus = 20.0
u_minus = -5.0
t_s = 0.5
avg_u = (2*t_s*u_plus + (T-2*t_s)*u_minus)/T
eigenvalues = np.array([-avg_u/2.0, 0.0, 1.0])
""",
            "call": "verify_stabilization_physics(T, u_plus, u_minus, t_s, avg_u, eigenvalues)",
            "gold_call": "_gold_verify_stabilization_physics(T, u_plus, u_minus, t_s, avg_u, eigenvalues)"
        },
        {
            # extreme stiffness ratio and switching time near boundary
            "setup": """
T = 5.0
u_plus = 100.0
u_minus = -1.0
t_s = 2.4
avg_u = (2*t_s*u_plus + (T-2*t_s)*u_minus)/T
eigenvalues = np.array([-avg_u/2.0, 0.5, 2.0])
""",
            "call": "verify_stabilization_physics(T, u_plus, u_minus, t_s, avg_u, eigenvalues)",
            "gold_call": "_gold_verify_stabilization_physics(T, u_plus, u_minus, t_s, avg_u, eigenvalues)"
        },
        {
            # symmetric stiffness magnitudes with small period
            "setup": """
T = 1.0
u_plus = 50.0
u_minus = -50.0
t_s = 0.1
avg_u = (2*t_s*u_plus + (T-2*t_s)*u_minus)/T
eigenvalues = np.array([-avg_u/2.0, 1.0, 3.0])
""",
            "call": "verify_stabilization_physics(T, u_plus, u_minus, t_s, avg_u, eigenvalues)",
            "gold_call": "_gold_verify_stabilization_physics(T, u_plus, u_minus, t_s, avg_u, eigenvalues)"
        },
        {
            # very small switching time (nearly all u_minus)
            "setup": """
T = 8.0
u_plus = 10.0
u_minus = -2.0
t_s = 0.05
avg_u = (2*t_s*u_plus + (T-2*t_s)*u_minus)/T
eigenvalues = np.array([-avg_u/2.0, 0.1, 0.5])
""",
            "call": "verify_stabilization_physics(T, u_plus, u_minus, t_s, avg_u, eigenvalues)",
            "gold_call": "_gold_verify_stabilization_physics(T, u_plus, u_minus, t_s, avg_u, eigenvalues)"
        }
    ]
