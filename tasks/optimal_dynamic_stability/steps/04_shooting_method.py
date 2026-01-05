"""
Find the optimal switching time for T-periodic stabilization using a shooting method.

The shooting method seeks a switching time t_s such that the trajectory 
starting with x'(0) = 0 returns to x'(T/2) = 0, ensuring T-periodicity 
for symmetric bang-bang profiles.
"""

import numpy as np
from scipy.optimize import brentq

# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def find_switching_time(T, u_plus, u_minus):
    """
    Finds the switching time t_s in [0, T/2] such that the solution is T-periodic.
    Using the shooting method, it finds roots of x'(T/2) = 0 for an initial state [1, 0].
    
    Parameters
    ----------
    T : float
        Period.
    u_plus : float
        Positive stiffness.
    u_minus : float
        Negative stiffness.
        
    Returns
    -------
    t_s : float
        The switching time.
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

def _gold_find_switching_time(T, u_plus, u_minus):
    def _objective(ts):
        z1 = _get_transition_matrix(u_plus, ts) @ np.array([1.0, 0.0])
        z2 = _get_transition_matrix(u_minus, T/2 - ts) @ z1
        return z2[1]

    try:
        t_s = brentq(_objective, 0, T/2 - 1e-10)
        return t_s
    except ValueError:
        return None

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "T=1.5; u_plus=10.0; u_minus=-5.0",
            "call": "find_switching_time(T, u_plus, u_minus)",
            "gold_call": "_gold_find_switching_time(T, u_plus, u_minus)"
        },
        {
            "setup": "T=2.5; u_plus=5.0; u_minus=-2.0",
            "call": "find_switching_time(T, u_plus, u_minus)",
            "gold_call": "_gold_find_switching_time(T, u_plus, u_minus)"
        },
        {
            "setup": "T=0.5; u_plus=100.0; u_minus=-1.0",
            "call": "find_switching_time(T, u_plus, u_minus)",
            "gold_call": "_gold_find_switching_time(T, u_plus, u_minus)"
        }
    ]
