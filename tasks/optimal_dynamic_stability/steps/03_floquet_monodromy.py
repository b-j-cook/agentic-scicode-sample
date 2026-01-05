"""
Construct the Floquet monodromy matrix for a periodically modulated system.

The monodromy matrix is the state transition matrix over one full period T,
formed by the composition of transition matrices for each piecewise-constant interval.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def compute_monodromy_matrix(T, u_plus, u_minus, t_s):
    """
    Computes the Floquet monodromy matrix for one period T.
    
    The profile is u_plus for [0, t_s], u_minus for [t_s, T - 2*t_s],
    and u_plus for [T - t_s, T].
    
    Parameters
    ----------
    T : float
        Period of the modulation.
    u_plus : float
        Stiffness value in the 'positive' regions.
    u_minus : float
        Stiffness value in the 'negative' region.
    t_s : float
        The first switching time.
        
    Returns
    -------
    M : np.ndarray (2, 2)
        The monodromy matrix Phi(T).
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

def _gold_compute_monodromy_matrix(T, u_plus, u_minus, t_s):
    Phi1 = _get_transition_matrix(u_plus, t_s)
    Phi2 = _get_transition_matrix(u_minus, T - 2 * t_s)
    Phi3 = _get_transition_matrix(u_plus, t_s)
    return Phi3 @ (Phi2 @ Phi1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "T=2.0; u_plus=1.0; u_minus=-1.0; t_s=0.5",
            "call": "compute_monodromy_matrix(T, u_plus, u_minus, t_s)",
            "gold_call": "_gold_compute_monodromy_matrix(T, u_plus, u_minus, t_s)"
        },
        {
            "setup": "T=10.0; u_plus=4.0; u_minus=-1.0; t_s=1.0",
            "call": "compute_monodromy_matrix(T, u_plus, u_minus, t_s)",
            "gold_call": "_gold_compute_monodromy_matrix(T, u_plus, u_minus, t_s)"
        },
        {
            "setup": "T=1.0; u_plus=0.0; u_minus=0.0; t_s=0.25",
            "call": "compute_monodromy_matrix(T, u_plus, u_minus, t_s)",
            "gold_call": "_gold_compute_monodromy_matrix(T, u_plus, u_minus, t_s)"
        }
    ]
