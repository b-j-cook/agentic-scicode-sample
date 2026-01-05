"""
Find the absolute minimal average stiffness threshold required for stability.

Performs a parameter sweep over u_minus to find the critical boundary 
where a stable T-periodic solution emerges, determining the minimal 
average stiffness <u(t)>.
"""

import numpy as np
from scipy.optimize import brentq

# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def find_minimal_average_stiffness(T, u_plus, u_minus_range):
    """
    Finds the absolute minimal average stiffness threshold by sweeping over u_minus.
    
    Parameters
    ----------
    T : float
        Period.
    u_plus : float
        Positive stiffness.
    u_minus_range : tuple (float, float)
        Range to search for the critical u_minus.
        
    Returns
    -------
    min_avg_u : float
        The minimal average stiffness found.
    optimal_ts : float
        The optimal switching time at the threshold.
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

def _gold_find_minimal_average_stiffness(T, u_plus, u_minus_range):
    u_minus_min, u_minus_max = u_minus_range
    
    def _get_min_abs_trace(u_neg):
        # Sample the trace over ts to find the minimum for this u_neg
        ts_vals = np.linspace(0, T/2, 500)
        traces = []
        for ts in ts_vals:
            M = _compute_monodromy_matrix(T, u_plus, u_neg, ts)
            traces.append(np.abs(np.trace(M)))
        return np.min(traces) - 2.0

    try:
        # The critical u_minus is where the minimum possible trace is exactly 2
        critical_u_minus = brentq(_get_min_abs_trace, u_minus_min, u_minus_max, xtol=1e-8)
    except (ValueError, RuntimeError):
        critical_u_minus = u_minus_min
        
    # Find the ts that achieves this minimum
    ts_vals = np.linspace(0, T/2, 2000)
    best_ts = 0.0
    min_tr = 1e10
    for ts in ts_vals:
        M = _compute_monodromy_matrix(T, u_plus, critical_u_minus, ts)
        tr = np.abs(np.trace(M))
        if tr < min_tr:
            min_tr = tr
            best_ts = ts
            
    avg_u = (2 * best_ts * u_plus + (T - 2 * best_ts) * critical_u_minus) / T
    return avg_u, best_ts

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "T=2.0; u_plus=10.0; u_minus_range=(-10.0, 0.0)",
            "call": "find_minimal_average_stiffness(T, u_plus, u_minus_range)",
            "gold_call": "_gold_find_minimal_average_stiffness(T, u_plus, u_minus_range)"
        },
        {
            "setup": "T=5.0; u_plus=5.0; u_minus_range=(-5.0, 0.0)",
            "call": "find_minimal_average_stiffness(T, u_plus, u_minus_range)",
            "gold_call": "_gold_find_minimal_average_stiffness(T, u_plus, u_minus_range)"
        },
        {
            "setup": "T=1.0; u_plus=20.0; u_minus_range=(-20.0, 0.0)",
            "call": "find_minimal_average_stiffness(T, u_plus, u_minus_range)",
            "gold_call": "_gold_find_minimal_average_stiffness(T, u_plus, u_minus_range)"
        }
    ]
