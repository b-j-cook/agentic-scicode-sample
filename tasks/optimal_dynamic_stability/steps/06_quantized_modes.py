"""
Identify higher-order quantized stabilization modes.

Extends the shooting method to find switching times for excited states, 
corresponding to stabilization modes with increasing numbers of zero-crossings.
"""

import numpy as np
from scipy.optimize import brentq

# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def find_quantized_switching_time(T, u_plus, u_minus, n_mode):
    """
    Finds the switching time t_s for the n-th stabilization mode.
    For mode n, x(t) has n zeros in the interval (0, T/2).
    
    Parameters
    ----------
    T : float
        Period.
    u_plus : float
        Positive stiffness.
    u_minus : float
        Negative stiffness.
    n_mode : int
        Mode index (0, 1, 2...).
        
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

def _gold_find_quantized_switching_time(T, u_plus, u_minus, n_mode):
    def _objective(ts):
        z1 = _get_transition_matrix(u_plus, ts) @ np.array([1.0, 0.0])
        z2 = _get_transition_matrix(u_minus, T/2 - ts) @ z1
        return z2[1]

    # Use a denser grid for root finding to catch all modes
    omega = np.sqrt(u_plus)
    n_pts = int(max(2000, 20 * (omega * T / (2 * np.pi))))
    ts_vals = np.linspace(0, T/2, n_pts)
    
    roots = []
    f_vals = [_objective(t) for t in ts_vals]
    for i in range(len(ts_vals) - 1):
        if f_vals[i] * f_vals[i+1] <= 0:
            try:
                roots.append(brentq(_objective, ts_vals[i], ts_vals[i+1], xtol=1e-12))
            except ValueError:
                continue
    
    if n_mode < len(roots):
        return roots[n_mode]
    return None

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "T=2.0; u_plus=50.0; u_minus=-10.0",
            "call": "[find_quantized_switching_time(T, u_plus, u_minus, 0), find_quantized_switching_time(T, u_plus, u_minus, 1)]",
            "gold_call": "[_gold_find_quantized_switching_time(T, u_plus, u_minus, 0), _gold_find_quantized_switching_time(T, u_plus, u_minus, 1)]"
        },
        {
            "setup": "T=10.0; u_plus=100.0; u_minus=-1.0",
            "call": "find_quantized_switching_time(T, u_plus, u_minus, 2)",
            "gold_call": "_gold_find_quantized_switching_time(T, u_plus, u_minus, 2)"
        },
        {
            "setup": "T=1.0; u_plus=1000.0; u_minus=-100.0",
            "call": "find_quantized_switching_time(T, u_plus, u_minus, 0)",
            "gold_call": "_gold_find_quantized_switching_time(T, u_plus, u_minus, 0)"
        }
    ]
