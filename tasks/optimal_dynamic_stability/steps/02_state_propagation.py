"""
Propagate the state of a linear mass-spring system with constant or varying stiffness.

Includes analytical solutions for different stiffness regimes (u > 0, u < 0, u = 0)
and special handling for the large T limit to maintain numerical stability.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def propagate_state(z0, u, dt, large_t=False):
    """
    Propagates the state [x, x'] over time dt with constant stiffness u.
    
    Parameters
    ----------
    z0 : np.ndarray (2,)
        Initial state [x0, x0'].
    u : float
        Stiffness value.
    dt : float
        Time interval.
    large_t : bool, optional
        If True, uses asymptotic formulas for u < 0 to avoid overflow.
        
    Returns
    -------
    z1 : np.ndarray (2,)
        The propagated state.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _get_transition_matrix(u, dt):
    """
    Computes the analytic transition matrix for a constant stiffness u over time dt.
    """
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

def _gold_propagate_state(z0, u, dt, large_t=False):
    if not large_t or u >= 0:
        Phi = _get_transition_matrix(u, dt)
        return Phi @ z0
    
    gamma = np.sqrt(-u)
    exp_pos = np.exp(gamma * dt)
    exp_neg = np.exp(-gamma * dt)
    
    c_plus = 0.5 * (z0[0] + z0[1] / gamma)
    c_minus = 0.5 * (z0[0] - z0[1] / gamma)
    
    x1 = c_plus * exp_pos + c_minus * exp_neg
    v1 = gamma * (c_plus * exp_pos - c_minus * exp_neg)
    
    return np.array([x1, v1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "z0 = np.array([1.0, 0.0]); u=4.0; dt=np.pi/4",
            "call": "propagate_state(z0, u, dt)",
            "gold_call": "_gold_propagate_state(z0, u, dt)"
        },
        {
            "setup": "z0 = np.array([1.0, 0.0]); u=-4.0; dt=1.0",
            "call": "[propagate_state(z0, u, dt, False), propagate_state(z0, u, dt, True)]",
            "gold_call": "[_gold_propagate_state(z0, u, dt, False), _gold_propagate_state(z0, u, dt, True)]"
        },
        {
            "setup": "z0 = np.array([1.0, 1.0]); u=0.0; dt=5.0",
            "call": "propagate_state(z0, u, dt)",
            "gold_call": "_gold_propagate_state(z0, u, dt)"
        },
        {
            "setup": "z0 = np.array([1.0, 0.0]); u=-1.0; dt=100.0",
            "call": "propagate_state(z0, u, dt, True)",
            "gold_call": "_gold_propagate_state(z0, u, dt, True)"
        }
    ]
