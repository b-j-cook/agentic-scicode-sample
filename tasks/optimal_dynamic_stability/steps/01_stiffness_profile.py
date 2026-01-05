"""
Generate a T-periodic bang-bang stiffness profile.

The profile alternates between u_plus and u_minus with switching times determined by t_s.
Specifically, it is u_plus in [0, t_s], u_minus in [t_s, T - t_s], and u_plus in [T - t_s, T].
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def stiffness_profile(t, T, u_plus, u_minus, t_s):
    """
    Returns the stiffness u(t) for a given time t, T-periodic.
    
    The profile is:
    u(t) = u_plus  for t in [0, t_s]
    u(t) = u_minus for t in [t_s, T - t_s]
    u(t) = u_plus  for t in [T - t_s, T]
    
    Parameters
    ----------
    t : float or np.ndarray
        Time at which to evaluate the stiffness.
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
    u : float or np.ndarray
        The stiffness value(s).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _gold_stiffness_profile(t, T, u_plus, u_minus, t_s):
    t_mod = np.mod(t, T)
    if np.isscalar(t_mod):
        if t_mod <= t_s or t_mod >= T - t_s:
            return u_plus
        else:
            return u_minus
    else:
        u = np.full_like(t_mod, u_minus)
        mask_plus = (t_mod <= t_s) | (t_mod >= T - t_s)
        u[mask_plus] = u_plus
        return u

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "T=10.0; u_plus=1.0; u_minus=-1.0; t_s=2.0",
            "call": "[stiffness_profile(1.0, T, u_plus, u_minus, t_s), stiffness_profile(5.0, T, u_plus, u_minus, t_s), stiffness_profile(9.0, T, u_plus, u_minus, t_s)]",
            "gold_call": "[_gold_stiffness_profile(1.0, T, u_plus, u_minus, t_s), _gold_stiffness_profile(5.0, T, u_plus, u_minus, t_s), _gold_stiffness_profile(9.0, T, u_plus, u_minus, t_s)]"
        },
        {
            "setup": "T=1.0; u_plus=10.0; u_minus=-5.0; t_s=0.1",
            "call": "stiffness_profile(np.array([0.05, 0.5, 0.95]), T, u_plus, u_minus, t_s)",
            "gold_call": "_gold_stiffness_profile(np.array([0.05, 0.5, 0.95]), T, u_plus, u_minus, t_s)"
        },
        {
            "setup": "T=5.0; u_plus=2.0; u_minus=-3.0; t_s=1.0",
            "call": "stiffness_profile(np.array([1.0, 4.0, 6.0]), T, u_plus, u_minus, t_s)",
            "gold_call": "_gold_stiffness_profile(np.array([1.0, 4.0, 6.0]), T, u_plus, u_minus, t_s)"
        }
    ]
