"""
Stage 3: Optimal Control Synthesis - Propagate the nonlinear satellite state forward by one control interval [t_k, t_{k+1}]
using the first element of the optimized control sequence.
"""

import numpy as np
from scipy.integrate import solve_ivp
import os
import sys

# Import dependencies from previous steps
import importlib
import sys
import os

_curr_dir = os.path.dirname(__file__)
if _curr_dir not in sys.path:
    sys.path.append(_curr_dir)

_mod02 = importlib.import_module('02_dynamics_ode')
_gold_dynamics_ode = _mod02._gold_dynamics_ode


# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def propagate_step(x_k, u_k, dt, inertia, h_w, B_eci, r_eci):
    """
    Integrate the nonlinear dynamics for one time step.
    
    Parameters
    ----------
    x_k : np.ndarray
        Initial state at t_k (7 elements).
    u_k : np.ndarray
        Applied control input (3 elements).
    dt : float
        Time step (s).
    inertia : np.ndarray
        3x3 inertia matrix.
    h_w : float
        Momentum wheel momentum magnitude.
    B_eci : np.ndarray
        Magnetic field in ECI frame.
    r_eci : np.ndarray
        Position vector in ECI frame.
    
    Returns
    -------
    x_next : np.ndarray
        State at t_{k+1} (7 elements).
    """
    return result  # <- Return hint for the model

# =============================================================================
# GOLD SOLUTION
# =============================================================================
def _gold_propagate_step(x_k, u_k, dt, inertia, h_w, B_eci, r_eci, orbit_params=None, t0=0.0):
    if orbit_params is not None:
        import importlib
        import sys
        import os
        _curr_dir = os.path.dirname(__file__)
        if _curr_dir not in sys.path:
            sys.path.append(_curr_dir)
            
        _mod01 = importlib.import_module('01_environmental_models')
        _mag_field_func = _mod01._gold_magnetic_field
        _orbit_pos_func = _mod01._gold_get_orbit_pos
        
        def _time_varying_ode(t, x, u, inertia, h_w, orbit_params, t0):
            t_abs = t0 + t
            B_t = _mag_field_func(t_abs, orbit_params)
            r_t = _orbit_pos_func(t_abs, orbit_params)
            return _gold_dynamics_ode(t, x, u, inertia, h_w, B_t, r_t)
            
        sol = solve_ivp(
            _time_varying_ode,
            [0, dt],
            x_k,
            args=(u_k, inertia, h_w, orbit_params, t0),
            method='RK45',
            rtol=1e-8, atol=1e-10
        )
    else:
        sol = solve_ivp(
            _gold_dynamics_ode,
            [0, dt],
            x_k,
            args=(u_k, inertia, h_w, B_eci, r_eci),
            method='RK45',
            rtol=1e-8, atol=1e-10
        )
    
    x_next = sol.y[:, -1]
    
    # Normalize quaternion
    q = x_next[0:4]
    q = q / np.linalg.norm(q)
    x_next[0:4] = q
    
    return x_next

# =============================================================================
# TEST CASES
# =============================================================================
def test_cases():
    return [
        {
            "setup": """
x_k = np.array([1.0, 0, 0, 0, 0.1, 0.1, 0.1])
u_k = np.array([0.5, 0.5, 0.5]) # Large control
dt = 1.0
inertia = np.diag([0.1, 0.1, 0.05])
h_w = 0.01
B_eci = np.array([3e-5, 0, 0])
r_eci = np.array([7e6, 0, 0])
""",
            "call": "propagate_step(x_k, u_k, dt, inertia, h_w, B_eci, r_eci)",
            "gold_call": "_gold_propagate_step(x_k, u_k, dt, inertia, h_w, B_eci, r_eci)"
        },
        {
            "setup": """
x_k = np.array([0.9914, 0.1305, 0, 0, 0, 0, 0])
u_k = np.array([0.2, 0, 0]) # Maximum dipole Am^2
dt = 2.0
inertia = np.diag([0.03, 0.03, 0.015])
h_w = 0.05
B_eci = np.array([0, 3e-5, 0])
r_eci = np.array([7e6, 0, 0])
""",
            "call": "propagate_step(x_k, u_k, dt, inertia, h_w, B_eci, r_eci)",
            "gold_call": "_gold_propagate_step(x_k, u_k, dt, inertia, h_w, B_eci, r_eci)"
        },
        {
            "setup": """
x_k = np.array([1.0, 0, 0, 0, 0, 0, 0.1])
u_k = np.zeros(3)
dt = 5.0 # Large dt
inertia = np.diag([0.1, 0.1, 0.1])
h_w = 0.0
B_eci = np.zeros(3)
r_eci = np.array([1e10, 0, 0])
""",
            "call": "propagate_step(x_k, u_k, dt, inertia, h_w, B_eci, r_eci)",
            "gold_call": "_gold_propagate_step(x_k, u_k, dt, inertia, h_w, B_eci, r_eci)"
        }
    ]

