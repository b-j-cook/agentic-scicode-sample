"""
Apply boundary conditions to the solution field.

Set Dirichlet boundary values on all four sides of the domain.
"""

import numpy as np
from typing import Dict, Tuple, List


# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def apply_boundary_conditions(u: np.ndarray, boundary_conditions: Dict) -> np.ndarray:
    """
    Apply boundary conditions to the solution field.
    
    Args:
        u: Solution field, shape (Ny, Nx).
        boundary_conditions: Dictionary with 'values' key containing dict with
                            'left', 'right', 'bottom', 'top' (scalar or array).
        
    Returns:
        Solution field with boundary conditions applied, shape (Ny, Nx).
    """
    return result  # <- Return hint for the model


# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _gold_apply_boundary_conditions(u: np.ndarray, boundary_conditions: Dict) -> np.ndarray:
    """Reference implementation."""
    if 'values' in boundary_conditions:
        vals = boundary_conditions['values']
        if isinstance(vals, dict):
            if 'left' in vals: u[:,0] = vals['left']
            if 'right' in vals: u[:,-1] = vals['right']
            if 'bottom' in vals: u[0,:] = vals['bottom']
            if 'top' in vals: u[-1,:] = vals['top']
    return u


# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Test case specifications."""
    return [
        {
            "setup": """u = np.zeros((10, 10))
bc = {'values': {'left': 0, 'right': 1, 'bottom': 0, 'top': 1}}""",
            "call": "apply_boundary_conditions(u.copy(), bc)",
            "gold_call": "_gold_apply_boundary_conditions(u.copy(), bc)",
        },
        {
            "setup": """u = np.ones((15, 20)) * 0.5
bc = {'values': {'left': 0, 'right': 0, 'bottom': 0, 'top': 0}}""",
            "call": "apply_boundary_conditions(u.copy(), bc)",
            "gold_call": "_gold_apply_boundary_conditions(u.copy(), bc)",
        },
        {
            "setup": """u = np.zeros((10, 10))
left_vals = np.linspace(0, 1, 10)
bc = {'values': {'left': left_vals, 'right': 0, 'bottom': 0, 'top': 0}}""",
            "call": "apply_boundary_conditions(u.copy(), bc)",
            "gold_call": "_gold_apply_boundary_conditions(u.copy(), bc)",
        },
    ]

