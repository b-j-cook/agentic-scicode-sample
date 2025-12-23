"""
Initialize the solution field.

Create initial solution array, either from provided initial condition or zeros.
"""

import numpy as np
from typing import Dict, Tuple, List


# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def initialize_solution(parameters: Dict, Ny: int, Nx: int) -> np.ndarray:
    """
    Initialize solution field.
    
    Args:
        parameters: Dictionary with optional 'initial_condition' key (np.ndarray).
        Ny, Nx: Number of grid points in y and x directions.
        
    Returns:
        Initial solution field, shape (Ny, Nx).
        If 'initial_condition' is provided, returns a copy; otherwise returns zeros.
    """
    return result  # <- Return hint for the model


# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _gold_initialize_solution(parameters: Dict, Ny: int, Nx: int) -> np.ndarray:
    """Reference implementation."""
    u = np.zeros((Ny, Nx))
    if 'initial_condition' in parameters:
        u = parameters['initial_condition'].copy()
    return u


# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Test case specifications."""
    return [
        {
            "setup": """params = {}
Ny, Nx = 10, 15""",
            "call": "initialize_solution(params, Ny, Nx)",
            "gold_call": "_gold_initialize_solution(params, Ny, Nx)",
        },
        {
            "setup": """Ny, Nx = 10, 15
u_init = np.ones((Ny, Nx)) * 5.0
params = {'initial_condition': u_init}""",
            "call": "initialize_solution(params, Ny, Nx)",
            "gold_call": "_gold_initialize_solution(params, Ny, Nx)",
        },
        {
            "setup": """Ny, Nx = 20, 20
X, Y = np.meshgrid(np.linspace(0,1,Nx), np.linspace(0,1,Ny))
u_init = np.sin(np.pi * X) * np.sin(np.pi * Y)
params = {'initial_condition': u_init}""",
            "call": "initialize_solution(params, Ny, Nx)",
            "gold_call": "_gold_initialize_solution(params, Ny, Nx)",
        },
    ]

