"""
Extract grid parameters from the discretization grid.

Parse the grid array to determine spacing (dx, dy) and dimensions (Ny, Nx).
Handles both meshgrid format and 1D coordinate arrays.
"""

import numpy as np
from typing import Dict, Tuple, List


# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def setup_grid(grid: np.ndarray) -> Tuple[float, float, int, int]:
    """
    Extract grid parameters from the discretization grid.
    
    Args:
        grid: Discretization grid, shape (2, Ny, Nx) or (2, N).
              grid[0] is X coordinates, grid[1] is Y coordinates.
        
    Returns:
        Tuple of (dx, dy, Ny, Nx) where:
            dx: Grid spacing in x direction
            dy: Grid spacing in y direction
            Ny: Number of grid points in y direction
            Nx: Number of grid points in x direction
    """
    return result  # <- Return hint for the model


# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _gold_setup_grid(grid: np.ndarray) -> Tuple[float, float, int, int]:
    """Reference implementation."""
    if grid.ndim == 3:
        Ny, Nx = grid.shape[1], grid.shape[2]
        x = grid[0]
        y = grid[1]
    else:
        x = grid[0]
        y = grid[1]
        if x.ndim == 1:
            Nx = len(x)
            Ny = len(y)
        else:
            Ny, Nx = x.shape
    
    if x.ndim == 2:
        dx = x[0, 1] - x[0, 0] if Nx > 1 else 1.0
        dy = y[1, 0] - y[0, 0] if Ny > 1 else 1.0
    else:
        dx = x[1] - x[0] if len(x) > 1 else 1.0
        dy = y[1] - y[0] if len(y) > 1 else 1.0
    
    return dx, dy, Ny, Nx


# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Test case specifications."""
    return [
        {
            "setup": """N = 21
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)
grid = np.array([X, Y])""",
            "call": "setup_grid(grid)",
            "gold_call": "_gold_setup_grid(grid)",
        },
        {
            "setup": """Nx, Ny = 11, 21
x = np.linspace(0, 2, Nx)
y = np.linspace(0, 1, Ny)
X, Y = np.meshgrid(x, y)
grid = np.array([X, Y])""",
            "call": "setup_grid(grid)",
            "gold_call": "_gold_setup_grid(grid)",
        },
    ]

