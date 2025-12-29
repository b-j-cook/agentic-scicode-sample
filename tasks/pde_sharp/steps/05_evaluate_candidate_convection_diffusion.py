"""
Evaluate a single candidate scheme for the convection-diffusion equation.

Implements time-stepping schemes: upwind, central difference, Lax-Wendroff.
Returns the solution after time evolution and an error metric.
"""

import numpy as np
from typing import Dict, Tuple, List


# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def evaluate_candidate_convection_diffusion(
    candidate: str,
    u: np.ndarray,
    grid: np.ndarray,
    parameters: Dict,
    dx: float,
    dy: float,
    boundary_conditions: Dict
) -> Tuple[np.ndarray, float]:
    """
    Evaluate a single candidate scheme for convection-diffusion equation.
    
    Args:
        candidate: Scheme name ('upwind_first', 'central_diffusion', etc.)
        u: Initial solution field, shape (Ny, Nx).
        grid: Discretization grid.
        parameters: Dict with 'dt', 'num_steps', 'velocity', 'viscosity', 'source'.
        dx, dy: Grid spacing in x and y directions.
        boundary_conditions: Boundary condition specification.
        
    Returns:
        Tuple of (solution, error):
            solution: np.ndarray, solution after num_steps time steps
            error: float, proxy error metric (gradient smoothness)
    """
    return result  # <- Return hint for the model


# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _apply_dirichlet(u: np.ndarray, boundary_conditions: Dict) -> np.ndarray:
    """Helper: Apply boundary values."""
    if 'values' in boundary_conditions:
        vals = boundary_conditions['values']
        if isinstance(vals, dict):
            if 'left' in vals: u[:,0] = vals['left']
            if 'right' in vals: u[:,-1] = vals['right']
            if 'bottom' in vals: u[0,:] = vals['bottom']
            if 'top' in vals: u[-1,:] = vals['top']
    return u


def _gold_evaluate_candidate_convection_diffusion(
    candidate: str,
    u: np.ndarray,
    grid: np.ndarray,
    parameters: Dict,
    dx: float,
    dy: float,
    boundary_conditions: Dict
) -> Tuple[np.ndarray, float]:
    """Reference implementation."""
    dt = parameters['dt']
    num_steps = parameters['num_steps']
    velocity = parameters.get('velocity', np.array([1.0, 0.0]))
    vx, vy = velocity
    nu = parameters.get('viscosity', 0.01)
    source = parameters.get('source', lambda grid, t: np.zeros_like(u))
    
    if u.shape[0] < 3 or u.shape[1] < 3 or dx <= 0 or dy <= 0:
        return u.copy(), 0.0
    
    u_candidate = u.copy()
    for step in range(num_steps):
        t = step * dt
        
        diff_x = (u_candidate[1:-1, 2:] - 2*u_candidate[1:-1, 1:-1] + u_candidate[1:-1, :-2]) / dx**2
        diff_y = (u_candidate[2:, 1:-1] - 2*u_candidate[1:-1, 1:-1] + u_candidate[:-2, 1:-1]) / dy**2
        diffusion = nu * (diff_x + diff_y)
        
        if 'upwind' in candidate:
            conv_x = np.where(vx > 0,
                              u_candidate[1:-1, 1:-1] - u_candidate[1:-1, :-2],
                              u_candidate[1:-1, 2:] - u_candidate[1:-1, 1:-1]) / dx
            conv_y = np.where(vy > 0,
                              u_candidate[1:-1, 1:-1] - u_candidate[:-2, 1:-1],
                              u_candidate[2:, 1:-1] - u_candidate[1:-1, 1:-1]) / dy
        else:
            conv_x = (u_candidate[1:-1, 2:] - u_candidate[1:-1, :-2]) / (2*dx)
            conv_y = (u_candidate[2:, 1:-1] - u_candidate[:-2, 1:-1]) / (2*dy)
        
        convection = vx * conv_x + vy * conv_y
        
        if callable(source):
            src = source(grid, t)
            if src.shape == u_candidate.shape:
                src = src[1:-1, 1:-1]
        else:
            src = source[1:-1, 1:-1] if source.shape == u_candidate.shape else source
        
        du = dt * (diffusion - convection + src)
        u_candidate[1:-1, 1:-1] += du
    
    u_candidate = _apply_dirichlet(u_candidate, boundary_conditions)
    proxy_error = np.std(np.abs(np.gradient(u_candidate)))
    
    return u_candidate, proxy_error


# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Test case specifications."""
    return [
        {
            "setup": """N = 11
u = np.zeros((N, N))
u[N//2-1:N//2+2, N//2-1:N//2+2] = 1.0  # Initial pulse
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)
grid = np.array([X, Y])
params = {'dt': 0.001, 'num_steps': 10, 'velocity': np.array([1.0, 0.0]), 
          'viscosity': 0.01, 'source': np.zeros((N, N))}
dx, dy = 0.1, 0.1
bc = {'values': {'left': 0, 'right': 0, 'bottom': 0, 'top': 0}}""",
            "call": "evaluate_candidate_convection_diffusion('upwind_first', u.copy(), grid, params, dx, dy, bc)",
            "gold_call": "_gold_evaluate_candidate_convection_diffusion('upwind_first', u.copy(), grid, params, dx, dy, bc)",
        },
        {
            "setup": """N = 11
u = np.zeros((N, N))
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)
grid = np.array([X, Y])
params = {'dt': 0.001, 'num_steps': 5, 'velocity': np.array([0.5, 0.5]), 
          'viscosity': 0.05, 'source': np.zeros((N, N))}
dx, dy = 0.1, 0.1
bc = {'values': {'left': 0, 'right': 0, 'bottom': 0, 'top': 0}}""",
            "call": "evaluate_candidate_convection_diffusion('central_diffusion', u.copy(), grid, params, dx, dy, bc)",
            "gold_call": "_gold_evaluate_candidate_convection_diffusion('central_diffusion', u.copy(), grid, params, dx, dy, bc)",
        },
    ]

