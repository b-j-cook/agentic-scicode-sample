"""
Evaluate a single candidate numerical scheme for the Poisson equation.

Implements iterative solvers: Jacobi, Gauss-Seidel, and SOR (Successive Over-Relaxation).
Returns the converged solution and residual error.
"""

import numpy as np
from typing import Dict, Tuple, List


# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def evaluate_candidate_poisson(
    candidate: str,
    u: np.ndarray,
    f: np.ndarray,
    dx: float,
    boundary_conditions: Dict,
    refinement_rounds: int
) -> Tuple[np.ndarray, float]:
    """
    Evaluate a single candidate numerical scheme for Poisson equation.
    
    Args:
        candidate: Scheme name ('jacobi', 'gauss_seidel', 'sor_omega_1.8', etc.)
        u: Initial solution field, shape (Ny, Nx).
        f: Source term array, shape (Ny, Nx).
        dx: Grid spacing (assumed uniform, dx = dy).
        boundary_conditions: Boundary condition specification.
        refinement_rounds: Number of refinement rounds (affects max iterations).
        
    Returns:
        Tuple of (solution, residual):
            solution: np.ndarray, shape (Ny, Nx), converged solution
            residual: float, mean squared residual error
    """
    return result  # <- Return hint for the model


# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _gold_evaluate_candidate_poisson(
    candidate: str,
    u: np.ndarray,
    f: np.ndarray,
    dx: float,
    boundary_conditions: Dict,
    refinement_rounds: int
) -> Tuple[np.ndarray, float]:
    """Reference implementation."""
    u_candidate = u.copy()
    omega = 1.0
    if 'sor' in candidate:
        omega = float(candidate.split('_')[-1])
    
    if u_candidate.shape[0] < 3 or u_candidate.shape[1] < 3:
        return u_candidate, 0.0
    
    max_iter = max(refinement_rounds * 5000, 20000)
    for iteration in range(max_iter):
        u_old = u_candidate.copy()
        interior = u_candidate[1:-1, 1:-1]
        
        if dx > 0 and not np.isnan(dx) and not np.isinf(dx):
            neighbors_sum = (u_candidate[2:, 1:-1] + u_candidate[:-2, 1:-1] +
                           u_candidate[1:-1, 2:] + u_candidate[1:-1, :-2])
            source_term = f[1:-1, 1:-1] * dx**2
            new_value = (neighbors_sum - source_term) / 4.0
            new_value = np.clip(new_value, -1e6, 1e6)
            u_candidate[1:-1, 1:-1] = (1 - omega) * interior + omega * new_value
            u_candidate = np.clip(u_candidate, -1e6, 1e6)
        
        u_candidate = apply_boundary_conditions(u_candidate, boundary_conditions)
        
        change = u_candidate[1:-1, 1:-1] - u_old[1:-1, 1:-1]
        residual = np.linalg.norm(change)
        norm_old = np.linalg.norm(u_old[1:-1, 1:-1])
        if norm_old > 1e-10:
            if residual / norm_old < 1e-10:
                break
        elif residual < 1e-10:
            break
    
    interior = u_candidate[1:-1, 1:-1]
    laplacian = (u_candidate[2:, 1:-1] + u_candidate[:-2, 1:-1] +
                 u_candidate[1:-1, 2:] + u_candidate[1:-1, :-2] - 4*interior)
    
    if dx > 0 and not np.isnan(dx) and not np.isinf(dx):
        residual_terms = np.clip(laplacian/dx**2 - f[1:-1, 1:-1], -1e10, 1e10)
        final_residual = np.mean(residual_terms**2)
    else:
        final_residual = np.inf
    
    return u_candidate, final_residual


# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Test case specifications."""
    return [
        {
            "setup": """N = 11
u = np.zeros((N, N))
f = np.zeros((N, N))
dx = 0.1
bc = {'values': {'left': 0, 'right': 0, 'bottom': 0, 'top': 0}}
refinement_rounds = 1""",
            "call": "evaluate_candidate_poisson('jacobi', u, f, dx, bc, refinement_rounds)",
            "gold_call": "_gold_evaluate_candidate_poisson('jacobi', u, f, dx, bc, refinement_rounds)",
        },
        {
            "setup": """N = 11
u = np.zeros((N, N))
f = np.ones((N, N))
dx = 0.1
bc = {'values': {'left': 0, 'right': 0, 'bottom': 0, 'top': 0}}
refinement_rounds = 2""",
            "call": "evaluate_candidate_poisson('sor_omega_1.8', u, f, dx, bc, refinement_rounds)",
            "gold_call": "_gold_evaluate_candidate_poisson('sor_omega_1.8', u, f, dx, bc, refinement_rounds)",
        },
    ]

