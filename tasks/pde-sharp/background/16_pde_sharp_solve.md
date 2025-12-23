# Background: The PDE-SHARP Framework

## Overview

PDE-SHARP implements a three-stage process for constructing and selecting PDE solvers. Rather than using a single fixed numerical scheme, it evaluates multiple candidates and selects the best one.

## Three-Stage Process

### Stage 1: Analysis
1. **Classify PDE type**: Identify whether we have Poisson, convection-diffusion, etc.
2. **Check linearity**: Determine if the PDE is linear or nonlinear
3. **Analyze stability**: For time-dependent problems, determine CFL constraints
4. **Classify mathematically**: Elliptic, parabolic, or hyperbolic
5. **Recommend schemes**: Generate list of candidate numerical methods

### Stage 2: Genesis
1. **Select discretization**: Choose finite difference stencil and time scheme
2. **Setup grid**: Extract spacing (dx, dy) and dimensions (Ny, Nx)
3. **Initialize solution**: Create initial condition array
4. **Apply boundaries**: Set Dirichlet boundary conditions

### Stage 3: Synthesis
1. **Generate candidates**: List all recommended numerical schemes
2. **Evaluate each**: Run each scheme and measure error/residual
3. **Rank and select**: Choose the solution with lowest error

## Key Innovation

The multi-shot evaluation approach allows the framework to:
- Adapt to problem-specific characteristics
- Find optimal relaxation parameters (e.g., SOR omega)
- Compare explicit vs. implicit schemes
- Select the most accurate result automatically

## Supported Problems

| Problem | Classification | Methods |
|---------|----------------|---------|
| Poisson | Elliptic | Jacobi, Gauss-Seidel, SOR |
| Convection-Diffusion | Parabolic | Upwind, Central, Implicit |

