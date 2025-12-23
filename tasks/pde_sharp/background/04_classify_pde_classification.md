# Background: PDE Mathematical Classification

## Overview

PDEs are classified into three mathematical categories based on the discriminant of their characteristic equation. This classification determines which numerical methods are appropriate.

## Classification Categories

### Elliptic PDEs
- **Characteristics**: No real characteristics; solution is smooth
- **Examples**: Laplace equation, Poisson equation
- **Physical meaning**: Equilibrium/steady-state problems
- **Numerical methods**: Iterative relaxation (Jacobi, Gauss-Seidel, SOR)

### Parabolic PDEs
- **Characteristics**: One family of real characteristics
- **Examples**: Heat equation, convection-diffusion
- **Physical meaning**: Diffusion-dominated, time-dependent
- **Numerical methods**: Explicit/implicit time-stepping, upwind schemes

### Hyperbolic PDEs
- **Characteristics**: Two families of real characteristics
- **Examples**: Wave equation, advection equation
- **Physical meaning**: Wave propagation
- **Numerical methods**: Characteristics methods, high-resolution schemes

## Mapping in PDE-SHARP

| PDE Type | Classification |
|----------|----------------|
| poisson_2d | elliptic |
| convection_diffusion_2d | parabolic |

