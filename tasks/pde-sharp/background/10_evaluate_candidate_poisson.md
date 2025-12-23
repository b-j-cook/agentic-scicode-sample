# Background: Iterative Solvers for Elliptic PDEs

## Overview

For elliptic PDEs like Poisson's equation, we use iterative relaxation methods to solve the discretized system. These methods start with an initial guess and iteratively improve until convergence.

## Discretization

The 5-point stencil discretization of the Laplacian on a uniform grid gives:

$$\frac{u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}}{h^2} = f_{i,j}$$

Rearranging for the update:

$$u_{i,j}^{new} = \frac{1}{4}(u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} + h^2 f_{i,j})$$

## Iterative Methods

### Jacobi Method
Uses only values from the previous iteration:
$$u^{n+1}_{i,j} = \frac{1}{4}(u^n_{i+1,j} + u^n_{i-1,j} + u^n_{i,j+1} + u^n_{i,j-1} + h^2 f_{i,j})$$

- Simplest method, easily parallelizable
- Slow convergence

### Gauss-Seidel Method
Uses updated values as soon as they're available:
$$u^{n+1}_{i,j} = \frac{1}{4}(u^n_{i+1,j} + u^{n+1}_{i-1,j} + u^n_{i,j+1} + u^{n+1}_{i,j-1} + h^2 f_{i,j})$$

- Faster than Jacobi
- Sequential updates

### Successive Over-Relaxation (SOR)
Accelerates Gauss-Seidel with relaxation parameter $\omega$:
$$u^{n+1}_{i,j} = (1-\omega)u^n_{i,j} + \omega \cdot u^{GS}_{i,j}$$

- Optimal $\omega$ typically between 1.5 and 1.9
- Much faster convergence for well-chosen $\omega$

