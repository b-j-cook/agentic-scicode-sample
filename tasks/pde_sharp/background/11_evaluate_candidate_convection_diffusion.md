# Background: Time-Stepping Schemes for Parabolic PDEs

## Overview

For time-dependent PDEs like convection-diffusion, we discretize in both space and time. The convection-diffusion equation:

$$\frac{\partial u}{\partial t} + \mathbf{v} \cdot \nabla u = \nu \nabla^2 u + f$$

has convection (first-order spatial derivatives) and diffusion (second-order) terms.

## Spatial Discretization

### Central Differences (for diffusion)
$$\frac{\partial^2 u}{\partial x^2} \approx \frac{u_{i+1} - 2u_i + u_{i-1}}{\Delta x^2}$$

Second-order accurate but can cause oscillations for convection-dominated flows.

### Upwind Scheme (for convection)
For positive velocity $v_x > 0$:
$$\frac{\partial u}{\partial x} \approx \frac{u_i - u_{i-1}}{\Delta x}$$

For negative velocity $v_x < 0$:
$$\frac{\partial u}{\partial x} \approx \frac{u_{i+1} - u_i}{\Delta x}$$

First-order accurate but stable and monotone.

## Time Integration

### Explicit Euler
$$u^{n+1} = u^n + \Delta t \cdot F(u^n)$$

Simple but has stability constraints (CFL condition).

### Stability Constraints

For explicit methods, the time step must satisfy:
- **Diffusion**: $\Delta t < \frac{\Delta x^2}{2\nu}$
- **Advection**: $\Delta t < \frac{\Delta x}{|v|}$

The CFL (Courant-Friedrichs-Lewy) condition must be satisfied for stability.

