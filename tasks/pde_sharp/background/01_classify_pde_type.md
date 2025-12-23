# Background: PDE Type Classification

## Overview

The first step in the PDE-SHARP framework is identifying what type of PDE we're solving. Different PDE types require fundamentally different numerical approaches.

## Supported PDE Types

### Poisson Equation (poisson_2d)
The 2D Poisson equation is:

$$\nabla^2 u = f$$

where $\nabla^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}$ is the Laplacian operator.

This is a steady-state (time-independent) equation commonly used for:
- Electrostatics
- Heat conduction at equilibrium
- Gravitational potential

### Convection-Diffusion Equation (convection_diffusion_2d)
The 2D convection-diffusion equation is:

$$\frac{\partial u}{\partial t} + \mathbf{v} \cdot \nabla u = \nu \nabla^2 u + f$$

where $\mathbf{v}$ is the velocity field and $\nu$ is the diffusion coefficient.

This is a time-dependent equation used for:
- Heat and mass transport
- Pollutant dispersion
- Fluid dynamics

## Implementation Note

The PDE type is specified in the input dictionary with a 'type' key. Valid values are 'poisson_2d' and 'convection_diffusion_2d'.

