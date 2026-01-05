# Background: Schrödinger Analogy

A core finding of Lazarus & Trélat (2025) is the mathematical equivalence between classical dynamical stabilization and the stationary Schrödinger equation in one dimension.

## The Mathematical Mapping

The classical ODE $\ddot{x} + u(t)x = 0$ can be mapped to the Schrödinger equation:
$$
-\frac{1}{2} \frac{d^2 \psi}{d\tau^2} + V(\tau) \psi = E \psi
$$
where time $t$ maps to space $\tau$. By setting $V(\tau) = -u(\tau)/2$, the stability problem becomes an eigenvalue problem for $E$. In the large $T$ limit, the stabilization threshold for a given mode matches the bound state energy of a quantum particle in a finite square-well potential.

## Quantum Bound States

- **Potential Well**: The region with positive stiffness $u^+$ corresponds to a potential well (where $V$ is low), allowing for oscillatory "trapped" states.
- **Barrier**: The region with negative stiffness $u^-$ corresponds to a potential barrier (where $V$ is high), leading to exponential decay (tunneling analogy).
- **Stabilization Threshold**: The minimal average stiffness required for stability is predicted by the ground state energy $E_0$ of the Hamiltonian $\mathbf{H} = -\frac{1}{2}\nabla^2 + V(\tau)$.
