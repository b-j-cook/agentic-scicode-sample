# Background: Floquet Monodromy Matrix

Floquet theory characterizes the stability of linear differential equations with $T$-periodic coefficients. The system $\ddot{x} + u(t)x = 0$ is stable if its solutions remain bounded as $t \to \infty$.

## The Monodromy Matrix

The **monodromy matrix** $\mathbf{M}$ is the state transition matrix over exactly one period $T$:
$$
\mathbf{z}(T) = \mathbf{M} \mathbf{z}(0)
$$
For the three-interval bang-bang profile, $\mathbf{M}$ is the product of three transition matrices:
$$
\mathbf{M} = \mathbf{\Phi}_{u^+}(t_s) \mathbf{\Phi}_{u^-}(T - 2t_s) \mathbf{\Phi}_{u^+}(t_s)
$$

## Trace Criterion for Stability

The eigenvalues $\mu_{1,2}$ of $\mathbf{M}$ (Floquet multipliers) determine stability. For this Hamiltonian system, $\det(\mathbf{M}) = \mu_1 \mu_2 = 1$. Stability requires $|\mu_i| \leq 1$, which is satisfied if and only if the trace of $\mathbf{M}$ obeys:
$$
|\text{Tr}(\mathbf{M})| \leq 2
$$
- **Stable region ($|\text{Tr}(\mathbf{M})| < 2$)**: All solutions are bounded and quasi-periodic.
- **Stability boundary ($|\text{Tr}(\mathbf{M})| = 2$)**: One multiplier is $\pm 1$; this corresponds to the existence of $T$-periodic or $2T$-periodic solutions.
- **Unstable region ($|\text{Tr}(\mathbf{M})| > 2$)**: Solutions diverge exponentially.

In the Schrödinger analogy, the stability boundary $|\text{Tr}(\mathbf{M})| = 2$ corresponds to the energy levels $E_n$ where bound states (stabilization modes) emerge. Specifically, the condition $\text{Tr}(\mathbf{M}) = 2$ corresponds to $T$-periodic modes (even $n$), while $\text{Tr}(\mathbf{M}) = -2$ corresponds to $2T$-periodic modes (odd $n$).