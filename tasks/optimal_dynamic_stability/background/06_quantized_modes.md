# Background: Quantized Stabilization Modes

Dynamical stabilization does not occur on a continuous spectrum; instead, stable trajectories emerge in discrete "quantized" modes, analogous to the energy levels of a quantum system.

## Modal Classification

Each mode $n = 0, 1, 2, \dots$ corresponds to a unique switching configuration $t_s^{(n)}$:
- **Ground State ($n=0$)**: The fundamental stabilization mode. The trajectory $x(t)$ has no zero-crossings within the interval $(0, T/2)$. This mode requires the absolute minimal average stiffness.
- **Excited States ($n > 0$)**: Higher-order stabilization modes where $x(t)$ oscillates $n$ times before the half-period.

## Admissible Control Space

For a fixed pair of values $\{u^+, u^-\}$, optimal stabilization is only possible at specific switching times $t_s$. These "quantized" solutions are the only trajectories that satisfy the periodicity and minimal stiffness conditions derived from the Schrödinger analogy.
