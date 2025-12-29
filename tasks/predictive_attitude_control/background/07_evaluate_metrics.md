# Background: Performance Evaluation

To assess the effectiveness of the Model Predictive Control system, we evaluate several key performance metrics based on the simulation history.

## Pointing Error

The primary objective is to point the satellite's boresight (assumed to be the body z-axis, $\mathbf{z}_b$) at an inertial target $\mathbf{t}_{eci}$. The pointing error $\theta$ is the angle between the actual boresight and the target vector:

$$
\cos(\theta) = \mathbf{z}_b \cdot \mathbf{t}_b
$$

We calculate:
- **Max Error**: The peak deviation during the simulation.
- **RMS Error**: The root-mean-square error, representing average precision.

## Power Consumption (Power Index)

Magnetic actuation consumes electrical power proportional to the square of the dipole moment $m$. We define a performance index $P$:

$$
P = \int_0^T \| \mathbf{m}(t) \|^2 dt
$$

Lower values of $P$ indicate a more efficient control strategy.

## Settling Time

The settling time is defined as the time required for the pointing error to enter and remain within a specific threshold (e.g., 1 degree) of the target. This measures the speed of the control system's response to large initial errors or disturbances.

