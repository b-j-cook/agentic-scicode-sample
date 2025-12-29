"""
Stage 3: Optimal Control Synthesis - Evaluate the performance of the attitude control system over a full simulation.
Calculate metrics like max pointing error, RMS error, and power consumption.
"""

import numpy as np


# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def evaluate_metrics(x_history, u_history, dt, target_q):
    """
    Calculate performance metrics from simulation history.
    
    Parameters
    ----------
    x_history : np.ndarray
        History of states (T, 7).
    u_history : np.ndarray
        History of control inputs (T-1, 3).
    dt : float
        Simulation time step (s).
    target_q : np.ndarray
        Target inertial quaternion.
    
    Returns
    -------
    metrics : dict
        Dictionary with 'max_error', 'avg_error', 'power_index', 'settling_time'.
    """
    return result  # <- Return hint for the model

# =============================================================================
# GOLD SOLUTION
# =============================================================================
def _gold_evaluate_metrics(x_history, u_history, dt, target_q):
    T = x_history.shape[0]
    errors = np.zeros(T)
    
    # Inverse rotation matrix for target
    target_q0, target_q1, target_q2, target_q3 = target_q
    C_target = np.array([
        [1 - 2*(target_q2**2 + target_q3**2), 2*(target_q1*target_q2 + target_q0*target_q3), 2*(target_q1*target_q3 - target_q0*target_q2)],
        [2*(target_q1*target_q2 - target_q0*target_q3), 1 - 2*(target_q1**2 + target_q3**2), 2*(target_q2*target_q3 + target_q0*target_q1)],
        [2*(target_q1*target_q3 + target_q0*target_q2), 2*(target_q2*target_q3 - target_q0*target_q1), 1 - 2*(target_q1**2 + target_q2**2)]
    ])
    
    # Target boresight in ECI frame (assumed body z-axis at target orientation)
    # Since C_target is ECI to Body, its rows are the body axes in ECI.
    # The 3rd row is the body z-axis in ECI.
    target_z_eci = C_target[2, :]
    
    for i in range(T):
        q = x_history[i, 0:4]
        # Normalize quaternion
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-10: q = q / q_norm
        
        q0, q1, q2, q3 = q
        # Rotation matrix ECI to Body
        C = np.array([
            [1 - 2*(q2**2 + q3**2), 2*(q1*q2 + q0*q3), 2*(q1*q3 - q0*q2)],
            [2*(q1*q2 - q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 + q0*q1)],
            [2*(q1*q3 + q0*q2), 2*(q2*q3 - q0*q1), 1 - 2*(q1**2 + q2**2)]
        ])
        
        # Actual boresight (z-axis) in ECI frame
        z_eci = C[2, :]
        
        # Pointing error is the angle between actual and target boresight
        dot_product = np.clip(np.dot(z_eci, target_z_eci), -1.0, 1.0)
        errors[i] = np.degrees(np.arccos(dot_product))
    
    max_error = np.max(errors)
    avg_error = np.sqrt(np.mean(errors**2)) # RMS error
    
    # Power index: integral of sum(m_i^2)
    power_index = np.sum(u_history**2) * dt
    
    # Settling time: time to reach and stay within 1 deg
    settling_time = 0.0
    for i in range(T-1, -1, -1):
        if errors[i] > 1.0:
            settling_time = (i + 1) * dt
            break
            
    metrics = {
        'max_error': float(max_error),
        'avg_error': float(avg_error),
        'power_index': float(power_index),
        'settling_time': float(settling_time),
        'is_stable': bool(max_error < 10.0) # Arbitrary threshold for stability in this context
    }
    
    return metrics

# =============================================================================
# TEST CASES
# =============================================================================
def test_cases():
    return [
        {
            "setup": """
x_history = np.zeros((10, 7))
x_history[:, 0] = 1.0 # Identity
u_history = np.zeros((9, 3))
dt = 1.0
target_q = np.array([1.0, 0, 0, 0])
""",
            "call": "evaluate_metrics(x_history, u_history, dt, target_q)",
            "gold_call": "_gold_evaluate_metrics(x_history, u_history, dt, target_q)"
        },
        {
            "setup": """
# Error decreases from 2.0 deg to 0.5 deg
t = np.linspace(0, 10, 11)
errors = np.linspace(2.0, 0.5, 11)
x_history = np.zeros((11, 7))
for i in range(11):
    angle = np.radians(errors[i])
    x_history[i, 0] = np.cos(angle/2)
    x_history[i, 1] = np.sin(angle/2)
u_history = np.zeros((10, 3))
dt = 1.0
target_q = np.array([1.0, 0, 0, 0])
""",
            "call": "evaluate_metrics(x_history, u_history, dt, target_q)",
            "gold_call": "_gold_evaluate_metrics(x_history, u_history, dt, target_q)"
        },
        {
            "setup": """
x_history = np.zeros((6, 7))
x_history[:, 0] = 1.0
u_history = np.array([
    [0.1, 0, 0],
    [0.2, 0, 0],
    [0.3, 0, 0],
    [0.2, 0, 0],
    [0.1, 0, 0]
])
dt = 2.0
target_q = np.array([1.0, 0, 0, 0])
""",
            "call": "evaluate_metrics(x_history, u_history, dt, target_q)",
            "gold_call": "_gold_evaluate_metrics(x_history, u_history, dt, target_q)"
        }
    ]

