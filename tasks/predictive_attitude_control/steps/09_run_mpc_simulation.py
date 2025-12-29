"""
Stage 3: Optimal Control Synthesis - Main orchestrator: Run the full MPC attitude control simulation.
This function ties together all previous steps: magnetic field modeling,
linearization, QP formulation, SL-MPC updates, and state propagation.
"""

import numpy as np
import importlib
import sys
import os

_curr_dir = os.path.dirname(__file__)
if _curr_dir not in sys.path:
    sys.path.append(_curr_dir)

_mod01 = importlib.import_module('01_environmental_models')
_mod03 = importlib.import_module('03_linearization_discretization')
_mod04 = importlib.import_module('04_mpc_solver')
_mod05 = importlib.import_module('05_sl_mpc_update')
_mod06 = importlib.import_module('06_propagate_step')
_mod07 = importlib.import_module('07_evaluate_metrics')
_mod08 = importlib.import_module('08_physics_verification')

_gold_magnetic_field = _mod01._gold_magnetic_field
_gold_linearize_dynamics = _mod03._gold_linearize_dynamics
_gold_discretize_ltv = _mod03._gold_discretize_ltv
_gold_formulate_qp = _mod04._gold_formulate_qp
_gold_solve_mpc_step = _mod04._gold_solve_mpc_step
_gold_sl_mpc_update = _mod05._gold_sl_mpc_update
_gold_propagate_step = _mod06._gold_propagate_step
_gold_evaluate_metrics = _mod07._gold_evaluate_metrics
_gold_verify_physics = _mod08._gold_verify_physics

# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def run_mpc_simulation(inertia, h_w, orbit_params, mpc_params, constraints, initial_state, duration):
    """
    Run a closed-loop MPC simulation for a dual-spin satellite.
    
    This function ties together the environmental models, dynamics, 
    linearization, and QP optimization to simulate the satellite's 
    attitude control performance over time.
    
    Parameters
    ----------
    inertia : np.ndarray
        3x3 inertia matrix (kg·m²).
    h_w : float
        Momentum wheel angular momentum (N·m·s), assumed along body z-axis.
    orbit_params : dict
        - 'altitude': Orbit altitude (m)
        - 'inclination': Orbit inclination (deg)
        - 'omega_orbit': Orbital angular velocity (rad/s)
    mpc_params : dict
        - 'horizon': Prediction horizon N (integer steps)
        - 'dt': Control/prediction time step (s)
        - 'Q': State weighting matrix (6x6)
        - 'R': Control weighting matrix (3x3)
    constraints : dict
        - 'm_max': Maximum magnetic dipole moment per axis (A·m²)
        - 'max_cone_angle': Maximum allowable pointing half-cone error (deg)
        - 'min_spin_rate': Minimum allowable spin rate around boresight (rad/s)
        - 'target_q': Target inertial quaternion [q0, q1, q2, q3]
    initial_state : np.ndarray
        Initial state [q0, q1, q2, q3, wx, wy, wz] (7 elements).
    duration : float
        Total simulation duration (s).
        
    Returns
    -------
    metrics : dict
        Performance metrics including 'max_error', 'avg_error', 
        'power_index', 'settling_time', 'is_stable', 'momentum_error', 
        and 'orthogonality_error'.
    """
    return result  # <- Return hint for the model

# =============================================================================
# GOLD SOLUTION
# =============================================================================
def _gold_run_mpc_simulation(inertia, h_w, orbit_params, mpc_params, constraints, initial_state, duration):
    dt_mpc = mpc_params['dt']
    N = mpc_params['horizon']
    
    # Use a smaller internal step for better physics verification
    dt_sim = min(dt_mpc, 1.0) 
    steps_per_mpc = int(dt_mpc / dt_sim)
    total_sim_steps = int(duration / dt_sim)
    
    x_history = np.zeros((total_sim_steps + 1, 7))
    u_history = np.zeros((total_sim_steps, 3))
    B_body_history = np.zeros((total_sim_steps, 3))
    tau_eci_history = np.zeros((total_sim_steps, 3))
    
    x_history[0] = initial_state
    current_x = initial_state.copy()
    
    # Extract constraints
    target_q = constraints.get('target_q', np.array([1.0, 0.0, 0.0, 0.0]))
    max_cone_angle = constraints.get('max_cone_angle', 20.0)
    min_spin_rate = constraints.get('min_spin_rate', 0.0)
    m_max = constraints.get('m_max', 0.5)
    
    # Initial guess for control sequence (zeros)
    u_seq = np.zeros(3 * N)
    
    mu_e = 3.986004418e14
    R_e = 6371200.0
    r_mag = R_e + orbit_params['altitude']
    inc = np.radians(orbit_params['inclination'])
    
    def _compute_external_torque(x, u, t):
        # Calculate rotation matrix
        q = x[0:4]
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-10:
            q = q / q_norm
        q0, q1, q2, q3 = q
        C = np.array([
            [1 - 2*(q2**2 + q3**2), 2*(q1*q2 + q0*q3), 2*(q1*q3 - q0*q2)],
            [2*(q1*q2 - q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 + q0*q1)],
            [2*(q1*q3 + q0*q2), 2*(q2*q3 - q0*q1), 1 - 2*(q1**2 + q2**2)]
        ])
        
        # Environmental vectors at time t
        B_eci = _gold_magnetic_field(t, orbit_params)
        theta = orbit_params['omega_orbit'] * t
        r_eci = r_mag * np.array([np.cos(theta), np.sin(theta)*np.cos(inc), np.sin(theta)*np.sin(inc)])
        
        B_body = C @ B_eci
        tau_mag_body = np.cross(u, B_body)
        
        r_body = C @ r_eci
        r_norm = np.linalg.norm(r_body)
        tau_gg_body = (3.0 * mu_e / r_norm**5) * np.cross(r_body, np.dot(inertia, r_body))
        
        tau_total_body = tau_mag_body + tau_gg_body
        tau_eci = C.T @ tau_total_body
        return tau_eci, B_body

    for k in range(total_sim_steps):
        t_k = k * dt_sim
        
        # 1. Update MPC every steps_per_mpc
        if k % steps_per_mpc == 0:
            # Successive Linearization
            x_ref_traj, B_traj_body, r_traj_body = _gold_sl_mpc_update(
                current_x, u_seq, dt_mpc, N, inertia, h_w, orbit_params
            )
            
            # Linearize and Discretize
            Ak_list = []
            Bk_list = []
            for i in range(N):
                A_cont, B_cont = _gold_linearize_dynamics(
                    x_ref_traj[i], u_seq[3*i:3*i+3], inertia, h_w, B_traj_body[i], r_traj_body[i]
                )
                Ak, Bk = _gold_discretize_ltv(A_cont, B_cont, dt_mpc)
                Ak_list.append(Ak)
                Bk_list.append(Bk)
            
            # Formulate and Solve QP
            x0_error = np.zeros(6)
            H, g, A_cons, b_cons = _gold_formulate_qp(
                Ak_list, Bk_list, mpc_params['Q'], mpc_params['R'], x0_error, N, 
                x_ref_traj, target_q, max_cone_angle, min_spin_rate
            )
            u_seq = _gold_solve_mpc_step(H, g, m_max, A_cons, b_cons)
            
        u_k = u_seq[0:3]
        u_history[k] = u_k
        
        # Calculate external torque at start of step
        tau_start, B_body_start = _compute_external_torque(current_x, u_k, t_k)
        B_body_history[k] = B_body_start
        
        # 2. Propagate dynamics
        current_x = _gold_propagate_step(current_x, u_k, dt_sim, inertia, h_w, None, None, orbit_params, t_k)
        x_history[k+1] = current_x
        
        # Calculate external torque at end of step with SAME control u_k
        tau_end, _ = _compute_external_torque(current_x, u_k, t_k + dt_sim)
        
        # Store average torque for impulse-momentum check
        tau_eci_history[k] = 0.5 * (tau_start + tau_end)
        
    metrics = _gold_evaluate_metrics(x_history, u_history, dt_sim, target_q)
    
    # Add physics verification metrics
    phys_metrics = _gold_verify_physics(x_history, u_history, B_body_history, tau_eci_history, dt_sim, inertia, h_w)
    metrics.update(phys_metrics)
    
    return metrics

# =============================================================================
# TEST CASES
# =============================================================================
def test_cases():
    return [
        {
            "setup": """
inertia = np.diag([0.03, 0.03, 0.015])
h_w = 0.05
orbit_params = {'altitude': 500000, 'inclination': 45.0, 'omega_orbit': 0.0011}
mpc_params = {
    'horizon': 10,
    'dt': 2.0,
    'Q': np.diag([500, 500, 500, 10, 10, 10]),
    'R': np.eye(3) * 1.0
}
constraints = {
    'm_max': 0.2,
    'max_cone_angle': 20.0,
    'min_spin_rate': 0.05,
    'target_q': np.array([1.0, 0, 0, 0])
}
initial_state = np.concatenate([np.array([0.9914, 0.1305, 0, 0]), np.array([0, 0, 0.1])])
duration = 20.0
""",
            "call": "run_mpc_simulation(inertia, h_w, orbit_params, mpc_params, constraints, initial_state, duration)",
            "gold_call": "_gold_run_mpc_simulation(inertia, h_w, orbit_params, mpc_params, constraints, initial_state, duration)"
        },
        {
            "setup": """
inertia = np.diag([0.03, 0.03, 0.015])
h_w = 0.005 # Very low momentum
orbit_params = {'altitude': 500000, 'inclination': 0.0, 'omega_orbit': 0.0011}
mpc_params = {
    'horizon': 5,
    'dt': 5.0,
    'Q': np.eye(6) * 100,
    'R': np.eye(3) * 0.1
}
constraints = {
    'm_max': 0.5,
    'max_cone_angle': 30.0,
    'min_spin_rate': 0.0,
    'target_q': np.array([1.0, 0, 0, 0])
}
initial_state = np.concatenate([np.array([0.9659, 0.2588, 0, 0]), np.zeros(3)]) # 30 deg
duration = 25.0
""",
            "call": "run_mpc_simulation(inertia, h_w, orbit_params, mpc_params, constraints, initial_state, duration)",
            "gold_call": "_gold_run_mpc_simulation(inertia, h_w, orbit_params, mpc_params, constraints, initial_state, duration)"
        },
        {
            "setup": """
inertia = np.diag([0.1, 0.1, 0.1])
h_w = 0.1
orbit_params = {'altitude': 600000, 'inclination': 90.0, 'omega_orbit': 0.001}
mpc_params = {
    'horizon': 10,
    'dt': 2.0,
    'Q': np.eye(6) * 1000,
    'R': np.eye(3) * 1.0
}
constraints = {
    'm_max': 1.0,
    'max_cone_angle': 95.0,
    'min_spin_rate': 0.1,
    'target_q': np.array([1.0, 0, 0, 0])
}
# 90 deg rotation about X
angle = np.radians(90)
initial_q = np.array([np.cos(angle/2), np.sin(angle/2), 0, 0])
initial_state = np.concatenate([initial_q, np.array([0, 0, 0.1])])
duration = 10.0
""",
            "call": "run_mpc_simulation(inertia, h_w, orbit_params, mpc_params, constraints, initial_state, duration)",
            "gold_call": "_gold_run_mpc_simulation(inertia, h_w, orbit_params, mpc_params, constraints, initial_state, duration)"
        }
    ]

