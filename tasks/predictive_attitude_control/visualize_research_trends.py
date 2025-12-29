"""
Visualize Research Trends: SL-MPC for Magnetically Actuated Dual-Spin Satellites.
Generates graphs similar to the Halverson & Caverly (2025) paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load necessary modules
base_dir = os.path.dirname(__file__)
steps_dir = os.path.join(base_dir, "tasks", "predictive_attitude_control", "steps")
mod01 = load_module("m01", os.path.join(steps_dir, "01_environmental_models.py"))
mod02 = load_module("m02", os.path.join(steps_dir, "02_dynamics_ode.py"))
mod03 = load_module("m03", os.path.join(steps_dir, "03_linearization_discretization.py"))
mod04 = load_module("m04", os.path.join(steps_dir, "04_mpc_solver.py"))
mod05 = load_module("m05", os.path.join(steps_dir, "05_sl_mpc_update.py"))
mod06 = load_module("m06", os.path.join(steps_dir, "06_propagate_step.py"))
mod07 = load_module("m07", os.path.join(steps_dir, "07_evaluate_metrics.py"))

# Setup dependencies for internal calls
mod06._gold_dynamics_ode = mod02._gold_dynamics_ode

def run_simulation_and_plot():
    # 1. Setup Simulation Parameters
    inertia = np.diag([0.03, 0.03, 0.015])
    h_w = 0.05
    orbit_params = {'altitude': 500000, 'inclination': 45.0, 'omega_orbit': 0.0011}
    mpc_params = {
        'horizon': 20, # Increased horizon
        'dt': 10.0,    # Total lookahead: 200s
        'Q': np.diag([1000, 1000, 1000, 50, 50, 50]), # More aggressive on attitude
        'R': np.eye(3) * 0.1 # Less penalty on control
    }
    constraints = {'m_max': 0.5, 'max_cone_angle': 20.0} # Increased m_max to 0.5 (typical for 3U)
    
    # Initial state: 15 deg pointing error
    initial_q = np.array([0.9914, 0.1305, 0, 0])
    initial_w = np.array([0.0, 0.0, 0.0])
    initial_state = np.concatenate([initial_q, initial_w])
    
    duration = 5000.0 
    dt = mpc_params['dt']
    steps = int(duration / dt)
        
    x_history = np.zeros((steps + 1, 7))
    u_history = np.zeros((steps, 3))
    errors = np.zeros(steps + 1)
    
    x_history[0] = initial_state
    target_q = np.array([1.0, 0.0, 0.0, 0.0])
    u_seq = np.zeros(3 * mpc_params['horizon'])
    
    current_x = initial_state.copy()
    
    print("Starting simulation (SL-MPC)...")
    for k in range(steps):
        t_k = k * dt
        
        # Calculate current pointing error for plotting
        q = current_x[0:4]
        # Rotation matrix ECI to Body
        q0, q1, q2, q3 = q / np.linalg.norm(q)
        C = np.array([
            [1 - 2*(q2**2 + q3**2), 2*(q1*q2 + q0*q3), 2*(q1*q3 - q0*q2)],
            [2*(q1*q2 - q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 + q0*q1)],
            [2*(q1*q3 + q0*q2), 2*(q2*q3 - q0*q1), 1 - 2*(q1**2 + q2**2)]
        ])
        z_eci = C[2, :] # Actual boresight
        target_z_eci = np.array([0, 0, 1]) # Target boresight for [1,0,0,0]
        dot_product = np.clip(np.dot(z_eci, target_z_eci), -1.0, 1.0)
        errors[k] = np.degrees(np.arccos(dot_product))
        
        # SL-MPC step (Manually corrected to use absolute time t_k)
        def _sl_update_fixed(x0, u_prev, dt, N, t_start):
            x_traj = np.zeros((N + 1, 7))
            x_traj[0] = x0
            B_traj = np.zeros((N, 3))
            r_traj = np.zeros((N, 3))
            curr = x0.copy()
            for i in range(N):
                t_abs = t_start + i * dt
                B_eci = mod01._gold_magnetic_field(t_abs, orbit_params)
                # Position calculation (ECI)
                theta = orbit_params['omega_orbit'] * t_abs
                R_e = 6371200.0
                r_orbit = R_e + orbit_params['altitude']
                inc = np.radians(orbit_params['inclination'])
                r_eci = r_orbit * np.array([np.cos(theta), np.sin(theta)*np.cos(inc), np.sin(theta)*np.sin(inc)])
                
                # Get Body B and r
                q = curr[0:4] / np.linalg.norm(curr[0:4])
                q0, q1, q2, q3 = q
                C = np.array([
                    [1 - 2*(q2**2 + q3**2), 2*(q1*q2 + q0*q3), 2*(q1*q3 - q0*q2)],
                    [2*(q1*q2 - q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 + q0*q1)],
                    [2*(q1*q3 + q0*q2), 2*(q2*q3 - q0*q1), 1 - 2*(q1**2 + q2**2)]
                ])
                B_traj[i] = C @ B_eci
                r_traj[i] = C @ r_eci
                
                # Propagate
                u_i = u_prev[3*i : 3*i+3]
                sol = mod06.solve_ivp(mod06._gold_dynamics_ode, [0, dt], curr, args=(u_i, inertia, h_w, B_eci, r_eci))
                curr = sol.y[:, -1]
                x_traj[i+1] = curr
            return x_traj, B_traj, r_traj

        x_ref_traj, B_traj_body, r_traj_body = _sl_update_fixed(
            current_x, u_seq, dt, mpc_params['horizon'], t_k
        )
        
        Ak_list, Bk_list = [], []
        for i in range(mpc_params['horizon']):
            A_cont, B_cont = mod03._gold_linearize_dynamics(
                x_ref_traj[i], u_seq[3*i:3*i+3], inertia, h_w, B_traj_body[i], r_traj_body[i]
            )
            Ak, Bk = mod03._gold_discretize_ltv(A_cont, B_cont, dt)
            Ak_list.append(Ak)
            Bk_list.append(Bk)
            
        x0_error = np.zeros(6) # Error state starts at zero relative to x_ref_traj[0]
        H, g, A_cons, b_cons = mod04._gold_formulate_qp(
            Ak_list, Bk_list, mpc_params['Q'], mpc_params['R'], x0_error, 
            mpc_params['horizon'], x_ref_traj, target_q, constraints['max_cone_angle']
        )
        
        u_seq = mod04._gold_solve_mpc_step(H, g, constraints['m_max'], A_cons, b_cons)
        u_k = u_seq[0:3]
        u_history[k] = u_k
        
        B_eci = mod01._gold_magnetic_field(t_k, orbit_params)
        R_e = 6371200.0
        r_orbit = R_e + orbit_params['altitude']
        inc = np.radians(orbit_params['inclination'])
        theta = orbit_params['omega_orbit'] * t_k
        r_eci = r_orbit * np.array([np.cos(theta), np.sin(theta)*np.cos(inc), np.sin(theta)*np.sin(inc)])
        
        current_x = mod06._gold_propagate_step(current_x, u_k, dt, inertia, h_w, B_eci, r_eci)
        x_history[k+1] = current_x
        
        if k % 50 == 0:
            print(f"Time: {t_k:.1f}s, Error: {errors[k]:.2f} deg")

    # Final error
    q = x_history[-1, 0:4]
    q0, q1, q2, q3 = q / np.linalg.norm(q)
    C = np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 + q0*q3), 2*(q1*q3 - q0*q2)],
        [2*(q1*q2 - q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 + q0*q1)],
        [2*(q1*q3 + q0*q2), 2*(q2*q3 - q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])
    z_eci = C[2, :]
    errors[-1] = np.degrees(np.arccos(np.clip(np.dot(z_eci, [0,0,1]), -1.0, 1.0)))

    # 2. Plotting
    time = np.linspace(0, duration, steps + 1)
    
    plt.figure(figsize=(12, 10))
    
    # Pointing Error
    plt.subplot(3, 1, 1)
    plt.plot(time, errors, 'b-', linewidth=1.5, label='Pointing Error')
    plt.axhline(y=constraints['max_cone_angle'], color='r', linestyle='--', label='Constraint')
    plt.ylabel('Error [deg]')
    plt.title('Satellite Attitude Control Performance (Halverson & Caverly 2025)')
    plt.grid(True)
    plt.legend()
    
    # Angular Velocity
    plt.subplot(3, 1, 2)
    plt.plot(time, x_history[:, 4], label='$\omega_x$')
    plt.plot(time, x_history[:, 5], label='$\omega_y$')
    plt.plot(time, x_history[:, 6], label='$\omega_z$')
    plt.ylabel('Angular Velocity [rad/s]')
    plt.grid(True)
    plt.legend()
    
    # Control Inputs
    plt.subplot(3, 1, 3)
    u_time = time[:-1]
    plt.step(u_time, u_history[:, 0], label='$m_x$')
    plt.step(u_time, u_history[:, 1], label='$m_y$')
    plt.step(u_time, u_history[:, 2], label='$m_z$')
    plt.axhline(y=constraints['m_max'], color='r', linestyle=':')
    plt.axhline(y=-constraints['m_max'], color='r', linestyle=':')
    plt.ylabel('Dipole Moment [Am$^2$]')
    plt.xlabel('Time [s]')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plot_path = "research_trends.png"
    plt.savefig(plot_path)
    print(f"Simulation complete. Graphs saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    run_simulation_and_plot()

