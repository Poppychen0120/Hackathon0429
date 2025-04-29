import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

from bicyleXYModel import BicycleXYModel


def create_mpc(model, N, Q, R, goal):
    nx = model.model_config.nx
    nu = model.model_config.nu

    # Decision variables
    X = ca.MX.sym('X', nx, N+1)
    U = ca.MX.sym('U', nu, N)

    # Objective function
    cost = 0
    for k in range(N):
        dx = X[:, k] - goal
        cost += ca.mtimes([dx.T, Q, dx]) + ca.mtimes([U[:, k].T, R, U[:, k]])
    dx_terminal = X[:, N] - goal
    cost += ca.mtimes([dx_terminal.T, Q, dx_terminal])

    # Constraints
    constraints = []
    for k in range(N):
        x_next = model.I(x0=X[:, k], p=U[:, k])['xf']
        constraints.append(X[:, k+1] - x_next)

    # Flatten constraints
    constraints = ca.vertcat(*constraints)

    # Define optimization variables
    opt_variables = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

    # Define the optimization problem
    nlp = {'f': cost, 'x': opt_variables, 'g': constraints}
    solver = ca.nlpsol('solver', 'ipopt', nlp)

    return solver, X, U

def run_simulation():
    # Initialize model
    sampling_time = 0.2
    model = BicycleXYModel(sampling_time)

    # MPC parameters
    N = 10  # Prediction horizon
    Q = np.diag([10, 10, 1])  # State cost
    R = np.diag([1, 1])       # Control cost
    goal = np.array([4.0, 4.0, 0.0])  # Goal position

    # Initial state
    x0 = np.array([0.0, 0.0, 0.0])

    # Create MPC solver
    solver, X_sym, U_sym = create_mpc(model, N, Q, R, goal)

    # Simulation parameters
    max_steps = 100
    tolerance = 0.1

    # Storage for trajectories
    x_trajectory = [x0]
    u_trajectory = []

    x_current = x0.copy()

    for step in range(max_steps):
        # Initial guess
        x_init = np.tile(x_current.reshape(-1, 1), (1, N+1))
        u_init = np.zeros((model.model_config.nu, N))

        # Flatten initial guesses
        x_init_flat = x_init.flatten()
        u_init_flat = u_init.flatten()
        init_guess = np.concatenate([x_init_flat, u_init_flat])

        # Solve MPC
        sol = solver(x0=init_guess, lbg=0, ubg=0)

        # Extract control input
        sol_x = sol['x'].full().flatten()
        u_opt = sol_x[model.model_config.nx*(N+1):model.model_config.nx*(N+1)+model.model_config.nu]
        u_trajectory.append(u_opt)

        # Apply control input
        x_next = model.I(x0=x_current, p=u_opt)['xf'].full().flatten()
        x_trajectory.append(x_next)
        x_current = x_next

        # Check if goal is reached
        if np.linalg.norm(x_current[:2] - goal[:2]) < tolerance:
            print(f"Goal reached at step {step+1}")
            break

    # Convert trajectories to arrays
    x_trajectory = np.array(x_trajectory).T
    u_trajectory = np.array(u_trajectory).T

    # Animate the simulation
    model.animateSimulation(x_trajectory, u_trajectory, num_agents=1)

if __name__ == "__main__":
    run_simulation()
