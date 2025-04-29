import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

from bicyleXYModel import BicycleXYModel

# Instantiate model
Ts = 0.2  # sampling time
model = BicycleXYModel(sampling_time=Ts)

# MPC setup
N = 15  # Prediction horizon
nx = model.model_config.nx
nu = model.model_config.nu

# Define optimization variables
opti = ca.Opti()
X = opti.variable(nx, N+1)  # state trajectory
U = opti.variable(nu, N)    # control trajectory

# Parameters
X0 = opti.parameter(nx)     # initial state
goal = opti.parameter(2)    # goal position (px, py)
obstacles = [(2.0, 2.0), (3.0, 3.0)]  # obstacle positions (px, py)

# Define objective and constraints
Q = np.diag([10.0, 10.0, 1.0])  # state cost
R = np.diag([1.0, 0.5])         # control cost

cost = 0
for k in range(N):
    # Stage cost: tracking + control effort
    pos_error = X[0:2, k] - goal
    cost += ca.mtimes([pos_error.T, Q[0:2,0:2], pos_error])
    cost += ca.mtimes([U[:,k].T, R, U[:,k]])

    # Dynamics constraint
    x_next = model.I(x0=X[:,k], p=U[:,k])['xf']
    opti.subject_to(X[:,k+1] == x_next)

    # Obstacle avoidance
    for obs in obstacles:
        dist = ca.sqrt((X[0,k] - obs[0])**2 + (X[1,k] - obs[1])**2)
        opti.subject_to(dist >= model.model_config.safety_radius)

# Terminal cost
pos_error_terminal = X[0:2, N] - goal
cost += ca.mtimes([pos_error_terminal.T, Q[0:2,0:2], pos_error_terminal])

# Initial condition constraint
opti.subject_to(X[:,0] == X0)

# Input constraints (optional)
steer_limit = np.deg2rad(30)
vel_limit = 2.0
opti.subject_to(opti.bounded(-steer_limit, U[0,:], steer_limit))
opti.subject_to(opti.bounded(0.0, U[1,:], vel_limit))

# Solver settings
opti.minimize(cost)
opti.solver('ipopt')

# Simulation loop
x_current = np.array([0.0, 0.0, 0.0])  # start position
goal_pos = np.array([4.0, 4.0])
x_traj = [x_current]
u_traj = []

sim_time = 30
for t in range(sim_time):
    opti.set_value(X0, x_current)
    opti.set_value(goal, goal_pos)

    sol = opti.solve()

    u_mpc = sol.value(U[:,0])
    u_traj.append(u_mpc)

    # simulate
    x_next = model.I(x0=x_current, p=u_mpc)['xf'].full().flatten()
    x_current = x_next
    x_traj.append(x_current)

    # Check if close to goal
    if np.linalg.norm(x_current[0:2] - goal_pos) < 0.2:
        break

x_traj = np.array(x_traj).T
u_traj = np.array(u_traj).T

# Visualization
def animate_simulation(x_trajectory, u_trajectory, model, num_agents=1, additional_lines_or_scatters=None, save_path=None):
    fig, ax = plt.subplots()
    wheel_long_axis = 0.4
    wheel_short_axis = 0.1
    nx = model.model_config.nx
    sim_length = u_trajectory.shape[1]

    def update(i):
        ax.clear()
        ax.set_aspect('equal')
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 5)
        ax.set_xlabel('px(m)', fontsize=14)
        ax.set_ylabel('py(m)', fontsize=14)
        ax.set_title(f'Bicycle Simulation: Step {i+1}')

        for i_agent in range(num_agents):
            front_x = x_trajectory[i_agent*nx, i] + model.model_config.lf * np.cos(x_trajectory[i_agent*nx+2, i])
            front_y = x_trajectory[i_agent*nx+1, i] + model.model_config.lf * np.sin(x_trajectory[i_agent*nx+2, i])
            rear_x = x_trajectory[i_agent*nx, i] - model.model_config.lr * np.cos(x_trajectory[i_agent*nx+2, i])
            rear_y = x_trajectory[i_agent*nx+1, i] - model.model_config.lr * np.sin(x_trajectory[i_agent*nx+2, i])

            ax.plot(x_trajectory[i_agent*nx, :i+1], x_trajectory[i_agent*nx+1, :i+1], color='tab:gray', linewidth=2)
            ax.scatter(x_trajectory[i_agent*nx, i], x_trajectory[i_agent*nx+1, i], color='tab:gray', s=50)
            ax.plot([front_x, rear_x], [front_y, rear_y], color='tab:blue', linewidth=3)
            if i < sim_length:
                wheel_f = patches.Ellipse((front_x, front_y), wheel_long_axis, wheel_short_axis, angle=np.degrees(x_trajectory[i_agent*nx+2, i] + u_trajectory[0, i]), color='tab:green')
                wheel_r = patches.Ellipse((rear_x, rear_y), wheel_long_axis, wheel_short_axis, angle=np.degrees(x_trajectory[i_agent*nx+2, i]), color='tab:green')
                ax.add_patch(wheel_f)
                ax.add_patch(wheel_r)
            safety_circle = patches.Circle((x_trajectory[i_agent*nx, i], x_trajectory[i_agent*nx+1, i]), model.model_config.safety_radius, color='tab:orange', alpha=0.3)
            ax.add_patch(safety_circle)

        if additional_lines_or_scatters is not None:
            for key, value in additional_lines_or_scatters.items():
                if value['type'] == 'scatter':
                    ax.scatter(value['data'][0], value['data'][1], color=value['color'], s=value['s'], marker=value['marker'])
                elif value['type'] == 'line':
                    ax.plot(value['data'][0], value['data'][1], color=value['color'], linewidth=2)

    ani = animation.FuncAnimation(fig, update, frames=sim_length+1, interval=300)

    if save_path is not None:
        ani.save(save_path, writer='pillow', fps=3)
    else:
        plt.show()

obstacle_lines = {}
for i, obs in enumerate(obstacles):
    obstacle_lines[f'Obstacle {i}'] = {
        'type': 'scatter',
        'data': ([obs[0]], [obs[1]]),
        'color': 'red',
        's': 100,
        'marker': 'X'
    }

animate_simulation(x_traj, u_traj, model, additional_lines_or_scatters=obstacle_lines, save_path='bicycle_simulation.gif')

# Plot the trajectory
plt.figure()
plt.plot(x_traj[0,:], x_traj[1,:], '-o', label='Trajectory')
plt.scatter(goal_pos[0], goal_pos[1], marker='*', color='green', s=200, label='Goal')
for obs in obstacles:
    circle = plt.Circle(obs, model.model_config.safety_radius, color='r', alpha=0.5)
    plt.gca().add_patch(circle)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('MPC Trajectory with Obstacle Avoidance')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
