import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

# ------------------ Model Configuration ------------------ #
class ModelConfig:
    def __init__(self):
        self.nx = 3  # State vector: [x, y, theta] (position and orientation)
        self.nu = 2  # Control vector: [delta, v] (steering angle and velocity)
        self.lf = 0.5  # Distance from rear axle to center of mass
        self.lr = 0.5  # Distance from front axle to center of mass
        self.safety_radius = 0.5  # Safety radius for collision avoidance
        self.dt = 0.2  # Sampling time (time step)

# bicycle's dynamics using CasADi
class BicycleXYModel:
    def __init__(self, sampling_time=0.2):
        self.model_config = ModelConfig()
        self.dt = sampling_time
        self._create_model()

    def _create_model(self):
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        delta = ca.SX.sym('delta')
        v = ca.SX.sym('v')


        #define variables for the state (x, y, theta) and control inputs (delta, v).
        state = ca.vertcat(x, y, theta)
        control = ca.vertcat(delta, v)

        beta = ca.atan((self.model_config.lr / (self.model_config.lr + self.model_config.lf)) * ca.tan(delta))
        dx = v * ca.cos(theta + beta)
        dy = v * ca.sin(theta + beta)
        dtheta = (v / self.model_config.lr) * ca.sin(beta)

        # Euler integration method is used to get the next state
        rhs = ca.vertcat(dx, dy, dtheta)
        x_next = state + self.dt * rhs

        #computes the next state from the current state and control input.
        self.I = ca.Function('I', [state, control], [x_next])

# ------------------ MPC Setup ------------------ #
#MPC controller
#minimize the cost function that considers the distance to the goal, velocity, and collision avoidance.
def create_mpc(model, static_obstacles, N=15):
    opti = ca.Opti()
    nx, nu = model.model_config.nx, model.model_config.nu

    X = opti.variable(nx, N + 1)
    U = opti.variable(nu, N)
    X0 = opti.parameter(nx)
    goal = opti.parameter(2)
    other_agent = opti.parameter(nx, N + 1)

    # Weighting matrices for the position error and control inputs, respectively. 
    # They determine the importance of minimizing position error and controlling the inputs.
    Q = np.diag([10.0, 10.0, 1.0])
    R = np.diag([1.0, 0.5])
    safety_radius = model.model_config.safety_radius

    cost = 0
    for k in range(N):
        #The error between the current position and the goal. The cost increases with the distance to the goal.
        pos_error = X[0:2, k] - goal
        cost += ca.mtimes([pos_error.T, Q[0:2, 0:2], pos_error])
        cost += ca.mtimes([U[:, k].T, R, U[:, k]])

        x_next = model.I(X[:, k], U[:, k])
        opti.subject_to(X[:, k + 1] == x_next)

        # Collision avoidance between agents
        dist_to_other = ca.norm_2(X[0:2, k] - other_agent[0:2, k])
        cost += 500 * ca.fmax(0, 2 * safety_radius - dist_to_other) ** 2

        # Static obstacle avoidance with reduced penalty
        for obs in static_obstacles:
            dist_to_obs = ca.norm_2(X[0:2, k] - obs)
            cost += 1000 * ca.fmax(0, safety_radius - dist_to_obs) ** 2

    #adds a penalty for approaching static obstacles.
    pos_error_terminal = X[0:2, N] - goal
    cost += ca.mtimes([pos_error_terminal.T, Q[0:2, 0:2], pos_error_terminal])
    opti.subject_to(X[:, 0] == X0)

    #the steering angle limit and the velocity limit to give the bicycles more flexibility in their movement.
    # SOFT control constraints
    steer_limit = np.deg2rad(35)  # Increased steering limit
    vel_limit = 3.0  # Increased velocity limit
    opti.subject_to(opti.bounded(-steer_limit, U[0, :], steer_limit))
    opti.subject_to(opti.bounded(0.0, U[1, :], vel_limit))

    opti.minimize(cost)
    
    # Solver settings: Increase iterations and tolerance
    opti.solver('ipopt', {
        'print_time': False, 
        'ipopt.print_level': 0,
        'ipopt.tol': 1e-8,  # Decreased tolerance for better convergence
        'ipopt.max_iter': 500,  # More iterations to improve solver behavior
        'ipopt.linear_solver': 'mumps',  # Robust solver
        'ipopt.acceptable_tol': 1e-6,  # Relax acceptable tolerance for smoothness
    })
    
    return opti, X, U, X0, goal, other_agent

# ------------------ Setup & Simulation ------------------ #
static_obstacles = [np.array([2.0, 2.0]), np.array([3.5, 3.0])]
model = BicycleXYModel()
N = 10 #The prediction horizon, i.e., the number of time steps into the future the controller will optimize for.
sim_steps = 100  # Increased for more steps
goal_path = [np.array([4.0, 4.0]), np.array([0.5, 4.5]), np.array([4.5, 0.5])]
goal_switch_interval = 5  # Goal changes every 5 steps

#creates the MPC problem using CasADi's Opti solver for each agent
opti1, X1, U1, X0_1, goal_1, other1 = create_mpc(model, static_obstacles, N)
opti2, X2, U2, X0_2, goal_2, other2 = create_mpc(model, static_obstacles, N)

x1 = np.array([0.0, 0.0, 0.0])
x2 = np.array([0.5, 0.5, 0.0])
traj1, traj2 = [x1], [x2]

# ------------------ Simulation Loop ------------------ #
for t in range(sim_steps):
    current_goal = goal_path[(t // goal_switch_interval) % len(goal_path)]

    # Agent 1
    opti1.set_value(X0_1, x1)
    opti1.set_value(goal_1, current_goal)
    opti1.set_value(other1, np.tile(x2, (N + 1, 1)).T)
    sol1 = opti1.solve()
    u1 = sol1.value(U1[:, 0])
    x1 = model.I(x1, u1).full().flatten()
    traj1.append(x1)

    # Agent 2
    opti2.set_value(X0_2, x2)
    opti2.set_value(goal_2, current_goal)
    opti2.set_value(other2, np.tile(x1, (N + 1, 1)).T)
    sol2 = opti2.solve()
    u2 = sol2.value(U2[:, 0])
    x2 = model.I(x2, u2).full().flatten()
    traj2.append(x2)

    # Stop simulation if any agent reaches the goal
    if np.linalg.norm(x1[0:2] - current_goal) < model.model_config.safety_radius:
        print("Agent 1 reached the goal!")
        break
    if np.linalg.norm(x2[0:2] - current_goal) < model.model_config.safety_radius:
        print("Agent 2 reached the goal!")
        break

traj1 = np.array(traj1).T
traj2 = np.array(traj2).T

# ------------------ Animation ------------------ #
def animate_race(traj1, traj2, model, static_obstacles, goal_path):
    fig, ax = plt.subplots()
    steps = traj1.shape[1]

    def update(i):
        ax.clear()
        ax.set_aspect('equal')
        ax.set_xlim(-1, 6)
        ax.set_ylim(-1, 6)
        ax.set_title(f'Step {i + 1}')
        ax.grid(True)

        ax.plot(traj1[0, :i + 1], traj1[1, :i + 1], 'b-', label='Agent 1')
        ax.plot(traj2[0, :i + 1], traj2[1, :i + 1], 'r-', label='Agent 2')
        ax.scatter(traj1[0, i], traj1[1, i], color='blue', s=50)
        ax.scatter(traj2[0, i], traj2[1, i], color='red', s=50)

        #If the bicycle is too close to the other agent a penalty is added.
        for obs in static_obstacles:
            ax.add_patch(patches.Circle(obs, model.model_config.safety_radius, color='gray', alpha=0.5))

        current_goal = goal_path[(i // goal_switch_interval) % len(goal_path)]
        ax.scatter(current_goal[0], current_goal[1], marker='*', s=200, color='green', label='Goal')
        ax.legend()

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=300)
    plt.show()

animate_race(traj1, traj2, model, static_obstacles, goal_path)

