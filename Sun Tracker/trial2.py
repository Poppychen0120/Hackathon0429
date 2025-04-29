import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

# Simulation parameters
dt = 0.1  # Time step
time = 0.0

# Sun path (rises from -90° to +90°)
def sun_angle(t):
    return -90 + 180 * (t / 12.0)  # 0 to 12 hours

# Initial panel angle
panel_angle = 0.0

# CasADi optimization function
def optimize_panel(current_angle, sun_angle):
    opti = ca.Opti()
    
    # Control variable: angular velocity
    omega = opti.variable()
    
    # Dynamics
    next_angle = current_angle + omega * dt
    
    # Objective: minimize tracking error + control effort
    error = next_angle - sun_angle
    cost = error**2 + 0.01 * omega**2
    opti.minimize(cost)
    
    # Constraint: Limit rotation speed
    opti.subject_to(opti.bounded(-30, omega, 30))
    
    opti.solver('ipopt', {'ipopt.print_level':0, 'print_time':0})
    sol = opti.solve()
    return sol.value(next_angle)

# Matplotlib setup 
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Manual Solar Tracker")

sun_dot, = ax.plot([], [], 'yo', markersize=15, label="Sun")
panel_line, = ax.plot([], [], 'b-', lw=3, label="Panel")

def update_plot():
    global time, panel_angle
    # Update sun position
    current_sun_angle = np.deg2rad(sun_angle(time))
    sun_x = np.cos(current_sun_angle)
    sun_y = np.sin(current_sun_angle)
    
    # Update panel
    panel_x = np.cos(np.deg2rad(panel_angle))
    panel_y = np.sin(np.deg2rad(panel_angle))
    
    sun_dot.set_data([sun_x], [sun_y])
    panel_line.set_data([0, panel_x], [0, panel_y])
    
    fig.canvas.draw()
    plt.pause(0.01)

def on_key(event):
    global time, panel_angle
    if event.key == ' ':
        # Move time forward
        time_step = 0.5
        time += time_step
        if time > 12: time = 0
        
        # Optimize panel
        target_sun = sun_angle(time)
        panel_angle_new = optimize_panel(panel_angle, target_sun)
        panel_angle = panel_angle_new
        
        update_plot()

fig.canvas.mpl_connect('key_press_event', on_key)
update_plot()
plt.legend()
plt.show()
