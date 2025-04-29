import casadi as ca
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy as np


def main():
    p_obj = ca.MX.sym('p_obj', 2)
    p_obs_1 = ca.MX.sym('p_obs_1', 2)
    p_obs_2 = ca.MX.sym('p_obs_2', 2)
    p_target = ca.MX.sym('p_target', 2)

    J = ca.sumsqr(p_obj - p_target)
    g = ca.vertcat(ca.sumsqr(p_obj - p_obs_1),
         ca.sumsqr(p_obj - p_obs_2))

    ocp = { 'x': p_obj,
            'p': ca.vertcat(p_obs_1, p_obs_2, p_target),
            'g': g,
            'f': J
        }
    opts = {'ipopt': {'print_level': 0, 'max_iter': 1000}, "print_time": False}
    solver = ca.nlpsol('solver', 'ipopt', ocp, opts)

    # Lower and Upper bound of constraints
    radius_obj = 0.3
    radius_obs_1 = 0.5
    radius_obs_2 = 0.8
    lbg = [(radius_obj+radius_obs_1)**2, (radius_obj+radius_obs_2)**2]
    ubg = [ca.inf, ca.inf]
    lb_x = -1.0
    ub_x = 2.0
    # Parameter values
    p_obs_1_val = np.array([1.0, 0.0])
    p_obs_2_val = np.array([0.0, 1.0])
    p_target_val = np.array([0.7, 0.5])
    solution = solver(
            x0=[1.0, 1.0],
            p=ca.vertcat(p_obs_1_val, p_obs_2_val, p_target_val),
            lbg=lbg,
            ubg=ubg,
            lbx=[lb_x+radius_obj, lb_x+radius_obj],
            ubx=[ub_x-radius_obj, ub_x-radius_obj],
        )
    p_obj_opt = solution['x'].full().flatten()
    print(f"The optimal position of the object is: {p_obj_opt}")
    print(f"The optimal cost is: {solution['f'].full().flatten()}")

    def plot_results():
        fontsize = 16
        params = {
            'text.latex.preamble': r"\usepackage{gensymb} \usepackage{amsmath} \usepackage{amsfonts} \usepackage{cmbright}",
            'axes.labelsize': fontsize,
            'axes.titlesize': fontsize,
            'legend.fontsize': fontsize,
            'xtick.labelsize': fontsize,
            'ytick.labelsize': fontsize,
            "mathtext.fontset": "stixsans",
            "axes.unicode_minus": False,
        }
        matplotlib.rcParams.update(params)

        fig, ax = plt.subplots()
        fig.suptitle("**Welcome to Workshop: Future PhD in control?!**\nMove the OBJECT to the TARGET as close as possible \nwhile not getting into obstacles.", fontsize=fontsize)
        ax.set_xlim(-1.1, 2.1)
        ax.set_ylim(-1.1, 2.1)
        ax.hlines([lb_x, ub_x], lb_x, ub_x, color='tab:brown', linestyle='--', linewidth=2)
        ax.vlines([lb_x, ub_x], lb_x, ub_x, color='tab:brown', linestyle='--', linewidth=2)
        ax.set_xlabel('px(m)', fontsize=fontsize)
        ax.set_ylabel('py(m)', fontsize=fontsize)
        ax.set_aspect('equal')
        ax.add_patch(patches.Circle((p_obs_1_val[0], p_obs_1_val[1]), radius_obs_1, color='tab:orange', alpha=0.5))
        ax.add_patch(patches.Circle((p_obs_2_val[0], p_obs_2_val[1]), radius_obs_2, color='tab:orange', alpha=0.5))
        ax.add_patch(patches.Circle((p_obj_opt[0], p_obj_opt[1]), radius_obj, color='tab:blue'))
        ax.scatter(p_obs_1_val[0], p_obs_1_val[1], color='tab:orange', s=50, label='Obstacle 1')
        ax.scatter(p_obs_2_val[0], p_obs_2_val[1], color='tab:orange', s=50, label='Obstacle 2')
        ax.scatter(p_obj_opt[0], p_obj_opt[1], color='black', s=40, label='OBJECT')
        ax.scatter(p_target_val[0], p_target_val[1], color='black', s=50, marker="x", label="TARGET")
        fig.subplots_adjust(right=0.6)
        ax.legend(loc='center', bbox_to_anchor=(1.4, 0.5))
        plt.show()
    plot_results()


if __name__ == "__main__":
    main()