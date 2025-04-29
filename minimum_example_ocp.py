import casadi as ca
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy as np
import os
import sys

@dataclass
class OCPConfig:
    nx: int = 4
    nu: int = 2
    n_hrzn: int = 50
    sampling_time = 0.05
    Q: np.ndarray = np.diag([1.0, 1.0, 0.01, 0.01])
    R: np.ndarray = np.diag([0.01, 0.01])


def main():
    ocp_config = OCPConfig()

    def define_ocp_problem():
        x = ca.MX.sym('x', ocp_config.nx, ocp_config.n_hrzn + 1)
        u = ca.MX.sym('u', ocp_config.nu, ocp_config.n_hrzn)
        x_init = ca.MX.sym('x_init', ocp_config.nx)
        p_obs_1 = ca.MX.sym('p_obs_1', 2)
        p_obs_2 = ca.MX.sym('p_obs_2', 2)
        x_ref = ca.MX.sym('x_ref', ocp_config.nx, ocp_config.n_hrzn + 1)

        # objective function
        J = 0.
        for i in range(ocp_config.n_hrzn):
            J += (x[:, i] - x_ref[:, i]).T @ ocp_config.Q @ (x[:, i] - x_ref[:, i])
            J += u[:, i].T @ ocp_config.R @ u[:, i]
        J += (x[:, ocp_config.n_hrzn] - x_ref[:, ocp_config.n_hrzn]).T @ ocp_config.Q @ (x[:, ocp_config.n_hrzn] - x_ref[:, ocp_config.n_hrzn])

        # constraints
        g = []
        for i in range(ocp_config.n_hrzn+1):
            g.append(ca.sumsqr(x[0:2, i] - p_obs_1))
            g.append(ca.sumsqr(x[0:2, i] - p_obs_2))
        g.append(x[:, 0] - x_init)
        for i in range(ocp_config.n_hrzn):
            g.append(x[0:2, i+1] - (x[0:2, i] + ocp_config.sampling_time * x[2:4, i] + 0.5 * ocp_config.sampling_time**2 * u[:, i]))
            g.append(x[2:4, i+1] - (x[2:4, i] + ocp_config.sampling_time * u[:, i]))
        g.append(x[2:4, ocp_config.n_hrzn])

        ocp = { 'x': ca.veccat(x, u),
                'p': ca.veccat(p_obs_1, p_obs_2, x_init, x_ref),
                'g': ca.veccat(*g),
                'f': J
            }
        opts = {'ipopt': {'print_level': 0, 'max_iter': 1000}, "print_time": False}
        solver = ca.nlpsol('solver', 'ipopt', ocp, opts)
        return solver
    solver = define_ocp_problem()

    # Parameter values
    radius_obj = 0.3
    radius_obs_1 = 0.35
    radius_obs_2 = 0.8
    lb_p = -1.0
    ub_p = 2.0
    p_obs_1_val = np.array([1.5, 0.0])
    p_obs_2_val = np.array([0.0, 1.0])
    p_target_val = np.array([1.3, 1.3])
    x_init_val = np.array([-0.5, -0.5, 0.0, 0.0])
    p_ref_val = (p_target_val[:, np.newaxis] - x_init_val[:2, np.newaxis]) @ np.linspace(0.0, 1.0, ocp_config.n_hrzn+1, endpoint=True)[np.newaxis, :] + x_init_val[:2, np.newaxis]
    v_ref_val = np.diff(p_ref_val, axis=1) / ocp_config.sampling_time
    v_ref_val = np.hstack((v_ref_val, np.zeros((2, 1))))
    x_ref_val = np.vstack((p_ref_val, v_ref_val))

    def solve_ocp():
        lbg = [(radius_obj+radius_obs_1)**2, (radius_obj+radius_obs_2)**2] * (ocp_config.n_hrzn+1) + [0.0] * ((ocp_config.n_hrzn+1) * ocp_config.nx + 2)
        ubg = [ca.inf, ca.inf] * (ocp_config.n_hrzn+1) + [0.0] * ((ocp_config.n_hrzn+1) * ocp_config.nx + 2)
        lbx = [lb_p, lb_p, -np.inf, -np.inf] * (ocp_config.n_hrzn+1) + [-np.inf, -np.inf] * ocp_config.n_hrzn
        ubx = [ub_p, ub_p, np.inf, np.inf] * (ocp_config.n_hrzn+1) + [np.inf, np.inf] * ocp_config.n_hrzn
        solution = solver(
                x0=ca.veccat(np.tile(x_init_val, (1, ocp_config.n_hrzn+1)), np.zeros((ocp_config.nu, ocp_config.n_hrzn))),
                p=ca.veccat(p_obs_1_val, p_obs_2_val, x_init_val, x_ref_val),
                lbg=lbg,
                ubg=ubg,
                lbx=lbx,
                ubx=ubx,
            )
        sol = solution['x'].full().flatten()
        print(f"The optimal cost is: {solution['f'].full().flatten()}")
        return sol
    sol = solve_ocp()
    x_sol = sol[:(ocp_config.n_hrzn+1) * ocp_config.nx].reshape((ocp_config.nx, ocp_config.n_hrzn+1), order='F')

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

        local_path = os.path.dirname(os.path.realpath(__file__))
        images_folder = os.path.join(local_path, "ocp_images")
        if not(os.path.exists(images_folder) and os.path.isdir(images_folder)):
            os.makedirs(images_folder)

        for i in range(ocp_config.n_hrzn + 1):
            fig = plt.figure(i)
            ax = fig.add_subplot(1, 1, 1)
            fig.suptitle(f"**Welcome to Workshop: Future PhD in control?!**\nMinium OCP, step={i}.", fontsize=fontsize)
            ax.set_xlim(-1.1, 2.1)
            ax.set_ylim(-1.1, 2.1)
            ax.hlines([lb_p, ub_p], lb_p, ub_p, color='tab:brown', linestyle='--', linewidth=2)
            ax.vlines([lb_p, ub_p], lb_p, ub_p, color='tab:brown', linestyle='--', linewidth=2)
            ax.set_xlabel('px(m)', fontsize=fontsize)
            ax.set_ylabel('py(m)', fontsize=fontsize)
            ax.set_aspect('equal')
            ax.add_patch(patches.Circle((p_obs_1_val[0], p_obs_1_val[1]), radius_obs_1, linewidth=0.0, color='tab:orange', alpha=0.5))
            ax.add_patch(patches.Circle((p_obs_2_val[0], p_obs_2_val[1]), radius_obs_2, linewidth=0.0, color='tab:orange', alpha=0.5))
            ax.add_patch(patches.Circle((x_sol[0, i], x_sol[1, i]), radius_obj, linewidth=0.0, color='tab:blue'))
            ax.scatter(p_obs_1_val[0], p_obs_1_val[1], color='tab:orange', s=50, label='Obstacle 1')
            ax.scatter(p_obs_2_val[0], p_obs_2_val[1], color='tab:orange', s=50, label='Obstacle 2')
            ax.scatter(x_sol[0, i], x_sol[1, i], color='black', s=50, marker="x", label="OBJECT", zorder=10)
            ax.plot(p_ref_val[0, :], p_ref_val[1, :], color='tab:gray', linewidth=3, label='Ref. traj.')
            fig.subplots_adjust(right=0.6)
            ax.legend(loc='center', bbox_to_anchor=(1.4, 0.5))
            plt.savefig(os.path.join(images_folder, "step_{:02d}.png".format(i)), dpi=600)
            plt.close()
    plot_results()


if __name__ == "__main__":
    main()


## TO Generate Video:
## $ ffmpeg -framerate 10 -i step_%02d.png -c:v libx264 -pix_fmt yuv420p -r 30 output.mp4