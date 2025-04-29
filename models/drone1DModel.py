import casadi as ca
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from models.baseModel import BaseModel

"""
| DroneZModel |

x = (pz, vz)
a = (thrust)

\dot{pz} = vz
\dot{vz} = -g + (thrust - c * vz) / m
c: drag coefficient
"""

@dataclass
class DroneZConfig:
    nx: int = 2
    nu: int = 1
    mass: float = 1.0
    gravity: float = 9.81
    drag_coefficient: float = 0.5


class DroneZModel(BaseModel):
    def __init__(self, sampling_time):
        super().__init__(sampling_time)
        self.model_name = "DroneZModel"
        self.model_config = DroneZConfig()

        x = ca.MX.sym('x', self.model_config.nx)
        u = ca.MX.sym('u', self.model_config.nu)
        x_dot = ca.vertcat(
            x[1],  # \dot{pz}
            -self.model_config.gravity + (u[0] - self.model_config.drag_coefficient * x[1]) / self.model_config.mass  # \dot{vz}
        )
        dae = {'x': x, 'p': u, 'ode': x_dot}
        opts = {'tf': self._sampling_time}
        self.I = ca.integrator('I', 'rk', dae, opts)


    def animateSimulation(self, x_trajectory, u_trajectory, additional_lines_or_scatters=None):
        sim_length = u_trajectory.shape[1]
        fig, axes = plt.subplots(1, 2, sharey=True, width_ratios=[1., 4.], figsize=(10, 4.5))
        px = 0.
        ts = np.arange(0, sim_length+1) * self._sampling_time
        for i in range(sim_length+1):
            axes[0].set_aspect('equal')
            axes[0].set_xlim(-1, 1)
            axes[0].set_ylim(-1, 5)
            axes[0].set_xlabel('px(m)', fontsize=14)
            axes[0].set_ylabel('pz(m)', fontsize=14)
            object_patch = patches.Circle((px, x_trajectory[0, i]), radius=0.2, color="tab:blue")
            axes[0].add_patch(object_patch)
            if i < sim_length:
                patch_thrust = patches.Arrow(px, x_trajectory[0, i], 0., 0.1*u_trajectory[0, i], color="tab:green")
                axes[0].add_patch(patch_thrust)

            axes[0].set_title(f"1D Drone Sim: Step {i+1}")

            axes[1].set_xlabel('Time(s)', fontsize=14)
            axes[1].set_xlim(0, sim_length * self._sampling_time)
            axes[1].plot(ts[:i+1], x_trajectory[0, :i+1], color="tab:gray", linewidth=2, zorder=1)

            if additional_lines_or_scatters is not None:
                for key, value in additional_lines_or_scatters.items():
                    if value["type"] == "scatter":
                        axes[value["idx_ax"]].scatter(value["data"][0], value["data"][1], color=value["color"], s=value["s"], label=key, marker=value["marker"])
                    elif value["type"] == "line":
                        axes[value["idx_ax"]].plot(value["data"][0], value["data"][1], color=value["color"], linewidth=2, label=key, zorder=0)
            axes[0].legend()

            plt.show(block=False)
            plt.pause(0.05)
            axes[0].clear()
        return