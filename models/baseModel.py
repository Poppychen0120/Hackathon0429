from collections.abc import Callable
import numpy as np

class BaseModel:
    model_name = "BaseModel"
    model_config = None
    _sampling_time = np.nan

    def __init__(self, sampling_time: float):
        self._sampling_time = sampling_time

    def animateSimulation(self, x_trajectory: np.ndarray, u_trajectory: np.ndarray):
        raise NotImplementedError("Subclasses must implement animateSimulation.")


    def simulateClosedLoop(self, sim_length:int, x_init:np.ndarray, controller: Callable, num_agents:int=1):
        nx = self.model_config.nx
        nu = self.model_config.nu
        x_trajectory = np.zeros((nx * num_agents, sim_length+1))
        u_trajectory = np.zeros((nu * num_agents, sim_length))
        x_trajectory[:, 0] = x_init.flatten()
        for i in range(sim_length):
            u = controller(x_trajectory[:, i])
            u_trajectory[:, i] = u
            for i_agent in range(num_agents):
                x_trajectory[i_agent*nx:(i_agent+1)*nx, i+1] = self.I(x0=x_trajectory[i_agent*nx:(i_agent+1)*nx, i], p=u[i_agent*nu:(i_agent+1)*nu])['xf'].full().flatten()
            print(f"Step {i+1}: x = {x_trajectory[:, i+1]}, u = {u}")
        return x_trajectory, u_trajectory


    def simulateOpenLoop(self, x_init: np.ndarray, u_trajectory: np.ndarray, num_agents:int=1):
        sim_length = u_trajectory.shape[1]
        nx = self.model_config.nx
        nu = self.model_config.nu
        x_trajectory = np.zeros((nx * num_agents, sim_length + 1))
        x_trajectory[:, 0] = x_init.flatten()
        for i in range(sim_length):
            for i_agent in range(num_agents):
                x_trajectory[i_agent*nx:(i_agent+1)*nx, i+1] = self.I(x0=x_trajectory[i_agent*nx:(i_agent+1)*nx, i], p=u_trajectory[i_agent*nu:(i_agent+1)*nu, i])['xf'].full().flatten()
            print(f"Step {i+1}: x = {x_trajectory[:, i+1]}, u = {u_trajectory[:, i]}")
        return x_trajectory