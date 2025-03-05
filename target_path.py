import numpy as np
import matplotlib.pyplot as plt

class target_path:
    def __init__(self, start_pos, start_vel, T_tot = 20.0, T_F = 40.0e-3):

        # Check if the start position is a 2x1 matrix
        if start_pos.shape != (2, 1):
            raise ValueError("Position must be a 2x1 matrix")
        # Check if the start velocity is a float
        if type(start_vel) != float:
            raise ValueError("Velocity must be a float")
        # Check if the total time is a float
        if type(T_tot) != float:
            raise ValueError("Total time must be a float")
        # Check if the frame time is a float
        if type(T_F) != float:
            raise ValueError("Frame time must be a float")
        
        # Define parameters for the target trajectory
        self.__start_pos = start_pos    # Initial position
        self.__start_vel = start_vel    # Initial velocity
        self.__T_tot = T_tot            # Total simulation time
        self.__T_F = T_F                # Frame time, temporal resolution

        # Define the target trajectory variables
        self.__x, self.__y, self.__vx, self.__vy = 0,0,0,0
    
    ##############################################################################################

    def generate_trajectory(self, trajectory_type="sinusoidal"):
        """
        Generate a simulated target trajectory. Ideal Point Target.

        Parameters:
            T (float): Total simulation time in seconds.
            dt (float): Time step in seconds / resolution.
            speed (float): Constant speed of the target in m/s.
            trajectory_type (str): Type of trajectory ("linear", "sinusoidal", "random").
        """
        num_steps = int(self.__T_tot / self.__T_F)
        # Initialize arrays for position and velocity
        x = np.zeros(num_steps)
        y = np.zeros(num_steps)
        # Initial position
        x[0], y[0] = self.__start_pos[0], self.__start_pos[1]
        # Generate trajectory based on type
        for i in range(1, num_steps):
            if trajectory_type == "linear":
                x[i] = x[i-1] + self.__start_vel * self.__T_F
                y[i] = y[i-1]  # Straight line
            elif trajectory_type == "sinusoidal":
                y[i] = y[i-1] + self.__start_vel * self.__T_F
                x[i] = self.__start_pos[0] + 10 * np.sin(0.1 * (y[i]-self.__start_pos[1]))  # Sine wave path
            elif trajectory_type == "random":
                x[i] = x[i-1] + self.__start_vel * self.__T_F
                y[i] = y[i-1] + np.random.uniform(-2, 2)  # Random small movements
        # Calculate velocity
        vx = np.gradient(x, self.__T_F)
        vy = np.gradient(y, self.__T_F)
        self.__x, self.__y, self.__vx, self.__vy = x, y, vx, vy
    
    ##############################################################################################

    def get_trajectory(self):
        """
        Get the target trajectory.

        Returns:
            x, y, vx, vy (numpy arrays)
        """
        return self.__x, self.__y, self.__vx, self.__vy