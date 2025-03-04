import numpy as np

class target:
    def __init__(self, 
                 start_position, 
                 start_velocity, 
                 T_total = 20.0, 
                 T_frame = 40.0e-3
                 ):
        
        """ 
        Setup a target path 
        
        This class sets up a target path with the given parameters.
        
        Parameters
        ----------

        start_position : np.array
            The initial position of the target in 2D space. The position is a 2x1 np.array.
        start_velocity : float
            The initial velocity of the target in m/s.
        T_total : float
            The total simulation time in seconds.
        T_frame : float
            The frame time, temporal resolution in seconds.

        Attributes
        ----------
        -||-

        Notes
        -----
        -||-
        """

        self.__start_position = start_position
        self.__start_velocity = start_velocity
        self.__T_total = T_total
        self.__T_frame = T_frame

        # Check if the start position is a 1x2 matrix
        if self.__start_position.shape != (1, 2):
            raise ValueError("Position must be a 2x1 matrix")
        # Check if the start velocity is a float
        if type(self.__start_velocity) != float:
            raise ValueError("Velocity must be a float")
        # Check if the total time is a float
        if type(self.__T_total) != float:
            raise ValueError("Total time must be a float")
        # Check if the frame time is a float
        if type(self.__T_frame) != float:
            raise ValueError("Frame time must be a float")


        # Define the target trajectory variables
        self.__x, self.__y = 0,0
    
    ##############################################################################################

    def generate_trajectory(self, 
                            trajectory_type="sinusoidal"
                            ):
        """
        Generate a simulated target trajectory. Ideal Point Target.

        Parameters:
            trajectory_type (str): Type of trajectory ("linear", "sinusoidal", "random").

        """
        num_steps = int(self.__T_total / self.__T_frame)

        # Initialize arrays for position and velocity
        x = np.zeros(num_steps)
        y = np.zeros(num_steps)

        # Initial position
        x[0], y[0] = self.__start_position[0,0], self.__start_position[0,1]

        # Generate trajectory based on type
        for i in range(1, num_steps):
            if trajectory_type == "linear":
                x[i] = x[i-1] + self.__start_velocity * self.__T_frame
                y[i] = y[i-1]  # Straight line
            elif trajectory_type == "sinusoidal":
                y[i] = y[i-1] + self.__start_velocity * self.__T_frame
                x[i] = self.__start_position[0,0] + 10 * np.sin(0.1 * (y[i]-self.__start_position[0,1]))  # Sine wave path
            elif trajectory_type == "random":
                x[i] = x[i-1] + self.__start_velocity * self.__T_frame
                y[i] = y[i-1] + np.random.uniform(-2, 2)  # Random small movements
        
        # Calculate velocity
        vx = np.gradient(x, self.__T_frame)
        vy = np.gradient(y, self.__T_frame)
        self.__x, self.__y = x, y
    
    ##############################################################################################

    def get_trajectory(self):
        """
        Get the target trajectory.

        Returns:
            x, y (numpy arrays)
        """
        return self.__x, self.__y