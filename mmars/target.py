import numpy as np
from abc import ABC, abstractmethod

class Target(ABC):
    def __init__(self):
        # Define the target trajectory variables
        self._x, self._y,self._vx, self._vy = 0,0,0,0
    
    ##############################################################################################
    def get_trajectory(self):
        """
        Get the target trajectory.

        Returns:
            x, y, vx, vy (numpy arrays)
        """
        return self._x, self._y, self._vx, self._vy
    
    ##############################################################################################
    @abstractmethod
    def generate_trajectory(self):
        pass