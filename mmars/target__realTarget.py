
import numpy as np
from .target import Target

class RealTarget(Target):
    def __init__(self, real_target_file_path):
        super().__init__()
        self.__real_target_file_path = real_target_file_path
        # Check if the file path is a string
        if type(self.__real_target_file_path) != str:
            raise ValueError("File path must be a string")
        # Check if the file exists.
        try:
            self.__file_content = np.loadtxt(self.__real_target_file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {self.__real_target_file_path} not found. Please check the file path.")
        
    ##############################################################################################
    def generate_trajectory(self):
        self.__file_content = np.loadtxt(self.__real_target_file_path)
        self.__time = self.__file_content[:, 0]
        self._x = self.__file_content[:, 1]
        self._y = self.__file_content[:, 2]
        self._vy = np.diff(self._y) / np.diff(self.__time)
        self._vx = np.append(self._vx, self._vx[-1])  # Extend vx to match the length of x
        self._vy = np.append(self._vy, self._vy[-1])  # Extend vy to match the length of y

    ##############################################################################################
    def get_delta_time(self):
        return np.mean(np.diff(self.__time))