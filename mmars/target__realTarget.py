
import numpy as np
from .tracking import Tracking

class RealTarget(Tracking):
    def __init__(self, real_target_file_path):
        self.__file_content = np.loadtxt(real_target_file_path, delimiter=',')
        self.__time = self.__file_content[:, 0]
        self.__x = self.__file_content[:, 1]
        self.__y = self.__file_content[:, 2]
        self.__vx = np.diff(self.__x) / np.diff(self.__time)
        self.__vy = np.diff(self.__y) / np.diff(self.__time)
    
    ##############################################################################################
    def get_trajectory(self):
        pass

    ##############################################################################################
    def get_delta_time(self):
        pass