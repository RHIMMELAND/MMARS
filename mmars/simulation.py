import numpy as np
from tqdm import tqdm

from .target import Target
from .fmcwRadar import FmcwRadar


class Simulation:
    def __init__(self, 
                 radar_setup: FmcwRadar, 
                 target_setup: Target
                 ):
        
        self.__radar_setup = radar_setup
        self.__target_setup = target_setup
    def run(self):
        print(f"Running simulation with {self.__radar_setup} and {self.__target_setup}")
        x,y,vx,vy = self.__target_setup.get_trajectory()

        frames = []

        for i in tqdm(range(len(x))):
            self.__radar_setup.radar_to_target_measures(x[i], y[i], vx[i], vy[i])
            frames.append(self.__radar_setup.get_IF_signal())