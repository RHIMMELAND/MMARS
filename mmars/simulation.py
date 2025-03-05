import numpy as np

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
        
        print(self.__radar_setup.__tx_antennas)
        """
        self.compute_radial_distances(self.__radar_setup.__tx_antennas, 
                                      self.__radar_setup.__rx_antennas, 
                                      self.__target_setup.__x, 
                                      self.__target_setup.__y)"""