import numpy as np
from abc import ABC, abstractmethod

class Target(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    ##############################################################################################
    @abstractmethod
    def generate_trajectory(self):
        pass
    
    ##############################################################################################
    @abstractmethod
    def get_trajectory(self):
        pass