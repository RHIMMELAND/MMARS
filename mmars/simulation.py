import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .target import Target
from .fmcwRadar import FmcwRadar


class Simulation():
    def __init__(self, 
                 radar_setup: FmcwRadar, 
                 target_setup: Target
                 ):
        
        self.__frames = None
        self.__x = None
        self.__y = None
        self.__vx = None
        self.__vy = None
        
        self.__radar_setup = radar_setup
        self.__target_setup = target_setup
        
    def run(self):
        print(f"Running simulation with {self.__radar_setup} and {self.__target_setup}")
        self.__x,self.__y,self.__vx,self.__vy = self.__target_setup.get_trajectory()
        idx = self.__radar_setup.get_IF_signal.shape
        self.__frames = np.zeros((len(self.__x), idx[0], idx[1], idx[2], idx[3]), dtype=complex)
        self.__SNRs = np.zeros(len(self.__x))

        for i in tqdm(range(len(self.__x))):
            self.__radar_setup.radar_to_target_measures(self.__x[i], self.__y[i], self.__vx[i], self.__vy[i])
            self.__frames[i] = self.__radar_setup.get_IF_signal
            self.__SNRs[i] = self.__radar_setup.get_current_SNR()

        
    def plot(self):
        if self.__x is None:
            raise ValueError("No data to plot. Run the simulation first.")
        else:

            plt.figure(figsize=(10,10))

            plt.plot(self.__x, self.__y, label="Ground truth", color="black", linewidth=5)
            if self.__tracking_algorithm is not None:
                plt.scatter(self.__tracking_data_x, self.__tracking_data_y, label="Estimated trajectory")
            plt.xlabel("x-position [m]")
            plt.ylabel("y-position [m]")
            plt.legend()
            plt.show()

    def get_data(self):
        return self.__frames
    
    def get_tracking_data(self):
        return self.__tracking_data_x, self.__tracking_data_y, self.__tracking_data_vx, self.__tracking_data_vy
    
    def get_SNR(self):
        return self.__SNRs
    
    def get_number_of_frames(self):
        return len(self.__frames)