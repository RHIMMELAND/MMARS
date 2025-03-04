import numpy as np

class simulation:
    def __init__(self, 
                 radar_Setup, 
                 target_Setup
                 

                 ):

        print("Simulation started")
        """ Simulation """


    def compute_radial_distances(TX_locations, RX_locations, target_locations):
        """Compute all radial distances from the target to the TXs and RXs"""
        target_location = np.array([x[time_step], y[time_step]])
        r_TX = np.linalg.norm(TX_locations - target_location, axis=1)
        r_RX = np.linalg.norm(RX_locations - target_location, axis=1)



    def compute_radial_velocity(x_target, y_target, x_target_velocity, y_target_velocity, x_base, y_base):
        dx = x_target - x_base
        dy = y_target - y_base
        distance = np.sqrt(dx**2 + dy**2)
        radial_velocity = (dx * x_target_velocity + dy * y_target_velocity) / distance
        return radial_velocity
    
    # THIS IS A DRAFT!