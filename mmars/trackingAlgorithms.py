import numpy as np

"""
class Bayesian_localisation_and_tracking:
    def __init__(self,
                 ):
        pass


class Kalman_filter:
    def __init__(self,
                 ):
        pass
"""

file = open("sim_data.pkl", "rb")

data = pickle.load(file)

file.close()

def BLaT(Z, mu_a, Lambda_a, x_start, y_start):

    sigma_tilde_hat = 1
    r_hat = np.sqrt(x_start**2 + y_start**2)
    v_hat = 0

    for n in range(Z.shape[0]):
        print(n)

BLaT(data, 0, 0, 0, 0)

        

