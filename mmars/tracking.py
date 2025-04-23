import numpy as np
import mmars
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

from tqdm import tqdm

import matplotlib.pyplot as plt


class Tracking():
    def __init__(self,
                iq_radar_data,
                radar_parameters,
                initial_kinematics):
    
        """
        docstring
        """
    
        if not isinstance(iq_radar_data, list):
            self.__iq_radar_data = [iq_radar_data]
        else:
            self.__iq_radar_data = iq_radar_data

        if not isinstance(radar_parameters, list):
            self.__radar_parameters = [radar_parameters]
        else:
            self.__radar_parameters = radar_parameters

        if len(self.__iq_radar_data) != len(self.__radar_parameters):
            raise ValueError("The number of radar data and radar parameters must be the same")

        self.__initial_kinematics = initial_kinematics
    
        self.__N_radar = len(self.__iq_radar_data)
    
    def run_mrblat(self, T_frame, N_iter = 100, N_frames = None, bound = [(-20,20), (1,100), (0.00001, 100), (0.00001, 100)], fifo_length = None):   

        """
        docstring
        """

        x0 = [self.__initial_kinematics[0,0], self.__initial_kinematics[1,0], 0.1, 0.1]

        T = np.array([[1, 0, T_frame, 0],
                    [0, 1, 0, T_frame],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
        T_inv = np.linalg.inv(T)
        T_inv_T = T_inv.T
        T_T = T.T

        G = np.diagflat([(T_frame**2)/2,(T_frame**2)/2,T_frame,T_frame])
        G_inv = np.linalg.inv(G)
        G_inv_T = G_inv.T
        G_T = G.T

        if N_frames is None:
            N_frames = self.__iq_radar_data[0].shape[0]
        if fifo_length is None:
            fifo_length = self.__iq_radar_data[0].shape[0]
    
        eps_bar_list = np.zeros((self.__N_radar, N_frames, 4, 1))
        eps_barbar_inv_list = np.zeros((self.__N_radar, N_frames, 4, 4))

        phi_bar_list = np.zeros((N_frames, 4, 1))
        phi_barbar_list = np.zeros((N_frames, 4, 4))

        initial_process_noise_precision = 1/8
        Lambda_a = np.eye((4))*initial_process_noise_precision

        fifo_counter = 0
        mrblat_functions_list = []
        
        for k in range(self.__N_radar):
            mrblat_functions_list.append(mmars.MRBLaT_Functions(self.__radar_parameters[k]))

        for N in tqdm(range(0, N_frames)):
            for k in range(self.__N_radar):
                frame_iq_radar_data = self.__iq_radar_data[k][N,:,:,0,:]
                data_fourier = np.fft.fft(frame_iq_radar_data, axis=-1).flatten()

                x0_k = x0.copy()
                x0_k[0] -= self.__radar_parameters[k]["position"][0,0]
                x0_k[1] -= self.__radar_parameters[k]["position"][0,1]

                D_KL_result = minimize(mrblat_functions_list[k].D_KL, x0_k, bounds = bound,  args=(data_fourier, x0_k[0], x0_k[1], (1,1,1,1), False), method='nelder-mead')
                eps_bar = np.array([[D_KL_result.x[0] + self.__radar_parameters[k]["position"][0,0]], [D_KL_result.x[1]+self.__radar_parameters[k]["position"][0,1]], [0.], [0.]])
                eps_bar_list[k, N] = eps_bar
                eps_barbar_inv_list[k, N] = (np.array([[1/D_KL_result.x[2],0,0,0], [0,1/D_KL_result.x[3],0,0], [0,0,0,0], [0,0,0,0]]))

                phi_bar_bar_inv = 0
                eps_barbar_inv_eps_bar_sum = 0
                for k in range(self.__N_radar):
                    phi_bar_bar_inv += eps_barbar_inv_list[k, N] 
                    eps_barbar_inv_eps_bar_sum += eps_barbar_inv_list[k, N] @ eps_bar_list[k, N]
                phi_bar_bar = np.linalg.pinv(phi_bar_bar_inv)
                phi_barbar_list[N] = phi_bar_bar

                phi_bar = phi_bar_bar @ eps_barbar_inv_eps_bar_sum
                phi_bar_list[N] = phi_bar
            for _ in range(N_iter):
                for n in range(N - fifo_counter, N+1):
                    if N == 0:
                        pass
                    elif n == 0:
                        phi_bar_bar_inv = T_T@G_inv_T@Lambda_a@G_inv@T
                        eps_barbar_inv_eps_bar_sum = (T_T@G_inv_T@Lambda_a@G_inv@T)@T_inv@phi_bar_list[n+1]
                        for k in range(self.__N_radar):
                            phi_bar_bar_inv += eps_barbar_inv_list[k, n]
                            eps_barbar_inv_eps_bar_sum += eps_barbar_inv_list[k, n] @ eps_bar_list[k, n]
                        phi_bar_bar = np.linalg.inv(phi_bar_bar_inv)
                        phi_barbar_list[n] = phi_bar_bar

                        phi_bar = phi_bar_bar @ eps_barbar_inv_eps_bar_sum
                        phi_bar_list[n] = phi_bar
                    elif n == N:
                        phi_bar_bar_inv = G_inv_T@Lambda_a@G_inv
                        eps_barbar_inv_eps_bar_sum = (G_inv_T@Lambda_a@G_inv)@T@phi_bar_list[n-1]
                        for k in range(self.__N_radar):
                            phi_bar_bar_inv += eps_barbar_inv_list[k, n]
                            eps_barbar_inv_eps_bar_sum += eps_barbar_inv_list[k, n] @ eps_bar_list[k, n]
                        phi_bar_bar = np.linalg.inv(phi_bar_bar_inv)
                        phi_barbar_list[n] = phi_bar_bar
                    
                        phi_bar = phi_bar_bar @ eps_barbar_inv_eps_bar_sum
                        phi_bar_list[n] = phi_bar
                    else:
                        phi_bar_bar_inv = G_inv_T@Lambda_a@G_inv + T_T@G_inv_T@Lambda_a@G_inv@T
                        eps_barbar_inv_eps_bar_sum = (G_inv_T@Lambda_a@G_inv)@T@phi_bar_list[n-1] + (T_T@G_inv_T@Lambda_a@G_inv@T)@T_inv@phi_bar_list[n+1]
                        for k in range(self.__N_radar):
                            phi_bar_bar_inv += eps_barbar_inv_list[k, n]
                            eps_barbar_inv_eps_bar_sum += eps_barbar_inv_list[k, n] @  eps_bar_list[k, n]
                        phi_bar_bar = np.linalg.inv(phi_bar_bar_inv)
                        phi_barbar_list[n] = phi_bar_bar

                        phi_bar = phi_bar_bar @ eps_barbar_inv_eps_bar_sum
                        phi_bar_list[n] = phi_bar

                if N >= 1:
                    alpha = fifo_counter+1
                    beta = np.zeros((4, 1))
                    for n in range(1 + N - fifo_counter, N+1):
                        outer_product_dummy = G_inv@(phi_bar_list[n]-T@phi_bar_list[n-1])
                        beta += np.abs(outer_product_dummy)**2 + np.linalg.diagonal(G_inv_T@(phi_barbar_list[n]+T@phi_barbar_list[n-1]@T_T)@G_inv)[:,np.newaxis]
                    Lambda_a = 1/(beta + 1) 
                    Lambda_a = alpha*np.eye(4)*Lambda_a 

            x0 = [phi_bar_list[N,0,0], phi_bar_list[N,1,0], phi_barbar_list[N,0,0], phi_barbar_list[N,1,1]]
            if fifo_counter < fifo_length:
                fifo_counter += 1
        return phi_bar_list, phi_barbar_list
    
    def run_kalman(self):
        """
        docstring
        """
        pass
    def run_pda(self):
        """
        docstring
        """
        pass
    def run_mht(self):
        """
        docstring
        """
        pass
    def run_jpda(self):
        """
        docstring
        """
        pass