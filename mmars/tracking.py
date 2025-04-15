import numpy as np
from scipy.optimize import minimize

#from typing import List
from tqdm import tqdm



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
    
    def run_mrblat(self, N_iter = 100, N_frames = None):

        """
        docstring
        """

        eps_bar_list = np.zeros((N_frames, 4, 1))
        eps_barbar_list = np.zeros((N_frames, 4, 4))

        phi_bar_list = np.zeros((N_frames+2, 4, 1))
        phi_barbar_list = np.zeros((N_frames+2, 4, 4))

        initial_variance = 1e-10

        phi_bar_list[0] = self.__initial_kinematics
        phi_barbar_list[0] = np.eye((4))*initial_variance

        # Initialze the Lambda a matrix
        initial_process_noise_precision = 1
        Lambda_a = np.eye((4))*initial_process_noise_precision

        fifo_counter = 1

        for N in tqdm(range(1, N_frames)):

            for k in range(self.__N_radar):
                frame_iq_radar_data = self.__iq_radar_data[k][N,:,:,0,:]

                data_fourier = np.fft.fft(frame_iq_radar_data, axis=-1).flatten()

                result = minimize(mrblatcomp.D_KL, (phi_bar_last_x_, phi_bar_last_y_, phi_barbar_last_0_, phi_barbar_last_1_), bounds = bound,  args=(data_fourier, phi_bar_last_x_, phi_bar_last_y_, (1,1,1,1), False), method='nelder-mead')
                
                return result.x[np.newaxis].T



            D_KL_res = broadcast_parameters(N, phi_bar_list[n,0,0], phi_bar_list[n,1,0], phi_barbar_list[n,0,0], phi_barbar_list[n,1,1])
            eps_bar = np.vstack((D_KL_res[:2], np.array([[0.], [0.]])))
            
            eps_bar_list[n] = eps_bar
            eps_barbar_list[n] = np.array([[D_KL_res[2][0],0,0,0], [0,D_KL_res[3][0],0,0], [0,0,0,0], [0,0,0,0]])
            
            for _ in range(N_iter):
                for i in range((n+1) - fifo, n+1):
                    eps_barbar_inv_extended = np.linalg.pinv(eps_barbar_list[i])
                    if (n+1) - i == 1:
                        phi_bar_bar_inv = eps_barbar_inv_extended + G_inv.T@Lambda_a@G
                        phi_bar_bar = np.linalg.pinv(phi_bar_bar_inv)
                        phi_barbar_list[i+1] = phi_bar_bar

                        phi_bar = phi_bar_bar @ (eps_barbar_inv_extended @ eps_bar_list[i] + (G_inv.T@Lambda_a@G)@T@phi_bar_list[i])
                        phi_bar_list[i+1] = phi_bar

                    else:
                        phi_bar_bar_inv = eps_barbar_inv_extended + G_inv.T@Lambda_a@G + T.T@G_inv.T@Lambda_a@G_inv@T
                        phi_bar_bar = np.linalg.pinv(phi_bar_bar_inv)
                        phi_barbar_list[i+1] = phi_bar_bar

                        phi_bar = phi_bar_bar @ (eps_barbar_inv_extended @ eps_bar_list[i] + (G_inv.T@Lambda_a@G)@T@phi_bar_list[i] + (T.T@G_inv.T@Lambda_a@G_inv@T)@T_inv@phi_bar_list[i+2])
                        phi_bar_list[i+1] = phi_bar     
        
                if n >= 1:
                    alpha = n+1
                    beta = np.zeros((4, 4))
                    for i in range((n+1) - fifo, n+1):
                        beta += np.linalg.norm(G_inv@(phi_bar_list[i+1]-T@phi_bar_list[i]))**2  + G_inv@(phi_barbar_list[i+1]+T@phi_barbar_list[i]@T.T)@G_inv.T
                    Lambda_a = np.linalg.pinv(beta/alpha)

            if fifo < 80:
                fifo += 1

        return phi_bar_list, phi_barbar_list



        pass
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