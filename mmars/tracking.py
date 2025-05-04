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

        x0 = [self.__initial_kinematics[0,0], self.__initial_kinematics[1,0], .2, .2]

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
        alpha_hat_list = []
        alpha_hat_S_list = []
        D_KL_phi_bar = np.zeros((N_frames, 2, 1))
        D_KL_phi_barbar = np.zeros((N_frames, 2, 2))
        
        for k in range(self.__N_radar):
            mrblat_functions_list.append(mmars.MRBLaT_Functions(self.__radar_parameters[k]))

        for N in tqdm(range(0, N_frames)):
            for k in range(self.__N_radar):
                frame_iq_radar_data = self.__iq_radar_data[k][N,:,:,0,:]
                data_fourier = np.fft.fft(frame_iq_radar_data, axis=-1).flatten()

                D_KL_result = minimize(mrblat_functions_list[k].D_KL, x0, bounds = bound,  args=(data_fourier, x0, (1,1,1,1), (1,1,1,1), False), method='nelder-mead')
                D_KL_phi_bar[N] = D_KL_result.x[:2,np.newaxis]
                alpha_hat = mrblat_functions_list[k].get_alpha_hat(data_fourier, x0[0], x0[1])[0]
                alpha_hat_list.append(alpha_hat)

                eps_bar = np.array([[D_KL_result.x[0]], [D_KL_result.x[1]], [0.], [0.]])
                eps_bar_list[k, N] = eps_bar
                D_KL_result = minimize(mrblat_functions_list[k].D_KL, D_KL_result.x, bounds = bound,  args=(data_fourier, D_KL_result.x, (0,0,1,1), (1,1,1,1), False), method='powell')
                D_KL_phi_barbar[N] = np.array([[D_KL_result.x[2], 0], [0, D_KL_result.x[3]]])
                eps_barbar_inv_list[k, N] = (np.array([[1/D_KL_result.x[2],0,0,0], [0,1/D_KL_result.x[3],0,0], [0,0,0,0], [0,0,0,0]]))#/(4*self.__N_radar)

                x0 = [eps_bar[0,0], eps_bar[1,0], D_KL_result.x[2], D_KL_result.x[3]]

                if N in [0,1]:
                    ### HEAT MAP ###

                    heatmap_res = 101

                    heatmap_pos = np.zeros((heatmap_res, heatmap_res))
                    heatmap_var = np.zeros((heatmap_res, heatmap_res))

                    heatmap_pos_x = np.linspace(-15,15, heatmap_res)
                    heatmap_pos_y = np.linspace(3,18, heatmap_res)
                    heatmap_var_x = 10**(np.linspace(-200, 20, heatmap_res)/20)
                    heatmap_var_y = 10**(np.linspace(-200, 20, heatmap_res)/20)

                    for i in range(heatmap_res):
                        for j in range(heatmap_res):
                            heatmap_pos[i,j] = mrblat_functions_list[k].D_KL([heatmap_pos_x[j], heatmap_pos_y[i], x0[2], x0[3]], data_fourier, x0, (1,1,1,1), (1,1,1,1), False)
                    argmin_heatmap_pos = np.unravel_index(np.argmin(heatmap_pos, axis=None), heatmap_pos.shape)

                    for i in range(heatmap_res):
                        for j in range(heatmap_res):
                            heatmap_var[i,j] = mrblat_functions_list[k].D_KL([heatmap_pos_x[argmin_heatmap_pos[0]], heatmap_pos_y[argmin_heatmap_pos[1]], heatmap_var_x[j], heatmap_var_y[i]], data_fourier, x0, (1,1,1,1), (1,1,1,1), False)
                    argmin_heatmap_var = np.unravel_index(np.argmin(heatmap_var, axis=None), heatmap_var.shape)

                    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                    
                    p0 = ax[0].pcolormesh(heatmap_pos_x, heatmap_pos_y, np.array(heatmap_pos), shading='auto')
                    fig.colorbar(p0, ax=ax[0])
                    ax[0].scatter(heatmap_pos_x[argmin_heatmap_pos[1]], heatmap_pos_y[argmin_heatmap_pos[0]], color='red', marker='x', s=100, label='ARGMIN')
                    ax[0].scatter(x0[0], x0[1], color='blue', marker='x', s=100, label='INITIAL')
                    ax[0].set_title(f'Position Heatmap (frame {N})')
                    ax[0].set_xlabel('X Position (m)')
                    ax[0].set_ylabel('Y Position (m)')
                    ax[0].grid()
                    ax[0].legend()
                    
                    p1 = ax[1].pcolormesh(heatmap_var_x, heatmap_var_y, np.array(heatmap_var), shading='auto')
                    fig.colorbar(p1, ax=ax[1])
                    ax[1].scatter(heatmap_var_x[argmin_heatmap_var[1]], heatmap_var_y[argmin_heatmap_var[0]], color='red', marker='x', s=100, label='ARGMIN')
                    ax[1].scatter(x0[2], x0[3], color='blue', marker='x', s=100, label='INITIAL')
                    ax[1].set_title(f'Variance Heatmap (frame {N})')
                    ax[1].set_xlabel('X Variance (m)')
                    ax[1].set_ylabel('Y Variance (m)')
                    ax[1].set_xscale('log')
                    ax[1].set_yscale('log')
                    ax[1].grid()
                    ax[1].legend()

                    plt.tight_layout()
                    plt.show()

                    ### HEATY MAP END ###

            # phi_bar_bar_inv = 0
            # eps_barbar_inv_eps_bar_sum = 0
            # for k in range(self.__N_radar):
            #     phi_bar_bar_inv += eps_barbar_inv_list[k, N] 
            #     eps_barbar_inv_eps_bar_sum += eps_barbar_inv_list[k, N] @ eps_bar_list[k, N]
            # phi_bar_bar = np.linalg.pinv(phi_bar_bar_inv)
            # phi_barbar_list[N] = phi_bar_bar
            # phi_bar = phi_bar_bar @ eps_barbar_inv_eps_bar_sum
            # phi_bar_list[N] = phi_bar
            
            # for _ in range(N_iter):
            #     for n in range(N - fifo_counter, N+1):
            #         if N == 0:
            #             pass
            #         elif n == N - fifo_counter:
            #             phi_bar_bar_inv = T_T@G_inv_T@Lambda_a@G_inv@T
            #             eps_barbar_inv_eps_bar_sum = (T_T@G_inv_T@Lambda_a@G_inv@T)@T_inv@phi_bar_list[n+1]
            #             for k in range(self.__N_radar):
            #                 phi_bar_bar_inv += eps_barbar_inv_list[k, n]
            #                 eps_barbar_inv_eps_bar_sum += eps_barbar_inv_list[k, n] @ eps_bar_list[k, n]
            #             phi_bar_bar = np.linalg.inv(phi_bar_bar_inv)
            #             phi_barbar_list[n] = phi_bar_bar

            #             phi_bar = phi_bar_bar @ eps_barbar_inv_eps_bar_sum
            #             phi_bar_list[n] = phi_bar
            #         elif n == N:
            #             phi_bar_bar_inv = G_inv_T@Lambda_a@G_inv
            #             eps_barbar_inv_eps_bar_sum = (G_inv_T@Lambda_a@G_inv)@T@phi_bar_list[n-1]
            #             for k in range(self.__N_radar):
            #                 phi_bar_bar_inv += eps_barbar_inv_list[k, n]
            #                 eps_barbar_inv_eps_bar_sum += eps_barbar_inv_list[k, n] @ eps_bar_list[k, n]
            #             phi_bar_bar = np.linalg.inv(phi_bar_bar_inv)
            #             phi_barbar_list[n] = phi_bar_bar
                    
            #             phi_bar = phi_bar_bar @ eps_barbar_inv_eps_bar_sum
            #             phi_bar_list[n] = phi_bar
            #         else:
            #             phi_bar_bar_inv = G_inv_T@Lambda_a@G_inv + T_T@G_inv_T@Lambda_a@G_inv@T
            #             eps_barbar_inv_eps_bar_sum = (G_inv_T@Lambda_a@G_inv)@T@phi_bar_list[n-1] + (T_T@G_inv_T@Lambda_a@G_inv@T)@T_inv@phi_bar_list[n+1]
            #             for k in range(self.__N_radar):
            #                 phi_bar_bar_inv += eps_barbar_inv_list[k, n]
            #                 eps_barbar_inv_eps_bar_sum += eps_barbar_inv_list[k, n] @  eps_bar_list[k, n]
            #             phi_bar_bar = np.linalg.inv(phi_bar_bar_inv)
            #             phi_barbar_list[n] = phi_bar_bar

            #             phi_bar = phi_bar_bar @ eps_barbar_inv_eps_bar_sum
            #             phi_bar_list[n] = phi_bar

            #     if N >= 1:
            #         alpha = fifo_counter+1
            #         beta = np.zeros((4, 1))
            #         for n in range(1 + N - fifo_counter, N+1):
            #             outer_product_dummy = G_inv@(phi_bar_list[n]-T@phi_bar_list[n-1])
            #             beta += np.abs(outer_product_dummy)**2 + np.linalg.diagonal(G_inv_T@(phi_barbar_list[n]+T@phi_barbar_list[n-1]@T_T)@G_inv)[:,np.newaxis]
            #         Lambda_a = 1/(beta + 1) 
            #         Lambda_a = alpha*np.eye(4)*Lambda_a 

            # x0 = [phi_bar_list[N,0,0], phi_bar_list[N,1,0], phi_barbar_list[N,0,0], phi_barbar_list[N,1,1]]
            # alpha_hat_S_list.append(alpha_hat*mrblat_functions_list[k].get_S_signal(x0[0], x0[1]))
            # if fifo_counter < fifo_length-1:
            #     fifo_counter += 1
        return phi_bar_list, phi_barbar_list, alpha_hat_list, alpha_hat_S_list, D_KL_phi_bar, D_KL_phi_barbar
    
    def run_kalman(self, T_frame, N_frames = None, bound = [(-100,100), (1,100), (0.00001, 100), (0.00001, 100)], music_buffer_bins = 4):
        """
        docstring
        """
        x0 = [self.__initial_kinematics[0,0], self.__initial_kinematics[1,0], self.__initial_kinematics[2,0], self.__initial_kinematics[3,0]]

        T = np.array([[1, 0, T_frame, 0],
                    [0, 1, 0, T_frame],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

        G = np.diagflat([(T_frame**2)/2,(T_frame**2)/2,T_frame,T_frame])
        sigma_a = (0.5/3)
        Lambda_a = np.eye((4))*sigma_a**2
        Q = G@G.T*sigma_a
        # Q = (G@(Lambda_a)@G.T) # np.linalg.inv

        H_k = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0]])
        H = np.vstack([H_k for _ in range(self.__N_radar)])

        # P0 = np.diag([1e-2, 1e-2, 1e1, 1e1])
        # P0 = np.diag([1e-10, 1e-10, 1e-10, 1e-10])
        P0 = np.diag([0, 0, 0, 0])
        
        measurement_noise = np.eye(self.__N_radar*2)*(0.17/3)**2#*1e-7#

        if N_frames is None:
            N_frames = self.__iq_radar_data[0].shape[0]

        range_values = 0

        phi_bar_list = []#np.zeros((N_frames, 4, 1))
        phi_barbar_list = []#np.zeros((N_frames, 4, 4))

        time_from_last_sample = 0
        previous_data_fourier_arg_max_median = -1
        data_fourier_arg_max_median = 0

        for N in tqdm(range(0, N_frames)):
            data_position = []
            for k in range(self.__N_radar):
                # print(k)
                range_values = np.linspace(0, self.__radar_parameters[k]["max_range"], self.__iq_radar_data[k].shape[-1])
                frame_iq_radar_data = self.__iq_radar_data[k][N,:,:,0,:]
                data_fourier = np.fft.fft(frame_iq_radar_data, axis=-1)
                data_fourier_arg_max = np.argmax(data_fourier, axis=-1)
                data_fourier_arg_max_median = np.median(data_fourier_arg_max)
                radial_distance = range_values[int(data_fourier_arg_max_median)]

                phasors = np.zeros((len(self.__radar_parameters[k]["tx_antennas"])*len(self.__radar_parameters[k]["rx_antennas"]), music_buffer_bins*2), dtype=complex)
                for i in range(len(self.__radar_parameters[k]["tx_antennas"])):
                    for j in range(len(self.__radar_parameters[k]["rx_antennas"])):
                        phasors[i*(len(self.__radar_parameters[k]["tx_antennas"])+1)+j] = data_fourier[i,j,int(data_fourier_arg_max_median-music_buffer_bins):int(data_fourier_arg_max_median+music_buffer_bins)]
                R = phasors @ phasors.conj().T*(1/self.__radar_parameters[k]["N_samples"])
                music = doa_music(R, 1, scan_angles = np.linspace(-90, 90, 1001))
                anglebins = np.linspace(-90, 90, 1001)
                detected_angle = anglebins[np.argmax(music)]

                tracking_data_x = -(radial_distance*np.cos(np.deg2rad(90-detected_angle))) + self.__radar_parameters[k]["position"][0,0]
                tracking_data_y = (radial_distance*np.sin(np.deg2rad(90-detected_angle))) + self.__radar_parameters[k]["position"][0,1]
                data_position.append([[tracking_data_x], [tracking_data_y]])
            data_position = np.array(data_position).flatten()

            if previous_data_fourier_arg_max_median == data_fourier_arg_max_median:
                time_from_last_sample += T_frame
                continue
            else:
                T = np.array([[1, 0, time_from_last_sample, 0],
                            [0, 1, 0, time_from_last_sample],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
                G = np.diagflat([(time_from_last_sample**2)/2,(time_from_last_sample**2)/2,time_from_last_sample,time_from_last_sample])
                time_from_last_sample = 0
                previous_data_fourier_arg_max_median = data_fourier_arg_max_median

            # Prediction step            
            x_prediction = T @ x0
            P_prediction = T @ P0 @ T.T + Q

            # Measurement step
            x0, P0 = kalman_gain_and_state_estimate(data_position, x_prediction, H, P_prediction, measurement_noise)

            phi_bar_list.append(x0[:,np.newaxis]) # KF.x[:,np.newaxis]# 
            phi_barbar_list.append(P0) # KF.P # 

        return np.array(phi_bar_list), np.array(phi_barbar_list)
        
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

def doa_music(R, n_sig, d = 0.5, scan_angles = range(-90, 91)):
    """ MUSIC algorithm implementation """
    n_array = np.shape(R)[0]
    array = np.linspace(0, (n_array - 1) * d, n_array)
    scan_angles = np.array(scan_angles)

    # 'eigh' guarantees the eigen values are sorted
    _, e_vecs = np.linalg.eigh(R)
    noise_subspace = e_vecs[:, :-n_sig]

    array_grid, angle_grid = np.meshgrid(array, np.radians(scan_angles), indexing = "ij")
    steering_vec = np.exp(-1.j * 2 * np.pi * array_grid * np.sin(angle_grid)) 
    
    # 2-norm (frobenius)
    ps = 1 / np.square(np.linalg.norm(steering_vec.conj().T @ noise_subspace, axis = 1))

    return 10 * np.log10(ps)

def kalman_gain_and_state_estimate(data, X_p, H, P, R):
    """
    docstring
    """
    # Kalman gain
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)

    # State estimate
    x_hat = X_p + K @ (data - H @ X_p)

    # Covariance estimate
    P = (np.eye(P.shape[0]) - K @ H) @ P @ (np.eye(P.shape[0]) - K @ H).T + K @ R @ K.T

    return x_hat, P
