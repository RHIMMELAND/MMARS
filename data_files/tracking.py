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
    
    def run_mrblat(self, T_frame, N_iter = 100, N_frames = None, bound = ((-10, 10), (0, 50), (0.00001, 10), (0.00001, 10)), fifo_length = None, heatmap_list = [-1]):   

        """
        docstring
        """

        x0 = [self.__initial_kinematics[0,0], self.__initial_kinematics[1,0], 0.001, 0.001]

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

        intermediate_reult_pos = []
        intermediate_reult_var = []
        
        for k in range(self.__N_radar):
            mrblat_functions_list.append(mmars.MRBLaT_Functions(self.__radar_parameters[k]))

        for N in tqdm(range(0, N_frames)):
            for k in range(self.__N_radar):
                frame_iq_radar_data = self.__iq_radar_data[k][N,:,:,0,:]
                data_fourier = np.fft.fft(frame_iq_radar_data, axis=-1).flatten()
                
                x0k = x0.copy()
                x0k[0] = x0[0] - self.__radar_parameters[k]["position"][0,0]
                x0k[1] = x0[1] - self.__radar_parameters[k]["position"][0,1]

                D_KL_result = minimize(mrblat_functions_list[k].D_KL, x0k, bounds = bound,  args=(data_fourier, x0k, (1,1,1,1), (1,1,1,1), False), method='nelder-mead', options={'xatol': 1e-10, 'disp': False})
                alpha_hat = mrblat_functions_list[k].get_alpha_hat(data_fourier, x0k[0], x0k[1])[0]

                D_KL_result.x[0] = D_KL_result.x[0] + self.__radar_parameters[k]["position"][0,0]
                D_KL_result.x[1] = D_KL_result.x[1] + self.__radar_parameters[k]["position"][0,1]

                alpha_hat_list.append(alpha_hat)

                eps_bar = np.array([[D_KL_result.x[0]], [D_KL_result.x[1]], [0.], [0.]])
                eps_bar_list[k, N] = eps_bar
                eps_barbar_inv_list[k, N] = (np.array([[1/(D_KL_result.x[2])**2,0,0,0], [0,1/(D_KL_result.x[3])**2,0,0], [0,0,0,0], [0,0,0,0]]))
                #intermediate = [x0[0], x0[1], D_KL_result.x[0], D_KL_result.x[1]]
                #D_KL_result = minimize(mrblat_functions_list[k].D_KL, D_KL_result.x, bounds = bound,  args=(data_fourier, intermediate, (0,0,0,0), (1,1,1,1), False), method='powell')
                #print("powell:", D_KL_result.x)

                if N in heatmap_list: # 
                    colors = ['red', 'purple', 'green', 'cyan', 'magenta']

                    ### HEAT MAP ###
                    heatmap_res = 101

                    heatmap_pos = np.zeros((heatmap_res, heatmap_res))
                    heatmap_var = np.zeros((heatmap_res, heatmap_res))

                    heatmap_pos_x = np.linspace(-22.48,22.48, heatmap_res)
                    heatmap_pos_y = np.linspace(1,22.48, heatmap_res)
                    heatmap_var_x = 10**(np.linspace(-200, 40, heatmap_res)/20)
                    heatmap_var_y = 10**(np.linspace(-200, 40, heatmap_res)/20)

                    for i in range(heatmap_res):
                        for j in range(heatmap_res):
                            heatmap_pos[i,j] = mrblat_functions_list[k].D_KL([heatmap_pos_x[j], heatmap_pos_y[i], x0k[2], x0k[3]], data_fourier, x0k, (1,1,1,1), (1,1,1,1), False)
                            if np.isnan(heatmap_pos[i,j]):
                                heatmap_pos[i,j] = np.inf
                            if np.sqrt(heatmap_pos_x[j]**2 + heatmap_pos_y[i]**2) > 22.4:
                                heatmap_pos[i,j] = np.inf
                    argmin_heatmap_pos = np.unravel_index(np.argmin(heatmap_pos, axis=None), heatmap_pos.shape)

                    for i in range(heatmap_res):
                        for j in range(heatmap_res):
                            heatmap_var[i,j] = mrblat_functions_list[k].D_KL([D_KL_result.x[0] - self.__radar_parameters[k]["position"][0,0], D_KL_result.x[1] - self.__radar_parameters[k]["position"][0,1], heatmap_var_x[j], heatmap_var_y[i]], data_fourier, x0k, (1,1,1,1), (1,1,1,1), False)
                    argmin_heatmap_var = np.unravel_index(np.argmin(heatmap_var, axis=None), heatmap_var.shape)

                    intermediate = [x0k[0], x0k[1], heatmap_pos_x[argmin_heatmap_pos[1]], heatmap_pos_y[argmin_heatmap_pos[0]]]
                    initial_guess = [heatmap_pos_x[argmin_heatmap_pos[1]], heatmap_pos_y[argmin_heatmap_pos[0]], heatmap_var_x[argmin_heatmap_var[1]], heatmap_var_y[argmin_heatmap_var[0]]]
                    D_KL_nelder_mead = minimize(mrblat_functions_list[k].D_KL, initial_guess, bounds = bound,  args=(data_fourier, intermediate, (0,0,1,1), (1,1,1,1), False), method='Nelder-mead', tol=1e-10)
                    D_KL_DE = differential_evolution(mrblat_functions_list[k].D_KL, bound, args=(data_fourier, intermediate, (1,1,1,1), (1,1,1,1), False), maxiter=1000, popsize=10, disp=False)
                    D_KL_L_BFGS_B = minimize(mrblat_functions_list[k].D_KL, initial_guess, bounds = bound,  args=(data_fourier, intermediate, (0,0,1,1), (1,1,1,1), False), method='L-BFGS-B', tol=1e-10)

                    fig, ax = plt.subplots(1, 1)
                    p0 = ax.pcolormesh(heatmap_pos_x, heatmap_pos_y, heatmap_pos, shading='auto', cmap='viridis')
                    fig.colorbar(p0, ax=ax)
                    ax.scatter(heatmap_pos_x[argmin_heatmap_pos[1]], heatmap_pos_y[argmin_heatmap_pos[0]], color='red', marker='x', s=100, label='ARGMIN')
                    # print("ARGMIN:", heatmap_pos_x[argmin_heatmap_pos[1]], heatmap_pos_y[argmin_heatmap_pos[0]])
                    # ax.scatter(x0k[0], x0k[1], color='blue', marker='x', s=100, label='INITIAL')
                    ax.scatter(D_KL_result.x[0] - self.__radar_parameters[k]["position"][0,0], D_KL_result.x[1] - self.__radar_parameters[k]["position"][0,1], color='green', marker='x', s=100, label='Optimised KL divergence')
                    # ax.scatter(D_KL_DE.x[0], D_KL_DE.x[1], color='purple', marker='+', s=100, label='DE - POS')
                    radar_point_1 = ax.scatter(self.__radar_parameters[k]["position"][0,0], self.__radar_parameters[k]["position"][0,1], c=colors[k], s=100, marker='o', zorder=3)
                    radar_point_2 = ax.scatter(self.__radar_parameters[k]["position"][0,0], self.__radar_parameters[k]["position"][0,1], c='white', s=100, marker='1', zorder=3)
                    radar_point_1.set_clip_on(False)
                    radar_point_2.set_clip_on(False)
                    
                    # ax.set_title(f'Position Heatmap (frame {N})')
                    ax.set_xlabel('$x$ [m]')
                    ax.set_ylabel('$y$ [m]')

                    plt.legend(loc='upper center', ncol=2)
                    plt.tight_layout()
                    plt.savefig(f'Figures/KL_mrblat_heatmap_pos_frame_{N}_radar_{k}.pdf')
                    plt.savefig(f'Figures/KL_mrblat_heatmap_pos_frame_{N}_radar_{k}.jpg')
                    plt.show()
                    

                    fig, ax = plt.subplots(1, 1)
                    p1 = ax.pcolormesh(heatmap_var_x, heatmap_var_y, 20*np.log10(heatmap_var-np.min(np.min(heatmap_var))), shading='auto', cmap='viridis')
                    fig.colorbar(p1, ax=ax)
                    ax.scatter(heatmap_var_x[argmin_heatmap_var[1]], heatmap_var_y[argmin_heatmap_var[0]], color='red', marker='x', s=100, label='ARGMIN')
                    # ax.scatter(x0[2], x0[3], color='blue', marker='x', s=100, label='INITIAL')
                    ax.scatter(D_KL_result.x[2], D_KL_result.x[3], color='green', marker='x', s=100, label='Optimised KL divergence')
                    # ax.scatter(D_KL_nelder_mead.x[2], D_KL_nelder_mead.x[3], color='yellow', marker='+', s=100, label='NELDER-MEAD - VAR')
                    # ax.scatter(D_KL_DE.x[2], D_KL_DE.x[3], color='purple', marker='+', s=100, label='DE - VAR')
                    # ax.scatter(D_KL_L_BFGS_B.x[2], D_KL_L_BFGS_B.x[3], color='orange', marker='+', s=100, label='L-BFGS-B - VAR')
                    # ax.set_title(f'Variance Heatmap (frame {N} - \sigma_x = {np.round(heatmap_var_x[argmin_heatmap_var[1]],6)} , \sigma_y = {np.round(heatmap_var_y[argmin_heatmap_var[0]],6)})')
                    ax.set_xlabel('$\sigma_x$ [m]')
                    ax.set_ylabel('$\sigma_y$ [m]')
                    ax.set_xscale('log')
                    ax.set_yscale('log')

                    plt.legend(loc='lower center', ncol=2)
                    plt.tight_layout()

                    # plt.savefig(f'Figures/KL_mrblat_heatmap_std_frame_{N}_radar_{k}.png')
                    plt.savefig(f'Figures/KL_mrblat_heatmap_std_frame_{N}_radar_{k}.pdf')
                    plt.savefig(f'Figures/KL_mrblat_heatmap_std_frame_{N}_radar_{k}.jpg')
                    plt.show()
                    

                    ### HEATY MAP END ###

            if N == -1:
                res = mrblat_functions_list[0].jacobian_S_H([0,10], print_output=True)
                plt.figure()
                plt.plot(np.abs(res[:,0]))
                plt.show()

                plt.figure()
                plt.plot(np.abs(res[:,1]))
                plt.show()


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
                    elif n == 0:#N - fifo_counter:
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
            
            intermediate_reult_pos.append(phi_bar_list.copy())
            intermediate_reult_var.append(phi_barbar_list.copy())

            PHI_NEXT = T@phi_bar_list[N]

            x0 = [PHI_NEXT[0,0], PHI_NEXT[1,0], np.sqrt(phi_barbar_list[N,0,0]), np.sqrt(phi_barbar_list[N,1,1])]

            #x0 = [phi_bar_list[N,0,0], phi_bar_list[N,1,0], np.sqrt(phi_barbar_list[N,0,0]), np.sqrt(phi_barbar_list[N,1,1])]
            alpha_hat_S_list.append(alpha_hat*mrblat_functions_list[k].get_S_signal(x0[0], x0[1]))
            if fifo_counter < fifo_length-1:
                fifo_counter += 1
        return phi_bar_list, phi_barbar_list, alpha_hat_list, alpha_hat_S_list, intermediate_reult_pos, intermediate_reult_var
    
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
