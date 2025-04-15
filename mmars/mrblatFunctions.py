import numpy as np
#import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from scipy.constants import c

from .target import Target
from .simulation import Simulation


class MRBLaT_Functions_new(Simulation):
    
    def __init__(self, radar_model, target_model): 
        super().__init__(radar_model, target_model)
        self.__radar_setup = radar_model

        self.__flatten_data_size = self.__radar_setup.get_IF_signal.shape[0]* self.__radar_setup.get_IF_signal.shape[1] * self.__radar_setup.get_IF_signal.shape[3]
        self.__standardDeviation = self.__radar_setup.get_standardDeviation
        self.__position = np.array([0, 0])
        self.__tx_antennas = self.__radar_setup.get_tx_antennas
        self.__rx_antennas = self.__radar_setup.get_rx_antennas
        self.__transmitPower = self.__radar_setup.get_transmitPower
        self.__gain = self.__radar_setup.get_gain
        self.__radarCrossSection = self.__radar_setup.get_radarCrossSection
        self.__wavelength = self.__radar_setup.get_wavelength
        self.__N_samples = self.__radar_setup.get_N_samples
        self.__chirp_Rate = self.__radar_setup.get_chirp_Rate
        self.__f_sampling = self.__radar_setup.get_f_sampling

        self.__lambda_z = np.eye(self.__flatten_data_size) * (self.__standardDeviation)**(-2)
        self.__lambda_z = csr_matrix(self.__lambda_z)

        self.__freqs = np.linspace(0, self.__N_samples, self.__N_samples, endpoint=False)[np.newaxis]
        self.__ds = np.arange(0, len(self.__tx_antennas) * len(self.__rx_antennas) * self.__wavelength/2, self.__wavelength/2)  # should be dynamic
        self.__A = np.sqrt(self.__transmitPower * self.__gain * self.__radarCrossSection * self.__wavelength**2 / (4 * np.pi)**3)
        self.__x_r = self.__position[0]
        self.__y_r = self.__position[1]

    def alpha_hat(self, s_n, data_fourier):
        # Compute conjugate transpose once
        s_n_H = s_n.conj().T  # Conjugate transpose
        return (s_n_H @ self.__lambda_z @ data_fourier) / (s_n_H @ self.__lambda_z @ s_n)
    
    def jacobian_S(self, epsilon):
            
        x, y = epsilon
        S_tilde = self.steering_matrix(x, y) @ self.sinc(x, y)

        partial_S_tilde_x = self.partial_steering_matrix_x(x, y) @ self.sinc(x, y) + self.steering_matrix(x, y) @ self.partial_sinc_x(x, y)
        S_jacobian_x = (partial_S_tilde_x * self.path_loss(x, y) + S_tilde * self.partial_path_loss_x(x, y)).flatten()[:, np.newaxis]

        partial_S_tilde_y = self.partial_steering_matrix_y(x, y) @ self.sinc(x, y) + self.steering_matrix(x, y) @ self.partial_sinc_y(y, x)

        S_jacobian_y = (partial_S_tilde_y * self.path_loss(x, y) + S_tilde * self.partial_path_loss_y(y, x)).flatten()[:, np.newaxis]

        S_jacobian = np.hstack((S_jacobian_x, S_jacobian_y))
        return S_jacobian

    def path_loss(self, x, y):
        r = np.sqrt((x - self.__x_r)**2 + (y - self.__y_r)**2)
        alpha = self.__A/r**2
        return alpha

    def partial_path_loss_x(self, x, y):
        r = np.sqrt((x - self.__x_r)**2 + (y - self.__y_r)**2)
        partial_alpha = - 2 * self.__A * r**(-4) * (x - self.__x_r)
        return partial_alpha

    def partial_path_loss_y(self, x, y):
        r = np.sqrt((x - self.__x_r)**2 + (y - self.__y_r)**2)
        partial_alpha = - 2 * self.__A * r**(-4) * (y - self.__y_r)
        return partial_alpha

    def steering_matrix(self, x, y):
        deltaR = np.sin(-np.arctan2(x - self.__x_r, y - self.__y_r)) * self.__ds
        phi = deltaR / self.__wavelength
        steering_mat = np.exp(1.j * 2 * np.pi * phi)
        return steering_mat[:, np.newaxis]
    
    def partial_steering_matrix_x(self, x, y):
        exp = (x - self.__x_r)**2 + (y - self.__y_r)**2
        partial_DeltaR_x = - (y - self.__y_r) * self.__ds / (np.sqrt(exp/(y - self.__y_r)**2) * exp)

        partial_phi_DeltaR = 1 / self.__wavelength

        DeltaR = np.sin(-np.arctan2(x - self.__x_r, y - self.__y_r)) * self.__ds
        phi = DeltaR / self.__wavelength
        partial_A_phi = 1.j * 2 * np.pi * np.exp(1.j * 2 * np.pi * phi)

        partial_A_x = partial_A_phi * partial_phi_DeltaR * partial_DeltaR_x
        return partial_A_x[:, np.newaxis]
    
    def partial_steering_matrix_y(self, x, y):
        exp = (x - self.__x_r)**2 + (y - self.__y_r)**2
        partial_DeltaR_y =  (x - self.__x_r) * self.__ds / (np.sqrt(exp/(y - self.__y_r)**2) * exp)

        partial_phi_DeltaR = 1 / self.__wavelength

        DeltaR = np.sin(-np.arctan2(x - self.__x_r, y - self.__y_r)) * self.__ds
        phi = DeltaR / self.__wavelength
        partial_A_phi = 1.j * 2 * np.pi * np.exp(1.j * 2 * np.pi * phi)

        partial_A_y = partial_A_phi * partial_phi_DeltaR * partial_DeltaR_y
        print(f"partial_A_y: {partial_A_y}")
        return partial_A_y[:, np.newaxis]
    
    def sinc(self, x, y):
        r = np.sqrt((x - self.__x_r)**2 + (y - self.__y_r)**2)
        f_IF = 2 * self.__chirp_Rate * r / c
        temp = 2 * np.pi * (f_IF/self.__f_sampling - self.__freqs/self.__N_samples)
        K = np.exp(1.j * (self.__N_samples - 1) * temp / 2)
        sinc_fnc = K * np.sin(self.__N_samples * temp / 2) / np.sin(temp / 2)
        return sinc_fnc/self.__N_samples
    
    def partial_sinc_x(self, x, y):
        N_s = self.__N_samples
        S = self.__chirp_Rate
        F_s = self.__f_sampling
        X_R = self.__x_r
        Y_R = self.__y_r
        j = 1.j
        f = self.__freqs

        # from Maple:
        partial_sinc = N_s * np.pi * S * ((x - X_R) ** 2 + (y - Y_R) ** 2) ** (-0.1e1 / 0.2e1) / c / F_s * (2 * x - 2 * X_R) * np.cos(N_s * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) / np.sin(np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) * np.exp(j * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s) * (N_s - 1)) - np.sin(N_s * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) / np.sin(np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) ** 2 * np.exp(j * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s) * (N_s - 1)) * np.pi * S * ((x - X_R) ** 2 + (y - Y_R) ** 2) ** (-0.1e1 / 0.2e1) / c / F_s * (2 * x - 2 * X_R) * np.cos(np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) + np.sin(N_s * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) / np.sin(np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) * j * np.pi * S * ((x - X_R) ** 2 + (y - Y_R) ** 2) ** (-0.1e1 / 0.2e1) / c / F_s * (2 * x - 2 * X_R) * (N_s - 1) * np.exp(j * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s) * (N_s - 1))
        return partial_sinc/self.__N_samples
    
    def partial_sinc_y(self, x, y):
        N_s = self.__N_samples
        S = self.__chirp_Rate
        F_s = self.__f_sampling
        X_R = self.__x_r
        Y_R = self.__y_r
        j = 1.j
        f = self.__freqs

        # from Maple:
        partial_sinc = N_s * np.pi * S * ((x - X_R) ** 2 + (y - Y_R) ** 2) ** (-0.1e1 / 0.2e1) / c / F_s * (2 * y - 2 * Y_R) * np.cos(N_s * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) / np.sin(np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) * np.exp(j * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s) * (N_s - 1)) - np.sin(N_s * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) / np.sin(np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) ** 2 * np.exp(j * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s) * (N_s - 1)) * np.pi * S * ((x - X_R) ** 2 + (y - Y_R) ** 2) ** (-0.1e1 / 0.2e1) / c / F_s * (2 * y - 2 * Y_R) * np.cos(np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) + np.sin(N_s * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) / np.sin(np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) * j * np.pi * S * ((x - X_R) ** 2 + (y - Y_R) ** 2) ** (-0.1e1 / 0.2e1) / c / F_s * (2 * y - 2 * Y_R) * (N_s - 1) * np.exp(j * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s) * (N_s - 1))
        return partial_sinc/self.__N_samples
    
    def D_KL(self, params, Z_data, phi_bar_last_x, phi_bar_last_y, outputmode=(1,1,1,1), print_output=False):

        eps_bar_x, eps_bar_y, eps_barbar_0, eps_barbar_1 = params

        # Last estimate of the trajectory
        self.__radar_setup.generate_S_signal(phi_bar_last_x, phi_bar_last_y)
        S_N_lack = self.__radar_setup.get_S_signal.flatten()[:, np.newaxis]/np.sqrt(256)

        # Generate the S signal with the new parameters
        self.__radar_setup.generate_S_signal(eps_bar_x, eps_bar_y)
        s_n = self.__radar_setup.get_S_signal.flatten()[:, np.newaxis]/np.sqrt(256)
        
        # Compute the alpha_hat value
        alpha_hat_xy = np.abs(self.alpha_hat(S_N_lack, Z_data))

        s_n_H = s_n.conj().T

        term_1 = -np.real(alpha_hat_xy * (s_n_H @ self.__lambda_z @ Z_data))
        term_2 = np.real(np.abs(alpha_hat_xy)**2 * s_n_H @ self.__lambda_z @ s_n)
        
        jac = self.jacobian_S(np.array([eps_bar_x, eps_bar_y]))
        term_3_inner_prod = np.real(jac.conj().T @ self.__lambda_z @ jac)
        
        term_3 = np.abs(alpha_hat_xy)**2 * np.trace(np.array([[eps_barbar_0, 0], [0, eps_barbar_1]]) @ term_3_inner_prod)
        k = 2
        entropy = k/2 * np.log(2*np.pi*np.e) + 1/2*np.log(eps_barbar_0 * eps_barbar_1)
        
        if print_output:
            print(term_1, term_2, term_3, entropy)

        return outputmode[0] * term_1 + outputmode[1] * term_2 + outputmode[2] * term_3 - outputmode[3] * entropy
    
    def run_tracking(self):
        pass