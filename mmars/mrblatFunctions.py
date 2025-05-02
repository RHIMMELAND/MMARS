import numpy as np
#import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from scipy.constants import c

from .fmcwRadar import FmcwRadar

from numba import njit

class MRBLaT_Functions():
    
    def __init__(self, radar_parameters): 

        self.__standardDeviation = radar_parameters["standardDeviation"]
        self.__position = radar_parameters["position"]
        self.__tx_antennas = radar_parameters["tx_antennas"]
        self.__rx_antennas = radar_parameters["rx_antennas"]
        self.__transmitPower = radar_parameters["transmitPower"]
        self.__gain = radar_parameters["gain"]
        self.__radarCrossSection = radar_parameters["radarCrossSection"]
        self.__wavelength = radar_parameters["wavelength"]
        self.__N_samples = radar_parameters["N_samples"]
        self.__chirp_Rate = radar_parameters["chirp_Rate"]
        self.__f_sampling = radar_parameters["f_sampling"]
        self.__flatten_data_size = len(self.__tx_antennas) * len(self.__rx_antennas) * self.__N_samples
        
        self.__lambda_z = np.eye(self.__flatten_data_size) * (self.__standardDeviation)**(-2)
        self.__lambda_z = csr_matrix(self.__lambda_z)

        self.__freqs = np.linspace(0, self.__N_samples, self.__N_samples, endpoint=False)[np.newaxis]
        self.__ds = np.arange(0, len(self.__tx_antennas) * len(self.__rx_antennas) * self.__wavelength/2, self.__wavelength/2)  # should be dynamic
        self.__A = np.sqrt(self.__transmitPower * self.__gain * self.__radarCrossSection * self.__wavelength**2 / (4 * np.pi)**3)
        self.__x_r = self.__position[0,0]
        self.__y_r = self.__position[0,1]

        self.__radar_setup = FmcwRadar(self.__position, self.__tx_antennas-self.__position, self.__rx_antennas-self.__position, self.__chirp_Rate, 1, radar_parameters["f_carrier"], self.__N_samples, self.__f_sampling, 1, self.__transmitPower, self.__gain, self.__radarCrossSection, [0.1,0.1]) 

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
    
    def jacobian_S_ez(self, epsilon):
        x, y = epsilon
        R = np.sqrt((x - self.__x_r)**2 + (y - self.__y_r)**2)
        dR__dx = (x - self.__x_r)/R
        dR__dy = (y - self.__y_r)/R

        R_RT = np.zeros((len(self.__tx_antennas), len(self.__rx_antennas)))
        dR_RT__dx = np.zeros((len(self.__tx_antennas), len(self.__rx_antennas)))
        dR_RT__dy = np.zeros((len(self.__tx_antennas), len(self.__rx_antennas)))
        for tx_idx in range(len(self.__tx_antennas)):
            for rx_idx in range(len(self.__rx_antennas)):
                R_RT[tx_idx,rx_idx] = np.sqrt((x - self.__tx_antennas[tx_idx,0])**2 + (y - self.__tx_antennas[tx_idx,1])**2) + np.sqrt((x - self.__rx_antennas[rx_idx,0])**2 + (y - self.__rx_antennas[rx_idx,1])**2)
                dR_RT__dx[tx_idx,rx_idx] = (x - self.__tx_antennas[tx_idx,0])**2/np.sqrt((x - self.__tx_antennas[tx_idx,0])**2 + (y - self.__tx_antennas[tx_idx,1])**2) + (x - self.__rx_antennas[rx_idx,0])**2/np.sqrt((x - self.__rx_antennas[rx_idx,0])**2 + (y - self.__rx_antennas[rx_idx,1])**2)
                dR_RT__dy[tx_idx,rx_idx] = (y - self.__tx_antennas[tx_idx,1])**2/np.sqrt((x - self.__tx_antennas[tx_idx,0])**2 + (y - self.__tx_antennas[tx_idx,1])**2) + (y - self.__rx_antennas[rx_idx,1])**2/np.sqrt((x - self.__rx_antennas[rx_idx,0])**2 + (y - self.__rx_antennas[rx_idx,1])**2)
        
        phi = (2*np.pi/self.__wavelength)*R_RT
        dphi__dx = (2*np.pi/self.__wavelength)*dR_RT__dx
        dphi__dy = (2*np.pi/self.__wavelength)*dR_RT__dy

        f_if = (2 * self.__chirp_Rate / c) * R
        df_if__dx = (2 * self.__chirp_Rate / c) * dR__dx
        df_if__dy = (2 * self.__chirp_Rate / c) * dR__dy

        T_E = 2*np.pi * (f_if/self.__f_sampling - self.__freqs/self.__N_samples)
        dT_E__dx = (2*np.pi/self.__f_sampling) * df_if__dx
        dT_E__dy = (2*np.pi/self.__f_sampling) * df_if__dy

        K = np.exp(1.j * (self.__N_samples - 1) * T_E / 2)
        dK__dx = ((1.j*self.__N_samples-1)/2) * np.exp(1.j * (self.__N_samples - 1) * T_E / 2) * dT_E__dx 
        dK__dy = ((1.j*self.__N_samples-1)/2) * np.exp(1.j * (self.__N_samples - 1) * T_E / 2) * dT_E__dy

        sin_expr = np.sin(self.__N_samples * T_E / 2) / np.sin(T_E / 2)
        dsin_expr__dx = (
            (self.__N_samples/2)*np.cos((self.__N_samples/2) * T_E )*dT_E__dx * np.sin(T_E / 2) 
            - np.sin(self.__N_samples * T_E / 2)*(1/2)*np.cos(T_E / 2)*dT_E__dx
            ) / ((np.sin(T_E / 2))**2)
        dsin_expr__dy = (
            (self.__N_samples/2)*np.cos((self.__N_samples/2) * T_E )*dT_E__dy * np.sin(T_E / 2) 
            - np.sin(self.__N_samples * T_E / 2)*(1/2)*np.cos(T_E / 2)*dT_E__dy
            ) / ((np.sin(T_E / 2))**2)
        
        

        

    def path_loss(self, x, y):
        return path_loss_speed(x, y, self.__x_r, self.__y_r, self.__A)

    def partial_path_loss_x(self, x, y):
        return partial_path_loss_x_speed(x, y, self.__x_r, self.__y_r, self.__A)

    def partial_path_loss_y(self, x, y):
        return partial_path_loss_y_speed(x, y, self.__x_r, self.__y_r, self.__A)

    def steering_matrix(self, x, y):
        return steering_matrix_speed(x, y, self.__x_r, self.__y_r, self.__ds, self.__wavelength)
    
    def partial_steering_matrix_x(self, x, y):
        return partial_steering_matrix_x_speed(x, y, self.__x_r, self.__y_r, self.__ds, self.__wavelength)
    
    def partial_steering_matrix_y(self, x, y):
        return partial_steering_matrix_y_speed(x, y, self.__x_r, self.__y_r, self.__ds, self.__wavelength)
    
    def sinc(self, x, y):
        return sinc_speed(x, y, self.__x_r, self.__y_r, self.__N_samples, self.__chirp_Rate, self.__f_sampling, self.__freqs)
    
    
    def partial_sinc_x(self, x, y):
        return partial_sinc_x_speed(x, y, self.__x_r, self.__y_r, self.__N_samples, self.__chirp_Rate, self.__f_sampling, self.__freqs)
    
    def partial_sinc_y(self, x, y):
        return partial_sinc_y_speed(x, y, self.__x_r, self.__y_r, self.__N_samples, self.__chirp_Rate, self.__f_sampling, self.__freqs)
    
    def D_KL(self, params, Z_data, phi_bar_last_x, phi_bar_last_y, outputmode=(1,1,1,1), print_output=False):

        eps_bar_x, eps_bar_y, eps_barbar_0, eps_barbar_1 = params

        # Last estimate of the trajectory
        self.__radar_setup.generate_S_signal(phi_bar_last_x, phi_bar_last_y)
        S_N_lack = self.__radar_setup.get_S_signal.flatten()[:, np.newaxis]

        # Generate the S signal with the new parameters
        self.__radar_setup.generate_S_signal(eps_bar_x, eps_bar_y)
        s_n = self.__radar_setup.get_S_signal.flatten()[:, np.newaxis]
        
        # Compute the alpha_hat value
        alpha_hat_xy = np.abs(self.alpha_hat(S_N_lack, Z_data))

        s_n_H = s_n.conj().T

        term_1 = -np.abs(alpha_hat_xy * (s_n_H @ self.__lambda_z @ Z_data))
        term_2 = np.real(np.abs(alpha_hat_xy)**2 * s_n_H @ self.__lambda_z @ s_n)
        
        jac = self.jacobian_S(np.array([eps_bar_x, eps_bar_y]))
        term_3_inner_prod = np.real(jac.conj().T @ self.__lambda_z @ jac)
        
        term_3 = np.abs(alpha_hat_xy)**2 * np.trace(np.array([[eps_barbar_0, 0], [0, eps_barbar_1]]) @ term_3_inner_prod)
        k = 2
        entropy = k/2 * np.log(2*np.pi*np.e) + 1/2*np.log(eps_barbar_0 * eps_barbar_1)
        
        if print_output:
            print(term_1, term_2, term_3, entropy)

        return outputmode[0] * term_1 + outputmode[1] * term_2 + outputmode[2] * term_3 - outputmode[3] * entropy

# @njit
def path_loss_speed(x, y, x_r, y_r, A):
    r = np.sqrt((x - x_r)**2 + (y - y_r)**2)
    alpha = A/r**2
    return alpha

# @njit
def partial_path_loss_x_speed(x, y, x_r, y_r, A):
    r = np.sqrt((x - x_r)**2 + (y - y_r)**2)
    partial_alpha = - 2 * A * r**(-4) * (x - x_r)
    return partial_alpha

# @njit
def partial_path_loss_y_speed(x, y, x_r, y_r, A):
    r = np.sqrt((x - x_r)**2 + (y - y_r)**2)
    partial_alpha = - 2 * A * r**(-4) * (y - y_r)
    return partial_alpha

# @njit
def steering_matrix_speed(x, y, x_r, y_r, ds, wavelength):
    deltaR = np.sin(-np.arctan2(x - x_r, y - y_r)) * ds
    phi = deltaR / wavelength
    steering_mat = np.exp(1.j * 2 * np.pi * phi)
    return steering_mat[:, np.newaxis]

# @njit
def partial_steering_matrix_x_speed(x, y, x_r, y_r, ds, wavelength):
    exp = (x - x_r)**2 + (y - y_r)**2
    partial_DeltaR_x = - (y - y_r) * ds / (np.sqrt(exp/(y - y_r)**2) * exp)

    partial_phi_DeltaR = 1 / wavelength

    DeltaR = np.sin(-np.arctan2(x - x_r, y - y_r)) * ds
    phi = DeltaR / wavelength
    partial_A_phi = 1.j * 2 * np.pi * np.exp(1.j * 2 * np.pi * phi)

    partial_A_x = partial_A_phi * partial_phi_DeltaR * partial_DeltaR_x
    return partial_A_x[:, np.newaxis]

# @njit
def partial_steering_matrix_y_speed(x, y, x_r, y_r, ds, wavelength):
    exp = (x - x_r)**2 + (y - y_r)**2
    partial_DeltaR_y =  (x - x_r) * ds / (np.sqrt(exp/(y - y_r)**2) * exp)

    partial_phi_DeltaR = 1 / wavelength

    DeltaR = np.sin(-np.arctan2(x - x_r, y - y_r)) * ds
    phi = DeltaR / wavelength
    partial_A_phi = 1.j * 2 * np.pi * np.exp(1.j * 2 * np.pi * phi)

    partial_A_y = partial_A_phi * partial_phi_DeltaR * partial_DeltaR_y
    return partial_A_y[:, np.newaxis]

# @njit
def sinc_speed(x, y, x_r, y_r, N_samples, chirp_Rate, f_sampling, freqs):
    r = np.sqrt((x - x_r)**2 + (y - y_r)**2)
    f_IF = 2 * chirp_Rate * r / c
    temp = 2 * np.pi * (f_IF/f_sampling - freqs/N_samples)
    K = np.exp(1.j * (N_samples - 1) * temp / 2)
    sinc_fnc = K * np.sin(N_samples * temp / 2) / np.sin(temp / 2)
    return sinc_fnc/N_samples

# @njit
def partial_sinc_x_speed(x, y, X_R, Y_R, N_s, S, F_s, f):
    partial_sinc = N_s * np.pi * S * ((x - X_R) ** 2 + (y - Y_R) ** 2) ** (-0.1e1 / 0.2e1) / c / F_s * (2 * x - 2 * X_R) * np.cos(N_s * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) / np.sin(np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) * np.exp(1.j * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s) * (N_s - 1)) - np.sin(N_s * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) / np.sin(np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) ** 2 * np.exp(1.j * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s) * (N_s - 1)) * np.pi * S * ((x - X_R) ** 2 + (y - Y_R) ** 2) ** (-0.1e1 / 0.2e1) / c / F_s * (2 * x - 2 * X_R) * np.cos(np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) + np.sin(N_s * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) / np.sin(np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) * 1.j * np.pi * S * ((x - X_R) ** 2 + (y - Y_R) ** 2) ** (-0.1e1 / 0.2e1) / c / F_s * (2 * x - 2 * X_R) * (N_s - 1) * np.exp(1.j * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s) * (N_s - 1))
    return partial_sinc/N_s

# @njit
def partial_sinc_y_speed(x, y, X_R, Y_R, N_s, S, F_s, f):
    partial_sinc = N_s * np.pi * S * ((x - X_R) ** 2 + (y - Y_R) ** 2) ** (-0.1e1 / 0.2e1) / c / F_s * (2 * y - 2 * Y_R) * np.cos(N_s * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) / np.sin(np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) * np.exp(1.j * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s) * (N_s - 1)) - np.sin(N_s * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) / np.sin(np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) ** 2 * np.exp(1.j * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s) * (N_s - 1)) * np.pi * S * ((x - X_R) ** 2 + (y - Y_R) ** 2) ** (-0.1e1 / 0.2e1) / c / F_s * (2 * y - 2 * Y_R) * np.cos(np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) + np.sin(N_s * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) / np.sin(np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) * 1.j * np.pi * S * ((x - X_R) ** 2 + (y - Y_R) ** 2) ** (-0.1e1 / 0.2e1) / c / F_s * (2 * y - 2 * Y_R) * (N_s - 1) * np.exp(1.j * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s) * (N_s - 1))
    return partial_sinc/N_s

