import numpy as np

from scipy.sparse import csr_matrix
from scipy.constants import c

from .target import Target
from .fmcwRadar import FmcwRadar


class MRBLaT_Functions(FmcwRadar):
    
    def __init__(self): 
        super().__init__()

        self.__flatten_data_size = self.get_IF_signal.shape[0]* self.get_IF_signal.shape[1] * self.get_IF_signal.shape[3]
        self.__standardDeviation = self.get_standardDeviation
        self.__position = np.array([0, 0])
        self.__tx_antennas = self.get_tx_antennas
        self.__rx_antennas = self.get_rx_antennas
        self.__transmitPower = self.get_transmitPower
        self.__gain = self.get_gain
        self.__radarCrossSection = self.get_radarCrossSection
        self.__wavelength = self.get_wavelength
        self.__N_samples = self.get_N_samples
        self.__chirp_Rate = self.get_chirp_Rate
        self.__f_sampling = self.get_f_sampling

        self.__lambda_z = np.eye(self.__flatten_data_size) * (self.__standardDeviation)**(-2)
        self.__lambda_z = csr_matrix(self.__lambda_z)

        self.__freqs = np.linspace(0, 2 * np.pi, self.__N_samples)[np.newaxis]
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
        partial_S_tilde_x = self.partial_steering_matrix(x, y, x_partial=True) @ self.sinc(x, y) + self.steering_matrix(x, y) @ self.partial_sinc(x, y)

        S_jacobian_x = (partial_S_tilde_x * self.path_loss(x, y) + S_tilde * self.partial_path_loss(x, y)).flatten()[:, np.newaxis]

        partial_S_tilde_y = self.partial_steering_matrix(x, y, x_partial=False) @ self.sinc(x, y) + self.steering_matrix(x, y) @ self.partial_sinc(y, x)
        S_jacobian_y = (partial_S_tilde_y * self.path_loss(x, y) + S_tilde * self.partial_path_loss(y, x)).flatten()[:, np.newaxis]

        S_jacobian = np.hstack((S_jacobian_x, S_jacobian_y))
        return S_jacobian

    def path_loss(self, x, y):
        """path loss"""
        r = np.sqrt((x - self.__x_r)**2 + (y - self.__y_r)**2)
        alpha = self.__A/r**2
        return alpha

    def partial_path_loss(self, x, y):
        """path loss alpha partial differentiated w.r.t first entry (works for x/y)"""
        r = np.sqrt((x - self.__x_r)**2 + (y - self.__y_r)**2)
        partial_alpha = - 2 * self.__A * r**(-4) * (x - self.__x_r)
        return partial_alpha

    def steering_matrix(self, x, y):
        """steering matrix"""
        deltaR = np.sin(-np.atan2(x - self.__x_r, y - self.__y_r)) * self.__ds
        phi = deltaR / self.__wavelength
        steering_mat = np.exp(1.j * 2 * np.pi * phi)
        return steering_mat[:, np.newaxis]

    def partial_steering_matrix(self, x, y, x_partial=True):
        """steering matrix partial differentiated w.r.t x/y"""
        exp1 = 1 + (x - self.__x_r)**2/(y - self.__y_r)**2
        partial_deltaR = (-(1/((y - self.__y_r) * np.sqrt(exp1))) + (x - self.__x_r)**2/((y - self.__y_r)**3 * exp1**(3/2))) * self.__ds
        
        if x_partial == False:
            partial_deltaR = ((x - self.__x_r)/((y - self.__y_r)**2 * np.sqrt(exp1)) - ((x - self.__x_r)**3/((y - self.__y_r)**4 * exp1**(3/2)))) * self.__ds

        partial_phi_deltaR = 1/self.__wavelength

        deltaR = np.sin(-np.atan2(x, y)) * self.__ds
        phi = deltaR / self.__wavelength
        partial_a_phi = 1.j * 2 * np.pi * np.exp(1.j * 2 * np.pi * phi)
        partial_steering_mat = partial_a_phi * partial_phi_deltaR * partial_deltaR
        return partial_steering_mat[:, np.newaxis]
        
    def sinc(self, x, y):
        """sinc function"""
        r = np.sqrt((x - self.__x_r)**2 + (y - self.__y_r)**2)
        f_IF = 2 * self.__chirp_Rate * r / c * (2 * np.pi / self.__f_sampling)
        K = np.exp(-1.j * self.__freqs * self.__N_samples/2)

        sinc_fnc = K * np.sin((self.__freqs - f_IF) * (self.__N_samples + 1) * 1/2)/np.sin((self.__freqs - f_IF)/2)
        return sinc_fnc
    
    def partial_sinc(self, x, y):
        """sinc fnc (fourier transform of window fnc) partially differentiated w.r.t. first entry"""
        K = np.exp(-1.j * self.__freqs * self.__N_samples/2)
        f = self.__freqs

        # from Maple:
        partial_sinc = (K * self.__chirp_Rate * ((x - self.__x_r) ** 2 + (y - self.__y_r) ** 2) ** (-0.1e1 / 0.2e1) / c * np.pi * (2 * x - 2 * self.__x_r) * (self.__N_samples + 1) 
                        * np.cos((f - self.__chirp_Rate * np.sqrt((x - self.__x_r) ** 2 + (y - self.__y_r) ** 2) / c * np.pi / 5000000) * (self.__N_samples + 1) / 2) 
                        / np.sin(-f / 2 + self.__chirp_Rate * np.sqrt((x - self.__x_r) ** 2 + (y - self.__y_r) ** 2) / c * np.pi / 10000000) / 20000000 + K 
                        * np.sin((f - self.__chirp_Rate * np.sqrt((x - self.__x_r) ** 2 + (y - self.__y_r) ** 2) / c * np.pi / 5000000) * (self.__N_samples + 1) / 2) 
                        / np.sin(-f / 2 + self.__chirp_Rate * np.sqrt((x - self.__x_r) ** 2 + (y - self.__y_r) ** 2) / c * np.pi / 10000000) ** 2 * self.__chirp_Rate 
                        * ((x - self.__x_r) ** 2 + (y - self.__y_r) ** 2) ** (-0.1e1 / 0.2e1) / c * np.pi * (2 * x - 2 * self.__x_r) 
                        * np.cos(-f / 2 + self.__chirp_Rate * np.sqrt((x - self.__x_r) ** 2 + (y - self.__y_r) ** 2) / c * np.pi / 10000000) / 20000000
                        )
        return partial_sinc


def D_KL(params, Z_data, phi_bar_last_x, phi_bar_last_y, Lambda_z, radar_model_1, mrblatcomp, outputmode=(1,1,1,1), print_output=False):

    eps_bar_x, eps_bar_y, eps_barbar_0, eps_barbar_1 = params

    # Last estimate of the trajectory
    radar_model_1.generate_S_signal(phi_bar_last_x, phi_bar_last_y)
    S_N_lack = radar_model_1.get_S_signal.flatten()[:, np.newaxis]/np.sqrt(256)

    # Generate the S signal with the new parameters
    radar_model_1.generate_S_signal(eps_bar_x, eps_bar_y)
    s_n = radar_model_1.get_S_signal.flatten()[:, np.newaxis]/np.sqrt(256)
    
    # Compute the alpha_hat value
    alpha_hat_xy = np.abs(mrblatcomp.alpha_hat(S_N_lack, Z_data))

    s_n_H = s_n.conj().T

    term_1 = -np.real(alpha_hat_xy * (s_n_H @ Lambda_z @ Z_data))
    term_2 = np.real(np.abs(alpha_hat_xy)**2 * s_n_H @ Lambda_z @ s_n)
    
    jac = mrblatcomp.jacobian_S(np.array([eps_bar_x, eps_bar_y]))
    term_3_inner_prod = np.real(jac.conj().T @ Lambda_z @ jac)
    
    term_3 = np.abs(alpha_hat_xy)**2 * np.trace(np.array([[eps_barbar_0, 0], [0, eps_barbar_1]]) @ term_3_inner_prod)
    k = 2
    entropy = k/2 * np.log(2*np.pi*np.e) + 1/2*np.log(eps_barbar_0 * eps_barbar_1)
    
    if print_output:
        print(term_1, term_2, term_3, entropy)

    return outputmode[0] * term_1 + outputmode[1] * term_2 + outputmode[2] * term_3 - outputmode[3] * entropy