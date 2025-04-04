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

    def alpha_hat(self, s_n, data_fourier):
        # Compute conjugate transpose once
        s_n_H = s_n.conj().T  # Conjugate transpose
        
        return (s_n_H @ self.__lambda_z @ data_fourier) / (s_n_H @ self.__lambda_z @ s_n)
    
    def jacobian_S(self, epsilon):
            
        N_vx_antennas = len(self.__tx_antennas) * len(self.__rx_antennas)
        x, y = epsilon
        x_r, y_r = self.__position

        A = np.sqrt(self.__transmitPower * self.__gain * self.__radarCrossSection * self.__wavelength**2 / (4 * np.pi)**3)
        ds = np.arange(0, N_vx_antennas * self.__wavelength/2, self.__wavelength/2) # should be dynamic
        freqs = np.linspace(0, 2 * np.pi, self.__N_samples)[np.newaxis]

        def path_loss(x, y):
            """path loss"""
            r = np.sqrt((x - x_r)**2 + (y - y_r)**2)
            alpha = A/r**2
            return alpha

        def partial_path_loss(x, y, x__R, y__R):
            """path loss alpha partial differentiated w.r.t first entry (works for x/y)"""
            r = np.sqrt((x - x__R)**2 + (y - y__R)**2)
            partial_alpha = - 2 * A * r**(-4) * (x - x__R)
            return partial_alpha

        def steering_matrix(x, y):
            """steering matrix"""
            deltaR = np.sin(-np.atan2(x - x_r, y - y_r)) * ds
            phi = 2 * np.pi * deltaR / self.__wavelength
            steering_mat = np.exp(1.j * 2 * np.pi * phi)
            return steering_mat[:, np.newaxis]

        def partial_steering_matrix(x, y, x__R, y__R, x_partial=True):
            """steering matrix partial differentiated w.r.t x/y"""
            exp1 = 1 + (x - x__R)**2/(y - y__R)**2
            partial_deltaR = (-(1/((y - y__R) * np.sqrt(exp1))) + (x - x__R)**2/((y - y__R)**3 * exp1**(3/2))) * ds
            
            if x_partial == False:
                partial_deltaR = ((x - x__R)/((y - y__R)**2 * np.sqrt(exp1)) - ((x - x__R)**3/((y - y__R)**4 * exp1**(3/2)))) * ds

            partial_phi_deltaR = 2 * np.pi/self.__wavelength

            deltaR = np.sin(-np.atan2(x, y)) * ds
            phi = 2 * np.pi * deltaR / self.__wavelength
            partial_a_phi = 1.j * 2 * np.pi * np.exp(1.j * 2 * np.pi * phi)
            partial_steering_mat = partial_a_phi * partial_phi_deltaR * partial_deltaR
            return partial_steering_mat[:, np.newaxis]
            
        def sinc(x, y):
            """sinc function"""
            r = np.sqrt((x - x_r)**2 + (y - y_r)**2)
            f_IF = 2 * self.__chirp_Rate * r / c * (2 * np.pi / self.__f_sampling)
            K = np.exp(-1.j * freqs * self.__N_samples/2)

            sinc_fnc = K * np.sin((freqs - f_IF) * (self.__N_samples + 1) * 1/2)/np.sin((freqs - f_IF)/2)
            return sinc_fnc
        
        def partial_sinc(x, y, x__R, y__R):
            """sinc fnc (fourier transform of window fnc) partially differentiated w.r.t. first entry"""
            K = np.exp(-1.j * freqs * self.__N_samples/2)
            f = freqs

            # from Maple:
            partial_sinc = (K * self.__chirp_Rate * ((x - x__R) ** 2 + (y - y__R) ** 2) ** (-0.1e1 / 0.2e1) / c * np.pi * (2 * x - 2 * x__R) * (self.__N_samples + 1) 
                            * np.cos((f - self.__chirp_Rate * np.sqrt((x - x__R) ** 2 + (y - y__R) ** 2) / c * np.pi / 5000000) * (self.__N_samples + 1) / 2) 
                            / np.sin(-f / 2 + self.__chirp_Rate * np.sqrt((x - x__R) ** 2 + (y - y__R) ** 2) / c * np.pi / 10000000) / 20000000 + K 
                            * np.sin((f - self.__chirp_Rate * np.sqrt((x - x__R) ** 2 + (y - y__R) ** 2) / c * np.pi / 5000000) * (self.__N_samples + 1) / 2) 
                            / np.sin(-f / 2 + self.__chirp_Rate * np.sqrt((x - x__R) ** 2 + (y - y__R) ** 2) / c * np.pi / 10000000) ** 2 * self.__chirp_Rate 
                            * ((x - x__R) ** 2 + (y - y__R) ** 2) ** (-0.1e1 / 0.2e1) / c * np.pi * (2 * x - 2 * x__R) 
                            * np.cos(-f / 2 + self.__chirp_Rate * np.sqrt((x - x__R) ** 2 + (y - y__R) ** 2) / c * np.pi / 10000000) / 20000000
                            )
            return partial_sinc

        S_tilde = steering_matrix(x, y) @ sinc(x, y)
        partial_S_tilde_x = partial_steering_matrix(x, y, x_r, y_r, x_partial=True) @ sinc(x, y) + steering_matrix(x, y) @ partial_sinc(x, y, x_r, y_r)

        S_jacobian_x = (partial_S_tilde_x * path_loss(x, y) + S_tilde * partial_path_loss(x, y, x_r, y_r)).flatten()[:, np.newaxis]

        partial_S_tilde_y = partial_steering_matrix(x, y, x_r, y_r, x_partial=False) @ sinc(x, y) + steering_matrix(x, y) @ partial_sinc(y, x, y_r, x_r)
        S_jacobian_y = (partial_S_tilde_y * path_loss(x, y) + S_tilde * partial_path_loss(y, x, y_r, x_r)).flatten()[:, np.newaxis]

        S_jacobian = np.hstack((S_jacobian_x, S_jacobian_y))
        return S_jacobian
    
def D_KL(params, Z_data, phi_bar_last_x, phi_bar_last_y):
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

    term_1 = -np.abs(alpha_hat_xy * (s_n_H @ Lambda_z @ Z_data))
    term_2 = np.real(np.abs(alpha_hat_xy)**2 * s_n_H @ Lambda_z @ s_n)
    
    jac = mrblatcomp.jacobian_S(np.array([eps_bar_x, eps_bar_y]))
    term_3_inner_prod = np.real(jac.conj().T @ Lambda_z @ jac)
    
    term_3 = np.abs(alpha_hat_xy)**2 * np.trace(np.array([[eps_barbar_0, 0], [0, eps_barbar_1]]) * term_3_inner_prod)
    k = 2
    entropy = k/2 * np.log(2*np.pi*np.e) + 1/2*np.log(eps_barbar_0 * eps_barbar_1)
    
    print(term_1, term_2, term_3, entropy)

    return alpha_hat_xy