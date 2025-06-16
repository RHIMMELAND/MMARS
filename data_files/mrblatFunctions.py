import numpy as np
import matplotlib.pyplot as plt

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
        self.__temp = 0
        
        self.__lambda_z = np.eye(self.__flatten_data_size) * (self.__standardDeviation)**(-2)
        self.__lambda_z = csr_matrix(self.__lambda_z)

        self.__freqs = np.linspace(0, self.__N_samples, self.__N_samples, endpoint=False)[np.newaxis]
        self.__ds = np.arange(0, len(self.__tx_antennas) * len(self.__rx_antennas) * self.__wavelength/2, self.__wavelength/2)  # should be dynamic
        self.__A = np.sqrt(self.__transmitPower * self.__gain * self.__radarCrossSection * self.__wavelength**2 / (4 * np.pi)**3)
        self.__x_r = self.__position[0,0]
        self.__y_r = self.__position[0,1]

        # self.__radar_setup = FmcwRadar(self.__position, self.__tx_antennas-self.__position, self.__rx_antennas-self.__position, self.__chirp_Rate, 1, radar_parameters["f_carrier"], self.__N_samples, self.__f_sampling, 1, self.__transmitPower, self.__gain, self.__radarCrossSection, [0.1,0.1])
        self.__radar_setup = FmcwRadar(self.__position, self.__tx_antennas, self.__rx_antennas, self.__chirp_Rate, 1, radar_parameters["f_carrier"], self.__N_samples, self.__f_sampling, 1, self.__transmitPower, self.__gain, self.__radarCrossSection, [0.1,0.1]) 

    def alpha_hat(self, s_n, data_fourier):
        # Compute conjugate transpose once
        s_n_H = s_n.conj().T  # Conjugate transpose
        return (s_n_H @ data_fourier) / (s_n_H @ s_n) # Fjernet Lambda_z
    
    def jacobian_S(self, epsilon):
            
        x, y = epsilon
        S_tilde = self.steering_matrix(x, y) * self.sinc(x, y)

        partial_S_tilde_x = self.partial_steering_matrix_x(x, y) * self.sinc(x, y) + self.steering_matrix(x, y) * self.partial_sinc_x(x, y)
        S_jacobian_x = (partial_S_tilde_x * self.path_loss(x, y) + S_tilde * self.partial_path_loss_x(x, y)).flatten()[:, np.newaxis]

        partial_S_tilde_y = self.partial_steering_matrix_y(x, y) * self.sinc(x, y) + self.steering_matrix(x, y) * self.partial_sinc_y(y, x)
        S_jacobian_y = (partial_S_tilde_y * self.path_loss(x, y) + S_tilde * self.partial_path_loss_y(y, x)).flatten()[:, np.newaxis]

        S_jacobian = np.hstack((S_jacobian_x, S_jacobian_y))
        return S_jacobian
    
    def jacobian_S_A(self, epsilon):

        x, y = epsilon
        c  = 3e8
        if y == 0:
            y = 1e-100
        theta = np.atan(x/y) #
        r = np.sqrt(x**2+y**2)
        if r < 1:
            r = 1
        
        Array_pos = np.zeros((len(self.__tx_antennas)*len(self.__rx_antennas),2))
        for TX_idx in range(len(self.__tx_antennas)):
            for RX_idx in range(len(self.__rx_antennas)):
                Array_pos[TX_idx*len(self.__rx_antennas)+RX_idx] = self.__tx_antennas[TX_idx] + self.__rx_antennas[RX_idx]
        Delta = np.linalg.norm(Array_pos - Array_pos[0], axis=1)[:, np.newaxis]

        f_IF =2*self.__chirp_Rate* r/c
        A = np.exp(1.j*2*np.pi*self.__wavelength**(-1)*np.sin(theta)*Delta)
        x_f = 2*np.pi*(f_IF/self.__f_sampling-self.__freqs/self.__N_samples)

        drdx = x/r
        drdy = y/r

        dsinthetadx = y/np.sqrt(x**2/y**2+1)/r**2
        dsinthetady = -x/np.sqrt(x**2/y**2+1)/r**2

        dAdx = 1.j*2*np.pi*self.__wavelength**(-1)*Delta*A*dsinthetadx
        dAdy = 1.j*2*np.pi*self.__wavelength**(-1)*Delta*A*dsinthetady

        df_IFdx = 2*self.__chirp_Rate/c*drdx
        df_IFdy = 2*self.__chirp_Rate/c*drdy

        dx_fdx = 2*np.pi*df_IFdx/self.__f_sampling
        dx_fdy = 2*np.pi*df_IFdy/self.__f_sampling
        expfx = np.exp(1.j*(self.__N_samples-1)*x_f/2)
        kernal = np.sin(self.__N_samples*x_f/2)/np.sin(x_f/2)

        dexpdx = np.exp(1.j*(self.__N_samples-1)*x_f/2)*1.j*(self.__N_samples-1)/2*dx_fdx
        dexpdy = np.exp(1.j*(self.__N_samples-1)*x_f/2)*1.j*(self.__N_samples-1)/2*dx_fdy

        dsindx = (-(self.__N_samples*np.sin(x_f/2)*np.cos((self.__N_samples * x_f)/2))/(np.cos(x_f) - 1) - (np.sin(x_f) * np.sin(x_f/2) * np.sin((self.__N_samples * x_f)/2))/(np.cos(x_f) - 1)**2)*dx_fdx
        dsindy = (-(self.__N_samples*np.sin(x_f/2)*np.cos((self.__N_samples * x_f)/2))/(np.cos(x_f) - 1) - (np.sin(x_f) * np.sin(x_f/2) * np.sin((self.__N_samples * x_f)/2))/(np.cos(x_f) - 1)**2)*dx_fdy

        Jx = dAdx*(expfx*kernal) + A*(dexpdx*kernal + expfx*dsindx)
        Jy = dAdy*expfx*kernal + A*dexpdy*kernal + A*expfx*dsindy


        Jx = Jx.flatten()[:, np.newaxis]
        Jy = Jy.flatten()[:, np.newaxis]

        J = np.hstack((Jx,Jy))/self.__N_samples
        return J
    
    def jacobian_S_H(self, epsilon):

        x, y = epsilon
        c  = 3e8
        # theta = np.atan(x/y)
        r = np.sqrt(x**2+y**2)
        if r < 1:
            r = 1
        
        # Array_pos = np.zeros((len(self.__tx_antennas)*len(self.__rx_antennas),2))
        # for TX_idx in range(len(self.__tx_antennas)):
        #     for RX_idx in range(len(self.__rx_antennas)):
        #         Array_pos[TX_idx*len(self.__rx_antennas)+RX_idx] = self.__tx_antennas[TX_idx] + self.__rx_antennas[RX_idx]
        # Delta = np.linalg.norm(Array_pos - Array_pos[0], axis=1)[:, np.newaxis]

        phi = np.zeros((len(self.__tx_antennas)*len(self.__rx_antennas), 1), dtype=np.complex128)
        dphidx = np.zeros((len(self.__tx_antennas)*len(self.__rx_antennas), 1), dtype=np.complex128)
        dphidy = np.zeros((len(self.__tx_antennas)*len(self.__rx_antennas), 1), dtype=np.complex128)

        for TX_idx in range(len(self.__tx_antennas)):
            for RX_idx in range(len(self.__rx_antennas)):
                X_TX = self.__tx_antennas[TX_idx, 0]
                Y_TX = self.__tx_antennas[TX_idx, 1]
                X_RX = self.__rx_antennas[RX_idx, 0]
                Y_RX = self.__rx_antennas[RX_idx, 1]
                X_TX0 = self.__tx_antennas[0, 0]
                Y_TX0 = self.__tx_antennas[0, 1]
                X_RX0 = self.__rx_antennas[0, 0]
                Y_RX0 = self.__rx_antennas[0, 1]
                phi[TX_idx*len(self.__rx_antennas)+RX_idx] = 2 * 1.j * np.pi * (np.sqrt((x - X_TX) ** 2 + (y - Y_TX) ** 2) + np.sqrt((x - X_RX) ** 2 + (y - Y_RX) ** 2)) / self.__wavelength - 2 * 1.j * np.pi * (np.sqrt((x - X_TX0) ** 2 + (y - Y_TX0) ** 2) + np.sqrt((x - X_RX0) ** 2 + (y - Y_RX0) ** 2)) / self.__wavelength
                dphidx[TX_idx*len(self.__rx_antennas)+RX_idx] = 2 * 1.j * np.pi * (((x - X_TX) ** 2 + (y - Y_TX) ** 2) ** (-0.1e1 / 0.2e1) * (2 * x - 2 * X_TX) / 2 + ((x - X_RX) ** 2 + (y - Y_RX) ** 2) ** (-0.1e1 / 0.2e1) * (2 * x - 2 * X_RX) / 2) / self.__wavelength - 2 * 1.j * np.pi * (((x - X_TX0) ** 2 + (y - Y_TX0) ** 2) ** (-0.1e1 / 0.2e1) * (2 * x - 2 * X_TX0) / 2 + ((x - X_RX0) ** 2 + (y - Y_RX0) ** 2) ** (-0.1e1 / 0.2e1) * (2 * x - 2 * X_RX0) / 2) / self.__wavelength
                dphidy[TX_idx*len(self.__rx_antennas)+RX_idx] = 2 * 1.j * np.pi * (((x - X_TX) ** 2 + (y - Y_TX) ** 2) ** (-0.1e1 / 0.2e1) * (2 * y - 2 * Y_TX) / 2 + ((x - X_RX) ** 2 + (y - Y_RX) ** 2) ** (-0.1e1 / 0.2e1) * (2 * y - 2 * Y_RX) / 2) / self.__wavelength - 2 * 1.j * np.pi * (((x - X_TX0) ** 2 + (y - Y_TX0) ** 2) ** (-0.1e1 / 0.2e1) * (2 * y - 2 * Y_TX0) / 2 + ((x - X_RX0) ** 2 + (y - Y_RX0) ** 2) ** (-0.1e1 / 0.2e1) * (2 * y - 2 * Y_RX0) / 2) / self.__wavelength
        
        
        A = np.exp(phi)
        dAdx = dphidx * np.exp(phi)
        dAdy = dphidy * np.exp(phi)

        f_IF =2*self.__chirp_Rate* r/c
        # A = np.exp(1.j*2*np.pi*self.__wavelength**(-1)*np.sin(theta)*Delta)
        # A = np.exp(2 * 1.j * np.pi * (np.sqrt((x - self.__tx_antennas[:,0]) ** 2 + (y - self.__tx_antennas[:,1]) ** 2) + np.sqrt((x - self.__rx_antennas[:,0]) ** 2 + (y - self.__rx_antennas[:,1]) ** 2)) / self.__wavelength)
        x_f = 2*np.pi*(f_IF/self.__f_sampling-self.__freqs/self.__N_samples)

        drdx = x/r
        drdy = y/r

        # dsinthetadx = y/np.sqrt(x**2/y**2+1)/r**2
        # dsinthetady = -x/np.sqrt(x**2/y**2+1)/r**2

        # dAdx = 1.j*2*np.pi*self.__wavelength**(-1)*Delta*A*dsinthetadx
        # dAdy = 1.j*2*np.pi*self.__wavelength**(-1)*Delta*A*dsinthetady

        df_IFdx = 2*self.__chirp_Rate/c*drdx
        df_IFdy = 2*self.__chirp_Rate/c*drdy

        dx_fdx = 2*np.pi*df_IFdx/self.__f_sampling
        dx_fdy = 2*np.pi*df_IFdy/self.__f_sampling
        expfx = np.exp(1.j*(self.__N_samples-1)*x_f/2)
        kernal = np.sin(self.__N_samples*x_f/2)/np.sin(x_f/2)#*self.__A

        dexpdx = np.exp(1.j*(self.__N_samples-1)*x_f/2)*1.j*(self.__N_samples-1)/2*dx_fdx
        dexpdy = np.exp(1.j*(self.__N_samples-1)*x_f/2)*1.j*(self.__N_samples-1)/2*dx_fdy

        dsindx = (-(self.__N_samples*np.sin(x_f/2)*np.cos((self.__N_samples * x_f)/2))/(np.cos(x_f) - 1) - (np.sin(x_f) * np.sin(x_f/2) * np.sin((self.__N_samples * x_f)/2))/(np.cos(x_f) - 1)**2)*dx_fdx
        dsindy = (-(self.__N_samples*np.sin(x_f/2)*np.cos((self.__N_samples * x_f)/2))/(np.cos(x_f) - 1) - (np.sin(x_f) * np.sin(x_f/2) * np.sin((self.__N_samples * x_f)/2))/(np.cos(x_f) - 1)**2)*dx_fdy

        Jx = dAdx*(expfx*kernal) + A*(dexpdx*kernal + expfx*dsindx)
        Jy = dAdy*(expfx*kernal) + A*(dexpdy*kernal + expfx*dsindy)            

        Jx = Jx.flatten()[:, np.newaxis]
        Jy = Jy.flatten()[:, np.newaxis]

        J = np.hstack((Jx,Jy))/self.__N_samples
        return J

    def path_loss(self, x, y):
        return path_loss_speed(x, y, self.__x_r, self.__y_r, self.__A)
        # r = np.sqrt((x - self.__x_r)**2 + (y - self.__y_r)**2)
        # alpha = self.__A/r**2
        # return alpha

    def partial_path_loss_x(self, x, y):
        return partial_path_loss_x_speed(x, y, self.__x_r, self.__y_r, self.__A)
        # r = np.sqrt((x - self.__x_r)**2 + (y - self.__y_r)**2)
        # partial_alpha = - 2 * self.__A * r**(-4) * (x - self.__x_r)
        # return partial_alpha

    def partial_path_loss_y(self, x, y):
        return partial_path_loss_y_speed(x, y, self.__x_r, self.__y_r, self.__A)
        # r = np.sqrt((x - self.__x_r)**2 + (y - self.__y_r)**2)
        # partial_alpha = - 2 * self.__A * r**(-4) * (y - self.__y_r)
        # return partial_alpha

    def steering_matrix(self, x, y):
        return steering_matrix_speed(x, y, self.__x_r, self.__y_r, self.__ds, self.__wavelength)
        # deltaR = np.sin(-np.arctan2(x - self.__x_r, y - self.__y_r)) * self.__ds
        # phi = deltaR / self.__wavelength
        # steering_mat = np.exp(1.j * 2 * np.pi * phi)
        # return steering_mat[:, np.newaxis]
    
    def partial_steering_matrix_x(self, x, y):
        return partial_steering_matrix_x_speed(x, y, self.__x_r, self.__y_r, self.__ds, self.__wavelength)
        # exp = (x - self.__x_r)**2 + (y - self.__y_r)**2
        # partial_DeltaR_x = - (y - self.__y_r) * self.__ds / (np.sqrt(exp/(y - self.__y_r)**2) * exp)

        # partial_phi_DeltaR = 1 / self.__wavelength

        # DeltaR = np.sin(-np.arctan2(x - self.__x_r, y - self.__y_r)) * self.__ds
        # phi = DeltaR / self.__wavelength
        # partial_A_phi = 1.j * 2 * np.pi * np.exp(1.j * 2 * np.pi * phi)

        # partial_A_x = partial_A_phi * partial_phi_DeltaR * partial_DeltaR_x
        # return partial_A_x[:, np.newaxis]
    
    def partial_steering_matrix_y(self, x, y):
        return partial_steering_matrix_y_speed(x, y, self.__x_r, self.__y_r, self.__ds, self.__wavelength)
        # exp = (x - self.__x_r)**2 + (y - self.__y_r)**2
        # partial_DeltaR_y =  (x - self.__x_r) * self.__ds / (np.sqrt(exp/(y - self.__y_r)**2) * exp)

        # partial_phi_DeltaR = 1 / self.__wavelength

        # DeltaR = np.sin(-np.arctan2(x - self.__x_r, y - self.__y_r)) * self.__ds
        # phi = DeltaR / self.__wavelength
        # partial_A_phi = 1.j * 2 * np.pi * np.exp(1.j * 2 * np.pi * phi)

        # partial_A_y = partial_A_phi * partial_phi_DeltaR * partial_DeltaR_y
        # return partial_A_y[:, np.newaxis]
    
    def sinc(self, x, y):
        return sinc_speed(x, y, self.__x_r, self.__y_r, self.__N_samples, self.__chirp_Rate, self.__f_sampling, self.__freqs)
        # r = np.sqrt((x - self.__x_r)**2 + (y - self.__y_r)**2)
        # f_IF = 2 * self.__chirp_Rate * r / c
        # temp = 2 * np.pi * (f_IF/self.__f_sampling - self.__freqs/self.__N_samples)
        # K = np.exp(1.j * (self.__N_samples - 1) * temp / 2)
        # sinc_fnc = K * np.sin(self.__N_samples * temp / 2) / np.sin(temp / 2)
        # return sinc_fnc/self.__N_samples
    
    def partial_sinc_x(self, x, y):
        return partial_sinc_x_speed(x, y, self.__x_r, self.__y_r, self.__N_samples, self.__chirp_Rate, self.__f_sampling, self.__freqs)
        # N_s = self.__N_samples
        # S = self.__chirp_Rate
        # F_s = self.__f_sampling
        # X_R = self.__x_r
        # Y_R = self.__y_r
        # j = 1.j
        # f = self.__freqs

        # # from Maple:
        # partial_sinc = N_s * np.pi * S * ((x - X_R) ** 2 + (y - Y_R) ** 2) ** (-0.1e1 / 0.2e1) / c / F_s * (2 * x - 2 * X_R) * np.cos(N_s * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) / np.sin(np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) * np.exp(j * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s) * (N_s - 1)) - np.sin(N_s * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) / np.sin(np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) ** 2 * np.exp(j * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s) * (N_s - 1)) * np.pi * S * ((x - X_R) ** 2 + (y - Y_R) ** 2) ** (-0.1e1 / 0.2e1) / c / F_s * (2 * x - 2 * X_R) * np.cos(np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) + np.sin(N_s * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) / np.sin(np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) * j * np.pi * S * ((x - X_R) ** 2 + (y - Y_R) ** 2) ** (-0.1e1 / 0.2e1) / c / F_s * (2 * x - 2 * X_R) * (N_s - 1) * np.exp(j * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s) * (N_s - 1))
        # return partial_sinc/self.__N_samples
    
    def partial_sinc_y(self, x, y):
        return partial_sinc_y_speed(x, y, self.__x_r, self.__y_r, self.__N_samples, self.__chirp_Rate, self.__f_sampling, self.__freqs)
        # N_s = self.__N_samples
        # S = self.__chirp_Rate
        # F_s = self.__f_sampling
        # X_R = self.__x_r
        # Y_R = self.__y_r
        # j = 1.j
        # f = self.__freqs

        # # from Maple:
        # partial_sinc = N_s * np.pi * S * ((x - X_R) ** 2 + (y - Y_R) ** 2) ** (-0.1e1 / 0.2e1) / c / F_s * (2 * y - 2 * Y_R) * np.cos(N_s * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) / np.sin(np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) * np.exp(j * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s) * (N_s - 1)) - np.sin(N_s * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) / np.sin(np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) ** 2 * np.exp(j * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s) * (N_s - 1)) * np.pi * S * ((x - X_R) ** 2 + (y - Y_R) ** 2) ** (-0.1e1 / 0.2e1) / c / F_s * (2 * y - 2 * Y_R) * np.cos(np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) + np.sin(N_s * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) / np.sin(np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s)) * j * np.pi * S * ((x - X_R) ** 2 + (y - Y_R) ** 2) ** (-0.1e1 / 0.2e1) / c / F_s * (2 * y - 2 * Y_R) * (N_s - 1) * np.exp(j * np.pi * (2 * S * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_s - f / N_s) * (N_s - 1))
        # return partial_sinc/self.__N_samples
    
    def D_KL(self, params, Z_data, phi_last, inputmode=(1,1,1,1), outputmode=(1,1,1,1), print_output=False):
        phi_bar_last_x, phi_bar_last_y, phi_bar_x_current, phi_bar_y_current  = phi_last
        eps_bar_x, eps_bar_y, eps_barbar_0, eps_barbar_1 = params

        if inputmode[0] == 0:
            eps_bar_x = phi_bar_x_current
        if inputmode[1] == 0:
            eps_bar_y = phi_bar_y_current
        
        res = (0 + 0.j)

        # Last estimate of the trajectory
        self.__radar_setup.generate_S_signal(phi_bar_last_x, phi_bar_last_y)
        S_N_lack = self.__radar_setup.get_S_signal.flatten()[:, np.newaxis]

        # Generate the S signal with the new parameters
        self.__radar_setup.generate_S_signal(eps_bar_x, eps_bar_y)
        s_n = self.__radar_setup.get_S_signal.flatten()[:, np.newaxis]
        
        # Compute the alpha_hat value
        alpha_hat_xy = self.alpha_hat(S_N_lack, Z_data)
        s_n_H = s_n.conj().T

        if outputmode[0] == 1:
            term_1 = - 2 * np.abs(alpha_hat_xy * (s_n_H @ Z_data))
        else:   
            term_1 = 0
        if outputmode[1] == 1:
            term_2 = np.abs(alpha_hat_xy)**2 * np.real(s_n_H @ s_n)
        else:
            term_2 = 0
        if outputmode[2] == 1:
            jac = self.jacobian_S_A(np.array([eps_bar_x, eps_bar_y]))
            term_3_inner_prod = jac.conj().T @ jac
            term_3 = np.abs(alpha_hat_xy)**2 * (term_3_inner_prod[0,0]*eps_barbar_0**2 + term_3_inner_prod[1,1]* eps_barbar_1**2) 
        else:
            term_3 = 0
        if outputmode[3] == 1:
            k = 2
            term_4 = k/2 * np.log(2*np.pi*np.e) + 1/2*np.log(eps_barbar_0**2 * eps_barbar_1**2)
        else:
            term_4 = 0
        return np.real((outputmode[0] * term_1 + outputmode[1] * term_2 + outputmode[2] * term_3)/((self.__standardDeviation*np.sqrt(self.__N_samples))**2) - outputmode[3] * term_4)
    
    def get_alpha_hat(self, Z_data, phi_bar_last_x, phi_bar_last_y):
        self.__radar_setup.generate_S_signal(phi_bar_last_x, phi_bar_last_y)
        S_N_lack = self.__radar_setup.get_S_signal.flatten()[:, np.newaxis]

        return np.abs(self.alpha_hat(S_N_lack, Z_data))
    
    def get_S_signal(self, phi_bar_last_x, phi_bar_last_y):
        self.__radar_setup.generate_S_signal(phi_bar_last_x, phi_bar_last_y)
        return self.__radar_setup.get_S_signal.flatten()[:, np.newaxis]

@njit
def path_loss_speed(x, y, __x_r, __y_r, __A):
    r = np.sqrt((x - __x_r)**2 + (y - __y_r)**2)
    alpha = __A/r**2
    return 1

@njit
def partial_path_loss_x_speed(x, y, __x_r, __y_r, __A):
    r = np.sqrt((x - __x_r)**2 + (y - __y_r)**2)
    partial_alpha = - 2 * __A * r**(-4) * (x - __x_r)
    return 0

@njit
def partial_path_loss_y_speed(x, y, __x_r, __y_r, __A):
    r = np.sqrt((x - __x_r)**2 + (y - __y_r)**2)
    partial_alpha = - 2 * __A * r**(-4) * (y - __y_r)
    return 0

@njit
def steering_matrix_speed(x, y, __x_r, __y_r, __ds, __wavelength):
    deltaR = np.sin(-np.arctan2(x - __x_r, y - __y_r)) * __ds
    phi = deltaR / __wavelength
    steering_mat = np.exp(1.j * 2 * np.pi * phi)
    return steering_mat[:, np.newaxis]

@njit
def partial_steering_matrix_x_speed(x, y, __x_r, __y_r, __ds, __wavelength):
    exp = (x - __x_r)**2 + (y - __y_r)**2
    partial_DeltaR_x = - (y - __y_r) * __ds / (np.sqrt(exp/(y - __y_r)**2) * exp)

    partial_phi_DeltaR = 1 / __wavelength

    DeltaR = np.sin(-np.arctan2(x - __x_r, y - __y_r)) * __ds
    phi = DeltaR / __wavelength
    partial_A_phi = 1.j * 2 * np.pi * np.exp(1.j * 2 * np.pi * phi)

    partial_A_x = partial_A_phi * partial_phi_DeltaR * partial_DeltaR_x
    return partial_A_x[:, np.newaxis]

@njit
def partial_steering_matrix_y_speed(x, y, __x_r, __y_r, __ds, __wavelength):
    exp = (x - __x_r)**2 + (y - __y_r)**2
    partial_DeltaR_y =  (x - __x_r) * __ds / (np.sqrt(exp/(y - __y_r)**2) * exp)

    partial_phi_DeltaR = 1 / __wavelength

    DeltaR = np.sin(-np.arctan2(x - __x_r, y - __y_r)) * __ds
    phi = DeltaR / __wavelength
    partial_A_phi = 1.j * 2 * np.pi * np.exp(1.j * 2 * np.pi * phi)

    partial_A_y = partial_A_phi * partial_phi_DeltaR * partial_DeltaR_y
    return partial_A_y[:, np.newaxis]

@njit
def sinc_speed(x, y, x_r, y_r, N_samples, chirp_Rate, f_sampling, freqs):
    r = np.sqrt((x - x_r)**2 + (y - y_r)**2)
    f_IF = 2 * chirp_Rate * r / c
    temp = 2 * np.pi * (f_IF/f_sampling - freqs/N_samples)
    K = np.exp(1.j * (N_samples - 1) * temp / 2)
    sinc_fnc = K * np.sin(N_samples * temp / 2) / np.sin(temp / 2)
    return sinc_fnc  / N_samples

@njit
def partial_sinc_x_speed(x, y, X_R, Y_R, N_SAMPLES, CHRIP_RATE, F_SAMPLING, FREQS):
    partial_sinc = N_SAMPLES * np.pi * CHRIP_RATE * ((x - X_R) ** 2 + (y - Y_R) ** 2) ** (-0.1e1 / 0.2e1) / c / F_SAMPLING * (2 * x - 2 * X_R) * np.cos(N_SAMPLES * np.pi * (2 * CHRIP_RATE * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_SAMPLING - FREQS / N_SAMPLES)) / np.sin(np.pi * (2 * CHRIP_RATE * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_SAMPLING - FREQS / N_SAMPLES)) * np.exp(1.j * (N_SAMPLES - 1) * np.pi * (2 * CHRIP_RATE * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_SAMPLING - FREQS / N_SAMPLES)) - np.sin(N_SAMPLES * np.pi * (2 * CHRIP_RATE * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_SAMPLING - FREQS / N_SAMPLES)) / np.sin(np.pi * (2 * CHRIP_RATE * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_SAMPLING - FREQS / N_SAMPLES)) ** 2 * np.exp(1.j * (N_SAMPLES - 1) * np.pi * (2 * CHRIP_RATE * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_SAMPLING - FREQS / N_SAMPLES)) * np.pi * CHRIP_RATE * ((x - X_R) ** 2 + (y - Y_R) ** 2) ** (-0.1e1 / 0.2e1) / c / F_SAMPLING * (2 * x - 2 * X_R) * np.cos(np.pi * (2 * CHRIP_RATE * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_SAMPLING - FREQS / N_SAMPLES)) + np.sin(N_SAMPLES * np.pi * (2 * CHRIP_RATE * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_SAMPLING - FREQS / N_SAMPLES)) / np.sin(np.pi * (2 * CHRIP_RATE * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_SAMPLING - FREQS / N_SAMPLES)) * 1.j * (N_SAMPLES - 1) * np.pi * CHRIP_RATE * ((x - X_R) ** 2 + (y - Y_R) ** 2) ** (-0.1e1 / 0.2e1) / c / F_SAMPLING * (2 * x - 2 * X_R) * np.exp(1.j * (N_SAMPLES - 1) * np.pi * (2 * CHRIP_RATE * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_SAMPLING - FREQS / N_SAMPLES))
    return partial_sinc  / N_SAMPLES

@njit
def partial_sinc_y_speed(x, y, X_R, Y_R, N_SAMPLES, CHRIP_RATE, F_SAMPLING, FREQS):
    partial_sinc = N_SAMPLES * np.pi * CHRIP_RATE * ((x - X_R) ** 2 + (y - Y_R) ** 2) ** (-0.1e1 / 0.2e1) / c / F_SAMPLING * (2 * y - 2 * Y_R) * np.cos(N_SAMPLES * np.pi * (2 * CHRIP_RATE * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_SAMPLING - FREQS / N_SAMPLES)) / np.sin(np.pi * (2 * CHRIP_RATE * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_SAMPLING - FREQS / N_SAMPLES)) * np.exp(1.j * (N_SAMPLES - 1) * np.pi * (2 * CHRIP_RATE * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_SAMPLING - FREQS / N_SAMPLES)) - np.sin(N_SAMPLES * np.pi * (2 * CHRIP_RATE * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_SAMPLING - FREQS / N_SAMPLES)) / np.sin(np.pi * (2 * CHRIP_RATE * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_SAMPLING - FREQS / N_SAMPLES)) ** 2 * np.exp(1.j * (N_SAMPLES - 1) * np.pi * (2 * CHRIP_RATE * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_SAMPLING - FREQS / N_SAMPLES)) * np.pi * CHRIP_RATE * ((x - X_R) ** 2 + (y - Y_R) ** 2) ** (-0.1e1 / 0.2e1) / c / F_SAMPLING * (2 * y - 2 * Y_R) * np.cos(np.pi * (2 * CHRIP_RATE * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_SAMPLING - FREQS / N_SAMPLES)) + np.sin(N_SAMPLES * np.pi * (2 * CHRIP_RATE * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_SAMPLING - FREQS / N_SAMPLES)) / np.sin(np.pi * (2 * CHRIP_RATE * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_SAMPLING - FREQS / N_SAMPLES)) * 1.j * (N_SAMPLES - 1) * np.pi * CHRIP_RATE * ((x - X_R) ** 2 + (y - Y_R) ** 2) ** (-0.1e1 / 0.2e1) / c / F_SAMPLING * (2 * y - 2 * Y_R) * np.exp(1.j * (N_SAMPLES - 1) * np.pi * (2 * CHRIP_RATE * np.sqrt((x - X_R) ** 2 + (y - Y_R) ** 2) / c / F_SAMPLING - FREQS / N_SAMPLES))
    return partial_sinc  / N_SAMPLES

