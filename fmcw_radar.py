import numpy as np
import matplotlib.pyplot as plt

class fmcw_radar:

    def __init__(self, pos, S = 30e6/1e-6, T_c = 25.66e-6, f_c = 77e9, N_s = 256, F_s = 20e6, N_c = 256, SNR = 10):
        # Check if the position is a 2x1 matrix
        if pos.shape != (2, 1):
            raise ValueError("Position must be a 2x1 matrix")
        self.__position = pos

        # Radar hardware:
        self.__rx_antennas = []
        self.__tx_antennas = []

        # Radar settings:
        self.__S = S     #30e6/1e-6 # chirp rate [MHz/µs]
        self.__T_c = T_c #25.66e-6  # pulse duration
        self.__f_c = f_c #77e9      # carrier frequency
        self.__N_s = N_s #256       # number of ADC samples
        self.__F_s = F_s #20e6      # sampling frequency
        self.__N_c = N_c #256       # number of chirps

        # Noise:
        self.__SNR = SNR #10       # signal-to-noise ratio [dB]
        self.__sigma = 10**(-self.__SNR/20) # noise standard deviation
        print(self.__sigma)
        
        # Constants and derived values:
        self.__c = 3e8 # speed of light
        self.__B = self.__S*self.__T_c # sweep bandwidth
        self.__lambda_c = self.__c/self.__f_c # wavelength

        # Show parameters:
        self.__R_max = self.__F_s*self.__c/(2*self.__S) # maximum unambiguous range
        self.__v_max = self.__lambda_c / (4 * self.__T_c) # maximum unambiguous velocity
        self.__angle_max = np.pi/2#(self.__lambda_c / (2 * 1.8e-3)) # maximum unambiguous angle
        
        # Initialize the data array
        self.__raw_radar_data = np.zeros((len(self.__tx_antennas), len(self.__rx_antennas), self.__N_c, self.__N_s), dtype=complex)
    
    def show_parameters(self):
        f_IF_max = self.__R_max*2*self.__B/(self.__c*self.__T_c)
        print(f"Maximum unambiguous range: {self.__R_max:.2f} m")
        print(f"Maximum unambiguous IF frequency: {f_IF_max/1e6:.2f} MHz")
        print(f"Maximum unambiguous velocity: {self.__v_max:.2f} m/s")
        print(f"Maximum unambiguous angle: {np.degrees(self.__angle_max):.2f} degrees")
    
    ##############################################################################################

    def add_rx_antenna(self, pos):
        # Check if the position is a 2x1 matrix
        if pos.shape != (2, 1):
            raise ValueError("Position must be a 2x1 matrix")
        self.__rx_antennas.append(self.__position + pos)
        self.__raw_radar_data = np.zeros((len(self.__tx_antennas), len(self.__rx_antennas), self.__N_c, self.__N_s), dtype=complex)
    def add_tx_antenna(self, pos):
        # Check if the position is a 2x1 matrix
        if pos.shape != (2, 1):
            raise ValueError("Position must be a 2x1 matrix")
        self.__tx_antennas.append(self.__position + pos)
        self.__raw_radar_data = np.zeros((len(self.__tx_antennas), len(self.__rx_antennas), self.__N_c, self.__N_s), dtype=complex)

    ##############################################################################################

    def measure_target(self, target_position, target_velocity):
        # Check if the target is a 2x1 matrix
        if target_position.shape != (2, 1):
            raise ValueError("Target must be a 2x1 matrix")
        # Calculate the distance between the radar and the target
        relative_distance = np.linalg.norm(target_position - self.__position)

        # Calculate the relative distances from TX_n to the target and from the target to RX_m
        distances = np.zeros((len(self.__tx_antennas), len(self.__rx_antennas)))
        for tx_idx in range(len(self.__tx_antennas)):
            for rx_idx in range(len(self.__rx_antennas)):
                distances[tx_idx,rx_idx] = np.linalg.norm(target_position - self.__tx_antennas[tx_idx]) + np.linalg.norm(target_position - self.__rx_antennas[rx_idx])

        # Calculate the phase difference between the TX and RX antennas
        phase_diff = np.zeros((len(self.__tx_antennas), len(self.__rx_antennas)))
        for tx_idx in range(len(self.__tx_antennas)):
            for rx_idx in range(len(self.__rx_antennas)):
                phase_diff[tx_idx,rx_idx] = 2 * np.pi * (distances[tx_idx,rx_idx] / self.__lambda_c) #% (2 * np.pi) # calculates how many times the wavelength fits in the distance 
        phase_diff = phase_diff-phase_diff[0,0]

        # print(phase_diff[0,0]-phase_diff[0,1])
        # print( np.arcsin(wave_length*(phase_diff[0,0]-phase_diff[0,1]) /(2*np.pi*(self.__lambda_c/2))) /np.pi*180 )
        # print( (wave_length*(4.06325307) /(2*np.pi*(self.__lambda_c/2))) /np.pi*180 )

        # Check if the velocity is a 2x1 matrix
        if target_velocity.shape != (2, 1):
            raise ValueError("Velocity must be a 2x1 matrix")
        # Calculate the radial velocity between the radar and the target
        radial_velocity = np.dot(target_velocity.T, target_position - self.__position) / relative_distance 

        # Calculate the IF signal frequency
        def IF_signal(distance, radial_velocity, travel_phase):
            f_IF = (2*distance*self.__B)/(self.__c*self.__T_c)
            time_from_vel = 2 * (radial_velocity * self.__T_c) / self.__c
            phi = 2 * np.pi * self.__f_c * time_from_vel
            t = np.linspace(0, self.__N_s/self.__F_s, self.__N_s)

            # Calculate the received signal
            ADC_samples = np.zeros((self.__N_c, self.__N_s), dtype=complex)
            for n_c in range(self.__N_c):
                ADC_samples[n_c,:] = np.exp( 1j * ( 2 * np.pi * (f_IF) * t + n_c * phi + travel_phase ) ) # / (distance**2)
            return ADC_samples

        for tx_idx in range(len(self.__tx_antennas)):
            for rx_idx in range(len(self.__rx_antennas)):
                self.__raw_radar_data[tx_idx,rx_idx,:,:] = IF_signal(relative_distance, radial_velocity, phase_diff[tx_idx,rx_idx])#+np.random.normal(0, self.__sigma, (self.__N_c, self.__N_s))+1j*np.random.normal(0, self.__sigma, (self.__N_c, self.__N_s))
    
    ##############################################################################################
    
    def plot_IF_signal(self, TX_idxs=[0], RX_idxs=[0]):
        plt.figure()
        for tx_idx in TX_idxs:
            for rx_idx in RX_idxs:
                t = np.linspace(0, self.__N_s/self.__F_s, self.__N_s)
                plt.plot(t, np.real(self.__raw_radar_data[tx_idx,rx_idx,0,:]), label=f"TX{tx_idx} RX{rx_idx}")
        plt.xlabel("Sample number")
        plt.ylabel("Amplitude")
        plt.title("IF signal")
        plt.legend()
        plt.grid()
        plt.show()
    def plot_range_fft(self, TX_idxs=[0], RX_idxs=[0]):
        plt.figure()
        for tx_idx in TX_idxs:
            for rx_idx in RX_idxs:
                range_fft_range = np.linspace(0, self.__R_max, self.__N_s)
                range_fft = np.fft.fft(self.__raw_radar_data[tx_idx,rx_idx,0,:], n=self.__N_s)
                plt.plot(range_fft_range, 10*np.log10(np.abs(range_fft[:self.__N_s])), label=f"TX{tx_idx} RX{rx_idx}")
        plt.ylabel("Amplitude [dB]")
        plt.xlabel("Range [m]")
        plt.title("Range FFT")
        plt.legend()
        plt.grid()
        plt.show()
    def plot_range_doppler_fft(self):
        plt.figure()
        range_fft_range = np.linspace(0, self.__R_max, self.__N_s)
        doppler_fft_range = np.linspace(-self.__v_max, self.__v_max, self.__N_c)
        range_doppler_fft_data = np.flip(np.fft.fftshift(np.fft.fft2(self.__raw_radar_data[0,0,:,:].T), axes=(1)), axis=0)
        plt.imshow(10*np.log10(np.abs(range_doppler_fft_data)), extent=[doppler_fft_range[0], doppler_fft_range[-1], range_fft_range[0], range_fft_range[-1]], aspect='auto', cmap='hot')
        plt.ylabel("Range [m]")
        plt.xlabel("Doppler [m/s]")
        plt.title("Range-Doppler FFT")
        plt.colorbar()
        plt.show()
    def plot_range_angle_fft(self):
        plt.figure()
        range_fft_range = np.linspace(0, self.__R_max, self.__N_s)
        angle_fft_range = np.linspace(np.degrees(-self.__angle_max), np.degrees(self.__angle_max), (len(self.__tx_antennas)*len(self.__rx_antennas)))
        angle_fft_data = np.zeros((len(self.__tx_antennas)*len(self.__rx_antennas), self.__N_s), dtype=complex)
        for tx_idx in range(len(self.__tx_antennas)):
            for rx_idx in range(len(self.__rx_antennas)):
                angle_fft_data[tx_idx*len(self.__rx_antennas)+rx_idx,:] = self.__raw_radar_data[tx_idx,rx_idx,0,:]
        range_angle_fft = np.fft.fftshift(np.fft.fft2(angle_fft_data), axes=(0))
        plt.imshow(10*np.log10(np.abs(range_angle_fft)), extent=[range_fft_range[0], range_fft_range[-1], angle_fft_range[0], angle_fft_range[-1]], aspect='auto', cmap='hot')
        plt.xlabel("Range [m]")
        plt.ylabel("Angle [degrees]")
        plt.title("Range-Angle FFT")
        plt.colorbar()
        plt.show() 
    def plot_angle_fft(self):
        plt.figure()
        angle_fft_range = np.arcsin(np.linspace(1, -1, 250))*180/np.pi
        angle_fft_data = np.zeros((len(self.__tx_antennas)*len(self.__rx_antennas)), dtype=complex)
        for tx_idx in range(len(self.__tx_antennas)):
            for rx_idx in range(len(self.__rx_antennas)):
                angle_fft_data[tx_idx*len(self.__rx_antennas)+rx_idx] = self.__raw_radar_data[tx_idx,rx_idx,0,0]
        # Print phase of angle fft data, in radians (no negative values)
        angle = np.angle(angle_fft_data)
        angle = angle = np.where(angle < 0, angle + 2*np.pi, angle)
        print(angle)
        angle_fft = np.fft.fftshift(np.fft.fft(angle_fft_data,n=250))
        print(angle_fft_range[np.argmax(np.abs(angle_fft))])
        plt.plot(angle_fft_range, 20*np.log10(np.abs(angle_fft)))
        plt.xlabel("Angle [degrees]")
        plt.ylabel("Amplitude [dB]")
        plt.title("Angle FFT")
        plt.grid()
        plt.show()
    
    ##############################################################################################
    
    def get_raw_radar_data(self):
        return self.__raw_radar_data
