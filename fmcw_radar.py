import numpy as np
import matplotlib.pyplot as plt

class fmcw_radar:
    def __init__(self, pos, S = 30e6/1e-6, T_c = 25.66e-6, f_c = 77e9, N_s = 256, F_s = 20e6, N_c = 128, plane_wave_approx = False, SNR = [100,10]):
        # Check if the position is a 2x1 matrix
        if pos.shape != (2, 1):
            raise ValueError("Position must be a 2x1 matrix")
        self.__position = pos

        # Radar hardware:
        self.__rx_antennas = []
        self.__tx_antennas = []

        # Radar settings:
        self.__S   = S   #30e6/1e-6 # chirp rate [MHz/µs]
        self.__T_c = T_c #25.66e-6  # pulse duration
        self.__f_c = f_c #77e9      # carrier frequency
        self.__N_s = N_s #256       # number of ADC samples
        self.__F_s = F_s #20e6      # sampling frequency
        self.__N_c = N_c #256       # number of chirps
        self.__transmitted_power = 1# transmitted power [W]
        self.__G = 1                # antenna gain
        self.__RCS = 1              # radar cross section
        
        # Constants and derived values:
        self.__c = 3e8                          # speed of light
        self.__B = self.__S*self.__T_c          # sweep bandwidth
        self.__lambda_c = self.__c/self.__f_c   # wavelength

        # Noise:
        self.__SNR = SNR                    # signal-to-noise ratio, at some distance [ SNR [dB], distance [m] ]
        recieved_power_SNR = self.__transmitted_power*self.__G*self.__lambda_c**2*self.__RCS/( (4*np.pi)**3 * self.__SNR[1]**4 )
        self.__sigma = 10*np.sqrt(recieved_power_SNR/10**(self.__SNR[0]/10)) # noise standard deviation

        # Show parameters:
        self.__R_max = self.__F_s*self.__c/(2*self.__S)   # maximum unambiguous range
        self.__v_max = self.__lambda_c / (4 * self.__T_c) # maximum unambiguous velocity
        self.__angle_max = np.pi/2                        # maximum unambiguous angle
        
        # Initialize the data array
        self.__raw_radar_data = np.zeros((len(self.__tx_antennas), len(self.__rx_antennas), self.__N_c, self.__N_s), dtype=complex)
        self.__window = np.ones((1,self.__N_s))

        # Class settings:
        self.__plane_wave_approx = plane_wave_approx # plane wave approximation, True or False.
    
    def show_parameters(self):
        f_IF_max = self.__R_max*2*self.__B/(self.__c*self.__T_c)
        print(f"Maximum unambiguous range: {self.__R_max:.2f} m")
        print(f"Maximum unambiguous IF frequency: {f_IF_max/1e6:.2f} MHz")
        print(f"Maximum unambiguous velocity: {self.__v_max:.2f} m/s")
        print(f"Maximum unambiguous angle: {np.degrees(self.__angle_max):.2f} degrees")
        print(f"SNR: {self.__SNR[0]} dB at {self.__SNR[1]} m")
    
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

    def set_window(self, window='rectangular'):
        if window == 'rectangular':
            self.__window = np.ones((self.__N_c,self.__N_s))
        elif window == 'hanning':
            window_ADC_samples = np.hanning(self.__N_s)
            window_chirps = np.hanning(self.__N_c)
            self.__window = np.outer(window_chirps, window_ADC_samples)
        elif window == 'hamming':
            window_ADC_samples = np.hamming(self.__N_s)
            window_chirps = np.hamming(self.__N_c)
            self.__window = np.outer(window_chirps, window_ADC_samples)
        elif window == 'blackman':
            window_ADC_samples = np.blackman(self.__N_s)
            window_chirps = np.blackman(self.__N_c)
            self.__window = np.outer(window_chirps, window_ADC_samples)
        else:
            raise ValueError("Window must be 'rectangular', 'hanning', 'hamming' or 'blackman'")
    
    def get_wavelength(self):
        return self.__lambda_c
    def get_raw_radar_data(self):
        return self.__raw_radar_data
    def get_radar_position(self):
        return self.__position
    def get_N_s(self):
        return self.__N_s
    def get_F_s(self):
        return self.__F_s
    def get_N_c(self):
        return self.__N_c
    def get_T_c(self):
        return self.__T_c
    def get_S(self):
        return self.__S
    def get_f_c(self):
        return self.__f_c
    def get_R_max(self):
        return self.__R_max
    def get_v_max(self):
        return self.__v_max
    def get_angle_max(self):
        return self.__angle_max
    def get_number_of_TX_antennas(self):
        return len(self.__tx_antennas)
    def get_number_of_RX_antennas(self):
        return len(self.__rx_antennas)
    def get_window(self):
        return self.__window

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
        if self.__plane_wave_approx:
            phase_diff = np.zeros((len(self.__tx_antennas), len(self.__rx_antennas)))
            for tx_idx in range(len(self.__tx_antennas)):
                for rx_idx in range(len(self.__rx_antennas)):
                    phase_diff[tx_idx,rx_idx] = np.linalg.norm(self.__tx_antennas[tx_idx] - self.__rx_antennas[rx_idx])* 2 * np.pi * np.sin(np.arccos(np.array([[0],[1]]).T@target_position/np.linalg.norm(target_position)))/self.__lambda_c
        else:
            phase_diff = np.zeros((len(self.__tx_antennas), len(self.__rx_antennas)))
            for tx_idx in range(len(self.__tx_antennas)):
                for rx_idx in range(len(self.__rx_antennas)):
                    phase_diff[tx_idx,rx_idx] = 2 * np.pi * (distances[tx_idx,rx_idx] / self.__lambda_c) # calculates how many times the wavelength fits in the distance 
            phase_diff = phase_diff-phase_diff[0,0]
        
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
            return ADC_samples*self.__window

        # Builds the raw radar data, based on TX-RX pairs
        for tx_idx in range(len(self.__tx_antennas)):
            for rx_idx in range(len(self.__rx_antennas)):
                self.__raw_radar_data[tx_idx,rx_idx,:,:] = IF_signal(relative_distance, radial_velocity, phase_diff[tx_idx,rx_idx])
        recieved_power = self.__transmitted_power*self.__G*self.__lambda_c**2*self.__RCS/( (4*np.pi)**3 * relative_distance**4 )
        self.__raw_radar_data *= np.sqrt(recieved_power) +np.random.normal(0, (self.__sigma), self.__raw_radar_data.shape)+1j*np.random.normal(0, (self.__sigma), self.__raw_radar_data.shape)
    
    ##############################################################################################
# RADAR PRESETS:
def default_77GHz_FMCW_radar(radar_position = np.array([[0], [0]])):
    """
    This function creates a default 77GHz FMCW radar with 3 TX antennas and 4 RX antennas.
    """
    # Check if the position is a 2x1 matrix
    if radar_position.shape != (2, 1):
        raise ValueError("Position must be a 2x1 matrix")
    # Create a default 77GHz FMCW radar, used for testing
    default_77GHz_FMCW_radar = fmcw_radar(radar_position, SNR=[20,50])
    wavelength = default_77GHz_FMCW_radar.get_wavelength()
    default_77GHz_FMCW_radar.add_tx_antenna(np.array([[0e-3], [0]]))
    default_77GHz_FMCW_radar.add_tx_antenna(np.array([[2*wavelength], [0]]))
    default_77GHz_FMCW_radar.add_tx_antenna(np.array([[4*wavelength], [0]]))

    default_77GHz_FMCW_radar.add_rx_antenna(np.array([[6*wavelength], [0]]))
    default_77GHz_FMCW_radar.add_rx_antenna(np.array([[6*wavelength+1*wavelength/2], [0]]))
    default_77GHz_FMCW_radar.add_rx_antenna(np.array([[6*wavelength+2*wavelength/2], [0]]))
    default_77GHz_FMCW_radar.add_rx_antenna(np.array([[6*wavelength+3*wavelength/2], [0]]))
    return default_77GHz_FMCW_radar


