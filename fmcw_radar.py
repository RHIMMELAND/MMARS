import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class fmcw_radar:

    def __init__(self, pos, debug = True):

        # Check if the position is a 2x1 matrix
        if pos.shape != (2, 1):
            raise ValueError("Position must be a 2x1 matrix")
        self.position = pos

        # Check if debug is a boolean
        if not isinstance(debug, bool):
            raise ValueError("Debug must be a boolean")
        self.debug = debug

        # Radar hardware:
        self.rx_antennas = []
        self.tx_antennas = []

        # Radar settings:
        self.S = 30e6/1e-6 # chirp rate [MHz/µs]
        self.T_c = 25.66e-6 # pulse duration
        self.f_c = 77e9 # carrier frequency
        self.N_s = 256 # number of ADC samples
        self.F_s = 20e6 # sampling frequency
        self.N_c = 40 # number of chirps
        self.T_b = 50e-6 # Time between chirps
        
        # Constants and derived values:
        self.c = 3e8 # speed of light
        self.B = self.S*self.T_c# sweep bandwidth
        self.lambda_c = self.c/self.f_c # wavelength

        # Show parameters:
        self.R_max = self.c * self.N_s / (2 * self.B)
        self.v_max = self.lambda_c / (4 * self.T_b)
        if debug:
            f_IF_max = self.R_max*2*self.B/(self.c*self.T_c)
            range_resolution = self.c / (2 * self.B)

            print(f"Maximum unambiguous range: {self.R_max:.2f} m")
            print(f"Maximum unambiguous velocity: {self.v_max:.2f} m/s")
            print(f"Maximum unambiguous IF frequency: {f_IF_max/1e6:.2f} MHz")
            print(f"Range resolution: {range_resolution:.2f} m")
        
        # Initialize the data array
        self.raw_radar_data = np.zeros((len(self.tx_antennas), len(self.rx_antennas), self.N_c, self.N_s), dtype=complex)
    
    ##############################################################################################

    def add_rx_antenna(self, pos):
        # Check if the position is a 2x1 matrix
        if pos.shape != (2, 1):
            raise ValueError("Position must be a 2x1 matrix")
        self.rx_antennas.append(pos)
        self.raw_radar_data = np.zeros((len(self.tx_antennas), len(self.rx_antennas), self.N_c, self.N_s), dtype=complex)
    def add_tx_antenna(self, pos):
        # Check if the position is a 2x1 matrix
        if pos.shape != (2, 1):
            raise ValueError("Position must be a 2x1 matrix")
        self.tx_antennas.append(pos)
        self.raw_radar_data = np.zeros((len(self.tx_antennas), len(self.rx_antennas), self.N_c, self.N_s), dtype=complex)

    ##############################################################################################

    def measure_target(self, target_position, target_velocity):
        # Check if the target is a 2x1 matrix
        if target_position.shape != (2, 1):
            raise ValueError("Target must be a 2x1 matrix")
        # Calculate the distance between the radar and the target
        relative_distance = np.linalg.norm(target_position - self.position)

        # Calculate the relative distances from TX_n to the target and from the target to RX_m
        distances = np.zeros((len(self.tx_antennas), len(self.rx_antennas)))
        for tx_idx in range(len(self.tx_antennas)):
            for rx_idx in range(len(self.rx_antennas)):
                distances[tx_idx,rx_idx] = np.linalg.norm(target_position - self.tx_antennas[tx_idx]) + np.linalg.norm(target_position - self.rx_antennas[rx_idx])
        print(distances)

        # Calculate the phase difference between the TX and RX antennas
        phase_diff = np.zeros((len(self.tx_antennas), len(self.rx_antennas)))
        for tx_idx in range(len(self.tx_antennas)):
            for rx_idx in range(len(self.rx_antennas)):
                phase_diff[tx_idx,rx_idx] = 2 * np.pi * (distances[tx_idx,rx_idx] / self.lambda_c) % (2 * np.pi) # calculates how many times the wavelength fits in the distance 
        print(phase_diff)

        # Check if the velocity is a 2x1 matrix
        if target_velocity.shape != (2, 1):
            raise ValueError("Velocity must be a 2x1 matrix")
        # Calculate the radial velocity between the radar and the target
        radial_velocity = np.dot(target_velocity.T, target_position - self.position) / relative_distance 

        # Calculate the IF signal frequency
        def IF_signal(distance, radial_velocity, travel_phase):
            f_IF = (2*distance*self.B)/(self.c*self.T_c)
            time_from_vel = 2 * (radial_velocity * self.T_c) / self.c
            phi = 2 * np.pi * self.f_c * time_from_vel
            self.t = np.linspace(0, self.N_s/self.F_s, self.N_s)

            # Calculate the received signal
            ADC_samples = np.zeros((self.N_c, self.N_s))
            for n_c in range(self.N_c):
                ADC_samples[n_c,:] = np.exp( 1j * ( 2 * np.pi * (f_IF) * self.t + n_c * phi + travel_phase ) )
            return ADC_samples

        for tx_idx in range(len(self.tx_antennas)):
            for rx_idx in range(len(self.rx_antennas)):
                self.raw_radar_data[tx_idx,rx_idx,:,:] = IF_signal(relative_distance, radial_velocity, phase_diff[tx_idx,rx_idx])+np.random.normal(0, 0.1, (self.N_c, self.N_s))

    ##############################################################################################

    def plot_IF_signal(self, TX_idxs=[0], RX_idxs=[0]):
        cmap = cm.get_cmap("tab10", len(TX_idxs)*len(RX_idxs))
        plt.figure()
        for tx_idx in TX_idxs:
            for rx_idx in RX_idxs:
                plt.plot(self.t, np.real(self.raw_radar_data[tx_idx,rx_idx,0,:]), label=f"TX{tx_idx} RX{rx_idx}", color=cmap(tx_idx*len(RX_idxs)+rx_idx))
        plt.xlabel("Sample number")
        plt.ylabel("Amplitude")
        plt.title("IF signal")
        plt.legend()
        plt.show()
    def plot_range_fft(self, TX_idxs=[0], RX_idxs=[0]):
        cmap = cm.get_cmap("tab10", len(TX_idxs)*len(RX_idxs))
        plt.figure()
        for tx_idx in TX_idxs:
            for rx_idx in RX_idxs:
                range_fft_range = np.linspace(0, self.R_max, self.N_s//2)
                range_fft = np.fft.fft(self.raw_radar_data[tx_idx,rx_idx,0,:])
                plt.plot(range_fft_range, (np.abs(range_fft[:self.N_s//2])), label=f"TX{tx_idx} RX{rx_idx}", color=cmap(tx_idx*len(RX_idxs)+rx_idx))
        plt.xlabel("Range [m]")
        plt.ylabel("Amplitude [dB]")
        plt.title("Range FFT")
        plt.legend()
        plt.show()
    def plot_range_angle_2D_fft(self):
        plt.figure()
        range_fft_range = np.linspace(0, self.R_max, self.N_s//2)
        angle_fft_range = np.linspace(np.degrees(-np.pi/2), np.degrees(np.pi/2), (len(self.tx_antennas)*len(self.rx_antennas)))
        angle_fft_data = np.zeros((len(self.tx_antennas)*len(self.rx_antennas), self.N_s), dtype=complex)
        for tx_idx in range(len(self.tx_antennas)):
            for rx_idx in range(len(self.rx_antennas)):
                angle_fft_data[tx_idx*len(self.rx_antennas)+rx_idx,:] = self.raw_radar_data[tx_idx,rx_idx,0,:]
        print(angle_fft_data.shape)
        range_angle_fft = np.fft.fft2(angle_fft_data)
        print(range_angle_fft[:range_angle_fft.shape[0]//2, :range_angle_fft.shape[1]//2].shape)
        plt.imshow(np.abs(range_angle_fft[:range_angle_fft.shape[0]//2, :range_angle_fft.shape[1]//2]), extent=[range_fft_range[0], range_fft_range[-1], angle_fft_range[0], angle_fft_range[-1]], aspect='auto', cmap='hot')
        plt.xlabel("Range [m]")
        plt.ylabel("Angle [rad]")
        plt.title("Range-Angle FFT")
        plt.colorbar()
        plt.show()

    ##############################################################################################
        
        


# Example usage:
radar_pos = np.array([[0], [0]])
r1 = fmcw_radar(radar_pos)
r1.add_tx_antenna(np.array([[0], [0]]))
r1.add_tx_antenna(np.array([[3e-3], [0]]))
r1.add_tx_antenna(np.array([[6e-3], [0]]))
#r1.add_tx_antenna(np.array([[9e-3], [0]]))
#r1.add_rx_antenna(np.array([[-12e-3], [0]]))
#r1.add_rx_antenna(np.array([[-9e-3], [0]]))
#r1.add_rx_antenna(np.array([[-6e-3], [0]]))
#r1.add_rx_antenna(np.array([[-3e-3], [0]]))
r1.add_rx_antenna(np.array([[0e-3], [0]]))
r1.add_rx_antenna(np.array([[3e-3], [0]]))
r1.add_rx_antenna(np.array([[6e-3], [0]]))
r1.add_rx_antenna(np.array([[9e-3], [0]]))
#r1.add_rx_antenna(np.array([[12e-3], [0]]))
r1.measure_target(np.array([[10], [0]]), np.array([[0], [10]]) )
r1.plot_range_angle_2D_fft()