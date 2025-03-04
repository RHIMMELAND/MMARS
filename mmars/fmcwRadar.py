import numpy as np

class FmcwRadar:
    def __init__(self, 
                 pos, 
                 S = 30e6/1e-6, 
                 T_c = 25.66e-6, 
                 f_c = 77e9, 
                 N_samples = 256, 
                 frequency_sampling = 20e6, 
                 N_chirps = 128, 
                 signalNoiseRatio = [10, 10]
                 ):
        
        """ Setup a FMCW radar
        
        This class sets up a FMCW radar with the given parameters. The radar is placed at the position given by the pos parameter.

        Parameters
        ----------
        pos : np.array
            The position of the radar in the coordinate system of the simulation.
        S : float
            The chirp rate [MHz/µs]
        
        
        
        
        
        """
        # Check if the position is a 2x1 matrix
        if pos.shape != (2, 1):
            raise ValueError("Position must be a 2x1 np.array")
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
        self.__sigma = np.sqrt(recieved_power_SNR/10**(self.__SNR[0]/10)) # noise standard deviation
        print(self.__sigma)

        # Show parameters:
        self.__R_max = self.__F_s*self.__c/(2*self.__S)   # maximum unambiguous range
        self.__v_max = self.__lambda_c / (4 * self.__T_c) # maximum unambiguous velocity
        self.__angle_max = np.pi/2                        # maximum unambiguous angle
        
        # Initialize the data array
        self.__raw_radar_data = np.zeros((len(self.__tx_antennas), len(self.__rx_antennas), self.__N_c, self.__N_s), dtype=complex)
        self.__window = np.ones((1,self.__N_s))

    
    def show_parameters(self):
        f_IF_max = self.__R_max*2*self.__B/(self.__c*self.__T_c)
        print(f"Maximum unambiguous range: {self.__R_max:.2f} m")
        print(f"Maximum unambiguous IF frequency: {f_IF_max/1e6:.2f} MHz")
        print(f"Maximum unambiguous velocity: {self.__v_max:.2f} m/s")
        print(f"Maximum unambiguous angle: {np.degrees(self.__angle_max):.2f} degrees")
        print(f"SNR: {self.__SNR[0]} dB at {self.__SNR[1]} m")
    