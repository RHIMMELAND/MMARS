import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .target import Target
from .fmcwRadar import FmcwRadar

class Simulation:
    def __init__(self, 
                 radar_setup: FmcwRadar, 
                 target_setup: Target
                 ):
        
        self.__frames = None
        self.__SNR_at_frames = None
        self.__x = None
        self.__y = None
        self.__vx = None
        self.__vy = None
        
        self.__radar_setup = radar_setup
        self.__target_setup = target_setup
    def run(self):
        print(f"Running simulation with {self.__radar_setup} and {self.__target_setup}")
        self.__x,self.__y,self.__vx,self.__vy = self.__target_setup.get_trajectory()
        idx = self.__radar_setup.get_IF_signal().shape
        self.__frames = np.zeros((len(self.__x), idx[0], idx[1], idx[2], idx[3]), dtype=complex)
        self.__SNR_at_frames = np.zeros((len(self.__x), 1), dtype=np.float64)

        for i in tqdm(range(len(self.__x))):
            self.__radar_setup.radar_to_target_measures(self.__x[i], self.__y[i], self.__vx[i], self.__vy[i])
            self.__frames[i] = self.__radar_setup.get_IF_signal()
            self.__SNR_at_frames[i] = self.__radar_setup.get_current_SNR()
    
    def run_tracking(self,
                     tracking_algorithm="maximum_value"
                     ):

        self.__tracking_algorithm = tracking_algorithm
        self.__tracking_data_x = np.zeros(len(self.__frames))
        self.__tracking_data_y = np.zeros(len(self.__frames))
        self.__tracking_data_vx = np.zeros(len(self.__frames))
        self.__tracking_data_vy = np.zeros(len(self.__frames))


        if self.__tracking_algorithm == "maximum_value":
            self.__max_val_tracking()
        elif self.__tracking_algorithm == "mrblat":
            self.__mrblat()
        else:
            raise ValueError("Tracking algorithm not supported")
        
    def plot(self):
        if self.__x is None:
            raise ValueError("No data to plot. Run the simulation first.")
        else:
            plt.figure(figsize=(10,10))
            plt.plot(self.__x, self.__y, label="Ground truth", color="black", linewidth=5)
            if self.__tracking_algorithm is not None:
                plt.scatter(self.__tracking_data_x, self.__tracking_data_y, label="Estimated trajectory")
            plt.xlabel("x-position [m]")
            plt.ylabel("y-position [m]")
            plt.legend()
            plt.show()

    def get_data(self, idx = None, flatten = False, fft_data = False):
        return_var = self.__frames
        if fft_data:
            return_var = np.fft.fft(self.__frames, axis=-1)
        if flatten:
            return_var = [return_var[frame_idx].flatten() for frame_idx in range(return_var.shape[0])]
        if idx is not None:
            return_var = return_var[idx]
        return return_var
    
    def get_number_of_frames(self):
        return len(self.__frames)
    
    def get_tracking_data(self):
        return self.__tracking_data_x, self.__tracking_data_y, self.__tracking_data_vx, self.__tracking_data_vy
    
    def get_SNR(self):
        return self.__SNR_at_frames

    def __max_val_tracking(self):
        max_range = self.__radar_setup.get_max_range()
        N_samples = self.__radar_setup.get_N_samples()
        range_values = np.linspace(0, max_range, N_samples)

        k = 0       

        for frame in tqdm(self.__frames):
            range_fft = np.fft.fft(frame[0][0][0])

            range_fft = np.abs(range_fft)
            radial_distance = range_values[np.argmax(range_fft)]
            

            idx = self.__radar_setup.get_IF_signal().shape
            phasors = np.zeros((idx[0]*idx[1], 256), dtype=complex)
            for i in range(idx[0]):
                for j in range(idx[1]):
                    phasors[i*(idx[0]+1)+j] = frame[i][j][0]
            #print(phasors.shape)
            
            R = phasors @ phasors.conj().T*(1/N_samples)
            ### MUSIC
            music = doa_music(R, 1, scan_angles = np.linspace(-90, 90, 1001))
            anglebins = np.linspace(-90, 90, 1001)

            detected_angle = anglebins[np.argmax(music)]

            self.__tracking_data_x[k] = radial_distance*np.cos(np.deg2rad(90-detected_angle))
            self.__tracking_data_y[k] = radial_distance*np.sin(np.deg2rad(90-detected_angle))
            self.__tracking_data_vx[k] = 0
            self.__tracking_data_vy[k] = 0

            k = k+1
    def __mrblat(self):
        MRBLAT_RADAR_CPY = copy.copy(self.__radar_setup)
        PHI = IndexFIFO(300, np.array([0,0,0,0]))
        print(PHI.get(return_numpy=True))

        for frame in tqdm(self.__frames):
            def GET_S_VECTOR(x,y):
                MRBLAT_RADAR_CPY.generate_S_signal(target_velocity_x=x, target_velocity_y=y)
                return (MRBLAT_RADAR_CPY.get_S_signal()).flatten()
            def JACOBIAN_S():
                pass
            def DKL(X,Y,Z_DATA):
                S_VECTOR = GET_S_VECTOR(X,Y)
                LAMBDA_Z = np.eye(len(S_VECTOR))
                S_T_LAMBDA_Z_MULT = S_VECTOR.T.conj() @ LAMBDA_Z

                
            current_Z = (frame).flatten()

    


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

class IndexFIFO:
    def __init__(self, size, dummy_value = None):
        self.size = size
        self.data = [dummy_value] * size
        self.idx = 0
    def append(self, value):
        self.data[self.idx] = value
        self.idx = (self.idx + 1) % self.size 
    def get(self, return_numpy = False):
        if return_numpy:
            return np.array(self.data)
        else:
            return self.data
    def get_from_idx(self, idx, return_numpy = False):
        if return_numpy:
            return np.array(self.data[self.idx + idx])
        else:
            return self.data[self.idx + idx]
    