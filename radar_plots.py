import numpy as np
import matplotlib.pyplot as plt
import fmcw_radar
import target_path
from target_detection import MAX_VAL_DETECTION

class radar_plots:
    def __init__(self, radar_objs, target_path_obj):
        # Check if the radar object is an instance of the fmcw_radar class or list of fmcw_radar class
        if isinstance(radar_objs, list):
            for radar in radar_objs:
                if not isinstance(radar, fmcw_radar.fmcw_radar):
                    raise ValueError("Radar object must be an instance of the fmcw_radar class")
        else:
            if not isinstance(radar_objs, fmcw_radar.fmcw_radar):
                raise ValueError("Radar object must be an instance of the fmcw_radar class")
        # Check if the target path object is an instance of the target_path class
        if not isinstance(target_path_obj, target_path.target_path):
            raise ValueError("Target path object must be an instance of the target_path class")
        
        # Save init params to the object
        self.__radar_objs = radar_objs
        self.__target_path_obj = target_path_obj
        
        # Predefined axis/bins for plots:
        if isinstance(radar_objs, list):
            self.__range_bins = [ np.linspace(0, radar_objs[0].get_R_max(), radar_objs[0].get_N_s()) for radar_idx in range(len(radar_objs)) ]
            self.__velocity_bins = [ np.linspace(-radar_objs[0].get_v_max(), radar_objs[0].get_v_max(), radar_objs[0].get_N_c()) for radar_idx in range(len(radar_objs)) ]
        else:
            self.__range_bins = np.linspace(0, radar_objs.get_R_max(), radar_objs.get_N_s())
            self.__velocity_bins = np.linspace(-radar_objs.get_v_max(), radar_objs.get_v_max(), radar_objs.get_N_c())
        self.__angle_bins = np.arcsin(np.linspace(1, -1, 181))*180/np.pi

    def plot_path(self, ax = None, show_radar_position = True, current_idx = None):
        """
        Plot the target path
        """
        if ax == None:
            fig, ax = plt.subplots()
        # Get the target trajectory
        x, y, _, _ = self.__target_path_obj.get_trajectory()
        # Plot the target path
        ax.plot(x, y)
        # Plot the current target position if required
        if current_idx != None:
            ax.plot(x[current_idx], y[current_idx], 'go', label='Target Position')
        # Plot the radar position if required
        if show_radar_position:
            if isinstance(self.__radar_objs, list):
                for radar in self.__radar_objs:
                    ax.plot(radar.get_radar_position()[0], radar.get_radar_position()[1], 'ro', label='Radar Position')
            else:
                ax.plot(self.__radar_objs.get_radar_position()[0], self.__radar_objs.get_radar_position()[1], 'ro', label='Radar Position')
        # Set the plot parameters
        ax.set_xlabel('X position [m]')
        ax.set_ylabel('Y position [m]')
        ax.set_title('Target Path')
        ax.grid()
        ax.set_aspect(1,)
        return ax
    def plot_radar_IF_signal(self, ax = None, radar_idx = 0, amplitude_lim = None, plot_time = True):
        """
        Plot the radar IF signal
        """

        radar_obj = self.__radar_objs[radar_idx] if isinstance(self.__radar_objs, list) else self.__radar_objs

        if ax == None:
            fig, ax = plt.subplots()

        # Get the IF signal
        IF_signals = radar_obj.get_raw_radar_data()*(1/radar_obj.get_window())
        
        # Plot the IF signal
        if plot_time:
            t = np.linspace(0, radar_obj.get_N_s()/radar_obj.get_F_s(), radar_obj.get_N_s())
            ax.plot(t,np.real(IF_signals[0,0,0,:]))
            ax.set_xlabel('Time [s]')
        else:
            ax.plot(np.real(IF_signals[0,0,0,:]))
            ax.set_xlabel('Sample')
        ax.set_ylabel('IF Signal')
        ax.set_title('Radar IF Signal')
        ax.grid()
        if amplitude_lim == None:
            pass
        else:
            ax.set_ylim(-amplitude_lim, amplitude_lim)
    def plot_range_fft(self, ax = None, radar_idx = 0, amplitude_range = None):
        """
        Plot the range FFT
        """
        radar_obj = self.__radar_objs[radar_idx] if isinstance(self.__radar_objs, list) else self.__radar_objs

        if ax == None:
            fig, ax = plt.subplots()

        # Get the range FFT
        IF_signals = radar_obj.get_raw_radar_data()*(1/radar_obj.get_window())

        # FFT of the IF signal
        range_fft = np.fft.fftshift(range_fft[0,0,0,:])

        # Plot the range FFT
        if isinstance(self.__radar_objs, list):
            ax.plot(self.__range_bins[radar_idx], 20*np.log10(np.abs(range_fft)))
        else:
            ax.plot(self.__range_bins, 20*np.log10(np.abs(range_fft)))
        ax.set_xlabel('Range')
        ax.set_ylabel('Amplitude')
        ax.set_title('Range FFT')
        ax.grid()
        if amplitude_range == None:
            pass
        else:
            ax.set_ylim(amplitude_range[0], amplitude_range[1])
    def plot_range_doppler_fft(self, ax = None, radar_idx = 0, amplitude_range = None):
        """
        Plot the range-Doppler
        """
        radar_obj = self.__radar_objs[radar_idx] if isinstance(self.__radar_objs, list) else self.__radar_objs

        if ax == None:
            fig, ax = plt.subplots()

        # Get the range-Doppler FFT
        IF_signals = radar_obj.get_raw_radar_data()

        # FFT of the IF signal
        range_doppler_fft = np.zeros((radar_obj.get_N_c(), radar_obj.get_N_s()), dtype=complex)
        for tx_idx in range(radar_obj.get_number_of_TX_antennas()):
            for rx_idx in range(radar_obj.get_number_of_RX_antennas()):
                range_doppler_fft += np.fft.fft2(IF_signals[0,0,:,:])
        range_doppler_fft = np.flip(np.fft.fftshift(range_doppler_fft,axes=0).T,axis=0)
        # Plot the range-Doppler FFT
        if isinstance(self.__radar_objs, list):
            im = ax.imshow(20*np.log10(np.abs(range_doppler_fft)), aspect='auto', extent=[self.__velocity_bins[radar_idx][0], self.__velocity_bins[radar_idx][-1], self.__range_bins[radar_idx][0], self.__range_bins[radar_idx][-1]])
        else:
            im = ax.imshow(20*np.log10(np.abs(range_doppler_fft)), aspect='auto', extent=[self.__velocity_bins[0], self.__velocity_bins[-1], self.__range_bins[0], self.__range_bins[-1]])
        ax.set_xlabel('Velocity')
        ax.set_ylabel('Range')
        ax.set_title('Range-Doppler FFT')
        if amplitude_range == None:
            pass
        else:
            plt.colorbar(im)
            im.set_clim(amplitude_range[0], amplitude_range[1])
    def plot_angle_fft(self, ax = None, radar_idx = 0, amplitude_range = None):
        """
        Plot the angle FFT
        """
        radar_obj = self.__radar_objs[radar_idx] if isinstance(self.__radar_objs, list) else self.__radar_objs

        if ax == None:
            fig, ax = plt.subplots()
        
        # Get the range-Doppler FFT
        IF_signals = radar_obj.get_raw_radar_data()

        # Find the phasors of the target:
        target_phasors = []
        for tx_idx in range(radar_obj.get_number_of_TX_antennas()):
            for rx_idx in range(radar_obj.get_number_of_RX_antennas()):
                range_doppler_fft = np.fft.fft2(IF_signals[tx_idx,rx_idx,:,:])
                detected_targets = MAX_VAL_DETECTION(range_doppler_fft)
                target_phasors.append(range_doppler_fft[detected_targets[0][0],detected_targets[0][1]])
        
        # Find the angle FFT
        angle_fft = np.fft.fftshift(np.fft.fft(target_phasors, n=len(self.__angle_bins)))
        
        # Mark the peak and write the angle
        peak_idx = np.argmax(np.abs(angle_fft))
        peak_angle = self.__angle_bins[peak_idx]
        ax.plot(peak_angle, 20*np.log10(np.abs(angle_fft[peak_idx])), 'ro', label='Peak Angle')
        ax.text(peak_angle, 20*np.log10(np.abs(angle_fft[peak_idx])), f'{peak_angle:.2f}', fontsize=12)

        # Plot the angle FFT
        ax.plot(self.__angle_bins, 20*np.log10(np.abs(angle_fft)))
        ax.set_xlabel('Angle [deg]')
        ax.set_ylabel('Amplitude')
        ax.set_title('Angle FFT')
        ax.grid()
        if amplitude_range == None:
            pass
        else:
            ax.set_ylim(amplitude_range[0], amplitude_range[1])
        ax.set_xlim(self.__angle_bins[-1],self.__angle_bins[0])
        
    ##################################################################################################
    def generate_gif(self,save_path='Figures/radar_plot.gif'):
        """
        Generate a gif based on a set of radar plots saved in the Figures/gif folder, then delete the images.
        """
        import imageio
        import os
        # Check if gif folder is empty
        if not os.listdir('Figures/gif/'):
            print("No images to create gif")
            return 1
        # Create a list of filenames
        filenames = []
        for file in os.listdir('Figures/gif/'):
            filenames.append('Figures/gif/'+file)
        # Create the gif
        with imageio.get_writer(save_path, mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
        # Delete the images
        for filename in filenames:
            os.remove(filename)
        print("Gif created and images deleted")
        return 0