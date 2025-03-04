#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 04 11:45:00 2025

@author: rhimmerland, MaltheRaschke
"""

import numpy as np
from scipy.constants import c

class FmcwRadar:
    def __init__(self, 
                 position,
                 tx_antennas = None,
                 rx_antennas = None, 
                 chirp_Rate = 30e6/1e-6, 
                 T_chirp = 25.66e-6, 
                 f_carrier = 77e9, 
                 N_samples = 256, 
                 f_sampling = 20e6, 
                 N_chirps = 128, 
                 transmitPower = 1,
                 gain = 1,
                 radarCrossSection = 1,
                 signalNoiseRatio = [10, 10]
                 ):
        
        """ Setup a FMCW radar
        
        This class sets up a FMCW radar with the given parameters. The radar is placed at the position given by the pos parameter.

        Parameters
        ----------
        position : np.array
            The position of the radar in 2D space. The position is a 2x1 np.array.
        chirp_Rate : float
            The chirp rate of the radar in MHz/µs.
        T_chirp : float
            The duration of the chirp in µs.
        f_carrier : float
            The carrier frequency of the radar in Hz.
        N_samples : int
            The number of ADC samples.
        f_sampling : float
            The sampling frequency in Hz.
        N_chirps : int
            The number of chirps.
        transmit_Power : float
            The transmitted power in W.
        gain : float
            The antenna gain.
        radarCrossSection : float
            The radar cross section
        signalNoiseRatio : list
            The signal-to-noise ratio of the radar. The first element is the SNR in dB and the second element is the distance in meters where the SNR is measured


        Attributes
        ----------
        -||-
        Notes
        -----
        -||-
        """

        self.__position = position
        self.__tx_antennas = tx_antennas
        self.__rx_antennas = rx_antennas
        self.__chirp_Rate = chirp_Rate
        self.__T_chirp = T_chirp
        self.__f_carrier = f_carrier
        self.__N_samples = N_samples
        self.__f_sampling = f_sampling
        self.__N_chirps = N_chirps
        self.__signalNoiseRatio = signalNoiseRatio
        self.__transmitPower = transmitPower
        self.__gain = gain
        self.__radarCrossSection = radarCrossSection


        # Constants and derived values:
        self.__c = c                                        # speed of light
        self.__B = self.__chirp_Rate*self.__T_chirp         # sweep bandwidth
        self.__wavelength_c = self.__c/self.__f_carrier     # self.__wavelength_c

        if self.__tx_antennas is None:
            self.__tx_antennas = np.array(([-12*(self.__wavelength_c/2), 0],
                                           [-8*(self.__wavelength_c/2), 0],
                                           [-4*(self.__wavelength_c/2), 0]))
            self.__tx_antennas = self.__tx_antennas + self.__position
        if self.__rx_antennas is None:
            rx_antennas = np.array(([-(3/2)*(self.__wavelength_c/2), 0],
                                    [-(1/2)*(self.__wavelength_c/2), 0],
                                    [(1/2)*(self.__wavelength_c/2), 0],
                                    [(3/2)*(self.__wavelength_c/2), 0]))
            self.__rx_antennas = self.__rx_antennas + self.__position

        # Check if the position is a 2x1 matrix
        if self.__position.shape != (2, 1):
            raise ValueError("Position must be a 2x1 np.array")
        
        # Noise:
        received_power_SNR = self.__transmitPower*self.__gain*self.__wavelength_c**2*self.__radarCrossSection/( (4*np.pi)**3 * self.__signalNoiseRatio[1]**4 )
        self.__standardDeviation = np.sqrt(received_power_SNR/10**(self.__signalNoiseRatio[0]/10)) # noise standard deviation

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
    