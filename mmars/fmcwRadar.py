#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 04 11:45:00 2025

@author: RHIMMELAND, MaltheRaschke and askebest
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
        self.__sweepBandwidth = self.__chirp_Rate*self.__T_chirp         # sweep bandwidth
        self.__wavelength = self.__c/self.__f_carrier     # self.__wavelength

        if self.__tx_antennas is None:
            self.__tx_antennas = np.array(([-12*(self.__wavelength/2), 0],
                                           [-8*(self.__wavelength/2), 0],
                                           [-4*(self.__wavelength/2), 0]))
            self.__tx_antennas = self.__tx_antennas + self.__position
        if self.__rx_antennas is None:
            self.__rx_antennas = np.array(([-(3/2)*(self.__wavelength/2), 0],
                                    [-(1/2)*(self.__wavelength/2), 0],
                                    [(1/2)*(self.__wavelength/2), 0],
                                    [(3/2)*(self.__wavelength/2), 0]))
            self.__rx_antennas = self.__rx_antennas + self.__position

        print(self.__tx_antennas)
        print(self.__rx_antennas)

        # Check if the position is a 1x2 matrix
        if self.__position.shape != (1, 2):
            raise ValueError("Position must be a 2x1 np.array")
        
        # Noise:
        received_power_SNR = self.__transmitPower*self.__gain*self.__wavelength**2*self.__radarCrossSection/( (4*np.pi)**3 * self.__signalNoiseRatio[1]**4 )
        self.__standardDeviation = np.sqrt(received_power_SNR/10**(self.__signalNoiseRatio[0]/10)) # noise standard deviation
        self.__current_SNR = 0

        # Show parameters:
        self.__R_max = self.__f_sampling*self.__c/(2*self.__chirp_Rate)   # maximum unambiguous range
        self.__v_max = self.__wavelength / (4 * self.__T_chirp) # maximum unambiguous velocity
        self.__angle_max = np.pi/2                        # maximum unambiguous angle

        # Data matrix:
        self.__IF_signal = np.zeros((self.__tx_antennas.shape[0], self.__rx_antennas.shape[0], self.__N_chirps, self.__N_samples),dtype=complex)
        self.__S_signal = np.zeros((self.__tx_antennas.shape[0], self.__rx_antennas.shape[0], self.__N_chirps, self.__N_samples),dtype=complex)

    def show_parameters(self):
        f_IF_max = self.__R_max*2*self.__sweepBandwidth/(self.__c*self.__T_chirp)
        print(f"Maximum unambiguous range: {self.__R_max:.2f} m")
        print(f"Maximum unambiguous IF frequency: {f_IF_max/1e6:.2f} MHz")
        print(f"Maximum unambiguous velocity: {self.__v_max:.2f} m/s")
        print(f"Maximum unambiguous angle: {np.degrees(self.__angle_max):.2f} degrees")
        print(f"SNR: {self.__signalNoiseRatio[0]} dB at {self.__signalNoiseRatio[1]} m")


    def radar_to_target_measures(self, 
                                 target_x=0, 
                                 target_y=10, 
                                 target_velocity_x=0, 
                                 target_velocity_y=5
                                 ):
        
        target_position = np.array([target_x, target_y])

        # Compute the radial distance to the target
        radial_distance = np.linalg.norm(self.__position - target_position, axis=1)

        # Compute the radial velocity of the target
        radial_velocity = np.dot(np.array([target_velocity_x, target_velocity_y]), (target_position - self.__position).flatten()) / radial_distance

        # Compute all distances between TX and RX antennas and the target
        distances = np.zeros((len(self.__tx_antennas), len(self.__rx_antennas)))
        for tx_idx in range(len(self.__tx_antennas)):
            for rx_idx in range(len(self.__rx_antennas)):
                distances[tx_idx,rx_idx] = np.linalg.norm(self.__tx_antennas[tx_idx] - target_position) + np.linalg.norm(self.__rx_antennas[rx_idx] - target_position)

        # Compute the phase difference between the antennas
        phase_diff_TX_RX = 2*np.pi*distances/self.__wavelength
        phase_diff_TX_RX -= phase_diff_TX_RX[0,0]

        # Compute the phase difference from the target moving during the chirp
        phase_from_velocity = 2 * np.pi * self.__f_carrier * 2 * (radial_velocity * self.__T_chirp) / self.__c 

        # Compute the Intermediate frequency (IF) frequency:
        f_IF = (2*radial_distance*self.__sweepBandwidth)/(self.__c*self.__T_chirp) 

        # Compute the received power:
        received_power = self.__transmitPower*self.__gain*self.__wavelength**2*self.__radarCrossSection/( (4*np.pi)**3 * radial_distance**4 )

        # Noise signal:
        white_noise = (np.random.normal(0, self.__standardDeviation, self.__IF_signal.shape) 
                        + 1j*np.random.normal(0, self.__standardDeviation, self.__IF_signal.shape))/np.sqrt(2)
        
        # Generate the IF signal
        time = np.linspace(0,self.__N_samples/self.__f_sampling,self.__N_samples)[np.newaxis]  # Time variable running from 0 to N_samples/F_sampling
        for tx_idx in range(self.__tx_antennas.shape[0]):
            for rx_idx in range(self.__rx_antennas.shape[0]):
                self.__IF_signal[tx_idx, rx_idx, :, :] = (np.exp(1j*2*np.pi*f_IF*(np.ones((self.__N_chirps,1))@time)) # Changes with ADC samples
                                                         *np.exp(1j*phase_diff_TX_RX[tx_idx,rx_idx]*(np.ones((self.__N_chirps,1))@np.ones((1,self.__N_samples)))) # Changes with antennas
                                                         *np.exp(1j*phase_from_velocity*(np.linspace(0,self.__N_chirps-1,self.__N_chirps)[:,np.newaxis]@np.ones((1,self.__N_samples)))) # Changes with chirps
                                                        )
        self.__IF_signal *= np.sqrt(received_power) # Scale the signal based on the received power
        signal_power = np.mean(np.abs(self.__IF_signal)**2) # Compute the signal power
        noise_power = np.mean(np.abs(white_noise)**2) # Compute the noise power
        self.__current_SNR = signal_power/noise_power # Compute the current SNR
        self.__IF_signal += white_noise # Add noise to the signal

    def generate_S_signal(self, 
                                 target_x=0, 
                                 target_y=10, 
                                 target_velocity_x=0, 
                                 target_velocity_y=5
                                 ):
        
        target_position = np.array([target_x, target_y])

        # Compute the radial distance to the target
        radial_distance = np.linalg.norm(self.__position - target_position, axis=1)

        # Compute the radial velocity of the target
        radial_velocity = np.dot(np.array([target_velocity_x, target_velocity_y]), (target_position - self.__position).flatten()) / radial_distance

        # Compute all distances between TX and RX antennas and the target
        distances = np.zeros((len(self.__tx_antennas), len(self.__rx_antennas)))
        for tx_idx in range(len(self.__tx_antennas)):
            for rx_idx in range(len(self.__rx_antennas)):
                distances[tx_idx,rx_idx] = np.linalg.norm(self.__tx_antennas[tx_idx] - target_position) + np.linalg.norm(self.__rx_antennas[rx_idx] - target_position)

        # Compute the phase difference between the antennas
        phase_diff_TX_RX = 2*np.pi*distances/self.__wavelength
        phase_diff_TX_RX -= phase_diff_TX_RX[0,0]

        # Compute the phase difference from the target moving during the chirp
        phase_from_velocity = 2 * np.pi * self.__f_carrier * 2 * (radial_velocity * self.__T_chirp) / self.__c 

        # Compute the Intermediate frequency (IF) frequency:
        f_IF = (2*radial_distance*self.__sweepBandwidth)/(self.__c*self.__T_chirp) 

        # Compute the received power:
        received_power = self.__transmitPower*self.__gain*self.__wavelength**2*self.__radarCrossSection/( (4*np.pi)**3 * radial_distance**4 )
        
        # Generate the IF signal
        freqs = np.linspace(0,self.__f_sampling,self.__N_samples)[np.newaxis]  # Time variable running from 0 to N_samples/F_sampling
        for tx_idx in range(self.__tx_antennas.shape[0]):
            for rx_idx in range(self.__rx_antennas.shape[0]):
                self.__S_signal[tx_idx, rx_idx, :, :] = (np.sin(((np.ones((self.__N_chirps,1))@freqs) - f_IF)*2*self.__T_chirp)/((np.ones((self.__N_chirps,1))@freqs) - f_IF) # Changes with ADC samples
                                                         *np.exp(1j*phase_diff_TX_RX[tx_idx,rx_idx]*(np.ones((self.__N_chirps,1))@np.ones((1,self.__N_samples)))) # Changes with antennas
                                                         *np.exp(1j*phase_from_velocity*(np.linspace(0,self.__N_chirps-1,self.__N_chirps)[:,np.newaxis]@np.ones((1,self.__N_samples)))) # Changes with chirps
                                                        )
        self.__S_signal *= np.sqrt(received_power) # Scale the signal based on the received power

    def get_current_SNR(self, decibels = True):
        if decibels:
            return 10*np.log10(self.__current_SNR)
        else:
            return self.__current_SNR
        
    def get_IF_signal(self):
        return self.__IF_signal
    
    def get_S_signal(self):
        return self.__S_signal

    def get_max_range(self):
        return self.__R_max
    
    def get_N_samples(self):
        return self.__N_samples