#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 04 11:45:00 2025

@author: RHIMMELAND, MaltheRaschke and askebest
"""

import numpy as np
from scipy.constants import c

from numba import njit

class FmcwRadar:
    def __init__(self, 
                 position = np.array([[0,0]]),
                 tx_antennas = None,
                 rx_antennas = None, 
                 chirp_Rate = 30e6/1e-6, 
                 T_between_chirps = 25.66e-6, 
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
        T_between_chirps : float
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
        self.__T_between_chirps = T_between_chirps
        self.__f_carrier = f_carrier
        self.__N_samples = N_samples
        self.__f_sampling = f_sampling
        self.__N_chirps = N_chirps
        self.__signalNoiseRatio = signalNoiseRatio
        self.__transmitPower = transmitPower
        self.__gain = gain
        self.__radarCrossSection = radarCrossSection


        # Constants and derived values:
        self.__c = c                                      # speed of light

        self.__wavelength = self.__c/self.__f_carrier     # self.__wavelength

        if self.__tx_antennas is None:
            self.__tx_antennas = np.array(([-12*(self.__wavelength/2), 0],
                                           [-8*(self.__wavelength/2), 0],
                                           [-4*(self.__wavelength/2), 0]))
            self.__tx_antennas = self.__tx_antennas + self.__position
        else: 
            self.__tx_antennas = self.__tx_antennas + self.__position
        if self.__rx_antennas is None:
            self.__rx_antennas = np.array(([-(3/2)*(self.__wavelength/2), 0],
                                    [-(1/2)*(self.__wavelength/2), 0],
                                    [(1/2)*(self.__wavelength/2), 0],
                                    [(3/2)*(self.__wavelength/2), 0]))
            self.__rx_antennas = self.__rx_antennas + self.__position
        else:
            self.__rx_antennas = self.__rx_antennas + self.__position

        # Check if the position is a 1x2 matrix
        if self.__position.shape != (1, 2):
            raise ValueError("Position must be a 2x1 np.array")
        
        # Noise:
        received_power_SNR = self.__transmitPower*self.__gain*self.__wavelength**2*self.__radarCrossSection/( (4*np.pi)**3 * self.__signalNoiseRatio[1]**4 )
        self.__standardDeviation = np.sqrt(received_power_SNR/10**((self.__signalNoiseRatio[0])/10)) # noise standard deviation
        self.__current_SNR = 0

        # Show parameters:
        self.__R_max = self.__f_sampling*self.__c/(2*self.__chirp_Rate)   # maximum unambiguous range
        self.__v_max = self.__wavelength / (4 * self.__T_between_chirps) # maximum unambiguous velocity
        self.__angle_max = np.pi/2                        # maximum unambiguous angle

        # Data matrix:
        self.__IF_signal = np.zeros((self.__tx_antennas.shape[0], self.__rx_antennas.shape[0], self.__N_chirps, self.__N_samples),dtype=complex)
        self.__S_signal = np.zeros((self.__tx_antennas.shape[0], self.__rx_antennas.shape[0], self.__N_chirps, self.__N_samples),dtype=complex)

        # Time and frequency axis:
        self.__freqs = np.linspace(0, self.__N_samples, self.__N_samples, endpoint=False)[np.newaxis] # Time variable running from 0 to N_samples/F_sampling

    def show_parameters(self):
        f_IF_max = self.__R_max*2*self.__chirp_Rate/(self.__c)
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
        phase_from_velocity = 2 * np.pi * self.__f_carrier * 2 * (radial_velocity * self.__T_between_chirps) / self.__c 

        # Compute the Intermediate frequency (IF) frequency:
        f_IF = (2*radial_distance*self.__chirp_Rate)/(self.__c) 

        # Compute the received power:
        received_power = self.__transmitPower*self.__gain*self.__wavelength**2*self.__radarCrossSection/( (4*np.pi)**3 * radial_distance**4 )

        # Noise signal:
        white_noise = ((np.random.normal(0, 1, self.__IF_signal.shape) 
                        + 1j*np.random.normal(0, 1, self.__IF_signal.shape)) * self.__standardDeviation) / np.sqrt(2)
        
        # Generate the IF signal
        time = np.linspace(0,self.__N_samples/self.__f_sampling,self.__N_samples, endpoint=False)[np.newaxis]  # Time variable running from 0 to N_samples/F_sampling
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
        
        self.__target_position = np.array([target_x, target_y])

        # Compute the radial distance to the target
        self.__radial_distance = np.sqrt(np.sum((self.__position - self.__target_position)**2, axis=1))

        # Compute the phase difference from the target moving during the chirp
        self.__phase_diff_TX_RX = compute_phase_matrix(self.__tx_antennas, self.__rx_antennas, self.__target_position, self.__wavelength)

        # Compute the Intermediate frequency (IF) frequency:
        self.__f_IF = 2*self.__radial_distance*self.__chirp_Rate/self.__c
        
        # Copmute the S-signal:
        signal_gain = 1#np.sqrt(self.__transmitPower * self.__gain * self.__wavelength**2 / (4 * np.pi)**3)
        self.__S_signal = compute_S_signal(self.__S_signal, self.__tx_antennas, self.__rx_antennas, self.__f_IF, self.__f_sampling, self.__N_samples, self.__freqs, self.__phase_diff_TX_RX, signal_gain)

    def get_current_SNR(self, decibels = True):
        if decibels:
            return 10*np.log10(self.__current_SNR)
        else:
            return self.__current_SNR

    @property
    def get_IF_signal(self):
        return self.__IF_signal
    
    @property
    def get_S_signal(self):
        return self.__S_signal
    
    @property
    def get_max_range(self):
        return self.__R_max
    
    @property
    def get_standardDeviation(self):
        return self.__standardDeviation
    
    @property
    def get_position(self):
        return self.__position

    @property
    def get_tx_antennas(self):
        return self.__tx_antennas

    @property
    def get_rx_antennas(self):
        return self.__rx_antennas

    @property
    def get_chirp_Rate(self):
        return self.__chirp_Rate

    @property
    def get_T_between_chirps(self):
        return self.__T_between_chirps

    @property
    def get_f_carrier(self):
        return self.__f_carrier

    @property
    def get_N_samples(self):
        return self.__N_samples

    @property
    def get_f_sampling(self):
        return self.__f_sampling

    @property
    def get_N_chirps(self):
        return self.__N_chirps

    @property
    def get_signalNoiseRatio(self):
        return self.__signalNoiseRatio

    @property
    def get_transmitPower(self):
        return self.__transmitPower

    @property
    def get_gain(self):
        return self.__gain

    @property
    def get_radarCrossSection(self):
        return self.__radarCrossSection
    
    @property
    def get_wavelength(self):
        return self.__wavelength
    
    @property
    def get_parameters(self):
        return { "position": self.__position,
                 "tx_antennas": self.__tx_antennas, 
                 "rx_antennas": self.__rx_antennas, 
                 "chirp_Rate": self.__chirp_Rate, 
                 "T_between_chirps": self.__T_between_chirps, 
                 "f_carrier": self.__f_carrier, 
                 "N_samples": self.__N_samples, 
                 "f_sampling": self.__f_sampling, 
                 "N_chirps": self.__N_chirps, 
                 "transmitPower": self.__transmitPower,
                 "gain": self.__gain,
                 "radarCrossSection": self.__radarCrossSection,
                 "signalNoiseRatio": self.__signalNoiseRatio,
                 "standardDeviation": self.__standardDeviation,
                 "wavelength": self.__wavelength,
                 "max_range": self.__R_max
                }

def radiation_pattern_fnc(x, y):
    theta = np.rad2deg(-np.arctan2(x, y))

    p1 = -0.0000
    p2 = -0.0000
    p3 = 0.0000
    p4 = 0.0000
    p5 = -0.0030
    p6 = -0.0433
    p7 = 10.2333

    return p1 * theta **6 + p2 * theta **5 + p3 * theta **4 + p4 * theta **3 + p5 * theta **2 + p6 * theta + p7
@njit
def compute_phase_matrix(__tx_antennas, __rx_antennas, __target_position, __wavelength):
    # Compute all distances between TX and RX antennas and the target
    __distances = np.zeros((len(__tx_antennas), len(__rx_antennas)))
    for tx_idx in range(len(__tx_antennas)):
        for rx_idx in range(len(__rx_antennas)):
            __distances[tx_idx,rx_idx] = np.sqrt(np.sum((__tx_antennas[tx_idx] - __target_position)**2)) + np.sqrt(np.sum((__rx_antennas[rx_idx] - __target_position)**2))

    # Compute the phase difference between the antennas
    __phase_diff_TX_RX = 2*np.pi*__distances/__wavelength
    __phase_diff_TX_RX -= __phase_diff_TX_RX[0,0]
    return __phase_diff_TX_RX

@njit
def compute_S_signal(__S_signal, __tx_antennas, __rx_antennas, __f_IF, __f_sampling, __N_samples, __freqs, __phase_diff_TX_RX,signal_gain):
    x = 2*np.pi*(__f_IF/__f_sampling-__freqs/__N_samples)
    __S_signal[:, :, :, :] = (np.exp(1.j*(__N_samples-1)*x/2)*np.sin(__N_samples*x/2)/np.sin(x/2))
    for tx_idx in range(__tx_antennas.shape[0]):
        for rx_idx in range(__rx_antennas.shape[0]):
            __S_signal[tx_idx, rx_idx, :, :] *= np.exp(1.j*__phase_diff_TX_RX[tx_idx,rx_idx])
    __S_signal *= signal_gain
    return __S_signal / __N_samples