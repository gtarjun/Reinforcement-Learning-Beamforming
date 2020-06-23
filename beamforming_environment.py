#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:42:02 2020

@author: arjun
"""

import pickle
import numpy as np
import time
import numpy as np
import math
from numpy.linalg import inv
from datagen import DataGen
import matplotlib.pyplot as plt
from matplotlib import style
import h5py
from PIL import Image  # for creating visual of our env
import cv2  # for showing our visual live



style.use("ggplot")

CARRIER_FREQUENCY = 1e9                                       #Frequency of operation
CARRIER_WAVELENGTH = 3e8/CARRIER_FREQUENCY                    #Wavelength
NUMBER_OF_SENSORS = 8                                        #Number of sensors
NOISE_VARIANCE = 0.5                                          #Thermal noise variance 
DEVIATION_FACTOR = 0.5                                          #Uncertainty in randomness of nonuniform sensor spacing
SNAPSHOTS_PER_REALIZATION = 200                               #Snapshots per iteration
NUMBER_OF_SOURCES = 1                                         #Number of sources
SPECTRUM_WIDTH = 60
ITERATIONS = 25000
REWARD = 50

PENALTY = 1
EPS_DECAY = 0.998
SHOW_EVERY = 1000
LEARNING_RATE = 0.01
DISCOUNT = 0.99

epsilon = 0.1

PLAYER_N = 1
SIGNAL_N = 2

d = {1: (0, 255, 0),  # green
     2: (0, 0, 255)}  # red

start_q_table = None

inter_elemental_distance = 0.5 * CARRIER_WAVELENGTH                    #uniform spacing of half wavelength
array_length = (NUMBER_OF_SENSORS - 1) * inter_elemental_distance
uniform_sensor_position = np.linspace(-array_length/2, array_length/2, NUMBER_OF_SENSORS)
uniform_sensor_position = uniform_sensor_position.reshape(NUMBER_OF_SENSORS, 1) #uniform positions for ULA
spectrum_resolution = np.arange(-90, 91, 1)
ber_profile = np.arange(0,1.1,0.1)


data_gen = DataGen()
H = data_gen.hermitian



class Beamformer:
    
    def __init__(self):
        
        self.x = np.int(90)  #np.random.randint(0,len(spectrum_resolution))
        self.y = np.random.randint(0,len(spectrum_resolution))
        
    def __str__(self):
        
        return f"{self.x}"
    
    def __sub__(self, bit_error_rate):
        
        return (0 - bit_error_rate)
    
    def action(self, choice):
        if choice == 0:
            self.move(x = 2)
        
        elif choice == 1:
            self.move(x = 2)       
        
        if choice == 2:
            self.move(x = 0)
        
            
    def move(self, x = False):
        if not x: 
            self.x += np.random.randint(-1,2)       
            
        else:
            self.x += x
            
        if self.x < 0:
            self.x = 0
            
        elif self.x >= len(spectrum_resolution):
            self.x =  180 
    
    def convert_to_rad(self, value_degree):
        """ convert_to_rad
        Arguments:
            value_degree  (float)  : angles in degrees

        Returns:
            angles in radians
        """
        output = value_degree * math.pi/180
        
        return output
    
    def generate_bpsk_signals(self,signal_dir):
        signal_dir_rad = self.convert_to_rad(signal_dir)                                #DOA to estimate   
        phi = 2*math.pi*np.sin(np.tile(signal_dir_rad,[NUMBER_OF_SENSORS, 1]))
        D_u = np.tile(uniform_sensor_position,[1, NUMBER_OF_SOURCES])
        steering_vector = np.exp(1j * phi * (D_u / CARRIER_WAVELENGTH)) 
        symbols = np.round(np.random.rand(NUMBER_OF_SOURCES, SNAPSHOTS_PER_REALIZATION)) * 2 -1                   #BPSK symbols
        noise = np.sqrt(NOISE_VARIANCE / 2)*(np.random.randn(NUMBER_OF_SENSORS, SNAPSHOTS_PER_REALIZATION) + 1j * np.random.randn(NUMBER_OF_SENSORS, SNAPSHOTS_PER_REALIZATION))   
        generated_signal = steering_vector.dot(symbols) + noise   
        
        return generated_signal, symbols, steering_vector
     
    def get_bit_error_rate_channel(self, sampled_signal, original_information):
        
        channel_weight = weights_q[self.x,:,:]
        reconstructed_received_signal = H(channel_weight).dot(sampled_signal)
        reconstructed_information = np.sign(np.real(reconstructed_received_signal))
        bit_error_rate_channel = np.sum(np.abs((original_information - reconstructed_information)/2))/ SNAPSHOTS_PER_REALIZATION
        bit_error_rate_channel = np.round(bit_error_rate_channel, 1)
        bit_error_rate_channel = np.int(bit_error_rate_channel * 10)
        
        return bit_error_rate_channel    
    
   
    def get_bit_error_rate_initial(self, sampled_signal, original_information ):
        
        normal_channel_weight = np.tile([1+1j], [NUMBER_OF_SENSORS, 1])
        reconstructed_received_signal = H(normal_channel_weight).dot(sampled_signal)
        reconstructed_information = np.sign(np.real(reconstructed_received_signal))
        bit_error_rate_channel = np.sum(np.abs((original_information - reconstructed_information)/2))/ SNAPSHOTS_PER_REALIZATION
        bit_error_rate_channel = np.round(bit_error_rate_channel, 2)
        bit_error_rate_channel = np.int(bit_error_rate_channel * 100)
        
        return bit_error_rate_channel 
    
    def read_weight_h5():
        hf = h5py.File('weights_q.h5','r')
        #print(hf.keys())
        weights_q = np.array(hf.get('weights'))
        hf.close()
        
        return weights_q
 
    
#if start_q_table is None:
q_table = {}
for angles in range(len(spectrum_resolution)):
    for ber in range(len(ber_profile)): 
                q_table[((angles),(ber))] = [np.random.uniform(-3, 0) for i in range(3)]
                    
#else: 
#    with open(start_q_table, 'rb') as f:
#        q_table = pickle.load(f)          

weights_q = Beamformer.read_weight_h5() #put this above the for loop below q_table
         
episode_rewards = []
for episode in range(ITERATIONS):
    player = Beamformer()
    signal = Beamformer()
    signal_dir = spectrum_resolution[signal.y]
    incoming_signal, symbols , steering_vector = player.generate_bpsk_signals(signal_dir)
    bit_error_rate = player.get_bit_error_rate_initial(sampled_signal=incoming_signal, original_information=symbols)
    
    if episode % SHOW_EVERY == 0:
        print(f"on # {episode}, epsilon: {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False                        
            
        
    episode_reward = 0  
    
    for i in range(100):
    
        obs = (player.x, player.get_bit_error_rate_channel(sampled_signal=incoming_signal, original_information=symbols))
        print(obs)
        
        if np.random.random() > epsilon:
            action = np.argmax(q_table)
        else:
            action = np.random.randint(0,2)
            
        player.action(action)
        print(action)
        
        new_ber = player.get_bit_error_rate_channel(sampled_signal=incoming_signal, original_information=symbols)
        
        
        if player.get_bit_error_rate_channel(sampled_signal=incoming_signal, original_information=symbols) == 0:
            reward = REWARD
        else:
            reward = -PENALTY
        
        
        new_observation = (player.x, new_ber)
        print(new_observation)
        max_future_q = np.max(q_table[new_observation])
        current_q = q_table[obs][action]
        
        if reward == REWARD:
            new_q = REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward +DISCOUNT * max_future_q)
            
        q_table[obs][action] = new_q
        
        if show:
            env = np.zeros((len(spectrum_resolution),1, 3), dtype=np.uint8)  
            env[player.x] = d[PLAYER_N]  
            env[signal.y] = d[SIGNAL_N]  
            img = Image.fromarray(env, 'RGB')  
            img = img.resize((300, 300))  
            cv2.imshow("image", np.array(img))  
            if reward == REWARD:                # crummy code to hang at the end 
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        episode_reward += reward
        if reward == REWARD:
            break
        
        
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY
            

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)            
            
        
        
            
            
               
            
            
            
            