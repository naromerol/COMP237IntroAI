# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 08:04:11 2021

Assignment - Neural Networks
@author: Nestor Romero - 301133331
Exercise 1

"""

import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt

np.random.seed(1)
min_value = -0.6
max_value = 0.6
#x1, x2 values
input_nestor = np.random.uniform(min_value,max_value,(10, 2))
#calculated y values y = x1 + x2
output_nestor = input_nestor[:,0] + input_nestor[:,1]
output_nestor = output_nestor.reshape(output_nestor.shape[0],1)

#range for each input feature
range_x1 = [input_nestor[:,0].min(),input_nestor[:,0].max()]
range_x2 = [input_nestor[:,1].min(),input_nestor[:,1].max()]

#number of label columns
num_outputs = output_nestor.shape[1]

#create ff neural network 2-6-1
neural_network = nl.net.newff([range_x1,range_x2], [6, num_outputs])

error_progress = neural_network.train(input_nestor, output_nestor, epochs=200, 
                                      show=15, goal=0.00001)

# plt.figure()
# plt.plot(error_progress)

result1 = neural_network.sim([[0.1,0.2]])
print('Predicted result for (0.1,0.2) neural network NN1. Result 1 =', result1)