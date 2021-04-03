# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 08:04:11 2021

Assignment - Neural Networks
@author: Nestor Romero - 301133331
Exercise 5 - Part B

"""

import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt

np.random.seed(1)
min_value = -0.6
max_value = 0.6
#x1, x2 values
input_nestor = np.random.uniform(min_value,max_value,(100, 3))
#calculated y values y = x1 + x2 + x3
output_nestor = input_nestor[:,0] + input_nestor[:,1] + input_nestor[:,2]
output_nestor = output_nestor.reshape(output_nestor.shape[0],1)

#range for each input feature
range_x1 = [input_nestor[:,0].min(),input_nestor[:,0].max()]
range_x2 = [input_nestor[:,1].min(),input_nestor[:,1].max()]
range_x3 = [input_nestor[:,2].min(),input_nestor[:,2].max()]

#number of label columns
num_outputs = output_nestor.shape[1]

#create ff neural network 2-5-3-1
neural_network6 = nl.net.newff([range_x1,range_x2,range_x3], [5,3,num_outputs])
neural_network6.trainf = nl.train.train_gd

error_progress = neural_network6.train(input_nestor, output_nestor, epochs=1000, 
                                      show=100, goal=0.00001)

result6 = neural_network6.sim([[0.2,0.1,0.2]])
print('Predicted result for (0.2,0.1,0.2) neural network NN6. Result 6 =', result6)