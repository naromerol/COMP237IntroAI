# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 08:04:11 2021

Assignment - Neural Networks
@author: Nestor Romero - 301133331
Exercise 4

"""

import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt

np.random.seed(1)
min_value = -0.6
max_value = 0.6
#x1, x2 values
input_nestor = np.random.uniform(min_value,max_value,(100, 2))
#calculated y values y = x1 + x2
output_nestor = input_nestor[:,0] + input_nestor[:,1]
output_nestor = output_nestor.reshape(output_nestor.shape[0],1)

#range for each input feature
range_x1 = [input_nestor[:,0].min(),input_nestor[:,0].max()]
range_x2 = [input_nestor[:,1].min(),input_nestor[:,1].max()]

#number of label columns
num_outputs = output_nestor.shape[1]

#create ff neural network 2-5-3-1
neural_network4 = nl.net.newff([range_x1,range_x2], [5,3,num_outputs])
neural_network4.trainf = nl.train.train_gd

error_progress = neural_network4.train(input_nestor, output_nestor, epochs=1000, 
                                      show=100, goal=0.00001)

plt.figure()
plt.title('Error progress NN4')
plt.xlabel('Epoch')
plt.ylabel('Error rate')
plt.plot(error_progress)

result4 = neural_network4.sim([[0.1,0.2]])
print('Predicted result for (0.1,0.2) neural network NN4. Result 4 =', result4)