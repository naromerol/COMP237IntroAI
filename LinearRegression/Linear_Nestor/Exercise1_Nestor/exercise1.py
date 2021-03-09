"""
@author Nestor Romero leon
StudentId 301133331
Linear regression Assignment - Exercise 1
"""

import numpy as np
import matplotlib.pyplot as plt

#Create sample and seed random generator
np.random.seed(31)
sample = np.random.uniform(-1,1,100)

#Verify interval for data
positive_vals = sample[sample > 0]
negative_vals = sample[sample < 0]

if len(positive_vals) > 0:
    print('Positive values found in sample: ', len(positive_vals))

if len(negative_vals) > 0:
    print('Negative values found in sample: ', len(negative_vals))

xdata = sample
#mp.hist(xdata)


ydata = 12 * xdata - 4
ydata_noise1 = 12 * xdata - 4 + np.random.normal()
ydata_noise2 = 12 * xdata - 4 + np.random.normal()
#print(ydata)

plt.figure(figsize = (16,8))

##PLOT 1
plt.subplot(1,3,1)
plt.grid()
plt.scatter(xdata,ydata,alpha=0.5)
plt.title('f(x) = 12x-4 (100 samples)')
plt.xlabel('x values')
plt.ylabel('f(x) values')
plt.axhline(0,color = 'black')
plt.axvline(0,color = 'black')

##PLOT 2
plt.subplot(1,3,2)
plt.grid()
plt.scatter(xdata,ydata_noise1,alpha=0.5)
plt.title('f(x)=12x-4 + noise')
plt.xlabel('x values')
plt.ylabel('f(x) values')
plt.axhline(0,color = 'black')
plt.axvline(0,color = 'black')

#PLOT 3 >> Amplify noise effect!!
plt.subplot(1,3,3)
plt.grid()
plt.scatter(xdata,ydata_noise2,alpha=0.5)
plt.title('f(x)=12x-4 + noise * 10')
plt.xlabel('x values')
plt.ylabel('f(x) values')
plt.axhline(0,color = 'black')
plt.axvline(0,color = 'black')

plt.tight_layout()

plt.savefig('./comparison_plot.png')
