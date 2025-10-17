#SINGLE NEURON
import numpy as np

inputs = np.array([2.5,3.1,5.0])  #height,weight,age (3features as inputs)

weights_1 = np.array([0.2,0.4,0.6]) #for neuron_1
bias_1 = 0.5

output_1 = np.dot(inputs,weights_1)+bias_1
print("output from neuron 1:",output_1)

#MULTIPLE NEURONS
import numpy as np

inputs = np.array([2.5,3.1,5.0])  #height,weight,age (3features as inputs)

weights = np.array([[0.2,0.4,0.6],[0.4,0.5,0.7],[0.3,0.5,0.7]]) #for neuron_1
bias = np.array([0.3,-1.2,0.5])

output = np.dot(inputs,weights)+bias
print("output from neurons:",output)