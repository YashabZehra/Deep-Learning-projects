import numpy as np

student_data = np.array([[2,8,70],
                         [4,6,80],
                         [6,5,90],
                         [8,4,95]])

desired_output = np.array([50,65,75,85])

weights = np.array([0.3,0.3,0.8])
bias=0.5

predictions= np.dot(student_data,weights)+bias

for i in range(len(student_data)):
    print(f"Input: {student_data[i]} -> Predicted Score: {predictions[i]:.2f} | Actual Score: {desired_output[i]}")