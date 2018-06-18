import csv
import random
import numpy as np
from mseiplib import *

input_Count = 4
hidden1_Count = 10
output_Count = 3
learningRate = 0.05

#Weights
input_hidden1_LayerWeight = np.random.uniform(low=-1, high=1, size=(input_Count,hidden1_Count))
hidden1_output_LayerWeight = np.random.uniform(low=-1, high=1, size=(hidden1_Count,output_Count))

input_hidden1_ThetaWeight = np.random.uniform(low=-1, high=1, size=(input_Count))
hidden1_output_ThetaWeight = np.random.uniform(low=-1, high=1, size=(hidden1_Count))

#Deltas  ^
#Deltas /_\
input_hidden1_Delta = np.zeros((input_Count, hidden1_Count))
hidden1_output_Delta = np.zeros((hidden1_Count, output_Count))


hidden_Output = np.zeros((hidden1_Count))
final_Output = np.zeros((output_Count))



#Start of Neural Network
#
input = np.array([])

with open('iris.csv', newline='') as csvfile:
	csvreader = csv.reader(csvfile, delimiter=',')
	
	
	for row in csvreader:
		input = np.append(input, row)
		
finalinput = np.reshape(input, (-1, 7))

#Testing out final numpy array
print(finalinput)
print(finalinput[0][0])
print(len(finalinput))
print(len(finalinput[0]))

print(MyClass.testfunction())
print(MyClass.returnfunction())
finalinput[0][0] = 100
print(finalinput[0])
finalinput[0][0] = 5.1