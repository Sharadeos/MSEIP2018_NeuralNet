import csv
import random
import math
#import numpy as np

input_Count = 4
hidden1_Count = 8
output_Count = 3
learningRate = 0.1
epoch = 1000


#Weights
input_hidden1_LayerWeight = [[random.uniform(-1, 1) for x in range(hidden1_Count)] for y in range(input_Count)] 
hidden1_output_LayerWeight = [[random.uniform(-1, 1) for x in range(output_Count)] for y in range(hidden1_Count)] 

input_hidden1_ThetaWeight = [random.uniform(-1, 1) for x in range(hidden1_Count)]
hidden1_output_ThetaWeight = [random.uniform(-1, 1) for x in range(output_Count)]

#Deltas  ^
#Deltas /_\

         
input_hidden1_WeightChange = [[0 for x in range(hidden1_Count)] for y in range(input_Count)] 
hidden1_output_input_hidden1_WeightChange = [[0 for x in range(output_Count)] for y in range(hidden1_Count)] 

accel_input_hidden1_WeightChange = [[0 for x in range(hidden1_Count)] for y in range(input_Count)] 
accel_hidden1_output_input_hidden1_WeightChange = [[0 for x in range(output_Count)] for y in range(hidden1_Count)] 



hidden_Output = [0 for x in range(hidden1_Count)]
final_Output = [0 for x in range(output_Count)]

input = []

def resetWeightChange():
	global input_hidden1_WeightChange
	global hidden1_output_input_hidden1_WeightChange
	
	input_hidden1_WeightChange = [[0 for x in range(hidden1_Count)] for y in range(input_Count)] 
	hidden1_output_input_hidden1_WeightChange = [[0 for x in range(output_Count)] for y in range(hidden1_Count)] 

def sigmoidFunction(input):
	return 1/(1+math.exp(-input))
	
def feedForward(flower):
	global hidden_Output
	global final_Output
	#global input
	
	#resets the outputs
	for i in range(hidden1_Count):
		hidden_Output[i] = 0
	for i in range(output_Count):
		final_Output[i] = 0
	
	#Before Hidden Layer Neurons
	for i in range(input_Count):
		for j in range(hidden1_Count):
			hidden_Output[j] += input[flower][i] * input_hidden1_LayerWeight[i][j] - input_hidden1_ThetaWeight[j] 
			
	for i in range(hidden1_Count):	
		hidden_Output[i] = sigmoidFunction(hidden_Output[i])
	
	#After Hidden Layer	Neurons
	for i in range(hidden1_Count):
		for j in range(output_Count):
			final_Output[j] += hidden_Output[i] * hidden1_output_LayerWeight[i][j]  - hidden1_output_ThetaWeight[j] 
		
	for i in range(output_Count):	
		final_Output[i] = sigmoidFunction(final_Output[i])
	
		
def backPropagation(flower):
	global input_hidden1_LayerWeight
	global hidden1_output_LayerWeight
	global hidden1_output_ThetaWeight
	global input_hidden1_ThetaWeight
	global accel_input_hidden1_WeightChange
	global accel_hidden1_output_input_hidden1_WeightChange
	
	sum = 0
	#Create temporary arrays for storing deltas
	deltaOutputLayer =  [0 for x in range(output_Count)]
	deltaHiddenLayer = [0 for x in range(hidden1_Count)]
	
	#Acceleration
	prevdeltaOutputLayer =  [0 for x in range(output_Count)]
	prevdeltaHiddenLayer = [0 for x in range(hidden1_Count)]
	
	for i in range(output_Count):
		deltaOutputLayer[i] = (input[flower][i+4] - final_Output[i]) * final_Output[i] * (1 - final_Output[i]) 
		hidden1_output_ThetaWeight[i] = hidden1_output_ThetaWeight[i] + (-1 * learningRate * deltaOutputLayer[i])
	
	
	for i in range(hidden1_Count):
		sum = 0
		for j in range(output_Count):
			sum += hidden1_output_LayerWeight[i][j] * deltaOutputLayer[j]
		
		deltaHiddenLayer[i] = hidden_Output[i] * (1 - hidden_Output[i]) * sum
		input_hidden1_ThetaWeight[i] = input_hidden1_ThetaWeight[i] + (-1 * learningRate * deltaHiddenLayer[i])
	
	for i in range(hidden1_Count):
		for j in range(output_Count):
			hidden1_output_input_hidden1_WeightChange[i][j] = (learningRate * deltaOutputLayer[j] * hidden_Output[i])
			hidden1_output_LayerWeight[i][j] += hidden1_output_input_hidden1_WeightChange[i][j] + 0.95*accel_hidden1_output_input_hidden1_WeightChange[i][j]
			accel_hidden1_output_input_hidden1_WeightChange[i][j] = hidden1_output_input_hidden1_WeightChange[i][j]
			
	for i in range(input_Count):
		for j in range(hidden1_Count):
			input_hidden1_WeightChange[i][j] = (learningRate * deltaHiddenLayer[j] * input[flower][i])
			input_hidden1_LayerWeight[i][j] += input_hidden1_WeightChange[i][j] + 0.95*accel_input_hidden1_WeightChange[i][j]
			accel_input_hidden1_WeightChange[i][j] = input_hidden1_WeightChange[i][j]
		
#Start of Neural Network

with open('iris.csv', newline='') as csvfile:
	csvreader = csv.reader(csvfile, delimiter=',')
	
	
	for row in csvreader:
		inputrow = []
		for counter in range(len(row)):
			inputrow.append(float(row[counter]))
		#print(inputrow)
		input.append(inputrow)
		
	
#for x in input:
    #print(*x, sep=" ")
	
#print(input[0][0])
#print(input_hidden1_LayerWeight)
#print(input_hidden1_LayerWeight[0][0])
#print(hidden1_output_input_hidden1_WeightChange[9][0])


#Main Program
for k in range(epoch):
	summation = 0
	for i in range(len(input)):
		
		resetWeightChange()

		feedForward(i)
		backPropagation(i)
		if(k == epoch-1):
			print("Flower:", i)
			print("Label: 1 ",input[i][4], "Result: ", final_Output[0])
			print("Label: 2 ",input[i][5], "Result: ",  final_Output[1])
			print("Label: 3 ",input[i][6], "Result: ",  final_Output[2])
		result = (abs(input[i][4] - final_Output[0])) + (abs(input[i][5] - final_Output[1])) + (abs(input[i][6] - final_Output[2]))
		summation += result

	print("Performance: ", summation/len(input))
	

			
