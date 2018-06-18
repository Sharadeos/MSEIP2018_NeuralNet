import csv
import random
#import numpy as np

input_Count = 4
hidden1_Count = 10
output_Count = 3
learningRate = 0.05



#Weights
input_hidden1_LayerWeight = [[0 for x in range(input_Count)] for y in range(hidden1_Count)] 
hidden1_output_LayerWeight = [[0 for x in range(hidden1_Count)] for y in range(output_Count)] 

input_hidden1_ThetaWeight = [0 for x in range(input_Count)]
hidden1_output_ThetaWeight = [0 for x in range(hidden1_Count)]

#Deltas  ^
#Deltas /_\
input_hidden1_WeightChange = [[0 for x in range(input_Count)] for y in range(hidden1_Count)] 
hidden1_output_input_hidden1_WeightChange = [[0 for x in range(hidden1_Count)] for y in range(output_Count)] 


hidden_Output = [0 for x in range(input_Count)]
final_Output = [0 for x in range(hidden1_Count)]



#Start of Neural Network

input = []

with open('iris.csv', newline='') as csvfile:
	csvreader = csv.reader(csvfile, delimiter=',')
	
	
	for row in csvreader:
		inputrow = []
		for counter in range(len(row)):
			inputrow.append(float(row[counter]))
		print(inputrow)
		input.append(inputrow)
		
	
for x in input:
    print(*x, sep=" ")
	
	
print(input[0][0])

