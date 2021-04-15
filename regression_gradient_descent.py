import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Returns a new array of coefficients based on the current array iteration
def gradient_descent(x_features, coefficients, y_values, learning_rate):
	prediction = np.dot(x_features, coefficients)
	prediction_diff = prediction - y_values
	inner_calc = np.dot(x_features.transpose() , prediction_diff)
	inner_calc = (1/len(y_values)) * learning_rate * inner_calc
	return coefficients - inner_calc

#Returns the sum of all squared differences between predicted vs actual y-values
def cost_function(x_features, coefficients, y_values):
	differences = np.dot(x_features, coefficients) - y_values
	squared_differences = np.square(differences)
	return sum(squared_differences)

#Classify a set of features using provided coefficients
#Return an array of labels (0 or 1)
def get_prediction(x_features, coefficients):
	product = np.dot(x_features, coefficients)
	result = np.array(product > 0.5) #Contains 0 if value is less than 0.5, 1 if greater
	return result

#Return the percentage of predictions that are actually correct
def get_accuracy(x_features, coefficients, y_values):
	this_prediction = get_prediction(x_features, coefficients)
	num_correct_values = compare_nparrays(this_prediction, y_values)
	return num_correct_values / len(y_values)

#Return the number of discrepencies between two nparrays
def compare_nparrays(nparray1, nparray2):
	diff_array = (nparray1 == nparray2)
	#print(diff_array)
	return sum(diff_array)

#Finds the optimal set of coefficients based on the pseudo inverse of the X input array
def find_b_opt(x_matrix, y_values):
	x_plus = np.linalg.pinv(x_matrix) #This is the pseudo inverse. https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html
	return np.dot(x_plus , y_values)

learning_rate = 0.001
num_iterations = 1000

#Load data
training_dataset = pd.read_csv('data/lr_training.csv')
#print(training_dataset)
training_dataset_np = np.array(training_dataset)
#print(training_dataset_np)
testing_dataset = pd.read_csv('data/lr_test.csv')
#print(testing_dataset)
testing_dataset_np = np.array(testing_dataset)
#print(testing_dataset_np)

training_features = training_dataset_np[:, 1:]
training_labels = training_dataset_np[:, 0] #These will be used when finding the optimal coefficients
testing_features = testing_dataset_np[:, 1:]
testing_labels = testing_dataset_np[:, 0] #These will be used for evaluating labellings and calculating accuracy

#Normalize all data to make things easier
#The target value threshold is 0.5, so let's normalize all features so that they lie between zero and one
#We can do this by dividing each feature's value by its maximum possible value (in this case, 255)
training_features_normalized = (1/255) * training_features
testing_features_normalized = (1/255) * testing_features

#Add a column of ones for linear regression's "X0"
ones_col = np.ones(training_features.shape[0])
training_features_normalized = np.insert(training_features_normalized, 0, ones_col, axis=1)
#print("Training Features Normalized:\n{}".format(pd.DataFrame(training_features_normalized)))

ones_col = np.ones(testing_features.shape[0])
testing_features_normalized = np.insert(testing_features_normalized, 0, ones_col, axis=1)
#print("Testing Features Normalized:\n{}".format(pd.DataFrame(testing_features_normalized)))

#Perform gradient descent
est_coeffs = np.zeros(training_features_normalized[0].shape)
costs = []
for i in range(num_iterations):
	#Calculate new coefficients
	est_coeffs = gradient_descent(training_features_normalized, est_coeffs, training_labels, learning_rate)
	est_cost = cost_function(training_features_normalized, est_coeffs, training_labels)
	costs.append(est_cost)
	#print("Iteration: {}, Cost: {}".format(i, est_cost))

#Plot our results of the Cost function over all iterations
figure = plt.figure()
plt.plot(costs)
plt.title("Linear Regression over {} iterations".format(num_iterations))
plt.xlabel("Num. Iterations")
plt.ylabel("Cost")
plt.show()

print("Final coefficients:\n{}".format(est_coeffs))

#Now apply these coefficients to perform labelling on the test dataset
testing_run_accuracy = get_accuracy(testing_features_normalized, est_coeffs, testing_labels)
print("Test run accuracy: {}".format(testing_run_accuracy))

b_opt = find_b_opt(training_features_normalized, training_labels)
coeff_diff = sum(abs(b_opt - est_coeffs))
print("Difference between b_opt and final coefficients: {}".format(coeff_diff))

print("Done!")
