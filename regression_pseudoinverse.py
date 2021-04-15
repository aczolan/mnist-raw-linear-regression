import pandas as pd
import numpy as np

#Finds the optimal set of coefficients based on the pseudo inverse of the X input array
def find_b_opt(x_matrix, y_values):
	x_plus = np.linalg.pinv(x_matrix) #This is the pseudo inverse. https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html
	return np.dot(x_plus , y_values)

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

#Return True if two dataframes are identical
def compare_dataframes(df1, df2):
	df1_np = np.array(df1)
	df2_np = np.array(df2)
	return np.allclose(df1_np, df2_np)

#Load data
training_dataset = pd.read_csv('data/lr_training.csv')
training_dataset_np = np.array(training_dataset)
testing_dataset = pd.read_csv('data/lr_test.csv')
testing_dataset_np = np.array(testing_dataset)

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

#Find b_opt
#b_opt = pinv(X) * y
# for all features X and label y in a dataset
b_opt = find_b_opt(training_features_normalized, training_labels)
print("Coefficients:\n{}".format(b_opt))

#Perform labelling on the test dataset, then determine accuracy
testing_run_accuracy = get_accuracy(testing_features_normalized, b_opt, testing_labels)
print("Test run accuracy: {}".format(testing_run_accuracy))

print("Done!")
