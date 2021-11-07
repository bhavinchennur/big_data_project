# Importing libraries 

import pandas as pd 

import numpy as np 

from sklearn.model_selection import train_test_split 

from sklearn.neighbors import KNeighborsRegressor 


import csv



# K Nearest Neighbors Regression 

class K_Nearest_Neighbors_Regressor() : 
	
	def __init__( self, K ) : 
		
		self.K = K 
		
	# Function to store training set 
		
	def fit( self, X_train, Y_train ) : 
		
		self.X_train = X_train 
		
		self.Y_train = Y_train 
		
		# no_of_training_examples, no_of_features 
		
		self.m, self.n = X_train.shape 
	
	# Function for prediction 
		
	def predict( self, X_test ) : 
		
		self.X_test = X_test 
		
		# no_of_test_examples, no_of_features 
		
		self.m_test, self.n = X_test.shape 
		
		# initialize Y_predict 
		
		Y_predict = np.zeros( self.m_test ) 
		
		for i in range( self.m_test ) : 
			
			x = self.X_test[i] 
			
			# find the K nearest neighbors from current test example 
			
			neighbors = np.zeros( self.K ) 
			
			neighbors = self.find_neighbors( x ) 
			
			# calculate the mean of K nearest neighbors 
			
			Y_predict[i] = np.mean( neighbors ) 
			
		return Y_predict 
	
	# Function to find the K nearest neighbors to current test example 
			
	def find_neighbors( self, x ) : 
		
		# calculate all the euclidean distances between current test 
		# example x and training set X_train 
		
		euclidean_distances = np.zeros( self.m ) 
		
		for i in range( self.m ) : 
			
			d = self.euclidean( x, self.X_train[i] ) 
			
			euclidean_distances[i] = d 
		
		# sort Y_train according to euclidean_distance_array and 
		# store into Y_train_sorted 
		
		inds = euclidean_distances.argsort() 
		
		Y_train_sorted = self.Y_train[inds] 
		
		return Y_train_sorted[:self.K] 
	
	# Function to calculate euclidean distance 
			
	def euclidean( self, x, x_train ) : 
		
		return np.sqrt( np.sum( np.square( x - x_train ) ) ) 

# Driver code 

def main() : 
	
	# Importing dataset 
	
	df = pd.read_csv( "bd_knn_1.csv" ) 

	X = df.iloc[:,:-1].values 

	Y = df.iloc[:,1].values 
	

	# Splitting dataset into train and test set 

	X_train, X_test, Y_train, Y_test = train_test_split( 
	X, Y, test_size = 1/3, random_state = 0 ) 
	
	# Model training 
	
	model = K_Nearest_Neighbors_Regressor( K = 3 ) 

	model.fit( X_train, Y_train ) 
	
	model1 = KNeighborsRegressor( n_neighbors = 3 ) 
	
	model1.fit( X_train, Y_train ) 
	
	# Prediction on test set 

	Y_pred = model.predict( X_test ) 
	
	Y_pred1 = model1.predict( X_test ) 
	
	

	
	
	
	
	
	
	file1 = open('/usr/local/hadoop/project_2/bd_output_1.txt', 'r') 
	Lines = file1.readlines() 
  

	rows, cols = (len(Lines), 1)

	arr = [[0 for i in range(cols)] for j in range(rows)] 


	count = 0

	# Strips the newline character 
	for line in Lines: 

		
	# String to store the resultant String 
		res = ""; 
  
    # Traverse the words and 
    # remove the first and last letter 
		 
		
    
		res = line[1:len(line) - 2]
		
		
		s1 = res.split(",")
		
		s2 = s1[1]
		
		s3 = s2[1:len(s2)-1]
		

	

		 

		
		arr[count][0] = float(s3)

		arr[count][0] = int(arr[count][0])	
		
		count = count + 1


	temp = model.predict(np.array(arr))
	
	output = []
	
	for i in range(len(temp)):
		if(temp[i] - int(temp[i]) < 0.5):
			output.append( int(temp[i]))
		else:
			output.append(int(temp[i])+1)
		
		
	for i in range(len(output)):	
		print(output[i],"\t",1)		# node number is output and mapper function
		
	file1 = open('final_output.txt', 'a') 
	for i in range(len(output)):
		s3 = "user " + str(i) + " : Node" + str(output[i]) + "\n"
		file1.writelines(s3) 
	file1.close()
		
	
	

    	
    	
    	
    	
    	
    	
    	
    	
	
	


if __name__ == "__main__" : 
	
	# Opening a file 
	file1 = open('final_output.txt', 'a') 
	
	L = ["\n\nThe above is the output of the spark program \n\n",
	 "The 3rd user input is the user_id  \n",
	 "And the 4th input is the number of jobs given by the user at that instace   \n\n",
	 "The spark program outputs the user id and the average number of jobs given by the user\n\n\n",
	 "Now,\n",
	 "we'll use a K-Nearest Neighbours algorithm to select nodes appropriately\n\n"
	 "Every worker node has a capacity, \n\ni.e., it can execute a certain number of jobs in ,say, 1 hour depending on it's capacity. \n\n"
	 "We have user_id's and their average number of jobs\n\n"
	 "We have to train the KNN Model, the training data is:\n\n",
	 "No. of jobs per hour \t Node no.\n"]
	 
	 
	 
	 
	file1.writelines(L) 
	
	with open('bd_knn_1.csv', mode = 'r') as file:
		csvFile = csv.reader(file)
		
		for lines in csvFile:
			s1 = lines[0] + "\t\t\t\t\t\t\t"+lines[1]
			file1.writelines(s1)
			file1.writelines("\n")
	
	
	L1 = ["\n\nUsing averages from spark output we predict the node to be assigned, according to its capacity, using kNN algorithm: \n\n"]
	
	file1.writelines(L1) 
	
	
	file1.close() 
	 
	 

	
	main()

