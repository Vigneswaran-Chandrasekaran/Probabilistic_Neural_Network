"""
Python code for Probabilistic Neural Network (PNN), which 
can be used for classification and pattern-recognition task.
PNN doesn't actually train on dataset instead it classify the 
test data on the flow, by estimating each class's posterior
probability approximated by Parzen window and the suitable class
is selected using Baye's Rule. It was introduced by D.F. Specht 
in 1966.

References:
1) Original work: https://www.sciencedirect.com/science/article/abs/pii/089360809090049Q
2) https://en.wikipedia.org/wiki/Probabilistic_neural_network

Architecture:
It have four layers:
1. Input layer
2. Patter layer
3. Summation layer
4. Output layer
"""
import numpy as np
class Probabilistic_Neural_Network(self):
    def __init__(self, X, Y, kernel = 'gauss', sigma = 2):
        """
        Constructor to create probability distribution for 
        the given data with given parent probability distribution
        function (pdf).

        Parameters
        ----------
        X : Numpy array
        Input data to form probability distribution

        Y : Numpy array
        Target class labels

        kernel : string (Optional)
        Select the type of parent pdf from available functions

        sigma : int
        Smoothning parameter
        """
        self.X = self.normalize(X)
        self.Y = self.Y
        # TODO: check validity of kernel function
        # TODO: check validitiy of each variables
        self.kernel = kernel
        self.sigmal = sigma
    
    def normalize(self, vector):
        vector = vector.T
        return ((vector - np.amin(vector, axis = 1)) / (np.amax(vector, axis = 1)- np.amin(vector, axis = 1))).T

    def predict(self, X_test):
        """
        Function to predict the class for input unknown vector

        Parameters
        ----------
        X_test : Numpy array
        Test vector for which class has to be predicted

        Returns
        -------
        Y_pred : Numpy array
        Predicted class labels
        """
        X_test = self.normalize(X_test)
        if self.kernel == 'gauss':
            Y_pred = []
            for unknown_vector in X_test:
                class_probability_score = []
                for class_label in np.unique(self.Y):
                    class_samples = self.X[np.where(self.Y == class_label), :]
                    distance_metric =  np.sum(np.exp( -1 * ((class_samples - unknown_vector)**2) / (2 * self.sigma**2)))
                    score = (1/np.where(self.Y == class_label).shape[0]) * distance_metric
                    class_probability_score.append(score)
                winner_class = np.unique(self.Y)[class_probability_score.index(max(class_probability_score))]
                Y_pred.append(winner_class)
        return np.array(Y_pred)