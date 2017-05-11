# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 18:22:59 2017
@author: Gulshan Madhwani, Ritesh Tawde
@version: Python 3.5
"""

# This is a face expression recognition projecct
# goal is to use the DNN methodology with soft-max to classify between one of the 4 face expressions
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
# load the packages / libraries / modules

# for plottin graph
import matplotlib.pyplot as plt
# for data shuffling
from sklearn.utils import shuffle
# load configurations
import configparser
# linear algebra library
import numpy as np


class FaceExp_dnn:
    def __init__(self):
        self.hiddenLayers = 200
        self.learningRate = 10e-7
        self.reg = 10e-7
        self.epochs = 10000
        self.training = "fer2013_training.csv"
        self.validation = "fer2013_validation.csv"
        self.testing = "fer2013_testing.csv"
        self.readConfig()
    
    def getValue(self,config,section, option, oldVal):
        if(config.has_option(section,option)):
            return config.get(section,option)
        else:
            return oldVal
    
    def readConfig(self):
        config = configparser.ConfigParser()
        config.read("./config.ini")
        self.training = self.getValue(config, 'Files', 'training', self.training)
        self.validation = self.getValue(config, 'Files', 'validation', self.validation)
        self.testing = self.getValue(config, 'Files', 'testing', self.testing)
        self.hiddenLayers = int(self.getValue(config, 'Parameters', 'hiddenLayers', self.hiddenLayers))
        self.learningRate = float(self.getValue(config, 'Parameters', 'learningRate', self.learningRate))
        self.reg = float(self.getValue(config, 'Parameters', 'reg', self.reg))
        self.epochs = int(self.getValue(config, 'Parameters', 'epochs', self.epochs))
    def LoadData(self, filepath):
        # this data is collected from the kaggle competition - link shared in our initial proposal report
        # https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge    
        # pixels are considered as the numpy array with spatial information of the image sacrified
        # The images are 48 X 48 pixel
        Header = True
        X_Features = []
        Y_predict =[]
	i = 0
        
        # read the source data file
        for imageRecord in open(filepath):
            if Header:
				# avoid headers
                Header = False
            else:
                Currentfeature=[]
                imageCurrentRow = imageRecord.split(',')
		#print(i,' ')
		#i = i + 1
                # get the x feature and the Y prediction values
                features = imageCurrentRow[1].split()
                for f in features:
                    Currentfeature.append(int(f))
                X_Features.append(Currentfeature)
                
                #X_Features.append([int(p) for p in imageCurrentRow[1].split()])
                
                pred = imageCurrentRow[0]
                Y_predict.append(int(pred))
                
        #normalization
        X_Features =  np.array(X_Features)  / 255.0
        Y_predict = np.array(Y_predict)
        
        return X_Features,Y_predict
    
    def LoadTestData(self, filepath, class_id):
        Header = True
        X_Features = []
        Y_predict =[]
        
        # read the source data file
        for imageRecord in open(filepath):
            if Header:
				# avoid headers
                Header = False
            else:
                Currentfeature=[]
                imageCurrentRow = imageRecord.split(',')
                # get the x feature and the Y prediction values
                if(str(class_id) == imageCurrentRow[0]):
                    features = imageCurrentRow[1].split()
                    for f in features:
                        Currentfeature.append(int(f))
                    X_Features.append(Currentfeature)
                    
                    #X_Features.append([int(p) for p in imageCurrentRow[1].split()])
                    
                    pred = imageCurrentRow[0]
                    Y_predict.append(int(pred))
                
        # normalization
        X_Features =  np.array(X_Features)  / 255.0
        Y_predict = np.array(Y_predict)
        
        return X_Features,Y_predict
    
    def ClassImbalanace(self, X_Features,Y_predict):
        # class 1 has very few training samples as compared to other classes
		# replicating each sample 10 times for class 1
        xNotOfClass1 = X_Features[Y_predict!=1,:]
        yNotOfClass1 = Y_predict[Y_predict!=1]
        
        xOfClass1 = X_Features[Y_predict==1]
        yOfClass1 = Y_predict[Y_predict==1,:]
		# repetitive selection
        XNew = np.repeat(xOfClass1,10,axis=0)
        
        X_Features = np.vstack([xNotOfClass1,XNew])
        Y_predict = np.concatenate((yNotOfClass1,[1]*len(XNew)))
        
        return X_Features,Y_predict
    
    
    def DeepNeuralNet(self, X_Features,Y_predict,X_Features_Validation,Y_predict_Validation):
        best_validation_error = float("inf")
        costs = []
        print("Started Deep Neaural Net learning..........")
        # shuffling
        X_Features, Y_predict = shuffle(X_Features, Y_predict)
        
        NumberOfImages, D_NumberOfFeatureColumns = X_Features.shape
        K_classesOfPrediction = len(set(Y_predict))
        
        # one-hot encoding
        Y_Indicator = np.zeros((len(Y_predict), len(set(Y_predict))))
        for i in range(len(Y_predict)):
            Y_Indicator[i, Y_predict[i]] = 1    
        T_targetClassOfPrediction = Y_Indicator
        
        # hidden and output layer weights initialized using gaussian and normalized
        self.HiddenLayersWeights = np.random.randn(D_NumberOfFeatureColumns,self.hiddenLayers) / np.sqrt(D_NumberOfFeatureColumns+self.hiddenLayers)
        self.OutPutLayerWeights =  np.random.randn(self.hiddenLayers,K_classesOfPrediction) / np.sqrt(self.hiddenLayers+K_classesOfPrediction)
        # initial bias to zero
        self.BiasForHiddenLayers = np.zeros(self.hiddenLayers)
        self.BiasForOutPutLayer = np.zeros(K_classesOfPrediction)  
        
        #deep neural net training #epoch times
        for i in range(self.epochs):        
            # Forward propagation        
            Forward_prop_result = self.forwardPropagation(X_Features)
            # last layer results
            LastlayerResults = Forward_prop_result.dot(self.OutPutLayerWeights) + self.BiasForOutPutLayer
            # softmax
            FinalClassificationResults = self.softmax(LastlayerResults)
            
			# error in forward propagation
            errorInForwardProp =  FinalClassificationResults - T_targetClassOfPrediction
            # backward propagation using chain rule 
            # Gradient descent
            self.OutPutLayerWeights = self.OutPutLayerWeights - self.learningRate*(Forward_prop_result.T.dot(errorInForwardProp) + self.reg*self.OutPutLayerWeights)
            self.BiasForOutPutLayer = self.BiasForOutPutLayer - self.learningRate*(errorInForwardProp.sum(axis=0) + self.reg*self.BiasForOutPutLayer)
            # activation function using tanh
            dZ = errorInForwardProp.dot(self.OutPutLayerWeights.T) * (1 - Forward_prop_result*Forward_prop_result) # tanh 
            #dZ = errorInForwardProp.dot(self.OutPutLayerWeights.T) * (Forward_prop_result > 0) # relu 
            # updation of hidden layer weights and bias
            self.HiddenLayersWeights = self.HiddenLayersWeights - self.learningRate*(X_Features.T.dot(dZ) + self.reg*self.HiddenLayersWeights)
            self.BiasForHiddenLayers = self.BiasForHiddenLayers - self.learningRate*(dZ.sum(axis=0) + self.reg*self.BiasForHiddenLayers)        
            
			# checking progress each 50th iteration to verify how well model is predicting
            if i % 50 == 0:
                # Forward propagation        
                Forward_prop_result = self.forwardPropagation(X_Features_Validation)
                # last layer results
                LastlayerResults = Forward_prop_result.dot(self.OutPutLayerWeights) + self.BiasForOutPutLayer
                # softmax
                FinalClassificationResults = self.softmax(LastlayerResults)
                
                N = len(Y_predict_Validation)
				# loss function using cross entropy
                costResult = -np.log(FinalClassificationResults[np.arange(N), Y_predict_Validation]).mean()
                
				# storing costs for plotting later
                costs.append(costResult)
                
                # predicted output based on maximum probability
                predictedResult = np.argmax(FinalClassificationResults, axis=1)
                error = self.Calcualte_errorRate(Y_predict_Validation, predictedResult)
                print("Iteration i:", i, " ||cost_function value :", costResult, " ||error Value:", error)
                
                # best validation error
                if error < best_validation_error:
                    best_validation_error = error
        print("best_validation_error:"+str(best_validation_error))
        np.savez("final_weights_bias.npz", output_weight=self.OutPutLayerWeights, hidden_weight=self.HiddenLayersWeights, output_bias=self.BiasForOutPutLayer, hidden_bias=self.BiasForHiddenLayers)
        plt.plot(costs)
        plt.show()        
    
                
    def Calcualte_errorRate(self, Yvalid, predictedResult):
        # error = # of wrong predictions OR 1 - # of correct predictions
        ErrorVal = 1 - np.mean(Yvalid == predictedResult)
        return ErrorVal
        
    def relu(self,x):
        # condition check return 0 or 1
        # we will pass values only greter than 0 and all others converted to 0
        return x * (x > 0)

    def tanh_f(self, x):
        return np.tanh(x)
        
    def softmax(self,layerResults):
        # softmax formula (stated in the report)
        return np.exp(layerResults)/ np.exp(layerResults).sum(axis=1,keepdims=True)
    
        
    def forwardPropagation(self,xfeatures):
        Z_resultsUntilLastLayer = self.tanh_f(xfeatures.dot(self.HiddenLayersWeights) + self.BiasForHiddenLayers)
        return Z_resultsUntilLastLayer 
    
    def test_classifier(self):
        weights = np.load("final_weights_bias.npz")
        self.OutPutLayerWeights = weights['output_weight']
        self.HiddenLayersWeights = weights['hidden_weight']
        self.BiasForOutPutLayer = weights['output_bias']
        self.BiasForHiddenLayers = weights['hidden_bias']
        confusion_matrix = np.zeros((7,7))
        for i in range(0,4):
            X_Features,_ = self.LoadTestData(obj.testing, i)
            Forward_prop_result = self.forwardPropagation(X_Features)
            LastlayerResults = Forward_prop_result.dot(self.OutPutLayerWeights) + self.BiasForOutPutLayer
            FinalClassificationResults =  self.softmax(LastlayerResults)
            predictedResult = np.argmax(FinalClassificationResults, axis=1)  # b
            #print(predictedResult)
            for item in predictedResult:
                confusion_matrix[i,item] = str(int(confusion_matrix[i,item]) + 1)                                                              
        print(confusion_matrix)

    
if __name__ == '__main__':

    obj = FaceExp_dnn()
    X_Features,Y_predict = obj.LoadData(obj.training)
    print("Data is loaded")
	
	# solving class imbalance
    X_Features,Y_predict = obj.ClassImbalanace(X_Features,Y_predict)
	
    # validation data to check classifier performance each 50th iteration
    X_Features_Validation,Y_predict_Validation = obj.LoadData(obj.validation)
    
    # training deep neural network
    obj.DeepNeuralNet(X_Features,Y_predict,X_Features_Validation,Y_predict_Validation)
    obj.test_classifier()
