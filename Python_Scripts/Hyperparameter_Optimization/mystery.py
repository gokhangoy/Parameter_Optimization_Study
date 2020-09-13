#Import the Necessary Libraries
##############################################################################################################

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import OneHotEncoder 
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from talos.utils import lr_normalizer
from keras.utils import multi_gpu_model
import keras.backend.tensorflow_backend as tfback
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.optimizers import Adam, Nadam
from keras.optimizers import RMSprop
from keras.activations import softmax
from keras.losses import categorical_crossentropy, logcosh
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
##############################################################################################################

#Preprocessing  --> Data Preperation

##############################################################################################################

myTrainData = pd.read_csv("/content/drive/HyperparameterOptimization/trainData2.csv",sep=";")
h_myTrainYData = pd.read_csv("/content/drive/HyperparameterOptimization/HyperParameterOptimizationYData.csv", sep=";")
h_myTrainYData_Array = h_myTrainYData.to_numpy()
h_myTrainYData_Array = h_myTrainYData_Array.astype(np.float)
firstColumn = myTrainData.iloc[:,0]
secondColumn = myTrainData.iloc[:,1]
thirdColumn = myTrainData.iloc[:,2]
fourthColumn = myTrainData.iloc[:,3]
fifthColumn = myTrainData.iloc[:,4]
sixthColumn = myTrainData.iloc[:,5]
seventhColumn = myTrainData.iloc[:,6]
eighthColumn = myTrainData.iloc[:,7]
ninethColumn = myTrainData.iloc[:,8]



deneme = np.stack(firstColumn.values.reshape(len(firstColumn),1))

#First Find The Different Labels with Label Encoder

##############################################################################################################
label_encoder = LabelEncoder()
fifthColumn_Enc = label_encoder.fit_transform(fifthColumn)
sixthColumn_Enc = label_encoder.fit_transform(sixthColumn)
seventhColumn_Enc = label_encoder.fit_transform(seventhColumn)
##############################################################################################################



#Then Use One Hot Encoding To Be Able To Use Machine Learning Techniques
##############################################################################################################
onehot_encoder = OneHotEncoder(sparse=False)
fifthColumn_Enc = fifthColumn_Enc.reshape(len(fifthColumn_Enc), 1)
fifth_Onehot_encoded = onehot_encoder.fit_transform(fifthColumn_Enc)

sixthColumn_Enc = sixthColumn_Enc.reshape(len(sixthColumn_Enc), 1)
sixth_Onehot_encoded = onehot_encoder.fit_transform(sixthColumn_Enc)


seventhColumn_Enc = seventhColumn_Enc.reshape(len(seventhColumn_Enc), 1)
seventh_Onehot_encoded = onehot_encoder.fit_transform(seventhColumn_Enc)

##############################################################################################################


#Merge the Columns
##############################################################################################################

preparedData = np.column_stack((firstColumn,secondColumn, thirdColumn,fourthColumn,fifth_Onehot_encoded, sixth_Onehot_encoded,seventh_Onehot_encoded,eighthColumn,ninethColumn))
##############################################################################################################

# Split the dataset(80%-20%)
##############################################################################################################
array = preparedData
X = array[:,0:605]
Y = array[:,605]

x_train, x_validation, y_train, y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)

##############################################################################################################




# Function to create model, required for KerasClassifier


def create_model(dropout_rate=0.0, weight_constraint=0):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=x_train.shape[1], kernel_initializer='uniform', activation='softplus'))
	model.add(Dropout(dropout_rate))
	model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
	#optimizer = RMSprop(lr=learn_rate)#learning rate and momentum optimization
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
	return model
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# create model
model = KerasClassifier(build_fn=create_model,epochs=50,batch_size=40, verbose=0)

# define the grid search parameters
#############################################################################################################
#Batch Size && Epochs Optimization
#Function Declaration should be like this : 

#def create_model():
	# create model
#	model = Sequential()
#	model.add(Dense(12, input_dim=8, activation='relu'))
#	model.add(Dense(1, activation='sigmoid'))
#	# Compile model
#	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#	return model

#batch_size = [10, 20, 40] batch_size and epochs optimization
#epochs = [10, 50] batch_size and epochs optimization
#param_grid = dict(batch_size=batch_size, epochs=epochs)
#############################################################################################################

#############################################################################################################
#Optimizer Optimization
#Function declaration should be like this  -->

#def create_model(optimizer='adam'):
	# create model
#	model = Sequential()
#	model.add(Dense(12, input_dim=8, activation='relu'))
#	model.add(Dense(1, activation='sigmoid'))
	# Compile model
#	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#	return model

#optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'] optimizer optimization
#param_grid =dict(optimizer=optimizer) optimizer optimization
##############################################################################################################

##############################################################################################################
#Activation Function Optimization
#Function declaration should be like this  -->

#def create_model(activation='relu'):
	# create model
#	model = Sequential()
#	model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation=activation))
#	model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
	# Compile model
#	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#	return model

#activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'] 
#param_grid = dict(activation=activation) 

##############################################################################################################
#Learning Rate && Momentum Optimization 
#Function declaration should be like this  --> 

#def create_model(learn_rate=0.01, momentum=0):
	# create model
#	model = Sequential()
#	model.add(Dense(12, input_dim=8, activation='relu'))
#	model.add(Dense(1, activation='sigmoid'))
	# Compile model
#	optimizer = SGD(lr=learn_rate, momentum=momentum)
#	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#	return model

#learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3] 
#momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
#param_grid =dict(learn_rate=learn_rate, momentum=momentum) 
##############################################################################################################


##############################################################################################################
#Dropout Rate Optimization 
#Function declaration should be like this  --> 

#def create_model(dropout_rate=0.0, weight_constraint=0):
	# create model
#	model = Sequential()
#	model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='linear', kernel_constraint=maxnorm(weight_constraint)))
#	model.add(Dropout(dropout_rate))
#	model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
	# Compile model
#	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#	return model

#weight_constraint = [1, 2, 3, 4, 5] 
#dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
##############################################################################################################



weight_constraint = [1, 2, 3, 4, 5] 
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)



#Call The Function and Fit the Model.
##############################################################################################################
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
grid_result = grid.fit(X, Y)
##############################################################################################################

# Print Out The Results
##############################################################################################################
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
##############################################################################################################	