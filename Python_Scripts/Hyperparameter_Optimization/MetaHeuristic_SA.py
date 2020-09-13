from hyperactive import SimulatedAnnealingOptimizer
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import OneHotEncoder 
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from keras.utils import multi_gpu_model
import keras.backend.tensorflow_backend as tfback
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.optimizers import Adam, Nadam
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


myTrainData = pd.read_csv("/content/drive/HyperparameterOptimization/trainData2.csv",sep=";")
h_myTrainYData = pd.read_csv("/content/drive/HyperparameterOptimization/HyperParameterOptimizationYData.csv", sep=";")
h_myTrainYData_Array = h_myTrainYData.to_numpy()
h_myTrainYData_Array = h_myTrainYData_Array.astype(np.float)

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
#print(fifthColumn)




label_encoder = LabelEncoder()
fifthColumn_Enc = label_encoder.fit_transform(fifthColumn)
sixthColumn_Enc = label_encoder.fit_transform(sixthColumn)
seventhColumn_Enc = label_encoder.fit_transform(seventhColumn)

onehot_encoder = OneHotEncoder(sparse=False)
fifthColumn_Enc = fifthColumn_Enc.reshape(len(fifthColumn_Enc), 1)
fifth_Onehot_encoded = onehot_encoder.fit_transform(fifthColumn_Enc)

sixthColumn_Enc = sixthColumn_Enc.reshape(len(sixthColumn_Enc), 1)
sixth_Onehot_encoded = onehot_encoder.fit_transform(sixthColumn_Enc)


seventhColumn_Enc = seventhColumn_Enc.reshape(len(seventhColumn_Enc), 1)
seventh_Onehot_encoded = onehot_encoder.fit_transform(seventhColumn_Enc)




preparedData = np.column_stack((firstColumn,secondColumn, thirdColumn,fourthColumn,fifth_Onehot_encoded, sixth_Onehot_encoded,seventh_Onehot_encoded,eighthColumn,ninethColumn))

# Split-out validation dataset
array = preparedData
X = array[:,0:605]
Y = array[:,605]
#X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)

(x_train_MNIST, y_train_MNIST), (x_test_MNIST, y_test_MNIST) = train_test_split(X, Y, test_size=0.20, random_state=1)

print(X_train)
print(Y_train)

#x_train_MNIST = x_train_MNIST.reshape(60000, 28, 28, 1)
#x_test_MNIST = x_test_MNIST.reshape(10000, 28, 28, 1)



#y_train_MNIST = to_categorical(y_train_MNIST)
#y_test_MNIST = to_categorical(y_test_MNIST)


# this defines the structure of the model and the search space in each layer
search_config = {
    "keras.compile.0": {"loss": ["categorical_crossentropy"], "optimizer": ["adam"]},
    "keras.fit.0": {"epochs": [20], "batch_size": [10], "verbose": [2]},
 #   "keras.layers.Conv2D.1": {
 #       "filters": [32, 64, 128],
  #      "kernel_size": range(3, 4),
   #     "activation": ["relu"],
   #     "input_shape": [(605, 605, 1)],
   # },
    #"keras.layers.MaxPooling2D.2": {"pool_size": [(2, 2)]},
    #"keras.layers.Conv2D.3": {
     #   "filters": [16, 32, 64],
      #  "kernel_size": [3],
       # "activation": ["relu"],
    #},
    #"keras.layers.MaxPooling2D.4": {"pool_size": [(2, 2)]},
    #"keras.layers.Flatten.1": {},
    "keras.layers.Dense.2": {"units": range(30, 200, 10), "activation": ["softmax"]},
    "keras.layers.Dropout.3": {"rate": list(np.arange(0.4, 0.8, 0.1))},
    "keras.layers.Dense.4": {"units": [1], "activation": ["softmax"]},
}

Optimizer = SimulatedAnnealingOptimizer(search_config, n_iter=10)

# search best hyperparameter for given data
Optimizer.fit(x_train_MNIST, y_train_MNIST)

# predict from test data
prediction = Optimizer.predict(x_test_MNIST)

# calculate accuracy score
score = Optimizer.score(x_test_MNIST, y_test_MNIST)