import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder 
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
# Load libraries
import talos
from talos.utils import lr_normalizer
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


deneme = np.stack(firstColumn.values.reshape(len(firstColumn),1))


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
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)
print(Y.dtype)
####print(len(y[1]))

# Spot Check Algorithms
#models = []
#models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
#models.append(('NB', GaussianNB()))
#models.append('RF',RandomForestClassifier(max_depth=2, random_state=0))
##models.append(('LDA', LinearDiscriminantAnalysis()))
##models.append(('KNN', KNeighborsClassifier()))
##models.append(('CART', DecisionTreeClassifier()))
##models.append(('SVM', SVC(gamma='auto')))
#print("dsfs")
## evaluate each model in turn
#results = []
#names = []
#for name, model in models:
#  kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#  cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
#  results.append(cv_results)
#  names.append(name)
#  print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
#  
#  
#  
#  
#

def _get_available_gpus():  

    if tfback._LOCAL_DEVICES is None:  
        devices = tf.config.list_logical_devices()  
        tfback._LOCAL_DEVICES = [x.name for x in devices]  
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]


tfback._get_available_gpus = _get_available_gpus
  
def logisticRegressionModel(x_train, y_train, x_val, y_val, params):

 model = multi_gpu_model(logisticRegressionModel, gpus=2)
 model.add(Dense(params['first_neuron'], input_dim=x_train.shape[1],activation='relu'))
 model.add(Dropout(params['dropout']))
 model.add(Dense(y_train.shape[1], activation=params['last_activation']))

 model.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['accuracy'])
 out = model.fit(x_train, y_train, batch_size=params['batch_size'], epochs=params['epochs'], verbose=0, validation_data=[x_val, y_val])


 return out, model

p = {'first_neuron':[4,8],
'batch_size': [2],
'dropout': (0, 0.40, 10),
'epochs': [100],
'loss': ['categorical_crossentropy'],
'last_activation': ['softmax']}


scan_object = talos.Scan(X,h_myTrainYData_Array,params=p,model=logisticRegressionModel,experiment_name='LRTuning')


print(scan_object.data.head())
print(scan_object.learning_entropy)
print(scan_object.details)

analyze_object = talos.Analyze(scan_object)
print(analyze_object.data)
print("Second Result")
print(analyze_object.rounds())
print("Third Result")
print(analyze_object.high('val_acc'))
print("Fourth Result")
print(analyze_object.rounds2high('val_acc'))
print("Fifth Result")
print(analyze_object.best_params('val_acc', ['acc', 'loss', 'val_loss']))
print("Sixth Result")
print(analyze_object.correlate('val_loss', ['acc', 'loss', 'val_loss']))