import pandas as pd
import numpy as np
np.set_printoptions(threshold=1)

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.combine import SMOTETomek

churnPredictionDataset = pd.read_csv('Categorical_Only_First_Month.csv', sep=';', decimal=',')

churnPredictionDataset=churnPredictionDataset.dropna()
print(churnPredictionDataset.shape)
print(len(churnPredictionDataset.columns))

input=churnPredictionDataset.iloc[:,0:-1]
output=churnPredictionDataset.iloc[:,-1]
###print(input)

smt = SMOTETomek(sampling_strategy=0.05, random_state=42)

nFV , labels = smt.fit_resample(input,output)

###print(nFV.shape)
print(labels)

result=np.append(nFV,labels[:,None],axis=1)
np.savetxt("SMOTETomekLast.csv", result, delimiter=",",fmt='%s')

##print(result.shape)
