# importing libraries

import numpy as np
import pandas as pd
import tensorflow as tf

# Data Preprocessing

# Importing Dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,-1].values
# print(X)
# print(y)

# Encoding Categorical Data

# Label Encoding Gender Column

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])

# print(X)

# One Hot Encoding the Geography Column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder',OneHotEncoder(),[1])],remainder ='passthrough')
X = np.array(ct.fit_transform(X))
# print(X)

# Splitting the dataset into Training Set and Test Set











