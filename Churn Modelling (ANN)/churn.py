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
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Building the ANN
# Initializing the ANN

ann = tf.keras.models.Sequential()

# First Layer and Hidden Layer
ann.add(tf.keras.layers.Dense(units = 6,activation = 'relu'))

# Second Hidden Layer
ann.add(tf.keras.layers.Dense(units = 6,activation = 'relu'))

# Output Layer
ann.add(tf.keras.layers.Dense(units = 1,activation = 'sigmoid'))


# Training The ANN

# Compiling the ann
ann.compile(optimizer ='adam' , loss ='binary_crossentropy' ,metrics=['accuracy'])

# Training the ANN
ann.fit(X_train,y_train,batch_size=32,epochs = 100)

# Making Predictions and Evaluating the model

# Prdicting the result of a single operation
ann.predict([])















