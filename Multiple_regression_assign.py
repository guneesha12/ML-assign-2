#data preprocessing

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
trainset=pd.read_csv('Train.csv')
testset=pd.read_csv('Test.csv')
X_train=trainset.iloc[:, 0:5].values
y_train=trainset.iloc[:,5].values
X_test=testset.iloc[:,0:5].values


from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)

