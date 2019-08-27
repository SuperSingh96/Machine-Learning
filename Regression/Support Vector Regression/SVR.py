# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 14:55:03 2019

@author: Navnit Singh
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datasets=pd.read_csv("D:\\ML\\Support Vector Regression\\Position_Salaries.csv")
X=datasets.iloc[:,1:2].values
Y=datasets.iloc[:,2].values
Y=Y.reshape(Y.shape[0],1)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_Y=StandardScaler()
X=sc_X.fit_transform(X)
Y=sc_Y.fit_transform(Y)

from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,Y)

V=np.array([6.5])
V=V.reshape(1,1)
y_pred=sc_Y.inverse_transform(regressor.predict(sc_X.transform(V)))


#X_grid=np.arange(min(X), max(X), 0.01)
#X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y,color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.show()

