# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 17:32:25 2019

@author: Navnit Singh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datasets=pd.read_csv("D:\\ML\\Decision Tree Regression\\Position_Salaries.csv")
X=datasets.iloc[:,1:2].values
Y=datasets.iloc[:,2].values

from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,Y)

V=np.array([6.5])
V=V.reshape(1,1)
regressor.predict(V)


X_grid=np.arange(min(X), max(X), 0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y,color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.show()
