# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 13:42:28 2019

@author: Navnit Singh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datasets=pd.read_csv("D:\\ML\\Polynomial Regression\\Position_Salaries.csv")
X=datasets.iloc[:,1:2].values
Y=datasets.iloc[:,2].values

"""
#X[:,1:3]=imputer.transform(X[:,1:3])
from sklearn.preprocessing import Imputer
imp=Imputer(missing_values="NaN", strategy="mean", axis=0)
X[:,1:3]=imp.fit_transform(X[:,1:3])

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le_x=LabelEncoder()
X[:,0]=le_x.fit_transform(X[:,0])
oe=OneHotEncoder(categorical_features=[0])
X=oe.fit_transform(X).toarray()
le_y=LabelEncoder()
Y=le_y.fit_transform(Y)
"""
"""
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2)
"""

from sklearn.linear_model import LinearRegression
len_reg=LinearRegression()
len_reg.fit(X,Y)

from sklearn.preprocessing import PolynomialFeatures
Poly_reg=PolynomialFeatures(degree=4)
X_poly=Poly_reg.fit_transform(X)
#Poly_reg.fit(X_poly, Y)
len_reg2=LinearRegression()
len_reg2.fit(X_poly, Y)

plt.scatter(X,Y,color='red')
plt.plot(X, len_reg.predict(X), color='blue')
plt.show()

X_grid=np.arange(min(X), max(X), 0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y,color='red')
plt.plot(X_grid, len_reg2.predict(Poly_reg.fit_transform(X_grid)), color='blue')
plt.show()

V=np.array([6.5])
V=V.reshape(1,1)

len_reg.predict(V)

len_reg2.predict(Poly_reg.fit_transform(V))



