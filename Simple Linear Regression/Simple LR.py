# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 20:53:11 2019

@author: Navnit Singh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datasets=pd.read_csv("D:\\ML\\SimpleLR\\Salary_Data.csv")
X=datasets.iloc[:,:-1].values
Y=datasets.iloc[:,1]

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

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, Y_train)
Y_pred=regressor.predict(X_test)

plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.show()

plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.show()

"""
data=np.array([15])
data=data.reshape(1,1)

z=regressor.predict(data)
"""
