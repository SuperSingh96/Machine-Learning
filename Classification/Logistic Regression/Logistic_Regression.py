# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 20:34:56 2019

@author: Navnit Singh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datasets=pd.read_csv("D:\\ML\\Logistic Regression\\Social_Network_Ads.csv")
X=datasets.iloc[:,[2,3]].values
Y=datasets.iloc[:,4].values

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
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)

Y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test, Y_pred)




