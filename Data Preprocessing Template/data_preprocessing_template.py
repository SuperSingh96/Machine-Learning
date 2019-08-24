# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 16:50:53 2019

@author: Navnit Singh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datasets=pd.read_csv("D:\\ML\\1\\Data.csv")
X=datasets.iloc[:,:-1].values
Y=datasets.iloc[:,3]

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
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2)