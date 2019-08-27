# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 21:59:15 2019

@author: Navnit Singh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("D:\\ML\\K-Means Clustering\\Mall_Customers.csv")
X=dataset.iloc[:,[3,4]].values

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.show()

kmeans=KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=0)
y_means=kmeans.fit_predict(X)

plt.scatter(X[y_means==0,0],X[y_means==0,1], s=100, c='red')
plt.scatter(X[y_means==1,0],X[y_means==1,1], s=100, c='blue')
plt.scatter(X[y_means==2,0],X[y_means==2,1], s=100, c='green')
plt.scatter(X[y_means==3,0],X[y_means==3,1], s=100, c='cyan')
plt.scatter(X[y_means==4,0],X[y_means==4,1], s=100, c='magenta')
plt.show()

