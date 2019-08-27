# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 23:32:37 2019

@author: Navnit Singh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("D:\\ML\\Hierarchical Clustering\\Mall_Customers.csv")
X=dataset.iloc[:,[3,4]].values

import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc=hc.fit_predict(X)

plt.scatter(X[y_hc==0,0],X[y_hc==0,1], s=100, c='red')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1], s=100, c='blue')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1], s=100, c='green')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1], s=100, c='cyan')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1], s=100, c='magenta')
plt.show()
