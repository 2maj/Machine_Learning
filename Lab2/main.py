# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 14:54:58 2022

@author: adjim
"""
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def knn(X, Y):
# Create KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)
# Fit the classifier to the data
    knn.fit(X, Y)
