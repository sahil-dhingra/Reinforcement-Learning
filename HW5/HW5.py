# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 21:47:50 2019

@author: sahil.d
"""
import numpy as np
from itertools import combinations

experiences = [[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]]
y = [0, 0, 1, 1, 0, 0, 0, 0]



d = len(experiences[0])
n = len(experiences)

H = np.zeros((d*(d-1), d))
comb = []
i = 0
for c in combinations(list(range(d)), 2):
    H[i][c[0]] =  1
    H[i][c[1]] = -1
    H[i+1][c[1]] =  1
    H[i+1][c[0]] = -1
    i = i + 2

out = np.zeros(n).astype(int)
h = np.ones(H.shape[0])
for i in range(n):
    H_predict = np.dot(H, experiences[i])
    H_active = np.take(H_predict, np.where(h==1))
    predictions = np.unique(H_active)
    if len(predictions) == 1:
        if predictions[0] == -1:
            predictions[0] = 0
        out[i] = predictions[0]
    else:
        h = h*(H_predict == y[i])
        out[i] = -1


#TEST
experiences = [[1, 1], [1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [1, 1]]
y = [0, 1, 0, 0, 0, 1, 0]
