import pandas as pd
import numpy as np
from collections import Counter
from math import *
from decimal import Decimal

class my_KNN:

    def __init__(self, n_neighbors=5, metric="minkowski", p=2):
        # metric = {"minkowski", "euclidean", "manhattan", "cosine"}
        # p value only matters when metric = "minkowski"
        # notice that for "cosine", 1 is closest and -1 is furthest
        # therefore usually cosine_dist = 1- cosine(x,y)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.p = p


    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        self.X = X
        self.y = y
        return

    def my_p_root(self, value, root):
        my_root_value = 1 / float(root)
        return round(Decimal(value) **
                     Decimal(my_root_value), 3)

    def dist(self,x):
        # Calculate distances of training data to a single input data point (distances from self.X to x)
        # Output np.array([distances to x])
        distances = list()
        rowIndex=0
        for train_row in self.X.to_numpy():
            distance = 0.0
            if self.metric == "minkowski":
                distance = self.my_p_root(sum(pow(abs(a-b), self.p) for a, b in zip(train_row, x)), self.p)

            elif self.metric == "euclidean":
                distance = sqrt(sum(pow(a-b, 2) for a, b in zip(train_row, x)))


            elif self.metric == "manhattan":
                distance = sum(abs(a-b) for a, b in zip(train_row, x))


            elif self.metric == "cosine":
                distance =  np.dot(train_row, x) / (np.sqrt(np.dot(train_row, x)) * np.sqrt(np.dot(train_row, x)))


            else:
                raise Exception("Unknown criterion.")
            distances.append((self.y[rowIndex], distance))
            rowIndex = rowIndex + 1
        return distances

    def k_neighbors(self,x):
        # Return the stats of the labels of k nearest neighbors to a single input data point (np.array)
        # Output: Counter(labels of the self.n_neighbors nearest neighbors)
        distances = self.dist(x)
        distances.sort(key=lambda tup: tup[1], reverse=True)
        neighbors = list()
        for i in range(self.n_neighbors):
            neighbors.append(distances[i][0])


        return Counter(neighbors)

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        probs = self.predict_proba(X)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]
        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        probs = []
        try:
            X_feature = X[self.X.columns]
        except:
            raise Exception("Input data mismatch.")

        for x in X_feature.to_numpy():
            neighbors = self.k_neighbors(x)
            probs.append({key: neighbors[key] / float(self.n_neighbors) for key in self.classes_})
        probs = pd.DataFrame(probs, columns=self.classes_)
        return probs
