import pandas as pd
import numpy as np
from copy import deepcopy
from pdb import set_trace
import math

class my_AdaBoost:

    def __init__(self, base_estimator = None, n_estimators = 50,learning_rate=1):
        # Multi-class Adaboost algorithm (SAMME)
        # base_estimator: the base classifier class, e.g. my_DT
        # n_estimators: # of base_estimator rounds
        self.base_estimator = base_estimator
        self.n_estimators = int(n_estimators)
        self.estimators = [deepcopy(self.base_estimator) for i in range(self.n_estimators)]
        self.learning_rate_ = learning_rate

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str

        self.classes_ = list(set(list(y)))
        k = len(self.classes_)
        n = len(y)  # number of samples
        w = np.array([1.0 / n] * n) # init weights for samples
        labels = np.array(y)
        self.alpha = []  # the weight of this tree
        for i in range(self.n_estimators):
            # Sample with replacement from X, with probability w
            sample = np.random.choice(n, n, p=w)
            # Train base classifier with sampled training data
            sampled = X.iloc[sample]
            sampled.index = range(len(sample))
            self.estimators[i].fit(sampled, labels[sample])
            predictions = self.estimators[i].predict(X)
            diffs = np.array(predictions) != y
            # Compute error rate and alpha for estimator i
            error = np.sum(diffs * w)
            while error >= (1 - 1.0 / k):
                w = np.array([1.0 / n] * n)
                sample = np.random.choice(n, n, p=w)
                # Train base classifier with sampled training data
                sampled = X.iloc[sample]
                sampled.index = range(len(sample))
                self.estimators[i].fit(sampled, labels[sample])
                predictions = self.estimators[i].predict(X)
                diffs = np.array(predictions) != y
                # Compute error rate and alpha for estimator i
                error = np.sum(diffs * w)

            # Compute alpha for estimator i (don't forget to use k for multi-class)
            alpha = self.learning_rate_ * math.log((1.0 - error) / error) + np.exp(k-1) # 1 is learning rate
            self.alpha.append(alpha)

            # Update wi
            # Missclassified samples gets larger weights and correctly classified samples smaller
            w1 = []
            for w, isError in zip(w, diffs):
                if isError:
                    w1.append(w * np.exp(alpha))  # change weight it got wrong predication
                else:
                    w1.append(w)  # keep weight got correct predication
            w = np.array(w1)

           # w *= np.exp((1-alpha)/alpha)+  np.exp(k-1)
            # Normalize to one
            w /= np.sum(w)

        # Normalize alpha
        self.alpha = self.alpha / np.sum(self.alpha)
        return

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        probs = self.predict_proba(X)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]
        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob: what percentage of the base estimators predict input as class C
        # prob(x)[C] = sum(alpha[j] * (base_model[j].predict(x) == C))
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # write your code below


        probs = []

        # find predictions for every tree and every row of data
        listOfPredictions = []
        for j in range(self.n_estimators):
                    predictions = self.estimators[j].predict(X)
                    listOfPredictions.append(predictions)

        dflistOfPredictions = pd.DataFrame(listOfPredictions)

        # find decsion by voting
        for col in range(dflistOfPredictions.shape[1]):

             # z init class
             pClass = dict()
             for className in self.classes_:
                pClass[className] = 0

             for row in range(dflistOfPredictions.shape[0]):
                classNameInRow = dflistOfPredictions.iloc[row, col]
                pClass[classNameInRow] = pClass[classNameInRow]+ self.alpha[row]

             probs.append({className:pClass[className] for className in self.classes_})


        probs = pd.DataFrame(probs, columns=self.classes_)
        return probs




