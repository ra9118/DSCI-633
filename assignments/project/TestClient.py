import pandas as pd
import time
import numpy as np

from sklearn.model_selection import train_test_split
from  project import  my_model

from sklearn import metrics


if __name__ == "__main__":
    print("Start loading dataset ....")
    start = time.time()
    # Load data
    data = pd.read_csv("../data/job_train.csv" ) #,nrows=1000)
    # Replace missing values with empty strings
    data = data.fillna("")

    # # extract same  samples from both classes
    # classes_ = list(set(list(data["fraudulent"])))
    # print(classes_)

    y = data["fraudulent"]
    X = data.drop(['fraudulent'], axis=1)

    print("split dataset to training and testing")
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y)
    split_point = int(0.8 * len(y))
    X_train = X.iloc[:split_point]
    X_test = X.iloc[split_point:]
    y_train = y.iloc[:split_point]
    y_test = y.iloc[split_point:]


    clf = my_model()
    clf.fit(X_train, y_train)
    runtime = (time.time() - start) / 60.0
    print("Run time:{}".format(runtime))

    predictions = clf.predict(X_test)
    print(metrics.classification_report(y_test, predictions))







