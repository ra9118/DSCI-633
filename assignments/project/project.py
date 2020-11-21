#
# Use this test client to make sure from the model accuracy
# by split data into 70% training and 30% testing
# Then measure model accuracy
#
import pandas as pd
import time
from TextNLP import  TextNLP
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

textNLP = TextNLP()

class my_model():

    def fit(self, X, y):
        print( "=============== Training the model ( takes up to 5 minutes )====================")
        X_train = textNLP.pre_pro_Cols(X)
        [X_train_balance, y_balance] = textNLP.balanceData(X_train,y)
        X_norm = min_max_scaler.fit_transform(X_train_balance)

        # Create Decision Tree classifer object
        self.clf = SGDClassifier()
        self.clf.fit(X_norm, y_balance)


    def predict(self, X):
        print("=============== Testing the model ====================")
        # remember to apply the same preprocessing in fit() on test data before making predictions
        X_test = textNLP.pre_pro_Cols(X)
        X_norm = min_max_scaler.fit_transform(X_test)
        return self.clf.predict(X_norm)


if __name__ == "__main__":
    print("Start loading dataset ....")
    start = time.time()
    # Load data
    data = pd.read_csv("../data/job_train.csv")
    # Replace missing values with empty strings
    data = data.fillna("")

    y = data["fraudulent"]
    X = data.drop(['fraudulent'], axis=1)

    clf = my_model()
    clf.fit(X, y)
    runtime = (time.time() - start) / 60.0
    print("Run time:{}".format(runtime))

    predictions = clf.predict(X)
    print(predictions)






