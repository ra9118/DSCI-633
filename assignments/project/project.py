#
# Use this test client to make sure from the model accuracy
# by split data into 70% training and 30% testing
# Then measure model accuracy
#
import pandas as pd
import time
from TextNLP import  TextNLP
from sklearn.tree import DecisionTreeClassifier

textNLP = TextNLP()

class my_model():

    def fit(self, X, y):
        print( "=============== Training the model ( takes up to 5 minutes )====================")
        [processed_docs, X_clean] = textNLP.pre_pro_Cols(X)
        [dictionary, corpus_tfidf, bow_corpus] = textNLP.tf_idf(processed_docs)
        self._lda_model = textNLP.lda(corpus_tfidf, dictionary)
        X_train = textNLP.apply_lda(self._lda_model, bow_corpus, X_clean)
        [X_train_balance, y_balance] = textNLP.balanceData(X_train,y)
        # Create Decision Tree classifer object
        self.clf = DecisionTreeClassifier()
        print("(Step 6 of 6)  Train Decision Tree Classifer")
        self.clf.fit(X_train_balance, y_balance)


    def predict(self, X):
        print("=============== Testing the model ====================")
        # remember to apply the same preprocessing in fit() on test data before making predictions
        [processed_docs, X_clean] = textNLP.pre_pro_Cols(X)
        [dictionary, corpus_tfidf, bow_corpus] = textNLP.tf_idf(processed_docs)
        print("(Step 4 of 6)  using existing {} LDA Topics ....".format(textNLP._numberOfTopics))
        #self._lda_model = textNLP.lda(corpus_tfidf, dictionary)
        X_test = textNLP.apply_lda(self._lda_model, bow_corpus, X_clean)
        print("(Step 6 of 6)  predictions using Decision Tree Classifer")
        return self.clf.predict(X_test)


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






