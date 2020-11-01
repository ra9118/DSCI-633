import pandas as pd
import time
from my_KNN import my_KNN
from sklearn.model_selection import train_test_split
import numpy as np
from TextNLP import  TextNLP
import gensim
from gensim import corpora, models
from my_evaluation import my_evaluation

if __name__ == "__main__":
    print("Start loading dataset ....")
    start = time.time()
    # Load data
    data = pd.read_csv("../data/job_train.csv")
    # Replace missing values with empty strings
    data = data.fillna("")
    y = data["fraudulent"]
    X = data.drop(['fraudulent','title','location','description','requirements'], axis=1)

    # pre process the
    print("Pre-proces coloums that has text (token,  remove stop word,remove  non alphabetic characters, and Lemmatizer....")
    X_pre = data[['title','location','description','requirements']]
    textNLP = TextNLP()
    textData = []
    for index, row in X_pre.iterrows():
        text = "{} {} {} {}" .format(row['title'],row['location'], row['description'], row['requirements'])
        textData.append(text)

    documents = pd.DataFrame(textData,columns=['headline_text'])
    processed_docs = documents['headline_text'].map(textNLP.textPreProcessing)

    print("Find how many times words appear ....")
    dictionary = gensim.corpora.Dictionary(processed_docs)
    # count = 0
    # for k, v in dictionary.iteritems():
    #     print(k, v)
    #     count += 1
    #     if count > 10:
    #         break

    print("Building Bag of Words dataset ....")
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    #TF-IDF
    # print("Apply TF-IDF")
    # tfidf = models.TfidfModel(bow_corpus)
    # corpus_tfidf = tfidf[bow_corpus]


    numberOfTopics = 10
    print("Starte generate {} LDA Topics ....".format(numberOfTopics))
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=numberOfTopics, id2word=dictionary, passes=2, workers=2)
    for idx, topic in lda_model.print_topics():
        print('Topic: {} \nWords: {}'.format(idx, [float(w.split("*")[0]) for w in topic.split("+")]))
        X["Topic {}".format(idx)] = np.nan
    #     print("=====")

    # Find LDA for every document
    print("\nFind LDA topic for every document ....")
    for row_index in range(len(bow_corpus)):
        for index, score in sorted(lda_model[bow_corpus[row_index]], key=lambda tup: -1 * tup[1]):
            #print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))
            idx = 0
            for w in lda_model.print_topic(index, 10).split("+"):
                X.at[row_index, "Topic {}".format(idx)] = float(w.split("*")[0])
                idx = idx + 1
            #print("=====")

    # info https://github.com/susanli2016/NLP-with-Python/blob/master/LDA_news_headlines.ipynb
    print("split dataset to training and testing")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Train model
    clf = my_KNN()
    clf.fit(X, y)


    print("Predict the lables and probabilities")
    predictions = clf.predict(X_test)
    # Predict probabilities
    probs = clf.predict_proba(X_test)

    runtime = (time.time() - start) / 60.0
    print("Run time:{}".format(runtime))

    print("Print results")
    for i,pred in enumerate(predictions):
        print("Class: %s\tprobs: %f" % (pred, probs[pred][i]))

    print("\n correct ration: {}".format((np.sum(np.array(predictions) == y_test))/len(y_test)))

    # Evaluate results
    metrics = my_evaluation(predictions,y_test, probs)
    result = {}
    for target in clf.classes_:
        result[target] = {}
        result[target]["prec"] = metrics.precision(target)
        result[target]["recall"] = metrics.recall(target)
        result[target]["f1"] = metrics.f1(target)
        result[target]["auc"] = metrics.auc(target)
    print(result)
    f1 = {average: metrics.f1(target=None, average=average) for average in ["macro", "micro", "weighted"]}
    print("Average F1 scores: ")
    print(f1)






