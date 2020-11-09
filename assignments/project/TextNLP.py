
import pandas as pd
import numpy as np
import string

from gensim.utils import lemmatize
from gensim.parsing.preprocessing import remove_stopwords
from gensim import corpora, models

# remove  non alphabetic characters.
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


class TextNLP:

     def __init__(self,pp_colums =['title','location','description','requirements'],outputCol='fraudulent', numberOfTopics =5):
        self._pp_colums = pp_colums
        self._numberOfTopics = numberOfTopics
        self._outputCol = outputCol


     # apply pre-processing on cloums that need pre-processing and return rest that dosenot need
     def pre_pro_Cols(self,data):
         print( "(Step 1 of 6) Pre-proces coloums that has text (token,  remove stop word,remove  non alphabetic characters, and Lemmatizer....")


         X_clean = data.drop(self._pp_colums, axis=1)
         self._shiftCols = X_clean.shape[1]

         X_pre = data[self._pp_colums]
         textData = []
         for index, row in X_pre.iterrows():
             text = "{} {} {} {}".format(row['title'],
                                         row['location'],
                                         row['description'],
                                         row['requirements'])
             textData.append(text)

         documents = pd.DataFrame(textData, columns=['headline_text'])
         processed_docs = documents['headline_text'].map(self.textPreProcessing)

         return [processed_docs, X_clean]


     def tf_idf(self,processed_docs):

         print("(Step 2 of 6) Building Bag of Words ....")
         dictionary = corpora.Dictionary(processed_docs)
         # count = 0
         # for k, v in dictionary.iteritems():
         #     print(k, v)
         #     count += 1
         #     if count > 10:
         #         break

         #print("Building Bag of Words and TF-IDF dataset ....")
         dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
         bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

         # TF-IDF
         print("(Step 3 of 6) Apply TF-IDF")
         tfidf = models.TfidfModel(bow_corpus)
         corpus_tfidf = tfidf[bow_corpus]

         return [dictionary, corpus_tfidf, bow_corpus]

     def lda(self,corpus_tfidf,dictionary):

         print("(Step 4 of 6)  Start generate {} LDA Topics ....".format(self._numberOfTopics))
         lda_model = models.LdaMulticore(corpus_tfidf,
                                         num_topics=self._numberOfTopics,
                                         id2word=dictionary,
                                         passes=2,
                                         workers=2)

         # Print the LDA topics
         for idx, topic in lda_model.print_topics():
             print('Topic: {} \nWords: {}'.format(idx, topic))

         return lda_model

     def apply_lda(self,lda_model,bow_corpus,X):

         print("(Step 5 of 6)  Find LDA topic for every document ....")
         #Add needed columns
         for idx, topic in lda_model.print_topics():
             X["Topic_{}".format(idx)] = np.nan

         # Find LDA for every document

         for row_index in range(len(bow_corpus)):
             for index, score in sorted(lda_model[bow_corpus[row_index]], key=lambda tup: -1 * tup[1]):
                 # print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))
                 idx = 0
                 for w in lda_model.print_topic(index, self._numberOfTopics).split("+"):
                     X.iloc[row_index, self._shiftCols + idx] = float(w.split("*")[0])
                     idx = idx + 1

         return X


     def textPreProcessing(self,text):
        #print("============ textPreProcessing2 ==========")

        # Removing stop words frequent words
        text_no_stopwords = remove_stopwords(text)
        #print(text_no_stopwords)

        # remove  punctuation
        text_no_punct = text_no_stopwords.translate(remove_punct_dict)
        #print(text_no_punct)

        #Lemmatizer
        token_lemmatized = [wd.decode('utf-8').split('/')[0] for wd in lemmatize(text_no_punct)]
        #print(token_lemmatized)

        # remove  non alphabetic characters.
        tokens_alphabetic = [word.lower() for word in token_lemmatized if word.isalpha()]
        #print(tokens_alphabetic)


        return tokens_alphabetic



## Test code
def main():
    textNLP = TextNLP()
    str= "Create a named mock of the request type from this builder. The same builder can be called to create multiple mocks.";
    print(str)
    newText= textNLP.textPreProcessing(str)
    print(newText)
    str= "Creates mock object of given class or interface. and See examples in javadoc for Mockito class";
    print(str)
    newText= textNLP.textPreProcessing(str)
    print(newText)


if __name__ == '__main__':main()