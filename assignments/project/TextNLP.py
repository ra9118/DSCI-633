
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer

pd.options.mode.chained_assignment = None

class TextNLP:

     def __init__(self,
                  pp_colums = ['title','location','description','requirements'],
                  outputCol ='fraudulent',
                  numberOfTopics = 45, # best so far 45
                  upSampling = False):

        self._pp_colums = pp_colums
        self._numberOfTopics = numberOfTopics
        self._outputCol = outputCol
        self._upSampling = upSampling
        self.vectorizer = None

     def balanceData(self,X,Y):

        print("(Step 1 of 2) balancing the data")
        data = X
        data["fraudulent"] = Y

        print("--- Data Before balance ----")
        print(data.fraudulent.value_counts())
        df_majority = data[data.fraudulent == 0]
        df_minority = data[data.fraudulent == 1]

        # Up-sampling is the process of randomly duplicating observations from the minority class in order to reinforce its signal.
        # There are several heuristics for doing so, but the most common way is to simply resample with replacement.
        # https://elitedatascience.com/imbalanced-classes
        if self._upSampling:
            # Upsample minority class
            df_minority_upsampled = resample(df_minority,
                                             replace=True,  # sample with replacement
                                             n_samples=df_majority.shape[0],  # to match majority class
                                             random_state=123)  # reproducible results

            # Combine majority class with upsampled minority class
            df_sampled = pd.concat([df_majority, df_minority_upsampled])

        else:
            # Downsample majority class
            df_majority_downsampled = resample(df_majority,
                                               replace=False,  # sample without replacement
                                               n_samples=df_minority.shape[0],  # to match minority class
                                               random_state=123)  # reproducible results

            # Combine minority class with downsampled majority class
            df_sampled = pd.concat([df_majority_downsampled, df_minority])

        # Display new class counts
        print("--- Data after balance ----")
        print(df_sampled.fraudulent.value_counts())

        return [df_sampled.drop(['fraudulent'], axis=1), df_sampled["fraudulent"]]


     # apply pre-processing on cloums that need pre-processing and return rest that dosenot need
     def pre_pro_Cols(self,data):
         print( "(Step 2 of 2) Pre-proces coloums that has text (token,  remove stop word,remove  non alphabetic characters, and Lemmatizer....")


         # Take only columns that has number
         X_clean = data.drop(self._pp_colums, axis=1)
         self._shiftCols = X_clean.shape[1]

         # Take only columns that has text
         X_pre = data[self._pp_colums]
         textData = []
         for index, row in X_pre.iterrows():
             text = "{} {} {} {}".format(row['title'],
                                         row['location'],
                                         row['description'],
                                         row['requirements'])
             textData.append(text)
         documents = pd.DataFrame(textData, columns=['headline_text'])


         # create the transform
         if self.vectorizer == None:
             #self.vectorizer = HashingVectorizer(n_features=self._numberOfTopics)
             self.vectorizer = TfidfVectorizer(stop_words='english', norm='l2', use_idf=False, smooth_idf=False)
             # encode document
             vectors = self.vectorizer.fit_transform(documents["headline_text"])
         else:
             vectors = self.vectorizer.transform(documents["headline_text"])

         # summarize encoded vector
         data_pro = pd.DataFrame(vectors.toarray(), columns=self.vectorizer.get_feature_names())

         for col in range(data_pro.shape[1]):
             X_clean["Topic_{}".format(col)] = np.nan
             if col > self._numberOfTopics:
                 break


         for row in range(data_pro.shape[0]):
             for col in range(data_pro.shape[1]):
                 X_clean.iloc[row, self._shiftCols + col] =  data_pro.iloc[row, col]
                 if col > self._numberOfTopics:
                     break

         return X_clean


