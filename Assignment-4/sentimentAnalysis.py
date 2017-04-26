#!/usr/bin/python3

"""
Sentiment analysis using NLTK package in Python
"""

__author__ = "Sunil"
__copyright__ = "Copyright (c) 2017 Sunil"
__license__ = "MIT License"
__email__ = "suba5417@colorado.edu"
__version__ = "0.1"

# Python modules
import sys
import random
import string

# NLTK modules
# import the movie review dataset
import nltk

from nltk.stem import SnowballStemmer
from nltk.corpus import movie_reviews

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# nltk module also provides a Naive bayes classifier.
# For now it's okay. Maybe in future i will use 
# advanced classifier's from sklearn (scikit learn package).
from nltk.classify import NaiveBayesClassifier

# import nltk util for checking accuracy
from nltk.classify import util

# import nltk's wrapper for sklearn modules
from nltk.classify.scikitlearn import SklearnClassifier

# scikit learn modules
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline


class Constants:
    kPOS = "+"
    kNEG = "-"

class SentimentAnalyzer(object):
    """
    Movie review sentiment analysis

    TODO: Figure out interfaces for this class. 

    Should take training file(s) and create a model &
    be able to predict the a given sentence/file has postive 
    or negative rating and also to what extent can the model 
    be trusted on the results.
    """
    def __init__(self):
        pass

# word lemmatizer. plurals to singular words
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

# keep only the stem words
stemmer = SnowballStemmer("english")

# remove the punctuations and digits from words
removeDigitsPunctuations = string.punctuation + string.digits

# ignore english stop words
englishStopwords = stopwords.words("english")

def wordSanitizer(wordOrg):
    word = wordOrg
    word = word.translate(str.maketrans('','', removeDigitsPunctuations))
    word = lemmatizer.lemmatize(word.lower())
    word = stemmer.stem(word)
    # print(wordOrg, "\t", word)
    return word

def wordFeatures(allWords):
    allcleanWords = [wordSanitizer(word) for word in allWords if word not in englishStopwords]
    
    # ignore emptry strings
    cleanWords = list(filter(None, allcleanWords))

    # create bigram words
    bigrams = list(zip(*[cleanWords[i:] for i in range(2)]))

    # create a feature word in the format required by NTLK Naive bayes
    words = [(" ".join(bigram), True) for bigram in bigrams]

    # words = [(word, True) for word in allWords]
    # words = [word for word in allWords if word not in englishStopwords]
    # create a dictionary of words 
    return dict(words)
    # return words

def getLabelledWords(fileAggregateName, label):
    words = []
    for fileid in movie_reviews.fileids(fileAggregateName):
        w = movie_reviews.words(fileid)
        words.append((wordFeatures(w), label))
    return words

def randomSplit(data, percent):
    random.shuffle(data)
    size = len(data)
    train_len = int(size*percent/100)
    test_len = size - train_len
    return data[:train_len], data[-test_len:]

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

def main(args):
    negativeWords = getLabelledWords("neg", Constants.kNEG)
    postiveWords = getLabelledWords("pos", Constants.kPOS)

    # Not a good idea to combine words and then split.
    # because +/- words maybe skewed/ may result in uneven split.
    # To increase changes of even distribution of words split then 
    # individually and then combine.

    # without kfold 
    # posTrain, posTest = randomSplit(postiveWords, 80)
    # negTrain, negTest = randomSplit(negativeWords, 80)

    # train = posTrain + negTrain
    # test = posTest + negTest

    # model = NaiveBayesClassifier.train(train)
    # accuracy = util.accuracy(model, test)
    # print("Accuracy of model = ", accuracy*100)

    reviews = np.array(negativeWords + postiveWords)
    
    # np.set_printoptions(threshold=np.nan)
    # print(reviews)

    # randomize the word distribution
    np.random.shuffle(reviews)



    splitter = KFold(n_splits=10).split
    accuracies = []

    for train_indices, test_indices in splitter(reviews):
        train, test = reviews[train_indices], reviews[test_indices]

        # pipeline = Pipeline([
        #     ('tfidf', TfidfTransformer()),
        #     ('chi2', SelectKBest(chi2, k=1000)),
        #     ('nb', svm.SVC())
        #     ])

        # model = SklearnClassifier(pipeline)
        # model.train(train)
        model = NaiveBayesClassifier.train(train)
        accuracy = util.accuracy(model, test)
        accuracies.append(accuracy)
        print("Accuracy of model = {0:.3f}".format(accuracy*100.0))

        # NLTK implementation of util.accuracy
        # count = 0
        # tt = pipeline.fit_transform(test)
        # print(tt)
        # for data, label in test:
        #     plabel = model.classify(pipeline.fit_transform(data))
        #     if plabel == label:
        #         count += 1

        # print("custom accuracy =", (count/len(test))*100)

    # Take average accuracy.
    avgAcc = mean(accuracies)
    print("Mean accuracy  = {0:.3f}".format(avgAcc*100.0))


if __name__ == "__main__":
    main(sys.argv)