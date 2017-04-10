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

# NLTK modules
# import the movie review dataset
from nltk.corpus import movie_reviews

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# nltk module also provides a Naive bayes classifier.
# For now it's okay. Maybe in future i will use 
# advanced classifier's from sklearn (scikit learn package).
from nltk.classify import NaiveBayesClassifier

# import nltk util for checking accuracy
from nltk.classify import util

# scikit learn modules
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold

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

def wordFeatures(allWords):
    englishStopwords = stopwords.words("english")
    words = [(word, True) for word in allWords if word not in englishStopwords]
    # create a dictionary of words 
    return dict(words)

def getWords(fileAggregateName, label):
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

def main(args):
    postiveWords = getWords("pos", Constants.kPOS)
    negativeWords = getWords("neg", Constants.kNEG)

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

    reviews = np.array(postiveWords + negativeWords)
    # randomize the word distribution
    random.shuffle(reviews)

    splitter = KFold(n_splits=10).split

    for train_indices, test_indices in splitter(reviews):
        train, test = reviews[train_indices], reviews[test_indices]

        model = NaiveBayesClassifier.train(train)
        accuracy = util.accuracy(model, test)
        print("Accuracy of model =", accuracy*100)

        # NLTK implementation of util.accuracy
        # count = 0
        # for data, label in test:
        #     plabel = model.classify(data)
        #     if plabel == label:
        #         count += 1

        # print("custom accuracy =", (count/len(test))*100)


if __name__ == "__main__":
    main(sys.argv)