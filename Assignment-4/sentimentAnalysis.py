#!/usr/bin/python3

"""
Sentiment analysis using NLTK package in Python
"""

__author__ = "Sunil"
__copyright__ = "Copyright (c) 2017 Sunil"
__license__ = "MIT License"
__email__ = "suba5417@colorado.edu"
__version__ = "0.1"

import collections
import itertools
import random
import string
# Python modules
import sys

import numpy as np
# NLTK modules
# import the movie review dataset
from nltk import pos_tag, stem
# import nltk util for checking accuracy
# nltk module also provides a Naive bayes classifier.
# For now it's okay. Maybe in future i will use 
# advanced classifier's from sklearn (scikit learn package).
from nltk.classify import (DecisionTreeClassifier, NaiveBayesClassifier,
                           maxent, util)
# import nltk's wrapper for sklearn modules
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.collocations import BigramCollocationFinder
from nltk.corpus import movie_reviews, stopwords
from nltk.metrics import BigramAssocMeasures
from nltk.probability import ConditionalFreqDist, FreqDist
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn import metrics, svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
# scikit learn modules
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline

# word lemmatizer. plurals to singular words
lemmatizer = stem.wordnet.WordNetLemmatizer()

# keep only the stem words
stemmer = SnowballStemmer("english")

# remove the punctuations and digits from words
removeDigitsPunctuations = string.punctuation + string.digits

# ignore english stop words
englishStopwords = stopwords.words("english")

class Constants(object):
    kPOS = "+"
    kNEG = "-"

def getTopNBestWords(N=10000):
    wordDist = FreqDist()
    labelWordDist = ConditionalFreqDist()
     
    for word in movie_reviews.words(categories=['pos']):
        wordDist[word.lower()] += 1
        labelWordDist['pos'][word.lower()] += 1
     
    for word in movie_reviews.words(categories=['neg']):
        wordDist[word.lower()] += 1
        labelWordDist['neg'][word.lower()] += 1
     
    pwords, nwords = labelWordDist['pos'].N(), labelWordDist['neg'].N()
    twords = pwords + nwords
    wordScores = dict()

    for word, freq in wordDist.items():
        # pos score
        pos_score = BigramAssocMeasures.chi_sq(labelWordDist['pos'][word], 
            (freq, pwords), twords)
        # neg score
        neg_score = BigramAssocMeasures.chi_sq(labelWordDist['neg'][word], 
            (freq, nwords), twords)
        # total word score.
        wordScores[word] = pos_score + neg_score

    best = sorted(wordScores.items(), key=lambda w: w[1], reverse=True)
    print(len(best))
    best = best[:N]
    bestwords = set([w for w, s in best])
    return bestwords

def wordSanitizer(wordOrg):
    word = wordOrg
    word = word.translate(str.maketrans('','', removeDigitsPunctuations))
    # word = lemmatizer.lemmatize(word.lower())
    # word = stemmer.stem(word)
    return word

def wordFeatures(allWords):
    # without any processing
    # words = [(word, True) for word in allWords]
    # return dict(words)

    allcleanWords = [wordSanitizer(word) for word in allWords] # if word not in englishStopwords]
    
    # ignore emptry strings
    cleanWords = list(filter(None, allcleanWords))

    # get the tagged words and use them as features. 
    # cleanWords = pos_tag(cleanWords)
    # taggedWords = []
    # for e in cleanWords:
    #     taggedWords.append("/".join(e))

    # cleanWords = taggedWords

    # create bigram words
    bigrams = list(zip(*[cleanWords[i:] for i in range(2)]))

    # create a feature word in the format required by NTLK Naive bayes
    # words = [(" ".join(bigram), True) for bigram in bigrams]
    words = [(bigram, True) for bigram in bigrams]
 
    # words = [(word, True) for word in allWords]
    # words = [word for word in allWords if word not in englishStopwords]
    # create a dictionary of words 
    return dict(words)

def BestBigrams(allWords, bestwords):
    # with word-tag features
    # allBestWords = [word for word in allWords if word in bestwords]
    # taggedWords = pos_tag(allWords)
    # return dict([(word, True) for word in taggedWords])
    return dict([(word, True) for word in allWords if word in bestwords])

def getLabelledWords(fileAggregateName, label):
    words = []
    for fileid in movie_reviews.fileids(fileAggregateName):
        w = movie_reviews.words(fileid)
        words.append((wordFeatures(w), label))
    return words

def getPosNegLabelledWords():
    bestwords = getTopNBestWords()
    
    pos = []
    for fileid in movie_reviews.fileids('pos'):
        w = movie_reviews.words(fileid)
        pos.append((BestBigrams(w, bestwords), Constants.kPOS))
    
    neg = []
    for fileid in movie_reviews.fileids('neg'):
        w = movie_reviews.words(fileid)
        neg.append((BestBigrams(w, bestwords), Constants.kNEG))
    
    return pos, neg

def classify(model, test):
    trueLabels, predLabels = [], []
    for feature, label in test:
        pred = model.classify(feature)
        predLabels.append(pred)
        trueLabels.append(label)

    return np.array(trueLabels), np.array(predLabels)

def main(args):
    # postiveWords, negativeWords = getLabelledWords("pos", Constants.kPOS), \
    #                 getLabelledWords("neg", Constants.kNEG)

    # # Bigram model with top n best words.
    postiveWords, negativeWords = getPosNegLabelledWords()

    reviews = np.array(negativeWords + postiveWords)
    
    # np.set_printoptions(threshold=np.nan)
    # print(reviews)

    # randomize the word distribution
    np.random.shuffle(reviews)

    splitter = KFold(n_splits=10).split
    accuracies = []
    precisions = []
    recalls = []
    fmeasures = []

    formatTo3Decimals = lambda header, decimal: "{0}:{1:.3f}".format(header, decimal)

    for train_indices, test_indices in splitter(reviews):
        train, test = reviews[train_indices], reviews[test_indices]

        # ================= SVC Classifier =================
        # pipeline = Pipeline([
        #     ('tfidf', TfidfTransformer()),
        #     # ('chi2', SelectKBest(chi2, k=1000)),
        #     ('nb', svm.SVC())
        #     ])
        # model = SklearnClassifier(pipeline)
        # model.train(train)

        # ================= Naive bayes classifier =================
        model = NaiveBayesClassifier.train(train)
        # print(model.show_most_informative_features(n=20))

        # ================= MaxEnt classifier =================
        # encoding = maxent.TypedMaxentFeatureEncoding.train( \
        #     train, count_cutoff=30, alwayson_features=True)

        # model = maxent.MaxentClassifier.train(train, bernoulli=False, encoding=encoding, trace=0)

        # ================= Decision Tree classifier =================
        # model = DecisionTreeClassifier.train(train)

        # ================= Statistics about the model =================
        accuracy = util.accuracy(model, test)

        trueLabels, predLabels = classify(model, test)
        # print(precision_recall_fscore_support(trueLabels, predLabels, beta = 1.0, average="macro"))
        precision, recall, f1, support = \
            precision_recall_fscore_support(trueLabels, predLabels, beta = 1.0, average="macro")

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        fmeasures.append(f1)

        # print(formatTo3Decimals("Accuracy", accuracy))
        # # print(precision, recall, f1, support)
        # print(
        #     formatTo3Decimals("Precision", precision),
        #     ";",
        #     formatTo3Decimals("Recall", recall),
        #     ";",
        #     formatTo3Decimals("F1", f1))

    # Take average accuracy.
    accuracy = np.mean(accuracies)
    precision = np.mean(precisions)
    recall = np.mean(recalls)
    f1 = np.mean(fmeasures)

    print(formatTo3Decimals("Accuracy", accuracy))
    print("{0};{1};{2}".format(
            formatTo3Decimals("Precision", precision),
            formatTo3Decimals("Recall", recall),
            formatTo3Decimals("F1", f1)
        ))

if __name__ == "__main__":
    main(sys.argv)
