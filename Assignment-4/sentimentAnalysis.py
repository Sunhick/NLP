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
import collections

import numpy as np

import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

# NLTK modules
# import the movie review dataset
from nltk import pos_tag
from nltk import stem
# from nltk.metrics import precision
# from nltk.metrics import recall
# from nltk.metrics import f_measure

from nltk.stem import SnowballStemmer
from nltk.corpus import movie_reviews

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# nltk module also provides a Naive bayes classifier.
# For now it's okay. Maybe in future i will use 
# advanced classifier's from sklearn (scikit learn package).
from nltk.classify import NaiveBayesClassifier
from nltk.classify import DecisionTreeClassifier
from nltk.classify import maxent

# import nltk util for checking accuracy
from nltk.classify import util

# import nltk's wrapper for sklearn modules
from nltk.classify.scikitlearn import SklearnClassifier

# scikit learn modules
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
from sklearn.model_selection import KFold

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

def getTopNBestWords(N=10000):
    wordDist = FreqDist()
    labelWordDist = ConditionalFreqDist()
     
    for word in movie_reviews.words(categories=['pos']):
        wordDist[word.lower()] += 1
        labelWordDist['pos'][word.lower()] += 1
     
    for word in movie_reviews.words(categories=['neg']):
        wordDist[word.lower()] += 1
        labelWordDist['neg'][word.lower()] += 1
     
    pos_word_count = labelWordDist['pos'].N()
    neg_word_count = labelWordDist['neg'].N()
    total_word_count = pos_word_count + neg_word_count
     
    wordScores = {}
     
    for word, freq in wordDist.items():
        # pos score
        pos_score = BigramAssocMeasures.chi_sq(labelWordDist['pos'][word], 
            (freq, pos_word_count), total_word_count)
        # neg score
        neg_score = BigramAssocMeasures.chi_sq(labelWordDist['neg'][word], 
            (freq, neg_word_count), total_word_count)
        # total word score.
        wordScores[word] = pos_score + neg_score

    best = sorted(wordScores.items(), key=lambda w: w[1], reverse=True)[:N]
    bestwords = set([w for w, s in best])
    return bestwords

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
lemmatizer = stem.wordnet.WordNetLemmatizer()

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

def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])

def wordFeatures(allWords):
    # without any processing
    # words = [(word, True) for word in allWords]
    # return dict(words)

    allcleanWords = [wordSanitizer(word) for word in allWords]# if word not in englishStopwords]
    
    # ignore emptry strings
    cleanWords = list(filter(None, allcleanWords))
    # return bigram_word_feats(cleanWords)

    # get the tagged words and use them as features. 
    cleanWords = pos_tag(cleanWords)

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

def randomSplit(data, percent):
    random.shuffle(data)
    size = len(data)
    train_len = int(size*percent/100)
    test_len = size - train_len
    return data[:train_len], data[-test_len:]

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

def classify(model, test):
    # refsets = collections.defaultdict(set)
    # testsets = collections.defaultdict(set)

    # for i, (feats, label) in enumerate(test):
    #     refsets[label].add(i)
    #     observed = model.classify(feats)
    #     testsets[observed].add(i)

    # return refsets, testsets
    trueLabels, predLabels = [], []
    for feature, label in test:
        pred = model.classify(feature)
        predLabels.append(pred)
        trueLabels.append(label)

    return np.array(trueLabels), np.array(predLabels)

def main(args):
    negativeWords = getLabelledWords("neg", Constants.kNEG)
    postiveWords = getLabelledWords("pos", Constants.kPOS)

    # # Bigram model with top n best words.
    # postiveWords, negativeWords = getPosNegLabelledWords()

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

        # ================= MaxEnt classifier =================
        # encoding = maxent.TypedMaxentFeatureEncoding.train( \
        #     train, count_cutoff=30, alwayson_features=True)

        # model = maxent.MaxentClassifier.train(train, bernoulli=False, encoding=encoding, trace=0)

        # ================= Decision Tree classifier =================
        # model = DecisionTreeClassifier.train(   \
        #     train, entropy_cutoff=0, support_cutoff=0)

        # ================= Statistics about the model =================
        accuracy = util.accuracy(model, test)
        accuracies.append(accuracy)

        # predictions = model.classify(test)
        # refset, predictset = classify(model, test)
        trueLabels, predLabels = classify(model, test)
        # print(precision_recall_fscore_support(trueLabels, predLabels, beta = 1.0, average="macro"))
        precision, recall, f1, support = \
            precision_recall_fscore_support(trueLabels, predLabels, beta = 1.0, average="macro")

        print(formatTo3Decimals("Accuracy", accuracy))
        # print(precision, recall, f1, support)
        print(
            formatTo3Decimals("Precision", precision),
            ";",
            formatTo3Decimals("Recall", recall),
            ";",
            formatTo3Decimals("F1", f1))
        
        # print("Accuracy:{0:.3f}".format(accuracy))
        # print("Precision:{}",precision(refset[Constants.kPOS], predictset[Constants.kPOS]))
        # print("Recall:", recall(refset[Constants.kPOS], predictset[Constants.kPOS]))
        # print("F1:", f_measure(refset[Constants.kPOS], predictset[Constants.kPOS]))


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