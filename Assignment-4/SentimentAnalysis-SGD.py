#!/usr/bin/python3

"""
Sentiment analysis using NLTK package in Python
"""

__author__ = "Sunil"
__copyright__ = "Copyright (c) 2017 Sunil"
__license__ = "MIT License"
__email__ = "suba5417@colorado.edu"
__version__ = "0.1"

import random
import sys

import numpy as np
from nltk.corpus import movie_reviews as reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn import svm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_recall_fscore_support)
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


class DocumentSanitizer(BaseEstimator, TransformerMixin):
    """
    Basic language processing before it's fed to other transformers like
    CountVectorizer or TfIdfVectorizer.
    """
    def __init__(self):
        pass

    def fit(self, X, Y):
        """
        Fit. Nothing todo here.
        """
        return self

    def transform(self, X):
        # Rmeove stop words, punctuations, numbers, stemming, lemmatization.
        # Bigram model if required. Since i'm using Tfidf it has an option to create 
        # N-grams.
        return X

def main():
    X = np.array([reviews.raw(fileid) for fileid in reviews.fileids()])
    y = np.array([reviews.categories(fileid)[0] for fileid in reviews.fileids()])

    data = np.array(list(zip(X, y)), dtype=np.dtype([('f1', np.object), ('f2', np.object)]))
    np.random.shuffle(data)

    X, y = zip(*data)
    X = np.array(X)
    y = np.array(y)

    labelTransformer = LabelEncoder()
    Y = labelTransformer.fit_transform(y)

    splitter = KFold(n_splits=10).split
    accuracies = []
    precisions = []
    recalls = []
    fmeasures = []
    
    formatTo3Decimals = lambda header, decimal: "{0}:{1:.3f}".format(header, decimal)

    for train_index, test_index in splitter(X):
        Xtrain, Xtest = X[train_index], X[test_index]
        Ytrain, Ytest = Y[train_index], Y[test_index]

        pipeline = Pipeline([
            ("DocumentProcessor", DocumentSanitizer()),
            ("TfIdfVec", TfidfVectorizer(tokenizer=None, preprocessor=None, lowercase=False, 
                ngram_range=(1,2))),
            # ("CountVec", CountVectorizer()),
            ("SGDclassifier", SGDClassifier())
            # ("svc", svm.SVC(kernel='linear'))
            # ("logreg", LogisticRegression())
            # ("KNeighborsClassifier", KNeighborsClassifier(n_neighbors=3))
            # ("MLPClassifier", MLPClassifier())
            # ("DecisionTreeClassifier", DecisionTreeClassifier(max_depth=6))
            # ("GaussianNB", GaussianNB()),
            # ("RandomForestClassifier", RandomForestClassifier())
            ])

        model = pipeline.fit(Xtrain, Ytrain)
        Ypred = model.predict(Xtest)
        # print(classification_report(Ytest, Ypred, target_names=labelTransformer.classes_))
        precision, recall, f1, support = \
            precision_recall_fscore_support(Ytest, Ypred, beta = 1.0, average="macro")

        accuracy = accuracy_score(Ytest, Ypred)

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
            formatTo3Decimals("F1", f1)))

if __name__ == "__main__":
    main()