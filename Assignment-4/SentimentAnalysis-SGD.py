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
import numpy as np

from nltk.corpus import movie_reviews as reviews

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

class DocumentSanitizer(BaseEstimator, TransformerMixin):
    """
    Basic language processing before it's fed to other transformers like
    CountVectorizer or TfIdfVectorizer.
    """
    def __init__(self):
        pass

    def fit(self, X, Y):
        return self

    def transform(self, X):
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
    
    formatTo3Decimals = lambda header, decimal: "{0}:{1:.3f}".format(header, decimal)

    for i, (train_index, test_index) in enumerate(splitter(X)):
        # print("Running for ", i)

        Xtrain, Xtest = X[train_index], X[test_index]
        Ytrain, Ytest = Y[train_index], Y[test_index]

        pipeline = Pipeline([
            ("DocumentProcessor", DocumentSanitizer()),
            ("TfIdfVec", TfidfVectorizer(tokenizer=None, preprocessor=None, lowercase=False)),
            ("SGDclassifier", SGDClassifier())
            ])

        model = pipeline.fit(Xtrain, Ytrain)
        Ypred = model.predict(Xtest)
        # print(classification_report(Ytest, Ypred, target_names=labelTransformer.classes_))
        precision, recall, f1, support = \
            precision_recall_fscore_support(Ytest, Ypred, beta = 1.0, average="macro")

        print(formatTo3Decimals("Accuracy", accuracy_score(Ytest, Ypred)))
        # print(precision, recall, f1, support)
        print(
            formatTo3Decimals("Precision", precision),
            ";",
            formatTo3Decimals("Recall", recall),
            ";",
            formatTo3Decimals("F1", f1))



if __name__ == "__main__":
    main()
