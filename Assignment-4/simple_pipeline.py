#!/usr/bin/python3

"""
Learning how to use sklearn pipeline

Choose kind of object you want to build.
1. ClassifierMixin - object to be used for classification task eg: Naive bayes
2. RegressorMixin - used for regression tasks such as, linear regression.
3. ClusterMixin - used for clustering task such as GMM, k-means
4. TransformerMixin - Just a transformer takes input x and transforms to x'. 
    eg: CountVectorizer, DictVectorizer, TfIdfVectorizer etc.

And also subclass your class with BaseEstimator.


"""

__author__ = "Sunil"
__copyright__ = "Copyright (c) 2017 Sunil"
__license__ = "MIT License"
__email__ = "suba5417@colorado.edu"
__version__ = "0.1"

# Python modules
import sys
import random

from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin

class SimpleTransformer1(BaseEstimator, TransformerMixin):
    """
    Simple transformer 1
    """
    def __init__(self):
        pass

    def fit(self, x, y):
        print("In Fit-1: ", x, y)
        return self

    def transform(self, x):
        print("In Transform-1: ", x)
        x += str(SimpleTransformer1)
        return x

class SimpleTransformer2(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y):
        print("In Fit-2: ", x, y)
        return self

    def transform(self, x):
        print("In Transform-2: ", x)
        x += str(SimpleTransformer2)
        return x

class SimpleEstimater1(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, x, y):
        print("simple estimator-1 fit:", x, y)
        return self

    def predict(self, x):
        print("simple estimator-1 predict:", x)
        return x

    def score(self, x, y=None):
        return 0.0

pipeline = Pipeline([
    ('st-1', SimpleTransformer1()),
    ('st-2', SimpleTransformer2()),
    ('est-1', SimpleEstimater1())
])

x = "Hello Sunil"
y = "#greet"

pipeline.fit(x, y)
ypredict = pipeline.predict(x)

print(x, "==>", y)
print(x, "==>", ypredict)