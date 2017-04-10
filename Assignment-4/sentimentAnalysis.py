#!/usr/bin/python3

"""
Sentiment analysis using NLTK package in Python
"""

__author__ = "Sunil"
__copyright__ = "Copyright (c) 2017 Sunil"
__license__ = "MIT License"
__email__ = "suba5417@colorado.edu"
__version__ = "0.1"

import sys

# import the movie review dataset
from nltk.corpus import movie_reviews

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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

def main(args):
    pass

if __name__ == "__main__":
    main(sys.argv)