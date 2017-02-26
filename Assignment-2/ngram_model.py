#!/usr/bin/python3

"""
N-gram language model. 

Build a unigram and bigram model with and without smoothing.
"""

__author__ = "Sunil"
__email__ = "suba5417@colorado.edu"

import re
import os
import sys
import math
from copy import deepcopy

# class Ngram(object):
#     def __init__(self):
#         pass

unigramsCounter = dict()
bigramsCounter = dict()

def GetNgrams(tokens, ngram):
    # begining of the sentence
    tokens.insert(0, "<s>")
    # end of sentence token
    tokens.append("</s>")
    return zip(*[tokens[i:] for i in range(ngram)])
    

def GetUnigramCounts(word):
    return unigramsCounter[word] if word in unigramsCounter else 0

def GetBigramCounts(word):
    return bigramsCounter[word] if word in bigramsCounter else 0

# Or GetUnigramAndBigramCounts()
def GetUnigramProbabilities(word):
    pass

def GetBigramProbabilities(word):
    pass

def GetSmoothedProbabilities(word):
    pass

# Or GetBigramAndSmoothedProb()
def GetUnigramSentenceProbability(sentence):
    for word in sentence.split():
        prob += math.log10(GetUnigramProbabilities(word))

    return prob

def GetBigramSentenceProbability(sentence):
    tokens = sentence.split()
    for biword in GetNgrams(tokens, 2):
        prob += GetBigramProbabilities(biword)

    return prob

def GetSmoothedSentenceProbability(sentence):
    pass

def UpdateCounter(tokens, counter):
    for token in tokens:
        if token not in counter:
            counter[token] = 1
        else:
            counter[token] += 1
    return counter

def main(cmdline):
    trainingFile = cmdline[0]
    testFile = cmdline[1]

    with open(trainingFile, 'r') as trainFile:
        for sentence in trainFile:
            unitokens = sentence.lower().split()
            bitokens = GetNgrams(deepcopy(unitokens), 2)

            unigramsCounter = UpdateCounter(unitokens, unigramsCounter)
            bigramsCounter = UpdateCounter(bitokens, bigramsCounter)

    # Now use the test file to calculate probabilities
    # What should i do now. read the test file and calculate 
    # unigram and bigram model probabilities for each sentence?
    with open(testFile, 'r') as testFile:
        for sentence in testFile:
            sentence = sentence.lower()
            uniProb = GetUnigramSentenceProbability(sentence)
            biProb = GetBigramSentenceProbability(sentence)
            biSmoothProb = GetSmoothedSentenceProbability(sentence)

            print("S = ", sentence)
            print("Unigrams: logprob(S) = ", uniProb)
            print("Bigrams: logprob(S) = ", biProb)

if __name__ == '__main__':
    main(sys.argv[1:])