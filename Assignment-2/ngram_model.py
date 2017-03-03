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

SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"

def GetNgrams(tokens, ngram):
    # begining of the sentence
    tokens.insert(0, SENTENCE_BEGIN)
    # end of sentence token
    tokens.append(SENTENCE_END)
    return zip(*[tokens[i:] for i in range(ngram)])
    

def GetUnigramCounts(word):
    return unigramsCounter[word] if word in unigramsCounter else 0

def GetBigramCounts(word):
    return bigramsCounter[word] if word in bigramsCounter else 0

# Or GetUnigramAndBigramCounts()
def GetUnigramProbabilities(word):
    prob = float(GetUnigramCounts(word)) / len(unigramsCounter)
    return prob

def GetBigramProbabilities(bigram):
    prob = 0.0
    first,_ = bigram
    prob = float(GetBigramCounts(bigram))/GetUnigramCounts((first,))
    return prob

def GetSmoothedProbabilities(word):
    V = len(unigramsCounter)
    k = .0001
    prob = float(GetBigramCounts(word)+k)/(GetUnigramCounts(word)+V)
    return prob

# Or GetBigramAndSmoothedProb()
def GetUnigramSentenceProbability(sentence):
    prob = 0.0
    for word in sentence.split():
        prob += math.log10(GetUnigramProbabilities((word,)))

    return prob

def GetBigramSentenceProbability(sentence):
    tokens = sentence.split()
    prob = 0.0
    for bigram in GetNgrams(tokens, 2):
        p = GetBigramProbabilities(bigram)
        if p == 0:
            prob = 0
            break
        else:
            prob += math.log10(p)
    return prob

def GetSmoothedSentenceProbability(sentence):
    prob = 0.0
    for word in sentence.split():
        prob += math.log10(GetSmoothedProbabilities(word))

    return prob

def UpdateCounter(tokens, counter):
    for token in tokens:
        if token not in counter:
            counter[token] = 1
        else:
            counter[token] += 1
    return counter

def main(cmdline):
    # use the global variables 
    global unigramsCounter
    global bigramsCounter
    trainingFile = cmdline[0]
    testFile = cmdline[1]

    with open(trainingFile, 'r') as trainFile:
        for sentence in trainFile:
            tokens = sentence.lower().split()
            unitokens = GetNgrams(deepcopy(tokens), 1)
            bitokens = GetNgrams(deepcopy(tokens), 2)

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
            print("Bigrams Smoothing: logprob(S) = ", biSmoothProb)

if __name__ == '__main__':
    main(sys.argv[1:])
